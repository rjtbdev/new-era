/*
 * cursor_capture.c - install an ao PH hook in osu!.exe's mouse_apply_handler
 * that stashes the cursor float32 x,y into CFG slots 2 and 3, then runs
 * the original instruction unchanged.
 *
 *   install                 sigscan, install hook into first free slot
 *   watch [ms=100]          poll the installed hook's CFG2/3 and print live xy
 *   remove                  unregister this hook (hunts by expected VA)
 *   status                  dump hook table + hit count for our site
 *
 * Build: tools/cursor-capture/build.bat
 */
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "ao_client.h"

#define TARGET_PROCESS "osu!.exe"

/* AQN mouse_apply_handler sig:
 *   d9 c1 d9 1c 24 d9 5d ? d9 5d ? ff 15
 * Function prologue starts about 0x2D bytes earlier; the match itself
 * (`d9 c1` = fld st(1)) is the hook point.
 *
 * At hook time (right at the MATCH):
 *   [ebp-0x14] = cursor.x (float32, written ~5 instructions earlier by the
 *                getter call at 0x1FE0C5C8 → stored at [ebp-0x14])
 *   [ebp-0x10] = cursor.y (float32, same getter)
 * ESI = `this` of the handler, EBP = frame pointer (valid since prologue
 * did `push ebp; mov ebp, esp` at 0x1FE0C5B2).
 */
#define SIG_PATTERN "d9c1d91c24d95d00d95d00ff15"
#define SIG_MASK    "xxxxxxx?xx?xx"

static void die(const char* msg) { fprintf(stderr, "error: %s\n", msg); exit(1); }

/* Reproduce hook_tool's sigscan via the hypercalls directly. Keeps this
 * tool self-contained and target-specific, per the `cli/` rule. */
static int sig_scan_range(uint64_t cr3, uint64_t start_va, uint64_t end_va,
                          const uint8_t* pat, const char* mask, uint32_t plen,
                          uint64_t* match_out)
{
    uint8_t window[8192];
    uint64_t va = start_va & ~0xFFFULL;
    uint32_t carry = 0;
    while (va < end_va) {
        uint64_t chunk = 0x1000;
        if (chunk > end_va - va) chunk = end_va - va;
        uint64_t rc = ao_read_mem(cr3, va, chunk, window + carry);
        if ((int64_t)rc < 0) { va += 0x1000; carry = 0; continue; }
        uint32_t have = carry + (uint32_t)chunk;
        if (have >= plen) {
            for (uint32_t i = 0; i + plen <= have; ++i) {
                int ok = 1;
                for (uint32_t j = 0; j < plen; ++j) {
                    if (mask[j] == '?') continue;
                    if (window[i + j] != pat[j]) { ok = 0; break; }
                }
                if (ok) { *match_out = (va - carry) + i; return 1; }
            }
            carry = plen - 1;
            memmove(window, window + have - carry, carry);
        } else {
            carry = have;
        }
        va += 0x1000;
    }
    return 0;
}

static int hexchar(char c) {
    if (c>='0'&&c<='9') return c-'0';
    if (c>='a'&&c<='f') return c-'a'+10;
    if (c>='A'&&c<='F') return c-'A'+10;
    return -1;
}
static int parse_pattern(const char* hex, uint8_t* out, uint32_t max) {
    uint32_t n = 0;
    while (*hex && n < max) {
        int hi = hexchar(*hex++); if (hi<0) return -1;
        int lo = hexchar(*hex++); if (lo<0) return -1;
        out[n++] = (uint8_t)((hi<<4)|lo);
    }
    return (int)n;
}

static uint64_t resolve_hook_va(uint64_t* cr3_out)
{
    if (!ao_present()) die("ao HV not detected");
    uint64_t cr3 = ao_get_process_cr3(TARGET_PROCESS);
    if (cr3 == 0 || cr3 == AO_NOT_PRESENT) die("osu!.exe not running");
    uint64_t base = ao_find_module_base(TARGET_PROCESS, TARGET_PROCESS);
    if (base == 0 || base == AO_NOT_PRESENT) die("osu! module base not found");

    uint8_t pat[256];
    int plen = parse_pattern(SIG_PATTERN, pat, sizeof(pat));
    if (plen <= 0) die("bad sig pattern");
    if ((int)strlen(SIG_MASK) != plen) die("sig mask length mismatch");

    uint64_t end = base + (uint64_t)2000 * 1024 * 1024;
    uint64_t match = 0;
    if (!sig_scan_range(cr3, base, end, pat, SIG_MASK, (uint32_t)plen, &match))
        die("mouse_apply_handler sig did not match");

    printf("[+] cr3=0x%llx  base=0x%llx  match=0x%llx\n",
           (unsigned long long)cr3, (unsigned long long)base,
           (unsigned long long)match);
    *cr3_out = cr3;
    return match;
}

static int do_install(void)
{
    uint64_t cr3;
    uint64_t hook_va = resolve_hook_va(&cr3);

    /* Script:
     *   READ4  R0 <- [EBP - 0x14]   ; cursor.x bits  (R0/R1 are scratch;
     *   READ4  R1 <- [EBP - 0x10]   ; cursor.y bits   CFG/GR are read-only)
     *   LOG    R0, R1               ; emit {cr3, va, x, y} into the log ring
     *   RETCONT                     ; run `d9 c1` and fall through
     *
     * Earlier version tried to READ4 directly into PH_CFG2/CFG3, which
     * faulted with PH_FAULT_WRITE_RO_REG — CFG regs are set from user-mode
     * and are read-only to the script. R0..R15 are the script's scratch.
     *
     * We use PH_LOG because the ao_client ABI exposes ao_ph_log_read()
     * (a drainable ring) but no ao_ph_get_cfg() to read CFG back to
     * user-mode. Log entries carry our xy bits in `a` / `b`; user-mode
     * drains and reinterprets as float.
     */
    ph_insn script[] = {
        PH_READ4(PH_R0, PH_GR_EBP, -0x14),
        PH_READ4(PH_R1, PH_GR_EBP, -0x10),
        PH_LOG(PH_R0, PH_R1),
        PH_RETCONT(),
    };
    uint32_t count = (uint32_t)(sizeof(script) / sizeof(script[0]));

    uint64_t slot = ao_ph_register(cr3, hook_va, script, count);
    if ((int64_t)slot < 0) {
        fprintf(stderr, "ao_ph_register failed rc=%lld\n", (long long)(int64_t)slot);
        return 1;
    }
    printf("[+] installed in slot %u at va=0x%llx (3 insns)\n",
           (uint32_t)slot, (unsigned long long)hook_va);
    printf("[+] run `cursor_capture watch` to see live cursor xy\n");
    return 0;
}

static int do_watch(int argc, char** argv)
{
    uint32_t interval_ms = (argc >= 3) ? (uint32_t)strtoul(argv[2], NULL, 0) : 100;
    if (!ao_present()) die("ao HV not detected");

    printf("[+] draining ph_log every %ums (Ctrl+C to stop)\n", interval_ms);
    ph_log_entry buf[256];
    uint32_t last_x_bits = 0, last_y_bits = 0;
    uint64_t total_entries = 0;
    while (1) {
        uint64_t n = ao_ph_log_read(buf, 256);
        if ((int64_t)n > 0) {
            /* Only print the most-recent entry if it changed (log rate
             * matches input poll rate from osu, which is very high). */
            ph_log_entry* last = &buf[n - 1];
            total_entries += n;
            uint32_t x_bits = (uint32_t)last->a;
            uint32_t y_bits = (uint32_t)last->b;
            if (x_bits != last_x_bits || y_bits != last_y_bits) {
                float x, y;
                memcpy(&x, &x_bits, 4);
                memcpy(&y, &y_bits, 4);
                printf("drained=%llu  cursor=(%.2f, %.2f)  [va=0x%llx]\n",
                       (unsigned long long)total_entries, (double)x, (double)y,
                       (unsigned long long)last->va);
                last_x_bits = x_bits; last_y_bits = y_bits;
            }
        }
        Sleep(interval_ms);
    }
    return 0;
}

static int do_remove(void)
{
    /* Simple: clear all hooks. (For now; targeted removal would require
     * tracking the slot we got at install time.) */
    uint64_t rc = ao_ph_clear();
    printf("clear rc=%lld\n", (long long)(int64_t)rc);
    return (int64_t)rc < 0 ? 1 : 0;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr,
            "cursor_capture -- capture osu! cursor xy via ao PH hook\n"
            "  install                sigscan osu! and install the PH script\n"
            "  watch [interval_ms]    poll CFG2/CFG3 and print live cursor xy\n"
            "  remove                 ao_ph_clear (removes ALL hooks for now)\n");
        return 1;
    }
    if (strcmp(argv[1], "install") == 0) return do_install();
    if (strcmp(argv[1], "watch")   == 0) return do_watch(argc, argv);
    if (strcmp(argv[1], "remove")  == 0) return do_remove();
    fprintf(stderr, "unknown command: %s\n", argv[1]);
    return 1;
}
