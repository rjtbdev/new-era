/*
 * key_inject.c  —  keyboard-injection hook for osu!stable via ao PH.
 *
 * STAGED BUILD. This file ships in log-only mode first; the real
 * force-press/release path lands in a follow-up commit once the log
 * mode confirms the hook fires cleanly.
 *
 *   resolve                run keybinding sig+deref, print K1/K2 VK + ptrs
 *   install                sigscan keyboard_get_state, install LOG-ONLY hook
 *   watch [ms=100]         drain ph_log, print ECX (state-buffer ptr) per hit
 *   remove                 ao_ph_clear (removes ALL hooks for now)
 *
 * Pass 2 script (log-only):
 *     PH_LOG(ECX, 0)        ; record the state-buffer pointer each call
 *     PH_RETCONT()          ; run original `call <GetKeyboardState>`
 *
 * Future Pass 3 script (force press/release, driven by CFG):
 *     PH_OP_CALL real GetKeyboardState(ECX)
 *     if CFG0 != 0: WRITE1 [ECX + CFG2] <- CFG0   // K1 (CFG2=0x5A, CFG0=0x80/0x00)
 *     if CFG1 != 0: WRITE1 [ECX + CFG3] <- CFG1   // K2 (CFG3=0x58, CFG1=0x80/0x00)
 *     RETSKIP 5                                    ; skip the original `e8 ...`
 */
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "ao_client.h"

#define TARGET_PROCESS "osu!.exe"

/* AQN sigs */
/* keyboard_get_state: hook AT the `call <GetKeyboardState>` opcode.
 *   e8 ? ? ? ? 33 f6 8b 5f 04 3b f3 73 ? f6 44 37 08 80
 */
#define KGS_PATTERN "e8000000003 3f68b5f043bf373 00f64437088 0"
#define KGS_PATTERN_CLEAN "e80000000033f68b5f043bf37300f64437 0880"
/* Note: my pattern-string composition has whitespace gremlins when hand-
 * typed. Keep the actual pattern construction in Python and pass clean
 * hex here. Below we use the clean form built manually: */
#define KGS_SIG_PAT  "e8000000003 3f68b5f043bf373 00f64437088 0"

/* We build the pattern programmatically from the AQN AOB to avoid typos. */
static const char *kgs_aob = "e8 ? ? ? ? 33 f6 8b 5f 04 3b f3 73 ? f6 44 37 08 80";

/* keybinding_aob (from AQN):
 *   ff 15 ? ? ? ? 89 ? ? 8d ? ? 50 8b 0d ? ? ? ? 8b 55 ? ? ? ff
 */
static const char *kb_aob = "ff 15 ? ? ? ? 89 ? ? 8d ? ? 50 8b 0d ? ? ? ? 8b 55 ? ? ? ff";

static void die(const char* msg) { fprintf(stderr, "error: %s\n", msg); exit(1); }

/* ------- pattern parsing + sigscan (same as cursor_capture) ------- */

static int hexchar(char c) {
    if (c>='0'&&c<='9') return c-'0';
    if (c>='a'&&c<='f') return c-'a'+10;
    if (c>='A'&&c<='F') return c-'A'+10;
    return -1;
}

/* Parse "aa bb ? cc" AOB into pattern+mask arrays. */
static int parse_aob(const char *aob, uint8_t *pat_out, char *mask_out, int cap)
{
    int n = 0;
    const char *p = aob;
    while (*p && n < cap) {
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;
        if (*p == '?') {
            while (*p == '?') p++;
            pat_out[n] = 0; mask_out[n] = '?'; n++;
        } else {
            int hi = hexchar(*p++); if (hi < 0) return -1;
            int lo = hexchar(*p++); if (lo < 0) return -1;
            pat_out[n] = (uint8_t)((hi<<4)|lo); mask_out[n] = 'x'; n++;
        }
    }
    mask_out[n] = 0;
    return n;
}

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

static uint64_t find_sig(uint64_t cr3, uint64_t base, const char *aob)
{
    uint8_t pat[256]; char mask[256];
    int plen = parse_aob(aob, pat, mask, sizeof(pat));
    if (plen <= 0) die("bad aob");
    uint64_t end = base + (uint64_t)2000 * 1024 * 1024;
    uint64_t match = 0;
    if (!sig_scan_range(cr3, base, end, pat, mask, (uint32_t)plen, &match))
        return 0;
    return match;
}

static uint32_t read_u32_at(uint64_t cr3, uint64_t va)
{
    uint32_t v = 0;
    uint64_t rc = ao_read_mem(cr3, va, 4, &v);
    if ((int64_t)rc < 0) die("read_u32 failed");
    return v;
}

/* ------- resolve ------- */

typedef struct {
    uint64_t cr3;
    uint64_t base;
    uint64_t kgs_match;       /* VA of the `e8 ... call GetKeyboardState` */
    uint32_t k1_vk, k2_vk;    /* byte indices into 256-byte state buffer */
} resolution;

static int do_resolve_core(resolution *r)
{
    if (!ao_present()) die("ao HV not detected");
    r->cr3 = ao_get_process_cr3(TARGET_PROCESS);
    if (r->cr3 == 0 || r->cr3 == AO_NOT_PRESENT) die("osu! not running");
    r->base = ao_find_module_base(TARGET_PROCESS, TARGET_PROCESS);
    if (r->base == 0 || r->base == AO_NOT_PRESENT) die("module base not found");

    uint64_t kb_match = find_sig(r->cr3, r->base, kb_aob);
    if (kb_match == 0) die("keybinding_aob did not match");
    uint32_t keybinding_ptr = read_u32_at(r->cr3, kb_match + 0xF);
    uint32_t struct_base = read_u32_at(r->cr3, keybinding_ptr);
    uint32_t binding = read_u32_at(r->cr3, struct_base + 8);
    r->k1_vk = read_u32_at(r->cr3, binding + 0x14);
    r->k2_vk = read_u32_at(r->cr3, binding + 0x24);

    r->kgs_match = find_sig(r->cr3, r->base, kgs_aob);
    if (r->kgs_match == 0) die("keyboard_get_state sig did not match");

    return 0;
}

static int do_resolve(void)
{
    resolution r;
    do_resolve_core(&r);
    printf("[+] cr3=0x%llx  base=0x%llx\n",
           (unsigned long long)r.cr3, (unsigned long long)r.base);
    printf("[+] keybinding resolved: K1=0x%02X ('%c')  K2=0x%02X ('%c')\n",
           r.k1_vk & 0xFF,
           (r.k1_vk>=32&&r.k1_vk<127)?(char)r.k1_vk:'?',
           r.k2_vk & 0xFF,
           (r.k2_vk>=32&&r.k2_vk<127)?(char)r.k2_vk:'?');
    printf("[+] keyboard_get_state call-site: 0x%llx\n", (unsigned long long)r.kgs_match);
    printf("    (hook would land AT this VA; `e8` opcode is 5 bytes)\n");
    return 0;
}

/* ------- install (log-only, pass 2) ------- */

static int do_install(void)
{
    resolution r;
    do_resolve_core(&r);

    /* Pass 3: force-press/release K1/K2 via post-call WRITE1.
     *
     * Design (no PH_OP_CALL — simpler, avoids calling-convention issues):
     *
     * Hook A at kgs_match (the `e8 ...` call-site): empty-body RETCONT.
     *   Purpose: keep ECX/EDI stable across the call by using the HV's
     *   normal RETCONT single-step flow. The real Windows GetKeyboardState
     *   runs and populates the byte[256] via ECX.
     *
     * Hook B at kgs_match + 5 (the `xor esi, esi` right after the call):
     *   At this point EDI = the byte[256] pointer (preserved by the call,
     *   as the surrounding osu! code immediately reads [EDI+esi+8] in a
     *   loop). Our script conditionally stamps K1/K2 bytes:
     *     if CFG0 != 0:  [EDI + 8 + K1_VK] = CFG0    (0x80 press, 0x00 release)
     *     if CFG1 != 0:  [EDI + 8 + K2_VK] = CFG1
     *   then RETCONT lets `xor esi, esi` run.
     *
     * CFG layout:
     *   CFG0 = K1 action byte (0 = pass-through, 0x80 = force press, 0x00 ... wait)
     *
     * Problem: "0x00 = release" collides with "0 = don't touch". Resolution:
     *   CFG0 == 0            -> leave K1 alone (user key passes through)
     *   CFG0 == 0x80         -> force K1 pressed
     *   CFG0 == 0x01         -> force K1 released (stored as 0x01 sentinel;
     *                            script converts 0x01 -> 0x00 before writing)
     * Same for CFG1 / K2.
     *
     * CFG2 / CFG3: baked K1/K2 byte indices at install time, so the
     * script doesn't need to re-resolve.
     */

    /* Hook A: empty passthrough at the call site. */
    ph_insn scriptA[] = {
        PH_RETCONT(),
    };
    uint64_t slotA = ao_ph_register(r.cr3, r.kgs_match, scriptA, 1);
    if ((int64_t)slotA < 0) {
        fprintf(stderr, "ao_ph_register A failed rc=%lld\n", (long long)(int64_t)slotA);
        return 1;
    }

    /* Hook B: post-call conditional WRITE1 into [EDI + 8 + K_VK].
     *
     *   R0 = CFG0   (K1 action: 0 pass, 0x80 press, 0x01 release-sentinel)
     *   cmp R0, 0   ; if zero -> skip K1 writes
     *   jeq +4
     *     load R1 = 0x01
     *     cmp R0, R1  ; if R0 == 0x01 (release sentinel), rewrite to 0x00
     *     jne +1      ; else keep R0 as-is (presumed 0x80)
     *     loadi R0 = 0
     *     write1 [EDI + K1_offset] <- R0
     *   (same for K2 using CFG1)
     *   retcont
     *
     * Simpler encoding: we skip the sentinel-remapping and let CFG hold
     * the literal byte to write (0x80 for press, 0x00 release)... but
     * then "0x00" means "both release and don't touch", so we lose the
     * distinction. That's bad.
     *
     * Simplest-but-safe: use TWO gating CFGs per key:
     *   CFG0 = K1 enable (nonzero = override)
     *   CFG1 = K1 value  (literal byte to write: 0x80 or 0x00)
     *   CFG2 = K2 enable
     *   CFG3 = K2 value
     * This also leaves CFG4+ free for baked K1/K2 indices.
     */

    uint32_t k1_off = 8 + (r.k1_vk & 0xFF);  /* offset into byte[256] data */
    uint32_t k2_off = 8 + (r.k2_vk & 0xFF);

    ph_insn scriptB[] = {
        /* K1 branch: if CFG0 != 0, write CFG1 to [EDI + k1_off] */
        PH_LOADI(PH_R0, 0),
        PH_CMP(PH_CFG0, PH_R0),
        PH_JEQ(+2),                           /* skip K1 write if CFG0 == 0 */
        PH_WRITE1(PH_GR_EDI, PH_CFG1, k1_off),

        /* K2 branch: if CFG2 != 0, write CFG3 to [EDI + k2_off] */
        PH_CMP(PH_CFG2, PH_R0),
        PH_JEQ(+2),                           /* skip K2 write if CFG2 == 0 */
        PH_WRITE1(PH_GR_EDI, PH_CFG3, k2_off),

        PH_RETCONT(),
    };
    uint32_t countB = (uint32_t)(sizeof(scriptB) / sizeof(scriptB[0]));
    uint64_t slotB = ao_ph_register(r.cr3, r.kgs_match + 5, scriptB, countB);
    if ((int64_t)slotB < 0) {
        /* unregister A so we don't leave half a hook pair */
        ao_ph_unregister((uint32_t)slotA);
        fprintf(stderr, "ao_ph_register B failed rc=%lld (unregistered A)\n",
                (long long)(int64_t)slotB);
        return 1;
    }

    printf("[+] installed hook pair:\n");
    printf("    slot %u  A: passthrough at 0x%llx (the `e8` call)\n",
           (uint32_t)slotA, (unsigned long long)r.kgs_match);
    printf("    slot %u  B: post-call WRITE at 0x%llx\n",
           (uint32_t)slotB, (unsigned long long)(r.kgs_match + 5));
    printf("[+] K1=0x%02X ('%c')  K2=0x%02X ('%c')\n",
           r.k1_vk & 0xFF,
           (r.k1_vk>=32&&r.k1_vk<127)?(char)r.k1_vk:'?',
           r.k2_vk & 0xFF,
           (r.k2_vk>=32&&r.k2_vk<127)?(char)r.k2_vk:'?');
    printf("[+] CFG layout (ao_ph_set_cfg targets slot %u):\n", (uint32_t)slotB);
    printf("      CFG0 = K1 enable (0 = pass-through, nonzero = override)\n");
    printf("      CFG1 = K1 value (0x80 = pressed, 0x00 = released)\n");
    printf("      CFG2 = K2 enable\n");
    printf("      CFG3 = K2 value\n");
    printf("[+] use `key_inject press k1|k2|both <ms>` to try it.\n");
    printf("[+] use `key_inject remove` when done.\n");
    return 0;
}

/* Find hook B (the post-call WRITE hook) by its VA, so press/release
 * can target the right slot without the caller having to remember. */
static int find_hook_b_slot(void)
{
    resolution r;
    do_resolve_core(&r);
    ph_status s;
    memset(&s, 0, sizeof(s));
    if ((int64_t)ao_ph_status(&s) < 0) die("ph_status failed");
    uint64_t want = r.kgs_match + 5;
    for (uint32_t i = 0; i < 8; i++) {
        if (!s.entries[i].active) continue;
        if (s.entries[i].target_va == want) return (int)i;
    }
    return -1;
}

static int do_press(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
            "usage: press <key> [hold_ms=100]\n"
            "  <key>  = k1 | k2 | both\n"
            "  holds the key(s) for hold_ms, then releases.\n");
        return 1;
    }
    const char *key = argv[2];
    uint32_t hold_ms = (argc >= 4) ? (uint32_t)strtoul(argv[3], NULL, 0) : 100;
    int do_k1 = (strcmp(key, "k1") == 0) || (strcmp(key, "both") == 0);
    int do_k2 = (strcmp(key, "k2") == 0) || (strcmp(key, "both") == 0);
    if (!do_k1 && !do_k2) { fprintf(stderr, "bad key: %s\n", key); return 1; }

    int slot = find_hook_b_slot();
    if (slot < 0) die("hook B not installed -- run `install` first");

    if (do_k1) {
        ao_ph_set_cfg((uint32_t)slot, 0, 1);      /* CFG0 = enable */
        ao_ph_set_cfg((uint32_t)slot, 1, 0x80);   /* CFG1 = value (pressed) */
    }
    if (do_k2) {
        ao_ph_set_cfg((uint32_t)slot, 2, 1);
        ao_ph_set_cfg((uint32_t)slot, 3, 0x80);
    }
    printf("[+] holding %s for %u ms (slot %d)\n", key, hold_ms, slot);
    Sleep(hold_ms);

    /* Release: disable override (osu! then sees whatever Windows reports). */
    if (do_k1) ao_ph_set_cfg((uint32_t)slot, 0, 0);
    if (do_k2) ao_ph_set_cfg((uint32_t)slot, 2, 0);
    printf("[+] released.\n");
    return 0;
}

static int do_watch(int argc, char** argv)
{
    uint32_t interval_ms = (argc >= 3) ? (uint32_t)strtoul(argv[2], NULL, 0) : 100;
    if (!ao_present()) die("ao HV not detected");
    printf("[+] draining ph_log every %ums (Ctrl+C to stop)\n", interval_ms);
    ph_log_entry buf[256];
    uint64_t total = 0;
    uint64_t last_a = 0;
    uint64_t last_b = 0;
    int first = 1;
    while (1) {
        uint64_t n = ao_ph_log_read(buf, 256);
        if ((int64_t)n > 0) {
            total += n;
            ph_log_entry *last = &buf[n - 1];
            if (first || last->a != last_a || last->b != last_b) {
                printf("drained=%llu  a=0x%08llx  b=0x%08llx  [va=0x%llx]\n",
                       (unsigned long long)total,
                       (unsigned long long)last->a,
                       (unsigned long long)last->b,
                       (unsigned long long)last->va);
                last_a = last->a; last_b = last->b;
                first = 0;
            }
        }
        Sleep(interval_ms);
    }
    return 0;
}

static int do_remove(void)
{
    uint64_t rc = ao_ph_clear();
    printf("clear rc=%lld\n", (long long)(int64_t)rc);
    return (int64_t)rc < 0 ? 1 : 0;
}

int main(int argc, char **argv)
{
    (void)KGS_PATTERN; (void)KGS_PATTERN_CLEAN; (void)KGS_SIG_PAT;
    if (argc < 2) {
        fprintf(stderr,
            "key_inject -- osu! keyboard-injection hook via ao\n"
            "  resolve                 run sig+deref, print K1/K2 VK + call-site VA\n"
            "  install                 install Pass-3 hook pair (A: passthrough, B: post-call write)\n"
            "  press k1|k2|both [ms]   force key(s) pressed for <ms> then release (default 100ms)\n"
            "  watch [interval_ms]     drain ph_log (useful if hook A also logs)\n"
            "  remove                  ao_ph_clear (removes ALL hooks)\n");
        return 1;
    }
    if (strcmp(argv[1], "resolve") == 0) return do_resolve();
    if (strcmp(argv[1], "install") == 0) return do_install();
    if (strcmp(argv[1], "watch")   == 0) return do_watch(argc, argv);
    if (strcmp(argv[1], "press")   == 0) return do_press(argc, argv);
    if (strcmp(argv[1], "remove")  == 0) return do_remove();
    fprintf(stderr, "unknown command: %s\n", argv[1]);
    return 1;
}
