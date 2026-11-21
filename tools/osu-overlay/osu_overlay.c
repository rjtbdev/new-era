/*
 * osu_overlay.c - Day 1 proof-of-life: install a CR3-filtered ao hook on
 * gdi32!SwapBuffers in osu!'s address space, log every hit, dump a rate.
 *
 * Build:
 *   ml64 /nologo /c ao_call.asm
 *   cl /nologo /O2 /W3 osu_overlay.c ao_call.obj /link /out:osu_overlay.exe
 *
 * Usage:
 *   osu_overlay.exe install                    - find osu, resolve SwapBuffers,
 *                                                install the log-only hook
 *   osu_overlay.exe watch [seconds]            - poll ph_log and report rate
 *                                                (default 10 sec)
 *   osu_overlay.exe status                     - dump hook table + hits
 *   osu_overlay.exe remove                     - clear this hook
 *
 * Day 1 handler is literally:
 *     PH_LOG(ARG1, ARG2);   // just records this frame happened
 *     PH_RETCONT();         // run original instruction (the jmp into gdi32full)
 *
 * The ao driver itself already enforces the CR3 filter via its
 * TargetCr3 slot on the hook. All we do up here is pass the right CR3.
 */

#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "user32.lib")

#include "ao_client.h"

#define TARGET_PROCESS  "osu!.exe"
#define TARGET_MODULE   "gdi32.dll"
#define TARGET_EXPORT   "SwapBuffers"
#define GL_MODULE       "opengl32.dll"

/* OpenGL state enum values. */
#define GL_PROJECTION    0x1701
#define GL_MODELVIEW     0x1700
#define GL_DEPTH_TEST    0x0B71
#define GL_TEXTURE_2D    0x0DE1
#define GL_QUADS         0x0007
#define GL_BLEND         0x0BE2
#define GL_SRC_ALPHA     0x0302
#define GL_ONE_MINUS_SRC_ALPHA 0x0303
#define GL_LINE_LOOP     0x0002

static void die(const char* msg)
{
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

/*
 * Find the osu! PID by image name.
 */
static DWORD find_pid(const char* image_name)
{
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) return 0;
    PROCESSENTRY32 pe;
    pe.dwSize = sizeof(pe);
    DWORD pid = 0;
    if (Process32First(snap, &pe)) {
        do {
            if (_stricmp(pe.szExeFile, image_name) == 0) {
                pid = pe.th32ProcessID;
                break;
            }
        } while (Process32Next(snap, &pe));
    }
    CloseHandle(snap);
    return pid;
}

/*
 * Find the 32-bit gdi32.dll base inside osu!'s address space by enumerating
 * its loaded modules with LIST_MODULES_32BIT.
 */
static uint64_t find_wow64_module_base(DWORD pid, const char* dll_name)
{
    HANDLE p = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid);
    if (!p) { fprintf(stderr, "OpenProcess(%lu) failed: %lu\n", pid, GetLastError()); return 0; }

    HMODULE mods[1024];
    DWORD need = 0;
    uint64_t base = 0;
    if (EnumProcessModulesEx(p, mods, sizeof(mods), &need, LIST_MODULES_32BIT)) {
        DWORD count = need / sizeof(HMODULE);
        for (DWORD i = 0; i < count; i++) {
            char name[MAX_PATH];
            if (GetModuleBaseNameA(p, mods[i], name, sizeof(name))) {
                if (_stricmp(name, dll_name) == 0) {
                    base = (uint64_t)(uintptr_t)mods[i];
                    break;
                }
            }
        }
    } else {
        fprintf(stderr, "EnumProcessModulesEx failed: %lu\n", GetLastError());
    }
    CloseHandle(p);
    return base;
}

/*
 * Parse the 32-bit gdi32.dll from SysWOW64 (DONT_RESOLVE_DLL_REFERENCES maps
 * it as a data file so we can walk its export table; we can't actually call
 * through it from our 64-bit process). Return the RVA of TARGET_EXPORT.
 */
static uint32_t find_export_rva(const char* dll_path, const char* export_name)
{
    HANDLE f = CreateFileA(dll_path, GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, 0, NULL);
    if (f == INVALID_HANDLE_VALUE) { fprintf(stderr, "open %s failed\n", dll_path); return 0; }
    HANDLE m = CreateFileMappingA(f, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(f);
    if (!m) { fprintf(stderr, "CreateFileMapping failed\n"); return 0; }
    void* v = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(m);
    if (!v) { fprintf(stderr, "MapViewOfFile failed\n"); return 0; }

    uint32_t rva = 0;
    IMAGE_DOS_HEADER* dos = (IMAGE_DOS_HEADER*)v;
    if (dos->e_magic == IMAGE_DOS_SIGNATURE) {
        /* NT headers (32-bit for WoW64 gdi32) */
        IMAGE_NT_HEADERS32* nt = (IMAGE_NT_HEADERS32*)((char*)v + dos->e_lfanew);
        if (nt->Signature == IMAGE_NT_SIGNATURE) {
            IMAGE_DATA_DIRECTORY* exp_dir =
                &nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
            if (exp_dir->VirtualAddress && exp_dir->Size) {
                /* We need to translate RVAs to file offsets since we loaded as a data file. */
                IMAGE_SECTION_HEADER* sec = IMAGE_FIRST_SECTION(nt);
                #define RVA2OFF(rva_) ({ \
                    uint32_t _off = 0; \
                    for (DWORD _i = 0; _i < nt->FileHeader.NumberOfSections; _i++) { \
                        if ((rva_) >= sec[_i].VirtualAddress && \
                            (rva_) < sec[_i].VirtualAddress + sec[_i].Misc.VirtualSize) { \
                            _off = sec[_i].PointerToRawData + ((rva_) - sec[_i].VirtualAddress); \
                            break; \
                        } \
                    } _off; })
                /* MSVC doesn't support GCC statement-expressions; inline it instead. */
                #undef RVA2OFF
                uint32_t exp_off = 0;
                for (DWORD i = 0; i < nt->FileHeader.NumberOfSections; i++) {
                    if (exp_dir->VirtualAddress >= sec[i].VirtualAddress &&
                        exp_dir->VirtualAddress < sec[i].VirtualAddress + sec[i].Misc.VirtualSize) {
                        exp_off = sec[i].PointerToRawData +
                                  (exp_dir->VirtualAddress - sec[i].VirtualAddress);
                        break;
                    }
                }
                if (exp_off) {
                    IMAGE_EXPORT_DIRECTORY* exp = (IMAGE_EXPORT_DIRECTORY*)((char*)v + exp_off);
                    /* AddressOfNames, AddressOfNameOrdinals, AddressOfFunctions are RVAs. */
                    uint32_t aon_off = 0, aono_off = 0, aof_off = 0;
                    for (DWORD i = 0; i < nt->FileHeader.NumberOfSections; i++) {
                        #define XLAT(rva_, out_) \
                            if ((rva_) >= sec[i].VirtualAddress && \
                                (rva_) < sec[i].VirtualAddress + sec[i].Misc.VirtualSize) \
                                (out_) = sec[i].PointerToRawData + ((rva_) - sec[i].VirtualAddress);
                        XLAT(exp->AddressOfNames, aon_off);
                        XLAT(exp->AddressOfNameOrdinals, aono_off);
                        XLAT(exp->AddressOfFunctions, aof_off);
                        #undef XLAT
                    }
                    if (aon_off && aono_off && aof_off) {
                        uint32_t* names = (uint32_t*)((char*)v + aon_off);
                        uint16_t* ords = (uint16_t*)((char*)v + aono_off);
                        uint32_t* funcs = (uint32_t*)((char*)v + aof_off);
                        for (DWORD i = 0; i < exp->NumberOfNames; i++) {
                            uint32_t name_rva = names[i];
                            uint32_t name_off = 0;
                            for (DWORD j = 0; j < nt->FileHeader.NumberOfSections; j++) {
                                if (name_rva >= sec[j].VirtualAddress &&
                                    name_rva < sec[j].VirtualAddress + sec[j].Misc.VirtualSize) {
                                    name_off = sec[j].PointerToRawData +
                                               (name_rva - sec[j].VirtualAddress);
                                    break;
                                }
                            }
                            if (name_off && strcmp((char*)v + name_off, export_name) == 0) {
                                rva = funcs[ords[i]];
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    UnmapViewOfFile(v);
    return rva;
}

static uint64_t resolve_swapbuffers_va_in_osu(DWORD pid)
{
    uint64_t base = find_wow64_module_base(pid, TARGET_MODULE);
    if (!base) die("gdi32.dll not found in osu! (32-bit modules)");
    uint32_t rva = find_export_rva("C:\\Windows\\SysWOW64\\gdi32.dll", TARGET_EXPORT);
    if (!rva) die("SwapBuffers export RVA not found in SysWOW64 gdi32.dll");
    return base + rva;
}

/* Resolve the full VA of a specific export in a specific module in a specific
 * 32-bit process. Returns 0 on failure. */
static uint64_t resolve_export_in_osu(DWORD pid, const char* module_name,
                                       const char* disk_path, const char* export_name)
{
    uint64_t base = find_wow64_module_base(pid, module_name);
    if (!base) { fprintf(stderr, "module %s not found in osu!\n", module_name); return 0; }
    uint32_t rva = find_export_rva(disk_path, export_name);
    if (!rva) { fprintf(stderr, "export %s not found in %s\n", export_name, disk_path); return 0; }
    return base + rva;
}

static int do_install(void)
{
    uint64_t ping = AO_HYPERCALL(AO_CMD_PING, 0, 0, 0, 0);
    fprintf(stderr, "ping = 0x%016llX\n", (unsigned long long)ping);
    if (ping == AO_NOT_PRESENT) die("hypervisor not present (UD2 unhandled)");

    uint64_t cr3 = ao_get_process_cr3(TARGET_PROCESS);
    fprintf(stderr, "get_cr3('%s') = 0x%016llX\n", TARGET_PROCESS, (unsigned long long)cr3);
    if (cr3 == AO_NOT_PRESENT || cr3 == 0) {
        /* Try to list what processes the HV CAN see, to figure out if the
         * name is slightly different (osu!..exe, differing case, etc). */
        unsigned char buf[PROC_ENTRY_SIZE * 256];
        uint64_t n = ao_list_procs(buf, 256);
        fprintf(stderr, "HV sees %llu processes:\n", (unsigned long long)n);
        for (uint64_t i = 0; i < n && i < 256; i++) {
            unsigned char* e = buf + i * PROC_ENTRY_SIZE;
            char name[17];
            memcpy(name, e, 16);
            name[16] = 0;
            uint64_t dtb = 0;
            memcpy(&dtb, e + 16, 8);
            fprintf(stderr, "  %-16s  cr3=0x%016llX\n", name, (unsigned long long)dtb);
        }
        die("osu CR3 lookup failed - name mismatch or not running");
    }

    /* osu! is 32-bit (WoW64). Use PSAPI with LIST_MODULES_32BIT to find
     * the 32-bit gdi32.dll base in osu!'s address space, then parse
     * SysWOW64\gdi32.dll on disk for the SwapBuffers export RVA. */
    DWORD pid = find_pid(TARGET_PROCESS);
    if (!pid) die("osu! PID not found");
    fprintf(stderr, "osu PID            = %lu\n", pid);

    uint64_t swap_va = resolve_swapbuffers_va_in_osu(pid);

    printf("osu CR3            = 0x%016llX\n", (unsigned long long)cr3);
    printf("SwapBuffers (osu)  = 0x%016llX\n", (unsigned long long)swap_va);

    /*
     * Day 1 script: log that we fired, run original.
     *   PH_LOG(PH_GR_EAX, PH_GR_ECX)  - two guest regs for quick visual id
     *   PH_RETCONT()                  - fall back to the Option 7 single-step
     *                                   path so the original jmp runs from the
     *                                   byte-identical shadow page
     */
    ph_insn script[] = {
        PH_LOG(PH_GR_EAX, PH_GR_ECX),
        PH_RETCONT(),
    };
    uint32_t n = (uint32_t)(sizeof(script) / sizeof(script[0]));

    uint64_t slot = ao_ph_register(cr3, swap_va, script, n);
    if ((int64_t)slot < 0) {
        fprintf(stderr, "ph_register failed: %lld\n", (long long)(int64_t)slot);
        return 1;
    }
    printf("installed, slot = %llu\n", (unsigned long long)slot);
    return 0;
}

/* ------------------------------------------------------------------ *
 * Day 2 draw: build a PH_CALL-based script that draws a red quad     *
 * in osu!'s GL context each frame, then runs the original SwapBuffers *
 * ------------------------------------------------------------------ */

/* Reinterpret a float as its uint32 bit pattern (for passing to PH_CALL
 * as a 32-bit arg; cdecl just takes 4 bytes off the stack). */
static uint32_t f2u(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

static int do_install_draw(void)
{
    if (!ao_present()) die("hypervisor not present");

    uint64_t cr3 = ao_get_process_cr3(TARGET_PROCESS);
    if (cr3 == AO_NOT_PRESENT || cr3 == 0) die("osu CR3 not found");
    DWORD pid = find_pid(TARGET_PROCESS);
    if (!pid) die("osu PID not found");

    uint64_t swap_va = resolve_swapbuffers_va_in_osu(pid);
    if (!swap_va) die("SwapBuffers unresolved");

    /* Resolve OpenGL exports from SysWOW64\opengl32.dll + osu's module base. */
    const char* gl_disk = "C:\\Windows\\SysWOW64\\opengl32.dll";
    uint64_t glMatrixMode   = resolve_export_in_osu(pid, GL_MODULE, gl_disk, "glMatrixMode");
    uint64_t glPushMatrix   = resolve_export_in_osu(pid, GL_MODULE, gl_disk, "glPushMatrix");
    uint64_t glLoadIdentity = resolve_export_in_osu(pid, GL_MODULE, gl_disk, "glLoadIdentity");
    uint64_t glPopMatrix    = resolve_export_in_osu(pid, GL_MODULE, gl_disk, "glPopMatrix");
    uint64_t glDisable      = resolve_export_in_osu(pid, GL_MODULE, gl_disk, "glDisable");
    uint64_t glColor3f      = resolve_export_in_osu(pid, GL_MODULE, gl_disk, "glColor3f");
    uint64_t glBegin        = resolve_export_in_osu(pid, GL_MODULE, gl_disk, "glBegin");
    uint64_t glVertex2f     = resolve_export_in_osu(pid, GL_MODULE, gl_disk, "glVertex2f");
    uint64_t glEnd          = resolve_export_in_osu(pid, GL_MODULE, gl_disk, "glEnd");
    if (!glMatrixMode || !glPushMatrix || !glLoadIdentity || !glPopMatrix ||
        !glDisable || !glColor3f || !glBegin || !glVertex2f || !glEnd)
        die("OpenGL export resolution failed");

    printf("osu CR3            = 0x%016llX\n", (unsigned long long)cr3);
    printf("SwapBuffers (osu)  = 0x%08llX\n", (unsigned long long)swap_va);
    printf("glMatrixMode       = 0x%08llX\n", (unsigned long long)glMatrixMode);
    printf("glBegin            = 0x%08llX\n", (unsigned long long)glBegin);
    printf("glVertex2f         = 0x%08llX\n", (unsigned long long)glVertex2f);
    printf("glEnd              = 0x%08llX\n", (unsigned long long)glEnd);

    /* Scratch register assignments. */
    enum {
        R_glMatrixMode   = PH_R0,
        R_glPushMatrix   = PH_R1,
        R_glLoadIdentity = PH_R2,
        R_glPopMatrix    = PH_R3,
        R_glDisable      = PH_R4,
        R_glColor3f      = PH_R5,
        R_glBegin        = PH_R6,
        R_glVertex2f     = PH_R7,
        R_glEnd          = PH_R8,
        /* Values. */
        R_PROJECTION     = PH_R9,
        R_MODELVIEW      = PH_R10,
        R_QUADS          = PH_R11,
        R_FLOAT_A        = PH_R12,  /* reused for color r / vertex x */
        R_FLOAT_B        = PH_R13,  /* reused for color g / vertex y */
        R_FLOAT_C        = PH_R14,  /* reused for color b */
    };

    ph_insn script[PH_SCRIPT_MAX_INSNS];
    uint32_t n = 0;
    #define EMIT(x) do { if (n >= PH_SCRIPT_MAX_INSNS) die("script overflow"); script[n++] = (x); } while(0)

    /* ---- One-time setup: load function VAs & constants into scratch. ---- */
    EMIT(PH_LOADI(R_glMatrixMode,   (uint32_t)glMatrixMode));
    EMIT(PH_LOADI(R_glPushMatrix,   (uint32_t)glPushMatrix));
    EMIT(PH_LOADI(R_glLoadIdentity, (uint32_t)glLoadIdentity));
    EMIT(PH_LOADI(R_glPopMatrix,    (uint32_t)glPopMatrix));
    EMIT(PH_LOADI(R_glDisable,      (uint32_t)glDisable));
    EMIT(PH_LOADI(R_glColor3f,      (uint32_t)glColor3f));
    EMIT(PH_LOADI(R_glBegin,        (uint32_t)glBegin));
    EMIT(PH_LOADI(R_glVertex2f,     (uint32_t)glVertex2f));
    EMIT(PH_LOADI(R_glEnd,          (uint32_t)glEnd));
    EMIT(PH_LOADI(R_PROJECTION,     GL_PROJECTION));
    EMIT(PH_LOADI(R_MODELVIEW,      GL_MODELVIEW));
    EMIT(PH_LOADI(R_QUADS,          GL_QUADS));

    /* ---- Projection matrix push + identity. ---- */
    EMIT(PH_CALL1(0, R_glMatrixMode, R_PROJECTION));
    EMIT(PH_CALL0(0, R_glPushMatrix));
    EMIT(PH_CALL0(0, R_glLoadIdentity));

    /* ---- Modelview matrix push + identity. ---- */
    EMIT(PH_CALL1(0, R_glMatrixMode, R_MODELVIEW));
    EMIT(PH_CALL0(0, R_glPushMatrix));
    EMIT(PH_CALL0(0, R_glLoadIdentity));

    /* ---- Disable depth test & textures so the quad always draws. ---- */
    EMIT(PH_LOADI(R_FLOAT_A, GL_DEPTH_TEST));
    EMIT(PH_CALL1(0, R_glDisable, R_FLOAT_A));
    EMIT(PH_LOADI(R_FLOAT_A, GL_TEXTURE_2D));
    EMIT(PH_CALL1(0, R_glDisable, R_FLOAT_A));

    /* ---- Red color (1.0, 0.0, 0.0). ---- */
    EMIT(PH_LOADI(R_FLOAT_A, f2u(1.0f)));  /* r */
    EMIT(PH_LOADI(R_FLOAT_B, f2u(0.0f)));  /* g */
    EMIT(PH_LOADI(R_FLOAT_C, f2u(0.0f)));  /* b */
    EMIT(PH_CALL3(0, R_glColor3f, R_FLOAT_A, R_FLOAT_B, R_FLOAT_C));

    /* ---- Begin quad. ---- */
    EMIT(PH_CALL1(0, R_glBegin, R_QUADS));

    /* ---- 4 vertices in NDC (-0.9..-0.5). ---- */
    #define VERT(x, y) \
        EMIT(PH_LOADI(R_FLOAT_A, f2u((float)(x)))); \
        EMIT(PH_LOADI(R_FLOAT_B, f2u((float)(y)))); \
        EMIT(PH_CALL2(0, R_glVertex2f, R_FLOAT_A, R_FLOAT_B));
    VERT(-0.9, -0.9);
    VERT(-0.5, -0.9);
    VERT(-0.5, -0.5);
    VERT(-0.9, -0.5);
    #undef VERT

    /* ---- End quad. ---- */
    EMIT(PH_CALL0(0, R_glEnd));

    /* ---- Pop modelview, pop projection. ---- */
    EMIT(PH_CALL1(0, R_glMatrixMode, R_MODELVIEW));
    EMIT(PH_CALL0(0, R_glPopMatrix));
    EMIT(PH_CALL1(0, R_glMatrixMode, R_PROJECTION));
    EMIT(PH_CALL0(0, R_glPopMatrix));

    /* ---- Let the original `jmp dword ptr [...]` run (into gdi32full). ---- */
    EMIT(PH_RETCONT());

    #undef EMIT

    printf("draw script: %u insns\n", n);
    uint64_t slot = ao_ph_register(cr3, swap_va, script, n);
    if ((int64_t)slot < 0) { fprintf(stderr, "ph_register failed: %lld\n", (long long)(int64_t)slot); return 1; }
    printf("installed, slot = %llu\n", (unsigned long long)slot);
    return 0;
}

/* =========================================================================
 * Menu overlay: 3-row toggle menu drawn by the HV into osu's GL context.
 *
 * Config slot layout (per-hook, 16x UINT32):
 *   CFG0  : menu_open (0/1). Script no-ops if 0.
 *   CFG1  : cursor_x_f32bits  (NDC X)
 *   CFG2  : cursor_y_f32bits  (NDC Y)
 *   CFG3-5  : row 0 color R/G/B as f32 bits   (alpha is a constant 0.85)
 *   CFG6-8  : row 1 color R/G/B as f32 bits
 *   CFG9-11 : row 2 color R/G/B as f32 bits
 *
 * All state/hover/on-off logic lives in the tool. Script just draws
 * whatever colors it's handed.
 * ========================================================================= */

struct gl_exports {
    uint64_t glMatrixMode;
    uint64_t glPushMatrix;
    uint64_t glLoadIdentity;
    uint64_t glPopMatrix;
    uint64_t glDisable;
    uint64_t glEnable;
    uint64_t glBlendFunc;
    uint64_t glColor4f;
    uint64_t glBegin;
    uint64_t glVertex2f;
    uint64_t glEnd;
    uint64_t glViewport;
    uint64_t glGetIntegerv;    /* for saving viewport we'll restore */
};

static int resolve_gl_exports(DWORD pid, struct gl_exports* e)
{
    const char* d = "C:\\Windows\\SysWOW64\\opengl32.dll";
    e->glMatrixMode   = resolve_export_in_osu(pid, GL_MODULE, d, "glMatrixMode");
    e->glPushMatrix   = resolve_export_in_osu(pid, GL_MODULE, d, "glPushMatrix");
    e->glLoadIdentity = resolve_export_in_osu(pid, GL_MODULE, d, "glLoadIdentity");
    e->glPopMatrix    = resolve_export_in_osu(pid, GL_MODULE, d, "glPopMatrix");
    e->glDisable      = resolve_export_in_osu(pid, GL_MODULE, d, "glDisable");
    e->glEnable       = resolve_export_in_osu(pid, GL_MODULE, d, "glEnable");
    e->glBlendFunc    = resolve_export_in_osu(pid, GL_MODULE, d, "glBlendFunc");
    e->glColor4f      = resolve_export_in_osu(pid, GL_MODULE, d, "glColor4f");
    e->glBegin        = resolve_export_in_osu(pid, GL_MODULE, d, "glBegin");
    e->glVertex2f     = resolve_export_in_osu(pid, GL_MODULE, d, "glVertex2f");
    e->glEnd          = resolve_export_in_osu(pid, GL_MODULE, d, "glEnd");
    return e->glMatrixMode && e->glPushMatrix && e->glLoadIdentity &&
           e->glPopMatrix && e->glDisable && e->glEnable && e->glBlendFunc &&
           e->glColor4f && e->glBegin && e->glVertex2f && e->glEnd;
}

/* Menu geometry (NDC). Panel is top-left; rows are stacked with equal
 * height and big enough gaps you can comfortably click them. Padding
 * matches the hit-test exactly (which uses ROW_* constants). */
#define MENU_X0 (-0.95f)
#define MENU_X1 (-0.45f)
#define MENU_Y0 ( 0.30f)
#define MENU_Y1 ( 0.95f)
#define ROW_X0  (-0.92f)
#define ROW_X1  (-0.48f)
#define ROW0_Y1 ( 0.88f)
#define ROW0_Y0 ( 0.72f)
#define ROW1_Y1 ( 0.68f)
#define ROW1_Y0 ( 0.52f)
#define ROW2_Y1 ( 0.48f)
#define ROW2_Y0 ( 0.32f)

/* Cfg slot indices (0..15). */
enum {
    CFG_MENU_OPEN    = 0,
    CFG_CURSOR_X     = 1,
    CFG_CURSOR_Y     = 2,
    CFG_R0_R = 3, CFG_R0_G = 4, CFG_R0_B = 5,
    CFG_R1_R = 6, CFG_R1_G = 7, CFG_R1_B = 8,
    CFG_R2_R = 9, CFG_R2_G = 10, CFG_R2_B = 11,
    CFG_VIEWPORT_W = 12,  /* uint px */
    CFG_VIEWPORT_H = 13,  /* uint px */
};

/* Cursor hit-test helper (pure math; used by tool). */
static int cursor_in_row(float cx, float cy, float y0, float y1)
{
    return cx >= ROW_X0 && cx <= ROW_X1 && cy >= y0 && cy <= y1;
}

/* Which row is the cursor over? 0..2, or -1 if none. */
static int hovered_row(float cx, float cy)
{
    if (cursor_in_row(cx, cy, ROW0_Y0, ROW0_Y1)) return 0;
    if (cursor_in_row(cx, cy, ROW1_Y0, ROW1_Y1)) return 1;
    if (cursor_in_row(cx, cy, ROW2_Y0, ROW2_Y1)) return 2;
    return -1;
}

/* Build a script that draws the full menu, reading colors from CFG slots.
 *
 * If `show` is 0, emits a bare PH_RETCONT.
 * Returns insn count. Dies on overflow. */
static uint32_t build_overlay_script(ph_insn* script, uint32_t cap,
                                     const struct gl_exports* e, int show)
{
    uint32_t n = 0;
    #define EMIT(x) do { if (n >= cap) die("script overflow"); script[n++] = (x); } while(0)

    if (!show) {
        EMIT(PH_RETCONT());
        return n;
    }

    enum {
        R_glMatrixMode   = PH_R0,
        R_glPushMatrix   = PH_R1,
        R_glLoadIdentity = PH_R2,
        R_glPopMatrix    = PH_R3,
        R_glDisable      = PH_R4,
        R_glEnable       = PH_R5,
        R_glBlendFunc    = PH_R6,
        R_glColor4f      = PH_R7,
        R_glBegin        = PH_R8,
        R_glVertex2f     = PH_R9,
        R_glEnd          = PH_R10,
        R_PROJECTION     = PH_R11,
        R_MODELVIEW      = PH_R12,
        R_TMP_A          = PH_R13,
        R_TMP_B          = PH_R14,
        R_ALPHA          = PH_R15,  /* holds f32 bits for 0.85 */
    };

    /* Gate: if CFG_MENU_OPEN == 0, skip everything and go straight to RETCONT. */
    /* PH_CMP sets flags; we use PH_JEQ to jump past the whole draw if open==0.
     * The JEQ delta is measured in insns; we fill it in after we know script size. */
    EMIT(PH_LOADI(R_TMP_A, 0));
    EMIT(PH_CMP(PH_CFG0 + CFG_MENU_OPEN, R_TMP_A));
    uint32_t jeq_patch_pc = n;
    EMIT(PH_JEQ(0));  /* delta to be filled in to point at final RETCONT */

    /* Func VAs. */
    EMIT(PH_LOADI(R_glMatrixMode,   (uint32_t)e->glMatrixMode));
    EMIT(PH_LOADI(R_glPushMatrix,   (uint32_t)e->glPushMatrix));
    EMIT(PH_LOADI(R_glLoadIdentity, (uint32_t)e->glLoadIdentity));
    EMIT(PH_LOADI(R_glPopMatrix,    (uint32_t)e->glPopMatrix));
    EMIT(PH_LOADI(R_glDisable,      (uint32_t)e->glDisable));
    EMIT(PH_LOADI(R_glEnable,       (uint32_t)e->glEnable));
    EMIT(PH_LOADI(R_glBlendFunc,    (uint32_t)e->glBlendFunc));
    EMIT(PH_LOADI(R_glColor4f,      (uint32_t)e->glColor4f));
    EMIT(PH_LOADI(R_glBegin,        (uint32_t)e->glBegin));
    EMIT(PH_LOADI(R_glVertex2f,     (uint32_t)e->glVertex2f));
    EMIT(PH_LOADI(R_glEnd,          (uint32_t)e->glEnd));
    EMIT(PH_LOADI(R_PROJECTION,     GL_PROJECTION));
    EMIT(PH_LOADI(R_MODELVIEW,      GL_MODELVIEW));
    EMIT(PH_LOADI(R_ALPHA,          f2u(0.85f)));

    /* Reset projection + modelview. */
    EMIT(PH_CALL1(0, R_glMatrixMode, R_PROJECTION));
    EMIT(PH_CALL0(0, R_glPushMatrix));
    EMIT(PH_CALL0(0, R_glLoadIdentity));
    EMIT(PH_CALL1(0, R_glMatrixMode, R_MODELVIEW));
    EMIT(PH_CALL0(0, R_glPushMatrix));
    EMIT(PH_CALL0(0, R_glLoadIdentity));

    /* Disable depth test + texture, enable blend. */
    EMIT(PH_LOADI(R_TMP_A, GL_DEPTH_TEST));
    EMIT(PH_CALL1(0, R_glDisable, R_TMP_A));
    EMIT(PH_LOADI(R_TMP_A, GL_TEXTURE_2D));
    EMIT(PH_CALL1(0, R_glDisable, R_TMP_A));
    EMIT(PH_LOADI(R_TMP_A, GL_BLEND));
    EMIT(PH_CALL1(0, R_glEnable, R_TMP_A));
    EMIT(PH_LOADI(R_TMP_A, GL_SRC_ALPHA));
    EMIT(PH_LOADI(R_TMP_B, GL_ONE_MINUS_SRC_ALPHA));
    EMIT(PH_CALL2(0, R_glBlendFunc, R_TMP_A, R_TMP_B));

    /* ---- Panel background (dark navy, semi-transparent). ---- */
    EMIT(PH_LOADI(R_TMP_A, f2u(0.05f)));   /* r */
    EMIT(PH_LOADI(R_TMP_B, f2u(0.05f)));   /* g */
    /* Stuff blue into R_PROJECTION as a temp (we reload it later). */
    EMIT(PH_LOADI(R_PROJECTION, f2u(0.15f)));
    EMIT(PH_CALL4(0, R_glColor4f, R_TMP_A, R_TMP_B, R_PROJECTION, R_ALPHA));

    EMIT(PH_LOADI(R_TMP_A, GL_QUADS));
    EMIT(PH_CALL1(0, R_glBegin, R_TMP_A));
    #define VERT(x, y) \
        EMIT(PH_LOADI(R_TMP_A, f2u((float)(x)))); \
        EMIT(PH_LOADI(R_TMP_B, f2u((float)(y)))); \
        EMIT(PH_CALL2(0, R_glVertex2f, R_TMP_A, R_TMP_B));
    VERT(MENU_X0, MENU_Y0);
    VERT(MENU_X1, MENU_Y0);
    VERT(MENU_X1, MENU_Y1);
    VERT(MENU_X0, MENU_Y1);
    EMIT(PH_CALL0(0, R_glEnd));

    /* ---- Row draws. Color read from CFG slots. ---- */
    #define ROW(cfg_r, cfg_g, cfg_b, y0, y1) \
        EMIT(PH_CALL4(0, R_glColor4f, PH_CFG0 + (cfg_r), PH_CFG0 + (cfg_g), \
                                       PH_CFG0 + (cfg_b), R_ALPHA));         \
        EMIT(PH_LOADI(R_TMP_A, GL_QUADS)); \
        EMIT(PH_CALL1(0, R_glBegin, R_TMP_A)); \
        VERT(ROW_X0, y0); VERT(ROW_X1, y0); VERT(ROW_X1, y1); VERT(ROW_X0, y1); \
        EMIT(PH_CALL0(0, R_glEnd));
    ROW(CFG_R0_R, CFG_R0_G, CFG_R0_B, ROW0_Y0, ROW0_Y1);
    ROW(CFG_R1_R, CFG_R1_G, CFG_R1_B, ROW1_Y0, ROW1_Y1);
    ROW(CFG_R2_R, CFG_R2_G, CFG_R2_B, ROW2_Y0, ROW2_Y1);
    #undef ROW

    /* ---- Cursor arrow: small white triangle at (CFG_CURSOR_X, CFG_CURSOR_Y). ----
     * Triangle: tip at (cx, cy), lower-right at (cx+0.03, cy-0.03),
     * lower-left at (cx+0.015, cy-0.045). Needs 3 add operations we don't
     * have as float ops; cheat: pass cursor pos as three pre-computed vertex
     * offsets. For MVP, draw a small white square instead - simpler. */
    /* White color. */
    EMIT(PH_LOADI(R_TMP_A, f2u(1.0f)));
    EMIT(PH_CALL4(0, R_glColor4f, R_TMP_A, R_TMP_A, R_TMP_A, R_ALPHA));
    /* For simplicity use integer math: cursor pos arrives as f32 bits in CFGs.
     * We can't do float arithmetic. Instead: tool computes 4 vertex coords for
     * the cursor-shape and sets them. But we're out of CFG slots (only 16).
     *
     * Trade-off: draw a FIXED-SIZE cursor by having PH_READ4 return the center,
     * then emitting 4 vertices at center +/- offsets... except we can't do
     * float add in the VM. Skip the cursor for MVP — tool draws a Win32 cursor
     * overlay separately, OR we just rely on the fact that mouse coords are
     * visible via OS cursor (which Windows will show during overlay). */

    /* Restore matrix stacks. */
    EMIT(PH_LOADI(R_MODELVIEW, GL_MODELVIEW));   /* reload in case we clobbered */
    EMIT(PH_LOADI(R_PROJECTION, GL_PROJECTION));
    EMIT(PH_CALL1(0, R_glMatrixMode, R_MODELVIEW));
    EMIT(PH_CALL0(0, R_glPopMatrix));
    EMIT(PH_CALL1(0, R_glMatrixMode, R_PROJECTION));
    EMIT(PH_CALL0(0, R_glPopMatrix));

    /* Patch the JEQ at the top to jump HERE (just before final RETCONT). */
    {
        /* PH_JEQ semantics (Host.c: pc = pc + imm; continue;):
         * next-executed insn index = jeq_patch_pc + imm.
         * We want next = n (the RETCONT we're about to emit).
         *  -> imm = n - jeq_patch_pc. */
        int32_t delta = (int32_t)n - (int32_t)jeq_patch_pc;
        script[jeq_patch_pc].imm = (uint32_t)delta;
    }

    /* Final RETCONT (either reached naturally after the draw, or by the gate
     * JEQ if menu_open==0). */
    EMIT(PH_RETCONT());

    #undef VERT
    #undef EMIT
    return n;
}

static int do_install_overlay(void)
{
    if (!ao_present()) die("hypervisor not present");
    uint64_t cr3 = ao_get_process_cr3(TARGET_PROCESS);
    if (cr3 == AO_NOT_PRESENT || cr3 == 0) die("osu CR3 not found");
    DWORD pid = find_pid(TARGET_PROCESS);
    if (!pid) die("osu PID not found");
    uint64_t swap_va = resolve_swapbuffers_va_in_osu(pid);
    if (!swap_va) die("SwapBuffers unresolved");

    struct gl_exports e;
    if (!resolve_gl_exports(pid, &e)) die("GL export resolution failed");

    ph_insn script[PH_SCRIPT_MAX_INSNS];
    uint32_t n = build_overlay_script(script, PH_SCRIPT_MAX_INSNS, &e, 1);

    printf("overlay script: %u insns\n", n);
    uint64_t slot = ao_ph_register(cr3, swap_va, script, n);
    if ((int64_t)slot < 0) { fprintf(stderr, "ph_register failed: %lld\n", (long long)(int64_t)slot); return 1; }
    printf("installed, slot = %llu\n", (unsigned long long)slot);
    return 0;
}

/* ------------------------------------------------------------------------
 * Menu loop: mouse-tracked, click-interactive, toggle with Scroll Lock.
 *
 * - Installs overlay script once with CFG_MENU_OPEN=0 initially.
 * - Scroll Lock press toggles menu_open (push CFG0, no re-register).
 * - While menu is open, sets a WH_MOUSE_LL hook to swallow mouse events
 *   whose cursor is inside the menu rectangle (so clicks don't reach osu!).
 * - Every tick: cursor pos → NDC, compute hovered row, compute per-row
 *   colors from state+hover, push to CFGs.
 * - Mouse click (left button) inside a row while menu open: toggle that
 *   row's state.
 * ------------------------------------------------------------------------ */

static uint32_t g_menu_slot = 0;
static volatile int g_menu_open = 0;
static volatile int g_row_state[3] = {0, 0, 0};
static HWND g_osu_hwnd = NULL;
/* Shared with mouse-hook thread. */
static CRITICAL_SECTION g_menu_cs;
static volatile int g_mouse_x_px = 0, g_mouse_y_px = 0;
static volatile int g_mouse_lclick_pending = 0;  /* set by hook, consumed by main loop */
static volatile uint64_t g_hook_call_count = 0;
static volatile uint64_t g_hook_swallow_count = 0;

static HWND find_osu_hwnd(DWORD pid)
{
    struct { DWORD pid; HWND hwnd; } ctx = { pid, NULL };
    WNDENUMPROC cb = NULL;  /* can't use lambda; define below */
    (void)cb;
    /* Use EnumWindows with a static callback and GetWindowThreadProcessId. */
    HWND found = NULL;
    HWND w = GetTopWindow(NULL);
    while (w) {
        DWORD wpid = 0;
        GetWindowThreadProcessId(w, &wpid);
        if (wpid == pid && IsWindowVisible(w)) {
            char title[64]; title[0] = 0;
            GetWindowTextA(w, title, sizeof(title) - 1);
            /* osu's main window has "osu!" in the title. Skip chatty ancillary ones. */
            if (strstr(title, "osu!") != NULL) { found = w; break; }
        }
        w = GetWindow(w, GW_HWNDNEXT);
    }
    return found;
}

/* Convert client-area pixel coords to the NDC the DRAW SCRIPT uses.
 *
 * osu! applies its own projection transform BEFORE we push ours; even after
 * glLoadIdentity on both matrices, the visible mapping from our NDC to
 * screen pixels is shrunk and flipped. Empirically calibrated at 2560x1440
 * from two points the user reported:
 *   draw-NDC (-0.92, 0.72) -> pixel (1110, 582)
 *   draw-NDC (-0.70, 0.60) -> pixel ( 780, 752)
 *
 * Fit as pixel = a*ndc + b per axis, express a,b as fractions of the window
 * size so the same transform scales with resolution:
 *   scale_x = -1500 px/NDC = -0.586 * client_w  @ 2560
 *   offset_x = -270 px     = -0.105 * client_w
 *   scale_y = -1417 px/NDC = -0.984 * client_h  @ 1440
 *   offset_y = +1602 px    = +1.113 * client_h
 * Resolution-relative so other window sizes still work (osu's projection
 * appears to scale with the window).
 *
 * Inverse (what we need here):
 *   ndc_x = (pixel_x - offset_x) / scale_x
 *   ndc_y = (pixel_y - offset_y) / scale_y
 */
static void px_to_ndc(int x_px, int y_px, int cw, int ch, float* nx, float* ny)
{
    const float sx = -0.586f * (float)cw;
    const float ox = -0.105f * (float)cw;
    const float sy = -0.984f * (float)ch;
    const float oy =  1.113f * (float)ch;
    *nx = ((float)x_px - ox) / sx;
    *ny = ((float)y_px - oy) / sy;
}

/* Low-level mouse hook. Runs on the thread that called SetWindowsHookEx and
 * has a message pump. Captures mouse pos; if menu is open and cursor is in
 * the menu panel region, swallow the event so osu! never sees it. */
static LRESULT CALLBACK ll_mouse_proc(int nCode, WPARAM wParam, LPARAM lParam)
{
    g_hook_call_count++;
    /* The hook callback MUST be fast and MUST NOT throw. If anything looks
     * off, fall through to CallNextHookEx so we never block system input. */
    if (nCode != HC_ACTION) return CallNextHookEx(NULL, nCode, wParam, lParam);
    if (!g_menu_open)        return CallNextHookEx(NULL, nCode, wParam, lParam);
    if (!g_osu_hwnd)         return CallNextHookEx(NULL, nCode, wParam, lParam);
    if (!IsWindow(g_osu_hwnd)) return CallNextHookEx(NULL, nCode, wParam, lParam);

    MSLLHOOKSTRUCT* m = (MSLLHOOKSTRUCT*)lParam;
    POINT p = { m->pt.x, m->pt.y };
    if (!ScreenToClient(g_osu_hwnd, &p)) return CallNextHookEx(NULL, nCode, wParam, lParam);

    EnterCriticalSection(&g_menu_cs);
    g_mouse_x_px = p.x;
    g_mouse_y_px = p.y;
    LeaveCriticalSection(&g_menu_cs);

    RECT cr;
    if (!GetClientRect(g_osu_hwnd, &cr)) return CallNextHookEx(NULL, nCode, wParam, lParam);
    int cw = cr.right - cr.left, ch = cr.bottom - cr.top;
    if (cw <= 0 || ch <= 0) return CallNextHookEx(NULL, nCode, wParam, lParam);

    float nx, ny; px_to_ndc(p.x, p.y, cw, ch, &nx, &ny);
    if (nx >= MENU_X0 && nx <= MENU_X1 && ny >= MENU_Y0 && ny <= MENU_Y1) {
        if (wParam == WM_LBUTTONDOWN) g_mouse_lclick_pending = 1;
        g_hook_swallow_count++;
        return 1;
    }
    return CallNextHookEx(NULL, nCode, wParam, lParam);
}

/* Message pump thread hosting the low-level mouse hook. */
static DWORD WINAPI mouse_hook_thread(LPVOID unused)
{
    (void)unused;
    HHOOK h = SetWindowsHookExA(WH_MOUSE_LL, ll_mouse_proc, GetModuleHandleA(NULL), 0);
    if (!h) { fprintf(stderr, "SetWindowsHookEx failed: %lu\n", GetLastError()); return 1; }
    MSG msg;
    while (GetMessageA(&msg, NULL, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
    UnhookWindowsHookEx(h);
    return 0;
}

static int do_menu_loop(void)
{
    if (!ao_present()) die("hypervisor not present");
    uint64_t cr3 = ao_get_process_cr3(TARGET_PROCESS);
    if (cr3 == AO_NOT_PRESENT || cr3 == 0) die("osu CR3 not found");
    DWORD pid = find_pid(TARGET_PROCESS);
    if (!pid) die("osu PID not found");
    uint64_t swap_va = resolve_swapbuffers_va_in_osu(pid);
    if (!swap_va) die("SwapBuffers unresolved");

    struct gl_exports e;
    if (!resolve_gl_exports(pid, &e)) die("GL export resolution failed");

    g_osu_hwnd = find_osu_hwnd(pid);
    if (!g_osu_hwnd) die("osu! HWND not found");
    printf("osu HWND = %p\n", (void*)g_osu_hwnd);

    ph_insn script[PH_SCRIPT_MAX_INSNS];
    uint32_t n = build_overlay_script(script, PH_SCRIPT_MAX_INSNS, &e, 1);
    printf("menu script: %u insns\n", n);
    uint64_t slot = ao_ph_register(cr3, swap_va, script, n);
    if ((int64_t)slot < 0) { fprintf(stderr, "ph_register failed: %lld\n", (long long)(int64_t)slot); return 1; }
    g_menu_slot = (uint32_t)slot;
    printf("installed, slot = %u\n", g_menu_slot);

    /* Initial CFG: menu closed, rows all off. */
    ao_ph_set_cfg(g_menu_slot, CFG_MENU_OPEN, 0);
    for (int i = 0; i < 3; i++) g_row_state[i] = 0;

    /* Set up mouse hook thread. */
    InitializeCriticalSection(&g_menu_cs);
    HANDLE th = CreateThread(NULL, 0, mouse_hook_thread, NULL, 0, NULL);
    if (!th) die("CreateThread failed");

    printf("press SCROLL LOCK to toggle menu, left-click rows to toggle, Ctrl+C to quit\n");

    int last_key_down = 0;
    int last_menu_open = 0;
    DWORD last_diag_ms = GetTickCount();
    while (1) {
        SHORT s = GetAsyncKeyState(VK_SCROLL);
        int key_down = (s & 0x8000) != 0;
        if (key_down && !last_key_down) {
            g_menu_open = !g_menu_open;
            ao_ph_set_cfg(g_menu_slot, CFG_MENU_OPEN, (uint32_t)g_menu_open);
            printf("menu %s\n", g_menu_open ? "OPEN" : "CLOSED");
        }
        last_key_down = key_down;

        /* Diagnostics every ~1s. */
        if (GetTickCount() - last_diag_ms >= 1000) {
            RECT cr; GetClientRect(g_osu_hwnd, &cr);
            int cw = cr.right - cr.left, ch = cr.bottom - cr.top;
            float nx = -2.0f, ny = -2.0f;
            if (cw > 0 && ch > 0)
                px_to_ndc(g_mouse_x_px, g_mouse_y_px, cw, ch, &nx, &ny);
            printf("  hook_calls=%llu swallowed=%llu cursor=(%d,%d) ndc=(%.3f,%.3f) client=(%dx%d) hov=%d open=%d\n",
                   (unsigned long long)g_hook_call_count,
                   (unsigned long long)g_hook_swallow_count,
                   g_mouse_x_px, g_mouse_y_px, nx, ny, cw, ch,
                   hovered_row(nx, ny), g_menu_open);
            last_diag_ms = GetTickCount();
        }

        if (g_menu_open) {
            /* Cursor → NDC. */
            EnterCriticalSection(&g_menu_cs);
            int mx = g_mouse_x_px, my = g_mouse_y_px;
            int lclick = g_mouse_lclick_pending;
            g_mouse_lclick_pending = 0;
            LeaveCriticalSection(&g_menu_cs);
            RECT cr; GetClientRect(g_osu_hwnd, &cr);
            int cw = cr.right - cr.left, ch = cr.bottom - cr.top;
            float nx = -2.0f, ny = -2.0f;
            if (cw > 0 && ch > 0) px_to_ndc(mx, my, cw, ch, &nx, &ny);
            int hov = hovered_row(nx, ny);

            /* Left-click in a row → toggle. */
            if (lclick && hov >= 0 && hov < 3) {
                g_row_state[hov] = !g_row_state[hov];
                printf("row %d → %s\n", hov, g_row_state[hov] ? "ON" : "OFF");
            }

            /* Push cursor pos to CFG. */
            ao_ph_set_cfg(g_menu_slot, CFG_CURSOR_X, f2u(nx));
            ao_ph_set_cfg(g_menu_slot, CFG_CURSOR_Y, f2u(ny));

            /* Per-row colors. On = green, off = gray. Hovered = brightened. */
            for (int i = 0; i < 3; i++) {
                float r, g, b;
                if (g_row_state[i]) { r = 0.10f; g = 0.70f; b = 0.10f; }
                else                { r = 0.30f; g = 0.30f; b = 0.30f; }
                if (hov == i) { r += 0.15f; g += 0.15f; b += 0.15f; }
                uint32_t cfg_r = CFG_R0_R + i * 3;
                ao_ph_set_cfg(g_menu_slot, cfg_r + 0, f2u(r));
                ao_ph_set_cfg(g_menu_slot, cfg_r + 1, f2u(g));
                ao_ph_set_cfg(g_menu_slot, cfg_r + 2, f2u(b));
            }
        }

        if (last_menu_open != g_menu_open) {
            last_menu_open = g_menu_open;
        }

        Sleep(16);
    }
    /* unreachable */
}

static int do_watch(int argc, char** argv)
{
    int seconds = 10;
    if (argc > 2) seconds = atoi(argv[2]);
    if (seconds <= 0) seconds = 10;

    ph_log_entry buf[256];
    uint64_t total = 0;
    DWORD t0 = GetTickCount();
    DWORD t_last_print = t0;

    printf("watching for %d seconds, draining ph_log...\n", seconds);

    while ((int)(GetTickCount() - t0) < seconds * 1000) {
        uint64_t n = ao_ph_log_read(buf, 256);
        if ((int64_t)n > 0) {
            total += n;
            /* Dump first entry per batch so we see cr3/va in case something's off. */
            printf("+%llu entries (cr3=%016llX va=%016llX a=%llu b=%llu)\n",
                   (unsigned long long)n,
                   (unsigned long long)buf[0].cr3,
                   (unsigned long long)buf[0].va,
                   (unsigned long long)buf[0].a,
                   (unsigned long long)buf[0].b);
        }
        /* second-ish rate line */
        DWORD now = GetTickCount();
        if (now - t_last_print >= 1000) {
            double elapsed = (now - t0) / 1000.0;
            printf("[t=%.1fs] total hits = %llu  (%.1f/sec avg)\n",
                   elapsed, (unsigned long long)total,
                   total / (elapsed > 0 ? elapsed : 1));
            t_last_print = now;
        }
        Sleep(50);
    }

    double elapsed = (GetTickCount() - t0) / 1000.0;
    printf("done: %llu hits in %.1fs (%.1f/sec avg)\n",
           (unsigned long long)total, elapsed,
           total / (elapsed > 0 ? elapsed : 1));
    return 0;
}

static int do_status(void)
{
    ph_status s;
    memset(&s, 0, sizeof(s));
    uint64_t rc = ao_ph_status(&s);
    if ((int64_t)rc < 0) {
        fprintf(stderr, "ph_status failed: %lld\n", (long long)(int64_t)rc);
        return 1;
    }
    printf("num_active = %u  dispatch = %llu  faults = %llu\n",
           s.num_active,
           (unsigned long long)s.dispatch_count,
           (unsigned long long)s.fault_count);
    for (int i = 0; i < PH_MAX_HOOKS; i++) {
        if (!s.entries[i].active) continue;
        printf("  [%d] cr3=%016llX va=%016llX hits=%llu insns=%u fault=%u@%u\n",
               i,
               (unsigned long long)s.entries[i].target_cr3,
               (unsigned long long)s.entries[i].target_va,
               (unsigned long long)s.entries[i].hits,
               s.entries[i].insn_count,
               s.entries[i].last_fault_reason,
               s.entries[i].last_fault_pc);
    }
    return 0;
}

static int do_remove(void)
{
    /* No elegant "find the one slot that's ours" helper; the user can call
     * `status` and then `hook_tool unregister <slot>` if they want precision.
     * For the day-1 tool, clear all PH hooks owned by this framework. If
     * other hooks (e.g. the osu! AR hook) are present we don't want to clobber
     * them, so for safety, only unregister the slot whose target VA matches
     * our SwapBuffers address.
     */
    DWORD pid = find_pid(TARGET_PROCESS);
    if (!pid) die("osu! PID not found (nothing to remove?)");
    uint64_t want_va = resolve_swapbuffers_va_in_osu(pid);
    ph_status s;
    memset(&s, 0, sizeof(s));
    if ((int64_t)ao_ph_status(&s) < 0) die("ph_status failed");
    int removed = 0;
    for (int i = 0; i < PH_MAX_HOOKS; i++) {
        if (!s.entries[i].active) continue;
        if (s.entries[i].target_va == want_va) {
            uint64_t rc = ao_ph_unregister((uint32_t)i);
            printf("unregister slot %d -> %lld\n", i, (long long)(int64_t)rc);
            removed++;
        }
    }
    if (!removed) printf("no matching hook found\n");
    return 0;
}

static void usage(void)
{
    fprintf(stderr,
        "usage: osu_overlay.exe <install|install_draw|install_overlay|menu_loop|watch [seconds]|status|remove>\n"
        "  install         Day-1 log-only hook on gdi32!SwapBuffers\n"
        "  install_draw    Day-2 draw red quad in osu's GL context each frame\n"
        "  install_overlay Phase A: install translucent overlay panel (always on)\n"
        "  menu_loop       Full menu: Scroll Lock toggles, mouse clicks toggle rows\n"
        "  watch [sec]     drain the PH_LOG ring and print rate\n"
        "  status          dump hook table + hits\n"
        "  remove          remove this hook (matches by SwapBuffers VA)\n");
}

int main(int argc, char** argv)
{
    if (argc < 2) { usage(); return 1; }
    if (!strcmp(argv[1], "install"))         return do_install();
    if (!strcmp(argv[1], "install_draw"))    return do_install_draw();
    if (!strcmp(argv[1], "install_overlay")) return do_install_overlay();
    if (!strcmp(argv[1], "menu_loop"))       return do_menu_loop();
    if (!strcmp(argv[1], "watch"))           return do_watch(argc, argv);
    if (!strcmp(argv[1], "status"))          return do_status();
    if (!strcmp(argv[1], "remove"))          return do_remove();
    usage();
    return 1;
}
