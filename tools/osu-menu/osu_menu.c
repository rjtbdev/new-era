/*
 * osu_menu.c — minimal localhost web menu for the ao HV overlay.
 *
 * Starts an HTTP + WS server on localhost:8080. Serves index.html on GET /.
 * On WS Upgrade: accepts browser clients, dispatches {cmd,key,val} messages
 * into ao hypercalls (sets PH_CFG slots), pushes status back.
 *
 * Build:
 *   ml64 /nologo /c ao_call.asm      (already built)
 *   cl /nologo /O2 /W3 osu_menu.c ao_call.obj /link /out:osu_menu.exe \
 *      psapi.lib user32.lib ws2_32.lib bcrypt.lib
 *
 * Usage:
 *   D:\osu_menu.exe
 *   # then visit http://localhost:8080 in a browser
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <psapi.h>
#include <tlhelp32.h>
#include <bcrypt.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "bcrypt.lib")

#include "ao_client.h"

#define TARGET_PROCESS "osu!.exe"
#define TARGET_MODULE  "gdi32.dll"
#define TARGET_EXPORT  "SwapBuffers"
#define GL_MODULE      "opengl32.dll"

#define GL_PROJECTION    0x1701
#define GL_MODELVIEW     0x1700
#define GL_DEPTH_TEST    0x0B71
#define GL_TEXTURE_2D    0x0DE1
#define GL_QUADS         0x0007
#define GL_BLEND         0x0BE2
#define GL_SRC_ALPHA     0x0302
#define GL_ONE_MINUS_SRC_ALPHA 0x0303

#define LISTEN_PORT 8080

/* Overlay hook CFG slots. In-game panel is a small status indicator whose
 * bar color reflects AR hook state. Browser menu is the real UI. */
enum {
    OVL_CFG_MENU_OPEN = 0,
    OVL_CFG_BAR_R    = 3,
    OVL_CFG_BAR_G    = 4,
    OVL_CFG_BAR_B    = 5,
};

/* AR hook CFG slots (a different hook entry). */
enum {
    AR_CFG_ENABLE  = 0,  /* 0 = pass-through, 1 = override EAX */
    AR_CFG_PREEMPT = 1,  /* preempt interval in ms */
};

/* ---- osu! AR hook signature (from the working hook_tool install_sig cmd). ---- */
static const char* AR_SIG_HEX  =
    "DD050000000083EC08DD1C24DD050000000083EC08DD1C24DD050000000083EC08DD1C248BCE8B018B4000FF5000DD5DE0F20F1045E0F20F2CC089461C";
static const char* AR_SIG_MASK =
    "xx????xxxxxxxx????xxxxxxxx????xxxxxxxxxxxx?xx?xxxxxxxxxxxxxxx";
#define AR_HOOK_OFFSET  0x3A
#define AR_SCAN_MB      256

/* ---- global state ---- */
static uint32_t g_overlay_slot = 0xFFFFFFFF;
static uint32_t g_ar_slot      = 0xFFFFFFFF;
static volatile int g_menu_open  = 1;   /* overlay visible */
static volatile int g_ar_enabled = 0;
static volatile float g_ar_value = 10.0f;  /* osu AR value 0..11 */
static DWORD    g_osu_pid = 0;
static uint64_t g_osu_cr3 = 0;
static uint64_t g_osu_base = 0;

/* osu!stable approach-rate preempt in ms. See: osu wiki (AR).
 *   AR 0   -> 1800
 *   AR 5   -> 1200
 *   AR 10  -> 450
 *   AR 11  -> 300  (extrapolated; outside standard range)
 */
static uint32_t ar_to_preempt(float ar) {
    if (ar < 0)  ar = 0;
    if (ar > 11) ar = 11;
    float ms;
    if (ar < 5)       ms = 1800.0f - 120.0f * ar;              /* AR 0..5: 1800..1200 */
    else              ms = 1200.0f - 150.0f * (ar - 5.0f);     /* AR 5..11: 1200..300 */
    if (ms < 100) ms = 100;
    return (uint32_t)(ms + 0.5f);
}

static void die(const char* m) { fprintf(stderr, "error: %s\n", m); exit(1); }

/* forward decls — C needs them before first use. */
static void push_ar_cfg(void);
static void push_overlay_cfg(void);
static void push_state_to_browsers(void);

/* ============================================================== *
 * helpers lifted/simplified from osu_overlay.c                    *
 * ============================================================== */

static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static DWORD find_pid(const char* name) {
    HANDLE s = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (s == INVALID_HANDLE_VALUE) return 0;
    PROCESSENTRY32 pe = { .dwSize = sizeof(pe) };
    DWORD pid = 0;
    if (Process32First(s, &pe)) do {
        if (_stricmp(pe.szExeFile, name) == 0) { pid = pe.th32ProcessID; break; }
    } while (Process32Next(s, &pe));
    CloseHandle(s);
    return pid;
}

static uint64_t find_wow64_module_base(DWORD pid, const char* dll) {
    HANDLE p = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid);
    if (!p) return 0;
    HMODULE mods[1024]; DWORD need = 0; uint64_t base = 0;
    if (EnumProcessModulesEx(p, mods, sizeof(mods), &need, LIST_MODULES_32BIT)) {
        DWORD cnt = need / sizeof(HMODULE);
        for (DWORD i = 0; i < cnt; i++) {
            char n[MAX_PATH];
            if (GetModuleBaseNameA(p, mods[i], n, sizeof(n))
                && _stricmp(n, dll) == 0) { base = (uint64_t)(uintptr_t)mods[i]; break; }
        }
    }
    CloseHandle(p);
    return base;
}

static uint32_t find_export_rva(const char* path, const char* name) {
    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
    if (f == INVALID_HANDLE_VALUE) return 0;
    HANDLE m = CreateFileMappingA(f, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(f); if (!m) return 0;
    void* v = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(m); if (!v) return 0;

    uint32_t rva = 0;
    IMAGE_DOS_HEADER* dos = (IMAGE_DOS_HEADER*)v;
    if (dos->e_magic == IMAGE_DOS_SIGNATURE) {
        IMAGE_NT_HEADERS32* nt = (IMAGE_NT_HEADERS32*)((char*)v + dos->e_lfanew);
        if (nt->Signature == IMAGE_NT_SIGNATURE) {
            IMAGE_DATA_DIRECTORY* ed = &nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
            if (ed->VirtualAddress && ed->Size) {
                IMAGE_SECTION_HEADER* sec = IMAGE_FIRST_SECTION(nt);
                #define XLAT(r) ({ uint32_t _o=0; for (DWORD _i=0;_i<nt->FileHeader.NumberOfSections;_i++) if ((r)>=sec[_i].VirtualAddress && (r)<sec[_i].VirtualAddress+sec[_i].Misc.VirtualSize) { _o=sec[_i].PointerToRawData+((r)-sec[_i].VirtualAddress); break; } _o; })
                /* MSVC: no statement exprs; inline instead. */
                #undef XLAT
                uint32_t eo = 0;
                for (DWORD i = 0; i < nt->FileHeader.NumberOfSections; i++)
                    if (ed->VirtualAddress >= sec[i].VirtualAddress && ed->VirtualAddress < sec[i].VirtualAddress + sec[i].Misc.VirtualSize)
                        { eo = sec[i].PointerToRawData + (ed->VirtualAddress - sec[i].VirtualAddress); break; }
                if (eo) {
                    IMAGE_EXPORT_DIRECTORY* exp = (IMAGE_EXPORT_DIRECTORY*)((char*)v + eo);
                    uint32_t aon=0, aono=0, aof=0;
                    for (DWORD i = 0; i < nt->FileHeader.NumberOfSections; i++) {
                        #define XR(r, d) if ((r)>=sec[i].VirtualAddress && (r)<sec[i].VirtualAddress+sec[i].Misc.VirtualSize) (d)=sec[i].PointerToRawData+((r)-sec[i].VirtualAddress);
                        XR(exp->AddressOfNames, aon);
                        XR(exp->AddressOfNameOrdinals, aono);
                        XR(exp->AddressOfFunctions, aof);
                        #undef XR
                    }
                    if (aon && aono && aof) {
                        uint32_t* names = (uint32_t*)((char*)v + aon);
                        uint16_t* ords  = (uint16_t*)((char*)v + aono);
                        uint32_t* funcs = (uint32_t*)((char*)v + aof);
                        for (DWORD i = 0; i < exp->NumberOfNames; i++) {
                            uint32_t nr = names[i], no = 0;
                            for (DWORD j = 0; j < nt->FileHeader.NumberOfSections; j++)
                                if (nr >= sec[j].VirtualAddress && nr < sec[j].VirtualAddress+sec[j].Misc.VirtualSize)
                                    { no = sec[j].PointerToRawData + (nr - sec[j].VirtualAddress); break; }
                            if (no && strcmp((char*)v + no, name) == 0) { rva = funcs[ords[i]]; break; }
                        }
                    }
                }
            }
        }
    }
    UnmapViewOfFile(v);
    return rva;
}

static uint64_t resolve_export(DWORD pid, const char* mod, const char* disk, const char* name) {
    uint64_t base = find_wow64_module_base(pid, mod);
    if (!base) return 0;
    uint32_t rva = find_export_rva(disk, name);
    if (!rva) return 0;
    return base + rva;
}

/* ============================================================== *
 * build overlay script (3 rows, colors from CFG)                  *
 * ============================================================== */

struct gl_exports {
    uint64_t glMatrixMode, glPushMatrix, glLoadIdentity, glPopMatrix;
    uint64_t glDisable, glEnable, glBlendFunc, glColor4f;
    uint64_t glBegin, glVertex2f, glEnd;
};

static int resolve_gl(DWORD pid, struct gl_exports* e) {
    const char* d = "C:\\Windows\\SysWOW64\\opengl32.dll";
    e->glMatrixMode   = resolve_export(pid, GL_MODULE, d, "glMatrixMode");
    e->glPushMatrix   = resolve_export(pid, GL_MODULE, d, "glPushMatrix");
    e->glLoadIdentity = resolve_export(pid, GL_MODULE, d, "glLoadIdentity");
    e->glPopMatrix    = resolve_export(pid, GL_MODULE, d, "glPopMatrix");
    e->glDisable      = resolve_export(pid, GL_MODULE, d, "glDisable");
    e->glEnable       = resolve_export(pid, GL_MODULE, d, "glEnable");
    e->glBlendFunc    = resolve_export(pid, GL_MODULE, d, "glBlendFunc");
    e->glColor4f      = resolve_export(pid, GL_MODULE, d, "glColor4f");
    e->glBegin        = resolve_export(pid, GL_MODULE, d, "glBegin");
    e->glVertex2f     = resolve_export(pid, GL_MODULE, d, "glVertex2f");
    e->glEnd          = resolve_export(pid, GL_MODULE, d, "glEnd");
    return e->glMatrixMode && e->glPushMatrix && e->glLoadIdentity && e->glPopMatrix
        && e->glDisable && e->glEnable && e->glBlendFunc && e->glColor4f
        && e->glBegin && e->glVertex2f && e->glEnd;
}

/* In-game status indicator: tiny panel top-left, single colored bar inside. */
#define MENU_X0 (-0.95f)
#define MENU_X1 (-0.75f)
#define MENU_Y0 ( 0.82f)
#define MENU_Y1 ( 0.95f)
#define ROW_X0  (-0.93f)
#define ROW_X1  (-0.77f)
#define ROW0_Y0 ( 0.84f)
#define ROW0_Y1 ( 0.93f)

static uint32_t build_script(ph_insn* s, uint32_t cap, const struct gl_exports* e) {
    uint32_t n = 0;
    #define EMIT(x) do { if (n >= cap) die("script overflow"); s[n++] = (x); } while(0)
    enum {
        R_glMatrixMode=PH_R0, R_glPushMatrix, R_glLoadIdentity, R_glPopMatrix,
        R_glDisable, R_glEnable, R_glBlendFunc, R_glColor4f,
        R_glBegin, R_glVertex2f, R_glEnd,
        R_PROJECTION, R_MODELVIEW, R_TMP_A, R_TMP_B, R_ALPHA
    };

    /* Gate on OVL_CFG_MENU_OPEN. */
    EMIT(PH_LOADI(R_TMP_A, 0));
    EMIT(PH_CMP(PH_CFG0 + OVL_CFG_MENU_OPEN, R_TMP_A));
    uint32_t jeq_pc = n;
    EMIT(PH_JEQ(0));  /* patched below */

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

    EMIT(PH_CALL1(0, R_glMatrixMode, R_PROJECTION));
    EMIT(PH_CALL0(0, R_glPushMatrix));
    EMIT(PH_CALL0(0, R_glLoadIdentity));
    EMIT(PH_CALL1(0, R_glMatrixMode, R_MODELVIEW));
    EMIT(PH_CALL0(0, R_glPushMatrix));
    EMIT(PH_CALL0(0, R_glLoadIdentity));

    EMIT(PH_LOADI(R_TMP_A, GL_DEPTH_TEST));
    EMIT(PH_CALL1(0, R_glDisable, R_TMP_A));
    EMIT(PH_LOADI(R_TMP_A, GL_TEXTURE_2D));
    EMIT(PH_CALL1(0, R_glDisable, R_TMP_A));
    EMIT(PH_LOADI(R_TMP_A, GL_BLEND));
    EMIT(PH_CALL1(0, R_glEnable, R_TMP_A));
    EMIT(PH_LOADI(R_TMP_A, GL_SRC_ALPHA));
    EMIT(PH_LOADI(R_TMP_B, GL_ONE_MINUS_SRC_ALPHA));
    EMIT(PH_CALL2(0, R_glBlendFunc, R_TMP_A, R_TMP_B));

    /* Panel bg (dark). */
    EMIT(PH_LOADI(R_TMP_A, f2u(0.05f)));
    EMIT(PH_LOADI(R_TMP_B, f2u(0.05f)));
    EMIT(PH_LOADI(R_PROJECTION, f2u(0.15f)));
    EMIT(PH_CALL4(0, R_glColor4f, R_TMP_A, R_TMP_B, R_PROJECTION, R_ALPHA));
    EMIT(PH_LOADI(R_TMP_A, GL_QUADS));
    EMIT(PH_CALL1(0, R_glBegin, R_TMP_A));
    #define V(x,y) \
        EMIT(PH_LOADI(R_TMP_A, f2u((float)(x)))); \
        EMIT(PH_LOADI(R_TMP_B, f2u((float)(y)))); \
        EMIT(PH_CALL2(0, R_glVertex2f, R_TMP_A, R_TMP_B));
    V(MENU_X0, MENU_Y0); V(MENU_X1, MENU_Y0); V(MENU_X1, MENU_Y1); V(MENU_X0, MENU_Y1);
    EMIT(PH_CALL0(0, R_glEnd));

    /* Status bar inside the panel: color from CFG (set to green when AR hook
     * is enabled, gray when disabled). */
    EMIT(PH_CALL4(0, R_glColor4f, PH_CFG0+OVL_CFG_BAR_R, PH_CFG0+OVL_CFG_BAR_G,
                                   PH_CFG0+OVL_CFG_BAR_B, R_ALPHA));
    EMIT(PH_LOADI(R_TMP_A, GL_QUADS));
    EMIT(PH_CALL1(0, R_glBegin, R_TMP_A));
    V(ROW_X0, ROW0_Y0); V(ROW_X1, ROW0_Y0); V(ROW_X1, ROW0_Y1); V(ROW_X0, ROW0_Y1);
    EMIT(PH_CALL0(0, R_glEnd));
    #undef V

    EMIT(PH_LOADI(R_MODELVIEW,  GL_MODELVIEW));
    EMIT(PH_LOADI(R_PROJECTION, GL_PROJECTION));
    EMIT(PH_CALL1(0, R_glMatrixMode, R_MODELVIEW));
    EMIT(PH_CALL0(0, R_glPopMatrix));
    EMIT(PH_CALL1(0, R_glMatrixMode, R_PROJECTION));
    EMIT(PH_CALL0(0, R_glPopMatrix));

    s[jeq_pc].imm = (uint32_t)(n - jeq_pc);
    EMIT(PH_RETCONT());
    #undef EMIT
    return n;
}

/* ============================================================== *
 * push current row colors to HV                                    *
 * ============================================================== */

/* Push overlay CFG: bar color reflects AR hook state, panel visibility is
 * independent (future: hide overlay entirely from browser). */
/* ---------- sig scan (lifted & simplified from hook_tool.c) ---------- */

static int hexchar(int c) {
    if (c>='0'&&c<='9') return c-'0';
    if (c>='a'&&c<='f') return c-'a'+10;
    if (c>='A'&&c<='F') return c-'A'+10;
    return -1;
}
static int parse_pattern(const char* hex, uint8_t* out, uint32_t max) {
    uint32_t n = 0;
    while (*hex && n < max) {
        while (*hex==' '||*hex=='\t') hex++;
        if (!*hex) break;
        int hi = hexchar(*hex++); if (hi<0) return -1;
        int lo = hexchar(*hex++); if (lo<0) return -1;
        out[n++] = (uint8_t)((hi<<4)|lo);
    }
    return (int)n;
}
static int sig_scan_range(uint64_t cr3, uint64_t start_va, uint64_t end_va,
                          const uint8_t* pat, const char* mask, uint32_t plen,
                          uint64_t* match_out) {
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
                    if (mask[j]=='?') continue;
                    if (window[i+j] != pat[j]) { ok = 0; break; }
                }
                if (ok) { *match_out = (va - carry) + i; return 1; }
            }
            carry = plen - 1;
            memmove(window, window + have - carry, carry);
        } else carry = have;
        va += 0x1000;
    }
    return 0;
}

/* ---------- AR hook recipe: CFG-driven, enable-gated ---------- */

/* Script:
 *   LOADI  R0, 0
 *   CMP    CFG_ENABLE, R0           # flagEq = (enable == 0)
 *   JEQ    skip                     # if disabled, fall through unchanged
 *   MOV    EAX, CFG_PREEMPT         # override EAX with preempt value
 * skip:
 *   RETCONT                         # run original `mov [esi+1C], eax`
 *   HALT
 */
static uint32_t build_ar_script(ph_insn* s, uint32_t cap) {
    uint32_t n = 0;
    #define E(x) do { if (n>=cap) die("script overflow"); s[n++]=(x); } while(0)
    E(PH_LOADI(PH_R0, 0));
    E(PH_CMP(PH_CFG0+AR_CFG_ENABLE, PH_R0));
    uint32_t jeq_pc = n;
    E(PH_JEQ(0));                                    /* patched */
    E(PH_MOV(PH_GR_EAX, PH_CFG0+AR_CFG_PREEMPT));
    s[jeq_pc].imm = (uint32_t)(n - jeq_pc);
    E(PH_RETCONT());
    E(PH_HALT());
    #undef E
    return n;
}

/* install AR hook by sig-scanning osu's module. Returns 0 on success. */
static int install_ar_hook(void) {
    if (g_ar_slot < PH_MAX_HOOKS) return 0;  /* already installed */
    if (!g_osu_cr3 || !g_osu_base) return -1;

    uint8_t pat[256];
    int plen = parse_pattern(AR_SIG_HEX, pat, sizeof(pat));
    if (plen <= 0) return -2;
    if ((int)strlen(AR_SIG_MASK) != plen) return -3;

    uint64_t end = g_osu_base + ((uint64_t)AR_SCAN_MB * 1024 * 1024);
    uint64_t found = 0;
    if (!sig_scan_range(g_osu_cr3, g_osu_base, end, pat, AR_SIG_MASK, (uint32_t)plen, &found))
        return -4;

    uint64_t hook_va = found + AR_HOOK_OFFSET;
    printf("[ar] sig match @ 0x%llX -> hook VA 0x%llX\n",
           (unsigned long long)found, (unsigned long long)hook_va);

    ph_insn script[PH_SCRIPT_MAX_INSNS];
    uint32_t n = build_ar_script(script, PH_SCRIPT_MAX_INSNS);
    uint64_t slot = ao_ph_register(g_osu_cr3, hook_va, script, n);
    if ((int64_t)slot < 0) return -5;
    g_ar_slot = (uint32_t)slot;
    printf("[ar] installed slot=%u insns=%u\n", g_ar_slot, n);

    /* push initial CFG. */
    push_ar_cfg();
    return 0;
}

static void uninstall_ar_hook(void) {
    if (g_ar_slot < PH_MAX_HOOKS) {
        ao_ph_unregister(g_ar_slot);
        printf("[ar] uninstalled slot=%u\n", g_ar_slot);
        g_ar_slot = 0xFFFFFFFF;
    }
}

static void push_overlay_cfg(void) {
    if (g_overlay_slot >= PH_MAX_HOOKS) return;
    float r, g, b;
    if (g_ar_slot < PH_MAX_HOOKS && g_ar_enabled) {
        /* AR hook installed AND enabled → green bar. */
        r = 0.15f; g = 0.75f; b = 0.20f;
    } else if (g_ar_slot < PH_MAX_HOOKS) {
        /* installed but off → amber. */
        r = 0.80f; g = 0.60f; b = 0.10f;
    } else {
        /* not installed → gray. */
        r = 0.35f; g = 0.35f; b = 0.35f;
    }
    ao_ph_set_cfg(g_overlay_slot, OVL_CFG_BAR_R, f2u(r));
    ao_ph_set_cfg(g_overlay_slot, OVL_CFG_BAR_G, f2u(g));
    ao_ph_set_cfg(g_overlay_slot, OVL_CFG_BAR_B, f2u(b));
    ao_ph_set_cfg(g_overlay_slot, OVL_CFG_MENU_OPEN, (uint32_t)g_menu_open);
}

static void push_ar_cfg(void) {
    if (g_ar_slot >= PH_MAX_HOOKS) return;
    uint32_t preempt = ar_to_preempt(g_ar_value);
    ao_ph_set_cfg(g_ar_slot, AR_CFG_ENABLE,  (uint32_t)g_ar_enabled);
    ao_ph_set_cfg(g_ar_slot, AR_CFG_PREEMPT, preempt);
}

/* ============================================================== *
 * tiny WebSocket + HTTP server                                     *
 * ============================================================== */

/* base64 encode (for Sec-WebSocket-Accept). */
static void b64(const uint8_t* in, int n, char* out) {
    static const char T[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    int j = 0;
    for (int i = 0; i < n; i += 3) {
        uint32_t v = (uint32_t)in[i] << 16;
        if (i+1 < n) v |= (uint32_t)in[i+1] << 8;
        if (i+2 < n) v |= (uint32_t)in[i+2];
        out[j++] = T[(v>>18)&63];
        out[j++] = T[(v>>12)&63];
        out[j++] = (i+1<n) ? T[(v>>6)&63]  : '=';
        out[j++] = (i+2<n) ? T[v&63]       : '=';
    }
    out[j] = 0;
}

static void sha1(const uint8_t* data, int len, uint8_t out[20]) {
    BCRYPT_ALG_HANDLE alg = NULL;
    BCRYPT_HASH_HANDLE h = NULL;
    BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA1_ALGORITHM, NULL, 0);
    BCryptCreateHash(alg, &h, NULL, 0, NULL, 0, 0);
    BCryptHashData(h, (PUCHAR)data, (ULONG)len, 0);
    BCryptFinishHash(h, out, 20, 0);
    BCryptDestroyHash(h);
    BCryptCloseAlgorithmProvider(alg, 0);
}

static const char INDEX_HTML[] =
"<!doctype html><html><head><meta charset=utf-8><title>osu! HV menu</title>"
"<style>"
"body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#0b0f1a;color:#dbe4f0;margin:0;padding:40px;max-width:500px;}"
"h1{font-weight:500;color:#9fbce0;margin:0 0 24px 0;font-size:18px;letter-spacing:2px;}"
"h1 .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:10px;background:#555;vertical-align:middle;}"
"h1 .dot.on{background:#23c;}"
".card{background:#161e30;border-radius:8px;padding:18px;margin-bottom:14px;}"
".card h2{margin:0 0 14px 0;font-size:14px;font-weight:600;color:#9fbce0;letter-spacing:1px;text-transform:uppercase;}"
".row{display:flex;align-items:center;justify-content:space-between;padding:8px 0;user-select:none;}"
".label{font-size:14px;}"
".switch{width:38px;height:22px;background:#2a3348;border-radius:11px;position:relative;cursor:pointer;transition:background 0.15s;flex-shrink:0;}"
".switch.on{background:#38c852;}"
".switch::after{content:'';position:absolute;top:2px;left:2px;width:18px;height:18px;background:#dbe4f0;border-radius:50%;transition:transform 0.15s;}"
".switch.on::after{transform:translateX(16px);}"
"button{background:#2a3348;color:#dbe4f0;border:none;padding:8px 16px;border-radius:6px;cursor:pointer;font-size:13px;transition:background 0.1s;}"
"button:hover{background:#3a4358;}"
"button.danger{background:#5a2a2a;}"
"button.danger:hover{background:#6a3a3a;}"
"button:disabled{opacity:0.4;cursor:not-allowed;}"
"input[type=range]{width:100%;margin:12px 0 4px 0;}"
".value{font-variant-numeric:tabular-nums;color:#9fbce0;font-size:18px;font-weight:500;}"
".sub{color:#6b7891;font-size:11px;margin-top:2px;}"
".meta{color:#6b7891;font-size:11px;margin-top:20px;font-variant-numeric:tabular-nums;}"
".dimmed{opacity:0.45;pointer-events:none;}"
"</style></head><body>"

"<h1><span class=dot id=dot></span>osu! hv menu</h1>"

"<div class=card>"
"<h2>in-game overlay</h2>"
"<div class=row><span class=label>Show indicator</span><div class=switch id=sw_overlay></div></div>"
"</div>"

"<div class=card id=card_ar>"
"<h2>approach rate</h2>"
"<div class=row>"
"<span class=label>Hook installed</span>"
"<button id=btn_ar_install>Install</button>"
"</div>"
"<div class=row id=row_ar_enable>"
"<span class=label>Override active</span>"
"<div class=switch id=sw_ar_enable></div>"
"</div>"
"<div id=row_ar_slider>"
"<div class=row style='padding-bottom:0;'><span class=label>AR value</span><span class=value id=v_ar>10.00</span></div>"
"<input type=range min=0 max=11 step=0.1 value=10 id=sl_ar>"
"<div class=sub>preempt: <span id=v_preempt>450</span> ms</div>"
"</div>"
"</div>"

"<div class=meta id=meta>connecting...</div>"

"<script>"
"const ws = new WebSocket('ws://' + location.host);"
"const $=(id)=>document.getElementById(id);"
"const dot=$('dot'),meta=$('meta');"
"const sw_overlay=$('sw_overlay'),sw_ar_enable=$('sw_ar_enable');"
"const btn_ar_install=$('btn_ar_install');"
"const row_ar_enable=$('row_ar_enable'),row_ar_slider=$('row_ar_slider');"
"const v_ar=$('v_ar'),v_preempt=$('v_preempt'),sl_ar=$('sl_ar');"
"let ar_installed=false,ar_enabled=false,overlay=true;"

"function set_sw(e,on){ if(on) e.classList.add('on'); else e.classList.remove('on'); }"
"function send(o){ if(ws.readyState===1) ws.send(JSON.stringify(o)); }"

"ws.onopen = ()=>{ dot.classList.add('on'); meta.textContent='connected'; };"
"ws.onclose= ()=>{ dot.classList.remove('on'); meta.textContent='disconnected'; };"
"ws.onmessage = (ev)=>{"
"  try{ const m = JSON.parse(ev.data);"
"    if(m.type!=='state') return;"
"    overlay = !!m.overlay;"
"    ar_installed = !!m.ar_installed;"
"    ar_enabled   = !!m.ar_enabled;"
"    set_sw(sw_overlay, overlay);"
"    set_sw(sw_ar_enable, ar_enabled);"
"    btn_ar_install.textContent = ar_installed ? 'Uninstall' : 'Install';"
"    btn_ar_install.className   = ar_installed ? 'danger' : '';"
"    row_ar_enable.className   = ar_installed ? 'row' : 'row dimmed';"
"    row_ar_slider.className   = ar_installed ? '' : 'dimmed';"
"    if(document.activeElement !== sl_ar) sl_ar.value = m.ar_value.toFixed(1);"
"    v_ar.textContent = (+m.ar_value).toFixed(2);"
"    v_preempt.textContent = m.ar_preempt;"
"    meta.textContent = 'connected · overlay '+m.overlay_hits+' hits · ar '+m.ar_hits+' hits';"
"  }catch(e){}"
"};"

"sw_overlay.addEventListener('click',()=>{ send({cmd:'set',key:'overlay',val:overlay?0:1}); });"
"sw_ar_enable.addEventListener('click',()=>{"
"  if(!ar_installed) return;"
"  send({cmd:'set',key:'ar_enable',val:ar_enabled?0:1});"
"});"
"btn_ar_install.addEventListener('click',()=>{"
"  send({cmd: ar_installed ? 'ar_uninstall' : 'ar_install'});"
"});"
"sl_ar.addEventListener('input',()=>{"
"  const v = parseFloat(sl_ar.value);"
"  v_ar.textContent = v.toFixed(2);"
"  send({cmd:'set',key:'ar_value',val:v});"
"});"
"</script></body></html>";

/* parse one line of HTTP header; returns 1 if found; writes value to out. */
static int get_header(const char* req, const char* name, char* out, int outsz) {
    char key[64];
    snprintf(key, sizeof(key), "\r\n%s:", name);
    const char* p = strstr(req, key);
    if (!p) return 0;
    p += strlen(key);
    while (*p == ' ') p++;
    int i = 0;
    while (*p && *p != '\r' && i < outsz-1) out[i++] = *p++;
    out[i] = 0;
    return 1;
}

/* send a plain HTTP response. */
static void http_respond(SOCKET s, const char* status, const char* ctype, const char* body, size_t blen) {
    char hdr[512];
    int hl = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
        status, ctype, blen);
    send(s, hdr, hl, 0);
    if (body && blen) send(s, body, (int)blen, 0);
}

/* WS frame send (server→client, no mask). */
static CRITICAL_SECTION g_send_cs;
static SOCKET g_clients[16] = {0};  /* max 16 browsers */

static void ws_send(SOCKET s, const char* data, int len) {
    uint8_t hdr[10]; int hlen;
    hdr[0] = 0x81;  /* FIN + text */
    if (len <= 125)        { hdr[1] = (uint8_t)len; hlen = 2; }
    else if (len <= 65535) { hdr[1] = 126; hdr[2]=(uint8_t)(len>>8); hdr[3]=(uint8_t)len; hlen=4; }
    else                   { hdr[1] = 127; for(int i=0;i<8;i++) hdr[2+i]=(uint8_t)(len>>(8*(7-i))); hlen=10; }
    EnterCriticalSection(&g_send_cs);
    send(s, (char*)hdr, hlen, 0);
    send(s, data, len, 0);
    LeaveCriticalSection(&g_send_cs);
}

static void ws_broadcast(const char* data) {
    int len = (int)strlen(data);
    for (int i = 0; i < 16; i++) {
        SOCKET s = g_clients[i];
        if (s) ws_send(s, data, len);
    }
}

static void push_state_to_browsers(void) {
    ph_status st; memset(&st, 0, sizeof(st));
    ao_ph_status(&st);
    uint64_t ovl_hits = (g_overlay_slot < PH_MAX_HOOKS) ? st.entries[g_overlay_slot].hits : 0;
    uint64_t ar_hits  = (g_ar_slot      < PH_MAX_HOOKS) ? st.entries[g_ar_slot].hits      : 0;
    char buf[512];
    snprintf(buf, sizeof(buf),
        "{\"type\":\"state\","
         "\"overlay\":%d,\"overlay_hits\":%llu,"
         "\"ar_installed\":%s,\"ar_enabled\":%d,"
         "\"ar_value\":%.2f,\"ar_preempt\":%u,"
         "\"ar_hits\":%llu}",
        g_menu_open, (unsigned long long)ovl_hits,
        (g_ar_slot < PH_MAX_HOOKS) ? "true" : "false",
        g_ar_enabled,
        (double)g_ar_value, ar_to_preempt(g_ar_value),
        (unsigned long long)ar_hits);
    ws_broadcast(buf);
}

/* parse incoming WS frame from client (masked). Fills out[], returns length or -1. */
static int ws_recv(SOCKET s, char* out, int outsz) {
    uint8_t h[2];
    if (recv(s, (char*)h, 2, MSG_WAITALL) != 2) return -1;
    /* opcode low 4 bits of h[0]. 0x8 = close. */
    if ((h[0] & 0x0F) == 0x8) return -1;
    int masked = h[1] & 0x80;
    uint64_t plen = h[1] & 0x7F;
    if (plen == 126) {
        uint8_t ext[2]; if (recv(s, (char*)ext, 2, MSG_WAITALL) != 2) return -1;
        plen = ((uint64_t)ext[0]<<8) | ext[1];
    } else if (plen == 127) {
        uint8_t ext[8]; if (recv(s, (char*)ext, 8, MSG_WAITALL) != 8) return -1;
        plen = 0; for (int i=0;i<8;i++) plen = (plen<<8) | ext[i];
    }
    if (plen > (uint64_t)(outsz-1)) return -1;
    uint8_t mask[4] = {0};
    if (masked) {
        if (recv(s, (char*)mask, 4, MSG_WAITALL) != 4) return -1;
    }
    if (recv(s, out, (int)plen, MSG_WAITALL) != (int)plen) return -1;
    if (masked) for (uint64_t i = 0; i < plen; i++) out[i] ^= mask[i & 3];
    out[plen] = 0;
    return (int)plen;
}


/* dispatch commands from browser.
 *   {cmd:"set",  key:"overlay",    val:0/1}
 *   {cmd:"set",  key:"ar_enable",  val:0/1}
 *   {cmd:"set",  key:"ar_value",   val:9.2}   # AR as float
 *   {cmd:"ar_install"}
 *   {cmd:"ar_uninstall"}
 */
static void dispatch(const char* json) {
    const char* cp = strstr(json, "\"cmd\":\"");
    if (!cp) return;
    cp += 7;
    char cmd[32] = {0}; int ci = 0;
    while (*cp && *cp != '"' && ci < 31) cmd[ci++] = *cp++;

    if (!strcmp(cmd, "ar_install")) {
        int rc = install_ar_hook();
        if (rc != 0) printf("[ar] install failed: %d\n", rc);
    } else if (!strcmp(cmd, "ar_uninstall")) {
        uninstall_ar_hook();
    } else if (!strcmp(cmd, "set")) {
        const char* kp = strstr(json, "\"key\":\"");
        const char* vp = strstr(json, "\"val\":");
        if (!kp || !vp) return;
        kp += 7;
        char key[32] = {0}; int ki = 0;
        while (*kp && *kp != '"' && ki < 31) key[ki++] = *kp++;
        double v = atof(vp + 6);

        if (!strcmp(key, "overlay")) {
            g_menu_open = v ? 1 : 0;
            printf("[set] overlay = %d\n", g_menu_open);
        } else if (!strcmp(key, "ar_enable")) {
            g_ar_enabled = v ? 1 : 0;
            printf("[set] ar_enable = %d\n", g_ar_enabled);
        } else if (!strcmp(key, "ar_value")) {
            g_ar_value = (float)v;
            printf("[set] ar_value = %.2f (preempt=%u)\n",
                   (double)g_ar_value, ar_to_preempt(g_ar_value));
        }
    }
    push_ar_cfg();
    push_overlay_cfg();
    push_state_to_browsers();
}

/* client thread: handle one TCP connection: either HTTP GET /, or WS upgrade. */
static DWORD WINAPI client_thread(LPVOID arg) {
    SOCKET s = (SOCKET)(uintptr_t)arg;
    char req[4096]; int rn = recv(s, req, sizeof(req)-1, 0);
    if (rn <= 0) { closesocket(s); return 0; }
    req[rn] = 0;

    /* WS upgrade? */
    char wskey[128] = {0};
    if (strstr(req, "Upgrade:") && get_header(req, "Sec-WebSocket-Key", wskey, sizeof(wskey))) {
        /* compute accept = base64(sha1(key + magic)) */
        char tomash[256];
        snprintf(tomash, sizeof(tomash), "%s258EAFA5-E914-47DA-95CA-C5AB0DC85B11", wskey);
        uint8_t digest[20];
        sha1((uint8_t*)tomash, (int)strlen(tomash), digest);
        char accept[64];
        b64(digest, 20, accept);
        char hs[512];
        int hl = snprintf(hs, sizeof(hs),
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Accept: %s\r\n\r\n", accept);
        send(s, hs, hl, 0);

        /* register this client in the broadcast table */
        int slot = -1;
        for (int i = 0; i < 16; i++) if (!g_clients[i]) { g_clients[i] = s; slot = i; break; }
        if (slot < 0) { closesocket(s); return 0; }
        printf("[ws] client %d connected\n", slot);

        /* push initial state */
        push_state_to_browsers();

        /* recv loop */
        char body[4096];
        while (1) {
            int n = ws_recv(s, body, sizeof(body));
            if (n < 0) break;
            dispatch(body);
        }
        g_clients[slot] = 0;
        closesocket(s);
        printf("[ws] client %d gone\n", slot);
        return 0;
    }

    /* Otherwise serve index.html (any GET returns the single page). */
    if (!strncmp(req, "GET ", 4)) {
        http_respond(s, "200 OK", "text/html; charset=utf-8",
                     INDEX_HTML, strlen(INDEX_HTML));
    } else {
        http_respond(s, "400 Bad Request", "text/plain", "", 0);
    }
    closesocket(s);
    return 0;
}

/* server thread: accept loop. */
static DWORD WINAPI server_thread(LPVOID arg) {
    (void)arg;
    WSADATA wsa; WSAStartup(MAKEWORD(2,2), &wsa);
    SOCKET ls = socket(AF_INET, SOCK_STREAM, 0);
    BOOL yes = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (char*)&yes, sizeof(yes));
    struct sockaddr_in sa = { .sin_family = AF_INET };
    sa.sin_addr.S_un.S_addr = htonl(INADDR_LOOPBACK);  /* localhost only */
    sa.sin_port = htons(LISTEN_PORT);
    if (bind(ls, (struct sockaddr*)&sa, sizeof(sa)) || listen(ls, 8)) {
        fprintf(stderr, "bind/listen failed: %lu\n", WSAGetLastError());
        return 1;
    }
    printf("http+ws server on http://localhost:%d\n", LISTEN_PORT);
    while (1) {
        SOCKET c = accept(ls, NULL, NULL);
        if (c == INVALID_SOCKET) continue;
        CreateThread(NULL, 0, client_thread, (LPVOID)(uintptr_t)c, 0, NULL);
    }
    return 0;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    if (!ao_present()) die("hypervisor not present");

    g_osu_cr3 = ao_get_process_cr3(TARGET_PROCESS);
    if (g_osu_cr3 == AO_NOT_PRESENT || g_osu_cr3 == 0) die("osu CR3 not found");
    g_osu_pid = find_pid(TARGET_PROCESS);
    if (!g_osu_pid) die("osu PID not found");
    g_osu_base = find_wow64_module_base(g_osu_pid, TARGET_PROCESS);
    if (!g_osu_base) die("osu base not found");

    /* Install overlay (status indicator) hook now; AR hook is browser-triggered. */
    uint64_t gdi_base = find_wow64_module_base(g_osu_pid, TARGET_MODULE);
    if (!gdi_base) die("gdi32.dll not in osu's 32-bit modules");
    uint32_t rva = find_export_rva("C:\\Windows\\SysWOW64\\gdi32.dll", TARGET_EXPORT);
    if (!rva) die("SwapBuffers RVA not found");
    uint64_t swap_va = gdi_base + rva;

    struct gl_exports e;
    if (!resolve_gl(g_osu_pid, &e)) die("GL export resolution failed");

    ph_insn script[PH_SCRIPT_MAX_INSNS];
    uint32_t n = build_script(script, PH_SCRIPT_MAX_INSNS, &e);
    uint64_t slot = ao_ph_register(g_osu_cr3, swap_va, script, n);
    if ((int64_t)slot < 0) die("overlay ph_register failed");
    g_overlay_slot = (uint32_t)slot;
    printf("overlay hook installed: slot=%u swap_va=0x%08llX script=%u insns\n",
           g_overlay_slot, (unsigned long long)swap_va, n);

    push_overlay_cfg();

    InitializeCriticalSection(&g_send_cs);
    CreateThread(NULL, 0, server_thread, NULL, 0, NULL);

    printf("ready. open http://localhost:%d in your browser.\n", LISTEN_PORT);
    DWORD last_push = GetTickCount();
    while (1) {
        Sleep(500);
        if (GetTickCount() - last_push >= 500) {
            /* refresh overlay color in case AR state changed via browser. */
            push_overlay_cfg();
            push_state_to_browsers();
            last_push = GetTickCount();
        }
    }
    return 0;
}
