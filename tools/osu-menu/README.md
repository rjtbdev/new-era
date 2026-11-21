# osu-menu

Single-file browser-UI control panel for ao hooks against osu!stable.
Runs a localhost HTTP + WS server on `:8080` and drives hook install /
uninstall, CFG slot updates (AR override), and live counter streaming.

## Build

```bat
cl /nologo /O2 /W3 osu_menu.c ..\..\..\ao\host\ao_call.obj ^
   /I..\..\..\ao\host /link psapi.lib user32.lib ws2_32.lib bcrypt.lib ^
   /out:osu_menu.exe
```

## Run

```
osu_menu.exe
```

Then open `http://localhost:8080`.

## UI

- Status indicator toggle (in-game overlay).
- AR hook install / uninstall (sigscans on click).
- AR enable / disable.
- AR value slider (0.0 – 11.0) with live preempt-ms readout.

Browser → tool JSON on WS:
```
{"cmd":"set","key":"ar_value","val":9.2}
{"cmd":"ar_install"}
{"cmd":"ar_uninstall"}
{"cmd":"set","key":"overlay","val":0}
```

Tool → browser state push every 500ms:
```
{"type":"state","overlay":1,"overlay_hits":1234,
 "ar_installed":true,"ar_enabled":1,
 "ar_value":9.2,"ar_preempt":570,"ar_hits":5678}
```

## AR formula

Baked into the tool:
```
AR < 5:   preempt_ms = 1800 - 120 * AR
AR >= 5:  preempt_ms = 1200 - 150 * (AR - 5)
clamp to [100, +inf)
```

Standard osu!: AR 0 → 1800, AR 5 → 1200, AR 10 → 450. Extends past AR 10
linearly (AR 11 → 300) for overclock experiments.

## Rules

- Sigs this tool uses live in `../../sigs/osu-stable.yaml` (TBD).
  Case-study prose for each sig lives in `../../../ao/docs/case-studies/osu-stable.md`.
