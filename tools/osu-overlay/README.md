# osu-overlay

Draws a quad into osu!'s GL context every frame. Reference consumer of
ao's `PH_OP_CALL` chaining — hooks `gdi32!SwapBuffers` and runs ~17 native
`opengl32` calls per frame to push a draw into the live render loop.

## Build

```bat
rem from this folder, after ..\..\..\ao\host\ao_call.obj is built
cl /nologo /O2 /W3 osu_overlay.c ..\..\..\ao\host\ao_call.obj ^
   /I..\..\..\ao\host /link psapi.lib user32.lib /out:osu_overlay.exe
```

## Run

```
osu_overlay.exe install            Day-1 log-only hook
osu_overlay.exe install_draw       red quad
osu_overlay.exe install_overlay    translucent panel
```

## Rules / caveats

- `gdi32.dll` is a shared system DLL; without a CR3 filter the hook
  fires for every process using it. The tool sets the CR3 filter to
  osu!.exe's CR3. See `../../../ao/docs/case-studies/osu-stable.md` for
  the hazard note.
- osu!'s GL projection is non-default ortho; drawing in NDC `[-1,+1]`
  does NOT land full-screen. Fine for a HUD; needs projection work for
  spatial overlays tied to game space.
