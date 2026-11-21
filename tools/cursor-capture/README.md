# cursor-capture

Installs an ao PH hook at osu!'s `mouse_apply_handler` call-site. The
hook reads the cursor float32 `x,y` out of `[ebp-0x14]` / `[ebp-0x10]`
into CFG slots 2 and 3, then runs the original instruction unchanged.

User-mode polls the slot's `cfg[2]` / `cfg[3]` via `ao_ph_status()`,
reinterprets as float, and gets the live cursor.

## Build

```
build.bat
```

## Run

```
cursor_capture install            sigscan + install
cursor_capture watch [ms=100]     poll & print live xy
cursor_capture remove             ao_ph_clear (removes ALL hooks)
```

## How the hook works

- Sig: AQN `mouse_apply_handler` = `d9 c1 d9 1c 24 d9 5d ? d9 5d ? ff 15`
- Match lands on `d9 c1` (= `fld st(1)`), ~0x2D bytes into the function.
- The function's getter call at `-0x17` from the match stores cursor xy
  into `[ebp-0x14]..[ebp-0x0C]`. By the time our hook fires those bytes
  are already there.
- Script: `PH_READ4(CFG2, EBP, -0x14) / PH_READ4(CFG3, EBP, -0x10) / PH_RETCONT`
- PH's RETCONT path runs the original `d9 c1` from the shadow page, then
  resumes guest execution at match+2. No visible bytes changed in osu's
  address space.
