# key-inject

Keyboard-injection hook for osu!stable, staged build.

The goal: force K1/K2 "pressed" or "released" in osu!'s own keyboard
state buffer, from an ao PH script, so osu! sees legitimate-looking key
state without any Win32 input API (no `SendInput`, no `keybd_event`).

## Staged plan

- **Pass 1** (done, in `tools/cursor-capture/` helpers / standalone
  resolver): resolve the `keybinding_aob` pointer chain. Confirmed K1
  and K2 VK codes.
- **Pass 2** (this file, log-only): install a `PH_LOG(ECX, 0)` script
  on the `keyboard_get_state` call-site. Watch that it fires without
  crashing and that ECX is the state-buffer pointer we expect.
- **Pass 3** (next): replace the script with the full PH_OP_CALL +
  WRITE1 variant driven by CFG. User-mode flips CFG0/CFG1 to force
  press/release of K1/K2.
- **Pass 4**: wire into `new-era/relax/live/` dispatcher so model
  decisions become key presses.

## Commands

```
key_inject resolve                 run sig+deref, print K1/K2 VK + call-site VA
key_inject install                 install LOG-ONLY hook (pass 2)
key_inject watch [interval_ms]     drain ph_log, print state-buffer ptrs
key_inject remove                  ao_ph_clear (removes ALL hooks)
```

## Build

```
build.bat
```
