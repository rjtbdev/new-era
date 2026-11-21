# new-era

Umbrella repo for osu!-adjacent projects. Downstream consumer of
[`ao`](../ao/), the generic hypervisor framework.

## Contents

```
new-era/
├─ relax/           ML replay-generation + live-play tool for osu!std relax
├─ tools/
│  ├─ osu-overlay/  GL-drawn overlays in osu!'s render loop (via ao PH_OP_CALL)
│  └─ osu-menu/     browser-UI control panel for ao hooks (AR override, etc.)
├─ sigs/            Centralized osu!stable sig recipes (source of truth)
├─ docs/
│  └─ architecture.md   how the new-era projects fit together
├─ harness/         harness run <name> entry point
└─ artifacts/       dated harness output (gitignored)
```

## Relationship to ao

- `ao` is a game/binary-agnostic hypervisor + hook framework. It knows
  nothing about osu.
- `new-era/` is a consumer of ao. All osu-specific knowledge — sig
  patterns, memory layouts, hook recipes, playstyle models — lives here.
- The handshake: tools in `new-era/tools/` include
  `../../ao/host/ao_client.h`, link against `ao_call.obj`, and talk to
  the ao driver via hypercalls.

If something in this repo would be useful to a non-osu consumer of ao
(e.g. a better sig-recipe schema, a cross-target hook-manager pattern),
it belongs upstream in `ao/sdk/`, not here.

## Sigs

`sigs/osu-stable.yaml` (TBD format) is the single source of truth for
every osu!stable AOB pattern and offset the tools in this repo use. See
`../ao/docs/case-studies/osu-stable.md` for the companion prose — what
each sig is, where it came from, what's unresolved.

## Projects

### relax/

Three-head ML model (timing / outcome / key) that generates human-like
osu!std relax replays. Offline for now; live-play integration planned
once the ao-side hook primitives for the needed sigs are implemented.

See `relax/README.md` for status.

### tools/osu-overlay/

Draws arbitrary quads into osu!'s GL context every frame via a
PH_OP_CALL chain hooked on `gdi32!SwapBuffers`. Reference implementation
for "chain native calls from an ao script."

### tools/osu-menu/

Localhost HTTP + WS server + browser UI that controls installed ao
hooks (install/uninstall, CFG slots for AR override, live counters).
Flagship consumer of ao as of hemlock-era.
