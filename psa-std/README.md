# psa_std — the psa standard library (L1)

Layering: **psa → psa_std → sla_std → sa_std → SA**. psa_std is
written in SLA against the dynamic-value surface (`PsaVal` and
the `__psa_*` runtime helpers a psa program emits whenever it
uses dynamic values). A psa_std module is therefore not
standalone-compilable: it is linked into a psa program whose
emitted code provides the runtime. User programs will reach it
through `import json` etc. The `json` wiring landed with psa
1.0.0 (`import json` binds this directory's json.sla; see
examples/json1.psa); other modules still ride the batch harness
until their own wiring lands.

Rules for modules here:

- Operate on `PsaVal` only; never invent a second value type.
- Bounds-guarded reads never ride on `&&` / `||` (SLA evaluates
  both sides eagerly — probe P4, compiler defect (xviii)).
  Guarded reads sit inside plain bounds branches.
- No `*b = xs` whole-value deref assignment (codegen error,
  probe P1); mutate through the proven helper surface.
- Errors are loud panics with psa_std codes (77130+), never
  silent defaults.

## Modules

- `json.sla` — `psa_json_dumps(v) -> str`, `psa_json_parse(s) -> v`.
  CPython `json` semantics for the v1 surface: separators
  `", "` / `": "` (the CPython defaults), tuples dump as
  arrays, dict keys must be strings, sets and other tags are
  refused (77130). Float dump: integral floats print as `N.0`;
  non-integral floats are refused (77130) until the runtime's
  float-repr builder lands — same stance as `repr(float)`.
  Parse: full JSON grammar incl. string escapes and `\uXXXX`
  (encoded to UTF-8); numbers go through the runtime's
  str→int / str→float parsers (their 77079 panic is the parse
  error surface for malformed numbers); trailing garbage is
  refused (77131). Divergence: dumps passes non-ASCII UTF-8
  bytes through raw (CPython's default `ensure_ascii=True`
  would `\uXXXX`-escape them); `\uXXXX` parse does not
  combine surrogate pairs. Verified 2026-11-25 by
  run_json_harness.sh (dumps + parse + round trip byte-equal
  to CPython json on nested values with escapes).

## Panic codes (psa_std range)

- 77130 — json: value not serializable / float not dumpable /
  malformed JSON structure
- 77131 — json: trailing characters after the top-level value
