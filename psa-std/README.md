# psa_std — the psa standard library (L1)

Layering: **psa → psa_std → sla_std → sa_std → SA**. psa_std is
written in SLA against the dynamic-value surface (`PsaVal` and
the `__psa_*` runtime helpers a psa program emits whenever it
uses dynamic values). A psa_std module is therefore not
standalone-compilable: it is linked into a psa program whose
emitted code provides the runtime. User programs will reach it
through `import json` etc. The `json` wiring landed with psa
1.0.0 (`import json` binds this directory's json.sla; see
examples/json1.psa). The `math` / `string` wiring was written
2026-11-26 (same intercept shape; see L1-WRITE-NOTES.md) and
was compiled and verified in the unified batch of
2026-11-26 (run_math_harness.sh / run_string_harness.sh
plus import-level end-to-end checks, byte-compared
against CPython). The csv / bisect / heapq / base64
wirings were verified in the same batch (harness +
negative phase + import e2e each).

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
- `math.sla` — the honest subset of CPython `math` over
  PsaVal (written 2026-11-26, unverified; harness:
  run_math_harness.sh). Rounding: `psa_math_floor` /
  `psa_math_ceil` / `psa_math_trunc` (int in, int out;
  float in, int out). Float: `psa_math_fabs`,
  `psa_math_sqrt`, `psa_math_sin`, `psa_math_cos`,
  `psa_math_degrees`, `psa_math_radians`,
  `psa_math_pow` (always float, as CPython `math.pow`;
  int base + int non-negative exponent + i64-fitting
  result computed exactly first, else libm), and
  `psa_math_isclose` (2-argument form, CPython default
  tolerances). Exact integer: `psa_math_gcd`,
  `psa_math_lcm`, `psa_math_factorial`, `psa_math_isqrt`,
  `psa_math_comb`, `psa_math_perm` (all overflow-checked).
  Float functions ride the sa_std libm externs
  (`sa_math_pow` / `sa_math_sqrt` / `sa_math_sin` /
  `sa_math_cos`) -- the same libm CPython calls, so
  results agree bit for bit on the same machine.
  Divergences, all loud: integers are i64, so factorial /
  comb / perm / lcm panic 77134 where CPython's bigint
  would return a bigger value; floor / ceil / trunc of a
  float outside the i64 range panic 77134 (CPython
  returns a bigint); gcd / lcm with an i64::MIN operand
  panic 77134; the `(-2)**63` corner of the integer pow
  path falls through to libm (which represents it
  exactly). Not covered: exp / log family, tan and the
  remaining trig / hyperbolic functions, hypot, fsum
  (CPython's is exactly rounded; a Kahan sum would be a
  silent approximation), dist, prod, remainder, the
  error / gamma functions, and the constants `pi` / `e`
  / `tau` / `inf` / `nan` (attribute reads of std
  modules are not wired in the emitter yet; degrees /
  radians use the same double pi internally). Reasons:
  sa_std exports no libm entry for them, or the exact
  algorithm needs a surface the runtime does not have.
- `string.sla` — the honest subset of CPython `string`
  over PsaVal (written 2026-11-26, unverified; harness:
  run_string_harness.sh). Constants are provided as
  zero-argument entry points returning dyn strs --
  `psa_string_ascii_lowercase`, `psa_string_ascii_uppercase`,
  `psa_string_ascii_letters`, `psa_string_digits`,
  `psa_string_hexdigits`, `psa_string_octdigits`,
  `psa_string_punctuation`, `psa_string_whitespace`,
  `psa_string_printable` -- because std-module attribute
  reads (`string.digits` without a call) are not wired
  in the emitter yet (same gap as the math constants).
  Function: `psa_string_capwords` /
  `psa_string_capwords_sep` for `string.capwords(s)` /
  `string.capwords(s, sep)`. Case work is ASCII-only
  (the runtime str-method stance): bytes >= 0x80 pass
  through unchanged, where CPython would case-map and
  split on Unicode whitespace; separator matching is at
  byte level, which for a valid UTF-8 separator can
  only match at code-point boundaries. Not covered:
  `string.Template` and `string.Formatter` (classes;
  psa_std modules are function sets in v1).

## Panic codes (psa_std range)

- 77130 — json: value not serializable / float not dumpable /
  malformed JSON structure
- 77131 — json: trailing characters after the top-level value
- 77132 — math: argument of the wrong type (CPython TypeError:
  non-numeric where a number is required, non-int where an
  integer function requires int)
- 77133 — math: domain error (CPython ValueError: sqrt of a
  negative, factorial / isqrt of a negative, pow domain,
  floor / ceil / trunc of NaN, sin / cos of infinity)
- 77134 — math: range error (CPython OverflowError, including
  the i64 boundary CPython's bigint would cross silently:
  factorial / comb / perm / lcm overflow, floor / ceil /
  trunc out of i64 range, pow overflow, i64::MIN operand)
- 77135 — string: argument of the wrong type (CPython
  TypeError: non-string where a string is required)
- 77136 — string: bad value (CPython ValueError: empty
  separator in capwords)
- 77140 — collections: argument of the wrong type
- 77141 — collections: bad value / missing key / empty
  container
- 77142 — itertools: argument of the wrong type
- 77143 — itertools: bad value
- 77144 — functools: argument of the wrong type / bad
  value
- 77146 — collections / itertools / functools (shared):
  argument is not callable
- 77148 — collections / itertools / functools (shared):
  registered-gap refusal (a feature deliberately not
  provided)
- 77149 — collections / itertools / functools (shared):
  argument is not iterable
- 77150 — random: argument of the wrong type
- 77151 — random: bad value
- 77152 — base64: argument of the wrong type
- 77153 — base64: invalid input (CPython binascii.Error)
- 77154 — heapq: argument of the wrong type
- 77155 — heapq: pop or replace on an empty heap
- 77156 — bisect: argument of the wrong type
- 77157 — bisect: bad value
- 77158 — random: the OS CSPRNG draw behind seed(None)
  failed
- 77160 — argparse: argument of the wrong type
- 77161 — argparse: parse error (the CPython-style
  error text is written to stderr first)
- 77162 — argparse: help requested (the help text is
  written to stdout first)
- 77163 — datetime: value error
- 77164 — os: general failure (including the scratch
  limit; never a silent truncation)
- 77165 — logging: argument of the wrong type
- 77166 — logging: bad format string or level name
- 77167 — csv: argument of the wrong type (including
  dialect validation)
- 77168 — csv: parse / write error (CPython _csv.Error)
- 77169 — argparse: definition error
- 77170 — re: argument of the wrong type
- 77171 — re: pattern syntax error (including the
  {m,n} and nesting limits)
- 77172 — re: unsupported construct (loud refusal:
  lookahead / lookbehind, named groups, backreferences
  in the pattern, conditional / atomic groups,
  possessive quantifiers)
- 77173 — re: backtracking step limit exceeded
- 77174 — re: bad escape in the replacement string
- 77175 — re: bad flags value
- 77176 — re: bad group reference in the replacement
  string
