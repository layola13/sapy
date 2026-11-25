# psa — the SA-Py compiler (v0.5)

`psa` compiles `.psa` source (a Python-syntax subset) to native
executables: `.psa` → psa frontend (lexer / parser / type inference) →
generated SLA source → the `sla` pipeline (checker + SA emission) →
the `sa` toolchain (`build-exe`). The binary is named `psa`, mirroring
how the SLA compiler binary is named `sla`.

```sh
psa emit      <file.psa>          # print the generated SLA source
psa check     <file.psa>          # type-check through the sla pipeline
psa build-exe <file.psa> [-o out] # build a native executable
```

The generated `.sla` file is written next to the source for inspection.
`psa` locates `sa_std` by walking ancestors of the source file, of the
`sla` binary (`PSA_SLA`), and of the cwd, unless `SA_STD_DIR` is set.

## The v0.1 subset — only what is listed here is supported

Everything else is a **loud compile error**, never a silent fallback.

- Top-level statements (wrapped into `main`) and `def` functions with
  positional parameters; recursion works (recursive return type must
  be `int` in v0.1).
- Types are inferred: `int` (i64), `float` (f64), `bool`, string
  literals. One type per variable; re-binding to a different type is
  an error (int → float widening on assignment is allowed).
- Integer arithmetic `+ - *`, floor division `//` and `%` with true
  Python floor semantics (via `__psa_floordiv` / `__psa_floormod`).
- `/` is Python true division and yields a float; float arithmetic
  and comparisons compute correctly.
- Comparisons `== != < <= > >=` (no chaining), `and` / `or` / `not`.
- `if` / `elif` / `else`, `while` with `break` / `continue`,
  `for i in range(...)` with one or two int bounds (no step, no
  `continue` inside `for` — use `while`), `return`, `pass`,
  augmented assignment `+= -= *=`.
- `print(...)`: ints, strings, bools (`True` / `False`), and
  multi-argument space-joined output (no bools in multi-arg print).
- `list[int]` (v0.2): list literals of ints, `xs.append(v)`, index
  read `xs[i]` (scalar elements copy out), index write `xs[i] = v`
  and `xs[i] += v`, `len(xs)`. Ownership follows SPEC-OWNERSHIP.md:
  assignment of a list **moves** it (use-after-move is a loud error
  naming both sites); a function parameter the callee only reads or
  writes through is inferred as a **borrow** (emitted as SLA
  `&Vec<i64>`, call sites pass `&name`); a parameter the callee moves
  out of is **consumed** (emitted as SLA `Vec<i64>`). Borrowed
  parameters can be re-borrowed onward but never moved, returned, or
  stored — those are loud errors (spec s7).
- `list[bool]` (v0.3): the v0.2 list machinery is element-type
  generic for scalar elements — literals, `append`, index read/write,
  `len`, and the same move/borrow inference (emitted as `Vec<bool>`,
  borrows as `&Vec<bool>`). Element types must agree within one
  literal; mixed literals are a loud error. Elements are still
  scalars only: string elements are heap values and stay a loud
  error, as does the empty list literal (its element type cannot be
  inferred). `list[float]` is implemented in the emitter but
  **blocked** by upstream defect (xxiv) below: it is a loud error
  until the `Vec<f64>` miscompilation is fixed.

- D0 (v0.4, the v1 charter's frontend batch): `**` (int ** int with
  a literal exponent is exact integer power; a negative literal
  exponent or any float operand goes through sa_std `sa_math_pow`;
  a non-literal int exponent is a loud error — its Python result
  type depends on the exponent's sign), bitwise `& | ^ ~ << >>`,
  chained comparisons (`a < b < c`; middle operands must be pure —
  a call in the middle is a loud error), conditional expressions
  (`x if c else y`, nestable; not in `while` conditions, where the
  materialised temp would only be computed once), and f-strings as
  the sole `print()` argument (interpolated ints and string
  literals; at most one bool interpolation; format specs and
  conversions are loud errors; an f-string anywhere else is a loud
  error until the D2 string runtime). All augmented-assignment
  forms (`//=`, `**=`, `&=`, ...) work through the same operators.
  `examples/d0.psa` runs end-to-end and its output matches CPython
  byte for byte.

- D1 (v0.5, the scalar dynamic layer): a variable or function whose
  type cannot be pinned to one scalar — mixed return branches, a
  conditional expression with mixed scalar branches, a `None`
  return next to a value return — becomes **dynamic**: its value is
  a tagged `PsaVal` (int / float / bool / None) and operations
  dispatch through a small runtime with Python semantics (true
  division always float, floor division/modulo follow Python sign
  rules, cross-kind numeric comparison, operand-returning
  short-circuit `and`/`or`, truthiness). Static and dynamic values
  coexist per-variable: static values box at dynamic boundaries,
  dynamic values unbox at static boundaries with a runtime kind
  check (a failed unbox is a loud panic, psa's TypeError surface —
  the static boundary is deliberately stricter than CPython).
  A dynamic value prints on its own line (int/bool/None; a float
  dynamic print is blocked by defect (xxiii) below and panics
  loudly rather than display a wrong value). A variable's type is
  still pinned by its first binding: a statically inferred variable
  cannot become dynamic later (loud error). Heap kinds (str/list/
  dict values as dynamic citizens) are D2. `examples/dyn1.psa`
  matches CPython byte for byte; the dynamic-style bench
  `bench/dynsum.psa` (10^6 dynamic additions) runs 0.12s vs
  CPython's 0.27s (best of 3).

## Known upstream defects (found 2026-11-25, each blocks one feature)

**(xxiii) f64 display.** The Rust `sla` **build-exe** route prints
f64 values through `println("{}", x)` with the fraction truncated:
`7.0 / 2.0` prints `3.0000000000`. The value itself is correct —
comparisons in the same executable confirm `z == 3.5` — and the Zig
route prints `3.5` correctly; only f64 display on this route is
wrong. Until that is fixed upstream, **psa refuses to print a
float** (loud error) rather than emit a program that displays a
wrong value. Scalar float computation and comparison are unaffected
and supported.

**(xxiv) `Vec<f64>` element access.** Hand-written SLA probes
independent of psa (push / index-write / index-read / equality over
`Vec<f64>`, with a `Vec<i64>` control passing) fail on **both**
routes: literal and variable pushes read back wrong on the Rust
route and the Zig route alike; index assignment `ys[0] = 2.25`
reads back correctly on the Rust route only. The SA-level `Vec`
storage is byte-typed, so both frontends' `Vec<f64>` element
lowerings are implicated. Until this is fixed upstream, **psa
refuses `list[float]`** (loud error) rather than emit silently
wrong programs. `list[int]` and `list[bool]` are unaffected.

**(xxv) String bindings.** On the Rust `sla` **build-exe** route a
string value cannot be held in a variable at all: an inferred
`let s = "world"` fails SAB lowering with
`UnsupportedSabDirectFeature: string literal let binding`, an
explicitly annotated `let s: &str = "world"` fails the type checker
(`expected borrow, actual raw_ptr`), and a `format(...)` String
cannot be printed either. The same shapes work inside `@test`
bodies via the `sla test` route (see the upstream corpus minim), so
this is specific to the build-exe lowering path. Until it is fixed
upstream, **psa refuses to bind, pass, or return a string value**
(loud error); string literals used directly in `print()` work, as
before. This also blocks string branches of conditional
expressions and string-value interpolation in f-strings.

**(xxvi) Self-referential call assignment (worked around in psa).**
The direct shape `x = f(x, ...)` — including the helper calls that
`//`, `%`, and `**` lower to — leaves a live register at function
exit (`MemoryLeak`, trap 1012) in the Rust build-exe lowering.
psa routes such values through a fresh temp before assigning
(`let t = f(x, ...); x = t;`), which lowers cleanly; the upstream
direct shape is still defective.

**(xxvii) Aggregate assignment and block-scoped locals in the SAB
lowering (worked around in psa).** Hand-written SLA probes
independent of psa (q1–q29, `@derive(copy)` struct and tuple)
establish four shapes on the Rust build-exe route: (a) a plain
`let t = v;` *moves* a Copy struct (the checker honours the
derive, `gen_assign` does not — the same missing-metadata family
as task 161 item ②); (b) *any* reassignment of an aggregate
variable — straight-line or in branches — traps
`RegisterRedefinition` (1006): every aggregate assignment is
emitted as a fresh register definition; (c) a block-scoped local
bound to a call result — scalar or aggregate, consumed or not —
is never retired and traps `MemoryLeak` (1012) at function exit;
(d) a through-borrow field-write setter leaks its borrow
parameter at function exit. psa's dynamic layer therefore never
reassigns a `PsaVal`: stores go through a non-Copy stage struct
whose scalar fields are copied into the destination and which is
then moved into a void sink (`__psa_stage_of`/`__psa_drop_stage`,
probe q28), and dynamic bindings first materialised inside a
block are hoisted to the function prologue. The upstream shapes
remain defective; a fix there would let psa simplify back to
plain assignments.

## Explicitly not in v0.1 (all loud errors)

Classes, attribute access, dicts/sets, lists of non-scalar elements
(strings, nested lists), string operations and comparisons,
string variables (blocked by (xxv) above), f-strings outside
`print()`, f-string format specs and conversions,
keyword arguments, defaults, annotations, imports, exceptions,
generators, first-class functions, type-changing re-binding,
`continue` in `for`, non-`range` iteration, float `//` / `%`,
printing a whole list.

Per the SA-Py whitepaper the dynamic face is a tagged-value
runtime; D1 (v0.5) landed its scalar layer (see the D1 bullet
above and DYNAMIC-DESIGN.md). Heap kinds are D2.
