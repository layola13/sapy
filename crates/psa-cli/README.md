# psa — the SA-Py compiler (v0.3)

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

## Explicitly not in v0.1 (all loud errors)

Classes, attribute access, dicts/sets, lists of non-scalar elements
(strings, nested lists), string operations and comparisons, `**`,
chained comparisons,
keyword arguments, defaults, annotations, imports, exceptions,
generators, first-class functions, type-changing re-binding,
`continue` in `for`, non-`range` iteration, float `//` / `%`,
printing a whole list.

Per the SA-Py whitepaper, dynamic typing is a later phase built on a
tagged-value runtime; v0.1 is the statically inferable subset only.
