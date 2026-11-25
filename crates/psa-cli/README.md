# psa — the SA-Py compiler (v0.1)

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

## Known upstream defect (found 2026-11-25, blocks one feature)

The Rust `sla` **build-exe** route prints f64 values through
`println("{}", x)` with the fraction truncated: `7.0 / 2.0` prints
`3.0000000000`. The value itself is correct — comparisons in the same
executable confirm `z == 3.5` — and the Zig route prints `3.5`
correctly; only f64 display on this route is wrong. Until that is
fixed upstream, **psa refuses to print a float** (loud error) rather
than emit a program that displays a wrong value. Float computation
and comparison are unaffected and supported.

## Explicitly not in v0.1 (all loud errors)

Classes, attribute access, subscripts, lists/dicts/sets, string
operations and comparisons, `**`, chained comparisons, keyword
arguments, defaults, annotations, imports, exceptions, generators,
first-class functions, type-changing re-binding, `continue` in
`for`, non-`range` iteration, float `//` / `%`.

Per the SA-Py whitepaper, dynamic typing is a later phase built on a
tagged-value runtime; v0.1 is the statically inferable subset only.
