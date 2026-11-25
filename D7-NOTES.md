# psa D7 — generators, async/await, match, overflow guards

Design settled 2026-11-25 (main session), following the
D0–D6 layering: everything rides the dynamic face (PsaVal);
anything not carried is refused loudly at parse or emit.

## 1. Generators (tag 13)

A def whose own body contains a `yield` statement is a
generator def. All its parameters are pinned dynamic (the
method discipline). Calling it does NOT run the body: the
emitter splits the def into

- a callable wrapper `__py_g(args) -> PsaVal` that creates a
  channel, spawns the producer thread, packs
  {chan, handle, done} into a boxed `PsaGenState` and returns
  it as a tag-13 value;
- a producer `__py_g__prod(ch, args) -> PsaVal` that runs the
  body with the channel in scope. `yield v` emits
  `ch.send(wrap(v, false))`; falling off the end or a bare
  `return` sends `wrap(None, true)`. `return <value>` in a
  generator def is refused loudly (CPython's StopIteration
  value is not modeled).

The channel is `Chan<PsaVal>` with capacity 0 (rendezvous),
so the producer blocks at each yield until the consumer
pulls — true lazy semantics for `next()`. The wrap is a
2-tuple (item, end-flag) so no user value can forge the end
marker. Surface shapes are the proven ones: `Chan::new(0)`,
`ch.send(v)`, `let (v, closed) = ch.recv()`,
`thread::spawn(^|| f(args))`, `JoinHandle<PsaVal>`
(sa_plugin_sla corpus test_unit_join_handle_vec_direct +
gosa demo 27). Imports added when a generator exists:
sa_std/task.sai, sa_std/thread.sa, sa_std/core/result.sa.

Consumption:
- `next(g)` (new builtin) -> `__psa_gen_next(g)`: recv one
  wrap; on the end marker (or a closed channel) it finishes
  the generator (join + destroy + done flag) and panics
  77081 (StopIteration). Calling next() on an exhausted
  generator panics 77081 again; on a non-generator, 77052.
- Iteration (for-each and comprehensions) goes through
  `__psa_iter_source(v)`: a tag-13 value is drained into a
  list (via next() until the end marker) and the existing
  index protocol runs on the list; every other value passes
  through unchanged.

Registered divergences:
- A for-loop over a generator drains it fully, even when
  the loop breaks early; an infinite generator combined
  with an early break therefore does not terminate in psa
  (it does in CPython). next() stays lazy and exact.
- A consumer that abandons a generator leaks the producer
  thread (blocked on send/recv) and the state box; psa has
  no thread cancellation surface to offer in v1.
- An uncaught exception inside a generator body aborts in
  the producer thread; CPython re-raises it at the next()
  call site. Exceptions caught inside the body work (the D6
  machinery is per-function and the producer is a function).
- `yield` is a statement in v1 (`yield expr` / bare
  `yield`); yield inside expressions, `yield from`, and
  generator .send()/.throw()/.close() are refused loudly.
- Generator expressions: supported only when the element
  expression references nothing from the enclosing scope
  except defs and literals (the clause iterables are passed
  to a synthetic producer def as arguments, evaluated
  eagerly — CPython evaluates only the outermost iterable
  eagerly). Otherwise refused loudly (closure capture).

## 2. async / await (tag 14)

`async def` marks a def (Func.is_async). Its parameters are
pinned dynamic. Calling it returns a tag-14 coroutine value
whose payload is [worker callable id, packed args]; the
body does not run. The body itself is emitted as a worker
def registered in the callable table.

`await e` is an expression: e must evaluate to a tag-14
value; `__psa_coro_run` dispatches the worker through the
ordinary callable machinery and returns its result. Await
of anything else panics 77082. Awaiting runs the coroutine
to completion at the await point — sequential semantics:
no event loop, no concurrency, no scheduling. `async with`
/ `async for`, top-level await (psa has no module-level
await context; await is legal in any function in v1 and
simply runs the coroutine), and asyncio are out of scope —
refused loudly where the parser can see them.

A coroutine that is never awaited never runs (and psa does
not warn; CPython warns at GC time — divergence, noted).

## 3. match

`match`/`case` are soft keywords (they stay valid
identifiers elsewhere). The parser commits to a match
statement only when a statement-initial `match` is followed
by a subject expression and a colon; otherwise the tokens
parse as an ordinary statement.

Patterns in v1: literals (int/float/str/bool/None, signed
numbers), capture names (a bare name always captures — true
Python semantics), wildcard `_`, sequence patterns over
lists/tuples `[p, ...]` / `(p, ...)` (no star), OR patterns
`p | p` (all alternatives must bind the same name set), and
guards `if expr`. Class patterns, mapping patterns, star
patterns and dotted value patterns are refused loudly.

Emission is a statement-level expansion in the emitter: the
subject is evaluated once into a dynamic temp; each case
computes an ok-flag with a prelude of checks (literal
equality via the dynamic eq, sequence tag/length/element
recursion), writing captures into per-case temps; on
success the temps are copied to the real names, the guard
(evaluated against the temps) passes, and the body runs.
Cases are tried in order inside plain ifs, so break /
continue / return in a case body keep their enclosing
meaning. No match arm firing is a silent no-op, as in
Python.

## 4. Dynamic int overflow (panic 77080)

CPython ints are arbitrary precision; psa does not
implement bigint in v1. Instead of silently wrapping, the
dynamic int operations now check: __psa_dyn_add/sub/mul
guard the i64 boundary, __psa_dyn_neg guards MIN, and the
dynamic floordiv guards MIN // -1. Overflow panics 77080.
The static face keeps SA i64 semantics (documented; a
static bigint is the same missing feature). __psa_ipow is
not guarded in v1 (noted gap).

## 5. Cyclic GC — evaluation only

psa's dynamic memory is Perceus-style reference counting by
construction plan (D0 decision); the current runtime leaks
payload boxes deliberately (the from_raw/into_raw dance
keeps every payload alive). A cycle collector is therefore
an evaluation item, not a v1 feature: with leak-by-design
semantics a collector changes nothing observable except
memory footprint. Verdict: no cyclic GC in v1; revisit if
the footprint becomes a measured problem. This section is
the D7 deliverable for the GC item.

## Panics added

77080 dynamic int overflow; 77081 StopIteration (next() on
an exhausted generator); 77082 await of a non-coroutine.

## Debug-batch probe items (D7 slice)

- Field-place method calls on the boxed state
  (`(*b).ch.recv()`, `(*b).ch.destroy()`, `(*b).h.join()`):
  the async.sla `(*c).lock()` chain is the precedent.
- `Chan::new(0)` rendezvous semantics under the text route
  (gosa demo 27 proves the shape on its own gate).
- thread::spawn with a multi-argument producer call and
  JoinHandle<PsaVal> storage in a struct field.
- Whether sa_std/task.sai + thread.sa + result.sa imports
  compose with psa's existing import set.

## Status addendum (written locally 2026-11-25 night, UNVERIFIED)

Implementation complete in psa-src, not yet compiled (no local
Rust toolchain by standing rule; first compile happens in the
unified debug batch on the box). What was written, by layer:

- lexer: Tok::Yield/Async/Await + keywords (match/case stay
  soft idents). ast: Stmt::Yield, Stmt::Match + MatchCase +
  Pattern {Lit, Capture, Wildcard, Seq, Or}, Expr::Await,
  Func.is_async. parser: async def, yield statement, await in
  unary position, match via soft-keyword lookahead, pattern
  parser (dotted value patterns, class patterns, star and
  mapping patterns refused loudly).
- emitter pre-pass (emit_program, after build_class_registry):
  pass 1 pre-registers all-Dyn sigs for every generator/async
  def (so sibling calls resolve to wrapper faces); pass 2
  emits. Methods that are generators or async are refused;
  async generators are refused.
- Generators: emit_generator_def writes wrapper (Chan::new(0)
  + thread::spawn drive) + drive (waits for a start token,
  runs producer, sends end wrap) + producer (body under a
  custom Ctx with gen_chan = "__psa_gench", dyn_ret). The
  start token (sent by __psa_gen_pull on first pull) makes the
  body lazy like CPython -- nothing runs at call time.
  gen_check_returns refuses value returns. D7_GEN_RUNTIME:
  PsaGenState {ch, h, done, started}, __psa_gen_new/wrap/
  wrap_end/pull/next, __psa_iter_source (drains tag 13 into a
  list; ForEach, comprehension-For and list/tuple/set
  constructors route through it when used_gen). next() builtin
  added (one argument only). Imports task.sai/thread.sa/
  result.sa gated on used_gen.
- Async: emit_async_def registers the body as a `::coro`
  worker def (all-Dyn callable, adapter-bound; defaults and
  *args/**kwargs refused), wrapper packs args into a list and
  returns __psa_coro_new(id, args) (tag 14). Await arm in
  emit_expr -> __psa_coro_run (D7_CORO_RUNTIME), sequential
  semantics. Top-level await is ACCEPTED by psa (a pragmatic
  extension; CPython scripts refuse it) -- divergence.
- Match: emit_match + match_check. Subject evaluated once to
  a Dyn temp; per-case ok flag + prologue-hoisted capture
  temps; captures bound to real names before the guard runs
  (CPython leaves guard-failed bindings in place; so does
  psa); bodies are plain blocks. Duplicate capture names in
  one pattern refused; or-alternatives must bind the same name
  set. Literal True/False patterns compare by dyn_eq (value),
  not identity -- divergence from CPython's singleton rule.
- Overflow: guards in __psa_dyn_add/sub/mul/neg (77080) and
  in the shared __psa_floordiv/__psa_floormod helpers
  (MIN//-1 -> 77080; MIN%-1 -> 0, which is the exact result).
  __psa_ipow stays unguarded (noted gap from the design).
- Runtime dispatch: tags 13/14 in __psa_dyn_truthy (true),
  __psa_type_base_name ("generator"/"coroutine"),
  __psa_str_of and both print paths (deterministic
  `<generator object>` / `<coroutine object>`, no address).
- GenExp: emit_genexp builds a synthetic generator def whose
  params are the clause iterables (evaluated eagerly at the
  site) and whose body is the clause nest yielding the
  element; element/condition expressions may reference only
  clause vars, defs, classes and builtins -- anything else
  keeps the loud no-capture refusal.

Examples (CPython outputs fixed 2026-11-25): gen1.psa
(0/1/1/12/[0, 1, 1, 2, 3]/made/start/10/20 -- "made" before
"start" pins the laziness), genexp1.psa (1/[9, 25]/
[(1, 10), (1, 20), (2, 10), (2, 20)]), async1.psa (18/15,
via an asyncio wrapper for the CPython reference),
match1.psa (greeting/zero/ascending pair/pair/small/other/
other/15). overflow1.psa is a NEGATIVE example: CPython
prints 9223372036854775808; psa must exit nonzero (77080).

Additional debug-batch check items (beyond the design list):
- gen_check_returns / func_is_generator / stmts_consume /
  def_calls do not descend into Match bodies uniformly
  (consume/recursion analysis treats match as opaque) --
  verify no regression surfaces for defs analysed through
  those walks.
- A static list passed as an argument to a generator def
  (pre-registered all-Dyn sig vs the Call arm's boxing rules).
- `except StopIteration` is not special-cased: StopIteration
  is not a registered builtin exception class, so the name is
  an unknown-class loud error at compile time; next()
  exhaustion is panic 77081, uncatchable in v1.


## Debug-batch addendum (2026-11-25 night, batches 5–7)

- **Generator channel payload.** SLA channel slots hold at
  most 16 bytes; PsaVal is 24. The generator channel is now
  `Chan<i64>` carrying the wrap tuple's heap pointer
  (wrap.ipay); the receiver rebuilds the PsaVal. The drive
  start token is a plain 0 (its content was never read).
- **Comprehension index collision.** ForEach temporaries
  were named by source line (`__psa_i_<line>`); synthetic
  comprehension/genexp loops share the host line, so
  nested clauses reused one index name and the flattener
  reported UseAfterMove on the outer increment (genexp1).
  Index/stop/unpack-element temps now carry a unique
  ctx.tmp suffix.
- **Static-face i64 arithmetic wraps silently** (fixed-width
  face); only the dynamic face raises the 77080 overflow
  panic. Registered divergence: CPython promotes to
  bigint. (overflow1 routes through the dyn face to pin
  the panic.)
- **tuple slice stop/count.** `__psa_tuple_slice`
  destructured `__psa_slice_bounds` as (start, count)
  where it returns (start, stop, step) — `t[1:]` read one
  element past the end (panic 32). Fixed to the list_slice
  shape.
