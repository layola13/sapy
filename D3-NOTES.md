# D3 notes — function dynamic face + ordered dict + name mangling

## Item 1: user function name mangling (code landed locally 2026-11-25, box verification pending)

Problem (D2 exit notes): a user function named like a std method the
runtime calls (`get`/`insert`/`push`/...) breaks SLA checker method
resolution — `(*b).get(k)` inside `__psa_map_get_h` was checked
against the user's 0-arg `get` ("call get with 2 args", loud).

Fix (psa-side, systematic): user-defined functions are emitted
under a `__py_` prefix via `sla_fn_name()` in emit.rs. Emission
points are exactly two: the definition site in `ensure_fn`
(`fn __py_<name>(...)`) and the single user-function call site in
`emit_expr` (`__py_<fname>(...)`). Internal bookkeeping
(defs/sigs maps, ctx.fname, def_calls) stays keyed by the source
name. The prefix is injective, cannot collide with `__psa_*`
runtime helpers or the generated `fn main` entry, and a user
function named `main` becomes `__py_main`, distinct from the
entry point.

Regression example: `namecollide1.psa` (local working copy;
ships to box examples/ at the next sync). User defs `get` and
`insert` coexist with dict indexing (goes through
`__psa_map_get_h`'s method call) and list append. Expected
CPython-identical output: 42 / 11 / 4 / 20. Before the fix the
emitted SLA fails the checker loudly; after, it must build and
print the four lines.

Status: code in ~/workspace/sapy/psa-src/src/emit.rs (helper at
the ListLit section, def site, call site). NOT yet compiled or
regression-run — local VM has no Rust toolchain (standing rule);
box build + full regression (34/34 must stay green, plus the new
example) waits for the box window after Kit's task-160 control
rerun finishes.

## Item 2a: default parameters + keyword arguments (code landed locally 2026-11-25, box verification pending)

Per DYNAMIC-DESIGN §6, D3 = function dynamic face: default
params, keyword params, *args/**kwargs, lambda/closures. This
slice lands the argument-binding machinery the rest builds on.

AST: `Func.params` is now `Vec<(String, Option<Expr>)>` (name +
optional default); `Expr::Call` gains a third field,
`Vec<(String, Expr)>` keyword arguments. Every Call pattern in
emit.rs (emit path, ExprStmt print branch, expr_consumes,
def_calls, expr_mentions) was updated and grep-audited.

Parser: defaults parsed in `parse_func` (Python's rule enforced:
a non-default parameter may not follow a default one); keyword
arguments parsed in call argument lists via an Ident+Assign
lookahead (duplicate keyword and positional-after-keyword are
parse errors); `*`/`**` in parameter lists and call-site
unpacking are refused with a named "D3 later item" message.

Emitter: one binding step in the Call path, before argument
emission — positionals in order, keywords matched by parameter
name (unknown keyword / multiple values are loud errors),
remaining parameters filled from their defaults. The bound list
always has exactly the parameter count, so ensure_fn and the
box/unbox + borrow machinery downstream see a plain positional
call. A fast path skips binding when the call has no keywords
and the callee declares no defaults.

Semantics boundary (honest): defaults are evaluated at each
call site, so they are restricted to literals (int/float/str/
bool/None) — indistinguishable from Python's evaluate-once for
immutable literals. Mutable or computed defaults are refused
loudly at the call that would use them. print()/len() reject
keyword arguments loudly (no sep/end in psa).

Ownership analysis: expr_consumes now traces a parameter passed
by keyword into the callee parameter of that name (same visiting
discipline as the positional path), so borrow/consume verdicts
for list parameters stay sound under keyword passing.

Acceptance example: `fnargs1.psa` — defaults, positional/keyword
mixing, keyword-only fill, bool default. Expected CPython output:
111 / 103 / 6 / 14 / 18 / 103 / 1 / 0 / 0. Negative cases to
verify on the box: missing required argument, unknown keyword,
multiple values for one argument, non-literal default — all must
fail loudly at emit.

## Item 2b: *args parameters (code landed locally 2026-11-25, box verification pending)

`Func` gains `vararg`/`kwarg` fields plus `param_names()` /
`param_count()` helpers (signature order: fixed, *args,
**kwargs); ensure_fn's arity check, borrow analysis, parameter
variables and the definition-site parameter list all go through
the helpers, so collection parameters are ordinary trailing
Dyn parameters at the SLA level.

Parser: `*name` and `**name` parameter forms parsed. Loud
refusals: bare `*` keyword-only marker, keyword-only parameters
after *args, any parameter after **kwargs, duplicate *args.

Binding (same step as 2a): surplus positionals are collected
into a ListLit bound to the vararg slot; the argument loop
builds that slot (and the kwargs slot) through
`emit_container_temp`, so the collection is ALWAYS a dynamic
list temp — the callee's vararg parameter is uniformly Dyn at
every call site regardless of the surplus element types.
**kwargs: the def parses and calls bind; surplus keywords
collect into a dynamic dict with str keys via the item-3
ordered dict (filled in when item 3 landed, same day).

Acceptance example: `fnargs2.psa` — len() of the collection,
indexing, a while loop accumulating dynamic elements. Expected
CPython output: 0 / 3 / 5 / 6 / 15. (The accumulator is seeded
from `args[0]` so it is dynamic from the start; whether a
static-typed accumulator may absorb dynamic elements is a
debug-phase probe, not assumed.)

## Item 3: ordered dict (code landed locally 2026-11-25, box verification pending)

Dict storage replaced: the payload is no longer a
Box<HashMap<i64, PsaVal>> under an i64 key normalisation, but a
Box<Vec<PsaVal>> of alternating key/value entries in insertion
order, manipulated only through the proven __psa_vec_*_h
helpers (the HashMap helpers and __psa_key_norm are deleted).
Lookup is a linear scan under Python key equivalence
(__psa_key_eq): the numeric family compares by value across
tags (1 == True == 1.0 are one key), None equals only None,
str compares by content; container keys are unhashable
(panic 77066 at set). Deletion writes a tombstone (a tag-4 key
slot, which can never be a live key), so surviving order is
kept and a re-inserted key goes to the end, exactly like
CPython. Overwrite keeps the original position.

Surfaces that came with it: str and float keys everywhere
(dict literals, subscript read/write, `in`, del); dict repr
({'k': v, ...} in insertion order, keys and values through the
existing repr printer); dict == dict (order-insensitive, by
key equivalence); for-each over a dict iterates keys in
insertion order (__psa_iter_get tag 5 now returns
__psa_dict_key_at); `del d[k]` on dynamic dicts (missing key
panics 77054, Python's KeyError). Retired panics: 77053 (key
normalisation), 77056 (dict ==), 77057 (print dict), 77061
(dict iteration). New panic: 77066 (unhashable dict key).

Lookup cost is O(n) per operation and len/key_at are O(n)
scans over tombstones; iteration is O(n^2) worst case. That is
a performance characteristic, not a semantic gap: if the v1
bench shows dict-heavy cells losing to CPython, the follow-up
is a hash index beside the entry list. Noted, not hidden.

Acceptance examples: `dict1.psa` (insertion order, overwrite
position, int/bool key unification, del + reinsertion order
via print order, repr, eq) and `fnargs3.psa` (**kwargs
collection: len, key iteration in call order, dict return,
empty call, and a fixed+*args+**kwargs mix). Expected outputs
are CPython's, recorded in the debug-batch checklist.

## Item 4: lambda/closures (code landed locally 2026-11-25, box verification pending)

Implemented exactly per the design below: lexer `lambda`
keyword, parser lambda expressions (plain identifier params;
defaults and *args/**kwargs in a lambda refused loudly), the
desugar pre-pass in emit_program (synthetic `__psa_lambda_<n>`
defs + FnVal nodes + capture refusal), callable pinning in
fn_val_code (all-Dyn signature; defaults/collection functions
refused as values), PsaVal tag 7 via __psa_box_fn, calling
through a Dyn variable (args boxed into a dynamic list temp,
kwargs refused), and the generated __psa_call_value dispatch
with per-function adapters (panics 77067 not-callable /
77068 arity). A def name in value position takes the same
FnVal path; a variable shadows a def name, as in Python.

Acceptance example: `lambda1.psa` — a lambda bound to a name,
a lambda passed inline, a def used as a value, a def passed
to a higher-order def. Expected CPython output: 42/42/42/20.
Negative case for the debug batch: a lambda referencing a
top-level variable must fail at emit with the capture
message. The psa version string in main.rs gets its D3 bump
when the debug batch lands on the box.

## Item 4 design record (settled 2026-11-25 before implementing)

Function-value representation, fitted to psa's architecture
(signatures are pinned per function by ensure_fn):

1. Lexer gains a `lambda` keyword; the parser accepts lambda
   at expression level with plain identifier parameters only
   (defaults/*args/**kwargs inside a lambda are refused loudly
   in v1). AST node Expr::Lambda.
2. A desugar pre-pass over the whole program replaces each
   lambda with a synthetic def `__psa_lambda_<n>` (body =
   return of the lambda body) plus an Expr::FnVal node.
   Capture analysis on the lambda body: free names that are
   user defs or builtins are referenced by name (no capture);
   any other free name is a captured local and is REFUSED
   LOUDLY — true closures need heap cells for captured
   variables (late binding), which is a separate emitter
   change, and a silent by-value capture would miscompile
   the loop-closure cases. Nested lambdas capturing an outer
   lambda's parameters refuse by the same rule.
3. Callable pinning: a function used as a value is pinned to
   an all-Dyn signature (Dyn params, Dyn return) through the
   ordinary ensure_fn/boundary machinery; if it was already
   emitted with a static signature, the value use is a loud
   error. Defs with defaults/vararg/kwarg cannot be used as
   values in v1 (direct calls unaffected).
4. Runtime: PsaVal tag 7 = callable, ipay = function id.
   The emitter generates a per-program dispatch
   `__psa_call_value(f, args)` with one adapter per callable
   function (arity panic 77068; calling a non-callable panics
   77067). Calling through a Dyn variable boxes the arguments
   into a dynamic list temp and goes through the dispatch;
   keyword arguments on a value call are refused loudly in v1.
   A def name used as a value (f = some_def) takes the same
   FnVal path; a variable shadows a def name, as in Python.

Everything else in the D3 list (name mangling, defaults/
kwargs, *args, **kwargs, ordered dict) is now implemented
locally; lambda implementation follows, then the unified box
debug/fix batch, per the implement-first order.
