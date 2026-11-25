# D6 notes — exceptions, with/files, import

Scope per DYNAMIC-DESIGN §6: 异常（try/except/finally/raise)、
with（文件最小形）、import（自有模块）。三者都是 v1 语法门的
核心项。实现先行、统一调试批后置（用户 2026-11-25 21:25 令）。

## Exceptions: shape

SLA has no exception unwinding, and psa's runtime panics
(77050 等） are NOT Python exceptions and stay uncatchable in
v1 (registered: mapping runtime failures to exceptions is
future work). The exception surface is explicit raise + try
only, and v1 propagation is INTRA-FUNCTION: a raise whose
dynamic extent does not include a try in the same function
aborts with panic 77100 (uncaught), and a raise never crosses
a call boundary. Nothing is silently swallowed anywhere.

Mechanics (all emitter-level, no runtime globals needed):

- Per-function pending slot: any function whose body contains
  a try or a raise gets a hoisted `__psa_excpend: PsaVal`
  (None literal default) in its prologue. Pending <=>
  slot.tag != 3. Set/take are emitter idioms (dyn store into
  the slot / read + reset), not runtime calls.
- Each try emits a single-iteration `while true { body;
  break; }` wrapper. While emitting a try body (exc_depth > 0)
  EVERY statement gets a trailing guard
  `if __psa_excpend.tag != 3 { break; };` — inside a user loop
  the guard breaks the loop, then the loop statement's own
  guard breaks the wrapper, so a raise at any nesting depth
  unwinds to the wrapper in order. Handler and finally bodies
  emit with exc_depth unchanged but outside the wrapper (no
  break guards there; see the raise restrictions below).
- Dispatch after the wrapper: if pending, handlers are tried
  in order; a match is an inline class test (the instance's
  class id against the handler class's descendant set, the
  isinstance machinery). A matched handler binds `as e` (Dyn)
  and clears the slot first. No match: the exception stays
  pending for the enclosing try (its guard fires) or the
  function epilogue check `if pending { panic(77100); }`.
- else runs when the body completed with no pending and no
  control flag set. finally ALWAYS runs: after the body on
  the normal path, after a matched handler, and before
  propagation — including the return/break/continue paths:
- Control flow crossing a try (with or without finally —
  the wrapper alone breaks naive break/continue) is rewritten
  through per-try flag slots: break/continue inside a try body
  set the try's flag and break the wrapper; the try epilogue
  re-emits the real break/continue (through any enclosing
  try's rewrite, recursively). Return inside a try-with-
  finally stores the value in a per-try slot of the function's
  return type, sets the flag, breaks the wrapper; after
  finally, `if flag { <real return of the slot> }` (itself
  rewritten by any enclosing try). Return inside a try
  WITHOUT finally is a plain return (no finally to run,
  handlers do not run on return — CPython agrees).

Restrictions (all loud, registered):

- raise inside an except body: only a bare `raise` (re-raise
  of the caught instance) and only as the block's LAST
  statement. Any other raise in a handler, and any raise in
  a finally body, is refused at emit.
- `raise X from Y` (chaining) is refused in v1.
- Exceptions raised by called functions do not propagate
  (the callee's own uncaught check panics 77100 first).
- ZeroDivisionError & friends are never raised by psa
  arithmetic in v1: division by zero stays a runtime panic,
  because psa panics are not exceptions (above).

Builtin exception classes, registered as synthetic classes
BEFORE user classes (ids 0..8): Exception (root), ValueError,
TypeError, KeyError, IndexError, AttributeError,
ZeroDivisionError, RuntimeError, NameError — all deriving
from Exception. Construction is special-cased in the Call
arm: `ValueError("msg")` / `ValueError()` emit
`__psa_exc_new(cid, msg)` (a tag-8 instance whose "msg"
entry holds the argument or None); the D4 __init__ arity
rules do not apply to them. User classes may derive from
them (handler matching is ancestry-based, so a user subclass
of ValueError is caught by `except ValueError`).
str(exc) is its msg (CPython); repr(exc) is `Name('msg')`
for a string msg, `Name()` for None (generated machinery
branches on the builtin-exception id range). A second
positional argument to an exception constructor is refused
in v1 (CPython packs args into a tuple; psa keeps the
single-message form).

raise forms: `raise ExcClass`, `raise ExcClass(args)`,
`raise <instance expr>`, bare `raise` (handler-last only).
Raising a value that is not an exception instance panics
77101 at runtime (CPython raises TypeError).

New panic codes: 77100 uncaught exception, 77101 raising a
non-exception, 77110 file IO failure, 77111 file closed /
wrong mode.

## with + files (tag 12)

`open(path[, mode])` is a Call-arm builtin (shadowable like
the D5 builtins). Modes: "r", "w", "a" (plus a no-op "t");
anything else (b, +, x) is refused loudly in v1. The file
object is tag 12 with payload Box<Vec<PsaVal>>:
[handle (int), mode code (int: 0 r / 1 w / 2 a), closed
(bool)]. The fs surface is sla_std/wrap/fs.sla
(sla_std_fs_open_read / sla_std_fs_open_write /
sla_std_fs_file_read / sla_std_fs_file_write /
sla_fs_file_close / sla_std_fs_file_seek), reached with
stack-buffer chunk copies (4096) because psa strings have
no data-pointer surface — paths longer than the chunk
panic 77110. Methods (in the D5 builtin table): read(n?),
readline(), write(s) -> int (code points written), close().
Read/write on a closed file or in the wrong mode panic
77111; IO failures panic 77110. Errors are panics, not
catchable exceptions, in v1 (see above).

`with` desugars at emit: items are evaluated left to right,
each bound to its `as` name (or a hidden temp), and the body
is emitted as a synthetic try whose finally closes the files
in reverse order — so close runs on normal, exception,
break/continue and return exits alike. Only file objects
(and expressions evaluating to them) are accepted as with
items in v1; a general __enter__/__exit__ protocol is
refused loudly.

## import (own modules only)

Driver-level (main.rs): after parsing the main program, each
top-level import is resolved to `<dir>/<module>.psa` next to
the main file, parsed, validated and merged BEFORE emission.
`import foo` in a function body is a parse error (top level
only in v1). A missing module file is a loud driver error.

Module shape restrictions (loud refusals): a module's top
level may contain only defs, classes and literal assignments
(readable as module attributes, evaluated at the read site
like D4 class attributes). Any other top-level statement is
refused. Module defs may reference their own module's defs
(bare-name calls are renamed to the dotted key at merge) and
module literal attributes (substituted at merge); other free
names are the importing program's business only if they are
builtins — cross-module global reads are refused at merge
when detectable, and registered as a v1 boundary.

Merge mechanics:

- Module defs enter prog.funcs under the dotted key
  `foo.bar`; their emitted SLA name is id-based
  (`__py_mod<modid>_<name>`, assigned in emit_program's defs
  loop), so no identifier encoding can collide.
- `from foo import bar` additionally binds the plain key
  `bar` to the same def (a collision with an existing def or
  class name is a loud merge error). `as` aliases bind the
  alias key instead.
- `import foo` + `foo.bar(...)`: MethodCall on the module
  name routes to the dotted def; `foo.X` reads a module
  literal attribute (Attr arm, module registry parallel to
  the class registry); `foo.SomeClass(...)` constructs a
  module class — module class names share the single class
  registry and must not collide with any other class name
  (loud merge error).
- Importing the same module twice is a no-op (module
  registry by name). Circular imports are refused loudly at
  merge (the merge is depth-first with a visiting set).

## Acceptance examples

exc1.psa (try/except matching, as-binding, else, finally,
re-raise shape, nested try propagation, raise in a helper
caught in the same function — note: same FUNCTION, helpers
called from inside a try do not propagate, so exc examples
keep raise and try in one frame, plus one uncaught case
verified as a NEGATIVE example with expected panic 77100),
with1.psa (write then read back a file under with, close on
exception exit), mod_main.psa + mod_a.psa (import, from-
import, module attr, module class). File examples use a
fixed relative path and clean up after themselves (the
debug batch runs them in a scratch dir).

## Status

Design settled 2026-11-25 (this file). Implementation in
progress locally; see the status addendum appended when the
code lands.

## Status addendum — 2026-11-25 (written locally, UNVERIFIED)

All D6 code is written locally (lexer/ast/parser/emit/main),
uncompiled; verification waits for the unified box debug
batch after D7 (user order 2026-11-25 21:25: write everything
first, then test once).

Implemented as designed: try/except/else/finally with the
per-function pending slot and TryFrame flag machinery;
handler matching against compile-time descendant sets;
builtin exception classes Exception..NameError (ids 0..8)
constructed via __psa_exc_new; raise forms (class call,
instance, class shorthand, bare raise restricted to a
handler's last statement); with over the same machinery with
reverse-order __psa_with_close; open() (tag 12 file values,
modes r/w/a + no-op t) over the sa_std fs externs with
read/read(n)/readline/write/close methods; import and
from-import merged by the driver (dotted def keys, plain
class names in the shared registry, literal module attrs
evaluated at the read site, circular imports refused).

Panics added: 77100 uncaught exception, 77101 raise of a
non-exception value, 77110 file IO failure (incl. path over
1000 bytes, non-str path), 77111 closed file / wrong mode.

Registered divergences and v1 limits:
- psa's own runtime panics (77050 etc.) are NOT Python
  exceptions: a try block does not catch them. Only explicit
  `raise` values participate in the exception machinery.
- KeyError's str is the bare message; CPython renders it as
  the repr of the argument (quotes). Not modeled in v1.
- File str omits the `encoding='...'` segment CPython shows.
- open() mode must be a string literal; binary/+ modes,
  encoding and newline arguments are refused loudly.
- Module top level: defs, classes and literal assignments
  only. From-import bindings are program-wide (CPython binds
  them module-locally). Aliasing a class via from-import is
  refused. Class names share one program-wide registry.
- A module def reading a module literal attribute gets the
  literal substituted at the read (no mutable module state).
- read(n) is byte-at-a-time internally (correct code-point
  counting, slow); read() chunks at 4096 bytes.

Examples (outputs fixed by running CPython 2026-11-25):
- exc1.psa — matching, as-binding, else, finally, subclass
  matching via Exception, nested propagation across calls,
  repr/str of an exception. Expected: `caught: div by zero`,
  `fin`, `y = 5`, `else2`, `exc caught`, `outer caught: deep`,
  `RuntimeError('boom')`, `boom`, `done`.
- exc2_uncaught.psa — negative: uncaught raise must exit
  nonzero (panic 77100); CPython exits 1 with a traceback.
- with1.psa — write/read/readline/append round trip through
  with blocks. Expected: `wrote 12`, `'hello\nworld\n'`,
  `'hello\n'`, `'world\n'`, `''`,
  `'hello\nworld\nthird\n'`, `done`. Creates with1_out.txt
  in the working directory (w mode truncates, reruns are
  deterministic).
- mod_a.psa + mod_main.psa — import, from-import, module
  attr reads, a module class. Expected: 10, 50, 20, 3, 11,
  12, `hi bob`. (CPython reference run needs mod_a copied
  to mod_a.py; psa resolves the .psa file itself.)

Debug-batch check items (D6 slice): the stack-buffer
variable-index reads in the file runtime (`buf[k]`) follow
the __psa_print_str precedent but are unverified against
the fs path; append mode relies on open_write(truncate=0)
plus seek END (whence 2); sa_std_close pairs with
sa_std_fs_open_* handles per the sci sendfile demo.


## Debug-batch addendum (2026-11-25 night, batches 5–7)

Three D6 defects found by the unified batch and fixed in
emit.rs:

1. **raise negated a void call.** The raise path emitted
   `if !(__psa_exc_check(t)) { panic(77101); }`, but
   `__psa_exc_check` returns void and panics 77101 itself.
   The SLA checker rejected the negation as a binary_expr
   type mismatch, failing every program containing a raise
   of a computed value (exc1/exc2). Fixed: plain call.

2. **Second `except ... as e` binding wrote to a dead
   variable.** Ctx registers names function-wide, so the
   second handler binding `e` took emit_bind_dyn_code's
   store path — but the first handler's textual `let e`
   sits in a sibling block, out of scope (UndefinedVariable
   at check). The handler site now always materialises a
   fresh textual let inside its own block; CPython deletes
   the name at handler exit anyway, so block-local binding
   is faithful. (Same family as the match-capture fix.)

3. **Propagation never crossed a call boundary.** The
   pending slot was a per-function local: a raise in a
   callee panicked 77100 at its own raise site even when
   the caller had a matching handler — exc1's div/outer
   cases could never pass, despite this file's example
   list promising "nested propagation across calls". The
   slot is now process-global: two sa_std globals u64
   cells (`psa.exc.tag`, `psa.exc.ipay`) behind
   `__psa_exc_tag/__psa_exc_get/__psa_exc_store` in the D6
   runtime (exception values are always tag 8 with fpay
   0.0, so the pair is lossless). Unwinding: a raise with
   no enclosing try in its function returns the function's
   default immediately (panic 77100 in main); every
   call-containing statement outside a guard wrapper gets
   a trailing `if __psa_exc_tag() != 3 { <unwind> }` check
   (suppressed inside finally bodies, whose epilogue check
   propagates instead); the try epilogue does the same
   when its frame pop empties the stack. Registered
   divergences: a call inside an if/while CONDITION is
   only checked after the whole statement completes; the
   slot is process-global, so an exception raised in a
   generator's producer thread is visible process-wide —
   generator/exception interaction stays best-effort in
   v1 (the producer unwinds, the generator ends).
