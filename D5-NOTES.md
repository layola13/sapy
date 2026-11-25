# D5 notes — comprehensions, tuples, sets, method sets, builtin face

Scope per DYNAMIC-DESIGN §6: 推导式、元组拆包、str/list/dict 方法集
高频部分. Plus the census §4 builtin face that rides the same
machinery (conversions, type/isinstance, print sep/end, f-string
as a general value, list()/tuple()/set() constructors), because
D5's method/runtime work makes them cheap and v1's gate is
"vast majority of Python syntax".

## New tags

- tag 10 = tuple: payload Box<Vec<PsaVal>>, immutable by
  convention (no mutating runtime entry points). Index/len/
  iteration/eq/repr/truthiness ride the dyn protocol; repr is
  `(a, b)` with a trailing comma at len 1 and `()` at len 0.
- tag 11 = set: payload Box<Vec<PsaVal>> in insertion order,
  membership under __psa_key_eq (extended: tuples compare
  elementwise, objects by identity -- both are hashable in
  Python). Unhashable elements (list/dict/set) panic 77066,
  the dict-key rule. Repr is `{a, b}`; the empty set prints
  `set()` (CPython). Iteration order is insertion order -- a
  documented divergence from CPython's hash order, same family
  as the ordered dict.

## Tuples and unpacking

- `(a, b)`, `(a,)`, `()` and bare `a, b` in return position are
  TupleLit; a tuple literal is a dynamic value materialised in
  a temp anywhere an expression is expected (immutable, so the
  temp shape is safe everywhere).
- Unpacking rides the existing MultiAssign: `a, b = <tuple or
  list or str>` evaluates the RHS once and binds elementwise at
  runtime (__psa_unpack_check + iter_get), arity mismatch
  panics 77077. Nested/starred targets stay loud in v1.

## Comprehensions

One AST node (Comp) with clauses in source order (For/If).
Emission materialises the result container in a temp and emits
the clause nest as statements around the element evaluation:
For over a dynamic iterable is an index loop over the dyn
iteration protocol (length re-read per round, matching
ForEach); For over range(...) is a numeric loop; For over a
static list is an index loop over the Vec. Loop variables live
in a pushed Ctx scope (Python 3 comprehension scoping: no
leak). Dict comps set key/value; set comps add. Generator
expressions parse into the same node with kind GenExp; their
semantics are D7 (generator machinery), refused loudly until
then.

## Method sets

Emitter dispatch by method NAME for dynamic bases: a builtin
method table maps the name to a runtime fn
`__psa_m_<name>(v, name, args)`; every such fn first defers
tag 8 to getattr + call_value (an object's own method always
wins over the container builtin of the same name), then checks
its container tag (wrong kind -> 77069 AttributeError), then
arity (77068). Name-base method calls (MethodCall) and general
base calls (Call of Attr) share the dispatch.

- list: extend, insert, pop, remove, clear, copy, index,
  count, reverse, sort (dyn_lt order; incomparable -> 77012).
- dict: get, keys, values, items (list of 2-tuples), pop,
  setdefault, update, clear, copy.
- str: upper, lower, strip, lstrip, rstrip, split, join,
  replace (optional count), startswith, endswith, find,
  count, isdigit, isalpha. Case mapping and classification
  are ASCII-only in v1 (bytes >= 0x80 pass through) -- a
  registered partial, Unicode tables are a follow-up.
- set: add, remove (missing -> 77076), discard, clear, copy,
  union, intersection, difference (any iterable argument).
- Static (monomorphic) lists keep append only; the other
  methods on a static list are refused loudly toward the
  dynamic face (documented v1 boundary).

## Builtin face

- int(x)/float(x)/str(x)/bool(x) in call position. int/float
  parse strings at runtime (failure -> 77079 ValueError).
  str() of a float stays blocked by (xxiii) (77030, never a
  wrong value); str() of containers is refused (77030) until
  repr-to-string builders exist; str() of an object rides
  __psa_obj_str.
- list(x)/tuple(x)/set(x) build from any iterable (incl.
  range(...) special-cased at the call site).
- type(x) returns the type NAME as a dynamic string (a type
  object is out of v1 scope; the name is what programs print).
  Class instances report their class name (generated
  __psa_class_name).
- isinstance(x, T): T a class name, a builtin type name, or a
  tuple of those. Class checks are subclass-aware (the
  emitter precomputes descendant id sets from the registry);
  bool counts as int (CPython: bool subclasses int).
- print: dynamic values join multi-argument print (printed via
  a no-newline dyn printer, space-separated); sep=/end= are
  accepted as string literals (non-literal refused loudly).
- f-string as a general value: parts concatenate into a
  dynamic string (ints via a runtime formatter, bools as
  True/False, strings as themselves; floats stay blocked by
  (xxiii)). The D0 print-only fast path (emit_print_fstr)
  stays in place for the sole-argument print position -- its
  output agrees with the value path on the D0 cases; the
  value path covers every other position and lifts the
  one-bool interpolation limit there.

## New panic codes

77075 method argument type error
77076 value not found (list.index/remove, set.remove)
77077 unpack arity mismatch
77079 int()/float() string parse failure (ValueError)

## Acceptance examples

tuple1.psa (literals, unpack, swap, return tuple, nested),
comp1.psa (list/dict/set comps, ifs, nested fors, range comps),
methods1.psa (list + dict methods), methods2.psa (str methods),
methods3.psa (set literal + methods + ops via methods),
builtins1.psa (conversions, type, isinstance, list/tuple/set
constructors, print sep/end, f-string value). Expected outputs
fixed against CPython at write time (all six run clean under
python3 on 2026-11-25; the debug batch diffs psa output
against a fresh python3 run).

## Status (2026-11-25, local, UNCOMPILED)

Implementation complete on the local VM: AST/parser (tuple
and set literals, comprehensions, Unpack statement, ForEach
multi-targets, method calls on general bases desugared to
Call-of-Attr), runtime (tags 10/11, the __psa_m_* method set,
str methods, conversions, constructors, type/isinstance
support, __psa_print_dyn_nl), emitter (builtin method
dispatch shared by MethodCall and Call-of-Attr, tuple/set/
comp temps, Unpack + multi-target ForEach arms, conversion
builtins, print render path with sep/end). Compilation and
the acceptance diff wait for the unified box debug batch
(user order 2026-11-25 21:25: write everything first).

Divergences registered while writing (all loud or
documented, none silent):

- dict.keys()/values()/items() return snapshot LISTS, not
  live views; their reprs differ from CPython's
  dict_keys([...]) form accordingly. Acceptance examples
  wrap them in list(...), which is byte-equal.
- type(x) returns the DISPLAY string `<class 'NAME'>`
  (byte-equal when printed); psa has no class values, so
  type() results cannot be compared with class objects or
  used to construct. User classes show no module prefix
  (CPython shows `__main__.`); psa has no modules until D6.
- set iteration/printing order is insertion order (see
  above); CPython's is hash order. For small-int sets the
  two coincide; string sets can differ run to run in
  CPython itself (hash randomisation), so no byte-parity
  example prints a multi-element string set.
- A method call on a general base goes through Call-of-Attr;
  keyword arguments in that position are refused (the
  MethodCall node for plain variables accepts positionals
  only too -- psa method calls take no kwargs in v1).
- int("1_000") and float("inf")/float("nan") are refused
  (77079 / compile-time literal error); CPython accepts
  them. Registered partials.
- sort() is a selection sort over dyn_lt (O(n^2)); a
  performance characteristic, not a semantic gap.

Debug-batch check items:

- `(*b).remove(i)` / `(*b).truncate(n)` in the vec surgery
  helpers are unverified SLA surface extensions.
- emit_container_temp on an empty ListLit (list() with no
  arguments routes there).
- The synthetic range comprehension inside list()/tuple()/
  set() (list(range(...)) family).
- print render path: sole static-list print still uses the
  historical path; multi-arg with a static list is refused.
- Str-literal method bases ("a,b".split(",")) ride the new
  parser desugar + Call-Attr Str special case.
