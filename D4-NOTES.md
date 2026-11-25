# D4 notes — class (attributes / methods / single inheritance) + attribute access

Design settled 2026-11-25 (main session), following the D3 pattern:
implement locally first, debug/fix in one batch on the box after the
D3 batch. Scope per DYNAMIC-DESIGN §6: class with attributes,
methods, single inheritance; the full attribute-access surface.
Census rows served: `class` （缺口-动态）, `obj.attr` 属性访问 （缺口-动态）.

## Representation (rides the proven D1–D3 machinery, no new concepts)

- PsaVal tag 8 = object instance. Payload mirrors the list/dict
  shape: `ipay = Box::into_raw(Box::new(Vec<PsaVal>))`, element 0 =
  box_int(class_id), elements 1.. = alternating key/value entries in
  insertion order — the D3 ordered-dict layout with a one-element
  header, so the `__psa_vec_*_h` helpers and find/set/del logic carry
  over with a header offset (`__psa_obj_find` etc.).
- PsaVal tag 9 = bound method. Payload `Box<Vec<PsaVal>>` of len 2:
  [box_int(callable_id), self]. `__psa_call_value` is refactored so
  the per-id dispatch becomes `__psa_call_id(id, args)`; tag 7 calls
  it directly, tag 9 builds args' = [self] + args and calls it.
  Adapters and arity panic 77068 are reused unchanged.
- Methods are ordinary psa defs pinned all-Dyn exactly like
  fn_val_code (D3 item 4): params Dyn, return Dyn, registered in the
  same callables table. The def key is `ClassName.method`; the SLA
  emission name is `__py_cls_<class_id>_<method>` (numeric id keeps
  it injective and a valid identifier — a source-level user function
  can never collide). The receiver is simply the method's first
  source parameter, whatever it is named.

## Compile-time class registry (emitter)

classes: Vec<ClassInfo> + by_name map. ClassInfo { id, name,
base_id, methods: flattened map name -> callable_id (own methods
merged over the base's, own wins — single inheritance flattened at
compile time), class_attrs: Vec<(String, Expr)> literal defaults }.

Per class, one generated runtime fn
`__psa_class_<id>_getattr(inst, name_val) -> PsaVal`: instance-entry
lookup is NOT here (see getattr below); this fn covers the class
side only — it byte-compares the name (tag-6 PsaVal, `__psa_str_eq`
against emitted str temps) against every flattened method name and
returns a tag-9 bound method; then class-attr dict lookup; else
panic 77069.

## Semantics settled

- Construction: the Call arm checks the class registry after the
  Dyn value-call hook and before builtins. `Name(args)`: emit
  `__psa_obj_new(id)`; if a flattened `__init__` exists, call its
  SLA fn directly with (inst, args...) through the ordinary def call
  path, check its result tag == 3 (None) else panic 77074, result =
  inst. No `__init__` and args non-empty -> panic 77073.
- getattr(obj, name): tag 8 -> instance entries find; hit returns
  the value; miss -> `__psa_class_<id>_getattr`. Non-object tags ->
  panic 77069 (dict/list method surfaces are D5, not D4).
- setattr(obj, name, v): tag 8 -> instance entries set (D3 dict set
  machinery, header offset). Non-object -> panic 77072.
  `del obj.attr`: tombstone via the dict del machinery; missing ->
  77069. Aug-assign and chained `a.b.c` ride the same get/set pair.
- Explicit base call: `Base.method(self, ...)` in call position
  (callee is an Attr whose base is a class name) emits a direct call
  to that class's method fn with the explicit receiver — this covers
  the `Base.__init__(self, ...)` pattern. `super()` is refused
  loudly in v1 (named in the refusal).
- Class attributes: PRIMARY shape — evaluated once at program
  start into a per-class dict held in a runtime registry
  (`__psa_class_attrs(id)`), so mutation persists and reads follow
  instance -> class -> (flattened) base order. This needs one
  mutable SLA global for the registry root: DEBUG-BATCH PROBE, not
  an assumption. FALLBACK if no usable global: literal class attrs
  evaluated at each class-side getattr miss (reads exact for
  literals), and class-attr assignment refused loudly. Either way
  no silent divergence.
- Dunders in D4: `__init__`, `__repr__`, `__str__`, `__bool__`.
  Print of a tag-8 value: flattened `__repr__`/`__str__` via the
  getattr machinery when defined; otherwise `<Name object>` (no
  address — CPython's address form is non-deterministic, so bare
  object print is excluded from byte-parity examples; documented
  here, not hidden). Truthiness: tag 8 is true unless `__bool__`
  is defined. Equality: pointer identity (ipay) unless __eq__ —
  operator dunders beyond the four above are D5.
- Parser/AST: `class Name:` / `class Name(Base):` at top level
  only (nested class definitions refused loudly). Class body may
  contain only method defs and attribute assignments; any other
  statement is a loud refusal. New AST: ClassDef collected beside
  funcs; `Expr::Attr { base, name }` postfix on any primary;
  assignment/aug-assign/del targets accept Attr. Bare class name in
  expression position (not a call, not an explicit-base callee) is
  refused loudly in v1.

## New panic codes

77069 attribute missing / getattr on a non-object (AttributeError)
77072 setattr on a non-object
77073 class with no __init__ called with arguments
77074 __init__ returned a non-None value

## Acceptance examples (written with the implementation)

class1.psa: Point(x, y) with __init__, a dist2 method, a class
attribute, a ColoredPoint(Point) subclass overriding one method and
calling Point.__init__ explicitly, attribute set/get/del,
dispatch through a Dyn alias, identity equality. Expected output
fixed against CPython 2026-11-25 (run locally):
25 / 3 / point / point / 41 / 9 / 9 / 105 / red / point / 105 /
True / False.

class2.psa: __repr__ + __bool__ (Flag), print, truthiness in if,
object repr inside a list repr. Expected (CPython):
Flag / no / Flag / yes / [Flag, Flag].

## Emitter slice decisions (2026-11-25, main session)

- Registry built in emit_program before main emission: bases
  must precede subclasses; class names colliding with a def or
  a builtin are refused; method defs register under the key
  `ClassName.method` with assigned SLA name
  `__py_cls_<id>_<method>` (Emitter.sla_names; sla_name()
  consults it at the definition site, the adapters and the
  def-call site).
- All methods of all classes are pinned as callables at
  assembly (method_callable_id): all-Dyn params; return may be
  Dyn/Void/Int/Float/Bool and the adapter normalises (Void ->
  None, scalars box). Defaults/*args/**kwargs methods cannot
  be bound (loud); explicit base calls still bind them through
  the ordinary def-call path. Defining a class therefore emits
  every method, even unused ones -- loud, CPython-spirited.
- __psa_call_value refactored: tag 9 -> __psa_call_bound
  (prepends the receiver, reuses __psa_call_id); tag 7 ->
  __psa_call_id. Both generated, together with __psa_call_bound,
  whenever callables or classes exist.
- Class attributes take the FALLBACK shape (the registry step
  waits for the mutable-global probe, per Jev R1202): literal
  attrs only (non-literal refused at registry time), evaluated
  at each class-side read -- instance getattr misses and
  `ClassName.attr` reads alike. Class-attr assignment and
  deletion are refused loudly.
- Construction refuses kwargs (v1); __init__ with defaults or
  *args/**kwargs is refused at the construction site; a static
  non-None __init__ return is an emit error, a dynamic one is
  tag-checked at runtime (77074); no __init__ + arguments ->
  runtime panic 77073 via __psa_obj_check_noargs.
- Generated machinery (gen_obj_machinery, emitted for every
  dynamic program because the static runtime references it):
  __psa_getattr, __psa_class_getattr dispatch, per-class
  __psa_class_<id>_getattr, __psa_obj_default_str (`<Name
  object>`), __psa_obj_repr (__repr__ only), __psa_obj_str
  (__str__ then __repr__), __psa_obj_truthy (__bool__).
  Runtime additions: __psa_obj_new/class/find/set,
  __psa_setattr (77072), __psa_delattr (77069),
  __psa_obj_check_noargs (77073), __psa_box_method,
  __psa_method_id/self; dyn_eq identity for tags 8/9;
  dyn_truthy tag 8 via dispatch, tag 9 true; print/repr tag 8.

## Debug-batch check items (D4)

- Forward references: static runtime text calls the generated
  __psa_obj_* dispatchers (defined later in the file). Expected
  to resolve program-wide like every other cross-reference;
  verify on the first class program.
- An expression-statement construction (`Point(1, 2)` alone)
  leaves a bare temp-name statement; check the SLA checker
  accepts it, else special-case ExprStmt.
- D3 wart spotted while wiring truthiness: a tag-7 callable in
  a boolean context panics 77010, but Python says a function is
  truthy. Check in the D3 batch; fix is one line in
  __psa_dyn_truthy if confirmed in scope.
- Negative cases: missing attribute (77069), setattr on an int
  (77072), Point(1) arity error at emit, class without
  __init__ called with args (77073), __init__ returning a value
  (77074 dynamic / emit error static), non-literal class attr
  refused at registry, class-attr assignment refused.

## Status

Design settled 2026-11-25 20:3x CST. Parser/AST slice landed
locally the same evening (uncompiled — no local Rust toolchain;
verified in the box debug batch after D3):
- lexer: Tok::Class + `class` keyword.
- ast: Expr::Attr, Stmt::AttrAssign, Stmt::AttrDel, ClassDef,
  Program.classes.
- parser: parse_class (top level only; single base; body = method
  defs + attribute assignments + pass, everything else refused
  loudly; multiple inheritance refused), Dot postfix -> Attr when
  not a call, attribute assignment with backtracking (chained
  `a.b.c = v` included), `del base.attr` single-target form.
- emit coherence: collect_names / desugar_expr / desugar_stmt /
  stmts_consume / expr_consumes arms for the new nodes (attribute
  names never counted as variables; Attr base is a borrow in the
  consume analysis), and loud placeholder arms in emit_expr /
  emit_stmt so nothing silently miscompiles before the emitter
  slice lands.

Emitter slice landed locally 2026-11-25 21:2x CST (same
uncompiled status; verified in the box debug batch after D3,
together with the mutable-global probe for class attributes):
registry + method defs, runtime tags 8/9 + obj helpers,
generated getattr/str/repr/truthy machinery, construction,
instance/class attribute reads, setattr/delattr (incl.
aug-assign), method calls via getattr + dispatch, explicit
base calls via the def path, class1.psa + class2.psa written
with CPython-fixed expected outputs. Remaining D4 work is only
the debug/fix batch itself (plus the probe deciding whether
the class-attr fallback is replaced by the registry shape).

## Addendum (2026-11-26): the runtime registry landed

The fallback shape above is replaced by the runtime registry
(the globals probe passed; R1202 probe_then_build 8/8):

- The registry root is a tag-4 list (indexed by class id)
  whose box pointer lives in the sa_std globals cell
  "psa.clsattrs"; `__psa_clsreg()` creates it lazily through
  `sa_global_u64_set_once` and re-reads the prevailing
  pointer on a lost race. `__psa_class_attrs_raw(cid)`
  grows the list with None slots and lazily creates each
  class's tag-5 dict.
- `__psa_class_attrs_seed()`, emitted as main's first
  statement, writes every class's OWN literal attributes
  into its dict (registry order: bases precede subclasses).
  Declared attributes stay scalar-literal-only at registry
  time; the registry governs mutation, not declaration.
- Reads: `__psa_class_<cid>_cgetattr(name)` walks the
  ancestor dicts in flattened order (own first), panicking
  77069 on a miss. Both instance getattr misses and
  `ClassName.attr` reads route through it, so attributes
  created by runtime assignment are found the same way.
  Method names on the class side stay an emit error.
- `ClassName.attr = v` writes the class's OWN dict
  (visible to subclasses and unshadowed instances, as in
  CPython); augmented assignment reads through cgetattr,
  applies the binop and writes back. `del ClassName.attr`
  deletes from the own dict only and panics 77069 when the
  name is absent there (an ancestor's attribute is not
  deleted through the subclass).
- Acceptance: examples/class3.psa (method-driven class
  mutation, inherited reads, runtime-created attribute,
  deletion) matches CPython byte-for-byte, and the unified
  batch re-ran green with the registry in place.
