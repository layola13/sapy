# SA-Py Ownership Semantics Spec — v1 draft

Status: draft for the v1 implementation line. Principles come from the
SA-Py Whitepaper 0.2 (dynamic types are separated from memory
ownership; zero GC; no default reference counting; when safety cannot
be proven the compiler refuses — it never silently degrades). Rules
marked PROPOSAL resolve the four open hard points from the whitepaper
evaluation and stand until the user rules otherwise; each rule names
the sla-checker module whose facts the inference must produce, so the
existing checker and the SA Referee remain the judges — psa infers,
it does not get the last word.

## 1. Values and categories

Every value has exactly one category, fixed at its binding site by
inference:

- **Scalar** — `int`, `float`, `bool`. Immutable-width machine values.
- **Text** — string values (v0.1: literals only, category Str).
- **Container** — lists, dicts, sets, tuples, and user objects.
  Not yet in the v0.1 surface; this spec fixes their semantics before
  the surface lands, so the surface cannot drift.
- **Dynamic** — tagged values of unknown static type. Phase 4;
  section 6 constrains them now.

## 2. Scalar rule (hard point 1, first half) — SETTLED by v0.1 practice

Scalars are **Copy**: assignment, argument passing, and return copy
the bits; the source binding stays usable. This matches the SLA
emission v0.1 already produces and the SA treatment of register-width
values. No annotation exists or is needed.

## 3. Container assignment (hard point 1, second half) — PROPOSAL

Assignment of a container **moves** it: after `b = a`, `a` is dead and
any later use is a loud compile error (use-after-move), mirroring
check_call_consume.rs's consumption facts at the statement level.
There is no implicit deep copy and no implicit reference count.
An explicit `copy(a)` (surface TBD) is the only duplication, and it
is shallow unless the element category is itself Copy-only.

Rationale: the ambiguity the whitepaper flagged (does `b = a` share
or copy?) is resolved by making the cheap reading (move) the only
reading; sharing must be written as a borrow (section 4), copying
must be written as a call.

## 4. Automatic borrow (the default read path)

A use of a container that only **reads** — field/index read,
iteration, passing to a parameter whose body only reads it — is
inferred as a **borrow**: no ownership transfer, the binding stays
alive. Inference is whole-function and flow-based, producing the
same borrow-origin facts as check_borrow_origin.rs /
check_field_borrow.rs. A use that **writes or consumes** (mutation,
storing into another container, returning, passing to a consuming
parameter) is a move (section 3) unless it is a place-mutation
through an existing borrow chain the checker can prove disjoint
(check_call_places.rs overlap rules).

If the analysis cannot prove a use is read-only, it must treat the
use as a move — and if a later use then fails, the diagnostic names
the first use as the move site. It must never guess "probably a
borrow" to make a program pass.

## 5. Element extraction (hard point 3) — PROPOSAL

Reading `xs[i]` or `d[k]` yields a **borrow of the element**, never
ownership: the value may be read and, for scalar elements, copied
out (scalars are Copy, so `n = xs[0]` copies the scalar — the common
case just works). A container element cannot be bound by plain
assignment from an index read; taking ownership out of a container
requires an explicit removal operation (`pop`, `take`) that the
checker models as consumption of that slot. Mutating an element in
place (`xs[i] = v`, `xs[i] += 1`) is a place write through the
container binding, not an extraction.

## 6. Function calls and the dynamic boundary (hard point 2) — PROPOSAL

- Calls to statically known functions follow sections 3–4 per
  parameter, inferred from the callee body (v0.1 already infers
  signatures monomorphically; ownership joins that inference).
- Calls through a **dynamic** callee (Phase 4) may only **borrow**
  their arguments: ownership never transfers into a call whose
  signature is not statically known, because consumption could not
  be modelled. A program that needs to hand ownership to a dynamic
  callee is refused loudly in v1.
- Returning a borrow of a local is refused (section 7).

## 7. Escaping references (hard point 4) — PROPOSAL

In v1, borrows are **scope-bound**: a borrow may not be stored into
a container, returned, or captured by a value that outlives the
borrowed binding's scope. The inference tracks borrow origins
(check_borrow_origin.rs) and refuses any flow that would let a
borrow escape. Generational handles are a later-phase design and
are not smuggled into v1.

## 8. Failure discipline

Every rule above fails **loudly and locally**: the diagnostic names
the binding, the rule, and the two conflicting sites (move site and
later use; borrow origin and escape site). A program the inference
cannot settle is rejected, never compiled under a guessed rule.
This is the whitepaper's core bargain and it outranks surface
convenience in every conflict.

## 9. Verification obligations

Each rule lands with: (a) a positive .psa example that compiles and
runs with Python-identical observable behaviour; (b) a negative
.psa example that fails with the rule's diagnostic; (c) a check
that the generated SLA passes the unmodified sla checker, i.e. the
inference's facts are the checker's facts, not a parallel system.
