# Known gaps found by the batch-1 diagnostics suite

## lambda capture has no psa-level diagnostic (registry PSA-E5002 unreachable here)

`cases/neg_lambda_capture.psa` (kept as the repro; deliberately NOT
in cases.txt):

```python
x = 5
f = lambda y: x + y
print(f(1))
```

- `psa emit` accepts it (rc=0) — the frontend produces no
  diagnostic, although the registry carries PSA-E5002
  `lambda-capture` and `examples/reject_d3_lambda_capture.psa`
  lists the shape as a loud refusal.
- `psa check` refuses it only downstream: the generated SLA fails
  to parse in the host pipeline (`error.SyntaxError` at the
  generated file), relayed as PSA-H003 with mapping `unmapped`.
  No PSA-E code, no site in the user's source.

So the refusal exists but is invisible to the structured
diagnostics: an agent consuming `--error-format=json` learns
nothing actionable. Suggested batch-2 fix: detect a lambda body
referencing a non-parameter, non-global name at emit time and
raise the already-registered PSA-E5002 with the lambda's line as
site. Until then this case is judged at host level only and is
excluded from the emit-level contrast suite.

## The same class covers 5 of the 13 repo `reject_*` examples

`audit_registry.py` over `examples/reject_*.psa` +
`crates/psa-cli/reject_list.psa` (2026-11-28, head f27db88 +
batch-1): 8 are refused at emit with real codes (P054, E3010,
E3017, E2053, E3016, E2032, E5069, E1001) and none lands on a
fallback code — but 5 are ACCEPTED by `psa emit` (rc=0, no
diagnostic): reject_d3_lambda_capture, reject_dyn_multiprint,
reject_fstr_spec, reject_fstr_value, reject_list. Their refusal
happens (if at all) in the host pipeline, without a PSA code.
Batch 2 should audit each of the five: either the frontend grows
the check (as for E5002 above) or the example's "reject" status
needs re-judging at the level that actually refuses it.

## Design-doc wording tension: JSON `sites` for position-less diagnostics

ERROR-FORMAT-DESIGN.md §4.2 marks `sites` as required with "at
least the main site", while its own §4.3 runtime example shows
`"sites":[]`, and host diagnostics such as PSA-H002 (input file
not found) have no source position and render with empty sites.
The suite follows §4.3 (empty sites allowed for host/runtime
renderings). The design doc should be reconciled in a later batch.
