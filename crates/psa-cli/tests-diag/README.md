# tests-diag — structured-diagnostics contrast tests

Batch-1 acceptance suite for the psa clear-error-feedback goal
(design: psa-std/ERROR-FORMAT-DESIGN.md). It pins the behaviour of
the diagnostic registry (diag_registry.rs), the `--error-format`
renderings (diag.rs) and the panic-code table (panic_codes.rs /
psa-std/PANIC-CODES.json) so refactors cannot silently renumber,
drop or reword a diagnostic.

## Run

```sh
cargo build --release -p psa-cli
PSA=$PWD/target/release/psa crates/psa-cli/tests-diag/run_tests.sh
python3 crates/psa-cli/tests-diag/check_panic_table.py .
PSA=$PWD/target/release/psa python3 crates/psa-cli/tests-diag/audit_registry.py examples/reject_*.psa
```

## What is covered

- `cases.txt`: 9 compile-time codes (PSA-E1001, E1002, L012,
  P001, E3018, E3016, E3017, E2053, E1021). Each has a negative
  case in `cases/neg_*.psa` that must be rejected at emit level
  with exactly that code in text AND JSON form (JSON checked
  against the design §4.2 schema: code / rule / kind / non-empty
  sites with line+role / message), and a legal twin
  `cases/ok_*.psa` that must emit cleanly (rc=0, empty stderr)
  in both formats.
- Host level: a missing input file must yield PSA-H002 in both
  formats (JSON sites may be empty — see KNOWN-GAPS.md).
- `explain_cases.txt`: `psa explain` for panic codes 77069,
  77040, 77100, 77083 and compile code PSA-E1001 — text form must
  carry the summary, JSON form the frozen rule name and code;
  unknown code 99999 must fail loudly.
- `check_panic_table.py`: the design §5.1 discipline — every
  panic code emitted in psa-std/*.sla or the psa-cli sources is
  a PANIC-CODES.json key, and every non-reserved key is emitted
  somewhere.
- `audit_registry.py`: coverage report over arbitrary negative
  corpora; flags any diagnostic landing on a fallback code
  (PSA-L900 / P900 / E9000).

Known gaps the suite deliberately does NOT pin yet are recorded
in KNOWN-GAPS.md (lambda capture / downstream-only refusals,
sites wording tension).
