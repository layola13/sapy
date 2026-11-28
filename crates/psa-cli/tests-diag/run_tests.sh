#!/bin/bash
# psa structured-diagnostics contrast tests (batch 1).
#
# Every registered compile-time code in cases.txt has:
#   - one negative .psa that must be rejected at the emit (check)
#     level with exactly that code, in BOTH text and JSON formats;
#   - one legal twin .psa that must emit cleanly (rc=0, no
#     diagnostic on stderr in either format).
# Plus host-level and `psa explain` checks (panic-code table).
#
# Usage: PSA=/path/to/psa ./run_tests.sh
# Exit code 0 iff every check passes.
set -u
cd "$(dirname "$0")"
PSA="${PSA:?set PSA to the psa binary path}"
FAIL=0
PASSN=0

note_fail() { echo "FAIL: $1"; FAIL=1; }
note_pass() { PASSN=$((PASSN+1)); echo "pass: $1"; }

# check_json_diag <stderr-file> <expected-code> <expected-kind> [min-sites]
check_json_diag() {
  python3 - "$1" "$2" "$3" "${4:-1}" <<'PYEOF'
import json, sys
path, code, kind = sys.argv[1], sys.argv[2], sys.argv[3]
min_sites = int(sys.argv[4])
lines = [l for l in open(path).read().splitlines() if l.strip()]
if not lines:
    print("no JSON diagnostics on stderr"); sys.exit(1)
try:
    objs = [json.loads(l) for l in lines]
except Exception as e:
    print("stderr is not NDJSON:", e); sys.exit(1)
d = objs[0]
for field in ("code", "rule", "kind", "sites", "message"):
    if field not in d:
        print("missing field", field); sys.exit(1)
if d["code"] != code:
    print("code", d["code"], "!=", code); sys.exit(1)
if d["kind"] != kind:
    print("kind", d["kind"], "!=", kind); sys.exit(1)
if len(d["sites"]) < min_sites:
    print("sites below minimum", min_sites); sys.exit(1)
for s in d["sites"]:
    if "line" not in s or "role" not in s:
        print("site missing line/role:", s); sys.exit(1)
sys.exit(0)
PYEOF
}

while read -r _tag name code kind; do
  [ -z "${_tag:-}" ] && continue
  case "$_tag" in \#*) continue;; esac
  neg="cases/neg_${name}.psa"
  ok="cases/ok_${name}.psa"
  # negative, text format
  if "$PSA" emit "$neg" >/dev/null 2>/tmp/diag_t.err; then
    note_fail "$name: negative accepted in text mode"
  elif ! head -1 /tmp/diag_t.err | grep -qF "error[$code]:"; then
    note_fail "$name: text first line lacks error[$code]: $(head -1 /tmp/diag_t.err)"
  else
    note_pass "$name text"
  fi
  # negative, JSON format
  if "$PSA" --error-format=json emit "$neg" >/dev/null 2>/tmp/diag_j.err; then
    note_fail "$name: negative accepted in JSON mode"
  elif ! check_json_diag /tmp/diag_j.err "$code" "$kind"; then
    note_fail "$name: JSON diagnostic mismatch ($(cat /tmp/diag_j.err | head -c 200))"
  else
    note_pass "$name json"
  fi
  # legal twin, both formats
  if ! "$PSA" emit "$ok" >/dev/null 2>/tmp/diag_o.err; then
    note_fail "$name: legal twin rejected in text mode: $(head -1 /tmp/diag_o.err)"
  elif [ -s /tmp/diag_o.err ]; then
    note_fail "$name: legal twin wrote to stderr in text mode"
  elif ! "$PSA" --error-format=json emit "$ok" >/dev/null 2>/tmp/diag_o2.err; then
    note_fail "$name: legal twin rejected in JSON mode"
  elif [ -s /tmp/diag_o2.err ]; then
    note_fail "$name: legal twin wrote to stderr in JSON mode"
  else
    note_pass "$name legal-twin"
  fi
done < cases.txt

# host-level negative: missing input file -> PSA-H002, both formats
if "$PSA" emit cases/__no_such_file__.psa >/dev/null 2>/tmp/diag_h.err; then
  note_fail "host-missing-file: accepted"
elif ! head -1 /tmp/diag_h.err | grep -qF "error[PSA-H002]:"; then
  note_fail "host-missing-file: text lacks PSA-H002: $(head -1 /tmp/diag_h.err)"
elif "$PSA" --error-format=json emit cases/__no_such_file__.psa >/dev/null 2>/tmp/diag_hj.err; then
  note_fail "host-missing-file: accepted in JSON mode"
elif ! check_json_diag /tmp/diag_hj.err "PSA-H002" "host" 0; then
  note_fail "host-missing-file: JSON mismatch"
else
  note_pass "host-missing-file"
fi

# explain: panic-code table lookups (text + JSON) and loud failure.
# explain_cases.txt rows are pipe-separated:
#   exp|<arg>|<rule expected in JSON>|<summary expected in text>
while IFS='|' read -r _tag arg rule summary; do
  [ -z "${_tag:-}" ] && continue
  case "$_tag" in \#*) continue;; esac
  if [ "$rule" = "LOUDFAIL" ]; then
    if "$PSA" explain "$arg" >/dev/null 2>&1; then
      note_fail "explain $arg: expected loud failure, got rc=0"
    else
      note_pass "explain $arg loud-fail"
    fi
    continue
  fi
  if ! "$PSA" explain "$arg" 2>/dev/null | grep -qF "$summary"; then
    note_fail "explain $arg: text output lacks summary '$summary'"
  elif ! "$PSA" --error-format=json explain "$arg" 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
assert d['rule']=='$rule', d
want_code = 'PSA-R$arg' if '$arg'.isdigit() else '$arg'
assert d['code']==want_code, d
"; then
    note_fail "explain $arg: JSON form mismatch"
  else
    note_pass "explain $arg"
  fi
done < explain_cases.txt

echo "----"
if [ "$FAIL" = 0 ]; then
  echo "ALL PASS ($PASSN checks)"
else
  echo "FAILURES PRESENT ($PASSN checks passed)"
fi
exit $FAIL
