#!/bin/bash
# psa_std json harness: emit the probe with psa (runtime +
# __py_test_* fns), cut psa's own main, splice json.sla +
# the hand-written harness main, build on the text route,
# compare against CPython's json module on the same values.
set -u
export PATH=/tools/zig:$PATH
export SA_STD_DIR=/content/sa_all/sci/sa_std
export SLA_STD_DIR=/content/sa_all/sapy/sla_std
export SA_PLUGIN_DEV=1
export SLA_NO_FRESHNESS_CACHE=1
SA=/content/sa_all/sci/zig-out/bin/sa
PSA=/content/sa_all/sapy/target/release/psa
W=/content/sapy-batch/jsonh
mkdir -p $W
cd $W
cp /content/sapy-batch/psa-std/json.sla .
cp /content/sapy-batch/psa-std/probe_json.psa .
cp /content/sapy-batch/psa-std/harness_main.sla .

if ! $PSA emit probe_json.psa > probe_full.sla 2> emit.err; then
    echo "EMIT-FAIL"; cat emit.err; exit 1
fi
# Cut psa's main: it is the last fn in the emitted file.
awk '/^fn main\(\) -> i32 \{/{exit} {print}' probe_full.sla > probe_nomain.sla
cat probe_nomain.sla json.sla harness_main.sla > harness.sla
rm -rf .sla-cache .sa_cache
if ! $SA build-exe harness.sla -o harness.bin > build.log 2>&1 || [ ! -x harness.bin ]; then
    echo "BUILD-FAIL"; head -5 build.log; exit 1
fi
./harness.bin > got.out 2> run.err
rc=$?
if [ $rc -ne 0 ]; then echo "RUN-FAIL rc=$rc"; cat run.err; fi
python3 - << 'PYEOF' > expected.out
import json
v = {"a": [1, 2.0, True, None, "x\ny"], "b": {"c": []}, "d": {}}
s = json.dumps(v)
print(s)
print(json.dumps(json.loads(s)))
t = '{"k": "A\\nB", "n": -12.0, "u": "\\u0041"}'
print(json.dumps(json.loads(t)))
PYEOF
if diff -u expected.out got.out; then
    echo "JSON-HARNESS PASS"
else
    echo "JSON-HARNESS FAIL"
fi
