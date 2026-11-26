#!/bin/bash
# run_e2e.sh <name> : psa emit <name>.psa -> sa build-exe -> run -> diff vs <name>.py output
set -u
export PATH=/data/nightfull-zig/tools/zig-x86_64-linux-0.17.0:$PATH
export HOME=/data/nightfull-zig/plhome
export LD_LIBRARY_PATH=/data/nightfull-zig/tools/llvm14/usr/lib/x86_64-linux-gnu
export SA_STD_DIR=/data/host-drift/host-0c4cb7be/sa_std
export SLA_STD_DIR=/data/nightfull-zig/all/sa_plugin_sla/sla_std
export SA_PLUGIN_DEV=1
export SLA_NO_FRESHNESS_CACHE=1
SA=/data/host-drift/host-0c4cb7be/bin/sa
PSA=/data/psa-v1/target/release/psa
N=$1
W=/data/psa-batch/e2e/$N
mkdir -p $W; cd $W
cp /data/psa-batch/e2e/$N.psa .
cp /data/psa-batch/e2e/$N.py .
rm -rf .sla-cache .sa_cache ~/.cache/sa/frontend
if ! $PSA emit $N.psa > $N.sla 2> emit.err; then echo "E2E-$N EMIT-FAIL"; cat emit.err; exit 1; fi
if ! $SA build-exe $N.sla -o $N.bin > build.log 2>&1 || [ ! -x $N.bin ]; then echo "E2E-$N BUILD-FAIL"; head -5 build.log; exit 1; fi
./$N.bin > got.out 2> got.err; rc=$?
if [ $rc -ne 0 ]; then echo "E2E-$N RUN-FAIL rc=$rc"; cat run.err; exit 1; fi
python3 $N.py > expected.out 2> expected.err
if diff -u expected.out got.out > diff.out && diff -u expected.err got.err > diff.err; then echo "E2E-$N PASS"; else echo "E2E-$N FAIL"; cat diff.out diff.err; fi
