#!/usr/bin/env python3
"""Registry coverage audit: run every known negative .psa in the
repo through `psa --error-format=json emit` and report which code
each one lands on. Anything on a fallback code (PSA-L900 / P900 /
E9000 / E6009-unclassified) is listed separately so unregistered
common messages are visible instead of silent.

Usage: PSA=/path/to/psa python3 audit_registry.py <file.psa ...>
"""
import json
import subprocess
import sys
import os

FALLBACK = {"PSA-L900", "PSA-P900", "PSA-E9000"}


def main() -> int:
    psa = os.environ.get("PSA")
    if not psa:
        print("set PSA to the psa binary path", file=sys.stderr)
        return 2
    fallback_hits = []
    rows = []
    for path in sys.argv[1:]:
        p = subprocess.run(
            [psa, "--error-format=json", "emit", path],
            capture_output=True, text=True)
        code = None
        for line in p.stderr.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    code = json.loads(line).get("code")
                except Exception:
                    pass
                break
        rows.append((path, p.returncode, code))
        if code in FALLBACK:
            fallback_hits.append((path, code))
    for path, rc, code in rows:
        print(f"{path}\trc={rc}\t{code}")
    print("----")
    print(f"{len(rows)} negatives, {len(fallback_hits)} on fallback codes")
    for path, code in fallback_hits:
        print(f"FALLBACK {code} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
