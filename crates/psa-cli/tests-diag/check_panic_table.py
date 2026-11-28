#!/usr/bin/env python3
"""Panic-code table discipline check (ERROR-FORMAT-DESIGN 5.1):
every panic code emitted anywhere in the sources must be a key of
PANIC-CODES.json, and every non-reserved key in the table must be
emitted somewhere (reserved keys are explicit placeholders).

Sources scanned:
  - psa-std/*.sla            `panic NNNNN`
  - crates/psa-cli/src/*.rs  `panic(NNNNN)` emission templates

Usage: python3 check_panic_table.py <repo-root>
Exit 0 iff both directions hold.
"""
import json
import re
import sys
from pathlib import Path


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    table = json.loads((root / "psa-std" / "PANIC-CODES.json").read_text())
    codes = table["codes"]

    emitted = set()
    for sla in (root / "psa-std").glob("*.sla"):
        # both SLA idioms occur: `panic 77134` and `panic(77130);`
        text = "\n".join(
            line.split("//", 1)[0] for line in sla.read_text().splitlines())
        emitted |= set(re.findall(r"\bpanic\s*\(?\s*(\d{5})\s*\)?", text))
    src = root / "crates" / "psa-cli" / "src"
    for rs in src.glob("*.rs"):
        if rs.name == "panic_codes.rs":
            continue  # the generated table itself
        emitted |= set(re.findall(r"panic\((\d{5})\)", rs.read_text()))

    missing = sorted(emitted - set(codes))
    unemitted = sorted(
        k for k, v in codes.items()
        if k not in emitted and not v.get("reserved"))
    print(f"emitted in sources: {len(emitted)}")
    print(f"table keys: {len(codes)} "
          f"(reserved: {sum(1 for v in codes.values() if v.get('reserved'))})")
    ok = True
    if missing:
        ok = False
        print("EMITTED BUT NOT IN TABLE:", " ".join(missing))
    if unemitted:
        ok = False
        print("IN TABLE, NOT RESERVED, NEVER EMITTED:", " ".join(unemitted))
    print("PANIC TABLE CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
