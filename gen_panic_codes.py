#!/usr/bin/env python3
"""Generate psa-std/PANIC-CODES.json from the master panic-code table.

Data source: crates/psa-cli/src/panic_codes.rs (the master table, 110 PanicInfo
entries). The JSON twin is the published single source of truth per
psa-std/ERROR-FORMAT-DESIGN.md section 5.1; it is generated from the same
data as the compiled-in table, never edited by hand.

History: the staging-errors original embedded its own 88-code table
(ERROR-FORMAT-DESIGN.md section 5.2) and emitted both files. That embedded
table is stale -- it predates the fstring renumber (77085/77086), the
pickle segment (77310-77319) and the bigint M10 revision of 77080's text.
The master .rs is now the only current complete table, so this master
copy derives the JSON from it (rs -> json), preserving the same-source
rule: one data set, the JSON generated, never hand-written.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RS_PATH = ROOT / "crates" / "psa-cli" / "src" / "panic_codes.rs"
JSON_PATH = ROOT / "psa-std" / "PANIC-CODES.json"

STR = r'"((?:[^"\\]|\\.)*)"'
ENTRY = re.compile(
    r"PanicInfo \{ code: (\d+), rule: " + STR +
    r", category: " + STR +
    r", cp_surface: (?:None|Some\(" + STR + r"\))" +
    r", summary: " + STR +
    r", fix: (?:None|Some\(" + STR + r"\))" +
    r", reserved: (true|false) \}"
)


def unescape(s):
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            out.append({"n": "\n", "t": "\t", "r": "\r",
                        "\\": "\\", '"': '"', "0": "\0"}[nxt])
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def parse_rs(text):
    rows = []
    for m in ENTRY.finditer(text):
        code, rule, cat, cp, summary, fix, reserved = m.groups()
        rows.append({
            "code": int(code),
            "rule": unescape(rule),
            "category": unescape(cat),
            "cp_surface": unescape(cp) if cp is not None else None,
            "summary": unescape(summary),
            "fix": unescape(fix) if fix is not None else None,
            "reserved": reserved == "true",
        })
    return rows


def main():
    text = RS_PATH.read_text(encoding="utf-8")
    rows = parse_rs(text)
    raw_entries = text.count("    PanicInfo { code:")
    if len(rows) != raw_entries:
        print(f"PARSE MISMATCH: regex matched {len(rows)} of "
              f"{raw_entries} PanicInfo entries in {RS_PATH}",
              file=sys.stderr)
        return 1
    codes = [r["code"] for r in rows]
    if len(set(codes)) != len(codes):
        print("DUPLICATE codes in master table", file=sys.stderr)
        return 1

    obj = {"version": 1, "codes": {}}
    for r in rows:
        entry = {
            "rule": r["rule"],
            "category": r["category"],
            "cp_surface": r["cp_surface"],
            "summary": r["summary"],
            "fix": r["fix"],
        }
        if r["reserved"]:
            entry["reserved"] = True
        obj["codes"][str(r["code"])] = entry

    JSON_PATH.write_text(
        json.dumps(obj, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"wrote {JSON_PATH} ({len(obj['codes'])} keys) "
          f"from {RS_PATH} ({len(rows)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
