import re
print(re.findall(r"\d+", "a1 b22"))
m = re.match(r"(\d+)-(\d+)", "2026-11")
print([m.group(0), m.group(1), m.group(2)])
