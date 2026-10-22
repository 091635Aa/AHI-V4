# -*- coding: utf-8 -*-
"""meeting53/实施/修汇总.py —— 只从臂清单重算汇总计数（修 R3 M1 计数 bug）。"""
import json
import sys


def 修(path):
    with open(path, encoding="utf-8") as f:
        果 = json.load(f)
    臂 = 果.get("臂", {})
    汇 = 果.setdefault("汇总", {})
    汇["臂数"] = len(臂)
    汇["PASS"] = sum(1 for v in 臂.values() if str(v["四态"]).startswith("PASS"))
    汇["FAIL"] = sum(1 for v in 臂.values() if str(v["四态"]).startswith("FAIL"))
    汇["UNDECIDABLE"] = sum(1 for v in 臂.values() if str(v["四态"]).startswith("UNDECIDABLE"))
    汇["INVALID"] = sum(1 for v in 臂.values() if str(v["四态"]).startswith("INVALID"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(果, f, ensure_ascii=False, indent=1, default=str)
    print(path, "→", {k: 汇[k] for k in ("臂数", "PASS", "FAIL", "UNDECIDABLE", "INVALID")})


if __name__ == "__main__":
    for p in sys.argv[1:]:
        修(p)
