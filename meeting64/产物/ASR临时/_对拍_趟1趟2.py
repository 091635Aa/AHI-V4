# -*- coding: utf-8 -*-
"""meeting64/产物/ASR临时/_对拍_趟1趟2.py —— C13 跨进程确定性对照（两趟基线 JSON 逐字段比）。

比：多句CER均值 / 逐样本 CER / 转写 / 波形 sha256 / 两趟一致 / sham。
不比：wav 文件名（每次唯一是**要求**）、run_id、临时目录、耗时、产物路径。

跑：
  & "<workspace>/\03_AHI\\.venv\\Scripts\\python.exe" "<workspace>/\03_AHI\\AHI4\\meeting64\\产物\\ASR临时\\_对拍_趟1趟2.py"
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
a = os.path.join(HERE, "_基线_趟1.json")
b = os.path.join(HERE, "_基线_趟2.json")


def 载(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    A, B = 载(a), 载(b)
    差 = []

    def 比(键, x, y):
        if x != y:
            差.append((键, x, y))

    比("多句CER均值", A["多句CER均值"], B["多句CER均值"])
    比("两趟一致", A["两趟一致"], B["两趟一致"])
    比("sham", A["sham"]["白噪"], B["sham"]["白噪"])
    比("sham静", A["sham"]["静音"], B["sham"]["静音"])
    比("基线", A["基线CER均值"], B["基线CER均值"])
    逐样本同 = 0
    for k in A["逐样本"]:
        x, y = A["逐样本"][k], B["逐样本"][k]
        比("%s.CER" % k, x["CER"], y["CER"])
        比("%s.转写" % k, x["转写"], y["转写"])
        比("%s.转写第二趟" % k, x["转写第二趟"], y["转写第二趟"])
        比("%s.sha256" % k, x["sha256"], y["sha256"])
        比("%s.对齐" % k, x["对齐"], y["对齐"])
        if x["sha256"] == y["sha256"] and x["CER"] == y["CER"] and x["转写"] == y["转写"]:
            逐样本同 += 1
    print("=" * 84)
    print("趟1 多句CER均值 = %s ｜ 趟2 = %s" % (A["多句CER均值"], B["多句CER均值"]))
    print("逐样本（CER/转写/对齐/波形sha256）全同 = %d/%d" % (逐样本同, len(A["逐样本"])))
    print("波形 sha256 全同 =", all(A["逐样本"][k]["sha256"] == B["逐样本"][k]["sha256"]
                              for k in A["逐样本"]))
    if 差:
        print("【差异】%d 项：" % len(差))
        for k, x, y in 差[:20]:
            print("  %-24s 趟1=%r 趟2=%r" % (k, x, y))
    else:
        print("【差异】0 项 —— 两趟逐字段同（C13 ✓）")
    通过 = (not 差) and 逐样本同 == len(A["逐样本"])
    print("C13 跨进程复现 =", 通过)
    raise SystemExit(0 if 通过 else 1)
