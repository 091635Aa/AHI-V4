# -*- coding: utf-8 -*-
"""meeting53/实施/diff_两趟.py —— 跨进程两趟逐键比对（忽略墙钟/内存）。"""
import json
import sys

A = json.load(open(sys.argv[1], encoding="utf-8"))
B = json.load(open(sys.argv[2], encoding="utf-8"))
for d in (A, B):
    d.get("备注", {}).pop("墙钟秒", None)
    d.get("备注", {}).pop("峰值内存MB", None)
    d.get("备注", {}).pop("替身登记", None)


def 平(x, 前=""):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out.update(平(v, 前 + "/" + str(k)))
        return out
    if isinstance(x, list):
        out = {}
        for i, v in enumerate(x):
            out.update(平(v, 前 + "[%d]" % i))
        return out
    return {前: x}


a, b = 平(A), 平(B)
差 = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
print("差异键数=%d" % len(差))
for k in 差[:40]:
    print("  %s: A=%r B=%r" % (k, a.get(k), b.get(k)))
print("跨进程两趟逐键同=", not 差)
sys.exit(0 if not 差 else 1)
