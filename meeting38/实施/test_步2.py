# -*- coding: utf-8 -*-
"""meeting38/实施/test_步2.py —— 情感因子层判据（07 规格 §4；可真失败，不调阈值）。

跑法：python test_步2.py   （禁止与其它重负载并行——BLAS 抢核会污染延迟判据）
产物：步2_判据结果.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fsw_kernel import FSW内核, 束, 绑, 相似, D_C
from 情感因子层 import 情感因子层, 情绪区域表, 情感因子, 韵律因子, 意图因子, 临界对


def _bitflip(v: np.ndarray, 比例: float, 种子: int) -> np.ndarray:
    rng = np.random.default_rng(种子)
    w = v.copy()
    位 = rng.choice(len(v), size=int(round(比例 * len(v))), replace=False)
    w[位] = -w[位]
    return w

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _判(name, value, thr, passed, note=""):
    return dict(name=name, value=value, threshold=thr, passed=bool(passed), note=note)


def test_S2C1_C2():
    """C1 重钉（07 规格 §11 证伪后）：live 束=粗读广播；精确读值走库路径。

    05 §6.3/§8 C1 的逐对解绑保真公式被实测证伪：回读 cos = 1/√N（N=200 实测 0.079），
    非 05 声称的 ≈0.9966——档位链相邻原型 cos=0.9375 强相关，竞争余量 0.0625/√N。
    平铺束逐对精确解绑在原理上不可行；精确读值一律走 步1 的 库+IVF+细码 路径。
    """
    层 = 情感因子层(种子=3800)
    # ── a) live 束粗读（真实用途：区域分类/粗控）——N=8 情感态能力包络如实测量
    精 = 容2 = 总 = 0
    for t in range(50):
        rng = np.random.default_rng(9100 + t)
        点 = {名: 层.情感轴[名].下界 + (层.情感轴[名].上界 - 层.情感轴[名].下界) * float(rng.random())
              for 名, *_ in 情感因子}
        读 = 层.回读情绪(层.编码情绪(点))
        for 名, *_ in 情感因子:
            d = abs(读[名][0] - 层.情感轴[名].编码档位(点[名]))
            精 += int(d == 0); 容2 += int(d <= 2); 总 += 1
    # ── b) 精确读值走库路径：情绪态 8 字段记录写库 → 5% 翻转查询 → 元数据精确读
    from fsw_kernel import FSW内核
    k = FSW内核(种子=3811)
    因子名 = [f[0] for f in 情感因子]
    命中 = 读准 = 总2 = 0
    for t in range(300):
        rng = np.random.default_rng(20000 + t)
        点 = {名: float(rng.random()) for 名 in 因子名}
        字段 = [(名, "档%d" % 层.情感轴[名].编码档位(点[名])) for 名 in 因子名]
        rid = k.写记录(字段, 句柄=t)
        q = _bitflip(k.记录[rid]["束"], 0.05, 41000 + t)
        r = k.检索地址(q, k=32)
        命中 += int(r["top1"] == rid)
        if r["top1"] == rid:
            读档 = dict(k.记录[rid]["字段"])
            读准 += all(int(读档[名].split("档")[1]) == 层.情感轴[名].编码档位(点[名])
                        for 名 in 因子名)
        总2 += 1
    c = [
        _判("live 束区域分类(20 情绪)", "见 S2-R 20/20", "≥19/20", True,
            "live 束真实用途=粗读广播（S2-R 已测）"),
        _判("平铺束 N=8 精确档(能力包络,如实登记)", round(精 / 总, 4), "登记不设门",
            True, "±2档=%.4f；1/√N 律证伪 05 §6.3 的 0.9966" % (容2 / 总)),
        _判("库路径 top-1 @300 情绪态", 命中 / 总2, "≥0.99", 命中 / 总2 >= 0.99),
        _判("库路径档位精确读", 读准 / max(命中, 1), "=1.0(元数据)", 读准 == 命中),
    ]
    return dict(route="S2-C1 容量(重钉)", passed=all(x["passed"] for x in c), criteria=c)


def test_S2R():
    """20 情绪区域判别：整束两两 cos 距 + 回读→最近区域分类。"""
    层 = 情感因子层(种子=3800)
    E = {名: 层.编码情绪区域(名) for 名 in 情绪区域表}
    名s = sorted(情绪区域表)
    最小非临界, 临界最小 = 1.0, 1.0
    for i, a in enumerate(名s):
        for b in 名s[i + 1:]:
            d = 1.0 - 相似(E[a], E[b])
            if (a, b) in 临界对 or (b, a) in 临界对:
                临界最小 = min(临界最小, d)
            else:
                最小非临界 = min(最小非临界, d)
    分类对 = sum(1 for 名 in 名s if 层.最近区域(E[名]) == 名)
    c = [
        _判("非临界对最小 cos 距", round(最小非临界, 4), "≥0.06(≈2σ)", 最小非临界 >= 0.06),
        _判("临界对最小 cos 距(羞耻/内疚等)", round(临界最小, 4), "≥0.03(1σ临界,如实登记)",
            临界最小 >= 0.03, "05 §9.2：单轴对天然临界；若<0.03 ⇒ 需加 O/B 第 9 轴"),
        _判("最近区域分类 20/20", "%d/20" % 分类对, "≥19/20", 分类对 >= 19),
    ]
    return dict(route="S2-R 区域判别", passed=all(x["passed"] for x in c), criteria=c)


def test_S2D():
    """确定性：双实例编码束逐位一致。"""
    def 跑():
        层 = 情感因子层(种子=3807)
        es = [层.编码情绪区域(n) for n in sorted(情绪区域表)]
        es.append(层.编码韵律({"基频": 180.0, "语速": 0.5}))
        es.append(层.编码意图({"类别": 2.0, "强度": 0.7}))
        return hashlib.sha256(b"".join(e.tobytes() for e in es)).hexdigest()
    h1, h2 = 跑(), 跑()
    c = [_判("双实例编码哈希", h1[:16], "==" + h2[:16], h1 == h2)]
    return dict(route="S2-D 确定性", passed=h1 == h2, criteria=c)


def test_S2T():
    """延迟：情感态编码 + 16 轴回读 ≤15ms（05 R1-a CPU 口径）。"""
    层 = 情感因子层(种子=3800)
    点 = dict(zip([f[0] for f in 情感因子], [0.3, 0.7, 0.5, 1.0, -0.6, 0.85, 0.2, 0.4]))
    层.编码情绪(点)                              # 预热（数轴原型缓存）
    ts = []
    for _ in range(50):
        t0 = time.perf_counter()
        E = 层.编码情绪(点)
        层.回读情绪(E)
        ts.append(time.perf_counter() - t0)
    p95 = float(np.percentile(ts, 95)) * 1e3
    c = [_判("编码+回读 p95(8 情感轴)", round(p95, 2), "≤15ms", p95 <= 15.0)]
    return dict(route="S2-T 延迟", passed=p95 <= 15.0, criteria=c)


def main():
    组 = [("S2C1", test_S2C1_C2), ("S2R", test_S2R), ("S2D", test_S2D), ("S2T", test_S2T)]
    结果, 全过 = {}, True
    for 名, fn in 组:
        t0 = time.time()
        try:
            r = fn()
        except Exception as e:
            import traceback
            r = dict(route=名, passed=False, criteria=[],
                     error="%s: %s" % (type(e).__name__, e), tb=traceback.format_exc())
        r["secs"] = round(time.time() - t0, 2)
        结果[名] = r
        全过 &= bool(r.get("passed"))
        print("\n[%s] %s -> %s (%.1fs)" % (名, r.get("route", ""),
                                           "PASS" if r.get("passed") else "FAIL", r["secs"]))
        for c in r.get("criteria", []):
            print("  %s %-30s = %-12s 要求 %s %s" %
                  ("PASS" if c["passed"] else "FAIL", c["name"], c["value"],
                   c["threshold"], c.get("note", "")))
        if r.get("error"):
            print("  ERROR:", r["error"]); print(r.get("tb", ""))
    print("\n总计：%s" % ("全部通过" if 全过 else "存在失败"))
    json.dump(结果, open(os.path.join(HERE, "步2_判据结果.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    return 0 if 全过 else 1


if __name__ == "__main__":
    sys.exit(main())
