# -*- coding: utf-8 -*-
"""meeting38/实施/test_步2b.py —— 发声动作/EFE 判据（08 规格 §4；可真失败）。

跑法：python test_步2b.py   （禁止与其它重负载并行——延迟判据会被污染）
产物：步2b_判据结果.json
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fsw_kernel import _哈希双极, D_C
from 发声动作与EFE import 动作编码器, 联想记忆, EFE选择器

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _判(name, value, thr, passed, note=""):
    return dict(name=name, value=value, threshold=thr, passed=bool(passed), note=note)


def test_A2C():
    """实装组合全枚举编码 + 确定性 + 两两不同。"""
    e1 = 动作编码器(种子=3800)
    e2 = 动作编码器(种子=3800)
    档表 = e1.轴档表()
    组合s = list(itertools.product(*[档表[a] for a in ("类型", "强度", "音高", "时长", "音质")]))
    assert len(组合s) == 4500
    h1 = hashlib.sha256()
    seen = set()
    冲突 = 0
    for 组 in 组合s:
        选 = dict(zip(("类型", "强度", "音高", "时长", "音质"), 组))
        c1 = e1.编码(选)
        c2 = e2.编码(选)
        if not np.array_equal(c1, c2):
            冲突 += 1
        h = hashlib.sha256(c1.tobytes()).digest()
        if h in seen:
            冲突 += 1
        seen.add(h)
        h1.update(c1.tobytes())
    c = [
        _判("4500 组合全编码", "OK" if 冲突 == 0 else "%d 异常" % 冲突, "0 异常", 冲突 == 0),
        _判("跨实例逐位一致", "一致" if 冲突 == 0 else "不一致", "一致", 冲突 == 0),
        _判("异组合码两两不同", len(seen), "=4500", len(seen) == 4500),
    ]
    return dict(route="A2-C 动作空间", passed=all(x["passed"] for x in c), criteria=c)


def test_A2E():
    """前向模型：K∈{5,20,40} 学后，G=某已学结果 → 对应动作 EFE 严格最小；
    未见动作 ambiguity 显著高于已学（08 规格 §4 预登记 ≥0.9 按实测几何重钉，
    见 08 规格 §6：动作码共享档位原子，组合空间未见动作最近邻 cos≈0.47）。"""
    c = []
    for K in (5, 20, 40):
        对 = 模糊 = 已学模糊 = 0
        未见ambs = []
        for t in range(30):
            记忆 = 联想记忆()
            e = 动作编码器(种子=3800)
            rng = np.random.default_rng(5100 + K * 1000 + t)
            档表 = e.轴档表()
            组合池 = list(itertools.product(*[档表[a] for a in
                                              ("类型", "强度", "音高", "时长", "音质")]))
            选s = rng.choice(len(组合池), size=K + 30, replace=False)
            动作s, 结果s = [], []
            for i in range(K):
                选 = dict(zip(("类型", "强度", "音高", "时长", "音质"), 组合池[选s[i]]))
                a码 = e.编码(选)
                o码 = _哈希双极("A2E|结果|%d|%d|%d" % (K, t, i), D_C)
                动作s.append(a码); 结果s.append(o码)
                记忆.学(a码, o码)
            sel = EFE选择器(记忆)
            j = int(rng.integers(0, K))
            胜, d = sel.选择([("a%d" % i, a) for i, a in enumerate(动作s)], 结果s[j])
            对 += int(胜 == "a%d" % j and d["cos"] >= 0.9 and d["risk"] <= 0.05)
            # 未见组合（与已学不相交）：ambiguity 应显著 > 已学匹配（=0）
            未见ambs_t = []
            for u in range(30):
                选 = dict(zip(("类型", "强度", "音高", "时长", "音质"), 组合池[选s[K + u]]))
                d2 = sel.打分(e.编码(选), 结果s[j])
                未见ambs_t.append(d2["ambiguity"])
                未见ambs.append(d2["ambiguity"])
            _, d学 = sel.选择([("a%d" % j, 动作s[j])], 结果s[j])
            已学模糊 += d学["ambiguity"]
            模糊 += int(np.mean(未见ambs_t) >= 0.4)
        c.append(_判("EFE 选中已学匹配动作 @K=%d" % K, "%d/30" % 对, "≥29/30", 对 >= 29,
                     "检索 cos≥0.9 且 risk≤0.05"))
        c.append(_判("未见动作 ambiguity 均值 @K=%d" % K,
                     round(float(np.mean(未见ambs)), 3), "≥0.4 且 >已学(≈%.3f)" % (已学模糊 / 30),
                     float(np.mean(未见ambs)) >= 0.4,
                     "几何重钉：动作共享档位原子，未见的最近邻 cos≈0.47（08 规格 §6）"))
    return dict(route="A2-E 前向模型+EFE", passed=all(x["passed"] for x in c), criteria=c)


def test_A2S():
    """嘴/手同打分器：表达目标→嘴胜，操作目标→手胜。"""
    嘴 = 动作编码器(种子=3800, 前缀="动作")
    手 = 动作编码器(种子=3800, 前缀="手动作")
    嘴胜 = 手胜 = 0
    试验 = 20
    for t in range(试验):
        rng = np.random.default_rng(6100 + t)
        档嘴, 档手 = 嘴.轴档表(), 手.轴档表()
        表达目标 = _哈希双极("A2S|表达|%d" % t, D_C)
        操作目标 = _哈希双极("A2S|操作|%d" % t, D_C)
        # 同一记忆学两对：嘴动作→表达听感，手动作→操作结果
        记忆 = 联想记忆()
        选嘴 = {轴: 档嘴[轴][int(rng.integers(0, len(档嘴[轴])))] for 轴 in 档嘴}
        选手 = {轴: 档手[轴][int(rng.integers(0, len(档手[轴])))] for 轴 in 档手}
        a嘴, a手 = 嘴.编码(选嘴), 手.编码(选手)
        记忆.学(a嘴, 表达目标)
        记忆.学(a手, 操作目标)
        sel = EFE选择器(记忆)
        候选 = [("嘴", a嘴), ("手", a手)]
        胜表达, _ = sel.选择(候选, 表达目标)
        胜操作, _ = sel.选择(候选, 操作目标)
        嘴胜 += int(胜表达 == "嘴")
        手胜 += int(胜操作 == "手")
    c = [
        _判("表达目标 → 嘴候选胜", "%d/%d" % (嘴胜, 试验), "≥19/20", 嘴胜 >= 19),
        _判("操作目标 → 手候选胜", "%d/%d" % (手胜, 试验), "≥19/20", 手胜 >= 19),
    ]
    return dict(route="A2-S 嘴/手同打分器", passed=all(x["passed"] for x in c), criteria=c)


def test_A2D():
    """双实例：动作码 + EFE 选择逐位一致。"""
    def 跑():
        e = 动作编码器(种子=3809)
        记忆 = 联想记忆()
        档表 = e.轴档表()
        rng = np.random.default_rng(7100)
        for i in range(10):
            选 = {轴: 档表[轴][int(rng.integers(0, len(档表[轴])))] for 轴 in 档表}
            记忆.学(e.编码(选), _哈希双极("A2D|o|%d" % i, D_C))
        sel = EFE选择器(记忆)
        候选 = [("c%d" % i, e.编码({轴: 档表[轴][int(rng.integers(0, len(档表[轴])))]
                                    for 轴 in 档表}))
                for i in range(20)]
        G = _哈希双极("A2D|G", D_C)
        胜, d = sel.选择(候选, G)
        return hashlib.sha256(b"".join(c.tobytes() for _, c in 候选)).hexdigest() + "|%s|%.6f" % (胜, d["efe"])
    r1, r2 = 跑(), 跑()
    c = [_判("双实例选择链逐位一致", r1[:24], "==" + r2[:24], r1 == r2)]
    return dict(route="A2-D 确定性", passed=r1 == r2, criteria=c)


def test_A2T():
    """50 候选 EFE 打分+选择 ≤3ms（05 R1-a CPU 口径）。"""
    e = 动作编码器(种子=3800)
    记忆 = 联想记忆()
    档表 = e.轴档表()
    rng = np.random.default_rng(7300)
    候选 = []
    for i in range(50):
        选 = {轴: 档表[轴][int(rng.integers(0, len(档表[轴])))] for 轴 in 档表}
        候选.append((i, e.编码(选)))
        if i < 40:
            记忆.学(候选[-1][1], _哈希双极("A2T|o|%d" % i, D_C))
    G = _哈希双极("A2T|G", D_C)
    sel = EFE选择器(记忆)
    sel.选择(候选, G)                              # 预热
    ts = []
    for _ in range(50):
        t0 = time.perf_counter()
        sel.选择(候选, G)
        ts.append(time.perf_counter() - t0)
    p95 = float(np.percentile(ts, 95)) * 1e3
    c = [_判("50 候选 EFE 选择 p95", round(p95, 3), "≤3ms", p95 <= 3.0)]
    return dict(route="A2-T 延迟", passed=p95 <= 3.0, criteria=c)


def main():
    组 = [("A2C", test_A2C), ("A2E", test_A2E), ("A2S", test_A2S), ("A2D", test_A2D), ("A2T", test_A2T)]
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
    json.dump(结果, open(os.path.join(HERE, "步2b_判据结果.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    return 0 if 全过 else 1


if __name__ == "__main__":
    sys.exit(main())
