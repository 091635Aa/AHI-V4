# -*- coding: utf-8 -*-
"""红队 R3 · M6 统一口径 + 双替身（协议脚本）

口径：绿队种子 53001–53005、D=256、K=8、800 事件、自源比 0.5、η=1/16、晚窗 t≥600、threads=1。
统一：去 DC（每题减均值）+ 捆绑（非 Hebbian：E 为 绑(键,观测) 的滑动平均）+ 平衡 AUC（等量重采样）
      + 统一任务（所有臂共用同一事件流）。
臂：正控 / 替身A 键置换 π1/π2 / 替身B 内容无绑定 B1/B2 / PV2 标签置换 / PV4 未见来源。
判据（预注册）：正控 AUC晚 ≥0.80 且 5 seeds 全过；双替身 AUC晚 ≤0.55 且 CI 上界 <0.60；
                 任一 seed ≥0.75 ⇒ 硬否证。
纪律：只读导入既有件；只写本目录 JSON；脚本经 `python -B` 运行；不改任何既有件。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_m = None


def 载能力核():
    global _m
    p = r"<workspace>/03_AHI\AHI4\meeting41\实施\算子_能力核.py"
    s = importlib.util.spec_from_file_location("能力核_m6_u", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def 双极(rng, n):
    return np.where(rng.integers(0, 2, size=int(n)) == 1, 1.0, -1.0)


def DC(v):
    v = np.asarray(v, dtype=np.float64)
    return v - float(v.mean())


def AUC(正, 负):
    正 = np.asarray(正, dtype=np.float64)
    负 = np.asarray(负, dtype=np.float64)
    if 正.size == 0 or 负.size == 0:
        return float("nan")
    return float(np.mean(正[:, None] > 负[None, :]) + 0.5 * np.mean(正[:, None] == 负[None, :]))


def 平衡AUC(正, 负, rng, 次数=100):
    """等量重采样（正负各取 min(n)），取均值 —— 消除不平衡的加权伪影。"""
    正 = np.asarray(正, dtype=np.float64)
    负 = np.asarray(负, dtype=np.float64)
    n = int(min(正.size, 负.size))
    if n < 2:
        return float("nan")
    值 = []
    for _ in range(int(次数)):
        a = 正[rng.choice(正.size, size=n, replace=False)]
        b = 负[rng.choice(负.size, size=n, replace=False)]
        值.append(AUC(a, b))
    return float(np.mean(值))


def 生成事件(种子, D=256, K=8, 事件数=800, 自源比=0.5):
    核 = _m.能力核(D=int(D), 种子=int(种子))
    rng = np.random.default_rng(int(种子))
    值 = np.array([双极(rng, D) for _ in range(K)])
    键 = np.array([双极(rng, D) for _ in range(K)])
    事件 = []
    for t in range(int(事件数)):
        k = int(rng.integers(0, K))
        自 = bool(rng.random() < 自源比)
        o = 值[k].copy() if 自 else 双极(rng, D)
        事件.append((k, 自, o))
    return 核, 值, 键, 事件


def 跑臂(核, 键, 值, 事件, η=1.0 / 16.0, D=256, 臂="正控", 置换=None, 内容置换=None, 替代=None):
    E = np.zeros(D, dtype=np.float64)
    正晚, 负晚 = [], []
    for t, (k, 自, o) in enumerate(事件):
        if 臂 == "正控":
            qk = k
        elif 臂 == "A_pi1":
            qk = (int(k) + 1) % len(键)
        elif 臂 in ("A_pi2", "B1", "B2"):
            qk = int(置换[k])
        else:
            qk = k
        ê = 核.解绑(键[qk], E)
        a = float(np.dot(DC(ê), DC(o))) / D          # 去 DC 后的归一化内积
        if t >= 600:
            (正晚 if 自 else 负晚).append(a)
        # 更新（非 Hebbian：绑(键, 观测) 的滑动平均）
        if 臂 == "B1":
            oo = 替代[t] if 自 else o                 # 自源内容被替换为当拍新随机向量
        elif 臂 == "B2":
            oo = 值[int(内容置换[k])] if 自 else o    # 自源内容被错配到别的内容
        else:
            oo = o
        E = 核.归((1.0 - η) * E + η * 核.绑(键[k], oo))
    return 正晚, 负晚


def main():
    开 = time.time()
    种子列 = [53001, 53002, 53003, 53004, 53005]
    结果 = {}
    逐种子 = []
    for 种子 in 种子列:
        核, 值, 键, 事件 = 生成事件(种子)
        K = len(键)
        置换 = np.random.default_rng(int(种子) + 1).permutation(K)      # π2 / 历史置换
        内容置换 = np.random.default_rng(int(种子) + 12).permutation(K)  # B2 内容错配
        rng替 = np.random.default_rng(int(种子) + 11)
        替代 = [双极(rng替, 256) for _ in 事件]                          # B1 当拍随机向量
        臂集 = {
            "正控": dict(臂="正控"),
            "A_pi1": dict(臂="A_pi1"),
            "A_pi2": dict(臂="A_pi2", 置换=置换),
            "B1": dict(臂="B1", 置换=np.arange(K), 替代=替代),
            "B2": dict(臂="B2", 置换=np.arange(K), 内容置换=内容置换),
        }
        行 = {}
        for 名, kw in 臂集.items():
            正, 负 = 跑臂(核, 键, 值, 事件, **kw)
            rng_ci = np.random.default_rng(int(种子) + 31)
            行[名] = dict(n正=len(正), n负=len(负),
                          AUC晚=float(AUC(正, 负)),
                          平衡AUC晚=float(平衡AUC(正, 负, rng_ci)),
                          正均值晚=float(np.mean(正)), 负均值晚=float(np.mean(负)))
            # bootstrap 95% CI（对平衡AUC）
            bs = []
            for b in range(300):
                r = np.random.default_rng(int(种子) * 1000 + b + 7)
                bs.append(平衡AUC(正, 负, r, 次数=8))
            行[名]["平衡AUC_CI95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        # PV2 标签置换（正控的晚窗分数 + 标签）
        正, 负 = 跑臂(核, 键, 值, 事件, 臂="正控")
        分 = np.array(list(正) + list(负)); 标 = np.array([1] * len(正) + [0] * len(负))
        perm_rng = np.random.default_rng(int(种子) + 41); 置AUC = []
        for _ in range(40):
            标p = perm_rng.permutation(标)
            置AUC.append(float(AUC(分[标p == 1], 分[标p == 0])))
        行["PV2_标签置换"] = dict(均值=float(np.mean(置AUC)),
                                P05=float(np.percentile(置AUC, 5)),
                                P95=float(np.percentile(置AUC, 95)), 次数=40)
        # PV4 未见来源（n=1000：500 自源 + 500 未见外源 0.5 幅度；另 500 频率×2）
        E = np.zeros(256)
        for (k, 自, o) in 事件:
            E = 核.归((1.0 - 1.0 / 16.0) * E + (1.0 / 16.0) * 核.绑(键[k], o))
        rng4 = np.random.default_rng(int(种子) + 51)
        正4, 负4, 负频 = [], [], []
        for _ in range(1000):
            k = int(rng4.integers(0, K)); 自 = bool(rng4.random() < 0.5)
            o = 值[k].copy() if 自 else 0.5 * 双极(rng4, 256)
            ê = 核.解绑(键[k], E); a = float(np.dot(DC(ê), DC(o))) / 256
            (正4 if 自 else 负4).append(a)
        for _ in range(500):
            k = int(rng4.integers(0, K)); o = 0.25 * (双极(rng4, 256) + 双极(rng4, 256))
            ê = 核.解绑(键[k], E); 负频.append(float(np.dot(DC(ê), DC(o))) / 256)
        行["PV4"] = dict(n=1000, AUC=float(AUC(正4, 负4)), AUC_频率x2=float(AUC(正4, 负频)))
        逐种子.append(dict(种子=int(种子), 臂=行))
        # 两趟确定性：正控复跑
        p1, n1 = 跑臂(核, 键, 值, 事件, 臂="正控")
        p2, n2 = 跑臂(核, 键, 值, 事件, 臂="正控")
        逐种子[-1]["两趟逐位同"] = bool(p1 == p2 and n1 == n2)
    结果["逐种子"] = 逐种子
    # 汇总 + 判定（预注册）
    臂名 = ["正控", "A_pi1", "A_pi2", "B1", "B2"]
    汇总 = {}
    for a in 臂名:
        vals = [s["臂"][a] for s in 逐种子]
        汇总[a] = dict(
            AUC晚=[v["AUC晚"] for v in vals],
            平衡AUC晚=[v["平衡AUC晚"] for v in vals],
            CI95=[v["平衡AUC_CI95"] for v in vals],
            最小AUC晚=float(min(v["AUC晚"] for v in vals)),
            最大AUC晚=float(max(v["AUC晚"] for v in vals)),
        )
    # 判定
    def 过(臂, 上界, 下界=None):
        for v in 汇总[臂]["AUC晚"]:
            if 下界 is not None and v < 下界:
                return False
            if v > 上界:
                return False
        return True
    判定 = {
        "正控_全_ge_0.80": bool(all(v >= 0.80 for v in 汇总["正控"]["AUC晚"])),
        "A_pi1_全_le_0.55": bool(all(v <= 0.55 for v in 汇总["A_pi1"]["AUC晚"])),
        "A_pi2_全_le_0.55": bool(all(v <= 0.55 for v in 汇总["A_pi2"]["AUC晚"])),
        "B1_全_le_0.55": bool(all(v <= 0.55 for v in 汇总["B1"]["AUC晚"])),
        "B2_全_le_0.55": bool(all(v <= 0.55 for v in 汇总["B2"]["AUC晚"])),
        "任一替身_ge_0.75_硬否证": bool(any(max(汇总[a]["AUC晚"]) >= 0.75 for a in ("A_pi1", "A_pi2", "B1", "B2"))),
        "PV4_全_ge_0.70": bool(all(s["臂"]["PV4"]["AUC"] >= 0.70 for s in 逐种子)),
        "PV2_均值_le_0.55": bool(all(s["臂"]["PV2_标签置换"]["均值"] <= 0.55 for s in 逐种子)),
        "两趟全同": bool(all(s["两趟逐位同"] for s in 逐种子)),
    }
    判定["CI上界全_lt_0.60"] = bool(all(v[1] < 0.60 for a in ("A_pi1", "A_pi2", "B1", "B2") for v in 汇总[a]["CI95"]))
    ok = all([判定["正控_全_ge_0.80"], 判定["A_pi1_全_le_0.55"], 判定["A_pi2_全_le_0.55"],
              判定["B1_全_le_0.55"], 判定["B2_全_le_0.55"], 判定["CI上界全_lt_0.60"],
              判定["PV4_全_ge_0.70"], 判定["PV2_均值_le_0.55"], 判定["两趟全同"]]) and (not 判定["任一替身_ge_0.75_硬否证"])
    结果.update(meta=dict(件="M6 统一口径 + 双替身（红队 R3/P0-2）",
                          时间=time.strftime("%Y-%m-%d %H:%M:%S"),
                          并发态="并发期；墙钟仅记录、不作效率宣称",
                          口径="种子 53001–53005; D=256; K=8; 800 事件; 自源比 0.5; η=1/16; 晚窗 t≥600; 去DC; 捆绑非Hebbian; 平衡AUC; 统一任务",
                          线程=os.environ.get("OPENBLAS_NUM_THREADS", "unset")),
                汇总=汇总, 判定=判定,
                四态=("复现 PASS" if ok else "不复现"),
                说明=("双替身均 ≤0.55 且 CI 上界<0.60、正控 5 seeds ≥0.80、PV4 全 ≥0.70、PV2 均值 ≤0.55 ⇒ 可升'已证'" if ok
                      else "未全部满足预注册条件 ⇒ M6 停'争议中/单队'；逐条见 判定"),
                墙钟_s=round(time.time() - 开, 3))
    out = os.path.join(HERE, "M6_统一口径_双替身.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(结果, f, ensure_ascii=False, indent=1)
    print(json.dumps(dict(判定=判定, 汇总={k: dict(AUC晚=v["AUC晚"], 平衡=v["平衡AUC晚"], CI=v["CI95"]) for k, v in 汇总.items()},
                         四态=结果["四态"], 墙钟_s=结果["墙钟_s"]), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _m = 载能力核()
    main()
