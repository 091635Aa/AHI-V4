# -*- coding: utf-8 -*-
"""红队 R3 · M6 统一口径 + 双替身 v2（一次性预注册；阈值不动）

预注册（运行前写死，跑后不改）：
  种子 53001–53005；D=256；K=8；学习流 800 事件；自源比 0.5；η=1/16。
  **评价窗 = 学习流晚窗 t>=600**（与绿队/v1/spec 同口径，n≈200）。统一任务：所有臂共用同一事件流。
  去 DC + 捆绑（非 Hebbian：E 为 绑(键,观测) 的滑动平均）+ 平衡 AUC（正负等量重采样）。
  Aπ1：循环移位；**Aπ2 = derangement（无不动点）**，报不动点数与尝试数。
  B1：自源更新改喂当拍新随机向量；**5 个独立替代抽样（rng(seed+71..75)）**；判据用 5 抽样均值；
      CI 上界取 5 抽样最大值（n 合计≈5×200≥1000）。
  B2：内容错配 **σ = derangement**，报不动点数。
  PV2：把晚窗正/负分数合并后做 **40 次标签置换**（随机选 n_pos 个为正），报均值+5%/95% 分位。
  PV4：**每键等量（平衡）未见样本**：8 键 × 125 = 1000（每键自源 63 / 外源 62；外源=0.5 幅度）；
      另频率×2 负样本 500（每键约 62）。每 seed 报 AUC + bootstrap 95% CI。
  判据（不变）：正控 ≥0.80 且 5/5；双替身 AUC ≤0.55、CI 上界 <0.60；最大替身 <0.75 才算无硬否证。
  结论：PASS（升已证）/ 不复现（保持争议）/ 硬否证（判死）。
纪律：只读导入既有件；只写本目录；python -B；不改既有件；线程=1（环境变量）。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_M = None


def 载核():
    global _M
    p = r"<workspace>/03_AHI\AHI4\meeting41\实施\算子_能力核.py"
    s = importlib.util.spec_from_file_location("能力核_m6_v2", p)
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


def 平衡AUC(正, 负, rng, 次=100):
    正 = np.asarray(正, dtype=np.float64)
    负 = np.asarray(负, dtype=np.float64)
    n = int(min(正.size, 负.size))
    if n < 2:
        return float("nan")
    值 = []
    for _ in range(int(次)):
        a = 正[rng.choice(正.size, size=n, replace=False)]
        b = 负[rng.choice(负.size, size=n, replace=False)]
        值.append(AUC(a, b))
    return float(np.mean(值))


def bootstrap_CI(正, 负, seed, 次=100):
    正 = np.asarray(正, dtype=np.float64)
    负 = np.asarray(负, dtype=np.float64)
    值 = []
    for b in range(int(次)):
        r = np.random.default_rng(int(seed) * 1000 + b + 3)
        a = 正[r.integers(0, 正.size, size=正.size)]
        c = 负[r.integers(0, 负.size, size=负.size)]
        值.append(AUC(a, c))
    return [float(np.percentile(值, 2.5)), float(np.percentile(值, 97.5))]


def derangement(rng, n):
    for 次 in range(10000):
        p = rng.permutation(int(n))
        if not np.any(p == np.arange(int(n))):
            return p, 次
    raise RuntimeError("derangement 失败")


def 生成(种子, 事件数=800, D=256, K=8):
    核 = _M.能力核(D=D, 种子=int(种子))
    rng = np.random.default_rng(int(种子))
    值 = np.array([双极(rng, D) for _ in range(K)])
    键 = np.array([双极(rng, D) for _ in range(K)])
    事件 = []
    for t in range(int(事件数)):
        k = int(rng.integers(0, K))
        自 = bool(rng.random() < 0.5)
        o = 值[k].copy() if 自 else 双极(rng, D)
        事件.append((k, 自, o))
    return 核, 值, 键, 事件


def 学评(核, 键, 事件, 更新臂="正控", 查询臂="正控", π=None, σ=None, 替代=None, 值=None, η=1.0 / 16.0, D=256):
    """一边学习一边评晚窗（t>=600）：E 使用当拍之前的在线状态（与 v1/绿队口径同）。"""
    E = np.zeros(D, dtype=np.float64)
    正, 负 = [], []
    for t, (k, 自, o) in enumerate(事件):
        if 查询臂 == "正控":
            qk = k
        elif 查询臂 == "A_pi1":
            qk = (int(k) + 1) % len(键)
        else:
            qk = int(π[k])
        ê = 核.解绑(键[qk], E)
        a = float(np.dot(DC(ê), DC(o))) / 256
        if t >= 600:
            (正 if 自 else 负).append(a)
        if 更新臂 == "B1":
            oo = 替代[t] if 自 else o
        elif 更新臂 == "B2":
            oo = 值[σ[k]] if 自 else o
        else:
            oo = o
        E = 核.归((1.0 - η) * E + η * 核.绑(键[k], oo))
    return E, 正, 负


def main():
    开 = time.time()
    种子列 = [53001, 53002, 53003, 53004, 53005]
    逐种子 = []
    for 种子 in 种子列:
        核, 值, 键, 事件 = 生成(种子)
        K = len(键)
        π2, 尝2 = derangement(np.random.default_rng(int(种子) + 1), K)
        σ, 尝σ = derangement(np.random.default_rng(int(种子) + 12), K)
        E正, 正, 负 = 学评(核, 键, 事件, 更新臂="正控", 查询臂="正控")
        行 = {}
        行["正控"] = dict(AUC=AUC(正, 负), 平衡AUC=平衡AUC(正, 负, np.random.default_rng(种子 + 3)),
                          CI=bootstrap_CI(正, 负, 种子))
        _, p1, n1 = 学评(核, 键, 事件, 更新臂="正控", 查询臂="A_pi1")
        行["A_pi1"] = dict(AUC=AUC(p1, n1), CI=bootstrap_CI(p1, n1, 种子))
        _, p2, n2 = 学评(核, 键, 事件, 更新臂="正控", 查询臂="A_pi2", π=π2)
        行["A_pi2"] = dict(AUC=AUC(p2, n2), CI=bootstrap_CI(p2, n2, 种子),
                          不动点=int(np.sum(π2 == np.arange(K))), 尝试=int(尝2))
        # B1：5 个独立替代抽样，每次评晚窗
        b1 = []
        for j in range(5):
            rngB = np.random.default_rng(int(种子) + 71 + j)
            替代 = [双极(rngB, 256) for _ in 事件]
            E1, pb, nb = 学评(核, 键, 事件, 更新臂="B1", 替代=替代)
            b1.append(dict(AUC=AUC(pb, nb), CI=bootstrap_CI(pb, nb, 种子 * 10 + j)))
        行["B1"] = {"5抽样": b1, "均值": float(np.mean([x["AUC"] for x in b1])),
                     "CI上界最大": float(max(x["CI"][1] for x in b1)),
                     "n合计": int(len(pb) + len(nb)) * 5}
        E2, pb2, nb2 = 学评(核, 键, 事件, 更新臂="B2", σ=σ, 值=值)
        行["B2"] = dict(AUC=AUC(pb2, nb2), CI=bootstrap_CI(pb2, nb2, 种子),
                       不动点=int(np.sum(σ == np.arange(K))), 尝试=int(尝σ))
        # PV2：合并晚窗分数后标签置换 40 次
        合 = np.array(list(正) + list(负)); npos = len(正); 置 = []
        for j in range(40):
            r = np.random.default_rng(int(种子) + 91 + j)
            标 = np.zeros(合.size, dtype=bool); 标[r.choice(合.size, npos, replace=False)] = True
            置.append(AUC(合[标], 合[~标]))
        行["PV2"] = dict(均值=float(np.mean(置)), P05=float(np.percentile(置, 5)),
                         P95=float(np.percentile(置, 95)))
        # PV4：每键等量平衡未见样本（n=1000，自源 500 / 外源 500）
        rng4 = np.random.default_rng(int(种子) + 51)
        正4, 负4, 负频 = [], [], []
        for k in range(K):
            for _ in range(63):
                o = 值[k].copy(); ê = 核.解绑(键[k], E正)
                正4.append(float(np.dot(DC(ê), DC(o))) / 256)
            for _ in range(62):
                o = 0.5 * 双极(rng4, 256); ê = 核.解绑(键[k], E正)
                负4.append(float(np.dot(DC(ê), DC(o))) / 256)
        for k in range(K):
            for _ in range(62):
                o = 0.25 * (双极(rng4, 256) + 双极(rng4, 256)); ê = 核.解绑(键[k], E正)
                负频.append(float(np.dot(DC(ê), DC(o))) / 256)
        行["PV4"] = dict(n=len(正4) + len(负4), AUC=AUC(正4, 负4),
                         CI=bootstrap_CI(正4, 负4, 种子 + 5),
                         AUC_频率x2=AUC(正4, 负频), n正=len(正4), n负=len(负4))
        # 两趟确定性
        E2b, p2b, n2b = 学评(核, 键, 事件, 更新臂="正控", 查询臂="正控")
        行["两趟逐位同"] = bool(np.array_equal(np.asarray(正), np.asarray(p2b)) and np.array_equal(np.asarray(负), np.asarray(n2b)))
        逐种子.append(dict(种子=int(种子), 臂=行))
    def 列(名):
        return [s["臂"][名]["AUC"] for s in 逐种子]
    汇总 = dict(
        正控=列("正控"), A_pi1=列("A_pi1"), A_pi2=列("A_pi2"),
        B1均值=[s["臂"]["B1"]["均值"] for s in 逐种子],
        B2=列("B2"),
        B1_CI上界最大=[s["臂"]["B1"]["CI上界最大"] for s in 逐种子],
        A_pi2_CI上界=[s["臂"]["A_pi2"]["CI"][1] for s in 逐种子],
        B2_CI上界=[s["臂"]["B2"]["CI"][1] for s in 逐种子],
        PV4=[s["臂"]["PV4"]["AUC"] for s in 逐种子],
        PV4_CI=[s["臂"]["PV4"]["CI"] for s in 逐种子],
        PV2均值=[s["臂"]["PV2"]["均值"] for s in 逐种子],
        两趟全同=bool(all(s["臂"]["两趟逐位同"] for s in 逐种子)),
    )
    替身AUC = 汇总["A_pi1"] + 汇总["A_pi2"] + 汇总["B1均值"] + 汇总["B2"]
    判 = {
        "正控_5seeds_ge_0.80": bool(all(v >= 0.80 for v in 汇总["正控"])),
        "A_pi1_全_le_0.55": bool(all(v <= 0.55 for v in 汇总["A_pi1"])),
        "A_pi2_全_le_0.55": bool(all(v <= 0.55 for v in 汇总["A_pi2"])),
        "B1均值_全_le_0.55": bool(all(v <= 0.55 for v in 汇总["B1均值"])),
        "B2_全_le_0.55": bool(all(v <= 0.55 for v in 汇总["B2"])),
        "CI上界_全_lt_0.60": bool(all(v < 0.60 for v in 汇总["A_pi2_CI上界"] + 汇总["B2_CI上界"] + 汇总["B1_CI上界最大"])),
        "任一替身_ge_0.75_硬否证": bool(any(v >= 0.75 for v in 替身AUC)),
        "PV4_5seeds_ge_0.70": bool(all(v >= 0.70 for v in 汇总["PV4"])),
        "PV2均值_全_le_0.55": bool(all(v <= 0.55 for v in 汇总["PV2均值"])),
        "两趟全同": 汇总["两趟全同"],
    }
    if 判["任一替身_ge_0.75_硬否证"]:
        四态 = "硬否证"
    elif all(判.values()):
        四态 = "PASS"
    else:
        四态 = "不复现"
    out = dict(meta=dict(件="M6 统一口径 + 双替身 v2（一次性预注册）",
                         时间=time.strftime("%Y-%m-%d %H:%M:%S"),
                         并发态="并发期；墙钟不作效率宣称",
                         线程=os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
                         预注册="学习流 800；评价=晚窗 t>=600；B1 5 抽样(n合计≥1000)；Aπ2/B2 derangement；PV4 每键平衡 n=1000；判据不变"),
                 逐种子=逐种子, 汇总=汇总, 判定=判, 四态=四态,
                 墙钟_s=round(time.time() - 开, 3),
                 说明=("PASS：双替身(derangement/5抽样)均 ≤0.55 且 CI 上界<0.60、正控 5/5≥0.80、PV4 5/5≥0.70 ⇒ 可升'已证'" if 四态 == "PASS"
                       else ("硬否证：derangement 下仍有替身 ≥0.75 ⇒ M6 判死" if 四态 == "硬否证"
                             else "不复现：逐条见 判定；保持争议")))
    p = os.path.join(HERE, "M6_统一口径_双替身_v2.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(dict(四态=out["四态"], 判定=判, 汇总=汇总, 墙钟_s=out["墙钟_s"]), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _M = 载核()
    main()
