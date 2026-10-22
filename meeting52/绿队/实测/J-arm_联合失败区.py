# -*- coding: utf-8 -*-
"""绿队 · J-arm 联合失败区实测（主控 2026-10-17 开窗；<2min，正确性 only）

单一合法变体（只调既有 API，不改件）：
  A_t ← P@特征(x_t)                      # 每拍重算，无历史（两区室核.设自我态向量）
  E_t ← 绑(键_t, o_t)                    # 单事件、不叠加（无历史）

测量：
  PS-B t50：切换点后，从 A_t 解码"上一规则"（最近邻 z-grid；判对=更接近旧边界）；
            窗口滚动准确率 ≤ 0.5+2SE 的首个 k 即 t50；另跑"有状态"对照臂（正常 V4 A）。
  PV AUC：预测 ê=解绑(键_t,E_{t−1})、a=cos(ê,o_t)，AUC(自源 vs 外源)；bootstrap CI。
  联合读数 Y = 1[PS-B 落失败区(t50=0)] ∧ 1[PV 落失败区(AUC≤0.55 或 CI 上界<0.60)]。

纪律：只读导入既有件（AHI4/meeting49/实施/算子_两区室核.py、算子_能力核.py）；只写本目录；
      并发期墙钟不宣称；两趟逐位同。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
绿队 = os.path.dirname(HERE)
meeting52 = os.path.dirname(绿队)
AHI4 = os.path.dirname(meeting52)
_核py = os.path.join(AHI4, "meeting41", "实施", "算子_能力核.py")
_两区室py = os.path.join(AHI4, "meeting49", "实施", "算子_两区室核.py")

阈值 = dict(t50失败=0, AUC失败=0.55, AUC上界=0.60, chance=0.5)


def 载(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


_m = 载("能力核_只读_jarm", _核py)
能力核 = _m.能力核
_归 = _m._归
_两 = 载("两区室核_只读_jarm", _两区室py)
两区室核 = _两.两区室核
生成序列 = _两.生成序列
特征 = _两.特征
边界训练 = _两.边界训练


def 双极(rng, n):
    return np.where(rng.integers(0, 2, size=int(n)) == 1, 1.0, -1.0)


def 解码z(核, A, 网格, A网格):
    """从 A 解码 自我标量 z：最近邻于 A(z)=P@特征(z) 的网格表示。"""
    d = np.sum((A网格 - np.asarray(A)[None, :]) ** 2, axis=1)
    return float(网格[int(np.argmin(d))])


def 跑PS(种子, 模式, 每种子=400):
    """模式 ∈ {"有状态","J_arm"}；返回 每切换点后的"旧规则解码"准确率曲线（长度 W）。"""
    核 = 两区室核(D=64, 种子=int(种子))
    P = np.asarray(核.P, dtype=np.float64)
    网格 = np.linspace(0.05, 0.95, 91)
    A网格 = np.array([_归(P @ 特征(float(z))) for z in 网格])
    序列 = 生成序列(int(种子) + 1, 每种子, 边界训练)
    x = 序列["x"]; b = 序列["边界"]; 切 = list(序列["切换点"])
    A列 = []
    for t in range(int(每种子)):
        o = 核.前向_试次(float(x[t]))
        ŷ = int(np.argmax(核.读出 @ o))
        c = 1 if float(x[t]) > float(b[t]) else 0
        奖 = int(ŷ == c)
        核.学读出(ŷ, 奖, 更新读出=True)
        核.学自我(ŷ, 奖)
        if 模式 == "J_arm":
            核.设自我态向量(_归(P @ 特征(float(x[t]))))
        A列.append(np.array(核.自我态, dtype=np.float64))
    W = 15
    曲线 = []          # 每个切换点后 k=0..W-1 的"旧规则解码"对/错
    for s in 切:
        if s < 120 or s + W >= 每种子:
            continue
        b旧 = float(b[s - 1]); b新 = float(b[s])
        for k in range(W):
            t = s + k
            ẑ = 解码z(核, A列[t], 网格, A网格)
            if abs(ẑ - b旧) < abs(ẑ - b新):
                曲线.append((k, 1.0))
            elif abs(ẑ - b旧) > abs(ẑ - b新):
                曲线.append((k, 0.0))
            else:
                曲线.append((k, 0.5))
    # 每个 k 的平均准确率与 2SE；t50=首个 k 使窗口 [k,k+3) 均值 ≤0.5+2SE
    ks = sorted(set(k for k, _ in 曲线))
    均 = {}
    for k in ks:
        v = [v for kk, v in 曲线 if kk == k]
        均[k] = float(np.mean(v))
    def 窗均(k):
        v = [v for kk, v in 曲线 if k <= kk < k + 3]
        return float(np.mean(v)) if v else float("nan")
    t50 = None
    for k in ks:
        if 窗均(k) <= 阈值["chance"] + 2 * (0.5 / np.sqrt(max(1, len([1 for kk, _ in 曲线 if kk == k])))):
            t50 = int(k); break
    return dict(种子=int(种子), 模式=模式, 切换点数=len(切), 曲线=均,
                t50=(t50 if t50 is not None else -1),
                曲线k0=窗均(0), 曲线k3=窗均(3))


def 跑PV(种子, K键=8, D=256, 事件数=400, 自源比=0.5):
    核 = 能力核(D=D, 种子=int(种子))
    rng = np.random.default_rng(int(种子))
    值 = np.array([双极(rng, D) for _ in range(K键)])
    键码 = np.array([双极(rng, D) for _ in range(K键)])
    自列 = [bool(rng.random() < 自源比) for _ in range(事件数)]
    正, 负 = [], []
    E = np.zeros(D, dtype=np.float64)          # 初始无历史
    for t in range(int(事件数)):
        k = int(t % K键)                        # 键循环（避免相邻重复键的假信号）
        o = 值[k].copy() if 自列[t] else 双极(rng, D)
        if t > 0:
            ê = 核.解绑(键码[k], E)
            a = float(np.dot(ê, o)) / D
            if t >= 20:
                (正 if 自列[t] else 负).append(a)
        E = 核.绑(键码[k], o)                   # ★J-arm：单事件，不叠加（每拍重置）
    def AUC(正, 负):
        正 = np.asarray(正); 负 = np.asarray(负)
        return float(np.mean(正[:, None] > 负[None, :]) + 0.5 * np.mean(正[:, None] == 负[None, :]))
    auc = AUC(正, 负)
    rngb = np.random.default_rng(int(种子) + 99)
    正 = np.asarray(正); 负 = np.asarray(负); boot = []
    for _ in range(1000):
        b1 = 正[rngb.integers(0, len(正), len(正))]; b2 = 负[rngb.integers(0, len(负), len(负))]
        boot.append(AUC(b1, b2))
    return dict(种子=int(种子), n正=len(正), n负=len(负), AUC=auc,
                CI低=float(np.quantile(boot, 0.025)), CI高=float(np.quantile(boot, 0.975)),
                a正均值=float(np.mean(正)), a负均值=float(np.mean(负)))


def 跑():
    PS = dict(有状态=[], J_arm=[])
    for 种子 in (55001, 55002, 55003):
        PS["有状态"].append(跑PS(种子, "有状态"))
        PS["J_arm"].append(跑PS(种子, "J_arm"))
    PV = [跑PV(种子) for 种子 in (55101, 55102, 55103)]
    # 判定
    j_t50 = [r["t50"] for r in PS["J_arm"]]
    j_曲线0 = float(np.mean([r["曲线k0"] for r in PS["J_arm"]]))
    有_t50 = [r["t50"] for r in PS["有状态"]]
    pv_auc = float(np.mean([r["AUC"] for r in PV]))
    pv_ci高 = max(r["CI高"] for r in PV)
    ps失败 = all(t == 0 for t in j_t50)
    pv失败 = (pv_auc <= 阈值["AUC失败"]) or (pv_ci高 < 阈值["AUC上界"])
    Y = int(ps失败 and pv失败)
    判 = dict(PS_B_j_t50=j_t50, PS_B_j_曲线k0=round(j_曲线0, 4),
              有状态_t50对照=有_t50,
              PV_AUC=round(pv_auc, 4), PV_CI高=round(pv_ci高, 4),
              PS落失败区=bool(ps失败), PV落失败区=bool(pv失败),
              联合Y=Y,
              结论=("J-arm 成立（可执行复合判据）" if Y == 1 else
                    ("部分失效（PV 半边不成立 ⇒ 需来源随机化臂补）" if ps失败 else "不可判定")))
    return dict(PS=PS, PV=PV, 判定=判)


def main():
    global 核
    核 = 能力核(D=64, 种子=55001)
    t0 = time.time()
    一 = 跑(); 二 = 跑()
    同 = bool(json.dumps(一, ensure_ascii=False, sort_keys=True) == json.dumps(二, ensure_ascii=False, sort_keys=True))
    果 = dict(meta=dict(件="J-arm 联合失败区", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
                        并发态="并发期；只作正确性判定，墙钟不宣称",
                        变体="A_t←P@特征(x_t)（每拍重算）+ E_t←绑(键_t,o_t)（单事件）",
                        种子="PS 55001–55003 / PV 55101–55103", 预注册阈值=阈值,
                        墙钟_s=round(time.time() - t0, 3), 两趟逐位同=同),
               result=一, 判定=一["判定"])
    with open(os.path.join(HERE, "J-arm_结果.json"), "w", encoding="utf-8") as f:
        json.dump(果, f, ensure_ascii=False, indent=1)
    print(json.dumps(果, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        np.seterr(all="ignore")
    except Exception:
        pass
    main()
