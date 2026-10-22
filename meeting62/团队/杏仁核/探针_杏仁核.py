# -*- coding: utf-8 -*-
"""meeting62/团队/杏仁核/探针_杏仁核.py —— 效价/唤醒 → 声带参数的因果驱动链探针。

只读 import：meeting32/声带.py、meeting61/实施/声带后端_v6.py（C14：一字不改）。
所有产物写入本目录（meeting62 内），不写 meeting61/产物。
CPU-only；纯 numpy；无外部模型；无参考音频。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEET62 = os.path.dirname(os.path.dirname(HERE))
AHI4 = os.path.dirname(MEET62)
sys.path.insert(0, os.path.join(AHI4, "meeting32"))
sys.path.insert(0, os.path.join(AHI4, "meeting61", "实施"))

import 声带 as 声带模块  # noqa: E402

基频区间 = (70.0, 400.0)


def 夹(v, lo, hi):
    return float(min(max(float(v), lo), hi))


# ─────────────── 提案：两级因果链 情感态 → 生理中间量 → 8 声学参数 ───────────────
# 依据（见 R1 §2）：唤醒 A → 交感激活（呼吸加深/声带张力）→ F0↑ 能量↑ 语速↑ 紧喉↑；
# 效价 V → 声门靠拢平衡/谱质：V− 低唤醒→气声↑（松弛叹息）、V− 高唤醒→粗糙/挤压↑、
# V+ → 清亮（气声↓）；支配 P → 声门下压/起音 → 幅度↑ 紧喉↑。
def 情感驱动(V, A, P, 基线=None):
    """(效价 V∈[-1,1], 唤醒 A∈[-1,1], 支配 P∈[-1,1]) → 律 dict。连续、单调、确定性。"""
    基 = dict(基频=190.0, 语速=3.2, 幅度=0.45, 气声度=0.15, 紧喉度=0.30,
               粗糙度=0.03, 送气量=0.20, 声区=0.05) if 基线 is None else dict(基线)
    V, A, P = 夹(V, -1, 1), 夹(A, -1, 1), 夹(P, -1, 1)
    # 半音域映射（唤醒主导，P 次之；V 小幅——正面略高、负面在低唤醒下拉低）
    半音 = 3.0 * A + 0.7 * V + 1.0 * P
    基频 = 夹(基["基频"] * 2.0 ** (半音 / 12.0), *基频区间)
    幅度 = 夹(基["幅度"] * (1.0 + 0.55 * A + 0.20 * P), 0.10, 1.0)
    语速 = 夹(基["语速"] * (1.0 + 0.30 * A + 0.08 * P), 1.5, 7.0)
    紧喉 = 夹(0.30 + 0.35 * A + 0.15 * max(-V, 0.0) * max(A, 0.0)
              - 0.15 * max(V, 0.0) + 0.10 * P, 0.0, 1.0)
    气声 = 夹(0.10 + 0.35 * max(-V, 0.0) * (1.0 - A) + 0.10 * max(-A, 0.0) * max(-V, 0.0),
              0.0, 0.8)
    粗糙 = 夹(0.03 + 0.20 * max(A, 0.0) * max(-V, 0.0) + 0.10 * max(A, 0.0), 0.0, 1.0)
    送气 = 夹(0.20 + 0.40 * 气声, 0.0, 1.0)
    声区 = 夹(0.05 + 0.30 * max(A, 0.0) + 0.20 * max(V, 0.0) * max(A, 0.0), 0.0, 1.0)
    return dict(基频=基频, 语速=语速, 幅度=幅度, 气声度=气声, 紧喉度=紧喉,
                粗糙度=粗糙, 送气量=送气, 声区=声区)


色 = ("基频", "语速", "幅度", "气声度", "紧喉度", "粗糙度", "送气量", "声区")


def 键序(律):
    return np.array([律[k] for k in 色], dtype=np.float64)


def 归一距(a, b, 尺度):
    return float(np.linalg.norm((a - b) / 尺度))


# ─────────────── 对照：现行 template（说_音色.情绪偏移 口径，离散 4 档） ───────────────
_模板 = {
    "高兴": dict(基频倍=1.15, 语速倍=1.18, 幅度倍=1.10, 气声加=-0.05, 紧喉加=0.10),
    "低落": dict(基频倍=0.88, 语速倍=0.82, 幅度倍=0.85, 气声加=0.15, 紧喉加=-0.10),
    "急促": dict(基频倍=1.10, 语速倍=1.35, 幅度倍=1.05, 气声加=-0.02, 紧喉加=0.12),
    "平静": dict(基频倍=1.00, 语速倍=1.00, 幅度倍=1.00, 气声加=0.00, 紧喉加=0.00),
}


def 模板律(情绪: str, 基线):
    m = _模板[情绪]
    return dict(基频=夹(基线["基频"] * m["基频倍"], *基频区间),
                语速=夹(基线["语速"] * m["语速倍"], 1.5, 7.0),
                幅度=夹(基线["幅度"] * m["幅度倍"], 0.1, 1.0),
                气声度=夹(基线["气声度"] + m["气声加"], 0.0, 0.8),
                紧喉度=夹(基线["紧喉度"] + m["紧喉加"], 0.0, 1.0),
                粗糙度=基线["粗糙度"], 送气量=基线["送气量"], 声区=基线["声区"])


# ─────────────── 音频级代理：F0 轨迹（自相关）+ RMS ───────────────
def F0中位(x, sr, 帧长=0.04, 帧移=0.01):
    n = max(1, int((len(x) - int(帧长 * sr)) // int(帧移 * sr) + 1))
    out = []
    for i in range(n):
        seg = x[i * int(帧移 * sr): i * int(帧移 * sr) + int(帧长 * sr)]
        if len(seg) < 100:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
        lo, hi = int(sr / 400), int(sr / 70)
        if hi >= len(ac):
            continue
        段 = ac[lo:hi]
        if np.max(段) <= 0:
            continue
        lag = lo + int(np.argmax(段))
        if ac[lag] / (ac[0] + 1e-9) < 0.3:
            continue
        out.append(sr / lag)
    return float(np.median(out)) if out else 0.0


def main():
    out = {}
    基线 = dict(基频=190.0, 语速=3.2, 幅度=0.45, 气声度=0.15, 紧喉度=0.30,
                粗糙度=0.03, 送气量=0.20, 声区=0.05)
    尺度 = np.array([80.0, 2.0, 0.3, 0.3, 0.3, 0.2, 0.3, 0.3])

    # 四象限情绪点（V, A, P）
    点 = {"高兴": (0.8, 0.8, 0.4), "低落": (-0.8, -0.7, -0.4),
          "愤怒": (-0.8, 0.9, 0.7), "恐惧": (-0.7, 0.8, -0.6),
          "平静": (0.0, 0.0, 0.0)}
    律s = {k: 情感驱动(*v, 基线=基线) for k, v in 点.items()}
    out["T1_情绪点律"] = {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in 律s.items()}

    # T1 换情绪必变：高兴 vs 低落 每参数差 + 归一距
    a, b = 键序(律s["高兴"]), 键序(律s["低落"])
    out["T1_高兴vs低落"] = dict(
        每参数差={k: round(float(a[i] - b[i]), 4) for i, k in enumerate(色)},
        归一距=round(归一距(a, b, 尺度), 4),
        最少变参数数=int(np.sum(np.abs(a - b) > 1e-6)), 门槛="≥6 参数变且归一距>1.0")

    # T2 唤醒单调性：固定 V=0,P=0，A∈[-1,1] 扫 9 点
    As = np.linspace(-1, 1, 9)
    沿A = [情感驱动(0.0, float(A), 0.0, 基线=基线) for A in As]
    单调 = all(沿A[i + 1]["基频"] >= 沿A[i]["基频"] - 1e-9 and
               沿A[i + 1]["幅度"] >= 沿A[i]["幅度"] - 1e-9 and
               沿A[i + 1]["语速"] >= 沿A[i]["语速"] - 1e-9 for i in range(len(沿A) - 1))
    out["T2_唤醒单调"] = dict(
        基频轨=[round(v["基频"], 1) for v in 沿A],
        幅度轨=[round(v["幅度"], 3) for v in 沿A],
        语速轨=[round(v["语速"], 2) for v in 沿A],
        F0_幅度_语速随A单调非减=bool(单调),
        门槛="F0/幅度/语速 随 A 单调非减（证据：唤醒→F0与强度）")

    # T3 效价可分：固定 A=0.6（高唤醒）与 A=-0.6（低唤醒）各扫 V∈[-1,1]
    沿V高 = [情感驱动(float(V), 0.6, 0.0, 基线=基线) for V in np.linspace(-1, 1, 9)]
    沿V低 = [情感驱动(float(V), -0.6, 0.0, 基线=基线) for V in np.linspace(-1, 1, 9)]
    紧高 = [round(v["紧喉度"], 3) for v in 沿V高]
    气低 = [round(v["气声度"], 3) for v in 沿V低]
    # 同 A 下 V− 与 V+ 的 声音质差（紧喉+粗糙 复合）
    def 质(律):
        return 律["紧喉度"] + 律["粗糙度"]
    out["T3_效价可分"] = dict(
        高唤醒_紧喉随V=[round(v, 3) for v in 紧高],
        低唤醒_气声随V=[round(v, 3) for v in 气低],
        V负减V正_声音质差_高唤醒=round(float(质(沿V高[0]) - 质(沿V高[-1])), 4),
        V负减V正_气声差_低唤醒=round(float(沿V低[0]["气声度"] - 沿V低[-1]["气声度"]), 4),
        门槛="同唤醒下 V− 与 V+ 的声音质（紧喉/粗糙/气声）差 >0.10")

    # T4 连续性（vs template 台阶）：5×5 网格 distinct 输出数
    网 = [情感驱动(float(v), float(a), 0.0, 基线=基线)
          for v in np.linspace(-1, 1, 5) for a in np.linspace(-1, 1, 5)]
    distinct = len({tuple(round(x, 6) for x in 键序(l)) for l in 网})
    模 = [模板律(e, 基线) for e in _模板]
    模distinct = len({tuple(round(x, 6) for x in 键序(l)) for l in 模})
    # 中点严格落在两端之间
    低 = 键序(情感驱动(-1, 0, 0, 基线=基线))
    高 = 键序(情感驱动(1, 0, 0, 基线=基线))
    中 = 键序(情感驱动(0, 0, 0, 基线=基线))
    之间 = bool(np.all(np.minimum(低, 高) - 1e-9 <= 中) and np.all(中 <= np.maximum(低, 高) + 1e-9))
    out["T4_连续性vs模板"] = dict(
        提案网格distinct=distinct, 提案网格总数=len(网),
        现行模板distinct=模distinct, 现行模板点数=len(模),
        中点落在两端之间=之间,
        门槛="提案：25 点 → 25 个不同输出且中点严格内插；模板：4 名 → ≤4 输出且无内插轴")

    # T5 must-fail 置零/置乱
    中性 = 键序(情感驱动(0, 0, 0, 基线=基线))
    与情绪点距 = {k: round(归一距(中性, 键序(v), 尺度), 3) for k, v in 律s.items()}
    原 = 键序(情感驱动(0.7, 0.6, 0.3, 基线=基线))
    置乱 = 键序(情感驱动(0.6, 0.7, 0.3, 基线=基线))   # 交换 V/A
    out["T5_mustfail"] = dict(
        置零回到中性基线=bool(np.array_equal(中性, 键序(基线))),
        中性到各情绪归一距=与情绪点距,
        置乱_交换V与A_与原配对不同=bool(not np.array_equal(原, 置乱)),
        置乱归一距=round(归一距(原, 置乱, 尺度), 4),
        门槛="置零⇒恰为中性基线且距各情绪>0.3；置乱⇒输出与原配对不同")

    # T6 确定性（参数层）
    A1 = 键序(情感驱动(0.3, 0.4, -0.2, 基线=基线))
    A2 = 键序(情感驱动(0.3, 0.4, -0.2, 基线=基线))
    out["T6_确定性_参数层"] = dict(两趟逐位同=bool(np.array_equal(A1, A2)))

    # ── T7 音频级实测：v6 后端（写 meeting62，不写 meeting61） ──
    try:
        目录62 = os.path.join(MEET62, "团队", "杏仁核", "产物_发声")
        os.makedirs(目录62, exist_ok=True)
        import 声带后端_v6 as B6  # noqa: E402
        b6 = B6.声带V6(seed=300, 目录=目录62,
                       台账路径=os.path.join(目录62, "台账.jsonl"))
        文本 = "你好，我很高兴。"
        条件 = {"高兴": (0.8, 0.8, 0.4), "低落": (-0.8, -0.7, -0.4),
                "愤怒": (-0.8, 0.9, 0.7), "平静": (0.0, 0.0, 0.0)}
        音频 = {}
        wavs = {}
        for 名, (V, A, P) in 条件.items():
            律 = 情感驱动(V, A, P, 基线=基线)
            音 = {k: float(律[k]) for k in 色}
            音.update(语速=float(律["语速"]), 增益=1.0)
            rec = b6.发声(np.zeros(1024), 文本, 音色=音)
            x = rec["波形"]
            wavs[名] = rec["wav"]
            音频[名] = dict(F0中位=round(F0中位(x, b6.采样率), 1),
                           RMS=round(float(np.sqrt(np.mean(x ** 2))), 4),
                           时长=round(float(len(x) / b6.采样率), 3),
                           实时率=rec["实时率"])
        # 两趟逐位同（同一条件重复）
        音重复 = {k: float(情感驱动(0.8, 0.8, 0.4, 基线=基线)[k]) for k in 色}
        音重复["增益"] = 1.0
        rec1 = b6.发声(np.zeros(1024), 文本, 音色=dict(音重复))
        rec2 = b6.发声(np.zeros(1024), 文本, 音色=dict(音重复))
        out["T7_音频级_v6"] = dict(
            各情绪=音频, wav=wavs,
            F0跨度=round(音频["高兴"]["F0中位"] - 音频["低落"]["F0中位"], 1),
            两趟逐位同=bool(np.array_equal(rec1["波形"], rec2["波形"])),
            门槛="高兴 vs 低落 F0 中位差 ≥10Hz；能量 ≥2 档；两趟逐位同")
    except Exception as e:  # noqa: BLE001
        out["T7_音频级_v6"] = dict(错误=repr(e), 状态="未测")

    p = os.path.join(HERE, "探针结果.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
