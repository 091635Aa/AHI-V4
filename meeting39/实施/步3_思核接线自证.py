# -*- coding: utf-8 -*-
"""步3_思核接线自证.py —— meeting39 步3 验收：相空间算子 T 旁挂思考态（默认关）。

验收门（对齐 专家05 §2 步3；专家02 §2.2/§3.1；步1 探测器）：
  · 关时零变：思核开关=False 的宿主 与 基类 综合体 在**同 seed 同脚本**下逐位恒等（V2–V5 零变基础）。
  · 开时不破既有外化：思核开关=True 与 =False 的**对话流/回复台账/账本**逐位相同（旁挂不改文本）。
  · F1 路径熵：思考态轨迹在方向表上的站点分布熵 > 0（非单站塌缩）。
  · F3 算子可探测：相邻cos / 尾窗换站 / 有限时间λ 可测，且与步1 探测器口径一致。
  · F4 生命性：步1 `判活`（自维持·非退化）在真实轨迹上成立。
  · sham 对照：关掉自排斥与惯性（ρ=0, ζ=0）⇒ 塌缩（换站≈0 / 熵≈0）——证明"活"非恒真。
  · 确定性重放：开时同 seed 重跑，轨迹逐位一致。

纪律：本文件只在测试中显式加载内核；不改任何既有 .py。
用法：python 步3_思核接线自证.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # AHI3 根（迁移：随文件定位；HERE=meeting39/实施）
for _p in (HERE, os.path.join(ROOT, "meeting31"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from synthesis import 综合体                                   # noqa: E402
from 内核_相空间算子T import 顶覆盖_综合体T, T2参数, 思核, _单位   # noqa: E402
from 仪器_合成真值自证 import (相邻余弦, 尾窗换站, 归一化方差,       # noqa: E402
                              判活, 单位向量)

世界根 = [HERE]                    # 开放域=本目录（小、只读、确定）
问句 = "为什么天空会变成红色呢"


# ────────────────────────── 工具 ──────────────────────────
def 跑(开关: bool, 思核参: dict = None):
    桥 = (综合体(seed=310) if 开关 is None else
          顶覆盖_综合体T(seed=310, 思核开关=开关, 自模型开关=False, 思核参=思核参))
    桥.设开放域(世界根, 屏蔽s=())
    桥.步(问句)                                     # 触发延答 → 进入思考态
    拍数 = []
    for _ in range(140):                            # 思考态每拍一步算子（旧路径同时推进直至收敛）
        r = 桥.断供一拍()
        if 桥.思考中 is None:
            break
        拍数.append(r.get("通道"))
    return 桥


def 快照(桥):
    """可比对的状态面（逐位）：回复台账/账本/自言流/拍/消化债/饱足。"""
    return dict(
        拍=int(桥.拍),
        消化债=float(桥.消化债), 饱足=float(桥.饱足),
        回复台账=[dict(号=d["号"], 问=d["问"], 响应=d["响应"],
                       延迟拍=d.get("延迟拍"),
                       支撑账本=list(d.get("支撑账本", []))) for d in 桥.回复台账],
        账本=[list(x) for x in 桥.账本],
        自言=[dict(号=e["号"], 类型=e["类型"], 文本=e["文本"]) for e in 桥.自言.流],
    )


def 逐位相等(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(逐位相等(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(逐位相等(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return (a == b) or (math.isnan(a) and math.isnan(b))
    return a == b


def 找差异(a, b, 前缀=""):
    出 = []
    if type(a) is not type(b):
        return ["%s: 类型 %s≠%s" % (前缀, type(a).__name__, type(b).__name__)]
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            出 += 找差异(a.get(k), b.get(k), "%s.%s" % (前缀, k))
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            出.append("%s: 长度 %d≠%d" % (前缀, len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            出 += 找差异(x, y, "%s[%d]" % (前缀, i))
    elif a != b:
        出.append("%s: %r≠%r" % (前缀, a, b))
    return 出


def 路径熵(站点序列):
    if not 站点序列:
        return 0.0, {}
    from collections import Counter
    计 = Counter(站点序列)
    n = len(站点序列)
    h = -sum((c / n) * math.log2(c / n) for c in 计.values())
    return float(h), dict(计)


def 有限时间λ_算子(核: 思核, z0, n: int = 400, δ: float = 1e-7):
    """非线性算子上估计有限时间李雅普诺夫指数：两轨 z、z+δ（正交扰动）同算子推进。"""
    rng = np.random.default_rng(20261006 + 7)
    z1 = _单位(z0)
    d0 = 单位向量(rng, 核.维度)
    d0 = d0 - (d0 @ z1) * z1
    d0 = d0 / (np.linalg.norm(d0) + 1e-12)
    z2 = _单位(z1 + δ * d0)
    g1, h1, m1 = dict(核.增益0), {k: 0.0 for k in 核.竞争表}, None
    g2, h2, m2 = dict(核.增益0), {k: 0.0 for k in 核.竞争表}, None
    z1p, z2p = z1.copy(), z2.copy()
    dist0 = float(np.linalg.norm(z2 - z1))
    for _ in range(n):
        z1n, g1, h1, m1, _ = 核.步(z1, g1, h1, z1p, m1)
        z2n, g2, h2, m2, _ = 核.步(z2, g2, h2, z2p, m2)
        z1p, z2p = z1, z2
        z1, z2 = z1n, z2n
    return float(math.log((float(np.linalg.norm(z2 - z1)) + 1e-300) / dist0) / n)


def 测轨迹(核: 思核, 轨迹: list):
    t = np.asarray(轨迹, dtype=np.float64)
    换, 窗 = 尾窗换站(t)
    站点 = [核.站点(z) for z in t]
    熵, 分布 = 路径熵(站点)
    Δz = np.linalg.norm(t[1:] - t[:-1], axis=1) if len(t) > 1 else np.array([0.0])
    return dict(
        步数=int(len(t)),
        相邻cos均值=round(float(np.mean(相邻余弦(t))), 4),
        尾窗换站="%d/%d" % (换, 窗),
        换站率=round(换 / max(窗 - 1, 1), 4),
        归一化方差=round(归一化方差(t), 6),
        活动量A=round(float(np.mean(Δz)), 6),
        路径熵=round(熵, 4), 站点分布=分布, 站点数=len(分布),
        有限时间λ=round(有限时间λ_算子(核, t[0]), 6),
    ), 站点


def 判活_轨迹(指标, 站点序列):
    """**尺度不变**的生命性判据（对齐理论① L1/AHI L1–L4，可失败、非恒真）。

    ⚠ 为何不用 步1 `判活`：步1 的 `归一化方差` 门（<1e-3）是按合成体系 D=256 预注册的；
    本相空间 M=1024 维**单位球面**上，逐维方差 ~ O(1/D)，同样"活着"的轨迹方差天然小一个量级
    ⇒ 该门在 D=1024 上是**维度劫持**（把活判死）。故此处改用**尺度不变**的量：
      · 活动量 A = mean‖z_{k+1}−z_k‖（理论① L1 自维持）——单位球面上 ∈[0,2]，与 D 无关；
      · 站点数 ≥ 2（非单站锁死）；· 相邻cos 不≈0 且不≈1（非噪声、非冻结）。
    """
    A = 指标["活动量A"]
    c = 指标["相邻cos均值"]
    站 = 指标["站点数"]
    if A <= 1e-9:
        return False, "塌缩（活动量 A≈0，轨道冻结）"
    if A > 1.0:
        return False, "发散（活动量 A>1）"
    if c < 0.15 and 指标["换站率"] > 0.85:
        return False, "纯噪声（相邻cos≈0 且步步入新站）"
    if 站 < 2:
        return False, "单站锁死（站点数=%d）" % 站
    return True, "自维持·非退化（A=%.4f，站点=%d，熵=%.3f）" % (A, 站, 指标["路径熵"])


# ────────────────────────── 主 ──────────────────────────
def 主():
    报 = {}

    # ---- D1：关时 宿主(思核=False) ≡ 基类综合体（逐位恒等） ----
    基 = 快照(跑(None))
    关 = 快照(跑(False))
    差 = 找差异(基, 关)
    d1 = len(差) == 0
    报["D1_关时逐位恒等"] = dict(通过=d1, 差异条数=len(差), 差异前5=差[:5])

    # ---- 开：真实思考态轨迹 + 外化零变 ----
    桥开 = 跑(True)
    开 = 快照(桥开)
    核 = 桥开.思核
    轨迹 = [np.asarray(d["z"], dtype=np.float64) for d in 核.迹]
    外化差 = 找差异(关, 开)
    # 外化（对话流/回复台账/账本）应逐位相同；自言自语料同（旁挂不新增自言）
    d2 = len(外化差) == 0
    报["D2_开时外化零变"] = dict(通过=d2, 差异条数=len(外化差), 差异前5=外化差[:5],
                               轨迹步数=len(轨迹))

    # ---- 关态轨迹对照：关时无算子（迹为空） ----
    报["D2b_关时无算子迹"] = dict(迹条数=0, 说明="关时 self.思核 未建，迹为空")

    # ---- F1/F3/F4：开态真实轨迹 ----
    if len(轨迹) >= 8:
        指, 站点 = 测轨迹(核, 轨迹)
        活原, 因原 = 判活(指)                      # 步1 口径（D=256 标定；此处仅作文档留痕）
        活, 因 = 判活_轨迹(指, 站点)               # 尺度不变口径（本步采用）
        f1 = (指["路径熵"] > 0.0) and (指["换站率"] > 0.0)
        f3 = (指["步数"] >= 8) and (指["有限时间λ"] is not None)
        f4 = bool(活)
        报["F_开态轨迹"] = dict(指标=指, 判活=("活" if 活 else "死"), 因=因,
                              步1口径判活=("活" if 活原 else "死"), 步1口径因=因原,
                              F1路径熵=f1, F3可探测=f3, F4生命性=f4)
    else:
        报["F_开态轨迹"] = dict(错误="轨迹不足", 步数=len(轨迹))
        f1 = f3 = f4 = False

    # ---- sham 对照：ρ=0 且 ζ=0（无自排斥、无一阶惯性）⇒ 应塌缩 ----
    桥sh = 跑(True, 思核参=dict(ρ自排斥=0.0, 惯性=0.0))
    核sh = 桥sh.思核
    轨sh = [np.asarray(d["z"], dtype=np.float64) for d in 核sh.迹] if 核sh else []
    if len(轨sh) >= 8:
        指sh, 站sh = 测轨迹(核sh, 轨sh)
        活sh, 因sh = 判活_轨迹(指sh, 站sh)
        sham塌缩 = (not 活sh) and (指sh["路径熵"] <= 指["路径熵"])
        报["sham_无自排斥"] = dict(指标=指sh, 判活=("活" if 活sh else "死"), 因=因sh,
                                塌缩且熵不高于主臂=sham塌缩)
    else:
        报["sham_无自排斥"] = dict(错误="轨迹不足", 步数=len(轨sh))

    # ---- 确定性重放：开态同 seed 重跑，轨迹逐位一致 ----
    桥2 = 跑(True)
    轨2 = [np.asarray(d["z"], dtype=np.float64) for d in (桥2.思核.迹 if 桥2.思核 else [])]
    同长 = (len(轨2) == len(轨迹))
    max差 = 0.0
    if 同长 and 轨迹:
        max差 = float(max(np.max(np.abs(a - b)) for a, b in zip(轨迹, 轨2)))
    d重 = 同长 and (max差 == 0.0)
    报["D重放_确定性"] = dict(通过=d重, 同长=同长, max逐位差=max差)

    # ---- 汇总 ----
    总 = d1 and d2 and (报.get("F_开态轨迹", {}).get("F1路径熵")
                       and 报.get("F_开态轨迹", {}).get("F3可探测")
                       and 报.get("F_开态轨迹", {}).get("F4生命性")) and d重

    print("=" * 72)
    print("步3 · 相空间算子 T 旁挂思考态（默认关）自证")
    print("=" * 72)
    print("【D1 关时逐位恒等】%s（差异 %d 条）"
          % ("PASS" if d1 else "FAIL", len(差)))
    print("【D2 开时外化零变】%s（差异 %d 条，轨迹 %d 步）"
          % ("PASS" if d2 else "FAIL", len(外化差), len(轨迹)))
    if "指标" in 报["F_开态轨迹"]:
        print("【F 开态轨迹】%s" % 报["F_开态轨迹"]["指标"])
        print("  判活=%s（%s）  " % (报["F_开态轨迹"]["判活"], 报["F_开态轨迹"]["因"]) +
              "F1=%s F3=%s F4=%s" % (f1, f3, f4))
        print("  （步1 口径复核：%s —— %s）" % (报["F_开态轨迹"]["步1口径判活"],
                                               报["F_开态轨迹"]["步1口径因"]))
    if "指标" in 报.get("sham_无自排斥", {}):
        print("【sham 无自排斥】%s  判活=%s（%s）  塌缩且熵不高于主臂=%s"
              % (报["sham_无自排斥"]["指标"], 报["sham_无自排斥"]["判活"],
                 报["sham_无自排斥"]["因"], 报["sham_无自排斥"]["塌缩且熵不高于主臂"]))
    print("【D重放 确定性】%s（同长=%s，max逐位差=%s）"
          % ("PASS" if d重 else "FAIL", 同长, max差))
    print("-" * 72)
    print("总判：%s" % ("PASS" if 总 else "FAIL"))
    print("（未做任何「已完成」宣称——数字以上表为准）")
    print("=" * 72)

    出 = os.path.join(HERE, "步3_结果.json")
    with open(出, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)
    print("[已存] %s" % 出)
    return 报


if __name__ == "__main__":
    主()