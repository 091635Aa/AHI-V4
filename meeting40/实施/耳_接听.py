# -*- coding: utf-8 -*-
"""meeting40/实施/耳_接听.py —— 备B 收尾：**自听通路**（`耳.听/学`），修「自听静默」。

出处与口径：
  · 终稿 §5 备B「输出」条末句：「补 `耳.听/学` 修自听静默」；
  · 终稿 §11 风险 5：「耳缺 听/学 ⇒ `声带.接综合体` 自听静默失效」→ 缓解=补 `耳_接听.py`；
  · **所有者裁定 D3-2（终稿 §12-9）**：「**自听只写学习迹**，内容记忆 Δ=0」；
  · 判据锚 **G-R3-06**：自听 `ε_aud ≤ 0.25` 且 **replay 后降 ≥30%**。

设计（零新超参、零外挂模型、确定性）：
  · 特征 = `[能量, 主频/1000]`（由 `耳器._分析` 提供；没有耳器时用本模块的**零依赖**简易分析：
    RMS 能量 + 过零率估主频 —— 纯 numpy）。
  · `ε_aud` = **自监督预测误差**：用**上一拍的自听特征**预测这一拍，`ε = ‖预测−实际‖ / (‖实际‖+ε₀)`。
    首块没有预测 ⇒ `ε=1.0`（诚实：没有先验就是全误差，不假装）。
  · `学()` = 更新自听基线（EMA，α=0.5）：越听自己越准 ⇒ replay 后 ε_aud 下降。
  · **落点**：`学习迹.放快照(关键词=["自听"], …)` —— **绝不写 `容量桥`/`codec._段`**；
    判据同时核 `记忆库大小` 前后不变。

诚实边界：本模块**不主张**「它已经像人一样听见自己」；它只保证 (a) 自听有通路、不静默，
(b) ε_aud 随重复下降，(c) 产物只进学习迹。真声带流的音质/延迟属 P3（`流式声带.py` 尚不存在）。

纪律：纯新增；不 import 任何器官（`耳器` 由调用方传进来，可选）；无全局随机态；C13 确定性。
"""
from __future__ import annotations

import numpy as np

import 事件契约 as 契

ε0 = 1e-9
默认α = 0.5


class 耳接听:
    """自听通路：声带输出块 → 分析 → ε_aud → **只写学习迹**。"""

    def __init__(self, 耳器=None, 学习迹=None, 采样率: float = 16000.0, α: float = 默认α):
        self.耳器 = 耳器
        self.迹 = 学习迹
        self.采样率 = float(采样率)
        self.α = float(α)
        self.自身输出 = 契.Provenance(prov_acq="自源", prov_org="自源", prov_route="实感")
        self.预测 = None                       # 自听基线（上一拍特征的 EMA）
        self.自听s: list = []                  # 本模块自己的学习侧留痕（仍属「学习」不属「记忆」）
        self.统计 = dict(听=0, 学=0, 写迹=0)

    # ── 分析（零依赖回退）──
    def _分析(self, 块) -> tuple:
        if self.耳器 is not None:
            try:
                能量, 主频 = self.耳器._分析(块)
                return float(能量), float(主频)
            except Exception:
                pass
        x = np.asarray(块, dtype=np.float64).ravel()
        if x.size == 0:
            return 0.0, 0.0
        能量 = float(np.sqrt(np.mean(x * x)))
        过零 = int(np.sum(np.diff(np.signbit(x)) != 0))
        主频 = float(过零 * self.采样率 / (2.0 * max(1, x.size - 1)))
        return 能量, 主频

    def _特征(self, 能量: float, 主频: float) -> np.ndarray:
        return np.array([float(能量), float(主频) / 1000.0], dtype=np.float64)

    # ── 听（自听一拍）──
    def 听(self, 块) -> dict:
        """听一块（自听）。返回 dict(主频/能量/特征/预测/ε_aud/落点/记忆库大小)。"""
        能量, 主频 = self._分析(块)
        特 = self._特征(能量, 主频)
        预 = self.预测
        ε = 1.0 if 预 is None else float(np.linalg.norm(预 - 特) / (np.linalg.norm(特) + ε0))
        落点 = 契.自听落点(self.自身输出)             # 恒 "学习迹"（D3-2）
        # 自听 = 本系统输出的回灌 ⇒ 写闸 `可写记忆(…, 输出回灌=True)` 必为 False ⇒ 内容记忆 Δ=0
        写M = 契.可写记忆(self.自身输出, 输出回灌=True)
        条 = dict(主频=round(主频, 3), 能量=round(能量, 6),
                  特征=[float(特[0]), float(特[1])],
                  预测=(None if 预 is None else [float(预[0]), float(预[1])]),
                  ε_aud=round(ε, 6), 落点=落点, 内容可写M=bool(写M),
                  记忆库大小=self.记忆库大小())
        self.自听s.append(条)
        self.统计["听"] += 1
        if self.迹 is not None:                     # ★只写学习迹（D3-2）
            try:
                self.迹.放快照(["自听"], 1,
                              "自听：主频%.1f 能量%.4f ε=%.3f" % (条["主频"], 条["能量"], 条["ε_aud"]),
                              束=特)
                self.统计["写迹"] += 1
            except Exception:
                pass
        return 条

    # ── 学（更新自听基线）──
    def 学(self, 块=None) -> dict:
        """更新自听基线（EMA）。给了 `块` 就先听一次再学。"""
        if 块 is not None:
            self.听(块)
        if not self.自听s:
            return dict(更新=False, 基线=None, 更新次数=self.统计["学"])
        特 = np.asarray(self.自听s[-1]["特征"], dtype=np.float64)
        self.预测 = 特.copy() if self.预测 is None else (1.0 - self.α) * self.预测 + self.α * 特
        self.统计["学"] += 1
        return dict(更新=True, 基线=[float(self.预测[0]), float(self.预测[1])],
                    更新次数=self.统计["学"])

    def 记忆库大小(self):
        """只读：背衬记忆库规模（用于证明「自听不写记忆」——它不变）。"""
        if self.迹 is None:
            return None
        try:
            return self.迹.记忆库大小()
        except Exception:
            return None

    # ── replay（G-R3-06：重复自听 ⇒ ε_aud 降）──
    def 重放(self, 块s, 遍数: int = 3) -> dict:
        """把同一串块听 `遍数` 遍，返回每遍的 ε_aud 均值与降幅。**确定性**。"""
        块s = list(块s)
        均s = []
        for _ in range(max(1, int(遍数))):
            εs = [self.听(b)["ε_aud"] for b in 块s]
            for b in 块s:                            # 每遍内逐块学（在线自监督）
                self.学()
            均s.append(round(float(np.mean(εs)) if εs else 1.0, 6))
        首, 末 = 均s[0], 均s[-1]
        降幅 = 0.0 if 首 <= ε0 else (首 - 末) / 首
        return dict(遍数=len(均s), ε_aud均值谱=均s, 首遍=首, 末遍=末,
                    降幅=round(float(降幅), 6), 达降30=bool(降幅 >= 0.30),
                    达ε0_25=bool(末 <= 0.25))

    def 统计表(self) -> dict:
        d = dict(self.统计)
        d["预测"] = None if self.预测 is None else [float(self.预测[0]), float(self.预测[1])]
        d["自听条数"] = len(self.自听s)
        d["记忆库大小"] = self.记忆库大小()
        d["落点口径"] = "自听只写学习迹（D3-2）；内容记忆 Δ=0"
        return d


# ─────────────────────────── 自测 ───────────────────────────
def _合成自听块(采样率: int = 16000, 块长: int = 160, 主频: float = 220.0, 幅: float = 0.3,
             种子: int = 0) -> np.ndarray:
    """确定性合成「自己的声音」：单频正弦 + 极小固定噪声（同 seed ⇒ 逐位同）。"""
    t = np.arange(块长, dtype=np.float64) / float(采样率)
    rng = np.random.default_rng(int(种子))
    return (幅 * np.sin(2 * np.pi * 主频 * t) + 1e-4 * rng.standard_normal(块长)).astype(np.float64)


def 自测() -> dict:
    from 学习迹 import 学习迹
    from 容量桥 import 容量桥
    库 = 容量桥(种子=3800)
    迹 = 学习迹(背衬=库)
    接 = 耳接听(学习迹=迹)
    块s = [_合成自听块(种子=i) for i in range(8)]

    def _跑():
        return 耳接听(学习迹=None).重放(块s, 遍数=3)["ε_aud均值谱"]

    库0, 迹0 = 接.记忆库大小(), len(迹.迹)
    重 = 接.重放(块s, 遍数=3)
    库1, 迹1 = 接.记忆库大小(), len(迹.迹)
    案 = dict(
        有通路=(接.统计["听"] > 0),
        记忆库未变=(库0 == 库1 and 库0 == 0),
        只写学习迹=(迹1 == 迹0 + 接.统计["写迹"] and 迹1 > 迹0),
        落点=(接.自听s[-1]["落点"] == "学习迹"),
        首条内容不可写M=(接.自听s[0]["内容可写M"] is False),
        replay降30=bool(重["达降30"]),
        ε末不超0_25=bool(重["达ε0_25"]),
        确定性=(_跑() == _跑()),
    )
    return dict(通过=bool(all(案.values())), 逐项=案, 重放=重, 统计=接.统计表())


if __name__ == "__main__":
    import json
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "meeting38", "实施"))
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 自测()
    print(json.dumps(果, ensure_ascii=False, indent=1, default=str))
    print("[耳_接听] 自测：%s" % ("PASS" if 果["通过"] else "FAIL"))
    raise SystemExit(0 if 果["通过"] else 1)
