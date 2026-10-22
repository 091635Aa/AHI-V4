# -*- coding: utf-8 -*-
"""meeting64/实施/韵律_v9.py —— 「先会说」第二轮：把**声调域 / 送气 VOT 时长 / 浊声母鼻共鸣 /
擦音部位带宽 / 语速**做成可单维 A/B 的结构旋钮（在 韵律_v8 之上，纯新增，C14）。

为什么不是又一轮参数搜索（meeting63 §8.6 的教训：9 旋钮坐标上升无法迁移）：
本轮的错误归因（`分析_错误归因.py`，18 样本 108 个错误操作）显示 v8.1 的剩余错误里
  · 77.8% 涉及声母，55.6% 涉及声调（声调**从未被测量/优化过**：`逐类归因` 只按声母+鼻尾分组）；
  · 纯声调错 9 条（同声母同韵母只错调），纯段错 42 条；
  · m→l 11 次（浊声母鼻共振缺失 + murmur 只有 40ms）；ing→in 5 次（-ng/-n 部位）；
  · 声母错误率最高：f 100%、p 100%、s 100%、j/q 83%、b/h 67%。
⇒ 这些是**结构线索**（声调域宽、VOT 时长、鼻零极点、擦音带宽、语速），不是 9 个混合旋钮的
   连续坐标；每个都必须**单维 A/B**，看 ASR CER 是否可迁移，再决定是否叠加。

本件只做「旋钮」，不做自动搜索；默认值 = v8.1 逐位同（`参数9` 全中性）。默认路径由
`audit` 断言：调域倍=1、VOT倍=1、浊时长倍=1、浊鼻增益=0、带宽倍=1、语速倍=1 ⇒ 与 v8 同输出。

只读 import meeting61/62/63 既有件；不改 v8 一字。引擎仍用 引擎_v7（本件不动引擎）。
"""
from __future__ import annotations

import copy
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M64 = os.path.dirname(HERE)
AHI4 = os.path.dirname(M64)
M63 = os.path.join(AHI4, "meeting63")
M62 = os.path.join(AHI4, "meeting62")
M61 = os.path.join(AHI4, "meeting61")
M58 = os.path.join(AHI4, "meeting58")
for _p in (HERE, os.path.join(M63, "实施"), os.path.join(M62, "实施"),
           os.path.join(M61, "实施"), os.path.join(M58, "实施")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import 音韵_v5 as Y  # noqa: E402
import 韵律_v7 as H7  # noqa: E402
import 韵律_v8 as H8  # noqa: E402  只读：基线（学习到的 θ）
from 接口契约_角色声线 import 声线档案  # noqa: E402

FS = 44100
帧率 = 100

# —— 单维结构旋钮（默认全中性 = v8.1）——
参数9 = dict(
    调域倍=1.0,        # 声调偏离 ×倍数（v8 的 1.2 半音/五度 → ×此值）。0 调域跨度 ±2.4 半音
    VOT倍=1.0,         # 送气塞 VOT 时长 ×倍数（v8 送气 VOT≈47ms；文献 90–110ms）
    不送气VOT倍=1.0,   # 不送气塞 VOT 时长 ×倍数（v8 下限 12ms）
    浊时长倍=1.0,      # 浊声母（m/n/l/r）murmur 时长 ×倍数（v8 用 self.缩 ⇒ 约 40ms；真人 60–90ms）
    浊鼻增益=0.0,      # 浊声母并联鼻支路增益（0=关；>0 时用 浊鼻极/浊鼻零 造鼻共鸣）
    浊鼻极Hz=250.0,    # 鼻极（m/n/ng 低频鼻共鸣）
    浊鼻零Hz=1000.0,   # 鼻零（m 的反谐振；n 更高）
    擦带宽倍=1.0,      # 擦音/塞擦摩擦带带宽 ×倍数（<1 更窄 ⇒ 部位谱更集中）
    擦时长倍=1.0,      # 擦音（擦方式）时长 ×倍数
    爆带宽倍=1.0,      # 塞音/塞擦爆破带带宽 ×倍数（v8 已收紧到 500/900/700）
    语速倍=1.0,        # 语速 ×倍数（<1 更慢 ⇒ 元音更长；不改音色）
    共振峰单次缩放=0.0,  # 1=消除「韵律×引擎」共振峰双重缩放（T3 实测：实际 F1 ×1.21、F2/F3 ×1.09）
    爆破秒倍=1.0,      # 除阻爆破时长 ×倍数（v8 的 5ms 被 10ms 帧栅格量化成 1 帧；3.0→15ms=2 帧）
    鼻尾借帧=1.0,      # 鼻尾帧从元音段借的比例（v8.1=1.0 借满 ⇒ 鼻韵母元音只剩 6 帧/60ms；0=追加不借）
)


def 造(参数: dict | None = None):
    """→ 韵律工厂(档, 种子) 供 `量具_v9.评多句` 直接使用。"""
    参 = dict(参数9)
    if 参数:
        参.update({k: float(v) for k, v in 参数.items()})
    return lambda 档, 种子=300: 人类韵律_v9(档, 种子, 参数=参)


class 人类韵律_v9(H8.人类韵律_v8):
    """文本 → 帧（v9：声调域/送气 VOT/浊声母鼻共鸣/擦音带宽/语速 可单维 A/B）。"""

    def __init__(self, 档案: 声线档案, 种子: int = 300, 辅音增益: float = 1.0,
                 参数: dict | None = None):
        self.参 = dict(参数9)
        if 参数:
            self.参.update({k: float(v) for k, v in 参数.items()})
        档 = 档案
        if abs(self.参["语速倍"] - 1.0) > 1e-9:
            档 = 档案.派生(语速=float(np.clip(档案.语速 * self.参["语速倍"], 1.5, 7.0)))
        super().__init__(档, 种子, 辅音增益)
        # —— 逐实例声母表：只缩放带宽/时长，中心频率与发音方式不变 ——
        self.声母表 = copy.deepcopy(H8.声母_v8)
        for _k, info in self.声母表.items():
            方 = info.get("方式")
            if info.get("爆破带"):
                fc, bw = info["爆破带"]
                info["爆破带"] = (float(fc), float(bw) * self.参["爆带宽倍"])
            if 方 == "擦":
                if info.get("频带"):
                    fc, bw = info["频带"]
                    info["频带"] = (float(fc), float(bw) * self.参["擦带宽倍"])
                info["时长"] = float(info["时长"]) * self.参["擦时长倍"]
            elif 方 == "塞擦":
                if info.get("频带"):
                    fc, bw = info["频带"]
                    info["频带"] = (float(fc), float(bw) * self.参["擦带宽倍"])
                info["擦时"] = float(info["擦时"]) * self.参["擦时长倍"]

    # ───────────────────────── F0：声调域可缩放 ─────────────────────────
    def _f0(self, 调: int, t: float, j: int, n: int, 标点: str, 重音: bool,
            首加成: float, 微颤相: float = 0.0, 时长: float = 0.12) -> float:
        if 调 == 5:
            半音 = -0.5
        else:
            度 = Y.音高轮廓(调, t)
            半音 = (度 - 3.0) * 1.2 * self.参["调域倍"]
        半音 += self._句调(t, j, n, 标点)
        if j == 0:
            半音 += 首加成
        if 重音:
            半音 += 0.3
        self.漂移 = float(np.clip(self.漂移 + self.rng.normal(0.0, 0.010), -0.10, 0.10))
        半音 += self.漂移 + float(self.rng.normal(0.0, 0.010))
        半音 += self._颤音半音(t * 时长, 时长, 微颤相)
        return self.基准F0 * 2.0 ** (半音 / 12.0) * (1.0 + 0.5 * self.声区)

    # ───────────────────────── 元音：鼻尾借帧可调（v8.1 借满 ⇒ 鼻韵母元音减半） ─────────────────────────
    def _元音(self, s: dict, j: int, n: int, 标点: str, 时长倍: float,
              重音: bool, 微颤相: float, 首加成: float, 末尾: bool, 起始F):
        韵 = s["韵"]
        目s = [np.array([t[0] * self.s[0], t[1] * self.s[1], t[2] * self.s[2]])
              for t in s["目标s"]]
        f4, f5 = H7._高共振峰(韵)
        f4 *= self.s4
        f5 *= self.s5
        关键 = H7._关键点(韵, len(目s))
        时长 = 0.200 * self.缩 * 时长倍 * (
            1.0 + float(H8.参数["复合韵时长倍"]) * max(len(目s) - 1, 0))
        帧数 = max(int(round(时长 * 帧率)), 3)
        过渡帧 = max(1, int(round(float(H8.参数["过渡秒"]) * 帧率)))
        尾 = s["尾"]
        鼻 = (np.asarray(H7.鼻尾目标[尾], dtype=np.float64) *
              np.array([self.s[0], self.s[1], self.s[2]])) if 尾 else None
        N尾 = int(H8.参数["鼻尾帧数"]) if 末尾 else int(H8.参数["鼻尾帧数句中"])
        A尾 = float(H8.参数["鼻尾幅"]) if 末尾 else float(H8.参数["鼻尾幅句中"])
        if 鼻 is not None:
            借 = int(round(N尾 * float(np.clip(self.参["鼻尾借帧"], 0.0, 1.0))))
            帧数 = max(帧数 - 借, 6)   # v8.1：借满 N尾；0=不借（追加，语速略降）
        for f in range(帧数):
            t = f / max(帧数 - 1, 1)
            目 = H7._插目标(目s, t, 关键)
            if 起始F is not None and f < 过渡帧:
                w = H7._平滑步((f + 1) / 过渡帧)
                目 = np.asarray(起始F) * (1.0 - w) + 目 * w
            if 鼻 is not None and t > 0.70:
                w = H7._平滑步((t - 0.70) / 0.30)
                目 = 目 * (1.0 - w) + 鼻 * w
            包 = 1.0
            if f < 2:
                包 *= (f + 1) / 2.0
            if f >= 帧数 - 2:
                包 *= max(0.55, 1.0 - 0.22 * (f - (帧数 - 3)))
            幅 = self.幅度 * 包 * (1.06 if 重音 else 1.0) * float(self.rng.uniform(0.97, 1.03))
            if s["调"] == 5:
                幅 *= 0.85
            f0 = self._f0(s["调"], t, j, n, 标点, 重音, 首加成, 微颤相, 时长)
            self._加(f0, (目[0], 目[1], 目[2], f4, f5), 幅, self.气声, 1, None, 0.0)
        if 鼻 is not None:
            f0l = self._f0(s["调"], 1.0, j, n, 标点, False, 首加成, 微颤相, 时长)
            for k in range(N尾):
                w = (k + 1.0) / N尾
                包 = A尾 * (0.92 - 0.22 * w)
                self._加(f0l * (1.0 - 0.008 * k),
                         (鼻[0], 鼻[1], 鼻[2], f4, f5),
                         self.幅度 * 包, min(self.气声 + 0.02, 0.8), 1, None, 0.0,
                         鼻增益=float(H8.参数["鼻增益"]),
                         鼻极=float(H8.参数["鼻极Hz"]),
                         鼻零=float(H8.参数["鼻零Hz"]))

    # ───────────────────────── 浊段：可加鼻支路 ─────────────────────────
    def _浊段(self, 秒, f0, F起, F终, 幅, 送气=0.05, 鼻=0.0):
        n = max(int(round(秒 * 帧率)), 1)
        起 = np.asarray(F起, dtype=np.float64)
        终 = np.asarray(F终, dtype=np.float64)
        用鼻 = float(鼻) > 1e-9
        for i in range(n):
            t = (i + 1) / n
            w = H7._平滑步(t)
            F = 起 + (终 - 起) * w
            包 = min((i + 1) / 1.5, 1.0)
            self._加(f0, (F[0], F[1], F[2], 3500.0, 4500.0),
                     幅 * 包, self.气声, 1, None, 送气,
                     鼻增益=float(鼻),
                     鼻极=float(self.参["浊鼻极Hz"]) if 用鼻 else 0.0,
                     鼻零=float(self.参["浊鼻零Hz"]) if 用鼻 else 0.0)

    # ───────────────────────── 声母：VOT/浊时长/鼻支路可缩放 ─────────────────────────
    def _声母(self, s: dict, 目标首: np.ndarray, f0起: float):
        声 = s["声"]
        if not 声:
            return None
        info = self.声母表.get(声)
        if info is None:
            return None
        方式 = info["方式"]
        缩 = self._缩辅()
        电平 = float(H8.参数["辅音电平倍"])
        增益 = self.辅音增益 * 电平
        轨 = np.asarray(info["轨迹"], dtype=np.float64)
        目 = np.asarray(目标首, dtype=np.float64)
        终点 = 轨 + (目 - 轨) * float(H8.参数["终点混合"])

        if 方式 == "塞":
            self._静音(info["塞闭"] * 缩)
            self._噪声过渡(float(H8.参数["爆破秒"]) * float(self.参["爆破秒倍"]),
                           info["爆破幅"] * float(H8.参数["爆幅倍"]) * 增益,
                           0, info["爆破带"], 轨, 轨, 送气=0.8 if info["送气"] else 0.3)
            if info["送气"]:
                self._噪声过渡(info["VOT"] * 缩 * float(self.参["VOT倍"]),
                               float(H8.参数["送气电平"]) * 增益, 1, None, 轨, 终点, 送气=1.0)
            else:
                self._噪声过渡(max(info["VOT"] * 缩 * float(self.参["不送气VOT倍"]), 0.012),
                               float(H8.参数["不送气电平"]) * 增益, 1, None, 轨, 终点, 送气=1.0)
            return 终点

        if 方式 == "擦":
            self._噪声过渡(info["时长"] * 缩, info["幅"] * 增益,
                           info.get("噪形", 0), info["频带"], 轨, 终点, 送气=1.0)
            return 终点

        if 方式 == "塞擦":
            self._静音(info["塞闭"] * 缩)
            self._噪声过渡(float(H8.参数["塞擦爆破秒"]) * float(self.参["爆破秒倍"]),
                           info["爆破幅"] * float(H8.参数["爆幅倍"]) * 增益,
                           0, info["爆破带"], 轨, 轨, 送气=0.8)
            self._噪声过渡(info["擦时"] * 缩 * float(H8.参数["塞擦擦时倍"]),
                           info["幅"] * 增益, 0, info["频带"], 轨, 终点, 送气=1.0)
            return 终点

        # 浊声母（m/n/l/r）
        鼻 = float(self.参["浊鼻增益"]) if 声 in ("m", "n") else 0.0
        self._浊段(info["时长"] * self.缩 * float(self.参["浊时长倍"]), f0起, 轨, 终点,
                   self.幅度 * info["幅"] * self.辅音增益 * H7.浊声母增益.get(声, 1.66),
                   送气=info.get("送气", 0.03), 鼻=鼻)
        return 终点

    # ───────────────────────── 音色：可消除共振峰双重缩放 ─────────────────────────
    def 音色(self) -> dict:
        d = super().音色()
        if float(self.参["共振峰单次缩放"]) > 0.5:
            # 韵律侧已把 `目标s` 乘过 self.s；引擎侧再乘一次 ⇒ 双重缩放。
            # 单次缩放 = 让引擎侧因子为 1（实际共振峰回到 `音韵_v5.目标s × 声线缩放`）。
            d["共振峰缩放"] = (1.0, 1.0, 1.0)
        return d


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    import json
    with open(os.path.join(M61, "产物", "声线档案", "EmoCompanion.json"), encoding="utf-8") as f:
        档 = 声线档案.从dict(json.load(f))
    print("v9 冒烟（EmoCompanion）—— 调域倍=1 应与 v8 同：")
    for 倍 in (1.0, 1.5):
        h = 人类韵律_v9(档, 300, 参数=dict(调域倍=倍))
        w, 元, 帧 = h.说("今天的风很轻，适合慢慢地说一会儿话。")
        f0 = np.array(帧["f0"]); src = np.array(帧["声源"])
        st = 12.0 * np.log2(f0[src == 1] / 档.基频)
        print("  调域倍=%.1f 时长=%.2fs 浊帧=%d 半音 %+.2f..%+.2f 跨度=%.2f 峰值=%.3f"
              % (倍, len(w) / FS, int((src == 1).sum()), st.min(), st.max(),
                 st.max() - st.min(), float(np.max(np.abs(w)))))
