# -*- coding: utf-8 -*-
"""meeting60/实施/韵律_44k.py —— 44.1kHz 高保真引擎上的人类化韵律（修复 v2「闷+失真」）。

与 meeting59/韵律_人类.py 的差别：
  1) 采样率 44.1kHz（每帧 10ms=441 样本）；帧里带 F4/F5（五共振峰），不再只有 F1–F3；
  2) 不再对 16k 三共振峰合成器做 ±8dB EQ —— 引擎自带唇辐射/F4/F5 并联支路，高频不再塌；
  3) 共振峰缩放受约束（拟合阶段限制在 0.85–1.10），避免 v2 的 1.18 卡通化；
  4) 保留 v2 的人类化要素：标点停顿/换气、句首重置、句内下倾、句末降调/问句上扬、
     三声变调与不/一、轻声、实词重音、句末延长+尾音、微抖动/微颤、句界音高重置。

只读 import `meeting32/声带.py` 的音韵表；合成本身由 `引擎_44k.py` 完成（纯 numpy/scipy）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEET60 = os.path.dirname(HERE)
AHI4 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AHI4, "meeting32"))
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))

import 声带 as 声带模块  # noqa: E402  只读音韵表
import 引擎_44k as E44  # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402

FS = 44100
帧率 = 100
每帧样本 = FS // 帧率

标点停顿 = {
    "，": 0.18, "、": 0.15, "；": 0.26, "：": 0.30,
    "。": 0.40, "！": 0.44, "？": 0.45, "…": 0.55, "—": 0.45,
    ",": 0.18, ";": 0.26, ":": 0.30, ".": 0.40, "!": 0.44, "?": 0.45,
}
句末类 = set("。！？.!?")
疑问类 = set("？?")
虚词首 = set("upcyeowrd")
实词首 = set("nvatmqsfzilj")

# F4/F5 目标（Hz）：按主元音取近似；缩放随 F3
_F4 = {"a": 3500.0, "o": 3300.0, "e": 3450.0, "i": 3800.0, "u": 3400.0, "v": 3600.0}
_F5 = {"a": 4500.0, "o": 4200.0, "e": 4450.0, "i": 4800.0, "u": 4300.0, "v": 4600.0}


def _高共振峰(韵: str):
    元 = None
    for c in str(韵):
        if c in _F4:
            元 = c
            break
    if 元 is None:
        元 = "e"
    return _F4[元], _F5[元]


def 切短语(文本: str) -> list:
    out = []
    cur = ""
    for ch in str(文本):
        if ch in 标点停顿:
            if cur.strip():
                out.append((cur, ch))
            cur = ""
        elif ch in " \t\r\n":
            continue
        else:
            cur += ch
    if cur.strip():
        out.append((cur, ""))
    return out


def 取音节(文本: str) -> list:
    from pypinyin import lazy_pinyin, Style
    拼 = lazy_pinyin(文本, style=Style.TONE3, neutral_tone_with_five=True,
                    errors=lambda x: [c for c in x if c.isalnum()])
    return [p for p in 拼 if p and p[0].isascii()]


def _变调(音节s: list) -> list:
    n = len(音节s)
    i = 0
    while i < n:
        if 音节s[i][2] == 3:
            j = i
            while j < n and 音节s[j][2] == 3:
                j += 1
            for k in range(i, j - 1):
                音节s[k][2] = 2
            i = j
        else:
            i += 1
    for i in range(n):
        声, 韵, 调, 拼 = 音节s[i]
        基 = 声 + 韵
        if 基 == "bu" and 调 == 4 and i + 1 < n and 音节s[i + 1][2] == 4:
            音节s[i][2] = 2
        if 基 == "yi" and 调 == 1:
            下 = 音节s[i + 1][2] if i + 1 < n else 0
            if 下 == 4:
                音节s[i][2] = 2
            elif 下 in (1, 2, 3):
                音节s[i][2] = 4
    return 音节s


def 词边界(文本: str, 音节数: int) -> list:
    起 = []
    try:
        import jieba.posseg as pseg
        for 词, 旗 in pseg.cut(str(文本)):
            if not 词.strip():
                continue
            实 = bool(旗) and 旗[0].lower() in 实词首
            for k in range(len(词)):
                起.append((k == 0, 实))
    except Exception:  # noqa: BLE001
        return [(False, False)] * 音节数
    if len(起) != 音节数:
        起 = [(k == 0, True) for k in range(音节数)]
    return 起


def 拆音(p: str):
    声, 韵, 调 = 声带模块._拆音节(p)
    return [声, 韵, int(调), p]


class 人类韵律44:
    def __init__(self, 档案: 声线档案, 种子: int = 300):
        self.档案 = 档案
        self.种子 = int(种子)
        self.rng = np.random.default_rng(int(种子) * 7919 + 13)
        self.基准F0 = float(np.clip(档案.基频, 70.0, 400.0))
        self.语速 = float(np.clip(档案.语速, 1.5, 7.0))
        self.声区 = float(np.clip(档案.声区, 0.0, 1.0))
        self.幅度 = float(np.clip(档案.幅度, 0.1, 1.0))
        self.气声 = float(np.clip(档案.气声度, 0.0, 0.8))
        self.紧喉 = float(np.clip(档案.紧喉度, 0.0, 1.0))
        self.粗糙 = float(np.clip(档案.粗糙度, 0.0, 1.0))
        self.送气 = float(np.clip(档案.送气量, 0.0, 1.0))
        self.s = tuple(float(np.clip(x, 0.75, 1.25)) for x in 档案.共振峰缩放)
        self.s4 = float(np.clip(self.s[2], 0.80, 1.20))
        self.s5 = float(np.clip(self.s[2], 0.80, 1.20))
        self.缩 = 3.2 / max(self.语速, 1.5)
        self.漂移 = 0.0
        self.帧 = None

    # ── 帧操作 ──
    def _清(self):
        self.帧 = {k: [] for k in ("f0", "f1", "f2", "f3", "f4", "f5",
                                  "幅", "气声", "声源", "噪带", "送气", "紧喉", "粗糙")}
        self.漂移 = 0.0

    def _加(self, f0, f1, f2, f3, f4, f5, 幅, 气声, 声源, 噪带, 送气, 紧喉, 粗糙):
        f = self.帧
        f["f0"].append(float(f0))
        f["f1"].append(float(f1))
        f["f2"].append(float(f2))
        f["f3"].append(float(f3))
        f["f4"].append(float(f4))
        f["f5"].append(float(f5))
        f["幅"].append(float(幅))
        f["气声"].append(float(气声))
        f["声源"].append(int(声源))
        f["噪带"].append(噪带)
        f["送气"].append(float(送气))
        f["紧喉"].append(float(紧喉))
        f["粗糙"].append(float(粗糙))

    def _目标(self, 韵: str):
        """元音目标序列 (F1,F2,F3,F4,F5)，已按档案缩放。"""
        目标s, 尾 = 声带模块._查韵母(韵)
        f4, f5 = _高共振峰(韵)
        out = []
        for (a, b, c) in 目标s:
            out.append((a * self.s[0], b * self.s[1], c * self.s[2],
                        f4 * self.s4, f5 * self.s5))
        return out, 尾

    def _静音(self, 秒: float):
        for _ in range(max(int(round(秒 * 帧率)), 0)):
            self._加(self.基准F0, 500, 1500, 2500, 3500, 4500, 0.0,
                     self.气声, -1, None, 0.0, self.紧喉, self.粗糙)

    def _换气(self, 秒: float = 0.14, 幅度倍: float = 0.09):
        for i in range(max(int(round(秒 * 帧率)), 1)):
            t = i / max(int(round(秒 * 帧率)) - 1, 1)
            包 = float(np.sin(np.pi * t) ** 0.8)
            self._加(self.基准F0, 700, 1500, 2500, 3500, 4500,
                     self.幅度 * 幅度倍 * 包, self.气声, 0, (1800.0, 2400.0),
                     float(np.clip(self.送气 + 0.45, 0.0, 1.0)), self.紧喉, self.粗糙)

    def _噪声(self, 秒: float, 频带, 送气=None):
        送 = self.送气 if 送气 is None else float(送气)
        for _ in range(max(int(round(秒 * 帧率)), 1)):
            self._加(self.基准F0, 500, 1500, 2500, 3500, 4500,
                     self.幅度 * 0.06, self.气声, 0, 频带, 送, self.紧喉, self.粗糙)

    def _浊(self, 秒: float, f0: float, 目: tuple, 幅度倍: float = 0.5):
        目s = (目[0] * self.s[0], 目[1] * self.s[1], 目[2] * self.s[2],
              目[3] if len(目) > 3 else 3500 * self.s4,
              目[4] if len(目) > 4 else 4500 * self.s5)
        for _ in range(max(int(round(秒 * 帧率)), 1)):
            self._加(f0, 目s[0], 目s[1], 目s[2], 目s[3], 目s[4],
                     self.幅度 * 幅度倍, self.气声, 1, None, 0.0, self.紧喉, self.粗糙)

    # ── 辅音 ──
    def _声母(self, 声: str, 调: int):
        if not 声:
            return
        表 = 声带模块.声母表.get(声)
        if 表 is None:
            return
        方式 = 表[0]
        if 方式 == "塞":
            self._静音(表[2] / 1000.0 * self.缩)
            self._噪声(0.006, (表[1], 700.0))
            送气 = 0.045 if 表[3] else 0.025
            self._噪声(送气 * self.缩, (表[1], 1300.0),
                       float(np.clip(self.送气 + (0.25 if 表[3] else 0.0), 0.0, 1.0)))
        elif 方式 == "擦":
            self._噪声(0.085 * self.缩, 表[1])
        elif 方式 == "塞擦":
            self._静音(0.010 * self.缩)
            self._噪声(0.006, (表[1][0], 700.0))
            self._噪声(0.042 * self.缩, 表[1])
        else:
            目 = 声带模块.浊声母表.get(声, (250.0, 1100.0, 2200.0))
            度0 = 声带模块._折线(声带模块.声调表.get(调, 声带模块.声调表[1]), 0.0)
            f0 = self.基准F0 * 2.0 ** (((度0 - 3.0) * 2.0 * 0.78) / 12.0)
            self._浊(0.05 * self.缩, f0 * (1.0 + 0.5 * self.声区), 目, 0.5)

    def _句调(self, t: float, j: int, n: int, 标点: str) -> float:
        v = 0.55 - 0.060 * j
        if 标点 in 疑问类:
            if j == n - 1:
                v += 2.2 * max(0.0, (t - 0.45) / 0.55)
            elif j == n - 2:
                v += 0.6
        elif 标点 in 句末类:
            if j == n - 1:
                v += -1.90
            elif j == n - 2:
                v += -0.85
        elif 标点 == "，":
            if j == n - 1:
                v += -0.30
        return float(v)

    def _元音(self, 韵: str, 调: int, j: int, n: int, 标点: str,
              时长倍: float, 重音: bool, 微颤相: float, 末尾: bool = False,
              首加成: float = 0.0):
        目标s, 尾 = self._目标(韵)
        时长 = 0.18 * (3.2 / self.语速) * 时长倍
        帧数 = max(int(round(时长 * 帧率)), 2)
        if 调 == 5:
            重音 = False
        for f in range(帧数):
            t = f / max(帧数 - 1, 1)
            数 = len(目标s)
            段 = min(int(t * 数), 数 - 1)
            段内 = (t * 数) - 段
            a = np.asarray(目标s[段], dtype=np.float64)
            b = np.asarray(目标s[min(段 + 1, 数 - 1)], dtype=np.float64)
            ff = a + (b - a) * 段内
            if 尾 and t > 0.7:
                w = (t - 0.7) / 0.3
                ff = ff + (np.array([ff[0], 1000.0, 2200.0, ff[3], ff[4]]) - ff) * w
            if 调 == 5:
                半音 = -0.80
            else:
                度 = 声带模块._折线(声带模块.声调表.get(调, 声带模块.声调表[1]), t)
                半音 = (度 - 3.0) * 2.0 * 0.78
            半音 += self._句调(t, j, n, 标点)
            if j == 0:
                半音 += 首加成
            if 重音:
                半音 += 0.45
            self.漂移 = float(np.clip(self.漂移 + self.rng.normal(0.0, 0.018), -0.18, 0.18))
            半音 += self.漂移 + float(self.rng.normal(0.0, 0.020))
            颤 = 0.0
            if 时长 >= 0.13 and 0.28 <= t <= 0.86:
                颤 = 0.10 * float(np.sin(2.0 * np.pi * 4.8 * t * 时长 + 微颤相))
                半音 += 颤
            f0 = self.基准F0 * 2.0 ** (半音 / 12.0) * (1.0 + 0.5 * self.声区)
            if t < 0.12:
                包 = t / 0.12
            elif t > 0.75:
                包 = 1.0 - 0.55 * (t - 0.75) / 0.25
            else:
                包 = 1.0
            幅 = self.幅度 * 包 * (1.10 if 重音 else 1.0) * float(self.rng.uniform(0.97, 1.03))
            幅 *= (0.80 if 调 == 5 else 1.0)
            self._加(f0, ff[0], ff[1], ff[2], ff[3], ff[4], 幅, self.气声, 1, None,
                     0.0, self.紧喉, self.粗糙)
            末 = (f0, ff, 幅)
        if 末尾:
            f0l, ffl, 幅l = 末
            for k in range(14):
                t = (k + 1) / 14.0
                self._加(f0l * (1.0 - 0.015 * t), ffl[0], ffl[1], ffl[2], ffl[3], ffl[4],
                         幅l * 0.95 * (1.0 - 0.80 * t), min(self.气声 + 0.04, 0.8),
                         1, None, 0.0, self.紧喉, self.粗糙)

    # ── 主流程 ──
    def 建帧(self, 文本: str, 语速倍: float = 1.0) -> dict:
        self._清()
        if abs(float(语速倍) - 1.0) > 1e-9:
            self.语速 = float(np.clip(self.语速 * float(语速倍), 1.5, 7.0))
            self.缩 = 3.2 / max(self.语速, 1.5)
        短语s = 切短语(文本)
        if not 短语s:
            return self.帧
        self._换气(0.12, 0.07)
        for pi, (短语, 标点) in enumerate(短语s):
            音节s = _变调([拆音(p) for p in 取音节(短语)])
            n = len(音节s)
            if n == 0:
                continue
            词 = 词边界(短语, n)
            prev标点 = 短语s[pi - 1][1] if pi > 0 else ""
            首加成 = 1.5 if (pi == 0 or prev标点 in 句末类) else 0.0
            for j, (声, 韵, 调, 拼) in enumerate(音节s):
                末尾 = (j == n - 1)
                词首, 实 = 词[j] if j < len(词) else (False, False)
                重音 = bool(词首 and 实 and 调 != 5)
                时长倍 = 1.0
                if 调 == 5:
                    时长倍 *= 0.55
                elif 调 == 3:
                    时长倍 *= 1.10
                if 末尾:
                    时长倍 *= 1.45
                if 重音:
                    时长倍 *= 1.06
                if j == 0 and 首加成 > 0.0:
                    时长倍 *= 1.10
                时长倍 *= float(self.rng.uniform(0.94, 1.06))
                self._声母(声, 调)
                微颤相 = float(self.rng.uniform(0.0, 2.0 * np.pi))
                self._元音(韵, 调, j, n, 标点, 时长倍, 重音, 微颤相, 末尾, 首加成)
            停 = float(标点停顿.get(标点, 0.0))
            if 标点 in 句末类:
                停 += 0.05
            if 停 > 0:
                self._静音(停)
                if 停 >= 0.30 and pi < len(短语s) - 1:
                    self._换气(0.13, 0.09)
        return self.帧

    def 说(self, 文本: str, 语速倍: float = 1.0, EQ=None) -> tuple:
        帧 = self.建帧(文本, 语速倍)
        if not 帧["f0"]:
            return np.zeros(0), dict(样本数=0), 帧
        波形, 元 = E44.合成(帧, 音色=self.音色(), 种子=self.种子)
        if EQ is not None:
            波形 = 应用EQ44(波形, EQ)
        return 波形, 元, 帧

    def 音色(self) -> dict:
        return dict(
            幅度=self.幅度, 气声度=self.气声, 紧喉度=self.紧喉, 粗糙度=self.粗糙,
            送气量=self.送气, 声区=self.声区,
            共振峰缩放=tuple(self.s), 带宽缩放=1.0,
            谱倾斜=float(np.clip(self.档案.谱倾斜, -0.5, 0.5)),
            增益=float(np.clip(self.档案.增益, 0.2, 1.5)))


# ─────────────────── 裸声（组B：无人类韵律，作基线） ───────────────────
class 裸声44:
    """最简逐字连读：无停顿/换气/句调/变调/重音，只保留四声与语速，作组B基线。"""

    def __init__(self, 档案: 声线档案, 种子: int = 300):
        self.a = 档案
        self.种子 = int(种子)
        self.rng = np.random.default_rng(int(种子) * 104729 + 7)
        self.基频 = float(np.clip(档案.基频, 70.0, 400.0))
        self.语速 = float(np.clip(档案.语速, 1.5, 7.0))
        self.声区 = float(np.clip(档案.声区, 0.0, 1.0))
        self.幅度 = float(np.clip(档案.幅度, 0.1, 1.0))
        self.气声 = float(np.clip(档案.气声度, 0.0, 0.8))
        self.紧喉 = float(np.clip(档案.紧喉度, 0.0, 1.0))
        self.粗糙 = float(np.clip(档案.粗糙度, 0.0, 1.0))
        self.送气 = float(np.clip(档案.送气量, 0.0, 1.0))
        self.s = tuple(float(np.clip(x, 0.75, 1.25)) for x in 档案.共振峰缩放)
        self.s4 = float(np.clip(self.s[2], 0.80, 1.20))
        self.s5 = float(np.clip(self.s[2], 0.80, 1.20))
        self.缩 = 3.2 / max(self.语速, 1.5)

    def _帧(self, 文本: str) -> dict:
        帧 = {k: [] for k in ("f0", "f1", "f2", "f3", "f4", "f5",
                               "幅", "气声", "声源", "噪带", "送气", "紧喉", "粗糙")}

        def 加(f0, f1, f2, f3, f4, f5, 幅, 气声, 声源, 噪带, 送气):
            帧["f0"].append(float(f0)); 帧["f1"].append(float(f1))
            帧["f2"].append(float(f2)); 帧["f3"].append(float(f3))
            帧["f4"].append(float(f4)); 帧["f5"].append(float(f5))
            帧["幅"].append(float(幅)); 帧["气声"].append(float(气声))
            帧["声源"].append(int(声源)); 帧["噪带"].append(噪带)
            帧["送气"].append(float(送气)); 帧["紧喉"].append(self.紧喉)
            帧["粗糙"].append(self.粗糙)

        音节s = [拆音(p) for p in 取音节(文本)]
        for 声, 韵, 调, 拼 in 音节s:
            表 = 声带模块.声母表.get(声) if 声 else None
            if 表:
                方式 = 表[0]
                if 方式 == "塞":
                    for _ in range(max(int(round(表[2] / 1000.0 * self.缩 * 帧率)), 1)):
                        加(self.基频, 500, 1500, 2500, 3500, 4500, 0.0, self.气声, -1, None, 0.0)
                    for _ in range(max(int(round(0.02 * self.缩 * 帧率)), 1)):
                        加(self.基频, 500, 1500, 2500, 3500, 4500, self.幅度 * 0.42,
                          self.气声, 0, (表[1], 1000.0), self.送气)
                elif 方式 in ("擦", "塞擦"):
                    频 = 表[1] if 方式 == "擦" else 表[1]
                    for _ in range(max(int(round(0.05 * self.缩 * 帧率)), 1)):
                        加(self.基频, 500, 1500, 2500, 3500, 4500, self.幅度 * 0.42,
                          self.气声, 0, 频, self.送气)
                else:
                    目 = 声带模块.浊声母表.get(声, (250, 1100, 2200))
                    for _ in range(max(int(round(0.05 * self.缩 * 帧率)), 1)):
                        加(self.基频, 目[0] * self.s[0], 目[1] * self.s[1], 目[2] * self.s[2],
                          3500 * self.s4, 4500 * self.s5, self.幅度 * 0.5, self.气声, 1, None, 0.0)
            目标s, 尾 = 声带模块._查韵母(韵)
            f4, f5 = _高共振峰(韵)
            目 = [(a * self.s[0], b * self.s[1], c * self.s[2], f4 * self.s4, f5 * self.s5)
                 for (a, b, c) in 目标s]
            帧数 = max(int(round(0.18 * (3.2 / self.语速) * 帧率)), 2)
            for f in range(帧数):
                t = f / max(帧数 - 1, 1)
                数 = len(目); 段 = min(int(t * 数), 数 - 1); 段内 = (t * 数) - 段
                a = np.asarray(目[段]); b = np.asarray(目[min(段 + 1, 数 - 1)])
                ff = a + (b - a) * 段内
                度 = 声带模块._折线(声带模块.声调表.get(调, 声带模块.声调表[1]), t)
                半音 = (度 - 3.0) * 2.0
                f0 = self.基频 * 2.0 ** (半音 / 12.0) * (1.0 + 0.5 * self.声区)
                包 = min(t / 0.15, 1.0) if t < 0.15 else (min((1.0 - t) / 0.25, 1.0) if t > 0.75 else 1.0)
                加(f0, ff[0], ff[1], ff[2], ff[3], ff[4], self.幅度 * 包, self.气声, 1, None, 0.0)
        return 帧

    def 说(self, 文本: str) -> tuple:
        帧 = self._帧(文本)
        if not 帧["f0"]:
            return np.zeros(0), dict(样本数=0), 帧
        音 = dict(幅度=self.幅度, 气声度=self.气声, 紧喉度=self.紧喉, 粗糙度=self.粗糙,
                送气量=self.送气, 声区=self.声区, 共振峰缩放=tuple(self.s),
                带宽缩放=1.0, 谱倾斜=float(np.clip(self.a.谱倾斜, -0.5, 0.5)),
                增益=float(np.clip(self.a.增益, 0.2, 1.5)))
        波形, 元 = E44.合成(帧, 音色=音, 种子=self.种子)
        return 波形, 元, 帧


# ─────────────────── 轻度 EQ（组D：250–8k 六带，≤±9dB；<250Hz/＞6.5k 衰减回 0） ───────────────────
带中心7 = (125.0, 354.0, 707.0, 1414.0, 2449.0, 3464.0, 5657.0)
带中心 = 带中心7[1:]          # 六带：354/707/1414/2449/3464/5657（v3 默认，不含 0–250 隆隆带）


def 响应44(增益dB, nfft: int = 8192) -> np.ndarray:
    g = np.asarray(增益dB, dtype=np.float64)
    if g.size == 6:
        lc = np.log2(np.asarray(带中心))
    elif g.size == 7:
        lc = np.log2(np.asarray(带中心7))
    else:
        raise ValueError("EQ 增益长度须为 6 或 7（得到 %d）" % g.size)
    f = np.fft.rfftfreq(nfft, 1.0 / FS)
    lf = np.log2(np.maximum(f, 20.0))
    db = np.interp(lf, lc, g)
    lo = f < 250.0                     # <250Hz 渐回 0dB（参考低端隆隆不参与拟合）
    if np.any(lo):
        db[lo] = db[lo] * np.clip((f[lo] - 60.0) / 190.0, 0.0, 1.0)
    hi = f > 6500.0                    # >6.5k 平滑回 0dB（保留引擎自然高频/齿音）
    if np.any(hi):
        db[hi] = db[hi] * np.clip(1.0 - (f[hi] - 6500.0) / 4000.0, 0.0, 1.0)
    return 10.0 ** (db / 20.0)


def 应用EQ44(x: np.ndarray, 增益dB) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    填 = int(0.1 * FS)
    y = np.concatenate([np.zeros(填), x, np.zeros(填)])
    nfft = 1 << int(np.ceil(np.log2(len(y))))
    out = np.fft.irfft(np.fft.rfft(y, nfft) * 响应44(增益dB, nfft), nfft)[:len(y)]
    return out[填:填 + x.size]


def 说(文本: str, 档案: 声线档案, 种子: int = 300, EQ=None) -> tuple:
    return 人类韵律44(档案, 种子).说(文本, EQ=EQ)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    import json
    档 = 声线档案(名="自检", 基频=220.0)
    结 = {}
    for 名 in ("裸声", "人类韵律"):
        器 = 裸声44(档, 300) if 名 == "裸声" else 人类韵律44(档, 300)
        w, 元, 帧 = 器.说("你好，今天天气不错，我们一起出去走走吧。")
        结[名] = dict(时长=round(len(w) / FS, 2), 帧数=len(帧["f0"]),
                      峰值=round(float(np.max(np.abs(w))), 3))
        print(名, 结[名])
