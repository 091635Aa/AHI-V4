# -*- coding: utf-8 -*-
"""meeting61/实施/韵律_v6.py —— 普通话语流（v6：声道滤波架构修正）。

v6 相对 v5 的唯一改动：合成引擎换成 引擎_v6（浊音道 Klatt 串联 + F4/F5 高通并联支路）。
韵律/音韵/协同发音部分与 韵律_v5 逐位同（辅音表听感不回退，C14）。

v5 原有的底层修改（保留）：
  1) 音韵层换成 音韵_v5（y/w 零声母、舌尖元音 ɿ/ʅ、全部复合/鼻韵母目标）；
  2) **CV 协同发音**：元音开头 30ms 从声母部位轨迹（locus）滑到元音目标；
     塞音 VOT / 喉擦 h 的噪声按后续元音 F1–F3 成形（引擎 噪形=1）；
  3) 辅音按发音方式给足时长与电平：塞音「塞闭→爆破→VOT」、擦音 50–60ms、
     塞擦「塞闭→爆破→擦」、浊声母 35–50ms；不再统一 0.06 幅度；
  4) 鼻韵尾：韵腹末 30% 滑向鼻尾目标（-n F2≈1600 / -ng F2≈1100）再补 20ms 鼻音；
  5) 停顿收紧（逗号 75ms / 句号 180ms / 句首换气 35ms），句内不再有 0.8s 空档。

只依赖 音韵_v5 + 引擎_v6（纯 numpy/scipy，无外挂模型）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEET61 = os.path.dirname(HERE)
AHI4 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))

import 音韵_v5 as Y  # noqa: E402
import 引擎_v6 as E5  # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402

FS = 44100
帧率 = 100
每帧样本 = FS // 帧率

标点停顿 = {
    "，": 0.075, "、": 0.055, "；": 0.100, "：": 0.100,
    "。": 0.180, "！": 0.190, "？": 0.200, "…": 0.260, "—": 0.200,
    ",": 0.075, ";": 0.100, ":": 0.100, ".": 0.180, "!": 0.190, "?": 0.200,
}
句末类 = set("。！？.!?")
疑问类 = set("？?")
虚词首 = set("upcyeowrd")
实词首 = set("nvatmqsfzilj")

# 高共振峰（F4/F5）按主元音取近似
_F4 = {"a": 3500.0, "o": 3300.0, "e": 3450.0, "i": 3800.0, "u": 3400.0,
       "v": 3600.0, "er": 3400.0, "ii": 3500.0, "iii": 3300.0}
_F5 = {"a": 4500.0, "o": 4200.0, "e": 4450.0, "i": 4800.0, "u": 4300.0,
       "v": 4600.0, "er": 4300.0, "ii": 4500.0, "iii": 4200.0}

中性F = (500.0, 1500.0, 2500.0)
鼻尾目标 = Y.鼻尾目标

# 浊声母（m/n/l/r 鼻音·边音·近音）电平补偿：
# 引擎_v6 的浊音道换成 Klatt 串联后，低 F1（鼻音 250–500Hz）的谐振峰增益 = F1/BW1
# 只有 3.6–7，而元音 /a/ 的 F1 是 13 ⇒ 同一帧下浊声母段比 v5 低 4.4dB（元音段只差 0.06dB）。
# 辅音层听感不许回归，故按**逐音位实测**把浊声母补偿回 v5 电平（残余 <1dB）。
浊声母增益 = {"m": 1.63, "n": 1.84, "l": 2.17, "r": 1.85}


def _平滑步(x: float) -> float:
    x = min(max(float(x), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def 切短语(文本: str) -> list:
    out, cur = [], ""
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


def _变调(音节s: list) -> list:
    n = len(音节s)
    i = 0
    while i < n:
        if 音节s[i]["调"] == 3:
            j = i
            while j < n and 音节s[j]["调"] == 3:
                j += 1
            for k in range(i, j - 1):
                音节s[k]["调"] = 2
            i = j
        else:
            i += 1
    for i in range(n):
        s = 音节s[i]
        基 = s["声"] + s["韵"]
        if 基 == "bu" and s["调"] == 4 and i + 1 < n and 音节s[i + 1]["调"] == 4:
            s["调"] = 2
        if 基 == "i" and s["调"] == 1:
            下 = 音节s[i + 1]["调"] if i + 1 < n else 0
            if 下 == 4:
                s["调"] = 2
            elif 下 in (1, 2, 3):
                s["调"] = 4
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


def _高共振峰(韵: str):
    元 = None
    for c in str(韵):
        if c in _F4:
            元 = c
            break
    if 元 is None:
        元 = "e"
    return _F4[元], _F5[元]


def _关键点(韵: str, n: int) -> list:
    """韵腹/韵尾目标的时间关键点（0..1）——介音短、韵尾在后段。"""
    if n <= 1:
        return [0.0]
    if n == 2:
        if 韵[:1] in ("i", "u", "v") and 韵 not in ("ai", "ei", "ao", "ou"):
            return [0.0, 0.30]
        return [0.0, 0.75]
    if 韵[:1] in ("i", "u", "v"):
        return [0.0, 0.36, 0.80]
    return [0.0, 0.45, 0.85]


def _插目标(目标s, t: float, 键s) -> np.ndarray:
    if len(目标s) == 1:
        return np.asarray(目标s[0], dtype=np.float64)
    if t <= 键s[0]:
        return np.asarray(目标s[0], dtype=np.float64)
    for k in range(len(键s) - 1):
        if t <= 键s[k + 1]:
            w = _平滑步((t - 键s[k]) / max(键s[k + 1] - 键s[k], 1e-6))
            a = np.asarray(目标s[k], dtype=np.float64)
            b = np.asarray(目标s[k + 1], dtype=np.float64)
            return a + (b - a) * w
    return np.asarray(目标s[-1], dtype=np.float64)


class 人类韵律_v5:
    """文本 → 帧（音素级协同发音）→ 44.1kHz 波形。"""

    def __init__(self, 档案: 声线档案, 种子: int = 300, 辅音增益: float = 1.0):
        self.档案 = 档案
        self.种子 = int(种子)
        self.辅音增益 = float(np.clip(辅音增益, 0.4, 3.0))
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

    # ── 帧容器 ──
    def _清(self):
        self.帧 = {k: [] for k in ("f0", "f1", "f2", "f3", "f4", "f5", "幅",
                                  "气声", "声源", "噪带", "送气", "紧喉", "粗糙", "噪形")}
        self.漂移 = 0.0

    def _加(self, f0, F, 幅, 气声, 声源, 噪带, 送气, 噪形=0):
        f = self.帧
        f["f0"].append(float(f0))
        f["f1"].append(float(F[0]))
        f["f2"].append(float(F[1]))
        f["f3"].append(float(F[2]))
        f["f4"].append(float(F[3] if len(F) > 3 else 3500.0))
        f["f5"].append(float(F[4] if len(F) > 4 else 4500.0))
        f["幅"].append(float(幅))
        f["气声"].append(float(气声))
        f["声源"].append(int(声源))
        f["噪带"].append(噪带)
        f["送气"].append(float(送气))
        f["紧喉"].append(float(self.紧喉))
        f["粗糙"].append(float(self.粗糙))
        f["噪形"].append(int(噪形))

    def _静音(self, 秒: float):
        for _ in range(max(int(round(秒 * 帧率)), 0)):
            self._加(self.基准F0, 中性F, 0.0, self.气声, -1, None, 0.0)

    def _换气(self, 秒: float = 0.035, 幅倍: float = 0.10):
        n = max(int(round(秒 * 帧率)), 1)
        for i in range(n):
            t = (i + 1) / n
            包 = float(np.sin(np.pi * t) ** 0.7)
            self._加(self.基准F0, 中性F, self.幅度 * 幅倍 * 包, self.气声, 0,
                     None, 0.9, 噪形=1)

    # ── 辅音 ──
    def _噪声过渡(self, 秒, 幅, 噪形, 频带, F起, F终, 送气=0.9):
        n = max(int(round(秒 * 帧率)), 1)
        起 = np.asarray(F起, dtype=np.float64) if F起 is not None else None
        终 = np.asarray(F终, dtype=np.float64) if F终 is not None else None
        for i in range(n):
            t = (i + 1) / n
            if 起 is None:
                F = 中性F
            else:
                w = _平滑步(min(t / 0.85, 1.0))
                F = 起 + ((终 if 终 is not None else 起) - 起) * w
            包 = 1.0 if n == 1 else max(
                min((i + 1) / 1.6, 1.0) * min((n - i) / 1.6, 1.0), 0.12)
            self._加(self.基准F0, (F[0], F[1], F[2], 3500.0, 4500.0),
                     幅 * 包, self.气声, 0, 频带, 送气, 噪形=噪形)

    def _浊段(self, 秒, f0, F起, F终, 幅, 送气=0.05):
        n = max(int(round(秒 * 帧率)), 1)
        起 = np.asarray(F起, dtype=np.float64)
        终 = np.asarray(F终, dtype=np.float64)
        for i in range(n):
            t = (i + 1) / n
            w = _平滑步(t)
            F = 起 + (终 - 起) * w
            包 = min((i + 1) / 1.5, 1.0)
            self._加(f0, (F[0], F[1], F[2], 3500.0, 4500.0),
                     幅 * 包, self.气声, 1, None, 送气)

    def _声母(self, s: dict, 目标首: np.ndarray, f0起: float):
        """按发音方式生成辅音帧；返回元音起始的共振峰（CV 过渡起点）。"""
        声 = s["声"]
        if not 声:
            return None
        info = Y.声母.get(声)
        if info is None:
            return None
        方式 = info["方式"]
        缩 = self.缩
        增益 = self.辅音增益
        轨 = np.asarray(info["轨迹"], dtype=np.float64)
        目 = np.asarray(目标首, dtype=np.float64)
        终点 = 轨 + (目 - 轨) * 0.45

        if 方式 == "塞":
            self._静音(info["塞闭"] * 缩)
            self._噪声过渡(0.008, info["爆破幅"] * 增益, 0, info["爆破带"],
                           轨, 轨, 送气=0.8 if info["送气"] else 0.3)
            if info["送气"]:
                self._噪声过渡(info["VOT"] * 缩, 0.80 * 增益, 1, None, 轨, 终点, 送气=1.0)
            else:
                self._噪声过渡(max(info["VOT"] * 缩, 0.012), 0.32 * 增益, 1, None,
                               轨, 终点, 送气=1.0)
            return 终点

        if 方式 == "擦":
            self._噪声过渡(info["时长"] * 缩, info["幅"] * 增益,
                           info.get("噪形", 0), info["频带"], 轨, 终点, 送气=1.0)
            return 终点

        if 方式 == "塞擦":
            self._静音(info["塞闭"] * 缩)
            self._噪声过渡(0.006, info["爆破幅"] * 增益, 0, info["爆破带"],
                           轨, 轨, 送气=0.8)
            self._噪声过渡(info["擦时"] * 缩, info["幅"] * 增益, 0, info["频带"],
                           轨, 终点, 送气=1.0)
            return 终点

        # 浊声母 m/n/l/r
        self._浊段(info["时长"] * 缩, f0起, 轨, 终点,
                   self.幅度 * info["幅"] * 增益 * 浊声母增益.get(声, 1.66),
                   送气=info.get("送气", 0.03))
        return 终点

    # ── 元音 ──
    def _f0(self, 调: int, t: float, j: int, n: int, 标点: str, 重音: bool,
            首加成: float, 微颤相: float = 0.0, 时长: float = 0.12) -> float:
        if 调 == 5:
            半音 = -0.5
        else:
            度 = Y.音高轮廓(调, t)
            半音 = (度 - 3.0) * 1.2
        半音 += self._句调(t, j, n, 标点)
        if j == 0:
            半音 += 首加成
        if 重音:
            半音 += 0.3
        self.漂移 = float(np.clip(self.漂移 + self.rng.normal(0.0, 0.010), -0.10, 0.10))
        半音 += self.漂移 + float(self.rng.normal(0.0, 0.010))
        if 时长 >= 0.13 and 0.28 <= t <= 0.86:
            半音 += 0.07 * float(np.sin(2.0 * np.pi * 4.8 * t * 时长 + 微颤相))
        return self.基准F0 * 2.0 ** (半音 / 12.0) * (1.0 + 0.5 * self.声区)

    def _句调(self, t: float, j: int, n: int, 标点: str) -> float:
        v = 0.30 - 0.030 * j
        if 标点 in 疑问类:
            if j == n - 1:
                v += 1.5 * max(0.0, (t - 0.45) / 0.55)
            elif j == n - 2:
                v += 0.35
        elif 标点 in 句末类:
            if j == n - 1:
                v += -1.10
            elif j == n - 2:
                v += -0.45
        elif 标点 == "，":
            if j == n - 1:
                v += -0.15
        return float(v)

    def _元音(self, s: dict, j: int, n: int, 标点: str, 时长倍: float,
              重音: bool, 微颤相: float, 首加成: float, 末尾: bool,
              起始F):
        韵 = s["韵"]
        目s = [np.array([t[0] * self.s[0], t[1] * self.s[1], t[2] * self.s[2]])
              for t in s["目标s"]]
        f4, f5 = _高共振峰(韵)
        f4 *= self.s4
        f5 *= self.s5
        关键 = _关键点(韵, len(目s))
        时长 = 0.200 * self.缩 * 时长倍
        帧数 = max(int(round(时长 * 帧率)), 3)
        过渡帧 = max(1, int(round(0.030 * 帧率)))
        尾 = s["尾"]
        鼻 = (np.asarray(鼻尾目标[尾], dtype=np.float64) *
             np.array([self.s[0], self.s[1], self.s[2]])) if 尾 else None

        for f in range(帧数):
            t = f / max(帧数 - 1, 1)
            目 = _插目标(目s, t, 关键)
            if 起始F is not None and f < 过渡帧:
                w = _平滑步((f + 1) / 过渡帧)
                目 = np.asarray(起始F) * (1.0 - w) + 目 * w
            if 鼻 is not None and t > 0.70:
                w = _平滑步((t - 0.70) / 0.30)
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

        if 鼻 is not None and 末尾:
            f0l = self._f0(s["调"], 1.0, j, n, 标点, False, 首加成, 微颤相, 时长)
            for k in range(2):
                包 = 0.30 * (1.0 - 0.35 * k)
                self._加(f0l * (1.0 - 0.008 * k),
                         (鼻[0], 鼻[1], 鼻[2], f4, f5),
                         self.幅度 * 包, min(self.气声 + 0.02, 0.8), 1, None, 0.0)

    # ── 主流程 ──
    def 建帧(self, 文本: str, 语速倍: float = 1.0) -> dict:
        self._清()
        if abs(float(语速倍) - 1.0) > 1e-9:
            self.语速 = float(np.clip(self.语速 * float(语速倍), 1.5, 7.0))
            self.缩 = 3.2 / max(self.语速, 1.5)
        短语s = 切短语(文本)
        if not 短语s:
            return self.帧
        self._换气(0.035, 0.10)
        for pi, (短语, 标点) in enumerate(短语s):
            音节s = _变调(Y.音节列表(短语))
            n = len(音节s)
            if n == 0:
                continue
            词 = 词边界(短语, n)
            prev标点 = 短语s[pi - 1][1] if pi > 0 else ""
            首加成 = 0.6 if (pi == 0 or prev标点 in 句末类) else 0.0
            for j, s in enumerate(音节s):
                末尾 = (j == n - 1)
                词首, 实 = 词[j] if j < len(词) else (False, False)
                重音 = bool(词首 and 实 and s["调"] != 5)
                时长倍 = 1.0
                if s["调"] == 5:
                    时长倍 *= 0.60
                elif s["调"] == 3:
                    时长倍 *= 1.15
                if 末尾:
                    时长倍 *= 1.18
                if 重音:
                    时长倍 *= 1.05
                if j == 0 and 首加成 > 0.0:
                    时长倍 *= 1.05
                时长倍 *= float(self.rng.uniform(0.95, 1.05))
                目标首 = np.array([s["目标s"][0][0] * self.s[0],
                                  s["目标s"][0][1] * self.s[1],
                                  s["目标s"][0][2] * self.s[2]])
                f0起 = self._f0(s["调"], 0.0, j, n, 标点, 重音, 首加成)
                起始F = self._声母(s, 目标首, f0起)
                微颤相 = float(self.rng.uniform(0.0, 2.0 * np.pi))
                self._元音(s, j, n, 标点, 时长倍, 重音, 微颤相, 首加成, 末尾, 起始F)
            停 = float(标点停顿.get(标点, 0.0))
            if 停 > 0:
                self._静音(停)
        return self.帧

    def 说(self, 文本: str, 语速倍: float = 1.0, EQ=None) -> tuple:
        帧 = self.建帧(文本, 语速倍)
        if not 帧["f0"]:
            return np.zeros(0), dict(样本数=0), 帧
        波形, 元 = E5.合成(帧, 音色=self.音色(), 种子=self.种子)
        return 波形, 元, 帧

    def 说音节(self, 拼音: str, 语速倍: float = 1.0) -> tuple:
        """单音节合成（供辅音/元音/四声测试表用，绕开汉字转拼音）。"""
        self._清()
        if abs(float(语速倍) - 1.0) > 1e-9:
            self.语速 = float(np.clip(self.语速 * float(语速倍), 1.5, 7.0))
            self.缩 = 3.2 / max(self.语速, 1.5)
        s = Y.解析音节(拼音)
        self._换气(0.030, 0.07)
        时长倍 = 0.60 if s["调"] == 5 else (1.15 if s["调"] == 3 else 1.0)
        目标首 = np.array([s["目标s"][0][0] * self.s[0],
                          s["目标s"][0][1] * self.s[1],
                          s["目标s"][0][2] * self.s[2]])
        f0起 = self._f0(s["调"], 0.0, 0, 1, "", False, 0.0)
        起始F = self._声母(s, 目标首, f0起)
        self._元音(s, 0, 1, "", 时长倍, False, 0.0, 0.0, True, 起始F)
        波, 元 = E5.合成(self.帧, 音色=self.音色(), 种子=self.种子)
        return 波, 元, self.帧

    def 音色(self) -> dict:
        return dict(
            幅度=self.幅度, 气声度=self.气声, 紧喉度=self.紧喉, 粗糙度=self.粗糙,
            送气量=self.送气, 声区=self.声区,
            共振峰缩放=tuple(self.s), 带宽缩放=1.0,
            谱倾斜=float(np.clip(self.档案.谱倾斜, -0.5, 0.5)),
            增益=float(np.clip(self.档案.增益, 0.2, 1.5)))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    import json
    import time
    with open(os.path.join(MEET61, "产物", "声线档案", "EmoCompanion.json"), encoding="utf-8") as f:
        档 = 声线档案.从dict(json.load(f))
    for 句 in ("你好，我是EmoCompanion。", "今天的风很轻，适合慢慢地说一会儿话。",
               "语言不只是声音，它把一个人的温度，送到另一个人的心里。"):
        t0 = time.perf_counter()
        w, 元, 帧 = 人类韵律_v5(档, 300).说(句)
        音节数 = len(Y.音节列表(句))
        print("%-28s 时长 %.2fs 音节 %d 音节率 %.2f/s 帧 %d 峰值 %.3f 用时 %.2fs"
              % (句[:22], len(w) / FS, 音节数, 音节数 / (len(w) / FS),
                 len(帧["f0"]), float(np.max(np.abs(w))), time.perf_counter() - t0))
