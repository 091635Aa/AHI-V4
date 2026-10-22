# -*- coding: utf-8 -*-
"""meeting59/实施/韵律_人类.py —— 「像人类一样说话」的韵律层（原生声带之上，纯 DSP）。

相对 meeting32 原生 `音节帧`（逐字四声 + 整句机械连读）补的人类化要素：
  1) 标点分短语：逗号/顿号/分号/句号/问号/省略号 各有停顿；句界重置语调；
  2) 换气：长停顿后加一段低幅带通噪声（吸气声，确定性种子）；
  3) 句调：句首微抬、句内下倾、句末降调；问句句末上扬；
  4) 三声变调（3+3→2+3）与「不/一」变调；轻声（第 5 声）缩短、降幅、平调；
  5) 重音：jieba 词性分级，实词首音节略高略长，虚词略轻略短；
  6) 句末延长（≥1.22×）；微抖动（F0 随机游走 + 幅度抖动）、长元音微颤（4.8Hz）；
  7) 辅音时长随语速缩放、浊声母/塞音/擦音/塞擦音按原生声母表重建；
  8) 共振峰三轴缩放与谱倾斜交给原生 14 维音色接口（`合成波形_快`）。

只读 import `meeting32/声带.py` 的音韵表与合成器；不改既有件（C14）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import 声学工具 as T  # noqa: E402
import 谱包络_校正 as EQ模块  # noqa: E402
from 声学工具 import 声带模块  # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402

标点停顿 = {
    "，": 0.18, "、": 0.15, "；": 0.26, "：": 0.30,
    "。": 0.40, "！": 0.44, "？": 0.45, "…": 0.55, "—": 0.45,
    ",": 0.18, ";": 0.26, ":": 0.30, ".": 0.40, "!": 0.44, "?": 0.45,
}
句末类 = set("。！？.!?")
疑问类 = set("？?")
虚词首 = set("upcyeowrd")          # 助/介/连/语气/叹/拟声/代词/副词：不重读
实词首 = set("nvatmqsfzilj")        # 名/动/形/数/量/处/方位/状态/成语/惯用/简称


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
    """3+3 连读变 2+3；不/一 基础变调。音节=[声,韵,调,原拼]。"""
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
    """jieba 词性 → 每音节 (词首?, 实词?)；切分失败/对不齐时全 False。"""
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
        # 对齐失败：按字符数兜底
        起 = [(k == 0, True) for k in range(音节数)]
    return 起


class 人类韵律:
    """固定 100Hz 帧率（10ms）的人类化韵律帧生成器。"""

    def __init__(self, 档案: 声线档案, 种子: int = 300, EQ=None):
        self.档案 = 档案
        self.种子 = int(种子)
        self.EQ = None if EQ is None else np.asarray(EQ, dtype=np.float64)
        self.rng = np.random.default_rng(int(种子) * 7919 + 13)
        self.律 = T.档案律(档案)
        self.音色 = T.档案音色(档案)
        self.基准F0 = float(self.律["基频"])
        self.语速 = float(self.律["语速"])
        self.声区 = float(self.律["声区"])
        self.幅度 = float(self.律["幅度"])
        self.气声 = float(self.律["气声度"])
        self.紧喉 = float(self.律["紧喉度"])
        self.粗糙 = float(self.律["粗糙度"])
        self.送气 = float(self.律["送气量"])
        self.s = tuple(float(x) for x in 档案.共振峰缩放)
        self.缩 = 3.2 / max(self.语速, 1.5)      # 辅音时长缩放
        self.漂移 = 0.0
        self.帧 = None

    # ── 帧操作 ──
    def _清(self):
        self.帧 = {k: [] for k in ("f0", "f1", "f2", "f3", "幅", "气声",
                                  "声源", "噪带", "送气", "紧喉", "粗糙")}
        self.漂移 = 0.0

    def _加(self, f0, f1, f2, f3, 幅, 气声, 声源, 噪带, 送气, 紧喉, 粗糙):
        f = self.帧
        f["f0"].append(float(f0))
        f["f1"].append(float(f1))
        f["f2"].append(float(f2))
        f["f3"].append(float(f3))
        f["幅"].append(float(幅))
        f["气声"].append(float(气声))
        f["声源"].append(int(声源))
        f["噪带"].append(噪带)
        f["送气"].append(float(送气))
        f["紧喉"].append(float(紧喉))
        f["粗糙"].append(float(粗糙))

    def _静音(self, 秒: float):
        帧数 = max(int(round(秒 * T.帧率)), 0)
        for _ in range(帧数):
            self._加(self.基准F0, 500.0, 1500.0, 2500.0, 0.0, self.气声, -1, None,
                     0.0, self.紧喉, self.粗糙)

    def _换气(self, 秒: float = 0.14, 幅度倍: float = 0.10):
        """吸气：低幅带通噪声（复用原生单谐振器噪声支路）。"""
        帧数 = max(int(round(秒 * T.帧率)), 1)
        for i in range(帧数):
            t = i / max(帧数 - 1, 1)
            包 = np.sin(np.pi * t) ** 0.8
            self._加(self.基准F0, 700.0, 1500.0, 2500.0,
                     self.幅度 * 幅度倍 * 包, self.气声, 0, (1600.0, 2200.0),
                     float(np.clip(self.送气 + 0.4, 0.0, 1.0)), self.紧喉, self.粗糙)

    def _噪声(self, 秒: float, 频带, 送气=None):
        帧数 = max(int(round(秒 * T.帧率)), 1)
        送 = self.送气 if 送气 is None else float(送气)
        for _ in range(帧数):
            self._加(self.基准F0, 500.0, 1500.0, 2500.0, self.幅度 * 0.4,
                     self.气声, 0, 频带, 送, self.紧喉, self.粗糙)

    def _浊(self, 秒: float, f0: float, 目: tuple, 幅度倍: float = 0.5):
        for _ in range(max(int(round(秒 * T.帧率)), 1)):
            目s = (目[0] * self.s[0], 目[1] * self.s[1], 目[2] * self.s[2])
            self._加(f0, 目s[0], 目s[1], 目s[2], self.幅度 * 幅度倍,
                     self.气声, 1, None, 0.0, self.紧喉, self.粗糙)

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
            self._噪声(0.005, (表[1], 600.0))
            送气 = 0.045 if 表[3] else 0.025
            self._噪声(送气 * self.缩, (表[1], 1200.0),
                       float(np.clip(self.送气 + (0.25 if 表[3] else 0.0), 0.0, 1.0)))
        elif 方式 == "擦":
            self._噪声(0.08 * self.缩, 表[1])
        elif 方式 == "塞擦":
            self._静音(0.010 * self.缩)
            self._噪声(0.005, (表[1][0], 600.0))
            self._噪声(0.04 * self.缩, 表[1])
        else:  # 浊声母 m/n/l/r
            目 = 声带模块.浊声母表.get(声, (250.0, 1100.0, 2200.0))
            度0 = T.声带模块._折线(T.声带模块.声调表.get(调, T.声带模块.声调表[1]), 0.0)
            f0 = self.基准F0 * 2.0 ** (((度0 - 3.0) * 2.0 * 0.78) / 12.0)
            self._浊(0.05 * self.缩, f0 * (1.0 + 0.5 * self.声区), 目, 0.5)

    # ── 元音 ──
    def _句调(self, t: float, j: int, n: int, 标点: str) -> float:
        v = 0.55 - 0.060 * j
        if 标点 in 疑问类:
            if j == n - 1:
                v += 2.2 * max(0.0, (t - 0.45) / 0.55)
            elif j == n - 2:
                v += 0.6
        elif 标点 in 句末类:
            if j == n - 1:
                v += -1.90          # 句末降调（qa ④-3：加强句界前下沉）
            elif j == n - 2:
                v += -0.85
        elif 标点 == "，":
            if j == n - 1:
                v += -0.30
        return float(v)

    def _元音(self, 韵: str, 调: int, j: int, n: int, 标点: str,
              时长倍: float, 重音: bool, 微颤相: float, 末尾: bool = False,
              首加成: float = 0.0):
        目标s, 尾 = T.声带模块._查韵母(韵)
        时长 = 0.18 * (3.2 / self.语速) * 时长倍
        帧数 = max(int(round(时长 * T.帧率)), 2)
        if 调 == 5:
            重音 = False
        for f in range(帧数):
            t = f / max(帧数 - 1, 1)
            # 元音目标滑移（原生同法），再做共振峰三轴缩放
            数 = len(目标s)
            段 = min(int(t * 数), 数 - 1)
            段内 = (t * 数) - 段
            a = np.asarray(目标s[段], dtype=np.float64)
            b = np.asarray(目标s[min(段 + 1, 数 - 1)], dtype=np.float64)
            ff = a + (b - a) * 段内
            if 尾 and t > 0.7:
                w = (t - 0.7) / 0.3
                ff = ff + (np.array([ff[0], 1000.0, 2200.0]) - ff) * w
            ff = np.array([ff[0] * self.s[0], ff[1] * self.s[1], ff[2] * self.s[2]])
            # 声调（音域收窄 0.78，避免过度歌唱化）
            if 调 == 5:
                半音 = -0.80
            else:
                度 = T.声带模块._折线(T.声带模块.声调表.get(调, T.声带模块.声调表[1]), t)
                半音 = (度 - 3.0) * 2.0 * 0.78
            半音 += self._句调(t, j, n, 标点)
            if j == 0:
                半音 += 首加成          # 句/段首音高重置（qa ④-3 可测）
            if 重音:
                半音 += 0.45
            # 微抖动：F0 随机游走 + 帧级小抖动（确定性）
            self.漂移 = float(np.clip(self.漂移 + self.rng.normal(0.0, 0.018), -0.18, 0.18))
            半音 += self.漂移 + float(self.rng.normal(0.0, 0.020))
            # 长元音微颤（4.8Hz，起落淡入淡出）
            颤 = 0.0
            if 时长 >= 0.13 and 0.28 <= t <= 0.86:
                颤 = 0.10 * float(np.sin(2.0 * np.pi * 4.8 * t * 时长 + 微颤相))
                半音 += 颤
            f0 = self.基准F0 * 2.0 ** (半音 / 12.0) * (1.0 + 0.5 * self.声区)
            # 幅度包络：起音 12%，收尾 25% 渐弱（到 45%）
            if t < 0.12:
                包 = t / 0.12
            elif t > 0.75:
                包 = 1.0 - 0.55 * (t - 0.75) / 0.25
            else:
                包 = 1.0
            幅 = self.幅度 * 包 * (1.10 if 重音 else 1.0) * float(self.rng.uniform(0.97, 1.03))
            幅 *= (0.80 if 调 == 5 else 1.0)
            self._加(f0, ff[0], ff[1], ff[2], 幅, self.气声, 1, None, 0.0,
                     self.紧喉, self.粗糙)
            末f0, 末ff, 末幅 = f0, ff, 幅
        # 句末尾音：元音收尾再拖 140ms 渐弱（qa ④-4 句末延长；也降低句尾截断感）
        if 末尾:
            for k in range(14):
                t = (k + 1) / 14.0
                self._加(末f0 * (1.0 - 0.015 * t), 末ff[0], 末ff[1], 末ff[2],
                         末幅 * 0.95 * (1.0 - 0.80 * t), min(self.气声 + 0.04, 0.8),
                         1, None, 0.0, self.紧喉, self.粗糙)

    # ── 主流程 ──
    def 建帧(self, 文本: str, 语速倍: float = 1.0) -> dict:
        self._清()
        if abs(float(语速倍) - 1.0) > 1e-9:
            self.语速 = float(np.clip(self.律["语速"] * float(语速倍), 1.5, 7.0))
            self.缩 = 3.2 / max(self.语速, 1.5)
        短语s = 切短语(文本)
        if not 短语s:
            return self.帧
        # 起始轻吸气（句首气口）
        self._换气(0.12, 0.07)
        for pi, (短语, 标点) in enumerate(短语s):
            拼 = 取音节(短语)
            音节s = []
            for p in 拼:
                声, 韵, 调 = T.声带模块._拆音节(p)
                音节s.append([声, 韵, int(调), p])
            音节s = _变调(音节s)
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
                    时长倍 *= 1.45          # 句末延长（qa ④-4：原 1.22 经测量仅 1.11×，提高并加尾音）
                if 重音:
                    时长倍 *= 1.06
                if j == 0 and 首加成 > 0.0:
                    时长倍 *= 1.10          # 句/段首字略长（配合音高重置，qa ④-3 可测）
                时长倍 *= float(self.rng.uniform(0.94, 1.06))
                self._声母(声, 调)
                微颤相 = float(self.rng.uniform(0.0, 2.0 * np.pi))
                self._元音(韵, 调, j, n, 标点, 时长倍, 重音, 微颤相, 末尾, 首加成)
            # 短语间停顿
            停 = float(标点停顿.get(标点, 0.0))
            if 标点 in 句末类:
                停 += 0.05
            if 停 > 0:
                self._静音(停)
                if 停 >= 0.30 and pi < len(短语s) - 1:
                    self._换气(0.13, 0.09)
        return self.帧

    def 说(self, 文本: str, 语速倍: float = 1.0, 写wav: str | None = None) -> tuple:
        帧 = self.建帧(文本, 语速倍)
        if not 帧["f0"]:
            return np.zeros(0), dict(样本数=0), 帧
        器 = T.取声带(self.种子)
        波形, 元 = 器.合成波形_快(帧, self.音色)
        if self.EQ is not None:
            波形 = EQ模块.应用(波形, self.EQ)      # 7 段静态谱包络校正（纯 numpy 零相位）
        if 写wav:
            T.写wav16(写wav, 波形)
        return 波形, 元, 帧


def 说(文本: str, 档案: 声线档案, 种子: int = 300, 语速倍: float = 1.0,
       写wav: str | None = None, EQ=None) -> tuple:
    return 人类韵律(档案, 种子, EQ=EQ).说(文本, 语速倍, 写wav)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    import glob as _glob
    import json as _json
    _p = _glob.glob(os.path.join(HERE, "..", "产物", "声线档案", "*.json"))
    if _p:
        with open(_p[0], encoding="utf-8") as _f:
            档 = 声线档案.从dict(_json.load(_f))
    else:
        档 = 声线档案(名="自检", 基频=210.0)
    for 试 in ("你好，我是洛天依。", "今天天气不错，我们一起出去走走吧。", "你最近，过得好吗？"):
        w, 元, 帧 = 说(试, 档, 300, 写wav=os.path.join(HERE, "..", "产物", "临时", "试听.wav"))
        f0, _, _ = T.基频轨迹(w)
        v = f0[np.isfinite(f0)]
        print("句=%s 时长=%.2fs 帧=%d F0=%.1f RMS=%.3f 停顿帧=%d"
              % (试, len(w) / T.FS, len(帧["f0"]),
                 float(np.median(v)) if v.size else 0.0, T.均方根(w),
                 int(sum(1 for x in 帧["声源"] if x == -1))))
