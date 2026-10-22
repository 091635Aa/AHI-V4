# -*- coding: utf-8 -*-
"""meeting62/实施/契约_谱与亚版本.py —— 「谱（md谱/乐谱/MIDI 谱）→ 亚版本 → 声库」的数据契约。

本契约为**纯新增件**（C14）：不改 AHI3/AHI4 任何既有文件；只被 meeting62 的实现 import。
只用 `meeting58/实施/接口契约_角色声线.py`（声线档案）与 `meeting58/实施/声线库.py` 的**只读**接口。

━━━ 术语（对齐 meeting62/共同简报 §0）━━━
  · **谱 / md谱**   ＝ 乐谱 / MIDI 谱：**音符序列** {歌词, 音高(midi 或 Hz), 时值, 连音}。
  · **亚版本**      ＝ 谱 → 声库 之间的**中间表示**；本件口径＝**音节序列**
                      {音素(声母/韵母/韵尾) × 声调(只取音素身份，**不取音高**) × 韵律轮廓(f0)
                       × 时长 × 音色引用}。
  · **声库**        ＝ 角色声线资源集合（`声线档案` 14 维）；本件通过 `取音色()` 只读接入。

━━━ 铁律 ━━━
  C1 禁注意力(并行序列)架构 ｜ C9 禁 OCR ｜ C10 禁外挂模型当器官 ｜ C13 确定性可复现 ｜ C14 纯新增
  口径：参数化模仿，非真声克隆；不得宣称是角色本人。

━━━ 归属（写越界=冲突）━━━
  `契约_谱与亚版本.py`  ← 本件（数据契约 + 校验 + 解析 + 建亚版本；无 DSP）
  `谱_亚版本_声库.py`   ← 实现（亚版本 → 帧 → 调用 v6 声带后端 → wav）
  只读 import：`meeting58/实施/接口契约_角色声线.py`、`meeting58/实施/声线库.py`
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

HERE = os.path.dirname(os.path.abspath(__file__))
MEET62 = os.path.dirname(HERE)
AHI4 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting61", "实施"))

from 接口契约_角色声线 import 声线档案  # noqa: E402  只读
import 声线库 as 库  # noqa: E402  只读

契约版本 = "score-subvoice-1.0"
A4_Hz = 440.0
A4_midi = 69

# 允许的声线引用（下拉可选；实际以声线库为准，未知名由 声线库.取 抛 E1006）
默认声线 = "EmoCompanion"

# ───────────────────────────── 错误码（码 + 字符串双写；E2xxx 独立命名空间） ─────────────────────────────
E2001 = "E2001"; E2002 = "E2002"; E2003 = "E2003"; E2004 = "E2004"
E2005 = "E2005"; E2006 = "E2006"; E2007 = "E2007"; E2008 = "E2008"
错误说明 = {
    E2001: "谱非法（空/音符数 0/拍速越界/单位未知）",
    E2002: "音符非法（音高越界/时值≤0/单位未知/歌词空且非休止）",
    E2003: "亚版本非法（音节数不符/时长≤0/音高越界/指纹失配）",
    E2004: "输入文本无法解析（简谱 token 非法）",
    E2005: "声线引用未知（由声线库 E1006 抛出）",
    E2006: "画像越界（音高单位↔音高不匹配）",
    E2007: "契约版本不兼容",
    E2008: "产物写盘失败",
}


class 契约错误(Exception):
    def __init__(self, 码: str, 细节: str = ""):
        self.码 = str(码)
        self.细节 = str(细节)
        super().__init__("%s %s %s" % (self.码, 错误说明.get(self.码, "?"), self.细节))


def 抛(码: str, 细节: str = "") -> None:
    raise 契约错误(码, 细节)


# ───────────────────────────── 音高/时值 工具 ─────────────────────────────
音高单位枚举 = ("midi", "hz")
时值单位枚举 = ("拍", "秒")
休止集 = frozenset(["", "-", "0", "rest", "R", "休止", "。"])
MIDI域 = (21.0, 108.0)
Hz域 = (55.0, 1200.0)
时值域 = (0.05, 16.0)
拍速域 = (30.0, 300.0)


def midi_转hz(m: float) -> float:
    return float(A4_Hz) * 2.0 ** ((float(m) - A4_midi) / 12.0)


def hz_转midi(h: float) -> float:
    return A4_midi + 12.0 * math.log2(max(float(h), 1e-9) / A4_Hz)


# 简谱音级 → C 大调半音（相对中央 C=60）；1' 升八度，1, 降八度
简谱音级 = {"1": 0, "2": 2, "3": 4, "4": 5, "5": 7, "6": 9, "7": 11}
简谱基midi = 60


# ───────────────────────────── 数据结构：谱 ─────────────────────────────
@dataclass(frozen=True)
class 音符:
    """谱的最小单元：一个音符（歌词 × 音高 × 时值 × 连音）。"""
    歌词: str = ""
    音高: float = 60.0
    音高单位: str = "midi"
    时值: float = 1.0
    时值单位: str = "拍"
    连音: bool = False
    备注: str = ""

    @property
    def 休止(self) -> bool:
        return str(self.歌词).strip() in 休止集

    def 音高Hz(self) -> float:
        if self.音高单位 == "midi":
            return midi_转hz(self.音高)
        if self.音高单位 == "hz":
            return float(self.音高)
        抛(E2002, "音高单位=%r" % (self.音高单位,))

    def 时值秒(self, 拍速: float) -> float:
        if self.时值单位 == "拍":
            return float(self.时值) * 60.0 / float(拍速)
        if self.时值单位 == "秒":
            return float(self.时值)
        抛(E2002, "时值单位=%r" % (self.时值单位,))

    def 校验(self) -> None:
        if self.音高单位 not in 音高单位枚举:
            抛(E2002, "音高单位=%r" % (self.音高单位,))
        if self.时值单位 not in 时值单位枚举:
            抛(E2002, "时值单位=%r" % (self.时值单位,))
        lo, hi = MIDI域 if self.音高单位 == "midi" else Hz域
        if not (lo <= float(self.音高) <= hi):
            抛(E2002, "音高=%r 不在 [%s,%s]" % (self.音高, lo, hi))
        if not (时值域[0] <= float(self.时值) <= 时值域[1]):
            抛(E2002, "时值=%r 不在 (0,%s]" % (self.时值, 时值域[1]))
        if not self.休止 and not str(self.歌词).strip():
            抛(E2002, "歌词为空且非休止")

    def 转dict(self) -> dict:
        return dict(歌词=str(self.歌词), 音高=float(self.音高), 音高单位=self.音高单位,
                    时值=float(self.时值), 时值单位=self.时值单位,
                    连音=bool(self.连音), 备注=str(self.备注))


@dataclass(frozen=True)
class 谱:
    """md谱/乐谱：拍速 + 声线引用 + 音符序列。**音高由音符决定**（见 亚版本 口径）。"""
    名: str = "无题"
    版本: str = "1.0"
    拍速: float = 100.0
    声线引用: str = 默认声线
    音符s: tuple = ()
    备注: str = ""

    def 校验(self) -> None:
        if self.版本 not in ("1.0",):
            抛(E2007, "版本=%r" % (self.版本,))
        if not (拍速域[0] <= float(self.拍速) <= 拍速域[1]):
            抛(E2001, "拍速=%r 不在 [%s,%s]" % (self.拍速, 拍速域[0], 拍速域[1]))
        if not self.声线引用:
            抛(E2001, "声线引用为空")
        if not self.音符s:
            抛(E2001, "音符序列为空")
        for i, n in enumerate(self.音符s):
            if not isinstance(n, 音符):
                抛(E2002, "第 %d 项不是 音符" % i)
            n.校验()

    def 总时长秒(self) -> float:
        return float(sum(n.时值秒(self.拍速) for n in self.音符s))

    def 指纹(self) -> str:
        h = hashlib.sha256()
        h.update(("score|%s|%s|%.6f|%s|" % (契约版本, self.名, self.拍速,
                                           self.声线引用)).encode())
        for n in self.音符s:
            d = n.转dict()
            h.update(json.dumps(d, ensure_ascii=False, sort_keys=True,
                                default=str).encode())
        return h.hexdigest()[:16]

    def 转dict(self) -> dict:
        return dict(契约版本=契约版本, 名=self.名, 版本=self.版本, 拍速=float(self.拍速),
                    声线引用=self.声线引用, 备注=self.备注,
                    音符s=[n.转dict() for n in self.音符s],
                    指纹=self.指纹(), 总时长秒=round(self.总时长秒(), 4))

    @staticmethod
    def 从dict(d: Mapping[str, Any]) -> "谱":
        ns = []
        for x in (d.get("音符s") or d.get("音符") or []):
            if isinstance(x, 音符):
                ns.append(x)
            else:
                x = dict(x)
                ns.append(音符(歌词=str(x.get("歌词", "")), 音高=float(x.get("音高", 60)),
                              音高单位=str(x.get("音高单位", "midi")),
                              时值=float(x.get("时值", 1.0)),
                              时值单位=str(x.get("时值单位", "拍")),
                              连音=bool(x.get("连音", False)),
                              备注=str(x.get("备注", ""))))
        s = 谱(名=str(d.get("名", "无题")), 版本=str(d.get("版本", "1.0")),
               拍速=float(d.get("拍速", 100.0)),
               声线引用=str(d.get("声线引用", d.get("声线", 默认声线))),
               音符s=tuple(ns), 备注=str(d.get("备注", "")))
        s.校验()
        return s

    @staticmethod
    def 从JSON文件(路径: str) -> "谱":
        with open(路径, encoding="utf-8") as f:
            return 谱.从dict(json.load(f))

    @staticmethod
    def 从简谱(文本: str, 声线引用: str = 默认声线, 名: str = "简谱") -> "谱":
        """极简简谱解析（1=C 大调）：每行 ``<音级[',登录]> <歌词> <时值(拍)> [+连音]``。

        音级 1–7（尾随 ``'`` 升八度、尾随 ``,`` 降八度）；``0``＝休止。
        ``#`` 起头的行为注释，``# 拍速=120`` 可覆盖 BPM。能跑即可（不做装饰音/变音记号）。
        """
        拍速 = 100.0
        音符s = []
        for 行 in str(文本).splitlines():
            行 = 行.strip()
            if not 行:
                continue
            if 行.startswith("#"):
                体 = 行[1:].strip()
                if "拍速" in 体 and "=" in 体:
                    拍速 = float(体.split("=")[-1].strip())
                continue
            toks = 行.split()
            if not toks:
                continue
            级 = toks[0]
            歌词 = toks[1] if len(toks) > 1 else ""
            时值 = float(toks[2]) if len(toks) > 2 else 1.0
            连音 = bool(len(toks) > 3 and toks[3].strip() == "+")
            if 级 == "0" or 级.startswith("0"):
                音符s.append(音符(歌词="0", 音高=60.0, 时值=时值, 时值单位="拍", 连音=连音))
                continue
            半 = 级[0]
            if 半 not in 简谱音级:
                抛(E2004, "音级=%r 非法（行 %r）" % (级, 行))
            oct = 0
            for c in 级[1:]:
                if c == "'":
                    oct += 12
                elif c == ",":
                    oct -= 12
                else:
                    抛(E2004, "八度记号=%r 非法（行 %r）" % (c, 行))
            m = 简谱基midi + 简谱音级[半] + oct
            音符s.append(音符(歌词=(歌词 or ("%s" % 半)), 音高=float(m), 音高单位="midi",
                              时值=时值, 时值单位="拍", 连音=连音))
        s = 谱(名=名, 拍速=拍速, 声线引用=声线引用, 音符s=tuple(音符s),
               备注="从简谱文本解析（1=C 大调，极简）")
        s.校验()
        return s


# ───────────────────────────── 数据结构：亚版本 ─────────────────────────────
@dataclass(frozen=True)
class 亚音节:
    """亚版本的一个音节：音素 × 声调(身份) × 韵律轮廓(f0) × 时长 × 音色引用。"""
    序: int = 0
    歌词: str = ""
    拼音: str = ""
    声: str = ""            # 声母（音素）
    韵: str = "a"           # 韵母（音素）
    尾: str = ""            # 鼻韵尾（'n'/'ng'/'')
    声调: int = 5           # 仅用于音素身份/元音目标；**音高不取自它**
    音高Hz: float = 261.63  # 由**谱音符**决定
    时长秒: float = 0.5
    连音: bool = False
    音色引用: str = 默认声线
    韵律轮廓: tuple = ()     # ((t, Hz), ...) 目标 f0 轨迹（含连音滑移；颤音在渲染期叠加）
    休止: bool = False
    目标s: tuple = ()        # 元音共振峰目标序列（供渲染；只读拷贝自音韵层）
    备注: str = ""

    def 校验(self) -> None:
        if not (0.01 <= float(self.时长秒) <= 30.0):
            抛(E2003, "时长秒=%r" % (self.时长秒,))
        if self.休止:
            return
        if not (Hz域[0] <= float(self.音高Hz) <= Hz域[1]):
            抛(E2003, "音高Hz=%r 不在 [%s,%s]" % (self.音高Hz, Hz域[0], Hz域[1]))
        if not str(self.韵):
            抛(E2003, "韵母为空")
        for pt in self.韵律轮廓:
            if len(pt) != 2 or not (0.0 <= float(pt[0]) <= 1.0):
                抛(E2003, "韵律轮廓点=%r" % (pt,))

    def 转dict(self) -> dict:
        return dict(序=int(self.序), 休止=bool(self.休止), 歌词=str(self.歌词),
                    拼音=str(self.拼音), 声=self.声, 韵=self.韵, 尾=self.尾,
                    声调=int(self.声调), 音高Hz=round(float(self.音高Hz), 4),
                    时长秒=round(float(self.时长秒), 4), 连音=bool(self.连音),
                    音色引用=self.音色引用,
                    韵律轮廓=[[round(float(a), 4), round(float(b), 4)] for a, b in self.韵律轮廓],
                    目标s=[[float(x) for x in t] for t in self.目标s], 备注=self.备注)


@dataclass(frozen=True)
class 亚版本:
    """谱 → 声库 的中间表示：谱指纹 + 声线引用 + 音节序列（可落盘、可校验、可比指纹）。"""
    契约版本: str = 契约版本
    谱指纹: str = ""
    声线引用: str = 默认声线
    音节s: tuple = ()
    备注: str = ""

    def 校验(self) -> None:
        if self.契约版本 != 契约版本:
            抛(E2007, "契约版本=%r" % (self.契约版本,))
        if not self.谱指纹 or len(self.谱指纹) != 16:
            抛(E2003, "谱指纹=%r" % (self.谱指纹,))
        if not self.声线引用:
            抛(E2003, "声线引用为空")
        if not self.音节s:
            抛(E2003, "音节序列为空")
        for i, s in enumerate(self.音节s):
            if s.序 != i:
                抛(E2003, "序不连续：第 %d 项 序=%d" % (i, s.序))
            s.校验()

    def 指纹(self) -> str:
        h = hashlib.sha256()
        h.update(("subvoice|%s|%s|%s|" % (self.契约版本, self.谱指纹,
                                         self.声线引用)).encode())
        for s in self.音节s:
            h.update(json.dumps(s.转dict(), ensure_ascii=False, sort_keys=True,
                                default=str).encode())
        return h.hexdigest()[:16]

    def 总时长秒(self) -> float:
        return float(sum(s.时长秒 for s in self.音节s))

    def 转dict(self) -> dict:
        return dict(契约版本=self.契约版本, 谱指纹=self.谱指纹, 声线引用=self.声线引用,
                    备注=self.备注, 音节数=len(self.音节s),
                    总时长秒=round(self.总时长秒(), 4), 指纹=self.指纹(),
                    音节s=[s.转dict() for s in self.音节s])

    @staticmethod
    def 从dict(d: Mapping[str, Any]) -> "亚版本":
        ss = []
        for x in (d.get("音节s") or []):
            x = dict(x)
            ss.append(亚音节(
                序=int(x.get("序", len(ss))), 歌词=str(x.get("歌词", "")),
                拼音=str(x.get("拼音", "")), 声=str(x.get("声", "")),
                韵=str(x.get("韵", "a")), 尾=str(x.get("尾", "") or ""),
                声调=int(x.get("声调", 5)), 音高Hz=float(x.get("音高Hz", 261.63)),
                时长秒=float(x.get("时长秒", 0.5)), 连音=bool(x.get("连音", False)),
                音色引用=str(x.get("音色引用", 默认声线)),
                韵律轮廓=tuple((float(a), float(b)) for a, b in x.get("韵律轮廓", ())),
                休止=bool(x.get("休止", False)),
                目标s=tuple(tuple(float(y) for y in t) for t in x.get("目标s", ())),
                备注=str(x.get("备注", ""))))
        v = 亚版本(契约版本=str(d.get("契约版本", 契约版本)),
                   谱指纹=str(d.get("谱指纹", "")), 声线引用=str(d.get("声线引用", 默认声线)),
                   音节s=tuple(ss), 备注=str(d.get("备注", "")))
        v.校验()
        return v


# ───────────────────────────── 谱 → 亚版本 ─────────────────────────────
def 建亚版本(谱对象: 谱, 声线引用: str | None = None) -> 亚版本:
    """谱 → 亚版本（确定性；**音高取自音符，不取自歌词声调**）。

    音素（声母/韵母/韵尾）与声调由歌词经 pypinyin + 音韵_v5 只读获得；
    音高 Hz 一律来自 ``音符.音高Hz()``；时长秒 来自 ``音符.时值秒()``；
    韵律轮廓 记录该音符的目标 f0 轨迹（连音⇒自上一音符滑入）。
    """
    import 音韵_v5 as Y  # noqa: E402  只读

    谱对象.校验()
    声 = str(声线引用 or 谱对象.声线引用)
    音节s = []
    前一Hz = None
    for i, n in enumerate(谱对象.音符s):
        时长 = float(n.时值秒(谱对象.拍速))
        if n.休止:
            音节s.append(亚音节(序=i, 休止=True, 歌词=str(n.歌词), 时长秒=时长,
                                连音=bool(n.连音), 音色引用=声,
                                韵律轮廓=((0.0, 0.0), (1.0, 0.0)), 备注="休止"))
            continue
        拼音 = _拼音(n.歌词)
        r = Y.解析音节(拼音)
        hz = float(n.音高Hz())
        连 = bool(n.连音) and 前一Hz is not None
        滑起 = float(前一Hz) if 连 else hz
        轮廓 = ((0.0, round(滑起, 4)), (0.12, round(hz, 4)), (1.0, round(hz, 4)))
        音节s.append(亚音节(序=i, 歌词=str(n.歌词), 拼音=拼音,
                            声=r["声"], 韵=r["韵"], 尾=r["尾"] or "", 声调=int(r["调"]),
                            音高Hz=round(hz, 4), 时长秒=round(时长, 4), 连音=连,
                            音色引用=声, 韵律轮廓=轮廓, 休止=False,
                            目标s=tuple(tuple(float(x) for x in t) for t in r["目标s"]),
                            备注=n.备注))
        前一Hz = hz
    v = 亚版本(契约版本=契约版本, 谱指纹=谱对象.指纹(), 声线引用=声,
               音节s=tuple(音节s), 备注="谱→亚版本（音高来自音符）")
    v.校验()
    return v


def _拼音(歌词: str) -> str:
    文 = str(歌词).strip()
    if 文 and all(c.isascii() for c in 文):
        # ASCII 歌词（如简谱/唱名 "la"/"ao"）：直接当拼音用，避免 pypinyin 把
        # "la" 拆成 ['l','a']（'l' 单声母 ⇒ 韵母空）。
        return 文.lower()
    from pypinyin import lazy_pinyin, Style
    拼 = lazy_pinyin(文, style=Style.TONE3, neutral_tone_with_five=True,
                    errors=lambda x: [c for c in x if c.isalnum()])
    拼 = [p for p in 拼 if p and p[0].isascii() and not _单声母(p)]
    if not 拼:
        return "la5"     # 无拼音（纯符号）→ 兜底为 la（可发声），不静默失败
    return 拼[0]


_声母集 = ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k",
          "h", "j", "q", "x", "r", "z", "c", "s")


def _单声母(p: str) -> bool:
    q = str(p).lower()
    if q and q[-1].isdigit():
        q = q[:-1]
    return q in _声母集


# ───────────────────────────── 亚版本 → 声库（只读接口） ─────────────────────────────
def 取音色(音色引用: str) -> 声线档案:
    """亚版本.音色引用 → 声线档案（声库只读；未知名透传 声线库 的 E1006）。"""
    try:
        return 库.取(str(音色引用))
    except 契约错误:
        raise
    except Exception as e:  # noqa: BLE001  声线库 抛的是 接口契约_角色声线.契约错误
        抛(E2005, "%s (%s)" % (音色引用, e))


@runtime_checkable
class 谱渲染协议(Protocol):
    """亚版本 → 音频（由实现类承担；本件只定义形状）。"""
    协议版本: str
    def 唱(self, 亚版本: 亚版本, *, 种子: int = 300) -> dict: ...


# ───────────────────────────── 自检 ─────────────────────────────
def _拒(fn, 期望) -> bool:
    try:
        fn()
    except 契约错误 as e:
        if isinstance(期望, str):
            return e.码 == 期望
        return isinstance(期望, type) and isinstance(e, 期望)
    except Exception as e:  # noqa: BLE001
        return isinstance(期望, type) and isinstance(e, 期望)
    return False


def 自检() -> dict:
    项 = []
    def 查(名, 条件, 细节=""):
        项.append(dict(项=名, 通过=bool(条件), 细节=str(细节)))

    s = 谱.从dict(dict(名="自检", 拍速=100, 声线引用="EmoCompanion", 音符s=[
        dict(歌词="啦", 音高=60, 时值=1.0),
        dict(歌词="啊", 音高=69, 时值=2.0, 连音=True)]))
    查("谱可解析", s.名 == "自检" and len(s.音符s) == 2)
    查("谱指纹16", len(s.指纹()) == 16, s.指纹())
    查("midi→Hz 正确", abs(音符(音高=69).音高Hz() - 440.0) < 1e-6, 音符(音高=69).音高Hz())
    查("时值→秒 正确", abs(s.音符s[0].时值秒(100.0) - 0.6) < 1e-9, s.音符s[0].时值秒(100.0))
    查("空谱被拒", _拒(lambda: 谱(音符s=()).校验(), E2001))
    查("音高越界被拒", _拒(lambda: 音符(音高=200.0).校验(), E2002))

    v = 建亚版本(s)
    查("亚版本音节数=音符数", len(v.音节s) == 2, len(v.音节s))
    查("亚版本指纹16", len(v.指纹()) == 16, v.指纹())
    查("音高取自音符", abs(v.音节s[0].音高Hz - midi_转hz(60.0)) < 1e-3, v.音节s[0].音高Hz)
    查("连音滑起", v.音节s[1].连音 and abs(v.音节s[1].韵律轮廓[0][1] - midi_转hz(60.0)) < 1e-3,
       v.音节s[1].韵律轮廓)
    查("音素存在", bool(v.音节s[0].韵), v.音节s[0].韵)
    v2 = 建亚版本(s)
    查("同输入同亚版本指纹", v.指纹() == v2.指纹(), v.指纹())

    简 = 谱.从简谱("# 拍速=120\n5 la 1\n6 la 1\n0 r 2\n7 la 2 +\n", 名="简谱自检")
    查("简谱解析音符数=4", len(简.音符s) == 4, len(简.音符s))
    查("简谱音级5→G4", abs(简.音符s[0].音高 - 67.0) < 1e-9, 简.音符s[0].音高)
    查("简谱休止", 简.音符s[2].休止, 简.音符s[2].歌词)
    查("简谱连音", bool(简.音符s[3].连音), 简.音符s[3].连音)
    查("简谱坏音级被拒", _拒(lambda: 谱.从简谱("9 la 1\n"), E2004))

    档 = 取音色("EmoCompanion")
    查("声库只读取音色", isinstance(档, 声线档案) and 档.名 == "EmoCompanion", 档.名)
    查("未知声线被拒", _拒(lambda: 取音色("不存在角色"), E2005))
    return dict(通过=all(x["通过"] for x in 项), 项=项)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    结 = 自检()
    for x in 结["项"]:
        print(("PASS " if x["通过"] else "FAIL ") + x["项"], x["细节"])
    print("总判:", 结["通过"])
