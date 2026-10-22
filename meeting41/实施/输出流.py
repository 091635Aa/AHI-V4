# -*- coding: utf-8 -*-
"""meeting41/实施/输出流.py —— L3 同时进行：思考链 ⇄ 手 ⇄ 正文 同拍三交替推进。

所有者原话（2026-10-08）：「**边思考边操作边输出正文**，相当于思考链的正文部分，
**思考流程和手上不会断**。」

━━━━━━━━━━━━━━ 冻结契约（接口契约.py 第四节；本文件逐字照做）━━━━━━━━━━━━━━
· 一拍() = 一个时基拍，内部**严格三交替**：
    ① 思考推进：思考链 +1，打点 "τ_q"                    （拍内偏移 0.0 拍）
    ② 手推进  ：有待执行技能 ⇒ 走 技能库/靶场，原子一步    （拍内偏移 0.4 拍）
    ③ 正文推进：发布 正文粒度拍 个分片，发布瞬间打点 "τ_arb"（拍内偏移 0.9 拍）
· 「不会断」= 三个**从真实时间线算出**的量（不写死）：
    最大空隙拍 ≤ 2 ｜ 思考跨操作 > 0 ｜ 孤立动作数 == 0（覆盖率 ≥ 0.9）
  打坏时间线即超标 —— 可失败性证明见 `事件契约_操作.反例()`（三种情形）。

━━━━━━━━━━━━━━ 计时口径（C13：禁 time.time()）━━━━━━━━━━━━━━
· 三层共用一个 `时基.拍钟`（钟=None ⇒ 自建）。
· 本层自持**虚拟拍栅格**：第 p 拍三步时刻 = 零点 + 时基.拍转秒(p + {0.0, 0.4, 0.9})，
  用 `拍钟.打点(名, t=显式值)` 写进共享钟 ⇒ 一切 τ 都是 时基 定点换算的结果，
  **不读墙钟**，故跨进程 / 跨 PYTHONHASHSEED 逐位同（T8）。
  注入的钟若已锚定 t0 则沿用其 t0；未锚定则以虚拟零点 0.0 锚定（拍 0 的 τ_q == 0.0）。
· 本文件不 import time。

━━━━━━━━━━━━━━ 与模块一「思考不公布」一致 ━━━━━━━━━━━━━━
· 思考项默认**不发布**（发布思考=False），对外只发正文分片；
· 正文分片一律过 `过滤元话语()`（与 meeting38/实施/输出层.py 同一口径的本地副本）；
· 过滤**非恒真**：自证里统计「净化前命中数 > 0 ∧ 净化后命中数 == 0」（见 净化台账()）。

━━━━━━━━━━━━━━ 不 import 任何器官（接口契约 §4 末条）━━━━━━━━━━━━━━
手只由构造参数传入的 技能库 / 靶场（合成靶场适配器）驱动；本文件不出现任何器官模块名。

━━━━━━━━━━━━━━ 诚实边界 ━━━━━━━━━━━━━━
· 手的原子只做「记录 + 过靶场（若注入）」；**不驱动真鼠标**（真机档在 真机探头.py，且须过令牌闸）。
· 若注入的技能对象只有整段 `执行(世界, 钟)` 而无 `原子序列`，本层**不调用它**
  （整段执行无法逐拍交替），只在 `手台账()` 里如实登记，并继续用内置手计划（不少产出）。
· 构造参数 `能力核` 只保存引用：本层产出**不依赖它**（能力核挂了也照常出思考/正文/动作）。
· 注入的引擎只在 `引擎台账()` 里留痕；不可用/报错时产出与"无引擎"逐位同（自证三臂实测）。
"""
from __future__ import annotations

import json
import os
import re
import sys

# ─────────────────────────── 路径自愈（照抄 00_共同简报 §二）───────────────────────────
_本目录 = os.path.dirname(os.path.abspath(__file__))
_AHI3 = os.path.dirname(os.path.dirname(_本目录))
for _p in (_本目录, os.path.join(_AHI3, "meeting40", "实施")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from 时基 import 拍钟, 拍转秒, 秒转拍, 分数拍转整数拍, 特征码 as 时基特征码  # noqa: E402

# 契约常量：从**只读真相源**读出；读不到就用同值兜底（绝不改契约文件）
try:
    import 接口契约 as 契约  # noqa: E402

    拍长秒 = float(契约.拍长秒)
    原子全 = tuple(契约.全部原子)
except Exception:  # pragma: no cover —— 契约缺失时按同值兜底，不少产出
    契约 = None
    拍长秒 = 0.1
    原子全 = ("移到", "左键", "右键", "中键", "滚轮", "拖拽", "键入")

_左 = ("左键", "右键", "中键")

# ═══════════════════════════ 元话语过滤（模块一口径的本地副本）═══════════════════════════
元话语 = (
    "我想到的是", "我想到", "我的内部状态是", "我的内部状态", "内部状态",
    "我的思考", "我在思考", "我思考", "我的脑", "脑子里", "心里想", "想了想",
    "想了一下", "我注意到", "沉思", "内省", "元话语", "神经元", "向量", "激活",
    "遥测", "telemetry", "特征码", "概率分布", "推理过程", "让我分析",
    "根据我的分析", "思考链", "思绪", "自我状态",
)
元话语_有序 = tuple(sorted(元话语, key=len, reverse=True))
_拍头_RE = re.compile(r"第\s*\d+\s*拍\s*[:：|｜]?")
_重标点_RE = re.compile(r"[\s，。、；：]{2,}")


def 命中元话语(文本) -> list:
    """返回命中的元话语词表（按 元话语 声明顺序，确定性）。"""
    s = "" if 文本 is None else str(文本)
    return [k for k in 元话语 if k in s]


def 过滤元话语(文本) -> tuple:
    """剥除元话语 → (净化后文本, 命中词表)。确定性、无副作用（模块一硬约束）。"""
    s = "" if 文本 is None else str(文本)
    命中 = 命中元话语(s)
    for k in 元话语_有序:
        s = s.replace(k, "")
    s = _拍头_RE.sub("", s)
    s = _重标点_RE.sub(lambda m: m.group(0)[0], s).strip(" ，。、；：|｜")
    return s, 命中


def _码(种子: int, *件) -> int:
    """确定性索引（不用内建 hash，避免 PYTHONHASHSEED 影响；C13）。"""
    import hashlib
    return int(hashlib.sha256(("|".join([str(种子)] + [str(x) for x in 件])).encode("utf-8"))
               .hexdigest()[:12], 16)


# ══════════════════════════════ 素材池（纯确定性；无生成式模型）══════════════════════════
_主题池 = ("眼下这一件", "手上这一步", "这一屏的落点", "刚接到的活", "要收口的那句")
_物池 = ("滚动条", "输入框", "标题行", "列表项", "按钮")
_元素池 = ("确认", "下一步", "提交", "搜索框", "正文区", "标签页", "关闭")
_键入池 = ("已核对", "按顺序来", "落点对上", "继续推进")
_正文池 = (
    "这一步先稳住，落点对上再往下走。",
    "把目标框住，手上这点活不耽误。",
    "先看当前这一屏，抓最靠上的那一项。",
    "顺序别乱：先定位，再动手，后核对。",
    "手上的动作保持连续，中间不停。",
    "这一拍的落点记下来，下一步照着走。",
    "把要紧的先办掉，剩下的排队等着。",
    "动作和话对齐，回头一看就能对上号。",
    "该点的点到了，接着看反馈。",
    "慢一点没关系，别把顺序丢了。",
)
_思考模板 = {
    "想": "第{拍}拍：我在思考{主题}该怎么落到手上，先把最靠前的一步定住。",
    "看": "第{拍}拍：我注意到画面里{物}的位置在动，先把它记下来。",
    "做": "第{拍}拍：手上正在推{原子}，我一边盯反馈一边盘下一步。",
    "说": "第{拍}拍：我在组织一句能落地的正文，内部状态不往外说。",
}
源序 = ("想", "看", "做", "说")


def 原子参数(原子: str, 序: int, 种子: int = 0) -> dict:
    """给一个原子配一份确定性参数（内置手计划用；纯函数，无全局态）。"""
    if 原子 == "移到":
        return {"点": (_码(种子, "x", 序) % 1200, _码(种子, "y", 序) % 800)}
    if 原子 == "拖拽":
        return {"起": (_码(种子, "x0", 序) % 1200, _码(种子, "y0", 序) % 800),
                "止": (_码(种子, "x1", 序) % 1200, _码(种子, "y1", 序) % 800)}
    if 原子 in _左:
        return {"元素": _元素池[_码(种子, "el", 序) % len(_元素池)]}
    if 原子 == "滚轮":
        return {"格数": int(_码(种子, "w", 序) % 5) - 2 or 3}
    if 原子 == "键入":
        return {"文本": _键入池[_码(种子, "kb", 序) % len(_键入池)]}
    return {}


class 输出流:
    """思考链 ⇄ 手 ⇄ 正文 三者同拍交替推进；思考不因操作而断。"""

    def __init__(self, 能力核=None, 技能库=None, 钟=None, 正文粒度拍: int = 1,
                 发布思考: bool = False, 种子: int = 0) -> None:
        self.能力核 = 能力核
        self.技能库 = 技能库
        self.正文粒度拍 = max(0, int(正文粒度拍))
        self.发布思考 = bool(发布思考)
        self.种子 = int(种子)

        # ── 共用拍钟 + 虚拟拍栅格（C13）──
        self.钟 = 钟 if 钟 is not None else 拍钟()
        self._零点 = float(self.钟.t0) if self.钟.t0 is not None else 0.0
        self.钟.打点("τ_in", t=self._时刻(0, 0.0))       # 流起点（τ=0 的同一时刻，与拍 0 的 τ_q 同瞬）

        # ── 三层时间线（链() 的三条列）──
        self.思考链: list = []
        self.正文分片列: list = []
        self.动作列: list = []
        self.拍数 = 0
        self._上次思考拍 = None
        self._首字打点 = None
        self._末正文拍 = None

        # ── 手（不 import 器官；只经 技能库/靶场）──
        self.靶场 = None
        self.手_队列: list = []          # list[dict(原子,参数,源,技能)]
        self.手_源轮转 = 0
        self.手_已排 = 0
        self.手_循环 = True              # 队列空 ⇒ 自动续（内置手计划 / 技能库）
        self.手_台账 = dict(执行数=0, 队列空补次数=0, 库来源=0, 内置来源=0,
                            靶场命中=0, 靶场未接=0, 靶场异常=0, 库异常=0, 库末次错误="",
                            整段执行技能=0)
        # 若技能库自带靶场适配器，自动取用（失败即不取，不影响产出）
        try:
            _场 = getattr(技能库, "靶场", None)
            if _场 is not None:
                self.靶场 = _场
        except Exception:
            pass

        # ── 引擎（可选；meeting39 六内核宿主；耦合失败不改产出）──
        self.引擎 = None
        self.引擎入口 = None
        self.引擎内态 = None
        self.引擎台账 = dict(有引擎=False, 入口名="", 探测说明="未注入",
                             调用数=0, 失败数=0, 命中试法=-1, 末次错误="")

        # ── 正文净化台账（证明过滤非恒真）──
        self.净化台账 = dict(候选数=0, 净化前命中=0, 净化后命中=0, 回退池=0)

    # ─────────────────────────── 计时 ───────────────────────────
    def _时刻(self, 拍: int, 偏移: float = 0.0) -> float:
        """第 拍 拍 + 偏移 的虚拟时刻（秒）。全部走 时基.拍转秒 定点换算，不读墙钟。"""
        return self._零点 + 拍转秒(int(拍) + float(偏移))

    # ═══════════════════════ 引擎注入（★try/except 全包住）═══════════════════════
    def 注入引擎(self, 引擎) -> None:
        """可选：meeting39 六内核宿主。★耦合失败**一个字都不许少产出**。

        只做「额外喂一拍思考」；引擎的返回只进 `引擎内态`/`引擎台账`，
        **不进** 思考链/正文/动作 三条时间线 ⇒ 引擎在不在，产出契约逐位同。
        接受形态（探测全在 try/except 内）：
          · dict(宿主=模块, 桥=对象) → 宿主.想一拍(桥, 问题, 拍数)
          · dict(想一拍=callable, ...) / 对象属性 想一拍/跑一拍/步进/推进/前向/一步/更新
          · 直接可调用对象
          · 桥形态（只有内态，无可调用推进口）→ 如实登记，不报错
        """
        self.引擎 = 引擎
        探 = dict(入口=None, 名="", 说明="未注入")
        try:
            探 = self._探引擎(引擎)
        except Exception as e:  # 探测本身也绝不外抛
            探 = dict(入口=None, 名="", 说明="探测异常:%s" % repr(e)[:120])
        self.引擎入口 = 探["入口"]
        self.引擎台账.update(有引擎=引擎 is not None, 入口名=探["名"], 探测说明=探["说明"])
        return None

    @staticmethod
    def _探引擎(引擎) -> dict:
        if 引擎 is None:
            return dict(入口=None, 名="", 说明="未注入")
        if isinstance(引擎, dict):
            宿主, 桥 = 引擎.get("宿主"), 引擎.get("桥")
            if 宿主 is not None and 桥 is not None:
                f = getattr(宿主, "想一拍", None)
                if callable(f):
                    return dict(入口=(lambda 问题, 拍数=1: f(桥, 问题, 拍数)),
                                名="宿主.想一拍(桥)", 说明="dict(宿主,桥)")
            for 名 in ("想一拍", "跑一拍", "步进", "推进", "前向", "一步", "更新"):
                f = 引擎.get(名)
                if callable(f):
                    return dict(入口=f, 名=名, 说明="dict 可调用口")
            return dict(入口=None, 名="", 说明="dict 里没有可调用推进口")
        if callable(引擎) and not any(hasattr(引擎, a) for a in ("思核", "自模型", "自我核")):
            return dict(入口=引擎, 名=type(引擎).__name__, 说明="直接可调用")
        for 名 in ("想一拍", "跑一拍", "步进", "推进", "前向", "一步", "更新"):
            f = getattr(引擎, 名, None)
            if callable(f):
                return dict(入口=f, 名=名, 说明="对象属性口(%s)" % type(引擎).__name__)
        if any(hasattr(引擎, a) for a in ("思核", "自模型", "自我核", "顿悟核", "画面核", "drv")):
            return dict(入口=None, 名="", 说明="桥形态：无可调用推进口（用 dict(宿主=,桥=) 注入可推进）")
        return dict(入口=None, 名="", 说明="认不出的引擎形态")

    def _问题(self, 输入s, 拍: int) -> str:
        主题 = self._主题(输入s, 拍)
        return "第%d拍：%s" % (拍, 主题)

    def _喂引擎(self, 输入s, 拍: int):
        if self.引擎入口 is None:
            return None
        问题 = self._问题(输入s, 拍)
        试法 = (
            lambda: self.引擎入口(问题, 拍数=1),
            lambda: self.引擎入口(问题, 1),
            lambda: self.引擎入口(问题),
            lambda: self.引擎入口(),
        )
        末错 = None
        for i, 试 in enumerate(试法):
            try:
                果 = 试()
                self.引擎台账["调用数"] += 1
                self.引擎台账["命中试法"] = i
                self.引擎内态 = self._摘要(果)
                return 果
            except TypeError as e:
                末错 = e
                continue
            except Exception as e:
                self.引擎台账["失败数"] += 1
                self.引擎台账["末次错误"] = repr(e)[:160]
                return None
        self.引擎台账["失败数"] += 1
        self.引擎台账["末次错误"] = repr(末错)[:160]
        return None

    @staticmethod
    def _摘要(果) -> dict:
        try:
            if isinstance(果, dict):
                return dict(形态="dict", 键=[str(k) for k in list(果.keys())[:12]])
            return dict(形态=type(果).__name__)
        except Exception:
            return dict(形态="不可摘要")

    # ═══════════════════════════ 手：队列 / 技能库 / 靶场 ═══════════════════════════
    def 设靶场(self, 世界) -> None:
        """注入合成靶场适配器（契约 §3：.元素列()/.点(名)/.滚(格数)/.打(文本)）。"""
        self.靶场 = 世界
        return None

    def 排计划(self, 原子列, 源: str = "外部") -> int:
        """外部排一条原子计划（list[dict(原子=,参数=)] 或 list[str]）。返回入队数。"""
        入 = 0
        for i, it in enumerate(list(原子列 or [])):
            原子, 参数 = self._规范原子(it, i)
            if 原子 is None:
                continue
            self.手_队列.append(dict(原子=原子, 参数=参数, 源=源, 技能=源))
            入 += 1
        self.手_已排 += 入
        return 入

    def 排技能(self, 名: str, 原子列=None) -> int:
        """按名从 技能库 取技能并排入队；给了 原子列 则直接用（便于靶场/真机适配）。"""
        if 原子列 is not None:
            return self.排计划(原子列, 源="技能:%s" % 名)
        原子列 = self._取技能原子列(名)
        if not 原子列:
            self.手_台账["整段执行技能"] += 1
            return 0
        return self.排计划(原子列, 源="技能:%s" % 名)

    def 停手(self) -> None:
        self.手_循环 = False
        return None

    def 手台账(self) -> dict:
        return dict(self.手_台账, 队列长=len(self.手_队列), 靶场已接=(self.靶场 is not None))

    @staticmethod
    def _规范原子(it, i: int):
        if isinstance(it, str):
            return it, 原子参数(it, i)
        if isinstance(it, dict):
            原子 = it.get("原子") or it.get("名") or it.get("动作")
            if 原子 is None:
                return None, {}
            参数 = it.get("参数")
            return str(原子), (dict(参数) if isinstance(参数, dict) else 原子参数(str(原子), i))
        return None, {}

    @staticmethod
    def _技能原子序列(技能):
        """按契约 §3：技能对象的 `原子序列`（list[dict(原子=,参数=)]）。"""
        if 技能 is None:
            return None
        if isinstance(技能, (list, tuple)):
            return list(技能)
        列 = getattr(技能, "原子序列", None)
        if 列 is None:
            列 = getattr(技能, "序列", None)
        if 列 is None:
            return None
        try:
            return list(列)
        except Exception:
            return None

    def _取技能原子列(self, 名: str):
        if self.技能库 is None:
            return None
        try:
            f = getattr(self.技能库, "取", None)
            技能 = f(名) if callable(f) else None
        except Exception as e:
            self.手_台账["库异常"] += 1
            self.手_台账["库末次错误"] = repr(e)[:160]
            技能 = None
        return self._技能原子序列(技能)

    def _库名单(self):
        if self.技能库 is None:
            return []
        for m in ("列表", "技能列表", "名单"):
            f = getattr(self.技能库, m, None)
            if callable(f):
                try:
                    列 = list(f())
                    if 列:
                        return 列
                except Exception as e:
                    self.手_台账["库异常"] += 1
                    self.手_台账["库末次错误"] = repr(e)[:160]
        return []

    def _内置手计划(self):
        """无外部技能时的内置手计划：一轮 7 个原子（全部原子各一次，起始按种子旋转）。"""
        排次 = self.手_已排 // max(1, len(原子全))
        return [dict(原子=原子全[(self.种子 + 排次 + i) % len(原子全)]) for i in range(len(原子全))]

    def _补队列(self) -> None:
        self.手_台账["队列空补次数"] += 1
        名单 = self._库名单()
        if 名单:
            名 = str(名单[self.手_源轮转 % len(名单)])
            self.手_源轮转 += 1          # ★无论能否单步都轮转：不卡在同一个不可单步技能上
            原子列 = self._取技能原子列(名)
            if 原子列:
                入 = self.排计划(原子列, 源="技能库:%s" % 名)
                if 入:
                    self.手_台账["库来源"] += 入
                    return
            self.手_台账["整段执行技能"] += 1
            表 = self.手_台账.setdefault("不可单步技能表", [])
            if 名 not in 表:
                表.append(名)
        elif self.技能库 is not None:
            self.手_台账["整段执行技能"] += 1   # 库在但取不到可单步技能：如实登记
        入 = self.排计划(self._内置手计划(), 源="内置手计划")
        self.手_台账["内置来源"] += 入

    def _过靶场(self, 原子: str, 参数: dict):
        """过合成靶场（若注入）；返回 命中(bool) 或 None（靶场未接该原子）。全部 try/except。"""
        世界 = self.靶场
        if 世界 is None:
            return None
        try:
            if 原子 in _左:
                目标 = 参数.get("元素") or getattr(世界, "当前目标", None)
                if 目标 is None:
                    return None
                return bool(世界.点(目标))
            if 原子 == "滚轮":
                return bool(世界.滚(int(参数.get("格数", 1))))
            if 原子 == "键入":
                return bool(世界.打(str(参数.get("文本", ""))))
            f = getattr(世界, "移到", None)
            if callable(f):
                return bool(f(参数.get("点")))
            return None
        except Exception:
            self.手_台账["靶场异常"] += 1
            return None

    def _手推进一步(self, 拍: int):
        """② 手推进：若有待执行技能 ⇒ 弹**一个**原子执行（拍内偏移 0.4 拍）。"""
        if not self.手_队列:
            if not self.手_循环:
                return None
            self._补队列()
        if not self.手_队列:
            return None
        项 = self.手_队列.pop(0)
        命中 = self._过靶场(项["原子"], 项["参数"])
        if 命中 is None:
            self.手_台账["靶场未接"] += 1
        elif 命中:
            self.手_台账["靶场命中"] += 1
        动作 = dict(原子=项["原子"], 参数=dict(项["参数"]),
                    τ=self.钟.分数拍(self._时刻(拍, 0.4)),
                    t_mono=self._时刻(拍, 0.4), 拍=int(拍), 源=项["源"], 命中=命中)
        self.动作列.append(动作)
        self.手_台账["执行数"] += 1
        return 动作

    # ═══════════════════════════ 思考 / 正文 ═══════════════════════════
    def _主题(self, 输入s, 拍: int) -> str:
        if isinstance(输入s, dict):
            for k in ("问题", "文本", "内容", "s", "主题", "输入"):
                if 输入s.get(k):
                    return str(输入s[k])[:40]
            return _主题池[_码(self.种子, "t", 拍) % len(_主题池)]
        if 输入s:
            return str(输入s)[:40]
        return _主题池[_码(self.种子, "t2", 拍) % len(_主题池)]

    def _思考文本(self, 拍: int, 源: str, 动作, 输入s) -> str:
        主题 = self._主题(输入s, 拍)
        原子 = (动作 or {}).get("原子") or 原子全[_码(self.种子, "a", 拍) % len(原子全)]
        物 = _物池[_码(self.种子, "o", 拍) % len(_物池)]
        return _思考模板[源].format(拍=拍, 主题=主题, 原子=原子, 物=物)

    def _生成正文(self, 拍: int, 动作, 输入s, 思考项) -> str:
        """③ 正文分片文本：正文＝思考链的正文部分 → 一律过元话语过滤，净化不足则回退实质句池。"""
        模 = int(拍) % 3
        if 模 == 0:
            候选 = _正文池[_码(self.种子, "z", 拍) % len(_正文池)]
        elif 模 == 1:
            候选 = 思考项["文本"]          # ★同一拍的思考内容 → 净化后作为正文
        else:
            原子 = (动作 or {}).get("原子")
            候选 = ("%s这一步已经落下，接着核对反馈。" % 原子) if 原子 else \
                _正文池[_码(self.种子, "z2", 拍) % len(_正文池)]
        净, 命中 = 过滤元话语(候选)
        self.净化台账["候选数"] += 1
        self.净化台账["净化前命中"] += len(命中)
        if len(净) < 6:
            self.净化台账["回退池"] += 1
            净, 命中2 = 过滤元话语(_正文池[_码(self.种子, "z3", 拍) % len(_正文池)])
            self.净化台账["净化前命中"] += len(命中2)
        self.净化台账["净化后命中"] += len(命中元话语(净))
        return 净

    def _正文推进(self, 拍: int, 动作, 输入s, 思考项):
        """③ 正文推进：发布 正文粒度拍 个分片；发布瞬间打点 "τ_arb"。返回首个分片 或 None。"""
        首 = None
        for k in range(self.正文粒度拍):
            文本 = self._生成正文(拍, 动作, 输入s, 思考项)
            if not 文本:
                continue
            序 = len(self.正文分片列)
            时刻 = self._时刻(拍, 0.9)
            打点 = self.钟.打点("τ_arb", t=时刻)      # ★发布瞬间
            if 序 == 0:
                self._首字打点 = dict(打点)
            分片 = dict(τ=打点["分数拍"], 拍=int(拍), 文本=文本, 序=序, 首字=bool(序 == 0))
            self.正文分片列.append(分片)
            self._末正文拍 = int(拍)
            if 首 is None:
                首 = 分片
            if self.发布思考:
                # 诊断臂：把思考项也作为分片发布（仍过元话语过滤，故与模块一不冲突）
                净, _ = 过滤元话语(思考项["文本"])
                净 = 净 or 文本
                序2 = len(self.正文分片列)
                分片2 = dict(τ=打点["分数拍"], 拍=int(拍), 文本=净, 序=序2, 首字=False)
                self.正文分片列.append(分片2)
                self._末正文拍 = int(拍)
        return 首

    # ═══════════════════════════ 契约方法 ═══════════════════════════
    def 一拍(self, 输入s=None) -> dict:
        """一拍 = 一个时基拍。内部严格三交替：思考 → 手 → 正文。"""
        拍 = self.拍数
        # ── ① 思考推进 ──
        τ_q = self.钟.打点("τ_q", t=self._时刻(拍, 0.0))["分数拍"]
        # 源：本拍手上有排队动作 ⇒ "做"；否则按 种子+拍 在 想/看/做/说 间确定性轮转
        源 = "做" if self.手_队列 else 源序[(拍 + self.种子) % len(源序)]
        动作预告 = self.手_队列[0]["原子"] if self.手_队列 else None
        思考项 = dict(τ=τ_q, 拍=int(拍), 源=源,
                      文本=self._思考文本(拍, 源, {"原子": 动作预告}, 输入s))
        self.思考链.append(思考项)
        空隙拍 = 0 if self._上次思考拍 is None else int(拍) - int(self._上次思考拍)
        self._上次思考拍 = int(拍)
        self._喂引擎(输入s, 拍)                      # 可选；失败不影响下面任何产出
        # ── ② 手推进 ──
        动作 = self._手推进一步(拍)
        # ── ③ 正文推进 ──
        首分片 = self._正文推进(拍, 动作, 输入s, 思考项)
        self.拍数 += 1
        return dict(拍=int(拍), τ=τ_q, 思考=思考项, 正文=首分片, 动作=动作, 空隙拍=空隙拍)

    def 链(self) -> dict:
        """完整可审计时间线。"""
        return dict(思考项列=list(self.思考链), 正文项列=list(self.正文分片列),
                    动作列=list(self.动作列))

    def 链指标(self) -> dict:
        """★「不会断」的四个量都从真实时间线算出（见 事件契约_操作 独立复算）。"""
        思考拍 = sorted(int(x["拍"]) for x in self.思考链)
        空隙 = [b - a for a, b in zip(思考拍, 思考拍[1:])]
        最大空隙拍 = max(空隙) if 空隙 else 0
        动作拍 = set(int(a["拍"]) for a in self.动作列)
        思考拍集 = set(思考拍)
        思考跨操作 = sum(1 for p in 动作拍 if p in 思考拍集)
        文拍 = sorted(int(x["拍"]) for x in self.正文分片列)
        孤立动作 = sum(1 for p in 动作拍 if not any(abs(p - q) <= 2 for q in 文拍))
        总字 = sum(len(str(x["文本"])) for x in self.正文分片列)
        经历秒 = 拍转秒(self.拍数)
        字速 = round(总字 / 经历秒, 4) if 经历秒 > 0 else 0.0
        if self._首字打点 is None:
            首字延迟 = -1.0
        else:
            try:
                首字延迟 = float(self.钟.延迟(self.钟.取埋点("τ_in"), self._首字打点)["毫秒"])
            except Exception:
                首字延迟 = -1.0
        return dict(思考数=len(self.思考链), 正文数=len(self.正文分片列), 动作数=len(self.动作列),
                    最大空隙拍=int(最大空隙拍), 思考跨操作=int(思考跨操作),
                    孤立动作=int(孤立动作), 字速=float(字速), 首字延迟毫秒=首字延迟)

    def 正文全文(self) -> str:
        return "".join(str(x["文本"]) for x in sorted(self.正文分片列, key=lambda d: d["序"]))

    def 特征码(self) -> str:
        """链指标 + 时间线的**时基**特征码（不含 t0/t_mono ⇒ 跨进程逐位同，C13/T8）。"""
        埋 = {名: self.钟.点[名]["分数拍"] for 名 in sorted(self.钟.点.keys())}
        return 时基特征码(dict(
            链指标=self.链指标(),
            思考=[[int(x["拍"]), str(x["源"]), str(x["文本"])] for x in self.思考链],
            正文=[[int(x["拍"]), int(x["序"]), str(x["文本"])] for x in self.正文分片列],
            动作=[[int(x["拍"]), str(x["原子"]), x["参数"]] for x in self.动作列],
            埋点τ={k: round(float(v), 6) for k, v in 埋.items()},
            净化=dict(self.净化台账),
        ))

    def 净化台账_只读(self) -> dict:
        return dict(self.净化台账)

    def 引擎台账_只读(self) -> dict:
        return dict(self.引擎台账, 内态=self.引擎内态)


# ═══════════════════════════════ 自证（真跑；出数字）═══════════════════════════════
def _链四项(流) -> dict:
    """本文件内联复算 T3/T4 四项（不依赖 事件契约_操作，保证可独立判据化）。"""
    思拍 = sorted(int(x["拍"]) for x in 流.思考链)
    空隙 = [b - a for a, b in zip(思拍, 思拍[1:])]
    文拍 = sorted(int(x["拍"]) for x in 流.正文分片列)
    孤立 = sum(1 for a in 流.动作列 if not any(abs(int(a["拍"]) - q) <= 2 for q in 文拍))
    动作拍 = [int(a["拍"]) for a in 流.动作列]
    思集 = set(思拍)
    跨 = sum(1 for p in 动作拍 if p in 思集)
    覆盖 = 0.0 if not 动作拍 else round(sum(
        1 for p in 动作拍 if any(abs(p - q) <= 2 for q in 文拍)) / float(len(动作拍)), 4)
    return dict(最大空隙拍=int(max(空隙) if 空隙 else 0), 思考跨操作=int(跨),
                孤立动作数=int(孤立), 覆盖率=float(覆盖))


def _跑一臂(拍数: int, 种子: int, 引擎=None, 引擎形态: str = "无"):
    流 = 输出流(种子=种子)
    if 引擎形态 != "无":
        流.注入引擎(引擎)
    for _ in range(int(拍数)):
        流.一拍(输入s="把文档里那一句改稳")
    return 流


class _坏引擎:
    """每次调用必抛的引擎（证明 try/except 真的兜住耦合失败）。"""

    def 想一拍(self, 问题, 拍数: int = 1):
        raise RuntimeError("内核宿主不在（故意）")


class _好引擎:
    """可调用但**不改时间线**的引擎（证明有引擎也不多不少）。"""

    def 想一拍(self, 问题, 拍数: int = 1):
        return dict(内态="引擎已思考", 拍=拍数, 问题=str(问题)[:24])


def 自证(拍数: int = 120, 种子: int = 0) -> dict:
    """直接跑出来的实测：四项指标 + 引擎三臂 + 确定性 + 元话语过滤非恒真。"""
    甲 = _跑一臂(拍数, 种子)
    乙 = _跑一臂(拍数, 种子)
    丙 = _跑一臂(拍数, 种子 + 1)
    无引擎 = _跑一臂(拍数, 种子)
    坏引擎 = _跑一臂(拍数, 种子, 引擎=_坏引擎(), 引擎形态="坏")
    好引擎 = _跑一臂(拍数, 种子, 引擎=_好引擎(), 引擎形态="好")

    指标 = 甲.链指标()
    四项 = _链四项(甲)
    全文 = 甲.正文全文()
    含元话语 = 命中元话语(全文)

    def _产(流):
        k = 流.链指标()
        return (k["思考数"], k["正文数"], k["动作数"])

    确定性 = dict(同种子两遍逐位同=bool(甲.特征码() == 乙.特征码()),
                  异种子不同=bool(甲.特征码() != 丙.特征码()),
                  码_甲=甲.特征码()[:16], 码_乙=乙.特征码()[:16], 码_丙=丙.特征码()[:16])
    三臂 = dict(
        无引擎=dict(产出=_产(无引擎), 码=无引擎.特征码()),
        坏引擎=dict(产出=_产(坏引擎), 码=坏引擎.特征码(),
                    失败数=int(坏引擎.引擎台账["失败数"]), 末次错误=坏引擎.引擎台账["末次错误"][:60]),
        好引擎=dict(产出=_产(好引擎), 码=好引擎.特征码(), 调用数=int(好引擎.引擎台账["调用数"])),
        特征码逐位同=bool(无引擎.特征码() == 坏引擎.特征码() == 好引擎.特征码()),
        产出逐项同=bool(_产(无引擎) == _产(坏引擎) == _产(好引擎)),
    )
    元话语果 = dict(净化前命中=int(甲.净化台账["净化前命中"]),
                    净化后命中=int(甲.净化台账["净化后命中"]),
                    回退池=int(甲.净化台账["回退池"]),
                    回退路径对=bool(len(过滤元话语("我在思考")[0]) < 6),
                    正文全文含元话语词表=含元话语, 正文全文=全文)
    通过 = bool(四项["最大空隙拍"] <= 2 and 四项["思考跨操作"] > 0
                and 四项["孤立动作数"] == 0 and 四项["覆盖率"] >= 0.9
                and 三臂["特征码逐位同"] and 确定性["同种子两遍逐位同"] and 确定性["异种子不同"]
                and 元话语果["净化后命中"] == 0 and 元话语果["净化前命中"] > 0
                and 元话语果["回退路径对"])
    return dict(通过=通过, 拍数=int(拍数), 种子=int(种子),
                链指标=指标, 四项=四项, 引擎三臂=三臂, 确定性=确定性, 元话语=元话语果,
                手台账=甲.手台账(), 契约=dict(拍长秒=拍长秒, 原子数=len(原子全)))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 自证()
    print(json.dumps(果, ensure_ascii=False, indent=1, default=str))
    print("[输出流] 自证：%s ｜ 最大空隙拍=%d ｜ 思考跨操作=%d ｜ 孤立动作数=%d ｜ 覆盖率=%.4f"
          % ("PASS" if 果["通过"] else "FAIL", 果["四项"]["最大空隙拍"],
             果["四项"]["思考跨操作"], 果["四项"]["孤立动作数"], 果["四项"]["覆盖率"]))
    raise SystemExit(0 if 果["通过"] else 1)
