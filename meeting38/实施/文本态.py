# -*- coding: utf-8 -*-
"""文本态.py —— AHI 纯文本对话态主程序（置顶任务：AHI2.1/meeting38/00_置顶_纯文本对话态.md）

只装配三件（其余器官一律冻结：不 import、不实例化、不起线程）：
  · 共存态  共享意识工作区（本阶段新增「对话槽」）
  · 综合体  meeting31 synthesis.py —— 「想 + 回忆 + 自言」的角色（不是工具调用、无 prompt 编排）
  · 嘴器    器官_嘴.py，用语音=False → 纯文本降级
外加 容量桥（FSW 长时记忆：每轮回合/自主思考记成「经历」）。
「世界」= **开放域（只读）**：文本态无屏幕（眼已冻结），故自主取材 = 只读探索自己的项目
  （步5 §4.5「看全屏挑没碰过的控件」在文本态下的等价物）；导航器构造即保证只读 + 有界。

对话回路（所有者「第一步/第二步」）：
  用户文本 → 脑「想一遍」+ 回忆历史（写「它·想」）→ 产出思考结果 → 转文本 → 说出去（写「它说」）。
  我不发消息 → 它自己作主：想就继续想（断供一拍 → 自言外显），实在没活就睡（内在状态驱动）。

「给世界，不给限制区」：全程序**不存在任何时间/拍数阈值、也没有节流阀**——
  醒/睡只听它自己**这一拍有没有冒出念头**（有想头＝醒、没得想＝歇），是 0 边界的存在/
  缺席判据，**不是**调出来的数：没有任何「空闲 N 秒就睡」「每 N 秒 ≤1 行」「饱足≥0.5」。
  念头多寡由引擎内驱（饱足/消化债→动作偏置）塑造，睡/醒因而是内在动力学的自然涌现。

铁律：C1 禁 Transformer｜C9 禁 OCR｜C10 禁外挂模型当器官｜C13 确定性可复现｜C14 不破既有。
"""
from __future__ import annotations

import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))          # .../meeting38/实施
M38 = os.path.dirname(HERE)                                # .../meeting38
AHI = os.path.dirname(M38)                                 # AHI3 根（迁移：随文件定位）
for _p in (HERE, M38, AHI, os.path.join(AHI, "meeting31")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from 共存态 import 共存态                                    # noqa: E402
from 器官_嘴 import 嘴器                                     # noqa: E402
from 容量桥 import 容量桥                                    # noqa: E402
from 通用接口 import 统一输入, 统一输出, 升维                # noqa: E402  (Phase2 通用 I/O)


def _行(谁: str, 文: str, t: float = None) -> str:
    """文本流一行：[谁 时:分:秒] 文。"""
    t = time.time() if t is None else t
    return "[%s %s] %s" % (str(谁).ljust(4), time.strftime("%H:%M:%S", time.localtime(t)), 文)


class 文本嘴(嘴器):
    """嘴的纯文本子类：不发声，按文本流格式打印（不改动 器官_嘴.py 的并发骨架）。

    多窗口（所有者 2026-10-06「同一生命·多窗口」）：表达槽每句带「会话」，
    故本子类**自己**跑发声循环（不走基类的队列/语音子线程）——按窗口分流打印，
    主窗口不标窗号、其它窗口标 `[它@A ...]`，让"一个我同时对着几个窗口说话"看得见。"""

    def __init__(self, 态, 打印锁):
        super().__init__(态, 用语音=False)                   # 纯文本降级，不起语音后端
        self._打印锁 = 打印锁

    def 运行(self, to_stop):
        while not to_stop.is_set() and not self._停.is_set():
            try:
                got = self.态.取下一句()
                if got is None:
                    to_stop.wait(0.05)                # 嘴的轮询节拍（事件等待，非生命睡阈）
                    continue
                _t, 文, 会话 = got
                self._发声(文, 会话)
            except Exception:
                try:
                    to_stop.wait(0.05)
                except Exception:
                    pass

    def _发声(self, 文本: str, 会话: str = "主"):
        谁 = "它" if str(会话) == "主" else "它@%s" % 会话
        try:
            with self._打印锁:
                print(_行(谁, 文本), flush=True)
        except Exception:
            pass
        self._回流(文本)                                     # 保持动作槽回流（与基类一致）


class 文本态:
    """纯文本对话生命：共存态 + 综合体（脑）+ 嘴 + 容量桥（记忆）。"""

    def __init__(self, 种子: int = 310, 挂记忆: bool = True):
        self.种子 = int(种子)
        self.态 = 共存态()
        self._打印锁 = threading.Lock()
        self.嘴 = 文本嘴(self.态, self._打印锁)
        # ★编码器接入：容量桥入库自动补语义向量 → 查经历可走真语义通路（见 容量桥 文件头）
        self.容量 = 容量桥(种子=3800, 编码器=getattr(self.态, "编码", None)) if 挂记忆 else None
        self.桥 = None
        self.接口 = None                                     # Phase2：统一输入（全接口预处理）
        self.输出 = 统一输出()                               # Phase2：统一输出（可插拔通道）
        self.停 = threading.Event()
        self.统计 = dict(回合=0, 自主拍=0, 自言外显=0, 醒拍=0, 睡拍=0, 延答=0)
        self._嘴线程 = None
        # meeting43 · P8 运行期自诊（纯旁路，默认关；AHI_自诊=1 开启）
        self.自诊 = None
        self._自诊开 = os.environ.get("AHI_自诊", "") in ("1", "true", "True", "yes")
        self._自诊间隔 = 20

    # ══════════ 装配 ══════════
    def 建(self, 教材行s=None, 接开放域: bool = True,
           根s=None, 屏蔽s=()) -> "文本态":
        """初始化综合体（想/回忆/自言）+ 「世界」。

        「世界」= **开放域（只读）**：文本态没有屏幕（眼已冻结），故「自己找事干 /
        看世界」= 只读地探索它自己所在的项目文件——即步5 §4.5「看全屏挑没碰过的
        控件」在文本态下的等价物。导航器构造即保证只读且有界（单文件 ≤2MB、每文件
        ≤200 行、单次 IO ≤1.5s），大文件/二进制自动跳过。

        ⚠ 为何必须接开放域（实测结论）：`饱足`/`消化债`（§3.1 指定的「想够了没有
        → 要不要先歇」信号）**只在文件域通道更新**；教材/行动通道下二者恒为
        1.0/0.0（死值），生命永不自主入睡。接开放域后信号才真正流动。

        教材（可选）：仅当**显式传入 教材行s** 时才设——避免教材通道在块边界通道
        竞争里长期压过文件域（用户语料短时也会让 未读率→0 后自动让位，但源码教材
        会占用很久），从而保证自主睡/醒可复现。"""
        from synthesis import 综合体                            # noqa: E402  (meeting31)
        桥 = 综合体(seed=self.种子)
        if 教材行s is not None:
            书 = [行.strip() for 行 in 教材行s if 6 <= len(行.strip()) <= 120]
            界 = list(range(1, len(书) + 1))                      # 扁平书=单文件
            桥.设教材(书, 界)
        if 接开放域:
            根 = list(根s) if 根s else [AHI]                      # 默认世界=自己的项目
            桥.设开放域(根, 屏蔽s)
        self.桥 = 桥
        self.接口 = 统一输入(桥.codec, seed=self.种子)          # Phase2：全接口预处理挂载
        if self._自诊开:                                        # meeting43·P8：默认关，零影响
            _p43 = os.path.join(AHI, "meeting43", "实施")
            if _p43 not in sys.path:
                sys.path.insert(0, _p43)
            from 自诊 import 自诊器
            self.自诊 = 自诊器(桥, seed=self.种子)
        return self

    # ══════════ 记忆底座（容量桥 FSW 长时库） ══════════
    def _写经历(self, 类型: str, 文本: str, 来源: str = "对话槽"):
        if self.容量 is None:
            return
        try:
            self.容量.记经历(拍=getattr(self.桥, "拍", 0), 类型=类型,
                             文本=文本, 来源=来源)
        except Exception:
            pass

    def _自评(self):
        """它自己先掂量一下自己（所有者：「每句输出都要经过我」）——全部取自引擎当下遥测。"""
        try:
            e = getattr(self.桥, "e", None)
            return ("先掂量一下自己：第 %d 拍，能量 %s，记忆 %d 段，账本 %d 条，"
                    "消化债 %.2f，饱足 %.2f。" %
                    (int(getattr(self.桥, "拍", 0)),
                     ("%.2f" % float(e)) if isinstance(e, (int, float)) else "—",
                     len(getattr(getattr(self.桥, "codec", None), "_段", []) or []),
                     len(getattr(self.桥, "账本", []) or []),
                     float(getattr(self.桥, "消化债", 0.0)),
                     float(getattr(self.桥, "饱足", 1.0))))
        except Exception:
            return None

    def _回忆(self, 文: str):
        """收到消息的第一动作：从记忆里检索相关经历（所有者：「有没有说过这个片段」）。"""
        try:
            hits = self.桥._检索(文, k=2)
        except Exception:
            hits = []
        if not hits:
            return None
        段, 分 = hits[0]
        阈值 = float(getattr(self.桥, "阈值", 0.2) or 0.2)
        if float(分) < 阈值:
            return None
        return "我回忆起来一段：『%s』（相似 %.2f）。" % (str(段)[:48], float(分))

    # ══════════ 回合：我一句 → 它想 + 回忆 → 回一句 ══════════
    def 回合(self, 文: str, 谁: str = "我", 向量=None, 会话: str = "主") -> dict:
        """一轮对话（先想在内部、再说出口）。会话= 哪个窗口（默认「主」）——
        内部念头（自评/回忆/自言）写「脑内」通道，**不进对话流**（正常对话流只留对白）。"""
        # ① 先自我评估（「每句输出都要经过我」）→ 再回忆历史（记为「脑内」念头）
        自评 = self._自评()
        if 自评:
            self.态.写它想(自评, 会话=会话)
        回忆 = self._回忆(文)
        if 回忆:
            self.态.写它想(回忆, 会话=会话)
        # ② 脑「想一遍」：引擎步进 + 检索 + 决策（综合体内部完成，非工具调用）
        #    向量 is not None ⇒ 通用输入直驱（文本/音频特征/任意特征经统一输入层转来的向量）
        n0 = len(self.桥.自言.流)
        r = self.桥.步_向量(向量, 文, 谁=谁, 会话=会话) if 向量 is not None \
            else self.桥.步(文, 谁=谁, 会话=会话)
        for e in self.桥.自言.流[n0:]:
            self.态.写它想(e["文本"], 会话=会话)             # 内部管线条目 → 脑内（审计用）
            self.统计["自言外显"] += 1
        if self.自诊 is not None:                            # meeting43·P8：只在开启时旁路自诊
            try:
                self.自诊.记输入(向量 if 向量 is not None else self.桥.codec.编码(文))
                if self.统计["回合"] % self._自诊间隔 == 0:
                    self.自诊.快照([文])
            except Exception:
                pass
        # ③ 转文本 → 说出去（写「它说」并入表达槽，由嘴走纯文本通道）
        响应 = str(r.get("响应", ""))
        self.态.写它说(响应, 会话=会话)
        self.态.排队说(响应, 会话=会话)
        # ④ 入长时记忆底座（记忆=全生命共享，与窗口无关；"说话人"用于归属）
        self._写经历("你说", 文, 来源=str(谁))
        self._写经历("它说", 响应)
        self.统计["回合"] += 1
        # ⑤ 自发承载（Phase3·自愿）：升维**不是**宿主每轮无条件强加给它的条件——
        #    而是**生命自己在「步」里自审**「学得多了、快装不下」后写下的自愿请求
        #    （synthesis.自审承载：落成它自己的「计划」自言 + 「自我」账本）。
        #    宿主只做**机械执行**：照它留下的 待扩容 请求重建，不做任何条件判断、
        #    不代替它决定要不要长。这就是「代码自愿 vs 硬性条件约束」的分界。
        self._执行扩容()
        return dict(行为="回合", 响应=响应, e=r.get("e"), 新异=r.get("ñ"), 会话=会话)

    # ══════════ 通用输入/输出（Phase2：任一模态进、任一通道出） ══════════
    def 入(self, 源, 模态: str = "文本", 标签: str = None, 谁: str = "我",
           会话: str = "主") -> dict:
        """通用入口：任一模态 → 一轮对话（先想后说）。
        文本走 codec 编码；向量/音频特征/任意特征经「全接口预处理」转成同一 D 维向量后直驱。
        眼/耳/手冻结：本方法不接任何器官——前端自己产出特征即可（契约见 通用接口.py）。"""
        d = self.接口.预处理(源, 模态)
        if d["文本"] is not None and 模态 == "文本":
            self.态.写用户说(d["文本"], 谁=谁, 会话=会话)      # 文本：走既有对话槽（判据 V1–V6 不变）
            return self.一拍(会话=会话)
        文 = d.get("文本") or (标签 or "（一段没有语词的感觉）")
        return self.回合(文, 谁=谁, 向量=d["向量"], 会话=会话)

    def 出(self, 文本: str, 目标: str = "文本", 会话: str = "主") -> dict:
        """通用输出：文本 → 目标通道。文本目标=入表达槽（嘴走纯文本流）；其它目标=已注册钩子。
        未注册钩子 ⇒ 确定性降级为文本（不假装说了——诚实边界）。"""
        d = self.输出.输出(文本, 目标)
        if d["通道"] == "文本":
            self.态.写它说(文本, 会话=会话)                    # 保持动作回流（与嘴同律）
            self.态.排队说(文本, 会话=会话)
        return d

    def 注册通道(self, 模态: str, 钩子) -> None:
        """注册一个输出通道（未来「说话」= 注册 TTS 钩子；本文件不 import 声带）。"""
        self.输出.注册钩子(模态, 钩子)

    # ══════════ 自主：它自己作主（想 / 睡） ══════════
    def 自主一拍(self, 会话: str = "主") -> dict:
        """自主一拍：它自己作主。引擎每拍照常步进（内在活动不停），

        「醒 / 睡」由引擎**这一拍有没有自己冒出念头**决定——**不给任何阈值/常量**：
          · 冒出念头（新自言）＝ 有想头 → 醒（念头写「脑内」，**不进对话流**）；
          · 没冒念头（心里静）＝ 没得想 → 歇（睡，不外显；但仍在对世界做事）。
        念头之所以会/不会冒出来，由引擎自身内驱塑造（饱足/消化债经 κd 偏置动作选择，
        见 synthesis._引擎步），故**睡/醒是内在动力学的自然涌现**——既不是定时器，
        也不是「建议阈值」；所有者「不想睡，想东西就给他想 / 实在没得想就自己睡着」
        即此判据（『有想头』与『没得想』是 0 边界的存在/缺席，不是调出来的数）。

        多窗口：会话= 本次自主拍挂靠的窗口（默认「主」）；延答投递回到它提问的窗口。"""
        n0 = len(self.桥.自言.流)
        try:
            r = self.桥.断供一拍()
        except Exception as ex:
            return dict(行为="想", 错=repr(ex))
        投递 = r.get("投递") if isinstance(r, dict) else None   # 延答作答：思考完成回传
        回窗 = (r.get("会话") if isinstance(r, dict) else None) or 会话   # 答完回原窗口
        新 = self.桥.自言.流[n0:]
        有念头 = len(新) > 0
        醒过 = self.态.对话态(会话=会话)["醒"]
        if 有念头:
            self.态.置醒(1)
            for e in 新:                                     # 有想头 → 内部外显（不进对话流）
                self.态.写它想(e["文本"], 会话=会话)
                self.统计["自言外显"] += 1
            self.统计["醒拍"] += 1
            行为 = "想" if 醒过 else "醒"
        elif 投递:
            self.态.置醒(1)                                  # 想好了要答你 → 也算有想头
            self.态.写它想("（想好了——答你刚才那句。）", 会话=回窗)
            self.统计["醒拍"] += 1
            行为 = "答"
        else:
            self.态.置醒(0)                                  # 没得想 → 它自己歇（无任何时间/数值阈值）
            self.统计["睡拍"] += 1
            行为 = "睡"
        if 投递:                                             # 把想好的作答说出去（先想后说）
            self.态.写它说(投递, 会话=回窗)
            self.态.排队说(投递, 会话=回窗)
            self._写经历("它说", 投递)
            self.统计["延答"] += 1
        self._执行扩容()                                     # 生命自主时自愿提出的承载请求，同样照办
        self.统计["自主拍"] += 1
        return dict(行为=行为, 通道=r.get("通道"), 醒=(有念头 or bool(投递)), 会话=会话)

    # ══════════ 一拍：有用户消息→回合，否则→自主 ══════════
    def 一拍(self, 会话: str = "主") -> dict:
        got = self.态.取用户说(会话=会话)
        if got is not None:
            self.态.置醒(1)                                  # 人来了 → 醒
            return self.回合(got[1], 谁=got[2], 会话=会话)
        return self.自主一拍(会话=会话)

    def 多窗一拍(self) -> dict:
        """多窗口分发一拍：把**所有**窗口的未读消息各走一轮（回合）；无消息的窗口不空转。
        返回 {会话: 行为}。用于"一个我同时回几个窗口"。"""
        报 = {}
        待 = self.态.待处理会话()
        for 会话 in 待:
            报[会话] = self.一拍(会话=会话)["行为"]
        return 报

    # ══════════ 快照（判据/审计用） ══════════
    def 快照(self, 对话上限: int = 200, 会话: str = "主") -> dict:
        return dict(统计=dict(self.统计), 对话态=self.态.对话态(会话=会话),
                    对话流=self.态.取对话(上限=对话上限, 会话=会话),
                    脑内=self.态.取脑内(上限=对话上限, 会话=会话),
                    会话们=self.态.会话们(),
                    自言总=len(self.桥.自言.流) if self.桥 else 0,
                    记忆库=self.容量.库大小() if self.容量 else 0)

    # ══════════ 持久化（事件重放式：同 seed+同事件序列=逐位再生） ══════════
    def 保存(self, 路径: str) -> str:
        """保存记忆：综合体（教史/事件史/开放域根）事件重放式入档。"""
        self.桥.保存(路径)
        return 路径

    def 载入(self, 路径: str) -> "文本态":
        """从档恢复记忆：重建综合体（重放事件，记忆段/账本/回复台账逐位再生）。"""
        from synthesis import 综合体                             # noqa: E402
        self.桥 = 综合体.载入(路径)
        self.接口 = 统一输入(self.桥.codec, seed=self.种子)      # 接口随 codec 重建（维/探针同步）
        return self

    # ══════════ 向量空间扩容（Phase2 手动原语；Phase3 自动化见 维度伸缩.py） ══════════
    def 扩容(self, 新D: int) -> dict:
        """把向量空间升降到 新D（唯一真相源=lang_codec.D）：升维(桥) → 事件重放 → 记忆再生。
        确定性（同 seed+同事件序列=逐位再生）；记忆保留（段数守恒，检索口径不变）。"""
        from 通用接口 import 记忆快照
        旧D = int(self.桥.codec._置换s[0].shape[0]) if self.桥 else 1024
        旧 = 记忆快照(self.桥) if self.桥 else {}
        self.桥 = 升维(self.桥, 新D)
        self.接口 = 统一输入(self.桥.codec, seed=self.种子)
        新 = 记忆快照(self.桥)
        return dict(旧D=旧D, 新D=int(新D), 旧=旧, 新=新,
                    段守恒=(旧.get("段数") == 新.get("段数")))

    def _执行扩容(self) -> dict:
        """机械执行生命自审后**自愿**提出的扩容请求（Phase3·自愿面）。

        宿主**不做任何条件判断**——不按学习量推算、不在每轮强推升维；它只把生命在自身节律
        （「步」学完一句 / 断供自审）里留下的 `待扩容` 请求如实重建。要不要长大，由生命自己
        自审（synthesis.自审承载）；宿主无权替它决定。旧的外部硬性 `维度自适()` 已删除。
        """
        if self.桥 is None:
            return dict(扩容=False)
        目 = getattr(self.桥, "待扩容", None)
        if not 目:
            return dict(扩容=False)                          # 生命觉得自己还装得下 → 原地不动
        self.桥.待扩容 = None                                # 请求已被受理（旧桥上清空）
        报 = self.扩容(int(目))                              # 升维=事件重放重建（记忆逐位再生）
        self.统计["升维"] = self.统计.get("升维", 0) + 1
        return dict(扩容=True, 新D=int(目), 报=报)

    # ══════════ 嘴：独立发声循环（与眼/手同一并发范式：只读写槽位，互不调用） ══════════
    def 起嘴(self) -> None:
        if self._嘴线程 is None:
            self._嘴线程 = threading.Thread(target=self.嘴.运行, args=(self.停,),
                                            name="文本态-嘴", daemon=True)
            self._嘴线程.start()

    def 停机(self) -> None:
        """停机：置停 + 收嘴线程（避免 daemon 线程在解释器退出时抢 stdout 锁而崩）。"""
        self.停.set()
        try:
            if self._嘴线程 is not None and self._嘴线程.is_alive():
                self._嘴线程.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.嘴.卒()
        except Exception:
            pass

    # ══════════ 运行：脚本（判据） / 交互（真人） ══════════
    def 运行_脚本(self, 消息s, 每轮自主拍: int = 2, 结尾自主拍: int = 60) -> dict:
        """跑一段脚本。每项可写：
          · "文本"               → 主窗口、谁="我"
          · (谁, 文)             → 主窗口、指定发言人
          · (会话, 谁, 文)       → 指定窗口 + 发言人（多窗口交错对话流）
        轮间/收尾自主拍挂在该项所属窗口（多窗口下各窗口各自生长）。"""
        self.起嘴()
        for 项 in 消息s:
            if isinstance(项, (tuple, list)) and len(项) >= 3:
                会话, 谁, 文 = 项[0], 项[1], 项[2]
            elif isinstance(项, (tuple, list)) and len(项) == 2:
                会话, 谁, 文 = "主", 项[0], 项[1]
            else:
                会话, 谁, 文 = "主", "我", 项
            self.态.写用户说(文, 谁=谁, 会话=会话)
            self.一拍(会话=会话)                                # 回合（该窗口）
            for _ in range(int(每轮自主拍)):
                self.一拍(会话=会话)                            # 轮间自主
        for _ in range(int(结尾自主拍)):
            self.一拍(会话="主")                                # 收尾自主（可自行入睡）
        return self.快照()

    def 运行_多窗(self, 窗口脚本: dict, 每轮自主拍: int = 2,
                 结尾自主拍: int = 60) -> dict:
        """多窗口演示：窗口脚本 = {会话ID: [(谁,文), ...]}——**交错**推进各窗口对话流
        （一个"我"同时跟几拨人聊）。返回 {会话: 该窗口快照} + 记忆库大小（记忆是共享的）。"""
        self.起嘴()
        名s = list(窗口脚本.keys())
        最 = max((len(v) for v in 窗口脚本.values()), default=0)
        for i in range(最):
            for 会话 in 名s:
                项s = 窗口脚本[会话]
                if i < len(项s):
                    项 = 项s[i]
                    谁, 文 = 项 if isinstance(项, (tuple, list)) else ("我", 项)
                    self.态.写用户说(文, 谁=谁, 会话=会话)
                    self.一拍(会话=会话)
                    for _ in range(int(每轮自主拍)):
                        self.一拍(会话=会话)
        for _ in range(int(结尾自主拍)):
            self.一拍(会话="主")
        return dict(每窗={s: self.快照(会话=s) for s in 名s},
                    记忆库=self.容量.库大小() if self.容量 else 0,
                    会话们=self.态.会话们())

    def 运行_交互(self):
        print("AHI 纯文本对话态 —— 直接打字说话；/save、/load 存取记忆；/q 退出。", flush=True)
        self.起嘴()

        def _读入():
            try:
                for 行 in sys.stdin:
                    命 = str(行).rstrip("\n").strip()
                    if 命 in ("/q", "/quit", "退出"):
                        break
                    if 命.startswith("/save"):
                        p = 命[5:].strip() or os.path.join(HERE, "_文本态记忆.pkl")
                        try:
                            self.保存(p)
                            print("[存档] OK → %s" % p, flush=True)
                        except Exception as ex:
                            print("[存档] 失败: %r" % ex, flush=True)
                        continue
                    if 命.startswith("/load"):
                        p = 命[5:].strip() or os.path.join(HERE, "_文本态记忆.pkl")
                        try:
                            self.载入(p)
                            print("[读档] OK → %s（记忆段 %d）" %
                                  (p, len(self.桥.codec._段)), flush=True)
                        except Exception as ex:
                            print("[读档] 失败: %r" % ex, flush=True)
                        continue
                    if 命:
                        self.态.写用户说(命)
            finally:
                self.停.set()                                 # EOF/退出 → 停机

        threading.Thread(target=_读入, name="文本态-读入", daemon=True).start()
        while not self.停.is_set():
            try:
                self.一拍()
            except Exception:
                pass
            self.停.wait(0.05)
        self.嘴.卒()


def main():
    态 = 文本态(种子=310).建()
    态.运行_交互()


if __name__ == "__main__":
    main()