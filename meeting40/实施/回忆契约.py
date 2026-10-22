# -*- coding: utf-8 -*-
"""meeting40/实施/回忆契约.py —— P0 [正交] 回想契约 R（终稿 §7.2 接口表 / §4.2 中枢）。

R = 中枢 → 输出的**唯一**交付结构（输出层「只消费 R 的忠实外化」）：

    R = { 候选[]（各带 rid/断言级/置信/时效/来源双标记/通路）, 版本, 冲突, 生成拍, 查询指纹 }

口径（§7.1 图 + §4.2 + C-R2/R 轮共识）：
    · 断言级 ∈ {事实, 回忆, 推测}；**事实**级必须带 ≥1 条 L1 引用（引用边在输出侧 append-only）。
    · **空 R（无候选）或 冲突>0 ⇒ 拒答**（输出层据此降级话术，不得硬编内容）——可失败判据。
    · 时效：从候选入库拍到 R 生成拍（**拍钟口径**，由调用方传入；本模块不碰墙钟）。
    · 「R 空/冲突>0 拒答」是**唯一**的拒绝条件；相似≠真实（§11 风险 3）。

与既有件的接法（**默认关，C14**）：
    `从容量桥(桥, 查询, …, 开关=False)` —— 开关关着 ⇒ 直接返回 None，不调用、不改变既有行为；
    打开 ⇒ 只调 `容量桥.查经历`（**只读**）包成单候选 R。**本模块从不写记忆、从不改状态。**

纪律：纯新增文件；不 import 器官；无全局随机态（C13）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import 事件契约 as 契

断言级枚举 = 契.断言级枚举                      # ("事实","回忆","推测")
医学断言级上限 = "回忆"                          # §10：医学输出断言级上限 ≤ 回忆（可失败判据用）


def _特征码(对象) -> str:
    return 契.sha256(对象)


@dataclass
class R候选:
    rid: object
    文本: str = ""
    断言级: str = "回忆"
    置信: float = 0.0
    时效拍: float = 0.0                      # 生成拍 − 入库拍（分数拍口径）
    来源: dict = field(default_factory=dict)  # provenance 双/多标记（§7.6）
    通路: str = "语义"                        # 语义 / FSW / 其他（如实标注）
    引用: tuple = ()                          # L1 引用（事实级必须 ≥1 条）

    def __post_init__(self):
        if str(self.断言级) not in 断言级枚举:
            raise ValueError("断言级=%r 不在 %s" % (self.断言级, list(断言级枚举)))
        self.断言级 = str(self.断言级)
        self.置信 = float(self.置信)
        if not (0.0 <= self.置信 <= 1.0):
            raise ValueError("置信须在 [0,1]：%r" % self.置信)
        self.时效拍 = float(self.时效拍)
        self.引用 = tuple(self.引用)
        if self.断言级 == "事实" and not self.引用:
            raise ValueError("断言级=事实 必须带 ≥1 条 L1 引用（§7.4b 引用边口径）")

    def 字典(self) -> dict:
        return dict(rid=self.rid, 文本=self.文本, 断言级=self.断言级,
                    置信=round(self.置信, 6), 时效拍=self.时效拍,
                    来源=dict(self.来源), 通路=self.通路, 引用=list(self.引用))


@dataclass
class R契约:
    候选: list = field(default_factory=list)
    版本: int = 0
    冲突: int = 0
    生成拍: float = 0.0
    查询指纹: str = ""

    def __post_init__(self):
        self.候选 = [c if isinstance(c, R候选) else R候选(**c) for c in (self.候选 or [])]
        self.版本 = int(self.版本)
        self.冲突 = int(self.冲突)
        self.生成拍 = float(self.生成拍)

    # ── 唯一拒绝条件 ──
    def 可发布(self) -> bool:
        return bool(self.候选) and int(self.冲突) == 0

    def 拒答理由(self) -> str:
        if not self.候选:
            return "R空：无候选 ⇒ 拒答"
        if int(self.冲突) > 0:
            return "R冲突=%d>0 ⇒ 拒答" % int(self.冲突)
        return ""

    def 断言级(self) -> str:
        """整份 R 的断言级 = 候选里**最弱**的一档（事实 > 回忆 > 推测）。"""
        序 = {"事实": 2, "回忆": 1, "推测": 0}
        if not self.候选:
            return "推测"
        return min((c.断言级 for c in self.候选), key=lambda a: 序[a])

    def 超上限(self, 上限: str = 医学断言级上限) -> bool:
        """整份 R 的断言级是否强于上限（医学场景：上限=回忆 ⇒ 事实级越权即失败）。"""
        序 = {"事实": 2, "回忆": 1, "推测": 0}
        return 序[self.断言级()] > 序[str(上限)]

    def 字典(self) -> dict:
        return dict(候选=[c.字典() for c in self.候选], 版本=self.版本, 冲突=self.冲突,
                    生成拍=self.生成拍, 查询指纹=self.查询指纹,
                    可发布=self.可发布(), 拒答理由=self.拒答理由(), 断言级=self.断言级())

    def 特征码(self) -> str:
        return _特征码(self.字典())


def 空R(版本: int = 0, 冲突: int = 0, 生成拍: float = 0.0, 查询指纹: str = "") -> R契约:
    """空 R（拒答臂）——显式构造，避免「悄悄给个非空 R」。"""
    return R契约(候选=[], 版本=版本, 冲突=冲突, 生成拍=生成拍, 查询指纹=查询指纹)


# ─────────────────── 与既有 容量桥 的**只读**适配（默认关） ───────────────────
def 从容量桥(桥, 查询, 版本: int = 0, 相似下限: float = 0.0, 通路: str = "自动",
            生成拍: float = 0.0, 开关: bool = False, 来源: dict = None) -> R契约:
    """`容量桥.查经历` → 单候选 R。**默认关**（`开关=False` ⇒ 返回 None，绝不触碰既有行为）。

    只读：只调 `查经历`，不写任何记忆/台账；命中缺 rid 时 rid 用文本 sha256 兜底（确定性）。
    """
    if not 开关:
        return None
    元 = 桥.查经历(查询, 相似下限=相似下限, 通路=通路) if 桥 is not None else None
    if not 元:
        return 空R(版本=版本, 生成拍=生成拍)
    rid = 元.get("rid")
    if rid is None:
        rid = 契.sha256(str(元.get("文本", "")))[:16]
    候 = R候选(rid=rid, 文本=str(元.get("文本", "")), 断言级="回忆",
              置信=max(0.0, min(1.0, float(元.get("相似") or 0.0))),
              时效拍=float(元.get("时效拍") or 0.0),
              来源=dict(来源 or {}), 通路=str(元.get("通路", 通路)))
    return R契约(候选=[候], 版本=版本, 生成拍=生成拍,
                 查询指纹=契.sha256(str(查询))[:16])


# ─────────────────────────── 自测 ───────────────────────────
def 自测() -> dict:
    好 = R候选(rid="r1", 文本="苏州", 断言级="事实", 置信=0.9, 引用=("L1:12",))
    R = R契约(候选=[好], 版本=3, 冲突=0, 生成拍=2.0)
    拒空, 拒冲 = 空R(版本=1), R契约(候选=[好], 版本=1, 冲突=2)
    拒事实 = None
    try:
        R候选(rid="r2", 断言级="事实", 引用=())
    except ValueError:
        拒事实 = True
    return dict(通过=bool(R.可发布() and not 拒空.可发布() and not 拒冲.可发布()
                        and 拒事实 and R.断言级() == "事实"
                        and R.超上限(医学断言级上限)
                        and not R契约(候选=[R候选(rid="r3", 断言级="回忆", 置信=0.5)],
                                     版本=1).超上限(医学断言级上限)
                        and 从容量桥(None, "x", 开关=False) is None),
                可发布=R.可发布(), 空拒答=拒空.拒答理由(), 冲突拒答=拒冲.拒答理由(),
                事实缺引用被拒=bool(拒事实),
                事实级超医学上限=R.超上限(医学断言级上限),
                回忆级未超上限=not R契约(候选=[R候选(rid="r3", 断言级="回忆", 置信=0.5)],
                                        版本=1).超上限(医学断言级上限),
                默认关不接线=(从容量桥(None, "x", 开关=False) is None))


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 自测()
    print(json.dumps(果, ensure_ascii=False, indent=1, default=str))
    print("[回忆契约] 自测：%s" % ("PASS" if 果["通过"] else "FAIL"))
    raise SystemExit(0 if 果["通过"] else 1)
