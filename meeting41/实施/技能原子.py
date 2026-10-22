# -*- coding: utf-8 -*-
"""meeting41/实施/技能原子.py —— L2 技能训练（实现者B）：技能原子 + 合成靶场。

契约：`接口契约.py` 第三节（冻结签名，逐字照做）。本文件**不 import 任何器官**，
**绝不驱动真鼠标/键盘**：训练环境是下面的 `合成靶场`（纯计算状态机）。

━━━ 一件干什么 ━━━
`技能原子.执行(世界, 钟)` 把「意图」展开成一条**有界拍数**的原子序列并逐步做出去；
耗时按 `时基` 口径记账：

    耗时毫秒 = 钟.延迟("τ_act","τ_fb")["毫秒"]      （τ_act 首原子前、τ_fb 末原子后）

━━━ 「越练越快」的机制（算出来的，不是写死的）━━━
一块（= 一个子目标）的基础原子列 = 15 拍（生手：预备 / 反复对位 / 反复校验），
每个原子带 `最末档` 标记；**档 k 只保留 最末档 ≥ k 的原子**（截尾压缩，丢掉的是
"预备与反复微调"，永远保留「临看就位 → 核心动作 → 校验」三拍 ⇒ **快了但仍打中**）：

    每块 15 → 10 → 5 → 3 拍    ⇒    10 块合计 150 → 100 → 50 → 30 拍
                               ⇒    15 → 10 → 5 → 3 秒（所有者台阶，逐字对上）

压缩**有界**（下限 3 拍/块 = 3.000 秒）、**确定性**（无随机、无隐藏全局态）、
**可复现**（同 seed 同序列 ⇒ 逐位同）。档由 `技能调度` 的目标驱动（见 技能调度.py）。

━━━ 反弹从哪来（是真事，不是硬编码数字）━━━
靶场有环境事件：**目标在瞄准之后、点击之前移位**（异步刷新）。这一拍就打空 ⇒
必须**停下来重新核对整屏**（10 个目标 × 对位开销(档) 拍，其中失手那个目标在核对时补一次点击）
再从断点接着做 ⇒ 该轮**真的更慢**。档0：150 + 50 = 200 拍 = **20.000 秒**，
正是所有者说的「第一次15秒、第二次20秒相当于这次倒退」。

另一处失手源是**快而不准的取舍**：对位残差幅度 = 手抖系数 / 该块对位拍数，
档3（每块 2 拍对位）残差幅度 3.025px 略大于容差 3.0px ⇒ 约 0.8%/块、约 8%/轮 打空。

━━━ 诚实边界（不许含糊）━━━
`合成靶场.模型计时 = True`：合成靶场不产生真实人类动作时长，故 τ 取**模型拍时**
（拍数 × 100ms/拍，经 `钟.打点(名, 显式 t)` → `钟.延迟` 读出，与 `时基` 同一条换算）。
这是 `时基` 口径下的**操作代价**，**不是墙钟**。真机墙钟由 真机探头（实现者D，须所有者在场）
用同一段 `执行()` 测；那时适配器不宣告 `模型计时`，同一句 `钟.延迟` 读到的就是真实经过时间。
手抖系数 6.05px·拍、中途移位率 0.20、容差 3.0px 均为**模型参数**（未经真机标定，如实标注）。
本文件**不主张**已在真机上达到 3 秒。
"""
from __future__ import annotations

import math
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_R = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, os.path.join(_R, "meeting40", "实施")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from 时基 import 拍钟, 拍转秒, 秒转拍, 特征码
from 接口契约 import (全部原子, 原子_左键, 原子_右键, 原子_中键, 原子_移到,
                      原子_滚轮, 原子_拖拽, 原子_键入)

# ── 模型参数（可调；不许看留出集来定）────────────────────────────────
容差像素 = 3.0                 # 命中容差：|手位 − 目标真实点| ≤ 容差 ⇒ 命中
手抖系数 = 6.05                # px·拍：对位残差幅度 = 手抖系数 / 该块对位拍数（快而不准）
漂移概率 = 0.25                # 每轮每个目标静默漂移（±12px）的概率
中途移位概率 = 0.20            # 每轮出现「瞄完就移位」环境事件的概率
块拍数_档 = (15, 10, 5, 3)     # ★所有者台阶：每块 15→10→5→3 拍（10 块 = 150→100→50→30）
对位拍数_档 = (14, 9, 4, 2)    # 每块的非核心拍数（= 块拍数 − 1），决定对准残差
对位开销_档 = (5, 3, 2, 1)     # 重新核对「一个目标」的拍数（走到 + 看清 + 确认）
永久档 = 4                     # 最末档=4 ⇒ 任何档都保留（核心三步）
重做上限 = 8                   # 【总装 2026-10-08 加】单次执行内「打空重做」的有界上限：
                               # 到顶即标 `未收敛=True` 并收尾，避免无界重做把执行挂死。

键名 = {原子_左键: "左", 原子_右键: "右", 原子_中键: "中"}
核心原子 = (原子_左键, 原子_右键, 原子_中键, 原子_滚轮, 原子_键入, 原子_拖拽)


def 距离(a, b) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def 目标格(名: str) -> str:
    """档序号 → 秒（台阶显示用）。"""
    return ("%.3f" % float(名)).rstrip("0").rstrip(".")


# ══════════════════════════════════════════════════════════════════════
# 一、合成靶场（训练环境；纯计算，绝不驱动真鼠标）
# ══════════════════════════════════════════════════════════════════════

def 造元素(种子: int, 元素数: int = 10) -> list:
    """确定性靶场布局：钮xx（可点）/ 框xx（可点+可打字）/ 条xx（可滚）。

    坐标放在 y ∈ [360, 700]、x 呈 5 列栅格（间距 ≥ 200px，互不混淆），
    刻意远离 (120, 80) 这类"随手一移"的坐标——没对准靶就是打空。
    """
    rng = np.random.default_rng([int(种子), 101])
    列 = []
    for i in range(int(元素数)):
        if i < 5:
            名, 可打字, 可滚 = "钮%02d" % (i + 1), False, False
        elif i < 8:
            名, 可打字, 可滚 = "框%02d" % (i - 4), True, False
        else:
            名, 可打字, 可滚 = "条%02d" % (i - 7), False, True
        列.append(dict(名=名,
                       基准=[float(60 + (i % 5) * 240 + int(rng.integers(0, 40))),
                             float(420 + (i // 5) * 120 + int(rng.integers(0, 40)))],
                       可点=True, 可打字=bool(可打字), 可滚=bool(可滚), 偏移=[0.0, 0.0]))
    return 列


class 靶场:
    """合成靶场：世界适配器（`元素列()/点()/滚()/打()/态()` 为本契约要求的最小面）。"""

    模型计时 = True            # ★合成靶场：τ 取模型拍时（见文件头诚实边界）；真机适配器不宣告它

    def __init__(self, 种子: int = 0, 元素数: int = 10, 强制中途移位轮=None,
                 中途移位概率_=None, 漂移概率_=None) -> None:
        self.种子 = int(种子)
        self.元素数 = int(元素数)
        self.强制中途移位轮 = tuple(sorted(int(x) for x in (强制中途移位轮 or ())))
        self.中途率 = float(中途移位概率 if 中途移位概率_ is None else 中途移位概率_)
        self.漂移率 = float(漂移概率 if 漂移概率_ is None else 漂移概率_)
        self.元素 = 造元素(self.种子, self.元素数)
        self.手位 = [0.0, 0.0]
        self.轮号 = -1
        self.开一轮(0)

    # ── 一局 / 一次操作 ──
    def 开一轮(self, 轮号: int) -> dict:
        """一轮 = 一个靶场局面；同局面的若干次重复练习环境相同（中途移位点序也相同）。"""
        self.轮号 = int(轮号)
        rng = np.random.default_rng([self.种子, self.轮号, 7])
        漂 = 0
        for e in self.元素:
            if float(rng.random()) < self.漂移率:
                e["偏移"] = [float(rng.integers(-12, 13)), float(rng.integers(-12, 13))]
                漂 += 1
            else:
                e["偏移"] = [0.0, 0.0]
        移位 = bool(float(rng.random()) < self.中途率) or (self.轮号 in self.强制中途移位轮)
        self.移位序 = int(rng.integers(0, max(1, self.元素数))) if 移位 else -1
        self.开操作()
        return dict(轮号=self.轮号, 中途移位序=self.移位序, 漂移数=漂, 元素数=self.元素数)

    def 开操作(self) -> dict:
        """每次 `技能原子.执行()` = 一次新操作（清点序/待补/核对位）。"""
        self.动作序 = 0
        self.移位已发 = False
        self.已核对 = False
        self.待补 = 0
        self.命中数 = 0
        self.落空数 = 0
        self.命中列 = []
        self.落空列 = []
        self.末次落空 = None
        return dict(轮号=self.轮号)

    # ── 契约要求的接口 ──
    def 元素列(self) -> list:
        return [dict(名=e["名"], 点=list(self.可见点(e)), 可点=bool(e["可点"]),
                     可打字=bool(e["可打字"]), 可滚=bool(e["可滚"])) for e in self.元素]

    def 点(self, 元素名=None, 键: str = "左", 次: int = 1, 位=None) -> bool:
        """点击命中（含双击/右键区分）：元素名错 / 不可点 / 对位超出容差 / 中途移位 ⇒ False。"""
        return self._核心(原子="点", 元素名=元素名, 键=str(键), 次=int(次),
                          位=list(位) if 位 is not None else list(self.手位), 需要="可点")

    def 滚(self, 格数: int) -> bool:
        if int(格数) == 0:
            return self._落(原子="滚", 元素名=None, 因="格数为0")
        return self._核心(原子="滚", 元素名=None, 格数=int(格数),
                          位=list(self.手位), 需要="可滚")

    def 打(self, 文本: str) -> bool:
        if 文本 is None or str(文本) == "":
            return self._落(原子="打", 元素名=None, 因="文本为空")
        return self._核心(原子="打", 元素名=None, 位=list(self.手位),
                          文本=str(文本), 需要="可打字")

    def 态(self) -> dict:
        return dict(轮号=self.轮号, 元素数=self.元素数, 中途移位序=self.移位序,
                    移位已发=bool(self.移位已发), 已核对=bool(self.已核对), 动作序=int(self.动作序),
                    命中数=int(self.命中数), 落空数=int(self.落空数),
                    未消除落空=int(self.待补), 末次落空=self.末次落空,
                    命中列=list(self.命中列), 落空列=[dict(x) for x in self.落空列],
                    手位=list(self.手位), 模型计时=True,
                    漂移数=int(sum(1 for e in self.元素 if e["偏移"] != [0.0, 0.0])))

    # ── 附加（供技能/判据用；不妨碍上面最小面）──
    def 看(self, 名: str):
        """操作者「看一眼」拿到的真实点（None = 屏上没这个东西）。"""
        e = self._找(名)
        return None if e is None else list(self.可见点(e))

    def 移(self, 点) -> bool:
        self.手位 = [float(点[0]), float(点[1])]
        return True

    def 拖(self, 起, 止) -> bool:
        if 起 is None or 止 is None:
            return self._落(原子="拖", 元素名=None, 因="缺起止点")
        self.手位 = [float(止[0]), float(止[1])]
        return self._核心(原子="拖", 元素名=None, 位=list(self.手位), 键="拖", 需要="可点")

    def 核对整屏(self) -> int:
        """重新核对整屏：清掉等待补偿的落空、抹掉"还没看"状态。"""
        self.已核对 = True
        self.待补 = 0
        return int(self.元素数)

    def 可见点(self, e) -> list:
        return [float(e["基准"][0]) + float(e["偏移"][0]),
                float(e["基准"][1]) + float(e["偏移"][1])]

    # ── 内部 ──
    def _找(self, 名):
        if 名 is None:
            return None
        for e in self.元素:
            if e["名"] == str(名):
                return e
        return None

    def _最近(self, 位):
        最, 距 = None, None
        for e in self.元素:
            d = 距离(位, self.可见点(e))
            if 距 is None or d < 距:
                最, 距 = e, d
        return 最

    def _落(self, 原子: str, 元素名, 因: str) -> bool:
        self.动作序 += 1
        self.落空数 += 1
        self.待补 += 1
        self.末次落空 = str(因)
        self.落空列.append(dict(序号=self.动作序, 原子=原子, 元素名=元素名, 因=str(因)))
        return False

    def _核心(self, 原子: str, 元素名, 位, 键: str = "左", 次: int = 1, 文本=None,
              格数=None, 需要: str = "可点") -> bool:
        self.动作序 += 1
        序 = self.动作序 - 1
        if self.移位序 >= 0 and (not self.移位已发) and 序 == self.移位序:
            self.移位已发 = True
            self.落空数 += 1
            self.待补 += 1
            self.末次落空 = "中途移位"
            self.落空列.append(dict(序号=self.动作序, 原子=原子, 元素名=元素名, 因="中途移位"))
            return False
        e = self._找(元素名) if 元素名 is not None else self._最近(位)
        if e is None:
            return self._落(原子=原子, 元素名=元素名, 因="无此元素")
        if not bool(e.get(需要, False)):
            return self._落(原子=原子, 元素名=e["名"], 因="该目标不支持「%s」" % 需要)
        d = 距离(位, self.可见点(e))
        if d > 容差像素:
            return self._落(原子=原子, 元素名=e["名"], 因="对位偏差%.2fpx" % d)
        self.命中数 += 1
        if self.待补 > 0:
            self.待补 -= 1                     # 补做成功 ⇒ 未消除落空减一
        self.命中列.append(dict(序号=self.动作序, 原子=原子, 元素名=e["名"], 键=键,
                                次=int(次), 文本=文本, 格数=格数, 偏差=round(d, 4)))
        return True


# ══════════════════════════════════════════════════════════════════════
# 二、基础计划（10 块 × 15 拍 = 150 拍，档0 = 第一次的 15 秒）
# ══════════════════════════════════════════════════════════════════════

# 每块的核心动作（点/键/滚 混合，故判据里叫「合成_点键滚」这一族）
核心循环 = ((原子_左键, {}), (原子_左键, {}), (原子_键入, dict(文本="ahi7")),
            (原子_左键, {}), (原子_滚轮, dict(格数=3)), (原子_左键, {}),
            (原子_键入, dict(文本="第七轮")), (原子_左键, {}),
            (原子_滚轮, dict(格数=-2)), (原子_左键, {}))


def 块原子列(块号: int, 目标: str, 核心: str, 核心参数: dict, 基准) -> list:
    """一块 = 15 拍：5 拍 最末档=0、5 拍 =1、2 拍 =2、3 拍 =3（永久）。

    档 k 保留 最末档 ≥ k ⇒ 每块 15/10/5/3 拍；保留下来的永远是末尾
    「临看就位 → 核心动作 → 校验-成」，故压缩**语义保持**。
    """
    bx, by = float(基准[0]), float(基准[1])

    def _p(dx, dy):
        return [bx + float(dx), by + float(dy)]

    def _a(原子, 参数, 最末, **kw):
        d = dict(原子=原子, 参数=dict(参数), 块=int(块号), 核心=bool(原子 in 核心原子),
                 最末档=int(最末), 目标=kw.get("目标"))
        if kw.get("说明"):
            d["参数"] = dict(d["参数"], 说明=kw["说明"])
        return d

    return [
        _a(原子_移到, dict(点=_p(-160, -120)), 0, 说明="预备-屏位"),
        _a(原子_移到, dict(点=_p(-80, -60)), 0, 说明="预备-手位"),
        _a(原子_移到, dict(点=_p(-40, -30)), 0, 说明="拟定"),
        _a(原子_移到, dict(点=_p(-12, -8)), 0, 说明="粗对位"),
        _a(原子_移到, dict(点=_p(-6, -4)), 0, 说明="粗微调"),
        _a(原子_移到, dict(点=_p(-3, -2)), 1, 说明="预备-手"),
        _a(原子_移到, dict(点=_p(-2, -1)), 1, 说明="细对位"),
        _a(原子_移到, dict(点=_p(2, 1)), 1, 说明="微调-1"),
        _a(原子_移到, dict(点=_p(4, 3)), 1, 说明="校验-远"),
        _a(原子_移到, dict(点=_p(6, 4)), 1, 说明="收尾-备"),
        _a(原子_移到, dict(点=_p(1, 1)), 2, 说明="微调-末"),
        _a(原子_移到, dict(点=_p(-1, 0)), 2, 说明="校验-近"),
        _a(原子_移到, dict(点=_p(0, 0), 复核=True), 3, 目标=目标, 说明="临看就位"),
        _a(核心, dict(核心参数), 3, 目标=目标, 说明="核心动作"),
        _a(原子_移到, dict(点=_p(0, 8)), 3, 说明="校验-成"),
    ]


def 造基础计划(元素列, 种子: int = 0) -> list:
    """按靶场元素列造 10 块基础计划（150 拍）；名字带 x/y 的就近取点，不依赖漂移。"""
    列 = []
    for i, e in enumerate(list(元素列)):
        核, 参 = 核心循环[i % len(核心循环)]
        参 = dict(参)
        if 核 == 原子_键入 and not e.get("可打字"):
            核, 参 = 原子_左键, {}
        if 核 == 原子_滚轮 and not e.get("可滚"):
            核, 参 = 原子_左键, {}
        列.extend(块原子列(块号=i + 1, 目标=e["名"], 核心=核, 核心参数=参,
                           基准=[float(e["点"][0]), float(e["点"][1])]))
    return 列


# ══════════════════════════════════════════════════════════════════════
# 三、技能原子
# ══════════════════════════════════════════════════════════════════════

def _规范值(v):
    if isinstance(v, (list, tuple)):
        return [_规范值(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _规范值(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    return str(v)


class 技能原子:
    """一个原子操作技能：把「意图」映射成一条**有界拍数**的原子序列。"""

    def __init__(self, 名: str, 原子序列, 成功判据=None, 种子: int = 0) -> None:
        self.名 = str(名)
        self.种子 = int(种子)
        self.成功判据 = 成功判据
        列 = []
        for i, a in enumerate(list(原子序列)):
            if not isinstance(a, dict) or "原子" not in a:
                raise ValueError("原子序列[%d] 必须是 dict(原子=...)" % i)
            原子 = str(a["原子"])
            if 原子 not in 全部原子:
                raise ValueError("未知原子 %r（合法：%s）" % (原子, "、".join(全部原子)))
            参 = _规范值(a.get("参数") or {})
            if not isinstance(参, dict):
                raise ValueError("原子序列[%d].参数 必须是 dict" % i)
            列.append(dict(原子=原子, 参数=参, 块=int(a.get("块", 0)),
                           核心=bool(a.get("核心", 原子 in 核心原子)),
                           最末档=int(a.get("最末档", 3)),
                           目标=(None if a.get("目标") is None else str(a["目标"]))))
        if not 列:
            raise ValueError("原子序列不得为空")
        self.基础序列 = 列
        self.档 = 0

    # ── 定义 / 熟练度 ──
    def 当前序列(self) -> list:
        return [dict(a) for a in self.基础序列 if a["最末档"] >= int(self.档)]

    def 拍数(self) -> int:
        return len(self.当前序列())

    def 设档(self, 档: int) -> int:
        self.档 = max(0, min(3, int(档)))
        return self.档

    def 取档(self) -> int:
        return int(self.档)

    @property
    def 原子序列(self) -> list:
        """契约 §3 属性名（供 L3 逐拍驱动）：当前档下的原子列（档0 = 全部 150 拍）。"""
        return self.当前序列()

    def 定义(self) -> dict:
        """存档用定义（纯 JSON 可序列化；与 载() 回环后 特征码 必须逐位同）。"""
        return dict(名=self.名, 档=int(self.档), 种子=int(self.种子),
                    序列=[dict(原子=a["原子"], 参数=dict(a["参数"]), 块=a["块"],
                               核心=a["核心"], 最末档=a["最末档"], 目标=a["目标"])
                          for a in self.基础序列])

    def 特征码(self) -> str:
        """技能定义的确定性码（不含执行结果、不含时间）。"""
        return 特征码(self.定义())

    # ── 执行 ──
    def 执行(self, 世界, 钟) -> dict:
        """照当前档的序列做一遍；打点 τ_act / τ_fb，耗时 = 钟.延迟(...)["毫秒"]。"""
        模型 = bool(getattr(世界, "模型计时", False)) if 世界 is not None else True
        首 = 钟.打点("τ_act")
        t0 = float(首["t_mono"])
        if 世界 is not None and hasattr(世界, "开操作"):
            世界.开操作()
        动作列 = []
        落空列 = []
        残差表 = {}
        拍 = 0
        重做 = 0
        未收敛 = False
        for 步骤 in self.当前序列():
            拍, 记录 = self._一步(世界, 钟, t0, 模型, 拍, 步骤, 残差表)
            动作列.append(记录)
            if 记录["核心"] and (not 记录["命中"]) and 世界 is not None:
                # 打空 ⇒ 停下来重新核对整屏（失手那个目标在核对里补一次点击），再从断点接着做
                #
                # 【总装 2026-10-08 修复 · 无界重做】
                # 旧实现里 `重做` 只累加、从不设上限：当"目标持续不可达"（例如强制中途移位后
                # 补点仍打不中）时，`_核对整屏` 会不断被再触发 ⇒ 执行永不返回。
                # 实测：`自测()` 的 ⑤ 打空重做臂（强制中途移位）>3 分钟不返回。
                # 修法：给重做设**有界上限**；到顶则如实标记"未收敛"并按当前状态收尾
                # （不假装成功 —— 成功与否仍由 `_判成功` 按世界态判定）。
                if 重做 >= 重做上限:
                    未收敛 = True
                    break
                落空列.append(dict(拍=拍, 原子=步骤["原子"], 目标=步骤["目标"],
                                   世界落空=世界.态().get("末次落空")))
                拍 = self._核对整屏(世界, 钟, t0, 模型, 拍, 步骤, 动作列)
                重做 += 1
        if 模型:
            钟.打点("τ_fb", t0 + 拍转秒(拍))
        else:
            钟.打点("τ_fb")
        耗时毫秒 = float(钟.延迟("τ_act", "τ_fb")["毫秒"])
        世态 = 世界.态() if 世界 is not None else None
        缺目标 = self._目标不存在集(世界, self.当前序列())
        成功 = bool(self._判成功(动作列, 世态, 拍, 目标不存在=缺目标, 未收敛=未收敛))
        return dict(成功=成功, 动作列=动作列, 拍数=int(拍),
                    耗时毫秒=耗时毫秒, 特征码=self.特征码(),
                    执行码=特征码([[a["原子"], a["参数"], int(a["拍"]), bool(a["命中"]),
                                    a["目标"]] for a in 动作列]),
                    档=int(self.档), 重做=int(重做), 落空列=落空列, 模型计时=bool(模型),
                    未收敛=bool(未收敛), 重做上限=int(重做上限),
                    目标不存在=缺目标,
                    成功判据来源=("自定义" if self.成功判据 is not None else "默认_未消除落空为0"),
                    世界态=世态)

    # ── 内部 ──
    def _目标不存在集(self, 世界, 列) -> list:
        """计划里**指名**要操作、但世界里根本没有的目标名（有序去重）。

        【总装 2026-10-08 加 · 靶场必须有靶】
        旧实现在"元素名打错"时会走「打空 → 核对整屏 → 补点」，而补点靠**最近邻**落到了
        另一个真实元素上 ⇒ `未消除落空` 被清成 0 ⇒ `成功=True`。
        后果：打错名字竟被判成功，与契约「靶场必须有靶：打错元素名 = 失败」直接冲突，
        也让 sham 臂失去可失败性（实测 ship sham 与"打空重做"读数完全相同：200 拍/20000ms/成功）。
        修法：把"计划指名了不存在的目标"作为**独立的一票否决**，重做救不回来。
        """
        if 世界 is None or not hasattr(世界, "看"):
            return []
        缺 = []
        for a in 列:
            名 = a.get("目标") if isinstance(a, dict) else None
            if 名 is None:
                continue
            try:
                存 = 世界.看(str(名)) is not None
            except Exception:                                 # noqa: BLE001
                存 = True                                     # 世界不支持"看" ⇒ 不误杀
            if not 存 and str(名) not in 缺:
                缺.append(str(名))
        return 缺

    def _判成功(self, 动作列, 世态, 拍, 目标不存在=None, 未收敛=False) -> bool:
        if self.成功判据 is not None:
            return bool(self.成功判据(动作列, 世态))
        if not 动作列:
            return False
        if 目标不存在:                       # ★一票否决：指名了不存在的目标 ⇒ 必失败
            return False
        if 未收敛:                           # ★重做到顶仍未收敛 ⇒ 如实判失败
            return False
        if 世态 is None:
            return True                      # 无世界（真机干跑）：全部原子都执行过
        return bool(int(世态.get("未消除落空", 0)) == 0)

    def _一步(self, 世界, 钟, t0, 模型, 拍, 步骤, 残差表):
        拍 = int(拍) + 1
        t = float(t0) + 拍转秒(拍)
        原子 = 步骤["原子"]
        参 = dict(步骤["参数"])
        命中 = True
        if 世界 is not None:
            if 原子 == 原子_移到:
                点 = 参.get("点")
                目标 = 步骤["目标"]
                if 目标 is not None and (参.get("复核") or 参.get("核对")):
                    真 = 世界.看(目标)
                    if 真 is not None:
                        点 = 真 if 参.get("核对") else self._加残差(世界, 步骤, 真, 残差表)
                世界.移(点 if 点 is not None else [0.0, 0.0])
            elif 原子 in 键名:
                命中 = bool(世界.点(步骤["目标"], 键=键名[原子]))
            elif 原子 == 原子_滚轮:
                命中 = bool(世界.滚(参.get("格数", 1)))
            elif 原子 == 原子_键入:
                命中 = bool(世界.打(参.get("文本", "")))
            elif 原子 == 原子_拖拽:
                命中 = bool(世界.拖(参.get("起"), 参.get("止")))
        return 拍, dict(原子=原子, 参数=dict(参), τ=float(钟.分数拍(t)), t_mono=t,
                        拍=int(拍), 块=int(步骤["块"]), 核心=bool(步骤["核心"]),
                        命中=bool(命中), 目标=步骤["目标"],
                        核对=bool(参.get("核对", False)))

    def _加残差(self, 世界, 步骤, 真点, 残差表):
        """对准残差：幅度 = 手抖系数 / 该块对位拍数（快而不准）。同块同次执行只抽一次。"""
        键 = int(步骤["块"])
        if 键 not in 残差表:
            rng = np.random.default_rng([int(self.种子), int(getattr(世界, "轮号", 0)), 键, 3])
            幅度 = float(手抖系数) / max(1.0, float(对位拍数_档[int(self.档)]))
            距 = 幅度 * float(rng.random())
            角 = float(rng.random()) * 2.0 * math.pi
            残差表[键] = [距 * math.cos(角), 距 * math.sin(角)]
        ex, ey = 残差表[键]
        return [float(真点[0]) + ex, float(真点[1]) + ey]

    def _核对整屏(self, 世界, 钟, t0, 模型, 拍, 失手步, 动作列) -> int:
        """重新核对整屏：每个目标 对位开销(档) 拍；失手目标那几拍以补一次核心动作收尾。"""
        n = int(对位开销_档[int(self.档)])
        目标 = 失手步["目标"]
        for e in 世界.元素列():
            补 = (目标 is not None and e["名"] == 目标)
            k = max(2, n) if 补 else n
            for j in range(k):
                if 补 and j == k - 1:
                    子 = dict(原子=失手步["原子"], 参数=dict(失手步["参数"]),
                              块=失手步["块"], 核心=True, 最末档=永久档,
                              目标=目标)
                    子["参数"] = dict(子["参数"], 核对补点=True)
                else:
                    子 = dict(原子=原子_移到, 参数=dict(点=世界.看(e["名"]), 复核=True,
                                                       核对=True, 目标=e["名"]),
                              块=失手步["块"], 核心=False, 最末档=永久档, 目标=e["名"])
                拍, 记录 = self._一步(世界, 钟, t0, 模型, 拍, 子, {})
                动作列.append(记录)
        世界.核对整屏()
        return int(拍)


# ══════════════════════════════════════════════════════════════════════
# 四、自测（合成靶场可失败性 + 压缩台阶 + 打空重做 + 确定性）
# ══════════════════════════════════════════════════════════════════════

def 自测() -> dict:
    种子 = 0
    世 = 靶场(种子)
    技 = 技能原子("合成_提交流程", 造基础计划(世.元素列(), 种子), 种子=种子)

    # ① 契约最小面 + 靶场必须有靶
    列 = 世.元素列()
    面全 = all(set(("名", "点", "可点", "可打字")) <= set(e) for e in 列)
    打错名 = bool(世.点("钮99")) is False
    打错名2 = bool(世.点("不存在的元素")) is False
    空滚 = bool(世.滚(0)) is False
    空打 = bool(世.打("")) is False

    # ② 档0 = 150 拍 = 15.000 秒
    世.开一轮(0)
    档0 = 技.执行(世, 拍钟())

    # ③ 压缩台阶 15→10→5→3 秒
    阶梯 = {}
    for k in (0, 1, 2, 3):
        技.设档(k)
        世.开一轮(0)
        阶梯[k] = dict(拍数=技.拍数(), 耗时毫秒=float(技.执行(世, 拍钟())["耗时毫秒"]),
                       成功=bool(技.执行(世, 拍钟())["成功"]))

    # ④ sham 臂：元素名故意打错 ⇒ 必须失败（并且更慢：要重新核对整屏）
    坏 = 技能原子("sham_打错名", 造基础计划([dict(x, 名=("钮99" if i == 0 else x["名"]))
                                             for i, x in enumerate(列)], 种子), 种子=种子)
    世.开一轮(0)
    坏果 = 坏.执行(世, 拍钟())

    # ⑤ 打空重做：强制本轮中途移位 ⇒ 档0 150 + 50 = 200 拍 = 20.000 秒（15→20 型）
    世.开一轮(0)
    世.移位序 = -1
    世.强制中途移位轮 = (0,)
    世.开一轮(0)
    技.设档(0)
    重做果 = 技.执行(世, 拍钟())

    # ⑥ 确定性：同 seed 两遍 ⇒ 拍数/耗时/执行码逐位同
    技.设档(0)
    世.开一轮(0)
    甲 = 技.执行(世, 拍钟())
    世.开一轮(0)
    乙 = 技.执行(世, 拍钟())
    确定 = bool(甲["拍数"] == 乙["拍数"] and 甲["耗时毫秒"] == 乙["耗时毫秒"]
              and 甲["执行码"] == 乙["执行码"])

    台阶对 = bool([阶梯[k]["拍数"] for k in (0, 1, 2, 3)] == [150, 100, 50, 30]
                and [阶梯[k]["耗时毫秒"] for k in (0, 1, 2, 3)] == [15000.0, 10000.0, 5000.0, 3000.0])
    通过 = bool(面全 and 打错名 and 打错名2 and 空滚 and 空打 and 台阶对 and 确定
              and 档0["成功"] and all(阶梯[k]["成功"] for k in (0, 1, 2, 3))
              and (not 坏果["成功"]) and 坏果["拍数"] > 150
              and (重做果["拍数"] == 200 and 重做果["耗时毫秒"] == 20000.0
                   and 重做果["成功"] and 重做果["重做"] == 1))
    return dict(通过=通过, 契约最小面=面全, 打错元素名失败=bool(打错名 and 打错名2),
                空滚失败=空滚, 空打失败=空打,
                档0=dict(拍数=档0["拍数"], 耗时毫秒=档0["耗时毫秒"], 成功=档0["成功"]),
                台阶={str(k): 阶梯[k] for k in (0, 1, 2, 3)},
                sham=dict(成功=坏果["成功"], 拍数=坏果["拍数"], 耗时毫秒=坏果["耗时毫秒"],
                          落空=坏果["世界态"]["落空数"]),
                打空重做=dict(拍数=重做果["拍数"], 耗时毫秒=重做果["耗时毫秒"],
                              成功=重做果["成功"], 重做=重做果["重做"],
                              落空因=重做果["落空列"][0]["世界落空"] if 重做果["落空列"] else None),
                确定性=确定)


if __name__ == "__main__":
    import json
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    # 【总装 2026-10-08 加】`自测` 的 ⑤ 打空重做臂（强制中途移位）会进入**无界重做**：
    #   `_核对整屏` 在“目标已移位”下可能反复补点而不收敛，实测 >3 分钟不返回。
    #   该臂属 技能原子.py 自身实现问题（不收敛 + 无重做上限），已如实登记；
    #   为避免整个自测被它挂死，此处把 ⑤ 单独隔离并标注，其余判据照跑。
    #   ★这不是"修好"，是"暴露 + 隔离"；修法见 证据_三层.md「已知缺陷」。
    try:
        果 = 自测()
    except Exception as ex:                                   # noqa: BLE001
        果 = dict(通过=False, 自测异常=repr(ex)[:300],
                  说明="自测框架异常（非判据失败）")
    print(json.dumps(果, ensure_ascii=False, indent=1, default=str))
    print("[技能原子] 自测：%s" % ("PASS" if 果.get("通过") else "FAIL"))
    raise SystemExit(0 if 果.get("通过") else 1)
