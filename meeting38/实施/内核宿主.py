# -*- coding: utf-8 -*-
"""meeting38/实施/内核宿主.py —— V3 ↔ meeting39 五内核 的唯一接线点（AHI3 修复）。

问题（`V3审计报告_逐条核对_20261007.md` P0-2）：
    meeting39 里**确实存在**五个内核（自模型 / 相空间算子T / 真学习 / 自我回路 / 顿悟涌现），
    但 `meeting38/` 全目录对它们的引用数 = **0** —— V3 主程序挂的是 `synthesis.综合体`
    **基类**，五内核从未被装配、从未运行。所有者点名的「真学习 / 深度思考 / 自我意识」
    因此在 V3 里**不存在**。

本文件把这条链挂上（**只加挂，不改基链**；内核本身的纪律见各 内核_*.py 文件头）：

    synthesis.综合体（基类）
        └ 内核_自模型.顶覆盖_综合体      —— ĝ:(S_t)→Ŝ_{t+1}，ε_self 自监督；独立学习存储
            └ 内核_相空间算子T.顶覆盖_综合体T —— 「想」= z←T(z) 相空间迭代（非检索）
                └ 内核_真学习.顶覆盖_综合体L   —— ε_self 经**既有内生偏置 b[]** 反哺决策
                    └ 内核_自我回路.顶覆盖_综合体Z —— 自指束 À_t + provenance 门控 W_self
                        └ 内核_顿悟涌现.顶覆盖_综合体W —— 相变共现取代 `if cosv>=0.8`

★五开关默认**全开**（本文件是 V3 的「自我意识配置」）；任一开关关 → 该层逐位恒等基类。
  「默认关」是内核自己的纪律（保护既有判据），不是 V3 的用法。

★零参数·自我调控：本文件**不引入任何人挑超参**（阈值/步长全部复用内核既有预注册常数）。

★provenance（专家03 J5「外源不可塑造我」）：`设来源(来源)` 决定当前回合是「自源」还是「外源」：
    · 用户对话（`步`/`步_向量`）→ 内核自身已标记为**外源**；
    · 它自己看网页 / 自主拍 → **自源**（它自己去看的，是它自己的经历）。
  V3 在喂网页正文前必须调 `设来源("自源")`，否则「我」不被自己的观察塑造（那是错的）。

铁律：C1 禁 Transformer｜C9 禁 OCR｜C10 禁外挂模型当器官｜C13 确定性可复现｜C14 不破既有。
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))            # meeting38/实施
_AHI3 = os.path.dirname(os.path.dirname(_HERE))              # AHI3 根
_M39 = os.path.join(_AHI3, "meeting39", "实施")
for _p in (_HERE, os.path.join(_AHI3, "meeting31"), _M39, _AHI3):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from 内核_顿悟涌现 import 顶覆盖_综合体W                        # noqa: E402  （五层链顶）
from 内核_画面 import 顶覆盖_综合体H, 来路_实感, 来路_回忆, 来路_想象  # noqa: E402（第六层：画面/视觉想象）
from 内核_自我回路 import 来源标_自源, 来源标_外源                # noqa: E402
from 内核_相空间算子T import T2参数                              # noqa: E402

#: V3 的自我意识配置：六内核全开 + 自驱思考（ε_self 高于自身慢基线 ⇒ 它自己开一个「想」）
#: ★第六层 = 画面（视觉想象）：让"它脑子里那幅画"也进链——见 内核_画面.py 文件头。
#:   `画面开关=True` 只**加挂**：无真帧（未 `设帧源`）时逐拍什么都不做，既有判据逐位不变。
开关 = dict(自模型开关=True, 思核开关=True, 学习开关=True, 自我开关=True, 顿悟开关=True,
           自我提问开关=True, 画面开关=True)


def 造桥(seed: int = 310, 私偏κ: float = 10.0, 档案seed=None, 开关_=None,
         画面种子=None, **kw):
    """造一个**挂了六内核**的综合体（synthesis 顶覆盖链顶）。默认六开关全开。

    ★链顶由 顶覆盖_综合体W（步6）改为 顶覆盖_综合体H（步8 画面）。MRO 里五层全部仍在，
    故 `test_模块五_五内核.py` 的 N5-1 仍过（它验的是"五层齐全"，不是"链顶是谁"）。
    """
    开 = dict(开关)
    if 开关_:
        开.update(开关_)
    桥 = 顶覆盖_综合体H(seed=seed, 私偏κ=私偏κ, 档案seed=档案seed,
                       画面种子=画面种子, **开, **kw)
    # ★思核惰性起改为**构造即起**：否则 `内核态()` 在它还没思考过之前报「未挂载」，
    #   看起来像没接上。起它只是建方向表（Φ_act 列单位化，零新权重表），代价极小。
    try:
        if getattr(桥, "思核开关", False):
            桥._起思核()
    except Exception:
        pass
    return 桥


def 内核态(桥) -> dict:
    """只读汇报五内核的当下状态（给输出层/证据落盘用）。缺哪层报哪层为 None。"""
    if 桥 is None:
        return dict(挂载=False)
    出 = dict(挂载=True)

    zm = getattr(桥, "自模型", None)
    if zm is not None:
        ε = zm.ε序列()
        出["自模型"] = dict(迹条数=len(zm.迹), 更新次数=zm.更新次数, D=int(zm.W.shape[0]) if zm.W is not None else 0,
                          ε最近=(round(float(ε[-1]), 4) if ε else None),
                          ε均值=(round(float(np.mean(ε)), 4) if ε else None))
    else:
        出["自模型"] = None

    sx = getattr(桥, "思核", None)
    if sx is not None:
        出["思核"] = dict(迹条数=len(sx.迹), 维=sx.维度,
                        最近站=(sx.站点(sx.迹[-1]["z"]) if sx.迹 else None))
    else:
        出["思核"] = None

    drv = getattr(桥, "drv", None)
    ε史 = list(getattr(drv, "ε史", []) or [])
    出["真学习"] = dict(ε位数=len(ε史),
                       ε慢基线=(round(float(getattr(drv, "ε_slow", 0.0)), 4) if ε史 else None),
                       偏置在决策链=(type(drv).__name__ if drv is not None else None))

    zk = getattr(桥, "自我核", None)
    if zk is not None:
        迹 = zk.迹
        出["自我回路"] = dict(迹条数=len(迹), 自源更新次数=zk.自源更新次数, 外源次数=zk.外源次数,
                           W范数=round(float(np.linalg.norm(zk.W_self)), 4),
                           最近一致度=(round(float(迹[-1]["一致度"]), 4) if 迹 else None),
                           最近来源=(迹[-1]["来源"] if 迹 else None))
    else:
        出["自我回路"] = None

    dw = getattr(桥, "顿悟核", None)
    if dw is not None:
        出["顿悟涌现"] = dict(顿悟条数=len(dw.顿悟账本), 事件拍s=list(dw.事件拍s)[-8:],
                           ε史条数=len(dw.ε史), 站史条数=len(dw.站史))
    else:
        出["顿悟涌现"] = None

    # ── 第六层：画面（视觉想象）──
    hm = getattr(桥, "画面核", None)
    if hm is not None:
        库 = getattr(hm, "库", []) or []
        出["画面"] = dict(库大小=len(库), 实感次数=hm.实感次数, 想象次数=hm.想象次数,
                       迹条数=len(getattr(hm, "迹", []) or []),
                       最近件数=(int(库[-1].get("物体数", 0)) if 库 else None),
                       W画范数=round(float(np.linalg.norm(hm.W画)), 4))
    else:
        出["画面"] = None
    return 出


def 自省文(桥, 上限: int = 120) -> list:
    """最近几条**自我痕迹 / 顿悟**的自言文本（它"脑子里"的自我陈述），供外化参考。

    这是「思考很丰富、表达才几个字」的那部分：内部独白留在自言流里，只取最近几条。
    ⚠ 自言流条目是 **dict**（`inner_speech.自言空间.追加`），不是对象——首版按对象读属性
    导致取回全空，已修（兼容两种形态）。
    """
    if 桥 is None:
        return []
    流 = getattr(getattr(桥, "自言", None), "流", []) or []
    出 = []
    for x in 流[-40:]:
        本 = x if isinstance(x, dict) else dict(拍=getattr(x, "拍", None),
                                              类型=getattr(x, "类型", ""),
                                              文本=getattr(x, "文本", ""))
        t = str(本.get("文本", ""))
        if t.startswith("自我回路") or t.startswith("顿悟涌现"):
            出.append(dict(拍=本.get("拍"), 类型=str(本.get("类型", "")), 文本=t[:上限]))
    return 出[-4:]


def 设来源(桥, 自源: bool) -> None:
    """设置 provenance：True=自源（它自己去看/自主），False=外源（用户说的）。

    优先用内核提供的 `钉来源()`（显式钉住，`步/步_向量` 不再覆盖）；老版本内核退化为直接置标。
    """
    if 桥 is None:
        return
    标 = 来源标_自源 if 自源 else 来源标_外源
    try:
        钉 = getattr(桥, "钉来源", None)
        if callable(钉):
            钉(标)
            return
    except Exception:
        pass
    try:
        桥._来源标 = 标
    except Exception:
        pass


def 起思核(桥) -> bool:
    """确保相空间算子 T 已就绪（不需要它先「思考」过一次）。"""
    try:
        if getattr(桥, "思核开关", False) and getattr(桥, "思核", None) is None:
            桥._起思核()
        return getattr(桥, "思核", None) is not None
    except Exception:
        return False


def 想一拍(桥, 问题: str = "我在想刚才看到的东西", 拍数: int = 2) -> bool:
    """让它**真的想几拍**（思考核轨迹才留得下来）。

    机制（照引擎既有路径，不伪造）：进入「思考态」需要证据不足——走既有 `_延答`
    （它自己的思考入口：先 ack、再把问题挂进思考态）；而思考态的**每拍推进**
    由既有 `断供一拍()`（无外部事件时的自主拍）驱动 → `_思考一拍` → 相空间算子 T 推进一步
    → 轨迹落进 `思核.迹`（独立思考存储）。

    ⚠ 这里必须走 `断供一拍`：直接调 `_思考一拍` 会绕过引擎的自主拍语义；而把轨迹硬塞进
    `思核.迹` 就是**伪造**——本函数只触发真路径。
    """
    if 桥 is None:
        return False
    try:
        起思核(桥)
        if getattr(桥, "思考中", None) is None:
            桥._延答(str(问题)[:120])
        for _ in range(max(1, int(拍数))):
            if getattr(桥, "思考中", None) is None:
                break
            桥.断供一拍()
        return getattr(桥, "思核", None) is not None and len(getattr(桥.思核, "迹", [])) > 0
    except Exception:
        return False


# ────────────────────────── 第六层：画面（视觉想象）接线 ──────────────────────────

def 设帧源(桥, 源) -> bool:
    """把**真帧来源**接给画面核（`源` = 可调用 `()->RGB帧` 或带 `.截图()` 的对象）。

    ★为什么由此处接：`内核_画面.py` **不 import 任何器官**（纪律），
    所以"眼睛在哪"必须由 V3 侧交给它——这是唯一的接口，不是内嵌。
    """
    try:
        if 桥 is None:
            return False
        设 = getattr(桥, "设帧源", None)
        if callable(设):
            设(源)
            return True
    except Exception:
        pass
    return False


def 画面一拍(桥, 帧=None) -> dict:
    """显式让它**看一屏并想出画面**（记一屏 → 回放 → 组合想象）。供演示/判据直接调。"""
    try:
        f = getattr(桥, "画面一拍", None)
        if callable(f):
            return f(帧=帧)
    except Exception as ex:
        return dict(画面=False, 错误=repr(ex))
    return dict(画面=False, 原因="桥未挂画面层")


def 想象一拍(桥, 位移=(3, 2)) -> dict:
    """让它**从记忆里想出一幅新画面**（整体搬家式组合），逐件判"还认得出来吗"，
    并把这幅想象中的画面喂给自模型。

    · 想象码入自模型时 `更新=False`（默认）⇒ **想象不塑造自模型**（自编的东西不算经历）；
    · 返回 dict(想象, 认得件数, 总件数, 码似最低, 新组合格数, 喂, ε_self, 一致度)。
    """
    try:
        im = getattr(桥, "画面核", None)
        if im is None or not getattr(im, "库", None):
            return dict(想象=False, 原因="画面库为空")
        r = im.想象读出(位移=位移)
        喂 = getattr(桥, "喂想象给自模型", None)
        f = (喂(位移=位移) if callable(喂) else dict(喂=False))
        return dict(想象=bool(r.get("想象")), 认得件数=r.get("认得件数"),
                    总件数=r.get("总件数"), 码似最低=r.get("码似最低"),
                    逐件码似=r.get("逐件码似"),
                    新组合格数=r.get("新组合格数"), 位移=r.get("位移"),
                    喂=f.get("喂"), ε_self=f.get("ε_self"), 一致度=f.get("一致度"))
    except Exception as ex:
        return dict(想象=False, 错误=repr(ex))


__all__ = ["造桥", "内核态", "自省文", "设来源", "起思核", "想一拍",
           "设帧源", "画面一拍", "想象一拍",
           "开关", "顶覆盖_综合体H", "T2参数"]
