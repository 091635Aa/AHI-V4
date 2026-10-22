# -*- coding: utf-8 -*-
"""test_模块五_五内核.py —— V3 判据：meeting39 五内核**真接线**（审计 P0-2 的验收口）。

背景：`V3审计报告_逐条核对_20261007.md` P0-2 指出 —— AHI3 里**存在**五个内核
（自模型 / 相空间算子T / 真学习 / 自我回路 / 顿悟涌现），但 `meeting38/` 对它们的引用数 = 0，
V3 主程序挂的是 `synthesis.综合体` **基类**，五内核从未装配。

本判据验的是「真接线」而不是「文件在不在」：

  N5-1 装配：`V3生命.起综合体()` 挂出的是**五层顶覆盖链**（MRO 里五层齐全，非基类）。
  N5-2 自模型真学：连续喂 → ε_self 有序列、W 有更新次数；**首窗 ε 高于末窗 ε**（真在降）。
  N5-3 真学习反哺：决策链上的偏置源是 `驱动_ε`（ε_self 经既有 内生偏置 b[] 接回引擎）；
        且 ε史 与 自模型迹 条数一致（同一时钟域）。
  N5-4 自我回路 + provenance：自源喂 → 自源更新次数 > 0、W_self 范数 > 0；
        **外源喂同一段 → W_self 逐位不变**（专家03 J5「外源不可塑造我」）。
  N5-5 思核（相空间算子 T）：进入思考态并经自主拍推进后，`思核.迹` 有轨迹且站点可判
        （「想」是相空间迭代，不是 _检索）。
  N5-6 顿悟核就位：ε史/站史随拍增长（前窗/后窗可比的原料）；**不**断言必然出现事件
        （事件是相变共现，可失败——不恒真才是判据）。
  N5-7 学习≠记忆（在五内核侧）：ε史/自模型迹/自我迹**都不是** `codec._段`；
        喂完之后 `len(codec._段)` 不因这些内核而增。
  N5-8 自指记忆有界（★第三轮新增）：长跑后 `‖W_self‖` **有界**（第三轮修复：Hebbian 后范数归一），
        且**方向仍在随经验改变**（不是把 W 冻住充数）——「我得是我，且我会变」。

跑法：python test_模块五_五内核.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M38 = os.path.dirname(HERE)
AHI3 = os.path.dirname(M38)
for p in (HERE, M38, AHI3, os.path.join(AHI3, "meeting31"), os.path.join(AHI3, "meeting39", "实施")):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── 出口编码自愈（同 P0-3）──
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import importlib.util                                            # noqa: E402


def _载实现():
    """按**文件路径**加载实现版 `V3主程序`（AHI3 根还有一个同名薄壳启动器，会遮蔽本模块）。"""
    路径 = os.path.join(HERE, "V3主程序.py")
    for k in [k for k in sys.modules if k == "V3主程序"]:
        del sys.modules[k]
    spec = importlib.util.spec_from_file_location("V3主程序_实现", 路径)
    模 = importlib.util.module_from_spec(spec)
    sys.modules["V3主程序_实现"] = 模
    spec.loader.exec_module(模)
    return 模


V = _载实现()
import 内核宿主 as HZ                                            # noqa: E402

# 连续语料：同一话题的多句，供自模型学「我下一步会怎样」
语料 = [
    "风从窗缝里进来，纸角轻轻翘起来。",
    "我把杯子挪到左手边，腾出桌角一块地方。",
    "屏幕光落在键盘上，键帽边缘有点发亮。",
    "楼下有人喊了一声，然后就没动静了。",
    "我把刚才那句读到一半，又退回去重读。",
    "时间过得慢，我又看了一眼窗外。",
    "灯还亮着，影子落在桌面上，随着我手动。",
    "我忽然想起昨天没做完的那件事。",
] * 6          # 48 句：够跑出自模型 ε 序列与前/后窗


def _ε序(桥):
    zm = getattr(桥, "自模型", None)
    return list(zm.ε序列()) if zm is not None else []


def main():
    命 = V.V3生命(种子=310)
    ok起 = 命.起综合体()
    桥 = 命.桥

    # ── N5-1 装配：五层链 ──
    mro名 = [c.__name__ for c in type(桥).__mro__]
    五层 = ["顶覆盖_综合体W", "顶覆盖_综合体Z", "顶覆盖_综合体L", "顶覆盖_综合体T", "顶覆盖_综合体"]
    N5_1 = bool(ok起 and all(x in mro名 for x in 五层))
    N5_1b = bool(type(桥).__module__.startswith("内核_"))

    # ── N5-2 自模型真学：ε 有序列且首窗 > 末窗 ──
    HZ.设来源(桥, 自源=True)
    for s in 语料:
        桥.步_向量(命._感知向量(s), 原文=None)
    ε = _ε序(桥)
    zm = getattr(桥, "自模型", None)
    首 = float(np.mean(ε[:max(1, len(ε) // 4)])) if ε else None
    末 = float(np.mean(ε[-max(1, len(ε) // 4):])) if ε else None
    N5_2 = bool(ε and 首 is not None and 末 is not None and 末 < 首)
    N5_2b = bool(zm is not None and zm.更新次数 > 0)

    # ── N5-3 真学习反哺：偏置源是 驱动_ε，且 ε史 与 自模型 ε 数**同域** ──
    #    口径说明：`学一拍` 首拍没有上一拍可比 ⇒ ε=None（不吸收），故
    #    len(ε史) = len(ε序列) = len(迹) − 1。这是**设计正确**，不是缺陷；
    #    判据因此比「迹条数」，而比「ε序列条数」。
    drv = getattr(桥, "drv", None)
    N5_3 = bool(type(drv).__name__ == "驱动_ε" and len(getattr(drv, "ε史", [])) > 0)
    N5_3b = bool(len(getattr(drv, "ε史", [])) == len(_ε序(桥))
                 and len(_ε序(桥)) == len(getattr(zm, "迹", [])) - 1)

    # ── N5-4 自我回路 + provenance 门控 ──
    zk = getattr(桥, "自我核", None)
    W0 = None if zk is None else np.array(zk.W_self, copy=True)
    自源次0 = 0 if zk is None else int(zk.自源更新次数)
    HZ.设来源(桥, 自源=True)
    for s in 语料[:8]:
        桥.步_向量(命._感知向量(s), 原文=None)
    W1 = None if zk is None else np.array(zk.W_self, copy=True)
    自源次1 = 0 if zk is None else int(zk.自源更新次数)
    N5_4a = bool(zk is not None and 自源次1 > 自源次0 and float(np.linalg.norm(W1)) > 0.0)

    # 外源喂同一段：W_self 必须**逐位不变**
    HZ.设来源(桥, 自源=False)
    for s in 语料[:8]:
        桥.步_向量(命._感知向量(s), 原文=s)
    外源次 = int(getattr(zk, "外源次数", 0))
    W2 = None if zk is None else np.array(zk.W_self, copy=True)
    差 = 0.0 if (W1 is None or W2 is None) else float(np.max(np.abs(W2 - W1)))
    N5_4b = bool(外源次 > 0 and 差 == 0.0)

    # ── N5-5 思核（相空间算子 T）真轨迹 ──
    HZ.设来源(桥, 自源=True)
    想ok = HZ.想一拍(桥, 问题="刚才看的那段话，我在想它跟我有什么关系", 拍数=3)
    sx = getattr(桥, "思核", None)
    迹 = list(getattr(sx, "迹", [])) if sx is not None else []
    N5_5 = bool(想ok and len(迹) >= 1 and sx.站点(迹[-1]["z"]) is not None)
    站s = [sx.站点(x["z"]) for x in 迹] if 迹 else []

    # ── N5-6 顿悟核就位（不断言必有事件）──
    dw = getattr(桥, "顿悟核", None)
    N5_6 = bool(dw is not None and len(getattr(dw, "ε史", [])) > 0
                and len(getattr(dw, "站史", [])) > 0)

    # ── N5-7 学习≠记忆：内核状态都不在 codec._段；喂完段数不因内核增 ──
    段0 = len(桥.codec._段)
    for s in 语料[:4]:
        桥.步_向量(命._感知向量(s), 原文=None)          # 原文=None ⇒ 不注册段
    段1 = len(桥.codec._段)
    N5_7 = bool(段0 == 段1 and 段0 >= 0)

    # ── N5-8 自指记忆有界（★第三轮新增）：长跑后 ‖W_self‖ 有界 ∧ 方向仍随经验变 ──
    #   修复前实测：400→3200 拍 ‖W_self‖ 12395→66327（每次更新约 +21，近似线性无界）。
    #   修复后：Hebbian 写完后归一到上限 ⇒ ‖W_self‖ 恒为上限；方向仍随经验漂移。
    zk8 = getattr(桥, "自我核", None)
    范s, 向s = [], []
    for _ in range(4):
        for _s_ in 语料:
            桥.步_向量(命._感知向量(_s_), 原文=None)
        n8 = float(np.linalg.norm(zk8.W_self))
        范s.append(n8)
        向s.append(np.array(zk8.W_self / max(n8, 1e-12), copy=True))
    上限 = 1.0
    有界 = all(abs(x - 上限) <= 1e-6 for x in 范s)
    方向变 = max((float(np.abs(向s[i + 1] - 向s[i]).max()) for i in range(len(向s) - 1)),
                 default=0.0)
    N5_8 = bool(有界 and 方向变 > 0.0)

    态 = 命.内核态()
    报 = dict(
        N5_1_装配五层链=dict(过=bool(N5_1 and N5_1b), MRO=mro名[:7],
                          桥模块=type(桥).__module__),
        N5_2_自模型真学=dict(过=bool(N5_2 and N5_2b), ε条数=len(ε),
                          首窗ε=(None if 首 is None else round(首, 4)),
                          末窗ε=(None if 末 is None else round(末, 4)),
                          更新次数=(None if zm is None else zm.更新次数),
                          W维=(None if zm is None or zm.W is None else int(zm.W.shape[0]))),
        N5_3_真学习反哺=dict(过=bool(N5_3 and N5_3b), 偏置源=type(drv).__name__,
                          ε史条数=len(getattr(drv, "ε史", [])),
                          自模型ε条数=len(_ε序(桥)),
                          自模型迹条数=len(getattr(zm, "迹", [])),
                          ε慢基线=round(float(getattr(drv, "ε_slow", 0.0)), 4)),
        N5_4_自我回路与门控=dict(过=bool(N5_4a and N5_4b),
                              自源更新=(自源次0, 自源次1), 外源次数=外源次,
                              W范数=(None if W1 is None else round(float(np.linalg.norm(W1)), 3)),
                              外源喂后W差=差),
        N5_5_思核相空间轨迹=dict(过=bool(N5_5), 想一拍=想ok, 迹条数=len(迹),
                              站点=sorted(set(站s)), 站点数=len(set(站s))),
        N5_6_顿悟核就位=dict(过=bool(N5_6),
                          ε史条数=len(getattr(dw, "ε史", [])),
                          站史条数=len(getattr(dw, "站史", [])),
                          顿悟条数=len(getattr(dw, "顿悟账本", []))),
        N5_7_学习非记忆=dict(过=bool(N5_7), 段前=段0, 段后=段1,
                          自模型迹=len(getattr(zm, "迹", []))),
        N5_8_自指记忆有界=dict(过=bool(N5_8), 范数s=[round(x, 6) for x in 范s],
                           上限=上限, 方向仍变=round(方向变, 6)),
        内核态=态,
    )
    全过 = all(报[k]["过"] for k in 报 if k.startswith("N5_"))
    报["全过"] = bool(全过)
    with open(os.path.join(HERE, "test_模块五_结果.json"), "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(报, ensure_ascii=False, indent=1, default=str))
    return 全过


if __name__ == "__main__":
    ok = main()
    print("\n[模块五·五内核] %s" % ("全过 ✅" if ok else "未全过 ❌"))
