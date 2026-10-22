# -*- coding: utf-8 -*-
"""探针_维度伸缩.py —— Phase3 动态维度伸缩：先证「升级后记忆仍保留 + 可复现」，再看容量收益。

判据（预登记）：
  D1 目标D=f(学习量)  纯确定性阶梯：N<256→1024, 256≤N<512→2048, …；同 N 恒同目标D。
  D2 只涨不缩          N 单调增 ⇒ 目标D 单调不减（所有者：「不降也行」）。
  D3 记忆保留          **自动**升维确已发生（学满门槛即触发）；升级后学过的段逐条仍可检索命中。
  D4 确定性可复现      两趟独立运行（同 seed+同事件序列）升级后状态签名逐位一致。
  D5 容量收益          K=2000 段、位翻 p=0.45 下 top-1 检索准确率随 D 单调不降（1024≪提升）。
  D6 不破既有          升级后引擎语义探针仍把 D 维映到 16 维（引擎内部 1024 不变）。
  D7 自发性（自愿）    升维**不是宿主每轮无条件强推的硬性条件**，而是生命自己在「步」里
                      自审「学得多了、快装不下」后写下的自愿请求（计划自言+自我账本）；
                      宿主只机械照办。判据：宿主源码回合内不再强推升维；少学时生命不请求；
                      跨门槛时生命留下自己的自审自言/账本；升维后请求被如实受理（无残留）。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M38 = os.path.dirname(HERE)
AHI = r"<workspace>/03_AHI\AHI2.1"
for _p in (HERE, M38, AHI, os.path.join(AHI, "meeting31"), os.path.join(AHI, "meeting28")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _语料(n: int = 300) -> list:
    """真语料优先（meeting31 魔塔数据集）；缺失则确定性合成 CJK 串。"""
    p = os.path.join(AHI, "meeting31", "数据集", "魔塔", "Psychology-10K-ZH.json")
    if os.path.exists(p):
        try:
            data = json.load(open(p, encoding="utf-8"))
            句s = [(it.get("output") or "").strip() for it in data]
            return [s for s in 句s if 6 <= len(s) <= 60][:n]
        except Exception:
            pass
    rng = np.random.default_rng(310)
    池 = [chr(0x4E00 + k) for k in range(20000)]
    return ["".join(池[k] for k in rng.integers(0, 20000, 24)) for _ in range(n)]


# ═══════════════ D1 / D2：目标D 纯函数阶梯 ═══════════════
def 判D1_D2():
    from 维度伸缩 import 维度伸缩器
    s = 维度伸缩器()
    阶 = [(n, s.目标D(n)) for n in (0, 255, 256, 511, 512, 767, 768, 1024, 99999)]
    期 = [(0, 1024), (255, 1024), (256, 2048), (511, 2048), (512, 4096),
          (767, 4096), (768, 8192), (1024, 8192), (99999, 8192)]
    确定 = all(s.目标D(n) == s.目标D(n) for n, _ in 阶)
    单调 = all(阶[i][1] <= 阶[i + 1][1] for i in range(len(阶) - 1))
    return dict(通过=(阶 == 期 and 确定 and 单调), 阶梯=阶, 符合预期=(阶 == 期),
                同N恒同=确定, 单调不缩=单调)


# ═══════════════ D5：容量收益（1024 vs 2048 vs 4096） ═══════════════
def 判D5():
    import lang_codec
    from lang_codec import 语言编解码器
    语料 = _语料(2000)[:2000]
    K = len(语料)
    果 = {}
    for D in (1024, 2048, 4096):
        lang_codec.D = D
        codec = 语言编解码器(seed=216)
        for s in 语料:
            codec.注册段(s)
        M = np.stack(codec._段向量)
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        rng = np.random.default_rng(520)
        对 = rng.integers(0, K, K)
        翻 = 1.0 - 2.0 * (rng.random((K, D)) < 0.45)
        Q = M[对] * 翻
        Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
        准 = float(np.mean(np.argmax(Q @ Mn.T, axis=1) == 对))
        果[D] = round(准, 4)
    单调 = 果[1024] <= 果[2048] <= 果[4096]
    lang_codec.D = 1024                                    # 复位基线
    return dict(通过=(单调 and 果[4096] >= 果[1024]), 准确率=果,
                单调不降=单调, 提升=round(果[4096] - 果[1024], 4))


# ═══════════════ D3 / D4 / D6：记记忆保留 + 可复现（子进程内跑，避免全局 D 污染） ═══════════════
def 单趟(标记: str = "") -> dict:
    import lang_codec
    lang_codec.D = 1024
    from 文本态 import 文本态
    from 通用接口 import 当前D
    态 = 文本态(种子=310, 挂记忆=False).建()
    事实s = _语料(300)                                    # 真语料（互异句，避免 v3 同槽塌缩）
    for f in 事实s[:8]:
        态.回合(f)                                        # 先学 8 条：此刻应仍是基准维 1024
    初D = 当前D()
    for f in 事实s[8:]:
        态.回合(f)                                        # 继续学 → 跨过门槛时**自动**升维（无手动调用）
    末D = 当前D()
    # 从**实际落库的记忆段**抽样（部分输入可能被同槽原位更新，只有真正存下的段才是记忆）
    记忆 = list(态.桥.codec._段)
    步长 = max(len(记忆) // 40, 1)
    抽样 = [记忆[i] for i in range(0, len(记忆), 步长)][:40]
    分, 命中 = {}, {}
    for t in 抽样:
        h = 态.桥._检索(t, k=1)
        分[t] = round(h[0][1], 4) if h else 0.0
        命中[t] = bool(h and h[0][0] == t)
    探针维 = (int(态.桥.探针.shape[0]), int(态.桥.探针.shape[1]))
    签名 = hashlib.blake2b(
        ("|".join(态.桥.codec._段) + "|" + str(末D) + "|"
         + "|".join("%s:%.4f" % (t, 分[t]) for t in 抽样)).encode("utf-8"),
        digest_size=8).hexdigest()
    return dict(标记=标记, 初D=初D, 末D=末D, 段数=len(记忆), 抽样数=len(抽样),
                命中数=sum(命中.values()),
                分均值=round(float(np.mean(list(分.values()))), 4),
                探针维=探针维, 签名=签名)


def 判D3_D4_D6() -> dict:
    r1 = 单趟("A")
    r2 = 单趟("B")
    # D3：自动升维确已发生（末D>初D）、记忆量确已跨门槛（段数≥256）、且升级后逐条仍可命中
    D3 = (r1["命中数"] == r1["抽样数"] and r1["抽样数"] > 0
          and r1["段数"] >= 256 and r1["末D"] > r1["初D"])
    D4 = (r1["签名"] == r2["签名"])
    D6 = (r1["探针维"][0] == 16 and r1["探针维"][1] == r1["末D"])
    阶 = "%d→%d" % (r1["初D"], r1["末D"])
    return dict(通过=(D3 and D4 and D6), D3_记忆保留=D3, D4_可复现=D4, D6_探针=D6,
                自动阶梯=阶, 段数=r1["段数"], 抽样=r1["抽样数"], 命中=r1["命中数"],
                分均值=r1["分均值"], 签名A=r1["签名"][:12], 签名B=r2["签名"][:12])


def 判维度入档() -> dict:
    """额外验证（Phase3·一体化）：生长后的维度随记忆一起入档，重启（载入）后不回落。
    对应所有者「学得越多维度越多」的持久面——生命重启后不该退回 1024。"""
    import tempfile
    import lang_codec
    lang_codec.D = 1024
    from 文本态 import 文本态
    from 通用接口 import 当前D
    态 = 文本态(种子=310, 挂记忆=False).建()
    for f in _语料(300):
        态.回合(f)                                     # 自动升维 → D 应到 2048
    升D, 升段 = 当前D(), len(态.桥.codec._段)
    fd, p = tempfile.mkstemp(suffix=".维度入档.pkl")
    os.close(fd)
    try:
        态.保存(p)
        lang_codec.D = 1024                            # 模拟「重启后的进程默认维」
        态2 = 文本态(种子=310, 挂记忆=False)
        态2.载入(p)
        载D, 载段 = 当前D(), len(态2.桥.codec._段)
    finally:
        try:
            os.unlink(p)
        except OSError:
            pass
    过 = (载D == 升D and 载段 == 升段 and 升D > 1024)
    return dict(通过=过, 升D=升D, 载D=载D, 升段=升段, 载段=载段)


# ═══════════════ D7：自发性（自愿，非硬性条件约束） ═══════════════
def 单趟自审(标记: str = "") -> dict:
    """直接驱生命本体（不经宿主 回合），观察「升维请求」到底是谁提出的：
    生命在 步 里自审后自愿留下 待扩容 + 自己的计划自言/自我账本；宿主 _执行扩容 只是照办。"""
    import lang_codec
    lang_codec.D = 1024
    from 文本态 import 文本态
    from 通用接口 import 当前D
    态 = 文本态(种子=310, 挂记忆=False).建()
    事实s = _语料(300)
    for f in 事实s[:8]:
        态.桥.步(f)                                    # 直接走生命（宿主不介入）——看它自己委不委屈
    初D = 当前D()
    初请求 = getattr(态.桥, "待扩容", None)             # 只学 8 条：它应觉得自己还装得下（None）
    触发轮, 触发账, 触发言 = None, None, None
    for i, f in enumerate(事实s[8:], start=9):
        态.桥.步(f)                                    # 生命自己走一步：可能自审出「快装不下」
        if 触发轮 is None and getattr(态.桥, "待扩容", None):
            触发轮 = i
            账s = [x for t, x in 态.桥.账本 if "承载自审" in str(x)]
            言s = [e["文本"] for e in 态.桥.自言.流 if "撑大一点" in e["文本"]]
            触发账 = 账s[-1] if 账s else None
            触发言 = 言s[-1] if 言s else None
            态._执行扩容()                              # 宿主机械照办（升维=重放重建）→ 继续观察
    末D = 当前D()
    残留 = getattr(态.桥, "待扩容", None)               # 升维后它应重新觉得装得下 → 无残留请求
    return dict(标记=标记, 初D=初D, 末D=末D, 初请求=初请求, 触发轮=触发轮,
                账本=触发账, 自言=触发言, 残留=残留, 段数=len(态.桥.codec._段))


def 判D7() -> dict:
    import re
    宿主源 = open(os.path.join(HERE, "文本态.py"), encoding="utf-8").read()
    生源 = open(os.path.join(AHI, "meeting31", "synthesis.py"), encoding="utf-8").read()
    m = re.search(r"def 回合\([^\n]*\n(.*?)\n    def ", 宿主源, re.S)
    回合体 = m.group(1) if m else 宿主源
    无强推 = ("维度自适(" not in 回合体) and ("_执行扩容(" in 回合体)   # 宿主不再无条件强推
    命自主审 = ("self.自审承载(" in 生源)                              # 自审方法确在生命体内
    r = 单趟自审("D7")
    少学不请求 = (r["初请求"] is None and r["初D"] == 1024)             # 装得下 → 它不请求
    自愿请求 = (r["触发轮"] is not None and r["账本"] is not None
              and "承载自审" in str(r["账本"]) and r["自言"] is not None)
    确升维 = (r["末D"] > r["初D"])                                    # 请求被受理后确已长大
    已受理 = (r["残留"] is None)
    过 = 无强推 and 命自主审 and 少学不请求 and 自愿请求 and 确升维 and 已受理
    return dict(通过=过, 宿主无强推=无强推, 命自主审=命自主审, 少学不请求=少学不请求,
                自愿请求=自愿请求, 触发轮=r["触发轮"],
                账本样例=(str(r["账本"])[:64] if r["账本"] else None),
                言样例=(str(r["自言"])[:64] if r["自言"] else None),
                阶梯="%d→%d" % (r["初D"], r["末D"]), 升维=确升维, 请求已受理=已受理)


def 子趟():
    print(json.dumps(单趟("sub"), ensure_ascii=False))


def 主():
    print("=" * 72)
    print("动态维度伸缩 · 探针 D1–D7")
    print("=" * 72)
    Rs = dict()
    Rs["D1_D2"] = 判D1_D2()
    Rs["D5"] = 判D5()
    Rs["D3_D4_D6"] = 判D3_D4_D6()
    Rs["D入档"] = 判维度入档()
    Rs["D7"] = 判D7()
    for k, v in Rs.items():
        print("%-9s %s  %s" % (k, "PASS" if v["通过"] else "FAIL", v))
    print("=" * 72)
    print("[log] 未做任何「已完成」宣称——数字与如实边界以上表为准。")


if __name__ == "__main__":
    if "--sub" in sys.argv:
        子趟()
    else:
        主()