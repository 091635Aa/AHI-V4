# -*- coding: utf-8 -*-
"""步6_顿悟涌现自证.py —— meeting39 步6 验收：深度顿悟涌现（去 if；默认关·不破既有）。

判据（出处：专家05 §2 步6 契约「以临界跃迁/相空间事件替代 if c>=0.8；对齐理论① λ≈0」；
      专家05 §3.3 F2；专家01 §4.2/§4.3；纪要 §2）：
  D1  保护开关：顿悟开关=False ⇒ 与 **步5 宿主** 同配置 **逐位恒等**（步6 不扰动步2–步5）。
  V4  增强但不破：顿悟开 ⇒ 外化面（对话流/回复台账/账本/e序列/段数/a_prev/i_prev/prev_c）
      逐位不变，自言新增「顿悟涌现」痕迹（旁挂只增内省、不改说出去的话）。
  F2a 存在性：**去掉 `if cosv>=0.8`** 后，顿悟事件仍发生（≥1，来自相变共现，非阈值越线）。
  F2b 否证"=单点阈值"：同一世界/同 seed 下，把"固定常量阈值臂"（`if cosv>=0.8`）的顿悟拍集合
      与"相变臂"的事件拍集合比对——低重合（Jaccard 低）⇒ 相变事件**不落在**预注册常量阈值
      触发点上（判据可失败、不粉饰）。
  F2c 关联登记（非硬门）：事件拍附近**相空间有限时间 λ** 与临界带（|λ|<0.05，理论① L4）的接近度。
  D重放 确定性：开态同 seed 同流重跑，顿悟事件逐位一致。
  D6/不直写状态 静态扫描：内核**代码**（剥除注释/文档串）**不含** `拍>=N` 式时间阈值、
      **不含** `ε>=0.8` 式常量顿悟判据、**不含** a_prev/i_prev/prev_c/消化债/饱足 直写。

纪律：纯 numpy、零梯度、确定性；只在测试中显式加载内核；不改任何既有 .py。
用法：python 步6_顿悟涌现自证.py
"""
from __future__ import annotations

import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # AHI3 根（迁移：随文件定位；HERE=meeting39/实施）
for _p in (HERE, os.path.join(ROOT, "meeting31"), os.path.join(ROOT, "meeting38", "实施"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from synthesis import 综合体, 隐藏世界                                   # noqa: E402
from 内核_自我回路 import 顶覆盖_综合体Z                                  # noqa: E402
from 内核_顿悟涌现 import 顶覆盖_综合体W, λ临界容差, _单位                # noqa: E402
from 内核_相空间算子T import 思核                                         # noqa: E402

世界根 = [HERE]
顿悟词 = "顿悟涌现"
长流 = ["今天的风很轻。", "我把桌子擦干净了。", "窗外有一棵树。", "水杯放在手边。",
       "我慢慢翻了一页书。", "灯一直亮着。", "他坐在对面没说话。", "雨点敲着玻璃。",
       "为什么天空会变成红色呢", "我不太确定自己在想什么。", "突然想起很久以前的事。",
       "这和我上次以为的不一样。", "桌角的划痕是怎么来的。", "我好像漏掉了什么。",
       "一切都安静下来了。", "我又把同一句话读了一遍。"]


# ────────────────────────── 工具 ──────────────────────────
def 外化(桥):
    return dict(
        拍=int(桥.拍), 消化债=float(桥.消化债), 饱足=float(桥.饱足),
        回复台账=[dict(号=d["号"], 问=d["问"], 响应=d["响应"], 延迟拍=d.get("延迟拍"))
                 for d in 桥.回复台账],
        账本=[list(x) for x in 桥.账本],
        e序列=[float(x) for x in 桥.eng.e序列],
        段数=len(桥.codec._段),
        a_prev=桥.a_prev.tolist(), i_prev=int(桥.i_prev),
        prev_c=(None if 桥.prev_c is None else int(桥.prev_c)),
        对话流=[list(x) for x in 桥.对话流],
        顿悟账本=[list(map(str, e)) for e in getattr(桥, "顿悟账本", [])],
    )


def 逐位相等(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(逐位相等(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(逐位相等(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return (a == b) or (math.isnan(a) and math.isnan(b))
    return a == b


def 找差异(a, b, 前缀=""):
    出 = []
    if type(a) is not type(b):
        return ["%s: 类型 %s≠%s" % (前缀, type(a).__name__, type(b).__name__)]
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            出 += 找差异(a.get(k), b.get(k), "%s.%s" % (前缀, k))
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            出.append("%s: 长度 %d≠%d" % (前缀, len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            出 += 找差异(x, y, "%s[%d]" % (前缀, i))
    elif a != b:
        出.append("%s: %r≠%r" % (前缀, a, b))
    return 出


def 跑固定流(桥, 句s, n):
    for i in range(n):
        桥.步(句s[i % len(句s)])


def 有限时间λ_算子(核: 思核, z0, n: int = 400, δ: float = 1e-7):
    """非线性算子上估计有限时间李雅普诺夫指数（与 步3 同口径）。"""
    rng = np.random.default_rng(20261006 + 7)
    z1 = _单位(z0)
    d0 = np.where(rng.random(核.维度) < 0.5, -1.0, 1.0)
    d0 = _单位(d0 - (d0 @ z1) * z1)
    z2 = _单位(z1 + δ * d0)
    g1, h1, m1 = dict(核.增益0), {k: 0.0 for k in 核.竞争表}, None
    g2, h2, m2 = dict(核.增益0), {k: 0.0 for k in 核.竞争表}, None
    z1p, z2p = z1.copy(), z2.copy()
    dist0 = float(np.linalg.norm(z2 - z1))
    for _ in range(n):
        z1n, g1, h1, m1, _ = 核.步(z1, g1, h1, z1p, m1)
        z2n, g2, h2, m2, _ = 核.步(z2, g2, h2, z2p, m2)
        z1p, z2p = z1, z2
        z1, z2 = z1n, z2n
    return float(math.log((float(np.linalg.norm(z2 - z1)) + 1e-300) / dist0) / n)


# ────────────────────────── D1：保护开关逐位恒等 ──────────────────────────
def 判D1():
    步5 = 顶覆盖_综合体Z(seed=310, 自我开关=True, 学习开关=True, 自模型开关=True)
    顿悟关 = 顶覆盖_综合体W(seed=310, 顿悟开关=False, 自我开关=True, 学习开关=True,
                          自模型开关=True)
    句s = 长流 * 3
    跑固定流(步5, 句s, 48)
    跑固定流(顿悟关, 句s, 48)
    差 = 找差异(外化(步5), 外化(顿悟关))
    return dict(通过=len(差) == 0, 差异条数=len(差), 差异前5=差[:5],
                顿悟核为空=(顿悟关.顿悟核 is None))


# ────────────────────────── V4：增强但不破 ──────────────────────────
def 判V4():
    关 = 顶覆盖_综合体W(seed=330, 顿悟开关=False, 学习开关=True, 自模型开关=True)
    开 = 顶覆盖_综合体W(seed=330, 顿悟开关=True, 学习开关=True, 自模型开关=True)
    跑固定流(关, 长流 * 8, 128)
    跑固定流(开, 长流 * 8, 128)
    外差 = 找差异(外化(关), 外化(开))
    自开 = [e for e in 开.自言.流 if 顿悟词 in e["文本"]]
    增强 = len(开.自言.流) > len(关.自言.流)
    不破 = (len(外差) == 0)
    return dict(通过=(增强 and 不破), 外化差异条数=len(外差), 外化差异前5=外差[:5],
                顿悟痕_开=len(自开),
                自言总数_关=len(关.自言.流), 自言总数_开=len(开.自言.流),
                事件数_开=len(开.顿悟核.事件拍s),
                样例=[e["文本"][:78] for e in 自开[:3]])


# ────────────────────────── F2：顿悟涌现（世界行动通道；原生 `if cosv>=0.8` 场景） ──────────────────────────
def 跑世界(seed: int, N: int, 开关: bool):
    桥 = 顶覆盖_综合体W(seed=seed, 顿悟开关=开关, 学习开关=True, 自模型开关=True)
    桥.设世界(隐藏世界(seed=77, 动作数=4))
    阈值拍s = []
    for _ in range(N):
        n0 = len(桥.顿悟账本)
        桥.断供一拍()
        if len(桥.顿悟账本) > n0:            # 仅关臂会增长（开臂该分支已被替换）
            阈值拍s.append(int(桥.拍))
    if 开关:                                 # 相变臂：事件在独立顿悟核账本
        return 桥, list(桥.顿悟核.事件拍s)
    return 桥, 阈值拍s                        # 阈值臂：基类 if cosv>=0.8


def 判F2():
    N = 300
    开桥, 相变拍s = 跑世界(331, N, True)          # 相变臂（去 if）
    关桥, 阈值拍s = 跑世界(331, N, False)         # 阈值臂（基类 if cosv>=0.8）

    # F2a 存在性：去掉 if cosv>=0.8 后，相变事件仍发生
    f2a = len(相变拍s) > 0

    # F2b 否证"=单点阈值"：两臂事件拍集合的低重合
    集1, 集2 = set(相变拍s), set(阈值拍s)
    交 = len(集1 & 集2)
    并 = len(集1 | 集2)
    jac = (交 / 并) if 并 > 0 else 0.0
    f2b = f2a and (jac < 0.5)

    # 纯文本态（当前冻结阶段）证据：相变事件同样在文本对话流上涌现
    文本桥 = 顶覆盖_综合体W(seed=331, 顿悟开关=True, 学习开关=True, 自模型开关=True)
    跑固定流(文本桥, 长流, 400)
    文本事件 = list(文本桥.顿悟核.事件拍s)

    # F2c 关联登记：事件拍附近相空间有限时间 λ 与临界带（|λ|<0.05）的接近度（非硬门）
    #   —— 用**每个事件各自的自态 z**（事件 dict 内）作轨初值，故各事件 λ 不再同值。
    事件λs = []
    if 相变拍s and hasattr(开桥, "eng"):
        Φ = np.asarray(开桥.eng.hdc.Φ_act, dtype=np.float64)
        方向表 = {"动%d" % j: _单位(Φ[:, j]) for j in range(int(Φ.shape[1]))}
        算核 = 思核(方向表)
        for ev in 开桥.顿悟核.顿悟账本[:6]:
            z0 = ev.get("z")
            if z0 is None:
                continue
            事件λs.append(round(有限时间λ_算子(算核, np.asarray(z0, dtype=np.float64)), 6))
    λ均值 = float(np.mean([abs(x) for x in 事件λs])) if 事件λs else None
    λ登记 = dict(事件λs=事件λs, 平均_abs_λ=(round(λ均值, 6) if λ均值 is not None else None),
                临界容差=λ临界容差,
                落临界带=(bool(λ均值 is not None and λ均值 <= λ临界容差)))

    return dict(通过=(f2a and f2b), F2a存在性=f2a, F2b否证阈值同构=f2b,
                相变事件数=len(相变拍s), 相变拍s=相变拍s[:16],
                阈值臂事件数=len(阈值拍s), 阈值拍s=阈值拍s[:16],
                Jaccard_相变vs阈值=round(jac, 4),
                文本态事件数=len(文本事件), 文本态拍s=文本事件[:16],
                事件样例=(开桥.顿悟核.顿悟账本[0] if 开桥.顿悟核.顿悟账本 else None),
                F2c关联登记=λ登记)


# ────────────────────────── D重放：确定性 ──────────────────────────
def 判D重放():
    def 跑():
        桥 = 顶覆盖_综合体W(seed=332, 顿悟开关=True, 学习开关=True, 自模型开关=True)
        桥.设世界(隐藏世界(seed=77, 动作数=4))
        for _ in range(200):
            桥.断供一拍()
        return list(桥.顿悟核.事件拍s)

    e1 = 跑()
    e2 = 跑()
    同 = 逐位相等(e1, e2)
    return dict(通过=同, 事件数1=len(e1), 事件数2=len(e2), 同=同, 拍s=e1[:12])


# ────────────────────────── D6/不直写状态：静态扫描（剥注释/文档串） ──────────────────────────
def _剥注释文档串(src: str) -> str:
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    src = re.sub(r"#[^\n]*", "", src)
    return src


def 静态扫描():
    p = os.path.join(HERE, "内核_顿悟涌现.py")
    src = _剥注释文档串(open(p, encoding="utf-8").read())
    时间阈 = re.findall(r"(?:拍数|拍|time|断供拍数)\s*(?:>=|>|<=|<)\s*\d", src)
    常量顿悟 = re.findall(r"(?:ε[^\n]{0,10}|cosv)\s*>=\s*0\.\d", src)
    直写 = re.findall(r"self\.(?:a_prev|i_prev|prev_c|消化债|饱足)\s*=", src)
    return dict(通过=(len(时间阈) == 0 and len(常量顿悟) == 0 and len(直写) == 0),
                时间阈命中=时间阈, 常量顿悟命中=常量顿悟, 直写状态命中=直写)


# ────────────────────────── 主 ──────────────────────────
def 主():
    报 = {}
    d1 = 判D1()
    v4 = 判V4()
    f2 = 判F2()
    dr = 判D重放()
    扫 = 静态扫描()
    报["D1_关时逐位恒等"] = d1
    报["V4_增强但不破"] = v4
    报["F2_顿悟涌现"] = f2
    报["D重放_确定性"] = dr
    报["D6_静态扫描"] = 扫

    总 = (d1["通过"] and v4["通过"] and f2["通过"] and dr["通过"] and 扫["通过"])
    报["总判"] = bool(总)

    print("=" * 82)
    print("步6 · 深度顿悟涌现（去 if；相变共现替代 if cosv>=0.8；默认关 · 不破既有）自证")
    print("=" * 82)
    print("【D1 关时逐位恒等】%s（差异 %d 条；顿悟核为空=%s）"
          % ("PASS" if d1["通过"] else "FAIL", d1["差异条数"], d1["顿悟核为空"]))
    if not d1["通过"]:
        print("   差异前5：%s" % d1["差异前5"])
    print("【V4 增强不破】%s（外化差异 %d 条；顿悟痕 %d；事件 %d；自言 关%d→开%d）"
          % ("PASS" if v4["通过"] else "FAIL", v4["外化差异条数"], v4["顿悟痕_开"],
             v4["事件数_开"], v4["自言总数_关"], v4["自言总数_开"]))
    for s in v4["样例"]:
        print("      例：%s" % s)
    print("【F2 顿悟涌现】%s（存在性=%s 否证阈值同构=%s）"
          % ("PASS" if f2["通过"] else "FAIL", f2["F2a存在性"], f2["F2b否证阈值同构"]))
    print("     相变臂事件 %d 个；阈值臂(if cosv>=0.8) %d 个；Jaccard=%.4f"
          % (f2["相变事件数"], f2["阈值臂事件数"], f2["Jaccard_相变vs阈值"]))
    print("     相变拍：%s" % f2["相变拍s"])
    print("     阈值拍：%s" % f2["阈值拍s"])
    print("     纯文本态事件 %d 个（当前冻结阶段证据）：%s"
          % (f2["文本态事件数"], f2["文本态拍s"]))
    print("     样例：%s" % f2["事件样例"])
    print("     F2c λ关联登记：%s" % f2["F2c关联登记"])
    print("【D重放 确定性】%s（事件数 %d/%d；拍 %s）"
          % ("PASS" if dr["通过"] else "FAIL", dr["事件数1"], dr["事件数2"], dr["拍s"]))
    print("【D6/不直写状态 静态扫描】%s（时间阈=%s 常量顿悟=%s 直写=%s）"
          % ("PASS" if 扫["通过"] else "FAIL", 扫["时间阈命中"], 扫["常量顿悟命中"],
             扫["直写状态命中"]))
    print("-" * 82)
    print("总判：%s" % ("PASS" if 总 else "FAIL"))
    print("（未做任何「已完成」宣称——数字以上表为准）")
    print("=" * 82)

    出 = os.path.join(HERE, "步6_结果.json")
    with open(出, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1, default=str)
    print("[已存] %s" % 出)
    return 报


if __name__ == "__main__":
    主()