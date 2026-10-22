# -*- coding: utf-8 -*-
"""步5_自我回路自证.py —— meeting39 步5 验收：自指束留痕 + provenance 门控（默认关·不破既有）。

判据（出处：专家05 §2 步5 契约「V4 增强但不破；A4 可测」；专家03 §4 J2/J5/J7 + §2.1/§2.2）：
  D1  保护开关：自我开关=False ⇒ 与 基类 综合体 **逐位恒等**。
  D1c 步骤隔离：自我开关=False ⇒ 与 **步4 宿主** 同配置 **逐位恒等**（步5 不扰动步2/3/4）。
  V4  增强但不破：自我开 ⇒ 自言新增「自我回路」痕迹(≥1)；**外化零变**（对话流/回复台账/账本/
      e序列/段数/a_prev/i_prev/prev_c 逐位同），即"旁挂只增内省、不改说出去的话"。
  J2  「经过我」显式痕迹：每条自我痕迹携带 À_t 派生字段（一致度）+ provenance 源号，
      类型∈六类，账本引用号 ∈ 账本（可审计）。
  J5  provenance 不可伪造：**只喂外源** ⇒ W_self 逐位不变（max|Δ|=0）；自源 ⇒ W_self 生长（≠0）。
  J1(机制) 反事实扰动：换 档案seed（持久身份分量）⇒ À_t **分化**（自我不是装饰；行为层分化待反哺）。
  J7  确定性重放：同 seed+同档案+同事件序列，两次运行 W_self / 迹 逐位一致。
  A4  ε 仍驱动行为：学习开+自我开时抬高 ε_self ⇒ 动作序列分化；且自我回路**本身**不改决策
      （自我开/关、学习关 时动作序列逐位同）。
  D6/不直写状态：新文件不含时间/拍数阈值、不含 a_prev/i_prev/prev_c/消化债/饱足 直接赋值。

纪律：纯 numpy、零梯度、确定性；只在测试中显式加载内核；不改任何既有 .py。
用法：python 步5_自我回路自证.py
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

from synthesis import 综合体                                              # noqa: E402
from 内核_真学习 import 顶覆盖_综合体L                                      # noqa: E402
from 内核_自我回路 import 顶覆盖_综合体Z, 来源标_自源, 来源标_外源            # noqa: E402

世界根 = [HERE]
熟悉句s = ["今天的风很轻。", "我把桌子擦干净了。", "窗外有一棵树。", "水杯放在手边。",
          "我慢慢翻了一页书。", "灯一直亮着。", "他坐在对面没说话。", "雨点敲着玻璃。"]
自我词 = "自我回路"
V4词 = ("复盘", "自评", "能量", "消化债", "账本", "先掂量")


# ────────────────────────── 工具 ──────────────────────────
def 外化(桥):
    """外化面（旁挂不得改动）：说出去的话 / 账 / 引擎态 / 决策态。"""
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


def 动作序列_引擎步(桥, us):
    出 = []
    for u in us:
        出.append(int(桥._引擎步(u)["a星序号"]))
    return 出


def 单位向量s(桥):
    return [桥.感知向量(桥.codec.编码(s)) for s in (熟悉句s * 4)]


# ────────────────────────── D1：保护开关逐位恒等 ──────────────────────────
def 判D1():
    基 = 综合体(seed=310)
    关 = 顶覆盖_综合体Z(seed=310, 自我开关=False, 学习开关=False,
                        自模型开关=False, 思核开关=False)
    句s = 熟悉句s * 3
    跑固定流(基, 句s, 24)
    跑固定流(关, 句s, 24)
    差 = 找差异(外化(基), 外化(关))
    return dict(通过=len(差) == 0, 差异条数=len(差), 差异前5=差[:5],
                自我核为空=(关.自我核 is None))


# ────────────────────────── D1c：步骤隔离（不扰动步4） ──────────────────────────
def 判D1c():
    步4 = 顶覆盖_综合体L(seed=318, 学习开关=True, 自模型开关=True, 思核开关=True)
    步5关 = 顶覆盖_综合体Z(seed=318, 自我开关=False, 学习开关=True,
                          自模型开关=True, 思核开关=True)
    句s = 熟悉句s * 3
    跑固定流(步4, 句s, 24)
    跑固定流(步5关, 句s, 24)
    差 = 找差异(外化(步4), 外化(步5关))
    return dict(通过=len(差) == 0, 差异条数=len(差), 差异前5=差[:5])


# ────────────────────────── V4：增强但不破 ──────────────────────────
def 判V4():
    # 走**自主（断供）**路径：既有周期自评（自评周期=50）会产出 V4 遥测词（能量/消化债/账本…）；
    # 自我回路在其之上**再加**自指束痕迹 ⇒ 增强；而**外化面（说出去的话/账/决策）逐位不变**。
    书 = ["今天的风很轻。", "我把桌子擦干净了。", "窗外有一棵树。", "水杯放在手边。"] * 3
    界 = list(range(1, len(书) + 1))
    关 = 顶覆盖_综合体Z(seed=320, 自我开关=False, 学习开关=True, 自模型开关=True)
    开 = 顶覆盖_综合体Z(seed=320, 自我开关=True, 学习开关=True, 自模型开关=True)
    关.设教材(书, 界)
    开.设教材(书, 界)
    for _ in range(60):
        关.断供一拍()
        开.断供一拍()
    外差 = 找差异(外化(关), 外化(开))                       # 旁挂不得改外化面
    自开 = [e for e in 开.自言.流 if 自我词 in e["文本"]]
    v4关 = [e for e in 关.自言.流 if any(k in e["文本"] for k in V4词)]
    v4开 = [e for e in 开.自言.流 if any(k in e["文本"] for k in V4词)]
    增强 = len(自开) > 0 and len(开.自言.流) > len(关.自言.流)
    不破 = (len(外差) == 0) and (len(v4关) > 0) and (len(v4开) > 0)
    return dict(通过=(增强 and 不破), 外化差异条数=len(外差), 外化差异前5=外差[:5],
                自我痕迹_开=len(自开),
                自言总数_关=len(关.自言.流), 自言总数_开=len(开.自言.流),
                V4迹_关=len(v4关), V4迹_开=len(v4开),
                样例=[e["文本"][:60] for e in 自开[:2]])


# ────────────────────────── J2：显式痕迹可审计 ──────────────────────────
def 判J2():
    桥 = 顶覆盖_综合体Z(seed=321, 自我开关=True, 学习开关=True, 自模型开关=True)
    跑固定流(桥, 熟悉句s * 6, 48)
    tr = [e for e in 桥.自言.流 if 自我词 in e["文本"]]
    型合 = all(e["类型"] in ("疑问", "假设", "预测", "复盘", "计划", "自评") for e in tr)
    字合 = all(("一致度" in e["文本"]) and ("ε_self" in e["文本"]) for e in tr)
    源合 = all((isinstance(e["触发源"], tuple) and len(e["触发源"]) >= 1
              and e["触发源"][0] in (来源标_自源, 来源标_外源)) for e in tr)
    引合 = all(e["引用"] and all(0 <= a < len(桥.账本) for a, _ in e["引用"]) for e in tr)
    通过 = (len(tr) > 0) and 型合 and 字合 and 源合 and 引合
    return dict(通过=通过, 痕迹条数=len(tr), 类型合规=型合, 字段含À_t=字合,
                源号合规=源合, 引用合规=引合,
                样例=[dict(类型=e["类型"], 触发源=list(e["触发源"]), 文本=e["文本"][:52])
                     for e in tr[:2]])


# ────────────────────────── J5：provenance 不可伪造 ──────────────────────────
def 判J5():
    def 跑(来源标):
        桥 = 顶覆盖_综合体Z(seed=322, 自我开关=True, 学习开关=False, 自模型开关=False)
        桥.设开放域(世界根, 屏蔽s=())
        us = 单位向量s(桥)
        W0 = 桥.自我核.W_self.copy()
        for u in us:
            桥._来源标 = 来源标
            桥._引擎步(u)
        return 桥, float(np.max(np.abs(桥.自我核.W_self - W0)))

    外桥, 外Δ = 跑(来源标_外源)
    自桥, 自Δ = 跑(来源标_自源)
    通过 = (外Δ == 0.0) and (自Δ > 0.0)
    return dict(通过=通过, 外源_maxΔ=外Δ, 自源_maxΔ=自Δ,
                外源次数=外桥.自我核.外源次数, 自源更新次数=自桥.自我核.自源更新次数)


# ────────────────────────── J1(机制)：换档案 ⇒ À_t 分化 ──────────────────────────
def 判J1():
    def 跑(档案seed):
        桥 = 顶覆盖_综合体Z(seed=324, 自我开关=True, 学习开关=True,
                          自模型开关=True, 档案seed=档案seed)
        跑固定流(桥, 熟悉句s * 5, 40)
        Às = [d["À"] for d in 桥.自我核.迹]
        一致度 = [d["一致度"] for d in 桥.自我核.迹]
        return Às, 一致度

    甲, 一甲 = 跑(310)
    乙, 一乙 = 跑(3719)                                  # 换持久身份分量（档案 seed）
    n = min(len(甲), len(乙))
    cos = [_余弦(甲[i], 乙[i]) for i in range(n)]
    同 = (len(甲) == len(乙)) and all(float(甲[i][0]) == float(乙[i][0])
                                     for i in range(min(3, n)))
    均cos = float(np.mean(cos)) if cos else 1.0
    通过 = (均cos < 0.9) and (not 同)                    # 分化（非逐位同）
    return dict(通过=通过, 迹条数=len(甲), 平均cos=round(均cos, 4),
                一致度均值_甲=round(float(np.mean(一甲)), 4),
                一致度均值_乙=round(float(np.mean(一乙)), 4))


# ────────────────────────── J7：确定性重放 ──────────────────────────
def 判J7():
    def 跑():
        桥 = 顶覆盖_综合体Z(seed=325, 自我开关=True, 学习开关=True,
                          自模型开关=True, 档案seed=310)
        桥.设开放域(世界根, 屏蔽s=())
        跑固定流(桥, 熟悉句s * 5, 40)
        return 桥.自我核.W_self.copy(), [d["À"].tolist() for d in 桥.自我核.迹]

    W1, T1 = 跑()
    W2, T2 = 跑()
    同W = 逐位相等(W1.tolist(), W2.tolist())
    同迹 = 逐位相等(T1, T2)
    return dict(通过=(同W and 同迹), 迹条数=len(T1), 同W=同W, 同迹=同迹)


# ────────────────────────── A4：ε 仍驱动行为 + 自我不改决策 ──────────────────────────
def 判A4():
    关 = 顶覆盖_综合体Z(seed=312, 自我开关=False, 学习开关=False, 自模型开关=False)
    us = [关.感知向量(关.codec.编码(s)) for s in (熟悉句s * 6)]   # 与步4 A4 同 seed/同流，可比
    关序 = 动作序列_引擎步(关, us)

    # 自我开/学习关：自我回路**本身**不得改决策 → 与关臂逐位同
    自开 = 顶覆盖_综合体Z(seed=312, 自我开关=True, 学习开关=False, 自模型开关=False)
    自序 = 动作序列_引擎步(自开, us)
    自我不改决策 = (自序 == 关序)

    # 学习开+自我开：抬高 ε_self ⇒ 动作分化（步4 通路仍有效）
    高 = 顶覆盖_综合体Z(seed=312, 自我开关=True, 学习开关=True, 自模型开关=False)
    高._ε源 = lambda b: 1.5
    高序 = 动作序列_引擎步(高, us)
    分化 = sum(1 for a, b in zip(关序, 高序) if a != b) / max(len(us), 1)
    通过 = 自我不改决策 and (分化 > 0.2)
    return dict(通过=通过, 拍数=len(us), 自我不改决策=自我不改决策,
                分化_ε驱动=round(分化, 4), 关序=关序[:12], 高序=高序[:12])


# ────────────────────────── 独立存储（记忆≠学习） ──────────────────────────
def 判独立():
    桥 = 顶覆盖_综合体Z(seed=327, 自我开关=True, 学习开关=True, 自模型开关=True)
    跑固定流(桥, 熟悉句s * 3, 24)
    迹 = 桥.自我核.迹
    迹类 = type(迹).__name__
    段类 = type(桥.codec._段).__name__
    迹s独立 = 迹 not in (桥.codec._段, getattr(桥.codec, "_段向量", None))
    清前 = len(迹)
    桥.codec._段 = []
    迹后 = len(桥.自我核.迹)
    return dict(通过=(迹类 == "list" and 迹s独立 and 迹后 == 清前 and 清前 > 0),
                迹类=迹类, 段类=段类, 迹条数=清前, 清记忆后迹条数=迹后)


# ────────────────────────── D6/不直写状态：静态扫描 ──────────────────────────
def 静态扫描():
    p = os.path.join(HERE, "内核_自我回路.py")
    src = open(p, encoding="utf-8").read()
    时间阈 = re.findall(r"(?:拍数|拍|time|断供拍数)\s*(?:>=|>|<=|<)\s*\d", src)
    直写 = re.findall(r"self\.(?:a_prev|i_prev|prev_c|消化债|饱足)\s*=", src)
    return dict(通过=(len(时间阈) == 0 and len(直写) == 0),
                时间阈命中=时间阈, 直写状态命中=直写)


def _余弦(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ────────────────────────── 主 ──────────────────────────
def 主():
    报 = {}
    d1 = 判D1()
    d1c = 判D1c()
    v4 = 判V4()
    j2 = 判J2()
    j5 = 判J5()
    j1 = 判J1()
    j7 = 判J7()
    a4 = 判A4()
    独 = 判独立()
    扫 = 静态扫描()
    报["D1_关时逐位恒等"] = d1
    报["D1c_步骤隔离"] = d1c
    报["V4_增强但不破"] = v4
    报["J2_显式痕迹"] = j2
    报["J5_provenance门控"] = j5
    报["J1_换档案À分化"] = j1
    报["J7_确定性重放"] = j7
    报["A4_ε驱动行为"] = a4
    报["独立存储"] = 独
    报["D6_静态扫描"] = 扫

    总 = (d1["通过"] and d1c["通过"] and v4["通过"] and j2["通过"] and j5["通过"]
         and j1["通过"] and j7["通过"] and a4["通过"] and 独["通过"] and 扫["通过"])
    报["总判"] = bool(总)

    print("=" * 78)
    print("步5 · 自我回路融合（自指束留痕 + provenance 门控；默认关 · 不破既有）自证")
    print("=" * 78)
    print("【D1 关时逐位恒等】%s（差异 %d 条；自我核为空=%s）"
          % ("PASS" if d1["通过"] else "FAIL", d1["差异条数"], d1["自我核为空"]))
    if not d1["通过"]:
        print("   差异前5：%s" % d1["差异前5"])
    print("【D1c 步骤隔离】%s（自我关 vs 步4宿主 差异 %d 条）"
          % ("PASS" if d1c["通过"] else "FAIL", d1c["差异条数"]))
    print("【V4 增强不破】%s（外化差异 %d 条；自我痕迹 开%d；自言总数 关%d→开%d；V4迹 关%d/开%d）"
          % ("PASS" if v4["通过"] else "FAIL", v4["外化差异条数"], v4["自我痕迹_开"],
             v4["自言总数_关"], v4["自言总数_开"], v4["V4迹_关"], v4["V4迹_开"]))
    for s in v4["样例"]:
        print("      例：%s" % s)
    print("【J2 显式痕迹】%s（痕迹 %d 条；型合=%s 字段含À=%s 源号=%s 引用=%s）"
          % ("PASS" if j2["通过"] else "FAIL", j2["痕迹条数"], j2["类型合规"],
             j2["字段含À_t"], j2["源号合规"], j2["引用合规"]))
    print("【J5 provenance 门控】%s（外源 maxΔ=%s ｜ 自源 maxΔ=%.3f）"
          % ("PASS" if j5["通过"] else "FAIL", j5["外源_maxΔ"], j5["自源_maxΔ"]))
    print("【J1 换档案⇒À_t 分化】%s（迹 %d 条；平均cos=%.4f；一致度均值 甲=%.3f 乙=%.3f）"
          % ("PASS" if j1["通过"] else "FAIL", j1["迹条数"], j1["平均cos"],
             j1["一致度均值_甲"], j1["一致度均值_乙"]))
    print("【J7 确定性重放】%s（迹 %d；同W=%s 同迹=%s）"
          % ("PASS" if j7["通过"] else "FAIL", j7["迹条数"], j7["同W"], j7["同迹"]))
    print("【A4 ε 驱动行为】%s（自我不改决策=%s；分化=%.4f）"
          % ("PASS" if a4["通过"] else "FAIL", a4["自我不改决策"], a4["分化_ε驱动"]))
    print("【独立存储 记忆≠学习】%s  %s" % ("PASS" if 独["通过"] else "FAIL", 独))
    print("【D6/不直写状态 静态扫描】%s（时间阈=%s 直写=%s）"
          % ("PASS" if 扫["通过"] else "FAIL", 扫["时间阈命中"], 扫["直写状态命中"]))
    print("-" * 78)
    print("总判：%s" % ("PASS" if 总 else "FAIL"))
    print("（未做任何「已完成」宣称——数字以上表为准）")
    print("=" * 78)

    出 = os.path.join(HERE, "步5_结果.json")
    with open(出, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)
    print("[已存] %s" % 出)
    return 报


if __name__ == "__main__":
    主()