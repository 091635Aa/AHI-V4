# -*- coding: utf-8 -*-
"""步4_真学习闭环自证.py —— meeting39 步4 验收：ε_self → 内生偏置 b[]（默认关·不破既有）。

判据（出处：专家05 §2 步4 门 + §3.1 D1/D6/D10 + §3.2 A2/A3/A4；专家01 §3.2/§3.3/§3.4）：
  D1  保护开关：学习开关=False ⇒ 与基类 综合体 **逐位恒等**（拍/账本/自言/e序列/段数/a_prev/i_prev/prev_c）。
  D6  无阈值  ：新文件**不含**时间/拍数阈值（静态扫描）。
  D10 段守恒  ：学习通路**不写记忆库**——连续 _引擎步 后 `codec._段` 段数不变。
  A2  真学习  ：闭环下 ε_self 随经验下降（末窗<首窗）。
  A3  记忆≠学习：删记忆（清 codec._段）后，学习迹/ε史**仍在且继续增长**；两者物理分离。
  A4  ε_self 驱动行为：抬高 ε_self（高于自身基线）⇒ 动作序列分化 vs 关臂；
      **sham**：注入同幅但**零异常**的 ε（恒等于其慢基线 ⇒ 项≡0）⇒ 不分化。
  A5' 确定性重放：同 seed 两次运行，ε史 与 动作序列 逐位一致。
  『不直写状态』：新文件**不含** a_prev/i_prev/prev_c 的直接赋值（静态扫描）。

纪律：纯 numpy、零梯度、确定性；只在测试中显式加载内核；不改任何既有 .py。
用法：python 步4_真学习闭环自证.py
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

from synthesis import 综合体                                    # noqa: E402
from 内核_真学习 import 顶覆盖_综合体L, 驱动_ε, 开关默认            # noqa: E402

世界根 = [HERE]
熟悉句s = ["今天的风很轻。", "我把桌子擦干净了。", "窗外有一棵树。", "水杯放在手边。",
          "我慢慢翻了一页书。", "灯一直亮着。", "他坐在对面没说话。", "雨点敲着玻璃。"]


# ────────────────────────── 工具 ──────────────────────────
def 快照(桥):
    return dict(
        拍=int(桥.拍),
        消化债=float(桥.消化债), 饱足=float(桥.饱足),
        回复台账=[dict(号=d["号"], 问=d["问"], 响应=d["响应"],
                       延迟拍=d.get("延迟拍"),
                       支撑账本=list(d.get("支撑账本", []))) for d in 桥.回复台账],
        账本=[list(x) for x in 桥.账本],
        自言=[dict(号=e["号"], 类型=e["类型"], 文本=e["文本"]) for e in 桥.自言.流],
        e序列=[float(x) for x in 桥.eng.e序列],
        段数=len(桥.codec._段),
        a_prev=桥.a_prev.tolist(), i_prev=int(桥.i_prev),
        prev_c=(None if 桥.prev_c is None else int(桥.prev_c)),
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
    """走既有 步()（含记忆注册），用于 D1 逐位恒等与闭环学习。"""
    for i in range(n):
        桥.步(句s[i % len(句s)])


def 动作序列_引擎步(桥, us):
    """直接走 _引擎步（不经记忆注册），返回 a星序号序列（决策面）。"""
    出 = []
    for u in us:
        出.append(int(桥._引擎步(u)["a星序号"]))
    return 出


# ────────────────────────── D1 保护开关：逐位恒等 ──────────────────────────
def 判D1():
    基 = 综合体(seed=310)
    关 = 顶覆盖_综合体L(seed=310, 学习开关=False, 自模型开关=False, 思核开关=False)
    句s = 熟悉句s * 4
    跑固定流(基, 句s, 24)
    跑固定流(关, 句s, 24)
    差 = 找差异(快照(基), 快照(关))
    return dict(通过=len(差) == 0, 差异条数=len(差), 差异前5=差[:5],
                drv为原类=(type(关.drv).__name__ == "内生驱动器"))


# ────────────────────────── A4：ε_self 驱动行为 ──────────────────────────
def 判A4():
    关 = 顶覆盖_综合体L(seed=312, 学习开关=False, 自模型开关=False)
    u维 = int(关.探针.shape[0])
    us = [关.感知向量(关.codec.编码(s)) for s in (熟悉句s * 6)]
    N = len(us)
    关序 = 动作序列_引擎步(关, us)

    # 高 ε：ε 恒定 1.5（远高于其自身基线）⇒ 项>0 ⇒ 应分化
    高 = 顶覆盖_综合体L(seed=312, 学习开关=True, 自模型开关=False)
    高._ε源 = lambda b: 1.5
    高序 = 动作序列_引擎步(高, us)
    分化高 = sum(1 for a, b in zip(关序, 高序) if a != b) / max(N, 1)

    # sham：ε 恒等于其慢基线（首拍注入 0.0 ⇒ ε_slow 恒 0 ⇒ 项≡0）⇒ 不分化
    sh = 顶覆盖_综合体L(seed=312, 学习开关=True, 自模型开关=False)
    sh._ε源 = lambda b: 0.0
    sh序 = 动作序列_引擎步(sh, us)
    分化sh = sum(1 for a, b in zip(关序, sh序) if a != b) / max(N, 1)

    # 诚实登记：随机注入（非自指形式）因 max(0, ε−slow) 仍含正偏置 ⇒ 不构成有效 sham
    rng = np.random.default_rng(20261006)
    随 = 顶覆盖_综合体L(seed=312, 学习开关=True, 自模型开关=False)
    随._ε源 = (lambda b: float(rng.uniform(0.0, 3.0)))
    随序 = 动作序列_引擎步(随, us)
    分化随 = sum(1 for a, b in zip(关序, 随序) if a != b) / max(N, 1)

    a4 = (分化高 > 0.2) and (分化sh == 0.0)
    return dict(通过=a4, 拍数=N,
                高ε项_eval=float(高.drv.ε_slow), 分化_高ε=round(分化高, 4),
                分化_sham零异常=round(分化sh, 4), 分化_随机诚实登记=round(分化随, 4),
                关序=关序[:12], 高序=高序[:12], sham序=sh序[:12])


# ────────────────────────── A2/A3：真学习 + 记忆≠学习 ──────────────────────────
def 判A2A3():
    桥 = 顶覆盖_综合体L(seed=313, 学习开关=True, 自模型开关=True)
    桥.设开放域(世界根, 屏蔽s=())
    句s = (熟悉句s * 12)
    for s in 句s[:60]:
        桥.步(s)
    ε = 桥.自模型.ε序列()
    k = 20
    首 = float(np.mean(ε[:k]))
    末 = float(np.mean(ε[-k:]))
    a2 = 末 < 首 - 1e-6

    # A3 分离：清记忆 → 学习迹/ε史仍在；继续跑 → 二者仍增长；段数不因学习而增
    迹前, ε前 = len(桥.自模型.迹), len(桥.drv.ε史)
    桥.codec._段 = []
    桥.codec._段向量 = []
    桥.codec._段字袋 = []
    桥.codec._共现语料 = []
    迹后, ε后 = len(桥.自模型.迹), len(桥.drv.ε史)
    删后不丢 = (迹后 == 迹前) and (ε后 == ε前) and (len(桥.codec._段) == 0)
    # 继续 _引擎步（不写记忆）
    段0 = len(桥.codec._段)
    us = [桥.感知向量(桥.codec.编码(s)) for s in 熟悉句s]
    动作序列_引擎步(桥, us)
    迹增 = len(桥.自模型.迹) > 迹后
    ε增 = len(桥.drv.ε史) > ε后
    段平 = (len(桥.codec._段) == 段0 == 0)
    a3 = 删后不丢 and 迹增 and ε增 and 段平
    return dict(通过A2=a2, 通过A3=a3,
                ε首窗=round(首, 4), ε末窗=round(末, 4), ε条数=len(ε),
                分离=dict(迹前=迹前, 迹后=迹后, ε前=ε前, ε后=ε后,
                          删记忆不丢=删后不丢, 继续学_迹增=迹增, 继续学_ε增=ε增,
                          学习不写记忆_段平=段平),
                存储对象独立=(type(桥.自模型.迹).__name__ != type(桥.codec._段).__name__))


# ────────────────────────── D10：段守恒 ──────────────────────────
def 判D10():
    桥 = 顶覆盖_综合体L(seed=314, 学习开关=True, 自模型开关=True)
    us = [桥.感知向量(桥.codec.编码(s)) for s in 熟悉句s]
    段0 = len(桥.codec._段)
    拍_段 = []
    for i in range(30):
        桥._引擎步(us[i % len(us)])
        拍_段.append(len(桥.codec._段))
    return dict(通过=all(x == 段0 for x in 拍_段), 段0=段0, 末段=拍_段[-1])


# ────────────────────────── A5'：确定性重放 ──────────────────────────
def 判重放():
    def 跑():
        桥 = 顶覆盖_综合体L(seed=315, 学习开关=True, 自模型开关=True)
        桥.设开放域(世界根, 屏蔽s=())
        for s in (熟悉句s * 5)[:40]:
            桥.步(s)
        us = [桥.感知向量(桥.codec.编码(s)) for s in 熟悉句s]
        序 = 动作序列_引擎步(桥, us)
        return 桥.drv.ε史, 序
    ε1, 序1 = 跑()
    ε2, 序2 = 跑()
    同ε = (len(ε1) == len(ε2)) and all((a == b) or (math.isnan(a) and math.isnan(b))
                                     for a, b in zip(ε1, ε2))
    同序 = (序1 == 序2)
    return dict(通过=(同ε and 同序), ε条数=len(ε1), 同ε=同ε, 同序=同序)


# ────────────────────────── D6/不直写状态：静态扫描 ──────────────────────────
def 静态扫描():
    p = os.path.join(HERE, "内核_真学习.py")
    src = open(p, encoding="utf-8").read()
    时间阈 = re.findall(r"(?:拍数|拍|time|断供拍数)\s*(?:>=|>|<=|<)\s*\d", src)
    直写 = re.findall(r"self\.(?:a_prev|i_prev|prev_c|消化债|饱足)\s*=", src)
    return dict(通过=(len(时间阈) == 0 and len(直写) == 0),
                时间阈命中=时间阈, 直写状态命中=直写)


# ────────────────────────── 主 ──────────────────────────
def 主():
    报 = {}
    d1 = 判D1()
    a4 = 判A4()
    aa = 判A2A3()
    d10 = 判D10()
    重 = 判重放()
    扫 = 静态扫描()
    报["D1_关时逐位恒等"] = d1
    报["A4_ε驱动行为"] = a4
    报["A2A3_真学习与分离"] = aa
    报["D10_段守恒"] = d10
    报["A5_确定性重放"] = 重
    报["D6_静态扫描"] = 扫

    总 = (d1["通过"] and a4["通过"] and aa["通过A2"] and aa["通过A3"]
         and d10["通过"] and 重["通过"] and 扫["通过"])
    报["总判"] = bool(总)

    print("=" * 74)
    print("步4 · 真学习闭环（ε_self → 内生偏置 b[]；默认关 · 不破既有）自证")
    print("=" * 74)
    print("【D1 关时逐位恒等】%s（差异 %d 条；drv 保持原类=%s）"
          % ("PASS" if d1["通过"] else "FAIL", d1["差异条数"], d1["drv为原类"]))
    if not d1["通过"]:
        print("   差异前5：%s" % d1["差异前5"])
    print("【A4 ε_self 驱动行为】%s（拍数=%d）"
          % ("PASS" if a4["通过"] else "FAIL", a4["拍数"]))
    print("   分化度：高ε=%.4f ｜ sham(零异常)=%.4f ｜ 随机(诚实登记，非有效sham)=%.4f"
          % (a4["分化_高ε"], a4["分化_sham零异常"], a4["分化_随机诚实登记"]))
    print("【A2 真学习下降】%s（ε首窗=%.4f → 末窗=%.4f，n=%d）"
          % ("PASS" if aa["通过A2"] else "FAIL", aa["ε首窗"], aa["ε末窗"], aa["ε条数"]))
    print("【A3 记忆≠学习】%s  %s" % ("PASS" if aa["通过A3"] else "FAIL", aa["分离"]))
    print("【D10 段守恒】%s（段0=%d → 末=%d）"
          % ("PASS" if d10["通过"] else "FAIL", d10["段0"], d10["末段"]))
    print("【A5 确定性重放】%s（ε条数=%d 同ε=%s 同序=%s）"
          % ("PASS" if 重["通过"] else "FAIL", 重["ε条数"], 重["同ε"], 重["同序"]))
    print("【D6/不直写状态 静态扫描】%s（时间阈=%s 直写状态=%s）"
          % ("PASS" if 扫["通过"] else "FAIL", 扫["时间阈命中"], 扫["直写状态命中"]))
    print("-" * 74)
    print("总判：%s" % ("PASS" if 总 else "FAIL"))
    print("（未做任何「已完成」宣称——数字以上表为准）")
    print("=" * 74)

    出 = os.path.join(HERE, "步4_结果.json")
    with open(出, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)
    print("[已存] %s" % 出)
    return 报


if __name__ == "__main__":
    主()