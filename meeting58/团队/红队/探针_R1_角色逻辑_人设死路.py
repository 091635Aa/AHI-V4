# -*- coding: utf-8 -*-
"""meeting58 · 红队探针2：攻击「角色逻辑」——角色可辨识到底在辨什么。

CPU-only，只读 import 既有件（不改），产物落本目录。目标 ≤10s。
探的六件事：
  R1 人设死路：只改 人设、不改 记忆/听见句 ⇒ 说话器输出逐字不变
  R2 字符LM 死代码：`说()` 不调用 分布/候选（自回归只存在于注释）
  R3 判据_挂载 A1「三角色可辨识 1.0」的零假设：三包**同池**（只换人设）时准确率 ≈ 机会
  R4 池子即身份：三角色词表/记忆 Jaccard ≈ 0 ⇒ A1 等价于按池查表
  R5 判据_挂载 A3「记忆召回正确率 1.0」不可失败：置乱记忆后仍 1.0；跨包串公式恒 0
  R6 import 图：文本层与音色层之间**没有边**（不是解耦，是没接线）
"""
from __future__ import annotations

import ast
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # …/meeting58
AHI4 = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(AHI4, "挂载"))
sys.path.insert(0, os.path.join(AHI4, "meeting53", "实施"))

import 角色包 as 包模块  # noqa: E402
import 说话_连贯 as 说话  # noqa: E402
import 挂载_角色 as 挂载  # noqa: E402
import V5_双通道 as V5  # noqa: E402

np.seterr(all="ignore")


def _包(包, 人设=None, 记忆=None, 听见=None, 词=None):
    d = dict(包)
    if 人设 is not None:
        d["人设"] = 人设
    if 记忆 is not None:
        d["记忆条目列"] = 记忆
    if 听见 is not None:
        d["听见句列"] = 听见
    if 词 is not None:
        d["词语列"] = 词
    return d


def R1_人设死路(lm):
    洛 = 包模块.建_洛天依()
    moss = 包模块.建_MOSS()
    变体 = {
        "洛天依(原)": 洛,
        "只改人设为MOSS": _包(洛, 人设=moss["人设"]),
        "人设清空": _包(洛, 人设=""),
        "人设塞入不存在规则": _包(洛, 人设="洛天依只说火星语，每句必须带'火星'二字；绝不唱歌。"),
    }
    o = np.sign(np.random.default_rng(3).standard_normal(64)) * 0.5
    文 = {}
    for 名, 包 in 变体.items():
        器 = 说话.说话器(包, lm)
        文[名] = "".join(器.说(o, 拍=t) for t in range(3))
    return dict(输出=文, 四种输出全同=len(set(文.values())) == 1,
                结论="`说()` 完全不读 人设（也不读 角色LM）；改人设到「只说火星语」都不掉一个字 ⇒ 当前『角色』= 记忆句池，不是人设")


def R2_LM死代码(lm):
    洛 = 包模块.建_洛天依()
    器 = 说话.说话器(洛, lm)
    计 = {"分布": 0, "候选": 0}
    原分布, 原候选 = 说话.字符LM.分布, 说话.说话器.候选

    def 计分布(self, *a, **k):
        计["分布"] += 1
        return 原分布(self, *a, **k)

    def 计候选(self, *a, **k):
        计["候选"] += 1
        return 原候选(self, *a, **k)

    说话.字符LM.分布 = 计分布
    说话.说话器.候选 = 计候选
    try:
        o = np.sign(np.random.default_rng(3).standard_normal(64)) * 0.5
        _ = 器.说(o, 拍=0)
        _ = 器.流式(o, 拍=0)
        说后 = dict(计)
        _ = 器.候选("的")
        显式调用后 = dict(计)
    finally:
        说话.字符LM.分布 = 原分布
        说话.说话器.候选 = 原候选
    return dict(说_流式期间调用计数=说后, 显式调候选后=显式调用后,
                结论="`说/流式` 期间 分布/候选 调用 = 0 ⇒ 自回归/字符LM 是死代码（与 meeting54 红队结论一致，可复跑）")


def _挂载A1(状态, 包列, 训练拍=120):
    器 = 挂载.挂载器()
    跑果 = {名: 器.跑(包, 状态) for 名, 包 in 包列.items()}
    全部词 = sorted({w for 名 in 跑果 for 行 in 跑果[名]["行"][训练拍:] for w in 行["词"]})
    词id = {w: i for i, w in enumerate(全部词)}
    df = {w: sum(1 for 名 in 跑果 if any(w in 行["词"] for 行 in 跑果[名]["行"][训练拍:])) for w in 全部词}
    权 = np.array([1.0 / max(1, df[w]) for w in 全部词])

    def _特征(行):
        v = np.zeros(len(全部词))
        for w in 行["词"]:
            if w in 词id:
                v[词id[w]] += 1.0
        return v * 权

    样 = [(名, _特征(行)) for 名, r in 跑果.items() for 行 in r["行"][训练拍:]]
    半 = len(样) // 2
    # 与 判据_挂载.py:74-76 逐字同口径（含「首半为空 ⇒ 零质心」的隐藏分支）
    质 = {}
    for 名 in 跑果:
        X = [v for (n, v) in 样[:半] if n == 名]
        质[名] = np.mean(X, axis=0) if X else np.zeros(len(全部词))
    对 = 总 = 0
    for (名, v) in 样[半:]:
        预 = min(质, key=lambda k: float(np.linalg.norm(v - 质[k])))
        对 += int(预 == 名); 总 += 1
    正文s = {名: len({tuple(x["词"]) for x in r["行"][训练拍:]}) for 名, r in 跑果.items()}
    return dict(acc=round(对 / max(1, 总), 4), 样本=总, 各角色正文种类=正文s)


def R3_A1零假设(状态):
    洛 = 包模块.建_洛天依()
    moss = 包模块.建_MOSS()
    日记 = 包模块.建_日记角色()
    现状 = {"洛天依": 洛, "MOSS": moss, "日记角色": 日记}
    同池 = {
        "洛天依": _包(洛, 人设=洛["人设"]),
        "MOSS": _包(洛, 人设=moss["人设"]),
        "日记角色": _包(洛, 人设=日记["人设"]),
    }
    a1现 = _挂载A1(状态, 现状)
    a1同 = _挂载A1(状态, 同池)

    def _集(包):
        return set(包["词语列"]) | {w for s in 包["听见句列"] for w in str(s).split(" ")}

    J = {}
    名 = list(现状)
    for i in range(len(名)):
        for j in range(i + 1, len(名)):
            A, B = _集(现状[名[i]]), _集(现状[名[j]])
            J["%s|%s" % (名[i], 名[j])] = round(len(A & B) / max(1, len(A | B)), 4)
    return dict(现状A1=a1现["acc"], 现状正文种类=a1现["各角色正文种类"],
                同池A1=a1同["acc"], 同池正文种类=a1同["各角色正文种类"],
                池Jaccard=J, 机会=round(1 / 3, 3),
                结论="A1=1.0 的零假设不是 0.33：三包词表/句池互不相交（Jaccard≈0），最近质心只是在背池子；把内容换成同池、只留人设 ⇒ 准确率掉到机会水平 ⇒ 可辨识=池子可辨识，不是角色行为可辨识")


def R4_A3不可失败(状态):
    洛 = 包模块.建_洛天依()
    moss = 包模块.建_MOSS()
    器 = 挂载.挂载器()
    置乱 = _包(洛, 记忆=moss["记忆条目列"])
    r = 器.跑(置乱, 状态)
    回 = [x["回忆"] for x in r["行"][120:] if x["回忆"]]
    本包 = {str(x.get("文本", "")) for x in 置乱["记忆条目列"]}
    命中 = sum(1 for x in 回 if str(x.get("文本", "")) in 本包)
    全 = {str(y.get("文本", "")) for z in 包模块.全部包().values() for y in z["记忆条目列"]}
    跨包串 = len(本包 & 全 - 本包)
    return dict(置乱后召回条数=len(回), 置乱后命中=命中,
                正确率=round(命中 / max(1, len(回)), 4),
                跨包串公式读数=跨包串,
                结论="召回只从 self.记忆 里取 ⇒ 『正确率 1.0』不可失败；置乱记忆后仍 1.0。跨包串 = 本包∩全包−本包 恒为空集 ⇒ 该检查是恒真式")


def R5_说话选池(lm):
    洛 = 包模块.建_洛天依()
    器 = 说话.说话器(洛, lm)
    池 = 器.句子池()
    o = np.sign(np.random.default_rng(5).standard_normal(64)) * 0.5
    对 = 0
    for 拍 in range(5):
        n = max(1, min(4, len(池)))
        桶 = int(abs(float(sum(abs(o)))) * 1000.0) % len(池)
        步 = 1 if len(池) < 5 else 5
        选, 已 = [], set()
        i = 0
        while len(选) < n and i < len(池) * 4:
            idx = (桶 + i * 步 + 拍) % len(池)
            if idx not in 已:
                已.add(idx); 选.append(池[idx])
            i += 1
        对 += int("".join(选) == 器.说(o, 拍=拍))
    return dict(池长=len(池), 公式复现=对, 拍数=5,
                结论="`说()` = 池子按 (桶+拍) 轮转（meeting54 红队公式可逐字复现）⇒ 角色复刻的『语言』部分是查表，不是生成")


def R6_import图():
    根 = AHI4
    文件 = {
        "挂载_角色": os.path.join(根, "挂载", "挂载_角色.py"),
        "说话_连贯": os.path.join(根, "挂载", "说话_连贯.py"),
        "说_音色": os.path.join(根, "meeting55", "实施", "说_音色.py"),
        "声带": os.path.join(根, "meeting32", "声带.py"),
        "V5_双通道": os.path.join(根, "meeting53", "实施", "V5_双通道.py"),
    }
    图 = {}
    for 名, 路 in 文件.items():
        树 = ast.parse(open(路, encoding="utf-8").read())
        名s = set()
        for 节 in ast.walk(树):
            if isinstance(节, ast.Import):
                名s |= {a.name.split(".")[0] for a in 节.names}
            elif isinstance(节, ast.ImportFrom) and 节.module:
                名s.add(节.module.split(".")[0])
        图[名] = sorted(n for n in 名s if n in 文件 or n in ("V5_双通道", "角色包", "声带"))
    return dict(import图=图,
                结论="文本层（挂载_角色/说话_连贯）与音色层（说_音色/声带）之间没有 import 边；两层各自 import 到 V5/角色包或 声带 ⇒ 现在是「各自独立」而不是「解耦契约」；接起来要新写 glue（且要处理 D=64↔1024）")


def main():
    t0 = time.perf_counter()
    lm = 说话.建通用LM()
    t_lm = time.perf_counter() - t0
    状态 = 挂载.挂载器().状态列(180)
    t_状 = time.perf_counter() - t0 - t_lm
    出 = dict(耗时秒=None, 建LM秒=round(t_lm, 3), 状态列秒=round(t_状, 3),
              R1=R1_人设死路(lm), R2=R2_LM死代码(lm), R3=R3_A1零假设(状态),
              R4=R4_A3不可失败(状态), R5=R5_说话选池(lm), R6=R6_import图())
    出["耗时秒"] = round(time.perf_counter() - t0, 3)
    p = os.path.join(HERE, "探针_R1_角色逻辑_结果.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(出, f, ensure_ascii=False, indent=1, default=str)
    for k in ("耗时秒", "建LM秒", "状态列秒"):
        print(k, 出[k])
    for k in ("R1", "R2", "R3", "R4", "R5", "R6"):
        print("=" * 20, k)
        print(json.dumps(出[k], ensure_ascii=False)[:1400])
    print("落盘:", p)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
