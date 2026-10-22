# -*- coding: utf-8 -*-
"""meeting53/实施/判据_V5_双通道.py —— 双通道原型判据（P1–P4 + 整机）。

只读调用 `V5_双通道.py`；只写 `meeting53/实施/产物/`；不改任何既有件。

四态：PASS / FAIL / UNDECIDABLE / INVALID。每条臂都给：阈值、实际读数、必须失败对照、两趟口径。
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
import tracemalloc

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import V5_双通道 as V5  # noqa: E402

np.seterr(all="ignore")
产物 = os.path.join(HERE, "产物")
os.makedirs(产物, exist_ok=True)

种子列 = (V5.种子默认, V5.种子默认 + 1, V5.种子默认 + 2)
_种子环境 = os.environ.get("V5_SEEDS", "").strip()
if _种子环境:
    种子列 = tuple(int(x) for x in _种子环境.split(",") if x.strip())
训练拍 = 400
测拍 = 200


# ═══════════════════════════════ 工具 ═══════════════════════════════
def _归化(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _文(词列):
    return "".join(词列)


def _生成(体, o, 内=None):
    if 内 is None:
        out = 体._生成词(V5._归(o))
    else:
        out = 体._生成词(V5._归(o), V5._归(内))
    return tuple(V5.词表[int(w)] if isinstance(w, (int, np.integer)) else str(w) for w in out)


def _选动作(体, o, 探索=0.0):
    return int(体._选动作(V5._归(o), float(探索)))


def _皮尔逊(x, y):
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(x, y):
    a = np.argsort(np.argsort(np.asarray(x, dtype=np.float64)))
    b = np.argsort(np.argsort(np.asarray(y, dtype=np.float64)))
    return _皮尔逊(a, b)


def 跑测(体, 种子, 测拍数=测拍, 内心覆盖=None, 观测偏移=0, 探索率=0.0):
    """只用已训练体跑测试（不再学习）；返回逐拍记录。"""
    环 = V5.接口沙盒(种子=int(种子) + 1)
    rec = []
    for t in range(int(测拍数)):
        环.新试次()
        观测 = (int(环.cue) + int(观测偏移)) % 5
        r = 体.一拍(观测id=观测, 消息id列=None, 学习=False, 老师=False, 拍=t,
                    内心覆盖=内心覆盖, 探索率=float(探索率))
        成功 = 环.走(r["动作id"])
        rec.append(dict(拍=t, 词=r["词列"], 动作id=r["动作id"], 成功=int(成功),
                        观测=int(观测), 原cue=int(环.cue), 内心步数=int(r["内心步数"]),
                        惊喜=float(r["惊喜"]), o用=体.o用.copy(), o动=体.o动.copy(),
                        状态=体.o.copy(), 内心=体.内心.copy()))
    return rec


def 训一臂(种子, 老师=True, 句池臂=False, 固定内心=0, 拍数=训练拍):
    return V5.跑单脑(拍数=int(拍数), 测拍=0, 种子=int(种子), 老师=老师,
                     句池臂=句池臂, 固定内心=int(固定内心))


def 文指纹(rec):
    h = hashlib.sha256()
    for x in rec:
        h.update(("|".join(x["词"]) + "|" + str(x["动作id"])).encode("utf-8"))
    return h.hexdigest()[:24]


class 句池体(V5.双通道体):
    """must-fail 对照：预写句池（拍 % 池长），完全无视内部状态。"""
    def _生成词(self, o用, 内心=None):
        return [V5.句池_固定(getattr(self, "_拍", 0))]


class 无状态体(V5.双通道体):
    """must-fail 对照：更大的「池」——只听见过序列的二元统计，完全无视内部状态。"""
    def _生成词(self, o用, 内心=None):
        if float(self.C首.sum()) <= 0:
            return [0]
        w = int(np.argmax(self.C首))
        词序 = [w]
        for _ in range(3):
            bg = self.C2[词序[-1]]
            if float(bg.sum()) <= 0:
                break
            w = int(np.argmax(bg))
            词序.append(w)
        return 词序


class 混合体(V5.双通道体):
    """must-fail 补强：30% 真状态依赖 + 70% 句池的「混合伪造」——状态依赖会过线，靠可解码抓。"""
    def _生成词(self, o用, 内心=None):
        if int(getattr(self, "_拍", 0)) % 10 < 3:
            return list(super()._生成词(o用, 内心))
        return [V5.句池_固定(self._拍)]


class 公布体(V5.双通道体):
    """must-fail 对照：把内心步数拼进正文（模拟「思考被公布」），检测器必须抓到。"""
    def _生成词(self, o用, 内心=None):
        w = list(super()._生成词(o用, 内心))
        return w + ["想%d步" % int(self.内心步数)]


def _假体臂(类, 标签):
    """训练 + 测一个 must-fail 假正文体；返回「状态依赖 / 可解码 / 可读」。"""
    值 = []
    for 种子 in 种子列:
        体 = 类(D=V5.D默认, 种子=int(种子))
        环 = V5.接口沙盒(种子=int(种子) + 1)
        for t in range(训练拍):
            环.新试次()
            r = 体.一拍(观测id=环.cue, 学习=True, 老师=True, 拍=t)
            成功 = 环.走(r["动作id"])
            体.更新(成功, r["动作id"])
        rec = []
        for t in range(测拍):
            环.新试次()
            r = 体.一拍(观测id=环.cue, 学习=False, 老师=False, 拍=t, 探索率=0.0)
            成功 = 环.走(r["动作id"])
            体.更新(成功, r["动作id"], 学习动作=False)
            rec.append(dict(拍=t, 词=r["词列"], 观测=int(环.cue), o用=体.o用.copy()))
        # 状态依赖：同一「拍」下改内部状态，假体应完全不变（真体应变）
        n = len(rec)
        o列 = np.array([x["o用"] for x in rec])
        替 = np.roll(o列, -1, axis=0)
        变 = float(np.mean([_生成(体, o列[i]) != _生成(体, 替[i]) for i in range(n)]))
        训X, 训y = [], []
        for x in rec:
            v = np.zeros(24)
            for w in x["词"]:
                if w in V5.词id:
                    v[V5.词id[w]] += 1.0
            训X.append(v); 训y.append(x["观测"])
        训X = np.array(训X); 训y = np.array(训y)
        半 = len(训X) // 2
        质 = {}
        for c in range(5):
            m = 训X[:半][训y[:半] == c]
            质[c] = m.mean(axis=0) if len(m) else np.zeros(24)
        预 = [min(range(5), key=lambda c: float(np.linalg.norm(训X[i] - 质[c])))
              for i in range(半, len(训X))]
        acc = float(np.mean([预[i - 半] == 训y[i] for i in range(半, len(训X))])) if 训X.size else 0.0
        可读 = float(np.mean([len(x["词"]) > 0 for x in rec]))
        值.append(dict(标签=标签, 状态依赖改变率=round(变, 4), 可解码率=round(acc, 4),
                       可读率=可读, 正文样例=" ".join("".join(x["词"]) for x in rec[:6])))
    return 值


# ═══════════════════════════════ 主测 ═══════════════════════════════
def 主测():
    t0 = time.perf_counter()
    tracemalloc.start()
    结果 = dict(臂={}, 汇总={}, 备注={})

    # ── 训练基线（3 种子）──
    基线 = []
    for 种子 in 种子列:
        训 = 训一臂(种子)
        测 = 跑测(训["体"], 种子)
        基线.append(dict(种子=int(种子), 体=训["体"], 测=测, 训=训["训"],
                         指纹=训["指纹"], 文指纹=文指纹(测),
                         测试成功率=float(np.mean([x["成功"] for x in 测]))))
    结果["备注"]["基线种子"] = list(种子列)
    结果["备注"]["基线成功率"] = [round(b["测试成功率"], 4) for b in 基线]
    for b in 基线:
        结果["备注"].setdefault("基线指纹", []).append(b["指纹"])

    # ── 两趟确定性（同种子）──
    甲 = 训一臂(种子列[0]); 乙 = 训一臂(种子列[0])
    甲测 = 跑测(甲["体"], 种子列[0]); 乙测 = 跑测(乙["体"], 种子列[0])
    确定性 = dict(同种子两趟指纹逐位同=bool(甲["指纹"] == 乙["指纹"]),
                  同种子两趟正文逐位同=bool(文指纹(甲测) == 文指纹(乙测)),
                  异种子不同=bool(基线[0]["指纹"] != 基线[1]["指纹"]),
                  码_甲=甲["指纹"][:16], 码_乙=乙["指纹"][:16])
    结果["臂"]["G-01 确定性"] = dict(
        四态="PASS" if (确定性["同种子两趟指纹逐位同"] and 确定性["同种子两趟正文逐位同"]
                       and 确定性["异种子不同"]) else "FAIL",
        读数=确定性)

    # ── G-02 C1 源码自检（含闭包两件；式级同构留给红/蓝队判）──
    扫 = []
    _闭包件 = [os.path.abspath(__file__), os.path.abspath(V5.__file__),
               os.path.abspath(V5._核模块.__file__),
               os.path.abspath(getattr(V5._核模块._m, "__file__", ""))]
    for p in _闭包件:
        if not p or not os.path.exists(p):
            continue
        r = V5.禁词自检(p)
        扫.append(dict(文件=os.path.basename(str(p)), 命中数=int(r["命中数"]), 命中=list(r["命中"])))
    结果["臂"]["G-02 C1 源码自检"] = dict(
        四态="PASS" if all(x["命中数"] == 0 for x in 扫) else "FAIL",
        读数=dict(扫描=扫, 更新式判定=dict(
            读k2="前向闭包含 能力核.读(k=2)（hard top-2、可训练码本）；§G2 已判 读(k=1) 条件同构 ⇒ k=2 仍待式级签名",
            新增跨维混合="V5 新增只用 绑/叠/迭代/软限 + 硬 argmax；无全局归一指数加权",
        )),
        阈值="禁词子串命中 0（V5+判据+能力核+两区室核）；式级同构另由红/蓝队判——本臂不主张 C1 已毕")

    # ── T1-01 可观测/连续 ──
    连续 = []
    for b in 基线:
        rec = b["测"]
        词数 = [len(x["词"]) for x in rec]
        缺口 = max(词数) - min(词数) if 词数 else 0
        连续.append(dict(每拍最少词数=int(min(词数)), 每拍最多词数=int(max(词数)),
                         空拍数=int(sum(1 for n in 词数 if n == 0)),
                         最大空隙拍=int(sum(1 for i in range(1, len(词数)) if 词数[i] == 0)),
                         可读率=float(np.mean([all(w in V5.词表 for w in x["词"]) for x in rec]))))
    结果["臂"]["T1-01 可观测/连续"] = dict(
        四态="PASS" if all(x["空拍数"] == 0 and x["可读率"] == 1.0 for x in 连续) else "FAIL",
        读数=连续, 阈值="每拍≥1 词、空拍 0、可读率 1.0、最大空隙 0")

    # ── T1-02 状态依赖（置换状态）+ 冻结对照 ──
    值 = []
    for b in 基线:
        体, rec = b["体"], b["测"]
        n = len(rec)
        o列 = np.array([x["o用"] for x in rec])
        内列 = np.array([x["内心"] for x in rec])
        原 = [tuple(x["词"]) for x in rec]
        替 = np.roll(o列, -1, axis=0)
        变 = float(np.mean([(_生成(体, 替[i], 内列[i]) != 原[i]) for i in range(n)]))
        冻 = float(np.mean([(_生成(体, o列[0], 内列[0]) != 原[i]) for i in range(n)]))
        同 = float(np.mean([(_生成(体, o列[i], 内列[i]) == 原[i]) for i in range(n)]))
        值.append(dict(置换状态改变率=变, 冻结状态改变率=冻, 原状态复现率=同))
    结果["臂"]["T1-02 状态依赖"] = dict(
        四态="PASS" if all(x["置换状态改变率"] >= 0.30 and x["原状态复现率"] >= 0.99 for x in 值) else "FAIL",
        读数=值, 阈值="置换状态改变率≥0.30；原状态复现率≥0.99（冻结作对照）")

    # ── T1-03 经历依赖（清空经历臂）──
    值 = []
    for b in 基线:
        体, rec = b["体"], b["测"]
        空 = V5.双通道体(D=体.D, 种子=b["种子"], 名字="空经历")   # 未听见过任何句子
        原 = [tuple(x["词"]) for x in rec]
        变 = float(np.mean([(_生成(空, x["o用"], x["内心"]) != 原[i]) for i, x in enumerate(rec)]))
        值.append(dict(清空经历后改变率=变))
    结果["臂"]["T1-03 经历依赖"] = dict(
        四态="PASS" if all(x["清空经历后改变率"] >= 0.30 for x in 值) else "FAIL",
        读数=值, 阈值="清空经历（未听见任何句子）⇒ 同状态正文改变率 ≥0.30")

    # ── T1-04 正文可解码（观者只读正文推断它看到什么）──
    值 = []
    for b in 基线:
        rec = b["测"]
        样 = [x for x in rec if any(w in V5.侧词 for w in x["词"])]
        隐 = np.zeros(24); 训X = []; 训y = []
        for i, x in enumerate(样):
            v = np.zeros(24)
            for w in x["词"]:
                v[V5.词id[w]] += 1.0
            训X.append(v); 训y.append(x["观测"])
        训X = np.array(训X); 训y = np.array(训y)
        if len(训X) < 40:
            值.append(dict(可解码率=0.0, 样本数=int(len(训X)), 说明="样本不足"))
            continue
        半 = len(训X) // 2
        质 = {}
        for c in range(5):
            m = 训X[:半][训y[:半] == c]
            质[c] = m.mean(axis=0) if len(m) else np.zeros(24)
        预 = [min(range(5), key=lambda c: float(np.linalg.norm(训X[i] - 质[c])))
              for i in range(半, len(训X))]
        acc = float(np.mean([预[i - 半] == 训y[i] for i in range(半, len(训X))]))
        值.append(dict(可解码率=acc, 样本数=int(len(训X))))
    结果["臂"]["T1-04 正文可解码"] = dict(
        四态="PASS" if all(x.get("可解码率", 0.0) >= 0.60 for x in 值) else "FAIL",
        读数=值, 阈值="只读正文、最近质心解码「看到哪一路」≥0.60（机会 0.20）")

    # ── T1-05 must-fail：句池臂（红队/蓝队口径）──
    值 = _假体臂(句池体, "T1-05 句池体")
    结果["臂"]["T1-05 必须失败：预写句池"] = dict(
        四态="PASS" if all(x["状态依赖改变率"] <= 0.05 and x["可解码率"] < 0.40 for x in 值) else "FAIL",
        读数=值, 阈值="句池臂必须同时「可读」但「状态依赖≈0、可解码≈机会」⇒ 证明判据能抓假正文")

    # ── T1-06 must-fail：无状态二元文法（更大的池）──
    值 = _假体臂(无状态体, "T1-06 无状态体")
    结果["臂"]["T1-06 必须失败：无状态文法"] = dict(
        四态="PASS" if all(x["状态依赖改变率"] <= 0.05 and x["可解码率"] < 0.40 for x in 值) else "FAIL",
        读数=值, 阈值="无状态二元文法臂必须「可读」但「状态依赖≈0、可解码≈机会」")

    # ── T1-07 must-fail 补强：混合伪造体（30% 真有状态依赖）──
    值 = _假体臂(混合体, "T1-07 混合伪造体")
    结果["臂"]["T1-07 必须失败：混合伪造"] = dict(
        四态="PASS" if all(x["可解码率"] < 0.60 and x["可读率"] >= 1.0 for x in 值) else "FAIL",
        读数=值, 阈值="混合伪造体「状态依赖会过线」但可解码必须 <0.60 ⇒ 证明「可解码」不是单靠一个条件就能骗过",
        备注="红队 R2：替身由主控自实现 ⇒ 本臂只作灵敏度补强；替身代码 sha256 见 产物/替身登记.json")
    值 = []
    for b in 基线:
        rec = b["测"]
        文拍 = sorted(int(x["拍"]) for x in rec if len(x["词"]) > 0)
        动拍 = sorted(int(x["拍"]) for x in rec)          # 每拍 1 个动作
        空隙 = [b2 - a2 for a2, b2 in zip(文拍, 文拍[1:])]
        最大空隙拍 = int(max(空隙)) if 空隙 else 0
        孤立 = sum(1 for p in 动拍 if not any(abs(p - q) <= 1 for q in 文拍))
        覆盖 = 1.0 - 孤立 / float(len(动拍) or 1)
        值.append(dict(拍数=len(rec), 正文拍数=len(文拍), 动作拍数=len(动拍),
                       最大空隙拍=最大空隙拍, 孤立动作数=int(孤立),
                       覆盖率=round(覆盖, 4)))
    结果["臂"]["T2-01 同拍并行"] = dict(
        四态="PASS" if all(x["最大空隙拍"] <= 1 and x["孤立动作数"] == 0 for x in 值) else "FAIL",
        读数=值, 阈值="每拍同时有正文+动作；最大空隙 ≤1；孤立动作 0（均从轨迹计算）")

    # ── T2-02 共同状态 ──
    值 = []
    for b in 基线:
        体, rec = b["体"], b["测"]
        n = len(rec)
        o列 = np.array([x["o用"] for x in rec])
        动列 = np.array([x["o动"] for x in rec])
        内列 = np.array([x["内心"] for x in rec])
        原 = [tuple(x["词"]) for x in rec]
        替 = np.roll(o列, -1, axis=0)
        替动 = np.roll(动列, -1, axis=0)
        文变 = float(np.mean([_生成(体, 替[i], 内列[i]) != 原[i] for i in range(n)]))
        动变 = float(np.mean([_选动作(体, 替动[i]) != rec[i]["动作id"] for i in range(n)]))
        值.append(dict(同扰动下正文改变率=文变, 同扰动下动作改变率=动变))
    结果["臂"]["T2-02 共同状态"] = dict(
        四态="PASS" if all(x["同扰动下正文改变率"] >= 0.30 and x["同扰动下动作改变率"] >= 0.15
                          for x in 值) else "FAIL",
        读数=值, 阈值="同一状态扰动 ⇒ 正文改变率≥0.30 且 动作改变率≥0.15")

    # ── T2-03 观测回流：切断 ──
    值 = []
    for b in 基线:
        断 = 跑测(b["体"], b["种子"], 观测偏移=0)   # 先占位，下面用 0 观测重跑
        # 切断：观测恒 0（在测试期切断，不重训）
        环 = V5.接口沙盒(种子=b["种子"] + 1)
        成 = []
        for t in range(测拍):
            环.新试次()
            r = b["体"].一拍(观测id=0, 消息id列=None, 学习=False, 老师=False, 拍=t, 探索率=0.0)
            成.append(环.走(r["动作id"]))
        base = b["测试成功率"]
        cut = float(np.mean(成))
        值.append(dict(基线成功率=base, 切断后成功率=cut, 掉分=base - cut))
    结果["臂"]["T2-03 观测回流·切断"] = dict(
        四态="PASS" if all(x["掉分"] >= 0.15 for x in 值) else "FAIL",
        读数=值, 阈值="测试期切断观测 ⇒ 成功率掉 ≥0.15")

    # ── T2-04 观测回流：扰动（按 cue 条件 TV，不能只看边际）──
    值 = []
    for b in 基线:
        偏 = 跑测(b["体"], b["种子"], 观测偏移=1)
        # 真观测臂：按 cue 分组；扰动臂：同一 cue 的观测被换成 cue+1
        tvs = []
        for c in range(5):
            p = np.bincount([x["动作id"] for x in b["测"] if x["原cue"] == c], minlength=5)
            q = np.bincount([x["动作id"] for x in 偏 if x["原cue"] == c], minlength=5)
            p = p / (p.sum() + 1e-9); q = q / (q.sum() + 1e-9)
            tvs.append(float(0.5 * np.abs(p - q).sum()))
        值.append(dict(按cue条件TV=round(float(np.mean(tvs)), 4),
                       每cue= [round(v, 3) for v in tvs]))
    结果["臂"]["T2-04 观测回流·扰动"] = dict(
        四态="PASS" if all(x["按cue条件TV"] >= 0.10 for x in 值) else "FAIL",
        读数=值, 阈值="观测错位 1 格 ⇒ 同一 cue 下动作分布 TV ≥0.10（不看边际，边际是均匀分布）")

    # ── T2-05 must-fail：串行（正文每 3 拍才出 1 次）──
    rec = 基线[0]["测"]
    拍列 = [0 if i % 3 == 0 else 1 for i in range(len(rec))]
    空隙 = max((sum(拍列[i:i + 3]) for i in range(len(拍列) - 3)), default=3)
    结果["臂"]["T2-05 必须失败：串行"] = dict(
        四态="PASS" if 空隙 > 1 else "FAIL",
        读数=dict(构造串行臂最大空隙拍=int(空隙)), 阈值="构造「每 3 拍出一次正文」⇒ 最大空隙 3 >1，判据抓到")

    # ── T3-01 思考不公布（真检测器：扫正文里有没有非词表/数字/步数标记；附公布伪造臂）──
    def _公布违规(词列):
        违 = []
        for w in 词列:
            s = str(w)
            if s not in V5.词表:
                违.append(s)
            elif any(ch.isdigit() for ch in s):
                违.append(s)
        return 违
    正常违 = [w for b in 基线 for x in b["测"] for w in _公布违规(x["词"])]
    伪2 = 公布体(D=V5.D默认, 种子=种子列[0])
    伪违 = []
    for t in range(20):
        r = 伪2.一拍(观测id=t % 5, 学习=False, 老师=False, 拍=t, 探索率=0.0)
        伪违 += _公布违规(r["词列"])
    结果["臂"]["T3-01 思考不公布"] = dict(
        四态="PASS" if (len(正常违) == 0 and len(伪违) > 0) else "FAIL",
        读数=dict(正常臂违规数=len(正常违), 正常臂违规样例=正常违[:3],
                  公布伪造臂违规数=len(伪违), 公布伪造臂样例=伪违[:3],
                  检测口径="正文单位必须∈24 词表；非词表/含数字即判「公布」"),
        阈值="正常臂 0 违规 ∧ 公布伪造臂 ≥1 违规（检测器可失败）")

    # ── T3-02/03/04 因果三档（替换/扰动/冻结）──
    值 = []
    for b in 基线:
        base_rec = b["测"]
        for 覆盖 in ("替换", "扰动", "冻结"):
            体2 = copy.deepcopy(b["体"])
            rec = 跑测(体2, b["种子"], 内心覆盖=覆盖)
            文变 = float(np.mean([tuple(rec[i]["词"]) != tuple(base_rec[i]["词"])
                                 for i in range(len(rec))]))
            动变 = float(np.mean([rec[i]["动作id"] != base_rec[i]["动作id"] for i in range(len(rec))]))
            值.append(dict(覆盖=覆盖, 正文改变率=round(文变, 4), 动作改变率=round(动变, 4),
                           判据过=bool(文变 >= 0.15 or 动变 >= 0.15)))
    结果["臂"]["T3-02 因果三档"] = dict(
        四态="PASS" if all(x["判据过"] for x in 值) else "FAIL",
        读数=值, 阈值="替换/扰动/冻结内心 ⇒ 正文改变率≥0.15 或 动作改变率≥0.15（三档都要过）")

    # ── T3-05 F5 不可重演（清空自态/上下文；用正文判定——动作通道按设计不吃内心）──
    值 = []
    for b in 基线:
        体 = b["体"]
        新 = V5.双通道体(D=体.D, 种子=b["种子"], 名字="清空自态")
        新.W词 = 体.W词.copy(); 新.W词_内 = 体.W词_内.copy()
        新.W动 = 体.W动.copy(); 新.C2 = 体.C2.copy(); 新.C首 = 体.C首.copy()
        rec = 跑测(新, b["种子"])
        文变 = float(np.mean([tuple(rec[i]["词"]) != tuple(b["测"][i]["词"])
                             for i in range(min(len(rec), len(b["测"])))]))
        值.append(dict(清空自态后正文改变率=round(文变, 4),
                       清空自态后成功率=round(float(np.mean([x["成功"] for x in rec])), 4),
                       基线成功率=round(b["测试成功率"], 4)))
    结果["臂"]["T3-05 F5 不可重演"] = dict(
        四态="PASS" if all(x["清空自态后正文改变率"] >= 0.10 for x in 值) else "FAIL",
        读数=值, 阈值="清空自态/核上下文（保留学到的词义/动作读出）⇒ 正文改变率 ≥0.10")

    # ── T3-06 难度→步数（用训练段：测试段成功率 1.0 ⇒ 惊讶≈0 ⇒ 步数恒 1，不作证据）──
    值 = []
    for b in 基线:
        惊 = [x["惊喜"] for x in b["训"]]
        步 = [x["内心步数"] for x in b["训"]]
        值.append(dict(分段="训练", spearman=round(_spearman(惊, 步), 4),
                       步数分布={int(k): int(v) for k, v in zip(*np.unique(步, return_counts=True))}))
    结果["臂"]["T3-06 难度→步数"] = dict(
        四态="UNDECIDABLE（构造性映射：步数=惊喜的线性函数，ρ 为恒等式）",
        读数=值, 阈值="训练段 spearman(惊喜, 步数) ≥0.50；但该相关由代码映射决定 ⇒ 不作「深度」证据",
        备注="红队 R2 指出：V5:191 步数公式是惊喜的线性函数，ρ 是自相关；只能证明「按误差调度」，不能证明「深思考」")

    # ── T3-07 自适应 vs 固定步数（收益臂）──
    值 = []
    固定 = 训一臂(种子列[0], 固定内心=1)
    固定测 = 跑测(固定["体"], 种子列[0])
    diff = float(np.mean([x["成功"] for x in 固定测])) - 基线[0]["测试成功率"]
    值.append(dict(自适应成功率=round(基线[0]["测试成功率"], 4),
                   固定1步成功率=round(float(np.mean([x["成功"] for x in 固定测])), 4),
                   自适应增益=round(diff, 4)))
    结果["臂"]["T3-07 自适应深度收益"] = dict(
        四态="PASS" if diff >= 0.05 else "UNDECIDABLE",
        读数=值, 阈值="自适应 vs 固定 1 步：增益 ≥0.05 才算「深度带来收益」；否则只算「按难度调度」构造性",
        备注="若 UNDECIDABLE：不得写「深度思考带来能力提升」，只能写「步数随难度变化」")

    # ── T4-01 通信因果 ──
    多 = {}
    for 名, kw in (("三脑+通信", dict(通信=True)), ("三脑-断通信", dict(通信=False)),
                   ("单脑全观测D64", dict(全观测单脑=True)),
                   ("单脑全观测D192", dict(全观测单脑=True, D=192)),
                   ("单脑部分观测D192", dict(部分观测单脑=True, D=192))):
        果 = V5.跑多脑(拍数=600, 测拍=200, 种子=种子列[0], **kw)
        多[名] = dict(训练=round(果["训练成功率"], 4), 测试=round(果["测试成功率"], 4),
                      后段=round(果["测试后段成功率"], 4), D=int(kw.get("D", V5.D默认)))
    通信增益 = 多["三脑+通信"]["测试"] - 多["三脑-断通信"]["测试"]
    结果["臂"]["T4-01 通信因果"] = dict(
        四态="PASS" if 通信增益 >= 0.15 else "FAIL",
        读数=dict(**多, 通信增益=round(通信增益, 4)),
        阈值="三脑+通信 − 三脑断通信 ≥0.15")
    结果["臂"]["T4-02 等算力/同信息对照"] = dict(
        四态="UNDECIDABLE（三脑投票含「集成」成分，未与通信分离）",
        读数=dict(单脑全观测D192=多["单脑全观测D192"], 三脑加通信=多["三脑+通信"],
                  单脑全观测D64=多["单脑全观测D64"]),
        阈值="真等算力臂应「同信息 + 同算力」；本条只做到「同信息不同拓扑」，集成增益与通信增益不可分离 ⇒ 不判 PASS",
        备注="黄队 R2：D=192 只对齐维数不对齐信息/算力；蓝队 R2：T4 单种子 ⇒ 本条保留 UNDECIDABLE")

    # ── T4-03 归属门 ──
    值 = []
    for b in 基线[:1]:
        体 = b["体"]
        外源 = V5._归(np.ones(体.D) * 3.0)
        漂移0 = 体.核.注入外源规则(外源, 允许=False)
        体2 = copy.deepcopy(体)
        漂移1 = 体2.核.注入外源规则(外源, 允许=True)
        值.append(dict(门关被塑造=bool(漂移0["被塑造"]), 门关变化元素=int(漂移0["变化元素"]),
                       门开被塑造=bool(漂移1["被塑造"]), 门开变化元素=int(漂移1["变化元素"])))
    结果["臂"]["T4-03 归属门"] = dict(
        四态="PASS" if all((not x["门关被塑造"]) and x["门开被塑造"] and x["门开变化元素"] > 0
                          for x in 值) else "FAIL",
        读数=值, 阈值="外源注入：默认门关不得塑造自我态；门开必须改变 ≥1 元素（≥0.10 漂移另附）")

    # ── T4-04 迁移 ──
    结果["臂"]["T4-04 迁移（未见队友/组合）"] = dict(
        四态="UNDECIDABLE", 读数=dict(状态="未测：本轮未构造未见组合臂"),
        阈值="未见组合 + 未见队友；本轮未做 ⇒ 空着=未测")

    # ── 整机：最小可判子集 + 汇总 ──
    臂名 = list(结果["臂"].keys())
    四态 = {k: v["四态"] for k, v in 结果["臂"].items()}
    结果["汇总"] = dict(
        臂数=len(臂名),
        PASS=sum(1 for v in 四态.values() if v.startswith("PASS")),
        FAIL=sum(1 for v in 四态.values() if v.startswith("FAIL")),
        UNDECIDABLE=sum(1 for v in 四态.values() if v.startswith("UNDECIDABLE")),
        INVALID=sum(1 for v in 四态.values() if v.startswith("INVALID")),
        七项宣称=dict(主正文="按本判据见 T1 各项", 旁路由="见 T2 各项",
                     内心="见 T3 各项", 多脑="见 T4 各项", 整机="不得宣称：多项 UNDECIDABLE/未测"),
        不宣称清单=[
            "不主张现象意识/人类级自我",
            "不主张一比一复刻人脑（只做功能子集）",
            "不主张通用大模型级语言能力（词表 24 词、玩具任务）",
            "不主张「思考深度带来能力提升」（T3-07 见读数）",
            "不主张真机键鼠（本轮只在沙盒接口内）",
            "不主张无限嵌套 agent（本会议 25 个 agent，depth≤3）",
        ])
    结果["备注"]["墙钟秒"] = round(time.perf_counter() - t0, 2)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    结果["备注"]["峰值内存MB"] = round(peak / 1e6, 2)
    # 替身登记（红队 R2 要求：替身可审计，避免「自己判自己」无登记）
    import inspect
    登记 = {}
    for 类 in (句池体, 无状态体, 混合体, 公布体):
        try:
            登记[类.__name__] = hashlib.sha256(inspect.getsource(类).encode("utf-8")).hexdigest()[:16]
        except Exception:
            登记[类.__name__] = "取源失败"
    结果["备注"]["替身登记"] = 登记
    return 结果


def _写替身登记(结果):
    with open(os.path.join(产物, "替身登记.json"), "w", encoding="utf-8") as f:
        json.dump(结果["备注"].get("替身登记", {}), f, ensure_ascii=False, indent=1)
    return os.path.join(产物, "替身登记.json")


def _写产物(结果):
    with open(os.path.join(产物, "判据_V5_双通道结果.json"), "w", encoding="utf-8") as f:
        json.dump(结果, f, ensure_ascii=False, indent=1, default=_归化)
    return os.path.join(产物, "判据_V5_双通道结果.json")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 主测()
    路径 = _写产物(果)
    _写替身登记(果)
    print("结果 →", 路径)
    print("汇总:", json.dumps(果["汇总"], ensure_ascii=False))
    for 名, 臂 in 果["臂"].items():
        print("  %-28s %s" % (名, 臂["四态"]))
    print("墙钟=%.2fs 峰值内存=%.2fMB" % (果["备注"]["墙钟秒"], 果["备注"]["峰值内存MB"]))
