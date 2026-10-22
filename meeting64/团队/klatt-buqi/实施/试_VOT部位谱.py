# -*- coding: utf-8 -*-
"""meeting64/团队/klatt-buqi/实施/试_VOT部位谱.py —— 立场书 A 的「跑前实测」探针（只读既有件）。

目的（只回答一个问题，不跑 ASR、不联网、不用任何模型）：
  1) 现行 v8.1 的**送气 VOT 段**到底有没有部位谱？（诊断_声学 §4.4 报 p/t/k 谱重心
     3264/3286/2852 Hz「几乎相同」——本件独立复现，并加测「谱平坦度」= 平直旁通占比的证据）
  2) 若把 VOT 帧从「元音共振峰成形（噪形=1）」改成「部位带（爆破带）× 逐帧滑向元音 F2（噪形=0+噪带）」，
     **纯帧层、引擎零改动**，能否把 p/t/k 的 VOT 谱分开？——这是 A 路线最小原型的可行性前置闸门。

纪律：只读 import meeting61/62/63/64 既有件；只写本目录；确定性（种子 300）；不引 ASR/torch/funasr。
输出：本目录 `试_VOT部位谱.json` + 终端表。
"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M64 = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # …/meeting64
AHI4 = os.path.dirname(M64)
for _p in (os.path.join(M64, "实施"), os.path.join(AHI4, "meeting63", "实施"),
           os.path.join(AHI4, "meeting62", "实施"), os.path.join(AHI4, "meeting61", "实施"),
           os.path.join(AHI4, "meeting58", "实施")):
    sys.path.insert(0, _p)

import 音韵_v5 as Y  # noqa: E402
import 韵律_v9 as H9  # noqa: E402
import 引擎_v7 as E7  # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402

FS = 44100
每帧 = FS // 100          # 441
音 = ("pa1", "ta1", "ka1", "ba1", "da1", "ga1", "sa1", "ca1")


def 读档(名="EmoCompanion"):
    with open(os.path.join(AHI4, "meeting61", "产物", "声线档案", 名 + ".json"), encoding="utf-8") as f:
        return 声线档案.从dict(json.load(f))


def VOT帧号(帧):
    """送气塞的 VOT 帧 = 声源==0、噪形==1、噪带 None、幅 > 换气幅（换气只有前 3 帧且幅≈0.07×幅度）。"""
    src = list(帧["声源"])
    噪形 = list(帧["噪形"])
    噪带 = list(帧["噪带"])
    幅 = list(帧["幅"])
    起 = 0
    for i, s in enumerate(src):          # 跳过开头的换气帧
        if s != 0:
            起 = i
            break
    候选 = [i for i in range(起, len(src))
            if src[i] == 0 and int(噪形[i]) == 1 and 噪带[i] is None and 幅[i] > 0.05]
    return 候选


def 谱(波, fi):
    blk = 波[fi * 每帧:(fi + 1) * 每帧]
    if len(blk) < 每帧:
        blk = np.pad(blk, (0, 每帧 - len(blk)))
    w = np.hamming(每帧)
    P = np.abs(np.fft.rfft(blk * w)) ** 2
    f = np.fft.rfftfreq(每帧, 1.0 / FS)
    return f, P


def 去辐射(波):
    """引擎的辐射是**一阶差分**（+6dB/oct）；积分近似撤掉它，用来验证「部位带是被 f² 压坏的」。"""
    y = np.cumsum(波)
    y = y - np.linspace(y[0], y[-1], len(y))     # 去线性趋势（差分引入的 DC 斜坡）
    return y


def 谱特征(f, P):
    s = P.sum()
    if s <= 0:
        return dict(重心=0.0, 峰=0.0, 展=0.0, 平坦=0.0)
    c = float((f * P).sum() / s)
    pk = float(f[int(np.argmax(P))])
    sp = float(np.sqrt(((f - c) ** 2 * P).sum() / s))
    正 = P[P > 0]
    flat = float(np.exp(np.mean(np.log(正))) / np.mean(正)) if len(正) else 0.0
    return dict(重心=c, 峰=pk, 展=sp, 平坦=flat)


def 部位化(帧, 声母, 倍频=1.0):
    """帧层改造：VOT 帧 → 噪形=0 + 噪带=(部位带→元音F2 逐帧滑移)。返回新帧 dict。"""
    新 = {k: copy.deepcopy(v) for k, v in 帧.items()}
    info = Y.声母[声母]
    带 = info.get("爆破带") or info.get("频带")
    if 带 is None:                 # 擦音（s/c）无爆破带：本探针只改塞音
        return 新, VOT帧号(帧)
    带 = 带[:2]                    # v5 口径：p/b (900,900)  t/d (3800,1800)  k/g (1900,1200)
    Fc0, BW0 = float(带[0]), float(带[1]) * float(倍频)
    idx = VOT帧号(帧)
    n = max(len(idx), 1)
    for k, fi in enumerate(idx):
        w = (k + 1.0) / n
        f2 = float(帧["f2"][fi])                       # 该帧的元音共振峰（已含声线缩放）
        Fc = Fc0 + (f2 - Fc0) * w
        BW = BW0 + (1400.0 - BW0) * w
        新["噪形"][fi] = 0
        新["噪带"][fi] = (float(Fc), float(BW))
    return 新, idx


def dprime(a, b, 去辐射后=False):
    sd = float(np.sqrt((np.var(a) + np.var(b)) / 2.0))
    return float(abs(np.mean(a) - np.mean(b)) / sd) if sd > 1e-9 else float("inf")


def 帧三类留一(样):
    """样 = [(特征4, 标签)]，按帧做留一最近质心（特征标准化）；n 足够才可读。"""
    X = np.array([s[0] for s in 样], dtype=np.float64)
    y = [s[1] for s in 样]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    类 = sorted(set(y))
    正确 = 0
    混淆 = {}
    for i in range(len(Z)):
        m = [j for j in range(len(Z)) if j != i]
        质 = {c: Z[[j for j in m if y[j] == c]].mean(0) for c in 类 if any(y[j] == c for j in m)}
        p = min(质, key=lambda c: float(np.linalg.norm(Z[i] - 质[c])))
        正确 += int(p == y[i])
        if p != y[i]:
            混淆["%s→%s" % (y[i], p)] = 混淆.get("%s→%s" % (y[i], p), 0) + 1
    return 正确 / len(Z), 混淆, len(Z)


def 跑(标, 改造):
    档 = 读档("EmoCompanion")
    行 = {}
    for 声 in 音:
        参 = H9.人类韵律_v9(档, 300, 参数={})
        波, 元, 帧 = 参.说音节(声)
        if 改造:
            帧2, idx = 部位化(帧, 声[0] if 声[:2] not in ("zh", "ch", "sh") else 声[:2])
            波, 元 = E7.合成(帧2, 音色=参.音色(), 种子=300)
        else:
            idx = VOT帧号(帧)
        波2 = 去辐射(波)
        fs = [谱特征(*谱(波, fi)) for fi in idx]
        fs2 = [谱特征(*谱(波2, fi)) for fi in idx]
        行[声] = dict(n帧=len(idx), 帧号=idx,
                      重心=[round(x["重心"], 1) for x in fs],
                      峰=[round(x["峰"], 1) for x in fs],
                      平坦=[round(x["平坦"], 4) for x in fs],
                      重心_去辐射=[round(x["重心"], 1) for x in fs2],
                      平均谱=dict(重心=round(float(np.mean([x["重心"] for x in fs])), 1) if fs else 0.0,
                              峰=round(float(np.median([x["峰"] for x in fs])), 1) if fs else 0.0,
                              展=round(float(np.mean([x["展"] for x in fs])), 1) if fs else 0.0,
                              平坦=round(float(np.mean([x["平坦"] for x in fs])), 4) if fs else 0.0,
                              重心_去辐射=round(float(np.mean([x["重心"] for x in fs2])), 1) if fs2 else 0.0))
    print("\n=== %s ===" % 标)
    print("%-4s %-4s %-8s %-8s %-8s %-7s %-10s %s"
          % ("声", "帧数", "重心Hz", "谱峰Hz", "谱展Hz", "平坦度", "去辐射重心", "逐帧重心"))
    for 声 in 音:
        d = 行[声]["平均谱"]
        print("%-4s %-4d %-8.1f %-8.1f %-8.1f %-7.4f %-10.1f %s"
              % (声, 行[声]["n帧"], d["重心"], d["峰"], d["展"], d["平坦"], d["重心_去辐射"], 行[声]["重心"]))
    return 行


def 主():
    基准 = 跑("现行 v8.1（噪形=1：VOT 由元音共振峰轨成形）", False)
    改后 = 跑("帧层部位化（噪形=0 + 噪带=部位带→元音F2，引擎零改动）", True)
    out = dict(基准=基准, 部位化=改后, 判据={})
    塞 = ("pa1", "ta1", "ka1")
    for 标, 行 in (("基准", 基准), ("部位化", 改后)):
        # 以「帧」为样本做 3 类留一（每声 5 帧 ⇒ 15 样本），比以「声」为样本可读
        acc, 混, n = 帧三类留一([((行[s]["重心"][k], 行[s]["峰"][k], 0.0, 0.0), s)
                                for s in 塞 for k in range(len(行[s]["重心"]))])
        acc2, 混2, _ = 帧三类留一([((x, 0.0, 0.0, 0.0), s) for s in 塞 for x in 行[s]["重心_去辐射"]])
        ps = [行[s]["平均谱"]["重心"] for s in 塞]
        ps2 = [行[s]["平均谱"]["重心_去辐射"] for s in 塞]
        print("\n[%s] VOT 重心（辐射后）= %.0f / %.0f / %.0f Hz ｜ 极差 %.0f ｜ 逐帧 3 类留一 %.0f%% %s"
              % (标, ps[0], ps[1], ps[2], max(ps) - min(ps), 100 * acc, 混))
        print("[%s] VOT 重心（去辐射）= %.0f / %.0f / %.0f Hz ｜ 极差 %.0f ｜ 逐帧 3 类留一 %.0f%% %s"
              % (标, ps2[0], ps2[1], ps2[2], max(ps2) - min(ps2), 100 * acc2, 混2))
        out["判据"][标] = dict(重心=ps, 极差=round(float(max(ps) - min(ps)), 1), 三类留一=round(float(acc), 3),
                              重心_去辐射=ps2, 极差_去辐射=round(float(max(ps2) - min(ps2)), 1),
                              三类留一_去辐射=round(float(acc2), 3), 混淆=混)
    for 标, 行 in (("基准", 基准), ("部位化", 改后)):
        d = {}
        for a in 塞:
            for b in 塞:
                if a < b:
                    d["%s/%s" % (a[:2], b[:2])] = [
                        round(dprime(np.array(行[a]["重心"]), np.array(行[b]["重心"])), 2),
                        round(dprime(np.array(行[a]["重心_去辐射"]), np.array(行[b]["重心_去辐射"])), 2)]
        print("[%s] d′ [辐射后, 去辐射] = %s" % (标, d))
        out["判据"][标]["d′"] = d
    # 设计带 vs 实测：噪声支路「带控」是否成立
    print("\n设计带 → 实测 VOT 重心（辐射后）：")
    for s in ("pa1", "ba1", "ta1", "da1", "ka1", "ga1"):
        带 = Y.声母[s[0]]["爆破带"]
        print("  %-4s 设计 %4.0f/%-4.0f Hz → 实测 %6.1f Hz（去辐射 %6.1f Hz）"
              % (s, 带[0], 带[1], 基准[s]["平均谱"]["重心"], 基准[s]["平均谱"]["重心_去辐射"]))
    with open(os.path.join(HERE, "试_VOT部位谱.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n已写：%s" % os.path.join(HERE, "试_VOT部位谱.json"))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    主()
