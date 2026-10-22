# -*- coding: utf-8 -*-
"""_诊断_自模型.py —— 临时诊断（步2 调试；不入交付，用后可删）。
问：为什么 1024 维富态 ĝ 在真机上学不动？分解到（键构造 / 读出 / 可达性）。
"""
from __future__ import annotations
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # AHI3 根（迁移：随文件定位；HERE=meeting39/实施）
for _p in (HERE, os.path.join(ROOT, "meeting31"), os.path.join(ROOT, "meeting38", "实施"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from 内核_自模型 import 顶覆盖_综合体

句s = ["今天的风很轻。", "我把桌子擦干净了。", "窗外有一棵树。", "水杯放在手边。",
      "我慢慢翻了一页书。", "灯一直亮着。", "他坐在对面没说话。", "雨点敲着玻璃。"]

def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(a @ b / (na * nb))

臂 = 顶覆盖_综合体(seed=311, 自模型开关=True, 学习=True)
# 手动收集 S_t（sign(x)）与动作向量
Ss, As, Us, Hs = [], [], [], []
for s in (句s * 22):
    u = 臂.感知向量(臂.codec.编码(s))
    rec = 臂._引擎步(u)
    h = rec["模块"]["hdc"]
    Ss.append(np.sign(np.asarray(h["x"], dtype=np.float64)))
    Hs.append(np.sign(np.asarray(h["H双极"], dtype=np.float64)))     # 引擎带记忆的内部态
    As.append(臂.eng.hdc._动作向量(np.asarray(rec["a星"], dtype=np.float32)))
    Us.append(np.asarray(u, dtype=np.float64))
Ss = np.array(Ss); Hs = np.array(Hs)

print("步数=%d 状态维=%d" % (len(Ss), Ss.shape[1]))
# 1) 输入编码 x 的结构：唯一度 / 相邻 cos
uniq = len({tuple(np.sign(Ss[i])[:8].tolist()) for i in range(len(Ss))})
adj = np.mean([cos(Ss[i], Ss[i+1]) for i in range(len(Ss)-1)])
print("S_t 相邻cos均值=%.4f（自相关，高=可预测）" % adj)
# 每个"句位"的 S 是否稳定（同一句是否总给出同一 S）
句S = {}
for i, S in enumerate(Ss):
    j = i % 8
    句S.setdefault(j, []).append(S)
print("同句 S 一致度：", {j: round(float(np.mean([cos(v[0], v[k]) for k in range(1, len(v))])), 3)
                        for j, v in 句S.items()})
# 2) 可达性：离线最小二乘（teacher-forced）能否拟合 k_t -> S_{t+1}
def 可达(键, 目标):
    K = np.array(键); T = np.array(目标)
    W, *_ = np.linalg.lstsq(K, T, rcond=None)     # (D,D)
    P = np.sign(K @ W)
    acc = float(np.mean(P == T))
    e = 1.0 - np.mean([cos(T[i], P[i]) for i in range(len(T))])
    return acc, e
kS = [Ss[i] for i in range(len(Ss)-1)]
kSa = [Ss[i] * As[i] for i in range(len(Ss)-1)]
kH = [Hs[i] for i in range(len(Hs)-1)]
kHa = [Hs[i] * As[i] for i in range(len(Hs)-1)]
for 名, 键, 目标 in (("x_t→S_{t+1}", kS, Ss[1:]),
                   ("S⊛a→S_{t+1}", kSa, Ss[1:]),
                   ("H_t→H_{t+1}", kH, Hs[1:]),
                   ("H⊛a→H_{t+1}", kHa, Hs[1:]),
                   ("S_t→H_{t+1}", kS, Hs[1:])):
    a, e = 可达(键, 目标)
    print("离线LS %-14s 逐维命中=%.3f  ε=%.4f" % (名, a, e))
# 3) 在线 NLMS 实际 W 的量级
W = 臂.自模型.W
k = Ss[-2] * As[-2]
print("在线W: ||W||=%.3f D=%d  ||W@k||=%.3f  sign命中率=%.3f" %
      (np.linalg.norm(W), W.shape[0], np.linalg.norm(W @ k),
       float(np.mean(np.sign(W @ k) == Ss[-1]))))
ε = 臂.自模型.ε序列()
print("在线 ε：首20=%.4f 末20=%.4f" % (np.mean(ε[:20]), np.mean(ε[-20:])))