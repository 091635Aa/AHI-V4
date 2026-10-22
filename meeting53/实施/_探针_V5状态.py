# -*- coding: utf-8 -*-
"""meeting53 探针：只读诊断 V5 的状态区分度（不改既有件）。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import V5_双通道 as V5

np.seterr(all="ignore")
体 = V5.双通道体(D=V5.D默认, 种子=V5.种子默认)
环 = V5.接口沙盒(种子=V5.种子默认 + 1)

# 训 200 拍但不学（只看状态）
记录 = {}
for t in range(200):
    环.新试次()
    r = 体.一拍(观测id=环.cue, 学习=False, 老师=False, 拍=t)
    B = 体.o
    记录.setdefault(环.cue, []).append(B.copy())
    环.走(r["动作id"])
    #############################################
print("cue → o 均值范数 / 两两 cos：")
import itertools
means = {c: np.mean(记录[c], axis=0) for c in 记录}
for c in sorted(记录):
    print(" cue=%d n=%d |o|mean=%.3f max=%.3f" % (c, len(记录[c]),
          np.mean(np.abs(记录[c])), np.max(np.abs(记录[c]))))
print(" 两两 cos：")
for a, b in itertools.combinations(sorted(记录), 2):
    ea, eb = means[a], means[b]
    cos = float(ea @ eb / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-9))
    print("  %d-%d cos=%.3f" % (a, b, cos))

# 再训 300 拍看动作学习
体2 = V5.双通道体(D=V5.D默认, 种子=V5.种子默认)
环2 = V5.接口沙盒(种子=V5.种子默认 + 1)
成功 = []
for t in range(300):
    环2.新试次()
    r = 体2.一拍(观测id=环2.cue, 学习=True, 老师=True, 拍=t)
    s = 环2.走(r["动作id"])
    体2.更新(s, r["动作id"], 学习动作=True)
    成功.append(s)
print("训练成功率(后50)=%.3f" % np.mean(成功[-50:]))
print("W动 行范数:", [round(float(np.linalg.norm(w)), 3) for w in 体2.W动])
print("b动:", [round(float(x), 3) for x in 体2.b动])
print("W词 行范数(前10):", [round(float(np.linalg.norm(w)), 3) for w in 体2.W词[:10]])
