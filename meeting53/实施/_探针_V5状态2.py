# -*- coding: utf-8 -*-
"""meeting53 探针2：状态可分辨性 + 动作学习（只读诊断，不改既有件）。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import V5_双通道 as V5

np.seterr(all="ignore")


def 测(标签):
    体 = V5.双通道体(D=V5.D默认, 种子=V5.种子默认)
    环 = V5.接口沙盒(种子=V5.种子默认 + 1)
    状 = {c: [] for c in range(5)}
    for t in range(200):
        环.新试次()
        r = 体.一拍(观测id=环.cue, 学习=False, 老师=False, 拍=t)
        状[环.cue].append(体.o.copy())
        环.走(r["动作id"])
    # 最近质心：前 100 拍建质心，后 100 拍测
    质 = {c: np.mean(状[c][:100 // 5 - 0:], axis=0) for c in 状}
    错 = 总 = 0
    for c in 状:
        for o in 状[c][-8:]:
            分 = {k: -float(np.linalg.norm(o - v)) for k, v in 质.items()}
            预 = max(分, key=分.get)
            错 += int(预 != c)
            总 += 1
    print("[%s] cue 最近质心 acc=%.3f" % (标签, 1 - 错 / 总), end="  ")
    # 动作学习
    体2 = V5.双通道体(D=V5.D默认, 种子=V5.种子默认)
    环2 = V5.接口沙盒(种子=V5.种子默认 + 1)
    成功 = []
    for t in range(400):
        环2.新试次()
        r = 体2.一拍(观测id=环2.cue, 学习=True, 老师=True, 拍=t)
        s = 环2.走(r["动作id"])
        体2.更新(s, r["动作id"], 学习动作=True)
        成功.append(s)
    print("后50成功率=%.3f  b动=%s" % (np.mean(成功[-50:]), [round(float(x), 2) for x in 体2.b动]))


# 变体：直接改模块常量（不改文件）
测("默认")
