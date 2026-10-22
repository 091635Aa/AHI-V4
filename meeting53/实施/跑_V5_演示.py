# -*- coding: utf-8 -*-
"""meeting53/实施/跑_V5_演示.py —— V5 双通道原型「实况演示」：落正文流/操作流/内心流/会议流。

只读调用 V5_双通道.py；只写 产物/；不改既有件。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import V5_双通道 as V5  # noqa: E402

np.seterr(all="ignore")
产物 = os.path.join(HERE, "产物")
os.makedirs(产物, exist_ok=True)


def 写单脑(种子=V5.种子默认, 训练拍=400, 演示拍=60):
    训 = V5.跑单脑(拍数=int(训练拍), 测拍=0, 种子=int(种子))
    体 = 训["体"]
    环 = V5.接口沙盒(种子=int(种子) + 1)
    正文 = []
    操作 = []
    内心 = []
    for t in range(int(演示拍)):
        环.新试次()
        r = 体.一拍(观测id=环.cue, 学习=False, 老师=False, 拍=t, 探索率=0.0)
        成功 = 环.走(r["动作id"])
        体.更新(成功, r["动作id"], 学习动作=False)
        行 = "第%03d拍 ｜%s ｜%s ｜%s" % (t, "".join(r["词列"]), r["动作"],
                                        "成" if 成功 else "败")
        正文.append(行)
        操作.append(dict(拍=t, 原cue=int(环.cue), 原子=r["动作"],
                         目标=环.目标, 成功=int(成功)))
        内心.append(dict(拍=t, 步数=int(r["内心步数"]), 惊喜=round(float(r["惊喜"]), 4),
                         自我标量=round(float(r["自我标量"]), 4), 备注="不发布"))
    with open(os.path.join(产物, "V5_正文流.txt"), "w", encoding="utf-8") as f:
        f.write("【主正文通道 · 外部可观测】\n")
        f.write("\n".join(正文) + "\n")
    with open(os.path.join(产物, "V5_操作流.jsonl"), "w", encoding="utf-8") as f:
        for x in 操作:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(os.path.join(产物, "V5_内心流.jsonl"), "w", encoding="utf-8") as f:
        for x in 内心:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    return 正文, 操作, 内心, 体.台账


def 写三脑(种子=V5.种子默认, 训练拍=600, 演示拍=60):
    果 = V5.跑多脑(拍数=int(训练拍), 测拍=0, 种子=int(种子), 通信=True)
    # 重跑演示段，逐拍记录会议消息与联合动作
    体列, 环 = 果["体列"], V5.多脑沙盒(种子=int(种子) + 2)
    动读出 = 果["动读出"]
    记录 = []
    for t in range(int(演示拍)):
        观测列 = 环.新试次()
        o列, 词列 = [], []
        for i, 体 in enumerate(体列):
            消息 = [观测列[j] for j in range(3) if j != i]
            r = 体.一拍(观测id=观测列[i], 消息id列=消息, 学习=False, 老师=False, 拍=t)
            o列.append(体.o用.copy())
            词列.append("".join(r["词列"]))
        联合 = 体列[0].o用
        u = V5._归(联合 / (float(np.max(np.abs(联合))) + 1e-9))
        动作 = int(np.argmax(动读出 @ u))
        for 体 in 体列:
            体.上动作 = 动作
        成功 = 环.走(动作)
        for 体 in 体列:
            体.更新(成功, 动作, 学习动作=False)
        记录.append(dict(拍=t, 观测=观测列, 各脑说=词列,
                         联合动作=V5.动作表[动作], 目标=int(环.目标), 成功=int(成功)))
    with open(os.path.join(产物, "V5_会议流.jsonl"), "w", encoding="utf-8") as f:
        for x in 记录:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    return 记录


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    正文, 操作, 内心, 台账 = 写单脑()
    会议 = 写三脑()
    print("单脑演示（前 12 拍）：")
    print("\n".join(正文[:12]))
    print("……")
    print("三脑会议演示（前 6 拍）：")
    for x in 会议[:6]:
        print("  拍%03d 观测=%s 说=%s 联合动作=%s 目标=%d %s" % (
            x["拍"], x["观测"], "/".join(x["各脑说"]), x["联合动作"], x["目标"],
            "成" if x["成功"] else "败"))
    print("产物：V5_正文流.txt / V5_操作流.jsonl / V5_内心流.jsonl / V5_会议流.jsonl")
    print("台账：", json.dumps(台账, ensure_ascii=False))
