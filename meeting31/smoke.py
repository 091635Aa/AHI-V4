# -*- coding: utf-8 -*-
"""meeting31/smoke.py —— #243 冒烟：综合体核心链路快速自检（开发用，非验收）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthesis import 综合体, 隐藏世界


def main():
    桥 = 综合体(seed=310)
    # ① 用户教学（2 条事实）
    for s in ("我最喜欢的颜色是蓝色。", "我养了一只猫叫团子。"):
        桥.步(s)
    # ② 教材接入（取标准库 json 源码前 40 行）
    语料目录 = os.path.join(os.path.dirname(os.__file__), "json")
    书, 界 = [], []
    for 名 in sorted(f for f in os.listdir(语料目录) if f.endswith(".py"))[:2]:
        行s = [ln.strip() for ln in open(os.path.join(语料目录, 名),
              encoding="utf-8", errors="ignore").read().splitlines()
              if 6 <= len(ln.strip()) <= 120][:20]
        界 += [len(书) + k for k in range(len(行s))]
        书 += 行s
    桥.设教材(书, 界)
    # ③ 接入世界并断供 300 拍
    桥.设世界(隐藏世界(seed=77, 动作数=8, 噪声=0.05))
    for _ in range(300):
        桥.断供一拍()
    # ④ 自言审计
    au = 桥.自言.审计(桥.账本)
    print("自言 %d 条，类型覆盖 %s" % (len(桥.自言.流), 桥.自言.类型覆盖()))
    print("审计：真实率 %.4f 违规 %d 通过=%s" %
          (au["真实率"], au["违规数"], au["通过"]))
    if au["违规"]:
        print("  违规样例:", au["违规"][:3])
    # ⑤ 自言流查询
    print("\n—— 自言流(5) ——")
    for e in 桥.自言.自言流(5):
        print("  #%d [%s] 拍%d %s" % (e["号"], e["类型"], e["拍"], e["文本"][:40]))
    # ⑥ 知识考试（v9 口径）
    from selflearn import 知识考试
    print("\n知识考试:", 知识考试(桥, 桥.世界))
    # ⑦ 对话：行动知识 / 用户事实 / 未知 / 溯源
    for q in ("动作 3 之后世界会往哪边？", "我最喜欢的颜色是什么？",
              "法国的首都是哪里？", "你为什么这么回答？"):
        r = 桥.步(q)
        print("\n你> %s\nAHI> %s" % (q, r["响应"][:120]))
    rep = 桥.回复台账[-1]
    print("回复台账尾条支撑：自言%s 账本%s" % (rep["支撑自言"], rep["支撑账本"][:6]))
    # ⑧ 持久化双循环
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke.pkl")
    桥.保存(p)
    桥2 = 综合体.载入(p)
    相同 = (桥.自言.流 == 桥2.自言.流 and 桥.账本 == 桥2.账本
            and 桥.回复台账 == 桥2.回复台账)
    i1 = 桥._自报告("介绍一下你自己")
    i2 = 桥2._自报告("介绍一下你自己")
    print("\n持久化：自言/账本/回复台账 逐位一致=%s" % 相同)
    print("身份自报告逐位一致=%s" % (i1 == i2))
    os.remove(p)


if __name__ == "__main__":
    main()
