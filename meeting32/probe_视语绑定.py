# -*- coding: utf-8 -*-
"""meeting32/probe_视语绑定.py —— 视觉词↔语言共现绑定小验证（判据先定）。

判据（先定）：
  B1 位置不变：3 个学习场景的物体挪位重画 → 看图说文 top-1 全中 3/3
  B2 内容区分：留置新场景（没学过的物体组合）top-1 相似度 < 训练命中相似度
  B3 反向检索：闻文找景 ≥ 2/3
  B4 确定性：同场景两次编码场景向量逐位一致
  B5 词典活性：视觉词典规模 > 10（真在用词典）
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "meeting31"))
sys.path.insert(0, os.path.join(HERE, "..", "meeting28"))

from 绑定 import 视语舱, 视  # noqa: E402

结果 = []


def 判(名, 条件, 明细=""):
    结果.append((名, bool(条件), 明细))
    print("[%s] %s %s" % ("PASS" if 条件 else "FAIL", 名, 明细), flush=True)


def 新图():
    return Image.new("RGB", (256, 192), (120, 120, 120))


def 场景1(偏移=(0, 0)):
    """红圆 + 蓝方。"""
    img = 新图(); d = ImageDraw.Draw(img)
    d.ellipse([30 + 偏移[0], 40 + 偏移[1], 110 + 偏移[0], 120 + 偏移[1]],
              fill=(220, 40, 40))
    d.rectangle([150 + 偏移[0], 90 + 偏移[1], 230 + 偏移[0], 170 + 偏移[1]],
                fill=(40, 60, 220))
    return img


def 场景2(偏移=(0, 0)):
    """绿三角 + 黄星。"""
    img = 新图(); d = ImageDraw.Draw(img)
    d.polygon([(60 + 偏移[0], 30 + 偏移[1]), (20 + 偏移[0], 130 + 偏移[1]),
               (100 + 偏移[0], 130 + 偏移[1])], fill=(40, 200, 60))
    星 = [(190 + 偏移[0] + 40 * np.cos(a - np.pi / 2),
           100 + 偏移[1] + 40 * np.sin(a - np.pi / 2))
          for a in np.linspace(0, 2 * np.pi, 10, endpoint=False)]
    点 = [(x, y) for i, (x, y) in enumerate(星)
          if i % 2 == 0] + [(x * 0.45 + 190 + 偏移[0] * 0.55,
                             y * 0.45 + 100 + 偏移[1] * 0.55)
                            for i, (x, y) in enumerate(星) if i % 2 == 1]
    序 = []
    for i in range(5):
        序.append(星[2 * i % 10]); 序.append(星[(2 * i + 1) % 10])
    d.polygon(序, fill=(240, 210, 40))
    return img


def 场景3(偏移=(0, 0)):
    """两个红方。"""
    img = 新图(); d = ImageDraw.Draw(img)
    d.rectangle([30 + 偏移[0], 50 + 偏移[1], 100 + 偏移[0], 120 + 偏移[1]],
                fill=(220, 40, 40))
    d.rectangle([150 + 偏移[0], 80 + 偏移[1], 220 + 偏移[0], 150 + 偏移[1]],
                fill=(220, 40, 40))
    return img


def 留置景():
    """紫圆（没学过的组合）。"""
    img = 新图(); d = ImageDraw.Draw(img)
    d.ellipse([80, 50, 180, 150], fill=(150, 40, 200))
    return img


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    视._原型词典.clear(); 视._使用次数.clear(); 视._生长史.clear()
    舱 = 视语舱()

    学习 = [(场景1(), "画面里有一个红色的圆和一个蓝色的方。"),
           (场景2(), "画面里有一个绿色的三角和一个黄色的星。"),
           (场景3(), "画面里有两个红色的方块。")]
    for img, 文 in 学习:
        舱.learn(img, 文)
    判("B5 词典活性", 视.词典规模() > 10, "视觉词典 %d 词" % 视.词典规模())

    # B4 确定性
    a = 舱.场景向量(场景1()); b = 舱.场景向量(场景1())
    判("B4 确定性", np.array_equal(a, b), "同场景两次编码逐位一致")

    # B1 位置不变（挪位重画）
    查询 = [(场景1((40, -20)), 学习[0][1]), (场景2((-30, 25)), 学习[1][1]),
            (场景3((20, 30)), 学习[2][1])]
    命中 = 0
    for img, 正确 in 查询:
        检 = 舱.看图说文(img, k=1)[0]
        命中 += int(检["文本"] == 正确)
        print("  查询→『%s』相似度=%.3f" % (检["文本"][:18], 检["相似度"]), flush=True)
    判("B1 位置不变", 命中 == 3, "挪位场景 3/3 命中（词袋级绑定天性）")

    # B2 内容区分（留置场景 top-1 相似度 < 训练命中）
    留检 = 舱.看图说文(留置景(), k=1)[0]
    训相似 = min(舱.看图说文(img, k=1)[0]["相似度"] for img, _ in 查询)
    判("B2 内容区分", 留检["相似度"] < 训相似,
       "留置 %.3f < 训练命中最低 %.3f（留置被认成『%s』）"
       % (留检["相似度"], 训相似, 留检["文本"][:16]))

    # B3 反向检索（闻文找景）
    对 = 0
    for img, 文 in 学习:
        检 = 舱.闻文找景(文, k=1)[0]
        对 += int(检["文本"] == 文)
    判("B3 反向检索", 对 >= 2, "%d/3" % 对)

    fail = [n for n, ok, _ in 结果 if not ok]
    print("\n===== 视语绑定小验证：%d/%d PASS =====" % (len(结果) - len(fail),
                                                        len(结果)))
    if fail:
        print("FAIL:", fail)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
