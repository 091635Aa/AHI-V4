# -*- coding: utf-8 -*-
"""meeting31/视觉测试2.py —— #290 复杂图形视觉测试：半月/折线/多边形/星形。

所有者指令：测试更复杂的——半月、不同折线、多角多边形，让他描述。
真值已知（确定性生成）→ 感知 vs 真值 = 可判读。
感知管线（传统 CV 无 NN）：
  折线=细长比（外接框长宽比+填充率）；星形=凹性（凸包面积/面积比大）+顶点≈10；
  半月=面积≈半圆（拟合圆）+含直弦边；五边/六边=approxPolyDP 顶点数；
  圆=霍夫+圆度>0.85。
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")

import cv2
import numpy as np
from PIL import Image, ImageDraw

图路径 = os.path.join(HERE, "视觉测试2_画面.png")
结果路径 = os.path.join(HERE, "视觉测试2_结果.json")
背景 = (245, 242, 235)


def 生成画面(seed=7, 宽=960, 高=720):
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (宽, 高), 背景)
    d = ImageDraw.Draw(img)
    真值 = []
    位置s = []

    def 占位(x, y, w, h):
        for px, py, pw, ph in 位置s:
            if abs(x - px) < (pw + w) / 2 + 15 and abs(y - py) < (ph + h) / 2 + 15:
                return False
        位置s.append((x, y, w, h))
        return True

    def 放(w, h):
        for _ in range(200):
            x, y = int(rng.integers(w // 2 + 15, 宽 - w // 2 - 15)), \
                   int(rng.integers(h // 2 + 15, 高 - h // 2 - 15))
            if 占位(x, y, w, h):
                return x, y
        return None, None

    def 画(形状, n):
        for i in range(n):
            色 = tuple(int(v) for v in np.random.default_rng(seed * 100 + i)
                       .integers(40, 200, 3))
            if 形状 == "半月":
                r = int(rng.integers(55, 85))
                x, y = 放(r * 2, r)
                if x is None:
                    continue
                d.pieslice([x - r, y - r, x + r, y + r], 180, 360, fill=色)
                d.rectangle([x - r, y - r - 2, x + r, y + r + 2], fill=背景)
                真值.append(dict(形状="半月", 中心=(x, y)))
            elif 形状 == "折线":
                w2, h2 = 200, 110
                x, y = 放(w2, h2)
                if x is None:
                    continue
                点s = [(x - w2 // 2, y + h2 // 2), (x - w2 // 6, y - h2 // 2),
                       (x, y + h2 // 4), (x + w2 // 6, y - h2 // 3),
                       (x + w2 // 2, y + h2 // 3)]
                d.line(点s, fill=色, width=14, joint="curve")
                真值.append(dict(形状="折线", 中心=(x, y)))
            elif 形状 == "五边形":
                r = int(rng.integers(48, 75))
                x, y = 放(r * 2, r * 2)
                if x is None:
                    continue
                d.polygon([(x + r * np.sin(a), y - r * np.cos(a))
                           for a in np.linspace(0, 2 * np.pi, 5, endpoint=False)],
                          fill=色)
                真值.append(dict(形状="五边形", 中心=(x, y)))
            elif 形状 == "六边形":
                r = int(rng.integers(48, 75))
                x, y = 放(r * 2, r * 2)
                if x is None:
                    continue
                d.polygon([(x + r * np.sin(a), y - r * np.cos(a))
                           for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)],
                          fill=色)
                真值.append(dict(形状="六边形", 中心=(x, y)))
            elif 形状 == "星形":
                r, r2 = 80, 34
                x, y = 放(r * 2, r * 2)
                if x is None:
                    continue
                点s = []
                for k in range(10):
                    rr = r if k % 2 == 0 else r2
                    a = k * np.pi / 5 - np.pi / 2
                    点s.append((x + rr * np.sin(a), y - rr * np.cos(a)))
                d.polygon(点s, fill=色)
                真值.append(dict(形状="星形", 中心=(x, y)))
            elif 形状 == "圆":
                r = int(rng.integers(45, 70))
                x, y = 放(r * 2, r * 2)
                if x is None:
                    continue
                d.ellipse([x - r, y - r, x + r, y + r], fill=色)
                真值.append(dict(形状="圆", 中心=(x, y)))

    for 形, n in (("圆", 3), ("半月", 2), ("折线", 2), ("五边形", 2),
                  ("六边形", 1), ("星形", 1)):
        画(形, n)
    img.save(图路径)
    return 真值


def 感知(路径):
    img = cv2.imdecode(np.fromfile(路径, dtype=np.uint8), cv2.IMREAD_COLOR)
    灰 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    圆s = cv2.HoughCircles(灰, cv2.HOUGH_GRADIENT, dp=1.2, minDist=60,
                           param1=100, param2=32, minRadius=35, maxRadius=95)
    圆s = [] if 圆s is None else np.round(圆s[0]).astype(int)
    前景 = (np.abs(img.astype(np.int32) - np.array(背景[::-1])
                   ).sum(axis=2) > 60).astype(np.uint8)
    前景 = cv2.morphologyEx(前景, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    连通 = cv2.connectedComponentsWithStats(前景, connectivity=8)
    检出 = []
    for i in range(1, 连通[0]):
        x, y, w, h, 面积 = 连通[2][i]
        if 面积 < 1500 or 面积 > 60000:
            continue
        cx, cy = x + w // 2, y + h // 2
        长宽比 = max(w, h) / max(1, min(w, h))
        填充 = 面积 / (w * h + 1e-9)
        # 几何特征（实测校准）：圆度/凹性/紧逼近顶点数
        域mask = (连通[1][y:y + h, x:x + w] == i).astype(np.uint8)
        cs, _ = cv2.findContours(域mask, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_NONE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        凸包面积 = cv2.contourArea(cv2.convexHull(c))
        凹比 = 凸包面积 / (面积 + 1e-9)
        周 = cv2.arcLength(c, True)
        圆度 = 4 * np.pi * 面积 / (周 * 周 + 1e-9)
        紧顶点 = cv2.approxPolyDP(c, 0.01 * 周, True).shape[0]
        内切 = float(cv2.distanceTransform(域mask, cv2.DIST_L2, 5).max())
        if 长宽比 > 1.5 and 填充 < 0.32 and 内切 < 15:
            形 = "折线"                     # 细笔画（线宽≈14px）
        elif 霍夫中(cx, cy, 圆s) or (圆度 > 0.88 and 凹比 < 1.05
                                     and 填充 > 0.75 and 面积 > 5000):
            形 = "圆"
        elif 凹比 > 2.4 and 紧顶点 >= 9:
            形 = "星形"                     # 深凹+多尖角
        elif 1.25 < 凹比 <= 2.4 and 紧顶点 >= 8 and 0.36 <= 填充 <= 0.56:
            形 = "半月"                     # 浅凹+近半圆填充
        elif 凹比 < 1.15 and 紧顶点 == 5:
            形 = "五边形"
        elif 凹比 < 1.15 and 紧顶点 == 6:
            形 = "六边形"
        else:
            形 = "未知"
        检出.append(dict(形状=形, 中心=(cx, cy), 圆度=round(圆度, 3),
                         凹比=round(凹比, 3), 紧顶点=紧顶点, 填充=round(填充, 2)))
    return 检出


def 霍夫中(cx, cy, 圆s) -> bool:
    return any((cx - ux) ** 2 + (cy - uy) ** 2 < (ur * 0.75) ** 2
               for ux, uy, ur in 圆s.tolist())


def 主():
    真值 = 生成画面()
    真值计: dict = {}
    for t in 真值:
        真值计[t["形状"]] = 真值计.get(t["形状"], 0) + 1
    print("[生成] 真值：" + "、".join("%s%d" % (k, v)
                                   for k, v in sorted(真值计.items())))
    检出 = 感知(图路径)
    检出计: dict = {}
    for x in 检出:
        检出计[x["形状"]] = 检出计.get(x["形状"], 0) + 1
    print("[感知] 检出：" + "、".join("%s%d" % (k, v)
                                   for k, v in sorted(检出计.items())))
    # 判读（按形状对齐）
    全对, 全真 = 0, 0
    for 形 in sorted(真值计):
        t, c = 真值计.get(形, 0), 检出计.get(形, 0)
        对 = "✓" if t == c else "✗"
        print("  %s：真值 %d 检出 %d %s" % (形, t, c, 对))
        全对 += min(t, c)
        全真 += t
    总检出 = len(检出)
    print("[判读] 形状召回 %d/%d（%.0f%%）；检出总数 %d（含误检 %d）"
          % (全对, 全真, 100.0 * 全对 / max(全真, 1), 总检出, 总检出 - 全对))
    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump(dict(真值计=真值计, 检出计=检出计, 检出=检出,
                       召回=全对, 真值总数=全真),
                  f, ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    主()
