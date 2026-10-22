# -*- coding: utf-8 -*-
"""meeting31/视觉测试.py —— #289 视觉保真度测试（可判读）：他能看清吗？

所有者测试设计：程序生成已知真值的复杂图形——一堆圆形中混入不同颜色、不同深浅、
不同锐度，以及方形/三角形等不同图形混合——问『有多少个圆』。
真值已知 → 他的视觉神经检测结果 vs 真值 = 可判读（计数误差/颜色正确率）。

感知管线（传统 CV，无神经网络）：
  圆=霍夫圆检测（纯几何）；方/三角=Canny+轮廓+多边形逼近（顶点数）；
  颜色=HSV 主色；深浅=V 亮度；锐度=Laplacian 方差。
描述注入他的账本/codec（同一向量空间）→ 让他回答 → 对真值判读。
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

from synthesis import 综合体

图路径 = os.path.join(HERE, "视觉测试_画面.png")
结果路径 = os.path.join(HERE, "视觉测试_结果.json")

颜色名s = {"红": 0, "橙": 20, "黄": 35, "绿": 60, "青": 90, "蓝": 120, "紫": 150}


def 最近色名(h_opencv: float) -> str:
    h = float(h_opencv) * 2          # OpenCV H∈[0,179] → 度数
    return min(颜色名s, key=lambda n: min(abs(h - 颜色名s[n]),
                                          360 - abs(h - 颜色名s[n])))


def 生成画面(seed=42, 宽=960, 高=720):
    """确定性生成混合图形：圆(主)+方+三角，颜色/深浅/锐度各异。真值返回。"""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (宽, 高), (245, 242, 235))    # 米白底
    d = ImageDraw.Draw(img)
    真值 = []
    n圆, n方, n三角 = 8, 3, 2
    色相s = [0, 25, 60, 120, 210, 260]
    for i in range(n圆):
        r = int(rng.integers(30, 75))
        x = int(rng.integers(r + 10, 宽 - r - 10))
        y = int(rng.integers(r + 10, 高 - r - 10))
        h = 色相s[int(rng.integers(0, len(色相s)))]
        深 = float(rng.uniform(0.35, 1.0))               # 深浅
        锐 = float(rng.uniform(0, 4))                    # 模糊半径 0=最锐
        色 = tuple(int(c * 深) for c in
                   (255 if h < 20 or h > 200 else 0, 0, 0)) if False else \
            tuple(int(v) for v in _hsv_rgb(h, 0.75, 深))
        层 = Image.new("RGB", (宽, 高), (0, 0, 0))
        dl = ImageDraw.Draw(层)
        dl.ellipse([x - r, y - r, x + r, y + r], fill=色)
        if 锐 > 0.3:
            层 = 层.filter(_模糊(锐))
        img = _叠底(img, 层, [x - r, y - r, x + r, y + r], 色)
        真值.append(dict(形状="圆", 色=_hsv_rgb(h, 0.75, 深), 深=round(深, 2),
                         锐=round(锐, 2), 中心=(x, y), r=r))
    for i in range(n方):
        s = int(rng.integers(45, 90))
        x = int(rng.integers(s + 10, 宽 - s - 10))
        y = int(rng.integers(s + 10, 高 - s - 10))
        h = 色相s[int(rng.integers(0, len(色相s)))]
        深 = float(rng.uniform(0.4, 0.95))
        色 = _hsv_rgb(h, 0.7, 深)
        d = ImageDraw.Draw(img)                  # 圆的叠底换了 img，画笔必须重绑
        d.rectangle([x - s // 2, y - s // 2, x + s // 2, y + s // 2], fill=色)
        真值.append(dict(形状="方", 色=色, 深=round(深, 2), 中心=(x, y)))
    for i in range(n三角):
        s = int(rng.integers(50, 95))
        x = int(rng.integers(s + 10, 宽 - s - 10))
        y = int(rng.integers(s + 10, 高 - s - 10))
        h = 色相s[int(rng.integers(0, len(色相s)))]
        深 = float(rng.uniform(0.4, 0.95))
        色 = _hsv_rgb(h, 0.7, 深)
        d = ImageDraw.Draw(img)
        d.polygon([(x, y - s // 2), (x - s // 2, y + s // 2),
                   (x + s // 2, y + s // 2)], fill=色)
        真值.append(dict(形状="三角", 色=色, 深=round(深, 2), 中心=(x, y)))
    img.save(图路径)
    return img, 真值


def _hsv_rgb(h, s, v):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360.0, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def _模糊(半径):
    from PIL import ImageFilter
    return ImageFilter.GaussianBlur(半径)


def _叠底(img, 层, 盒, 色):
    """把模糊后的圆形区域贴回底图（其余透明处理：按色距混合）。"""
    a = np.asarray(img, dtype=np.float32)
    l = np.asarray(层, dtype=np.float32)
    非黑 = (l.sum(axis=2) > 30)[:, :, None]
    混 = np.where(非黑, l, a)
    return Image.fromarray(混.astype(np.uint8))


def 感知(路径):
    """传统 CV 感知管线：数圆/辨形/颜色/深浅/锐度。返回描述 dict。"""
    img = cv2.imdecode(np.fromfile(路径, dtype=np.uint8),
                       cv2.IMREAD_COLOR)          # 中文路径必须 imdecode
    灰 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 圆：霍夫（纯几何）——param2 放宽：模糊圆边缘弱
    圆s = cv2.HoughCircles(灰, cv2.HOUGH_GRADIENT, dp=1.2,
                           minDist=40, param1=100, param2=30,
                           minRadius=25, maxRadius=95)
    圆s = [] if 圆s is None else np.round(圆s[0]).astype(int)
    # 方/三角：背景色距分割→连通域→圆度判形（4πA/P²：圆≈1.0 方≈0.79 三角≈0.6）
    前景 = (np.abs(img.astype(np.int32) - np.array([235, 242, 245])
                   ).sum(axis=2) > 60).astype(np.uint8)
    前景 = cv2.morphologyEx(前景, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    连通 = cv2.connectedComponentsWithStats(前景, connectivity=8)
    形s = []
    for i in range(1, 连通[0]):
        x, y, w, h, 面积 = 连通[2][i]
        if 面积 < 1500 or 面积 > 60000:
            continue
        域 = (连通[1][y:y + h, x:x + w] == i).astype(np.uint8)
        周 = cv2.arcLength(cv2.findContours(
            域, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0], True)
        圆度 = 4 * np.pi * 面积 / (周 * 周 + 1e-9)
        cx, cy = x + w // 2, y + h // 2
        if any((cx - ux) ** 2 + (cy - uy) ** 2 < (ur * 0.7) ** 2
               for ux, uy, ur in 圆s.tolist()):
            continue                       # 圆心落在已检圆内：跳过
        hsvv = hsv[cy, cx]
        色名 = 最近色名(float(hsvv[0]))
        if 圆度 > 0.85:
            continue                       # 圆形（已由霍夫统计）
        形 = "方" if 圆度 > 0.68 else "三角"
        形s.append(dict(形状=形, 色=色名, 深=round(float(hsvv[2]) / 255, 2),
                        中心=(cx, cy), 圆度=round(圆度, 3)))
    圆详 = []
    for ux, uy, ur in 圆s.tolist():
        hsvv = hsv[uy, ux]
        锐区 = 灰[max(0, uy - ur):uy + ur, max(0, ux - ur):ux + ur]
        锐度 = cv2.Laplacian(锐区, cv2.CV_64F).var()
        圆详.append(dict(中心=(int(ux), int(uy)), r=int(ur),
                         色=最近色名(float(hsvv[0])),
                         深=round(float(hsvv[2]) / 255, 2),
                         锐度=round(float(锐度), 1)))
    return dict(圆数=len(圆详), 圆s=圆详, 其他=形s)


def 主():
    img, 真值 = 生成画面()
    print("[生成] 画面已生成：真值 圆 %d 个（颜色/深浅/锐度各异）+ 方 %d + 三角 %d"
          % (sum(1 for t in 真值 if t["形状"] == "圆"),
             sum(1 for t in 真值 if t["形状"] == "方"),
             sum(1 for t in 真值 if t["形状"] == "三角")))
    感 = 感知(图路径)
    print("[感知] 他看到：圆 %d 个、方 %d 个、三角 %d 个"
          % (感["圆数"], sum(1 for x in 感["其他"] if x["形状"] == "方"),
             sum(1 for x in 感["其他"] if x["形状"] == "三角")))
    # 详细描述（他的视觉报告）
    描述 = "画面里有 %d 个圆。" % 感["圆数"]
    颜色计: dict = {}
    for c in 感["圆s"]:
        颜色计[c["色"]] = 颜色计.get(c["色"], 0) + 1
    描述 += "颜色：" + "、".join("%s%d 个" % (k, v)
                              for k, v in sorted(颜色计.items(),
                                                 key=lambda kv: -kv[1])) + "。"
    最锐 = max(感["圆s"], key=lambda c: c["锐度"]) if 感["圆s"] else None
    最浅 = min(感["圆s"], key=lambda c: c["深"]) if 感["圆s"] else None
    if 最锐:
        描述 += "最锐利的圆在%s（锐度 %.0f）。" % (最锐["中心"], 最锐["锐度"])
    if 最浅:
        描述 += "最浅的圆深度 %.2f。" % 最浅["深"]
    描述 += "另有 %s。" % ("、".join("%s %d 个" % (x["形状"], 1)
                                for x in 感["其他"]) or "无其他图形")
    print("[描述] %s" % 描述)
    # 让他回答：描述入账+codec，然后问
    体 = 综合体(seed=777)
    体.设开放域([], [])
    体.挂舱("感官")
    体.codec.注册段(描述)
    体.账本.append(("检验", 描述))
    体.舱s["感官"].存行(描述)
    体.步("画面里有多少个圆？")
    n = 0
    while 体.思考中 is not None and n < 40:
        体.断供一拍()
        n += 1
    答 = str(体.回复台账[-1]["响应"]) if 体.回复台账 else "（无回答）"
    print("[他的回答] %s" % 答[:200])
    # 判读
    真圆 = sum(1 for t in 真值 if t["形状"] == "圆")
    检出 = 感["圆数"]
    误差 = abs(检出 - 真圆)
    判 = "PASS" if 误差 == 0 else ("PASS(±1)" if 误差 == 1 else "FAIL")
    print("[判读] 真值 圆=%d ｜ 检出=%d ｜ 误差=%d → %s" % (真圆, 检出, 误差, 判))
    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump(dict(真值=真值, 感知=感, 描述=描述, 回答=答[:300],
                       真圆数=真圆, 检出圆数=检出, 判读=判),
                  f, ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    主()
