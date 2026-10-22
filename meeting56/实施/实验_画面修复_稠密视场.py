# -*- coding: utf-8 -*-
"""实验_画面修复_稠密视场.py —— meeting56 修复验证：旧「稀疏件画面」vs 新「保拓扑稠密视场画面」。

【为什么有这个实验】所有者看 meeting56 的回放屏/想象屏后指出「画面有问题」。诊断（R0 臂）：
  旧 内核_画面 的 16 件里 **11 件落在纯背景格**（结构度≈噪声），且件特征只是中心 4×4 px 色块，
  **方块/横条/竖条只要中心同色就是同一个码**（形状不可分）；成像又把归一化结构均值映射成假色。
  本脚本把这三条做成可复现的数字，并验证新 [`画面_稠密场`](画面_稠密场.py) 修复：
  保拓扑稠密缓冲 + 外周 8×8 / 中央凹 3×3 为 16×16 + 缓冲直读成像 + 只搬件、背景留原样。

跑法：python 实验_画面修复_稠密视场.py   （合成屏、固定种子、无网络；两趟逐位同）
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))                 # AHI4/meeting56/实施
AHI4 = os.path.dirname(os.path.dirname(HERE))                    # AHI4
M39 = os.path.join(AHI4, "meeting39", "实施")
for p in (HERE, M39):
    if p not in sys.path:
        sys.path.insert(0, p)

from 画面_稠密场 import (稠密画面, 结构阈, 结构度s, 形状描述,     # noqa: E402
                        _缩放, _写png)
from 内核_画面 import 画面核                                      # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

产物 = os.path.join(os.path.dirname(HERE), "产物")
种子_屏 = 7
格x, 格y = 16, 9


def 造屏(种子: int = 种子_屏, H: int = 144, W: int = 256) -> np.ndarray:
    """7 件分离色块（与 meeting56 既有实验同源；确定性）。"""
    rng = np.random.default_rng(种子)
    图 = np.full((H, W, 3), 40, np.float32)
    件 = [(16, 16, 32, 32, (230, 60, 60)), (80, 16, 32, 32, (60, 200, 90)),
          (144, 16, 32, 32, (80, 120, 240)), (208, 16, 32, 32, (240, 220, 40)),
          (16, 96, 64, 16, (250, 250, 250)), (112, 96, 32, 48, (20, 20, 20)),
          (192, 96, 48, 32, (150, 90, 200))]
    for x, y, w, h, c in 件:
        图[y:y + h, x:x + w] = np.asarray(c, np.float32)
    return 图 + rng.normal(0, 3, 图.shape).astype(np.float32)


形状框 = {"方块": (120, 56, 32, 32), "横条": (112, 66, 48, 12),
          "竖条": (130, 48, 12, 48), "圆": (120, 56, 32, 32)}


def 造形状屏(形: str, H: int = 144, W: int = 256, 种子: int = 90) -> np.ndarray:
    """同色（210,70,70）的形状，中心都在格 (8,4) 的中点 (136,72)：
       方块 32×32 / 横条 48×12 / 竖条 12×48 / 圆 32×32（与方块同 bbox）。
       方块的"中心 4×4"与横条/竖条/圆的中心 4×4 完全相同 ⇒ 专测"形状信息在不在码里"。"""
    rng = np.random.default_rng(种子)
    图 = np.full((H, W, 3), 40, np.float32)
    x0, y0, w, h = 形状框[形]
    if 形 == "圆":
        Y, X = np.ogrid[0:H, 0:W]
        cx, cy, r = x0 + w // 2, y0 + h // 2, w // 2
        掩 = (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
        图[掩] = np.asarray((210, 70, 70), np.float32)
    else:
        图[y0:y0 + h, x0:x0 + w] = np.asarray((210, 70, 70), np.float32)
    return 图 + rng.normal(0, 3, 图.shape).astype(np.float32)


def 形状标签_宽高比(形: str) -> str:
    """复刻 器官_眼 元素s.形状 的现有口径（只按 bbox 宽高比打标签）：宽/高 ≥1.6 → 横条，
    高/宽 ≥1.6 → 竖条，否则 → 方块。**它看不到轮廓**（圆与方块同 bbox ⇒ 同为"方块"）。"""
    x0, y0, w, h = 形状框[形]
    if w / h >= 1.6:
        return "横条"
    if h / w >= 1.6:
        return "竖条"
    return "方块"


def 图cos(a, b) -> float:
    x = np.asarray(a, np.float64).ravel()
    y = np.asarray(b, np.float64).ravel()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def 旧画面(帧) -> tuple:
    """旧路径：内核_画面 稀疏件 + 画格（一格一色），放大 16 ⇒ 256×144，便于和真值同尺寸比。"""
    核 = 画面核(D=1024, 种子=3800)
    记 = 核.记一屏(帧, 拍=1)
    if not 记.get("记"):
        return None, dict(件数=0, 件s=[])
    位s = [tuple(x) for x in 核.库[-1]["位s"]]
    r = 核.读出(核.库[-1]["场s"], 码库=核.库[-1]["码s"], 位置s=位s,
                 外观库=核.库[-1]["外观s"])
    格 = 核.画格(位s, r["重建"])
    图 = np.repeat(np.repeat(格, 16, 0), 16, 1)
    return 图, dict(件数=len(位s), 件s=位s)


def 旧件背景数(帧, 件s) -> dict:
    """旧件里有多少落在"无结构格"（用新件的同一判据：块内结构+背景色差 < 结构阈）。"""
    照 = 稠密画面()
    块s = 照.取格s(帧, 注视=(8, 4))
    度 = 结构度s(块s)
    无结构 = [k for k in 件s if 度.get(k, 0.0) < 结构阈]
    return dict(件数=len(件s), 无结构格件数=len(无结构), 无结构格件=无结构)


def main():
    报 = {"设置": dict(屏="合成屏 144×256（固定种子 7）", 网格=(格x, 格y),
                       外周块=8, 中央凹="3×3 格 × 16×16", 结构阈=结构阈,
                       对齐="会议_视觉 §0 保拓扑视场 + §3 中央凹/外周 + D3 原始像素块")}
    os.makedirs(产物, exist_ok=True)

    # ── R0 旧画面：背景虚构 + 与真值差 ──
    帧 = 造屏()
    旧图, 旧元 = 旧画面(帧)
    真图 = _缩放(帧[:, :, :3], 格x * 16, 格y * 16).astype(np.uint8)
    _写png(真图, os.path.join(产物, "修复_真值.png"))
    _写png(旧图, os.path.join(产物, "修复_旧画面.png"))
    旧PSNR = float(10.0 * np.log10(255.0 ** 2 / max(
        float(np.mean((真图.astype(np.float64) - 旧图.astype(np.float64)) ** 2)), 1e-12)))
    旧MAE = float(np.mean(np.abs(真图.astype(np.float64) - 旧图.astype(np.float64))))
    R0 = dict(旧件数=旧元["件数"], **旧件背景数(帧, 旧元["件s"]),
              旧画面PSNR=round(旧PSNR, 2), 旧画面MAE=round(旧MAE, 2),
              说明="旧路径只画 16 件、其余格暗底；件含纯背景噪声格 ⇒ 画面散块/假色")

    # ── R1 新画面：稠密缓冲回放 + 忠实度 ──
    照 = 稠密画面()
    记 = 照.记一屏(帧, 拍=1, 注视=(8, 4))
    照.回放(路径=os.path.join(产物, "修复_新画面.png"), 放大=1)
    比 = 照.比对(0, 帧, 放大=1)
    背件 = [k for k in 记["件s"] if 记["屏"]["结构s"][k] < 结构阈]
    R1 = dict(结构格数=记["结构格数"], 无结构格数=记["无结构格数"],
              件s=记["件s"], 背景虚构件数=len(背件), **比,
              说明="稠密场整屏成像；结构门只用于'哪些是件'，成像不依赖件列表")

    # ── R2 新画面：组合想象（只搬件，背景留原样）──
    想 = 照.想象(位移=(3, 2), 路径=os.path.join(产物, "修复_新画面_想象.png"), 放大=1)
    件s集 = set(记["件s"])
    白黑件 = [k for k in 件s集 if k in ((1, 6), (2, 6), (3, 6), (4, 6),
                                        (7, 6), (8, 6), (7, 7), (8, 7), (7, 8), (8, 8))]
    R2 = dict(位移=list(想["位移"]), 原件数=len(件s集), 物件数=想.get("物件数"),
              搬动物件数=想.get("搬动物件数"), 新组合格数=想["新组合格数"],
              被搬_白黑平件数=len(白黑件),
              说明="整幅画仍稠密；按物件整体搬、越界反弹（不环面撕开）⇒ 画面不坏；"
                   "白/黑平件也搬（R0 缺陷修复）")

    # ── R3 形状可分：同色、同中心 4×4 的 方块/横条/竖条/圆 ──
    旧码, 新形状, 新图, 标签 = {}, {}, {}, {}
    for 形 in ("方块", "横条", "竖条", "圆"):
        f = 造形状屏(形)
        核 = 画面核(D=1024, 种子=3800)
        核.记一屏(f, 拍=1)
        旧码[形] = 核.特征码(核.块s(f)[(8, 4)])          # 旧：中心 2×2 块的码（不依赖件选没选上）
        照x = 稠密画面()
        照x.记一屏(f, 拍=1, 注视=(8, 4))
        新块 = 照x.库[-1]["块s"][(8, 4)]
        x0, y0, w, h = 形状框[形]
        补块 = f[max(0, y0 - 6):y0 + h + 6, max(0, x0 - 6):x0 + w + 6, :3]
        新形状[形] = 形状描述(补块)
        新图[形] = 照x.成像(照x.库[-1]["块s"], 放大=1)
        标签[形] = 形状标签_宽高比(形)
        _写png(新图[形], os.path.join(产物, "形状_新_%s.png" % 形))
    _写png(旧画面(造形状屏("方块"))[0], os.path.join(产物, "形状_旧_方块.png"))
    旧cos = {("%s-%s" % (a, b)): (None if 旧码[a] is None or 旧码[b] is None
                                 else round(图cos(旧码[a], 旧码[b]), 4))
             for a in 旧码 for b in 旧码 if a < b}
    新形cos = {("%s-%s" % (a, b)): (None if 新形状[a] is None or 新形状[b] is None
                                   else round(float(新形状[a] @ 新形状[b]), 4))
               for a in 新形状 for b in 新形状 if a < b}
    # 中央 3×3 格（±48px）区域的成像差：形状差应体现在轮廓/边界附近
    核区 = np.s_[(4 - 1) * 16:(4 + 2) * 16, (8 - 1) * 16:(8 + 2) * 16]
    图差 = {}
    for a in 新图:
        for b in 新图:
            if a < b:
                图差["%s-%s" % (a, b)] = round(float(np.mean(np.abs(
                    新图[a][核区].astype(np.float64) - 新图[b][核区].astype(np.float64)))), 2)
    R3 = dict(旧中心码相似=旧cos, 新形状描述相似=新形cos, 新画面中央区MAE=图差,
              宽高比标签=标签, 旧窄门=0.75,
              病灶=("器官_眼 的 元素s.形状 只看 bbox 宽高比：圆(32×32) → 「%s」，方块(32×32) → 「%s」，"
                  "两者标签相同；旧画面核中心码 cos=%.4f（同码）；新结构描述 cos=%.4f"
                  "（弱可分：<1 但仍 > 识别门 0.75 ⇒ 轮廓通道仍缺，见诚实边界）" %
                  (标签["圆"], 标签["方块"], 旧cos.get("圆-方块", 0) or 0.0,
                   新形cos.get("圆-方块", 0) or 0.0)),
              结论="旧：中心 4×4 同色 ⇒ 四种形状码几乎相同（分不出形，与初心「不知道方块」同病灶）；"
                   "新：形状描述（含轮廓的 972 维色无关结构）与画面中央区都不同")

    # ── R4 确定性：同种子两趟，画面字节逐位同 ──
    照B = 稠密画面()
    照B.记一屏(造屏(), 拍=1, 注视=(8, 4))
    h1 = hashlib.sha256(照.成像(照.库[-1]["块s"], 放大=1).tobytes()).hexdigest()
    h2 = hashlib.sha256(照B.成像(照B.库[-1]["块s"], 放大=1).tobytes()).hexdigest()
    R4 = dict(两趟画面逐位同=bool(h1 == h2), sha256=h1[:16])

    报["臂"] = dict(R0_旧画面=R0, R1_新画面=R1, R2_新想象=R2,
                   R3_形状可分=R3, R4_确定性=R4)
    out = os.path.join(HERE, "实验_画面修复_稠密视场结果.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(报["臂"], ensure_ascii=False, indent=1, default=str))
    print("\n[画面修复] 产物目录：%s\n结果：%s" % (产物, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
