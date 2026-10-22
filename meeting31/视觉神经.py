# -*- coding: utf-8 -*-
"""meeting31/视觉神经.py —— #288 真视觉：多尺度保留细节+视觉原型词典（可热插拔感官）。

回应所有者两难题：
  看到=帧进入编码器（全景网格，视野级）；
  看清=显著点→中央凹高清裁剪（焦点细节保留，人类 fovea 同构）；
  看懂=视觉原型词典——相似补丁贪心聚类成"视觉词"（词#id），词典持续生长；
       细节保留度可判读：词典重建帧 vs 真帧的误差（自测）。
设计约束：无 OCR、无神经网络——全部随机投影+统计聚类（确定性 seed）。
"""
from __future__ import annotations

import numpy as np

_rng = np.random.default_rng(2025)
_D = 1024
_全景投影 = _rng.standard_normal((32 * 24 * 3, _D)).astype(np.float32)
_中央凹投影 = _rng.standard_normal((32 * 24 * 3, _D)).astype(np.float32)

_原型词典: list = []       # 原型补丁向量——视觉词的"词表"（7维描述子或旧3维色向量）
_原型阈值 = 0.90           # 兼容旧签名保留；新路径用 两档阈值
_紧阈值 = 0.99             # ≥紧=同一视觉词（原样吸收）
_松阈值 = 0.96             # 松≤cos<紧=同一视觉词且 EMA 吸收变化（防词典碎片化）
# 档位定标依据（probe_视觉词 实测）：黄/橙 mean-cos≈0.90 须分词 → 松≥0.96；
# 灰度微差 cos≈0.999 须同词 → 松≤0.96 与 紧≈0.99 双界同时满足；
# 条纹块 vs 纯色块 ≈0.93 → 分词（结构差异成词）。
_词典上限 = 2048
_使用次数: list = []        # 每词被命中次数（生长/淘汰统计）
_生长史: list = []          # (补丁序号, 词典规模) 采样记录


def 细网格(img, 块=8):
    """7 维补丁描述子网格：块内 RGB 均值(3)+RGB 标准差(3)+边缘密度(1)。
    旧 3 维色均值无结构信息（同色不同纹理同词→词典不生长的病根），方差+边缘补上。"""
    a = np.asarray(img.convert("RGB").resize((128, 96)), dtype=np.float32) / 255.0
    H, W = 96 // 块, 128 // 块
    m = a.reshape(H, 块, W, 块, 3)
    均值 = m.mean(axis=(1, 3))
    标准差 = m.std(axis=(1, 3))
    灰 = a.mean(axis=2)
    边缘 = np.zeros((H, W), dtype=np.float32)
    for by in range(H):
        for bx in range(W):
            b = 灰[by * 块:(by + 1) * 块, bx * 块:(bx + 1) * 块]
            边缘[by, bx] = float(np.abs(np.diff(b, axis=1)).mean() +
                                 np.abs(np.diff(b, axis=0)).mean())
    描述 = np.concatenate([均值, 标准差 * 2.0,
                           边缘[:, :, None] * 2.0], axis=2).astype(np.float32)
    # ×2 加权依据（probe_视觉词 实测）：白条纹对品红/橙的均值偏移使 cos≥0.96 被吞；
    # std/边缘 ×2 后四色条纹块 cos 全部 <0.96（结构差异获得与颜色相当的话语权），
    # 灰度微差（std=0, 边缘=0）不受影响仍合并。
    return 描述                                                  # (H,W,7)


def 视觉词(补丁向量) -> int:
    """补丁→原型词：贪心最近邻两档匹配；无近邻则生长新词。
    两档：cos≥紧=同词；松≤cos<紧=同词+EMA 吸收变化（原型随数据微调，
    同序确定性）；<松=新词。词典满=-1（细节溢出，如实）。"""
    n2 = np.linalg.norm(补丁向量)
    if n2 > 1e-9 and _原型词典:
        矩阵 = np.stack(_原型词典)
        n1 = np.linalg.norm(矩阵, axis=1)
        相似 = (矩阵 @ 补丁向量) / (n1 * n2 + 1e-9)
        i = int(np.argmax(相似))
        c = float(相似[i])
        if c >= _紧阈值:
            _使用次数[i] += 1
            return i
        if c >= _松阈值:
            # EMA 吸收（α=0.1）：词义随输入流微调，确定性
            _原型词典[i] = (0.9 * _原型词典[i] + 0.1 * np.asarray(补丁向量,
                                                                 np.float32))
            _使用次数[i] += 1
            return i
    if len(_原型词典) < _词典上限:
        _原型词典.append(np.asarray(补丁向量, np.float32).copy())
        _使用次数.append(1)
        _生长史.append((len(_使用次数), len(_原型词典)))
        return len(_原型词典) - 1
    return -1


def 视觉词表(img, 块=8):
    """整图→视觉词 id 网格（7 维描述子路径）。返回 (H,W) int 网格。"""
    描述 = 细网格(img, 块)
    H, W = 描述.shape[:2]
    out = np.zeros((H, W), dtype=np.int64)
    for by in range(H):
        for bx in range(W):
            out[by, bx] = 视觉词(描述[by, bx])
    return out


def 词典规模() -> int:
    return len(_原型词典)


def 词典统计() -> dict:
    """生长健康度：规模、命中分布（top 词占比过高=分辨率退化）。"""
    if not _使用次数:
        return dict(规模=0, top占比=0.0, 命中中位=0)
    u = np.asarray(_使用次数)
    return dict(规模=len(u), top占比=round(float(u.max() / u.sum()), 3),
                命中中位=int(np.median(u)), 生长史长度=len(_生长史))


def 编码帧(img):
    """全景：PIL Image→(24×32×3 网格, D 维全景向量)。"""
    a = np.asarray(img.convert("RGB").resize((128, 96)), dtype=np.float32) / 255.0
    g = a.reshape(24, 4, 32, 4, 3).mean(axis=(1, 3))
    return g, g.flatten() @ _全景投影


def 显著点(g, 前g=None, k=3):
    """显著性=局部对比度+变化加权（动过的东西更醒目）。返回 [(行,列,分)]。"""
    d = np.zeros(g.shape[:2], dtype=np.float32)
    d[:, :-1] += np.abs(g[:, 1:] - g[:, :-1]).mean(axis=2)
    d[:-1, :] += np.abs(g[1:, :] - g[:-1, :]).mean(axis=2)
    if 前g is not None:
        d += np.abs(g - 前g).mean(axis=2) * 2.0
    平 = d.flatten()
    序 = np.argsort(-平)[:k]
    return [(int(i // 32), int(i % 32), round(float(平[i]), 4))
            for i in 序 if 平[i] > 0.02]


def 中央凹(img, cx, cy, 宽=320, 高=240):
    """焦点：以 (cx,cy) 为中心的高清裁剪→(24×32×3 细网格, D 维焦点向量, 实际焦点)。"""
    x0 = max(0, min(img.width - 宽, int(cx - 宽 / 2)))
    y0 = max(0, min(img.height - 高, int(cy - 高 / 2)))
    crop = np.asarray(img.convert("RGB").crop((x0, y0, x0 + 宽, y0 + 高)),
                      dtype=np.float32) / 255.0
    g = crop.reshape(高 // 10, 10, 宽 // 10, 10, 3).mean(axis=(1, 3))
    return g, g.flatten() @ _中央凹投影, (x0 + 宽 // 2, y0 + 高 // 2)


def 区域特征(g):
    """看清（无NN 启发式）：边缘密度+颜色熵→文本状/图片状/均匀块。"""
    灰 = g.mean(axis=2)
    边缘 = float(np.abs(np.diff(灰, axis=1)).mean())
    量化 = (g * 4).astype(int).reshape(-1, 3)
    _, c = np.unique(量化, axis=0, return_counts=True)
    p = c / c.sum()
    熵 = float(-(p * np.log(p + 1e-9)).sum())
    if 边缘 > 0.02 and 熵 < 1.8:
        类 = "文本状"
    elif 熵 > 2.4:
        类 = "图片状"
    else:
        类 = "均匀块"
    return 类, round(边缘, 4), round(熵, 3)


def 重建误差(g, 抽样=48, img=None) -> float:
    """细节保留度自测：抽样补丁用最近原型近似的余弦距离均值（越小保留越好）。
    img 给出时走 7 维富描述子路径（与 视觉词表 同表示）；否则旧 3 维路径
    （自动跳过维度不符的原型，防混拼崩溃）。"""
    if not _原型词典:
        return -1.0
    if img is not None:
        描述 = 细网格(img)
        H, W = 描述.shape[:2]
        rs = np.random.default_rng(7).choice(H * W, min(抽样, H * W),
                                             replace=False)
        err, n = 0.0, 0
        矩阵 = [p for p in _原型词典 if p.size == 7]
        if not 矩阵:
            return -1.0
        M = np.stack(矩阵)
        Mn = np.linalg.norm(M, axis=1) + 1e-9
        for i in rs:
            v = 描述[i // W, i % W]
            n2 = np.linalg.norm(v)
            if n2 < 1e-9:
                continue
            best = float(np.max(M @ v / (Mn * n2)))
            err += 1.0 - best
            n += 1
        return round(err / max(n, 1), 4)
    rs = np.random.default_rng(7).choice(g.shape[0] * g.shape[1],
                                         min(抽样, g.shape[0] * g.shape[1]),
                                         replace=False)
    err, n = 0.0, 0
    for i in rs:
        v = g[i // g.shape[1], i % g.shape[1]].flatten()
        n2 = np.linalg.norm(v)
        if n2 < 1e-9:
            continue
        best = 1.0
        for p in _原型词典:
            if p.size != v.size:                     # 维度护栏：混表示跳过
                continue
            n1 = np.linalg.norm(p)
            if n1 > 1e-9:
                cosv = float(p @ v) / (n1 * n2)
                if 1.0 - cosv < best:
                    best = 1.0 - cosv
        err += best
        n += 1
    return round(err / max(n, 1), 4)
