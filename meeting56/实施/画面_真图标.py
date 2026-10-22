# -*- coding: utf-8 -*-
"""画面_真图标.py —— meeting56 四轮：**真桌面图标**（真实截图）的检出 / 轮廓原型 / 接形状束·图标簿 / 识别门重标。

【为什么有本文件（meeting56 §12.6 第一步原话）】
  「接真图标：把 §12 的"轮廓原型"接到 形状束/图标簿 与真实桌面截图（非判据演示先跑），用真图标重标识别门。」
  三轮（`画面_轮廓想象.py`）只跑了**合成线画图标**；本件把同一套轮廓原型搬进**真桌面截图**，并
  真的走一遍既有 `形状描述子`（1024 维双极 `形状束`）与 `器官_脑.簿`（`识别门 = 0.75`，`器官_脑.py:409`）。

【本文件做什么（全部确定、零训练、零外挂模型、零 OCR、非 Transformer）】
  1) **看**：真截图 → 梯度能量图 → **桌面栅格**（Windows 图标栅距，实测 76×100）→ 有能量的格 = 图标格；
     格内上半部再收tight字形框（标签文字在格下半部，**不读字**，C9）。
     位置 = (行, 列) 格键 —— 来自外部空间帧（桌面栅格），不是像素投影（与 §5 接缝一致）。
  2) **轮廓原型**：字形补块 → 灰度 → DoG 高通过滤（去掉壁纸大尺度渐变）→ 边缘强度图 E；
     `轮廓图` = E 的三阈值层的边界并集（同 `画面_轮廓想象.轮廓图` 的律，只是输入换成**边缘通道**，
     因为真图标是彩色/透明底，直接对灰度做阈值会被壁纸吃掉）。
  3) **接形状束/图标簿**：
     - `码_形状束`：把 **E** 当 patch 交给既有 `形状束.形状描述子(编码).形状束()` → **1024 维双极码**；
       这就是把"轮廓原型"塞进既有**形状束**表示（同律：64×64、三阈值、16/8/2 池化、层内去共模）。
     - `图标簿`：`dict(束/码, 名, 类, 位置, 成员s)`；识别 = 余弦最近邻 + **门**（默认 0.75）。
       `接图标簿()` 提供把条目灌进真实 `器官_脑.脑器.簿` 的桥（簿条目格式见 `器官_脑.py:65,234`）。
  4) **门重标**：用真图标（含真实退化：DPI 缩放/平移/JPEG/噪声/模糊）做 genuine/impostor 打分，
     `标定门()` 给出 ROC 上的 τ，并与既有 `识别门=0.75` 同口径对比。

【为什么不改既有件】C14：`形状束.py` / `器官_脑.py` / `画面_轮廓想象.py` **一字未改**，本件是旁挂新文件；
`识别门` 若要在实验里换值，只在实例上遮蔽（`脑.识别门 = x`），不动源码。

【依赖】numpy + cv2（cv2 只用于 Sobel/DoG/缩放/连通域；`形状束.py` 本身就依赖 cv2）。
       `共存态`（取 1024 维编码）只在 `码_形状束`/`接图标簿` 里**惰性导入**，不 import 本模块就付 7s 代价。

铁律：C1 禁 Transformer ｜ C9 禁 OCR（不读字符，只看结构/轮廓）｜ C10 禁外挂模型当器官
      ｜ C13 确定性可复现（固定栅格、固定种子、无网络）｜ C14 不破既有。
"""
from __future__ import annotations

import hashlib
import os

import cv2
import numpy as np

# ────────────────────────── 口径常量（全部显式，便于复核）──────────────────────────
列距, 行距 = 76, 100          # Windows 桌面图标栅距（实测：三张真屏 ACF 峰 76/100）
能量阈 = 55.0                 # 格存在判据：字形带（格内上 46 行）平均边缘能量 ≥ 55
字形带 = (8, 64)              # 格内字形带的纵向范围（相对格顶）：实测字形在 12..62，标签在 68 以下
掩阈 = 120.0                  # 字形 tight 框用的二值掩阈值（梯度能量）
补块边 = 1.20                 # 补块边长 = max(字框宽, 高) × 1.20
旧识别门 = 0.75               # 既有 `器官_脑.py:409`
任务栏比 = 0.06               # 排除底部任务栏（~6% 高）


# ────────────────────────── 读 / 写 / 显示 ──────────────────────────
def 读图(路径: str) -> np.ndarray:
    """Unicode 路径安全读 PNG/JPG → RGB uint8。`cv2.imread` 对中文路径静默失败，一律 imdecode。"""
    buf = np.fromfile(路径, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError("读图失败：%s" % 路径)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def 写图(图: np.ndarray, 路径: str) -> str:
    """RGB/RGBA/gray uint8 → PNG（Unicode 路径安全）。"""
    a = np.asarray(图)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    a = np.clip(a, 0, 255).astype(np.uint8)[:, :, :3]
    ok, enc = cv2.imencode(".png", cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError("编码失败：%s" % 路径)
    os.makedirs(os.path.dirname(os.path.abspath(路径)), exist_ok=True)
    enc.tofile(路径)
    return 路径


def 文件sha(路径: str) -> str:
    h = hashlib.sha256()
    with open(路径, "rb") as f:
        for 块 in iter(lambda: f.read(1 << 20), b""):
            h.update(块)
    return h.hexdigest()


# ────────────────────────── 看：能量 / 栅格 / 图标格 ──────────────────────────
def 灰度(图) -> np.ndarray:
    a = np.asarray(图, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] >= 3:
        return (0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]).astype(np.float32)
    return a


def 能量图(图, 窗: int = 7) -> np.ndarray:
    """梯度能量（Sobel 幅值 × 7×7 均值）：图标笔画/文字 >> 壁纸大尺度渐变。"""
    g = 灰度(图)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.blur(np.hypot(gx, gy), (窗, 窗))


def _能量掩(能量, 图侧) -> np.ndarray:
    m = (能量 > 掩阈).astype(np.uint8)
    m[int(图侧[0] * (1 - 任务栏比)):, :] = 0
    return m


def 估算栅格(能量, 图侧, 排除底: int = 0, 列距_: int = None, 行距_: int = None) -> dict:
    """真屏栅格标定：(列距, 行距) 先验固定（实测 76×100），**相位**由"图标+标签连通域中心"的中位残差给出。

    做法（只读、确定性）：低阈掩 → 膨胀 → 连通域 → 取尺寸像"图标+标签"的盒子
    （高 55..110、宽 40..110，按 `列距_/76` 的倍率缩放）→ 中心 mod 栅距 的中位数 = 相位。
    相位测量口径：`列相`/`行相` = **图标字形中心**在栅格上的位置（行相不是格中心）。
    `列距_/行距_` 用于跨 DPI 模拟（整屏重采样后栅距按倍率缩放）。
    """
    H, W = 图侧
    P = int(列距_) if 列距_ else 列距
    Q = int(行距_) if 行距_ else 行距
    比 = P / float(列距)
    m = (能量 > 35.0 * max(0.5, 比)).astype(np.uint8)
    if 排除底:
        m[H - 排除底:, :] = 0
    m = cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=2)
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(m)
    cx, cy = [], []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 900 * 比 * 比:
            continue
        if not (55 * 比 <= h <= 110 * 比 and 40 * 比 <= w <= 110 * 比):
            continue
        if not (0.5 <= w / float(h) <= 1.8):
            continue
        cx.append(x + w / 2.0)
        cy.append(y + h / 2.0)
    if len(cx) < 3:                                    # 退化：用固定相位
        return dict(列距=P, 行距=Q, 列相=38.0 * 比, 行相=42.0 * 比, 格子源=len(cx))
    return dict(列距=P, 行距=Q,
                列相=float(np.median(np.mod(cx, P))),
                行相=float(np.median(np.mod(cy, Q))),
                格子源=len(cx))


def 格位s(图侧, 栅: dict, 排除底: int = 0) -> list:
    """枚举整张图的所有候选格（行, 列, 格框, 可见字形带）。

    字形带 = 格内纵向 [8, 64)（实测：字形在格内 12..62、标签在 68 以下；不读字，C9）；
    允许贴边裁剪（A/B 是局部截图），要求可见字形带 ≥ 60 宽 × 40 高。
    """
    H, W = 图侧
    P, Q = 栅["列距"], 栅["行距"]
    ox, oy = 栅["列相"], 栅["行相"]
    比 = P / float(列距)
    带 = (int(round(字形带[0] * 比)), int(round(字形带[1] * 比)))
    c0 = int(np.floor((0 - ox) / P)) - 1
    c1 = int(np.ceil((W - ox) / P)) + 1
    r0 = int(np.floor((0 - oy) / Q)) - 1
    r1 = int(np.ceil((H - oy) / Q)) + 1
    格s = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            x0 = int(round(ox + P * c - P / 2.0))
            y0 = int(round(oy + Q * r - Q / 2.0))       # 行相 = 字形中心 ⇒ 格顶 = 行相 - 行距/2
            bx0, bx1 = max(0, x0), min(W, x0 + P)
            by0, by1 = max(0, y0 + 带[0]), min(H - 排除底, y0 + 带[1])
            if bx1 - bx0 < 60 * 比 or by1 - by0 < 40 * 比:
                continue
            格s.append(dict(行=r, 列=c, 框=(x0, y0, P, Q),
                            字带=(bx0, by0, bx1 - bx0, by1 - by0)))
    return 格s


def 检出(图, 密阈: float = 0.08, 排除底: int = 0, 栅=None,
         列距_: int = None, 行距_: int = None) -> dict:
    """真截图 → 图标格（桌面栅格键 = 位置）+ 字形框（tight，供人眼核对）。

    `排除底`：整屏截图才有任务栏（C 用 48）；局部截图（A/B）传 0。
    `列距_/行距_`：跨 DPI 模拟用（整屏重采样后传缩放后的栅距）。
    存在判据：字形带内"强边缘像素（能量>120）占比" ≥ 密阈（默认 0.08），
    且掩的 bbox 尺寸/位置/填充率像**一个居中的图标字形**（壁纸旋涡的细长边会被拒）。
    这不是"显著性/误差场涌现"，是**本轮演示通道的启发式**（诚实边界见台账）。
    """
    H, W = 图.shape[:2]
    能量 = 能量图(图)
    栅 = 栅 or 估算栅格(能量, (H, W), 排除底=排除底,
                        列距_=列距_, 行距_=行距_)
    P, Q = 栅["列距"], 栅["行距"]
    比 = P / float(列距)
    m = (能量 > 掩阈 * max(0.6, (比 ** 0.5))).astype(np.uint8)
    if 排除底:
        m[H - 排除底:, :] = 0
    图标s = []
    for 格 in 格位s((H, W), 栅, 排除底):
        bx0, by0, bw, bh = 格["字带"]
        band = m[by0:by0 + bh, bx0:bx0 + bw]
        密 = float(band.mean())
        if 密 < 密阈:
            continue
        ys, xs = np.nonzero(band)
        if xs.size < 20:
            continue
        gw, gh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        if not (12 * 比 <= gw <= 76 * 比 and 10 * 比 <= gh <= 62 * 比):
            continue
        cx, cy = xs.min() + gw / 2.0, ys.min() + gh / 2.0
        if abs(cx - bw / 2.0) > 20 * 比 or abs(cy - bh / 2.0) > 24 * 比:
            continue
        if float(xs.size) / (gw * gh) < 0.10:
            continue
        字框 = (int(bx0 + xs.min()), int(by0 + ys.min()), gw, gh)
        图标s.append(dict(行=格["行"], 列=格["列"], 格框=格["框"], 字带=格["字带"], 字框=字框,
                          密=round(密, 4), 中心=(int(字框[0] + gw / 2), int(字框[1] + gh / 2))))
    图标s.sort(key=lambda t: (t["行"], t["列"]))
    return dict(图标s=图标s, 栅格=栅)


def 图格号(图路径: str, 图标: dict) -> str:
    """图标在真屏上的可复读身份：文件 + 行/列（行列为桌面栅格键，不是像素）。"""
    return "%s#r%dc%d" % (os.path.basename(图路径), 图标["行"], 图标["列"])


def 外形标记(图, 图标s, 色=(0, 255, 0)) -> np.ndarray:
    """人眼核对用：把检出结果画在图上（只画不判，不改原图）。"""
    out = np.array(图, dtype=np.uint8).copy()
    for it in 图标s:
        x, y, w, h = it["字框"]
        cv2.rectangle(out, (x, y), (x + w, y + h), 色, 1)
    return out


def 补块(图, 图标, 边=补块边) -> np.ndarray:
    """**固定取景**的字形补块（RGB）：格内字形带（不含标签），供编码用。

    为什么固定取景而不是 tight bbox：真图标有透明底，背景深浅会让 tight bbox 抖动；
    固定取景把所有图标放在同一坐标系里，码差才反映形状差（而不是框差）。
    """
    x, y, w, h = 图标["字带"]
    sub = np.asarray(图[y:y + h, x:x + w], np.uint8)
    if sub.shape[0] < 12 or sub.shape[1] < 12:
        return None
    return sub


def 字形补块(图, 图标, 边=补块边) -> np.ndarray:
    """tight 字形补块（仅用于产物/人眼核对，不用于编码）。"""
    H, W = 图.shape[:2]
    x, y, w, h = 图标["字框"]
    s = int(round(max(w, h) * 边)) + 2
    cx, cy = x + w / 2.0, y + h / 2.0
    x0 = int(round(cx - s / 2.0))
    y0 = int(round(cy - s / 2.0))
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + s), min(H, y0 + s)
    sub = np.asarray(图[ya:yb, xa:xb], np.uint8)
    if sub.shape[0] < 6 or sub.shape[1] < 6:
        return None
    return sub


# ────────────────────────── 轮廓原型（真图标版）──────────────────────────
def 边缘图(补块, σ: float = 6.0) -> np.ndarray:
    """DoG 高通边缘强度（色无关）：消掉壁纸大尺度渐变，留图标笔画/边界。

    返回 0..255 float32（按 99.5 分位归一，确定性）。
    """
    g = 灰度(补块)
    dog = np.abs(g - cv2.GaussianBlur(g, (0, 0), σ))
    hi = float(np.percentile(dog, 99.5))
    if hi < 1e-6:
        return np.zeros_like(dog, np.float32)
    return np.clip(dog / hi * 255.0, 0, 255).astype(np.float32)


def 边缘色差图(补块, σ: float = 6.0) -> np.ndarray:
    """Lab 三通道 DoG 的 L2 合成（用色差找边界，仍**不把颜色当身份**）：对彩色图标更稳。"""
    rgb = np.asarray(补块, np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    acc = None
    for k in range(3):
        ch = lab[:, :, k]
        d = np.abs(ch - cv2.GaussianBlur(ch, (0, 0), σ))
        acc = d * d if acc is None else acc + d * d
    dog = np.sqrt(acc)
    hi = float(np.percentile(dog, 99.5))
    if hi < 1e-6:
        return np.zeros(dog.shape, np.float32)
    return np.clip(dog / hi * 255.0, 0, 255).astype(np.float32)


def 轮廓图(边缘或补块, 模式: str = "边缘") -> np.ndarray:
    """轮廓图 = 三阈值结构层的**边界并集**（64×64 0/1）。

    - 模式="边缘"：直接对 `边缘色差图(补块)` 三层取边界（真图标主用）；
    - 模式="灰度"：对灰度补块三层取边界（合成图标老律，保留对照）。
    退化层（占比 >0.97 或 <0.03）跳过；补块最外 1px 不算轮廓（避免假大框）。
    """
    if 模式 == "边缘":
        E = 边缘或补块
        if E.ndim == 3:
            E = 边缘色差图(E)
        small = cv2.resize(E.astype(np.float32), (64, 64), interpolation=cv2.INTER_AREA)
    else:
        g64 = cv2.resize(灰度(边缘或补块), (64, 64), interpolation=cv2.INTER_AREA)
        return 结构层轮廓(g64)
    return _三层边界(small)


def _边界(B: np.ndarray) -> np.ndarray:
    内 = np.asarray(B).astype(bool)
    r = np.pad(内, 1, constant_values=False)
    邻全 = r[2:, 1:-1] & r[:-2, 1:-1] & r[1:-1, 2:] & r[1:-1, :-2]
    return (内 & ~邻全).astype(np.uint8)


def _三层边界(图64) -> np.ndarray:
    a = np.asarray(图64, np.float32)
    μ, σ = float(a.mean()), float(a.std())
    out = np.zeros((64, 64), np.uint8)
    for 阈 in (μ - 0.6 * σ, μ, μ + 0.6 * σ):
        B = (a > 阈).astype(np.uint8)
        占比 = float(B.mean())
        if 占比 > 0.97 or 占比 < 0.03:
            continue
        out |= _边界(B)
    out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = 0
    return out


def 结构层轮廓(灰64) -> np.ndarray:
    return _三层边界(np.asarray(灰64, np.float32))


# ────────────────────────── 背景分离 / 质心对齐（五轮）──────────────────────────
def 背景色(补块, 边宽: int = 4) -> np.ndarray:
    """补块边缘带的 RGB 中位色 = 该图标所在处的局部背景（壁纸局部近似平滑，不做颜色恒常）。"""
    a = np.asarray(补块, np.float32)
    H, W = a.shape[:2]
    e = max(1, int(边宽))
    边 = np.concatenate([a[:e, :, :].reshape(-1, 3), a[-e:, :, :].reshape(-1, 3),
                         a[:, :e, :].reshape(-1, 3), a[:, -e:, :].reshape(-1, 3)], 0)
    return np.median(边, 0).astype(np.float32)


def 前景对比图(补块, 分位: float = 99.5) -> np.ndarray:
    """背景分离：补块 → (RGB-Lab) 与局部背景色的距离图（0..255）。

    为什么：原图标有透明底，补块里混着壁纸；直接对原像素做形状束会被背景拉成同码
    （四轮实测：束(原始像素) 的"原样"只有 .914、文件夹 8 组同码/21 格）。
    这里只用"离局部背景多远"当地形，不做颜色恒常、不把颜色当身份。
    """
    rgb = np.asarray(补块, np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    H, W = lab.shape[:2]
    e = max(1, min(4, H // 8, W // 8))
    边 = np.concatenate([lab[:e, :, :].reshape(-1, 3), lab[-e:, :, :].reshape(-1, 3),
                         lab[:, :e, :].reshape(-1, 3), lab[:, -e:, :].reshape(-1, 3)], 0)
    bg = np.median(边, 0)
    d = np.sqrt(((lab - bg) ** 2).sum(2))
    hi = float(np.percentile(d, 分位))
    if hi < 1e-6:
        return np.zeros((H, W), np.float32)
    return np.clip(d / hi * 255.0, 0, 255).astype(np.float32)


def 字形质心(前景, 下限比: float = 0.25) -> tuple:
    """前景图的亮度加权质心（= 字形锚点；用于把位置抖动归一）。"""
    a = np.asarray(前景, np.float32)
    hi = float(np.percentile(a, 99.0))
    w = np.where(a > max(20.0, 下限比 * hi), a, 0.0)
    和 = float(w.sum())
    if 和 < 1e-6:
        return a.shape[1] / 2.0, a.shape[0] / 2.0
    ys, xs = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    return float((w * xs).sum() / 和), float((w * ys).sum() / 和)


def 对齐补块(补块, 前景=None, 最大移比: float = 0.25) -> dict:
    """把补块按字形质心平移到中心（整数/亚像素 warpAffine，确定性）。

    这就是四轮最弱臂（移 ±3px 掉到 .30–.53）的补法：**先对齐再编码**，不是把位置信息丢掉
    （位置仍由桌面栅格的（行,列）承担；这里对齐的是"格内字形落点抖动"）。
    """
    rgb = np.asarray(补块, np.uint8)
    fg = 前景对比图(rgb) if 前景 is None else np.asarray(前景, np.float32)
    H, W = rgb.shape[:2]
    cx, cy = 字形质心(fg)
    dx = float(np.clip(W / 2.0 - cx, -最大移比 * W, 最大移比 * W))
    dy = float(np.clip(H / 2.0 - cy, -最大移比 * H, 最大移比 * H))
    bg = 背景色(rgb)
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], np.float32)
    出 = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT,
                       borderValue=tuple(float(x) for x in bg))
    出f = cv2.warpAffine(fg, M, (W, H), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return dict(补块=出, 前景=出f, 位移=(round(dx, 2), round(dy, 2)), 质心=(round(cx, 2), round(cy, 2)))


# ────────────────────────── 码：972 轮廓码 / 1024 形状束 ──────────────────────────
def _池化(图, n: int) -> np.ndarray:
    h = 图.shape[0] // n
    return 图[:h * n, :h * n].reshape(n, h, n, h).mean(axis=(1, 3)).ravel()


def 码_972(轮廓或边缘, 已轮廓: bool = False) -> np.ndarray:
    """972 维轮廓码（三轮同律：3 阈值 × 16/8/2 池化，层内去共模，L2 归一）。

    输入：轮廓图（0/1，`已轮廓=True`）或边缘/灰度图（0..255，自动缩 64×64）。
    """
    a = np.asarray(轮廓或边缘, np.float32)
    if a.ndim == 3:
        a = 灰度(a)
    if not 已轮廓:
        a = cv2.resize(a, (64, 64), interpolation=cv2.INTER_AREA)
    μ, σ = float(a.mean()), float(a.std())
    特征 = []
    for 阈 in (μ - 0.6 * σ, μ, μ + 0.6 * σ):
        B = (a > 阈).astype(np.float32)
        for n in (16, 8, 2):
            v = _池化(B, n)
            v = v - v.mean()
            特征 += v.ravel().tolist()
    v = np.asarray(特征, np.float32)
    return v / (float(np.linalg.norm(v)) + 1e-12)


_形状描缓存 = {}


def 形状描述子(编码=None):
    """惰性取既有 `形状束.形状描述子`（同律、不改源）：`编码` 缺省用 `共存态.共存态(种子=3800).编码`。"""
    import sys
    AHI4 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for p in (os.path.join(AHI4, "meeting38"), os.path.join(AHI4, "meeting38", "实施"),
              os.path.join(AHI4, "meeting39", "实施")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from 形状束 import 形状描述子 as _类
    if 编码 is None:
        key = "默认"
        if key not in _形状描缓存:
            import 共存态
            _形状描缓存[key] = 共存态.共存态(种子=3800, 用gpu=False).编码
        编码 = _形状描缓存[key]
    return _类(编码)


def 码_形状束(补块或边缘, 编码=None) -> np.ndarray | None:
    """既有 `形状束` 律（1024 维双极码），**归一化到单位向量**后返回。

    为什么要归一：`编码.特征码` 输出的是双极向量（范数 ≠ 1）；既有识别用
    `编码.相似` = 归一化点积（余弦）。本件把归一化做在返回处，之后一律用点积=余弦，
    这样门值（0.75 等）与既有 `器官_脑.识别门` 同口径可比较。
    """
    a = np.asarray(补块或边缘)
    if a.ndim == 2:
        a = np.stack([a, a, a], -1)
    v = 形状描述子(编码).形状束(np.clip(a, 0, 255).astype(np.uint8))
    if v is None:
        return None
    v = np.asarray(v, np.float32)
    return v / (float(np.linalg.norm(v)) + 1e-12)


def 相似(a, b) -> float:
    return float(np.dot(a, b))


# ────────────────────────── 图标簿（旁挂；格式对齐 器官_脑.簿）──────────────────────────
class 图标簿:
    """条目：{名, 类, 码, 束, 位置, 成员s, 原型s}。识别 = 余弦最近邻 + 门。"""

    def __init__(self, 门: float = 旧识别门):
        self.条目s: list = []
        self.门 = float(门)

    def 记(self, 名: str, 码: np.ndarray, 类: str = "", 位置=None, 束=None, 成员s=None):
        self.条目s.append(dict(名=名, 类=类, 码=np.asarray(码, np.float32),
                              束=束, 位置=位置, 成员s=list(成员s or [])))
        return len(self.条目s) - 1

    def 识(self, 码, 门: float | None = None) -> dict:
        门 = self.门 if 门 is None else float(门)
        if not self.条目s:
            return dict(认得=False, 原因="簿为空")
        拟 = [(相似(码, 条["码"]), i) for i, 条 in enumerate(self.条目s)]
        拟.sort(key=lambda t: (-t[0], t[1]))
        s, i = 拟[0]
        return dict(认得=bool(s >= 门), 名=self.条目s[i]["名"], 类=self.条目s[i]["类"],
                    下标=i, 相似=round(float(s), 4),
                    次佳=round(float(拟[1][0]), 4) if len(拟) > 1 else None, 门=round(门, 4))

    def 接图标簿(self, 脑, 名称键: str = "名") -> int:
        """把本簿条目灌进**真实** `器官_脑.脑器.簿`（格式 `dict(束, 心, 名称, 成功, 大变)`）。

        需要 `脑` 是已构造的 `脑器`（`脑.簿` 是 dict）；返回灌入条数。
        只写实例状态，不改 `器官_脑.py`（C14）。`识别门` 仍由 `脑.识别门` 决定。
        """
        n = 0
        for 条 in self.条目s:
            束 = 条.get("束")
            if 束 is None:
                continue
            号 = len(脑.簿)
            位 = 条.get("位置") or (0, 0)
            脑.簿[号] = dict(束=束, 心=(int(位[0]), int(位[1])), 名称=条[名称键],
                            成功=2, 大变=2)
            n += 1
        return n


# ────────────────────────── 门重标：genuine / impostor → τ ──────────────────────────
def 标定门(真s, 假s, 旧门: float = 旧识别门) -> dict:
    """用真图标打分重标识别门。

    真s = genuine 相似（同一图标的退化探针 vs 自己原型）；假s = impostor（vs 别的原型）。
    判据：τ 在候选网格上取 **Youden J = TPR − FPR** 最大；同分取 τ 较大者（保守）。
    返回 τ、旧门同口径指标（TPR/FPR/准确）、以及"接受即正确"的混淆表。
    """
    真s = np.asarray(真s, np.float64)
    假s = np.asarray(假s, np.float64)
    cand = np.round(np.arange(0.30, 0.991, 0.005), 4)

    def _指标(τ):
        tpr = float((真s >= τ).mean()) if 真s.size else 0.0
        fpr = float((假s >= τ).mean()) if 假s.size else 0.0
        return tpr, fpr, tpr - fpr

    评 = [(float(_指标(t)[2]), float(t)) for t in cand]
    评.sort(key=lambda x: (x[0], x[1]))
    J, τ = 评[-1]
    旧tpr, 旧fpr, 旧J = _指标(旧门)
    新tpr, 新fpr, 新J = _指标(τ)
    return dict(τ=round(τ, 4), J=round(J, 4),
                旧门=round(旧门, 4), 旧门_TPR=round(旧tpr, 4), 旧门_FPR=round(旧fpr, 4),
                旧门_J=round(旧J, 4),
                τ_TPR=round(新tpr, 4), τ_FPR=round(新fpr, 4),
                真s_均=round(float(真s.mean()), 4) if 真s.size else None,
                真s_最小=round(float(真s.min()), 4) if 真s.size else None,
                假s_均=round(float(假s.mean()), 4) if 假s.size else None,
                假s_最大=round(float(假s.max()), 4) if 假s.size else None,
                候选数=int(cand.size))


# ────────────────────────── 想象：从记忆的轮廓重画（虚线/碎片）──────────────────────────
def 稀疏(轮廓: np.ndarray, 生动度: float, 种子: int = 5601) -> np.ndarray:
    """同 `画面_轮廓想象._稀疏` 律：固定种子保留一部分轮廓像素（确定性虚线/碎片）。"""
    有效 = np.flatnonzero(np.asarray(轮廓).ravel() > 0)
    if 有效.size == 0:
        return np.zeros_like(轮廓, np.float32)
    k = int(round(float(生动度) * 有效.size))
    乱 = np.random.default_rng(int(种子)).permutation(有效.size)
    留 = np.zeros(轮廓.size, np.float32)
    留[有效[乱[:max(0, min(k, 有效.size))]]] = 1.0
    return (留.reshape(轮廓.shape) * 轮廓).astype(np.float32)


def 想象(轮廓: np.ndarray, 生动度: float = 0.45, 种子: int = 5601) -> np.ndarray:
    """记忆轮廓 → 稀疏轮廓（0..255 灰度，供人眼核对"想出来的虚线/碎片"）。"""
    return 稀疏(轮廓, 生动度, 种子) * 255.0


def 预测误差IoU(想象轮廓: np.ndarray, 真轮廓: np.ndarray, 容差: int = 2) -> dict:
    """想象（预期轮廓）vs 真轮廓：严格 IoU + 容差 IoU（同三轮 C8 口径）。"""
    想 = np.asarray(想象轮廓) > 0
    真 = np.asarray(真轮廓) > 0

    def _胀(m, 次):
        m = np.asarray(m, bool)
        for _ in range(int(次)):
            r = np.pad(m, 1, constant_values=False)
            m = (m | r[2:, 1:-1] | r[:-2, 1:-1] | r[1:-1, 2:] | r[1:-1, :-2])
        return m

    def _iou(a, b):
        u = float((a | b).sum())
        return round(float((a & b).sum()) / max(u, 1.0), 4)

    return dict(IoU=_iou(想, 真), 容差IoU=_iou(_胀(想, 容差), _胀(真, 容差)),
                交=float((想 & 真).sum()), 并=float((想 | 真).sum()), 容差=容差)


# ────────────────────────── 退化探针（真实桌面会遇到的）──────────────────────────
def 退化探针(补块, 种: int = 0) -> dict:
    """真图标退化族（确定性）：原样 / 缩放 / 平移 / JPEG / 噪声 / 模糊。

    平移用**补块自身边缘中位色**填边（不是黑边）——否则边界黑框会伪装成"形状剧变"，
    把数值压成假阴性（实测：填黑时平移臂 top1≈0.07，填自身底色后才是在测"位置抖动"）。
    这是"演示通道"的模拟退化，不冒充真机采样的全部真实扰动。
    """
    rgb = np.asarray(补块, np.uint8)
    H, W = rgb.shape[:2]
    边 = np.concatenate([rgb[0, :], rgb[-1, :], rgb[:, 0], rgb[:, -1]], 0).astype(np.float32)
    背 = np.median(边, 0)
    out = {"原样": rgb}
    for 尺 in (0.60, 0.80, 1.25, 1.50):
        n = max(8, int(round(W * 尺)))
        out["缩%s" % 尺] = cv2.resize(rgb, (n, max(8, int(round(H * 尺)))),
                                     interpolation=cv2.INTER_AREA if 尺 < 1 else cv2.INTER_LINEAR)
    for 移 in (-3, 3):
        M = np.array([[1.0, 0.0, float(移)], [0.0, 1.0, float(移)]], np.float32)
        out["移%d" % 移] = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=tuple(float(x) for x in 背))
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    out["jpeg60"] = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    rng = np.random.default_rng(3800 + int(种))
    out["噪5"] = np.clip(rgb.astype(np.float32) + rng.normal(0, 5.0, rgb.shape), 0, 255).astype(np.uint8)
    out["糊1"] = cv2.GaussianBlur(rgb, (0, 0), 1.0)
    return out


def 边缘码(补块, 编码=None, 用色差: bool = True) -> dict:
    """一条图形补块 → 三种表示（全部确定性）。"""
    E = 边缘色差图(补块) if 用色差 else 边缘图(补块)
    轮 = 轮廓图(E, "边缘")
    return dict(边缘=E, 轮廓=轮,
                码972=码_972(E),                      # 972 维（三轮律，实数单位向量）
                束=码_形状束(E, 编码))                # 1024 维（既有 形状束 律，双极）


def 色差度(补块) -> float:
    """补块内彩度（人眼核对"颜色在真图标里有多重要"，不作为身份特征）。"""
    rgb = np.asarray(补块, np.float32)
    if rgb.ndim < 3:
        return 0.0
    hsv = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    return round(float(np.percentile(hsv[:, :, 1], 90)), 1)
