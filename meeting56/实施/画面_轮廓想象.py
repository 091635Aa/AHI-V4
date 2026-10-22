# -*- coding: utf-8 -*-
"""画面_轮廓想象.py —— meeting56 三轮：**轮廓型想象画面**（人脑式：边缘/虚线/碎片/虚影）。

【为什么重做（所有者 2026-10-18 原话，语音转写）】
  「人脑的想象画面是边框啊之类的一些虚线啊之类的……它的画面不应该是看到什么就有什么东西，然后再映射……
   哪怕是一个非常模糊的虚影，或者是边框，或者是一些碎片，它至少来说是要看到一些东西……
   比如说看到电脑画面文件夹，他至少来说能幻想到文件夹一些轮廓之类的……你这个彩色方块我没有看出有什么联想性……
   这个视觉幻想我是不满意的，至少来说对他有用的思考，对他有用才行。」
  ⇒ 上一版（`画面_稠密场`）修的是**视网膜缓冲直读**：忠实但只是"重放像素"，不是想象；
    彩色方块也没有**联想性**（说不出"这是什么"）、更没**用途**（不驱动思考/行动）。

【本文件做什么】
  想象画面 = **形状原型（prototype）驱动的轮廓示意图**，不是像素照片：
    1) 从看到的图标补块（框±6px）算 **三阈值结构图**（μ±0.6σ, μ —— 与 `形状束.py` 同律，色无关）；
    2) **轮廓图** = 三层二值图的**边界**（区域∈、4 邻域∉）—— 正是人脑想象里那种"边框线"；
    3) **想象** = 从记忆里的轮廓图重画：按"生动度"参数只保留一部分轮廓像素（**虚线/碎片/虚影**）；
       位置来自记忆（空间参考帧），不是像素投影；
    4) **识别**：用轮廓图的池化码（972 维）在库中最近邻 → "这是什么"（联想性）；
    5) **有用**：输出**身份 + 位置 + 去那里**的计划（想象 → 想清楚 → 行动），而不是一张图。
  全部确定性（固定种子、无 cv2 也能跑）、零训练、零外挂模型、非 Transformer。

【验证口径（修上一轮的错误）】
  不用 PSNR 当主判据（PSNR 只配叫"像素像不像"，不配叫"想得对不对"）。
  主判据 = **①认得出来（top-1）②碎片下还认得（生动度扫描）③认得出还能用（位置/目标）④反例会失败**。

铁律：C1 禁 Transformer ｜ C9 禁 OCR ｜ C10 禁外挂模型当器官 ｜ C13 确定性 ｜ C14 不破既有。
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

格X, 格Y = 16, 9
图标底 = 40.0              # 图标画布底色（灰度）
框边距 = 6                 # 图标框外扩（让轮廓包含边界）
缩放 = 64                  # 与 形状束.py 同：统一 64×64
活力度 = 0.45              # 想象默认保留的轮廓像素比例（"虚影/虚线"程度；越低越模糊）


# ────────────────────────── 基础：灰度 / 缩放 / PNG ──────────────────────────
def 灰度(帧) -> np.ndarray:
    a = np.asarray(帧, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] >= 3:
        return (0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2])
    return a


def _缩放(图, n: int = 缩放) -> np.ndarray:
    """确定性面积平均缩放（无 cv2 依赖；与 形状束 的 64×64 口径一致）。"""
    a = np.asarray(图, dtype=np.float32)
    h0, w0 = a.shape[:2]
    ys = np.linspace(0, h0, n + 1).astype(int)
    xs = np.linspace(0, w0, n + 1).astype(int)
    出 = np.zeros((n, n), np.float32)
    for j in range(n):
        for i in range(n):
            区 = a[ys[j]:max(ys[j + 1], ys[j] + 1), xs[i]:max(xs[i + 1], xs[i] + 1)]
            出[j, i] = 区.mean() if 区.size else 0.0
    return 出


def 写png(图: np.ndarray, 路径: str = None) -> str:
    a = np.asarray(图, dtype=np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    h, w, _ = a.shape
    原始 = b"".join(b"\x00" + a[y].tobytes() for y in range(h))

    def _块(typ: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + typ + data
                + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))
    数据 = (b"\x89PNG\r\n\x1a\n"
            + _块(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + _块(b"IDAT", zlib.compress(原始, 6))
            + _块(b"IEND", b""))
    if 路径:
        with open(路径, "wb") as f:
            f.write(数据)
        return 路径
    return ""


# ────────────────────────── 结构图 / 轮廓图 / 池化码 ──────────────────────────
def 结构图层s(灰64) -> list:
    """三阈值二值图（暗结构 / 中间域 / 亮结构），与 形状束.py 同律。"""
    μ, σ = float(灰64.mean()), float(灰64.std())
    层s = []
    for 阈 in (μ - 0.6 * σ, μ, μ + 0.6 * σ):
        层s.append((灰64 > 阈).astype(np.uint8))
    return 层s


def _边界(B: np.ndarray) -> np.ndarray:
    """区域边界：区域像素中"4 邻域不全在区域内"的那些（轮廓线；纯 numpy，不依赖 cv2）。"""
    内 = np.asarray(B).astype(bool)
    r = np.pad(内, 1, constant_values=False)
    邻全 = r[2:, 1:-1] & r[:-2, 1:-1] & r[1:-1, 2:] & r[1:-1, :-2]
    return (内 & ~邻全).astype(np.uint8)


def 轮廓图(灰64) -> np.ndarray:
    """三层结构图的边界并集 → 64×64 uint8 轮廓（0/1）。

    ★ 退化层要跳过：补块里背景占多数时，"暗结构"层（灰度 > μ−0.6σ）会把**整块**都算进区域，
      它的"边界"就是补块外框（实测：想象图里每个图标外面多出一个虚线大框——假轮廓）。
      所以某层若几乎全真/全假（占比 >0.97 或 <0.03）就不取边界；并清掉补块边框 1px。
    """
    out = np.zeros((缩放, 缩放), np.uint8)
    for B in 结构图层s(灰64):
        占比 = float(B.mean())
        if 占比 > 0.97 or 占比 < 0.03:
            continue
        out |= _边界(B)
    out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = 0       # 补块边框不算轮廓
    return out


def _池化(图, n: int) -> np.ndarray:
    h = 图.shape[0] // n
    return 图[:h * n, :h * n].reshape(n, h, n, h).mean(axis=(1, 3)).ravel()


def 结构码(图64) -> np.ndarray:
    """64×64 灰度/轮廓 → 972 维色无关池化码（与 形状束.py 同律：3 阈值 × 16/8/2 池化，层内去共模）。"""
    a = np.asarray(图64, dtype=np.float32)
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


def _乱序(码: np.ndarray, 种子: int) -> np.ndarray:
    """反例臂：同范数但打乱坐标（破坏形状信息、保留统计量）——必须认错。"""
    rng = np.random.default_rng(int(种子))
    return 码[rng.permutation(码.shape[0])]


# ────────────────────────── 图标库（原型 + 同形变体）──────────────────────────
def _线(图, x1, y1, x2, y2, v=235.0):
    n = int(max(abs(x2 - x1), abs(y2 - y1), 1))
    xs = np.linspace(x1, x2, n + 1).round().astype(int)
    ys = np.linspace(y1, y2, n + 1).round().astype(int)
    图[np.clip(ys, 0, 图.shape[0] - 1), np.clip(xs, 0, 图.shape[1] - 1)] = v


def _框(图, x, y, w, h, v=235.0):
    _线(图, x, y, x + w, y, v)
    _线(图, x, y + h, x + w, y + h, v)
    _线(图, x, y, x, y + h, v)
    _线(图, x + w, y, x + w, y + h, v)


def _圆(图, cx, cy, r, v=235.0):
    t = np.linspace(0, 2 * np.pi, max(24, int(4 * r)), endpoint=False)
    图[np.clip((cy + r * np.sin(t)).round().astype(int), 0, 图.shape[0] - 1),
      np.clip((cx + r * np.cos(t)).round().astype(int), 0, 图.shape[1] - 1)] = v


def 画图标(名: str, 图, x, y, w, h, v=235.0):
    """线画图标（同色/线画 = 形状是唯一身份，颜色不参与）。"""
    if 名 == "文件夹":
        _框(图, x, y + int(h * 0.22), w, h - int(h * 0.22), v)
        _线(图, x, y + int(h * 0.22), x + int(w * 0.45), y + int(h * 0.22), v)
        _线(图, x, y + int(h * 0.22), x, y, v)
        _线(图, x + int(w * 0.45), y + int(h * 0.22), x + int(w * 0.45), y, v)
        _线(图, x + int(w * 0.2), y + int(h * 0.6), x + int(w * 0.8), y + int(h * 0.6), v)
    elif 名 == "文件":
        _框(图, x, y, w, h, v)
        _线(图, x + int(w * 0.65), y, x + w, y + int(h * 0.3), v)
        _线(图, x + int(w * 0.65), y, x + int(w * 0.65), y + int(h * 0.3), v)
        _线(图, x + int(w * 0.65), y + int(h * 0.3), x + w, y + int(h * 0.3), v)
        _线(图, x + int(w * 0.2), y + int(h * 0.55), x + int(w * 0.8), y + int(h * 0.55), v)
    elif 名 == "窗口":
        _框(图, x, y, w, h, v)
        _线(图, x, y + int(h * 0.22), x + w, y + int(h * 0.22), v)
        for i in range(3):
            _圆(图, x + int(w * (0.12 + 0.1 * i)), y + int(h * 0.11), 2, v)
        _线(图, x + int(w * 0.15), y + int(h * 0.5), x + int(w * 0.85), y + int(h * 0.5), v)
    elif 名 == "箭头":
        _线(图, x, y + h // 2, x + int(w * 0.8), y + h // 2, v)
        _线(图, x + int(w * 0.8), y + h // 2, x + int(w * 0.55), y + int(h * 0.25), v)
        _线(图, x + int(w * 0.8), y + h // 2, x + int(w * 0.55), y + int(h * 0.75), v)
    elif 名 == "齿轮":
        _圆(图, x + w // 2, y + h // 2, min(w, h) // 3, v)
        cx, cy, r = x + w // 2, y + h // 2, min(w, h) // 3
        for k in range(8):
            t = 2 * np.pi * k / 8
            _线(图, cx + int(r * np.cos(t)), cy + int(r * np.sin(t)),
                cx + int((r + 5) * np.cos(t)), cy + int((r + 5) * np.sin(t)), v)


def 造图标屏(名s=("文件夹", "文件", "窗口", "箭头", "齿轮"),
             变体: int = 0, H: int = 144, W: int = 256, 种子: int = 2026) -> np.ndarray:
    """一行排开的线画图标屏（同色、背景暗、带噪声）。变体=1 时尺寸/位置/间隔不同（泛化用）。"""
    rng = np.random.default_rng(int(种子))
    图 = np.full((H, W), 图标底, np.float32)
    n = len(名s)
    for i, 名 in enumerate(名s):
        if 变体 == 0:
            w, h = (44, 34) if 名 != "箭头" else (44, 22)
            x = 10 + i * 50
            y = 55 if 名 != "箭头" else 64
        else:
            w, h = (36, 28) if 名 != "箭头" else (36, 18)
            x = 16 + i * 47
            y = 40 if 名 != "箭头" else 62
        画图标(名, 图, x, y, w, h)
    return 图 + rng.normal(0, 3, 图.shape).astype(np.float32)


# ────────────────────────── 轮廓想象核 ──────────────────────────
def _膨胀(掩: np.ndarray, 次: int = 2) -> np.ndarray:
    """4 邻域膨胀：把线画图标的稀疏线连成块（否则阈值后一条线会碎成几十个 1px 连通域）。"""
    m = np.asarray(掩).astype(bool)
    for _ in range(int(次)):
        r = np.pad(m, 1, constant_values=False)
        m = (m | r[2:, 1:-1] | r[:-2, 1:-1] | r[1:-1, 2:] | r[1:-1, :-2])
    return m


class 轮廓想象:
    def __init__(self):
        self.库: list = []          # 每个图标一条：dict(名, 框, 位置, 轮廓, 码, 屏)

    # 找图标：亮线掩码 → 膨胀 → 连通域 → 框（纯算术，确定性）
    def 取图标s(self, 帧) -> list:
        灰 = 灰度(帧)
        H, W = 灰.shape
        掩 = _膨胀(灰 > (float(np.median(灰)) + 30.0), 2)
        见 = np.zeros_like(掩, bool)
        框s = []
        for y in range(H):
            for x in range(W):
                if not 掩[y, x] or 见[y, x]:
                    continue
                队 = [(y, x)]
                见[y, x] = True
                ys, xs = [], []
                while 队:
                    cy, cx = 队.pop()
                    ys.append(cy)
                    xs.append(cx)
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and 掩[ny, nx] and not 见[ny, nx]:
                            见[ny, nx] = True
                            队.append((ny, nx))
                y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
                if (y1 - y0) < 8 or (x1 - x0) < 8:
                    continue
                框s.append((x0, y0, x1 - x0 + 1, y1 - y0 + 1))
        return sorted(框s, key=lambda b: (b[0], b[1]))

    def 取补块(self, 帧, 框) -> np.ndarray:
        灰 = 灰度(帧)
        H, W = 灰.shape
        x0, y0, w, h = 框
        xa, ya = max(0, x0 - 框边距), max(0, y0 - 框边距)
        xb, yb = min(W, x0 + w + 框边距), min(H, y0 + h + 框边距)
        return _缩放(灰[ya:yb, xa:xb], 缩放)

    def 记一屏(self, 帧, 屏号: int = 0, 名s=None) -> dict:
        灰 = 灰度(帧)
        框s = self.取图标s(帧)
        条s = []
        for i, b in enumerate(框s):
            补 = self.取补块(帧, b)
            轮 = 轮廓图(补)
            码 = 结构码(轮.astype(np.float32) * 255.0)
            位置 = (int((b[0] + b[2] / 2) / (灰.shape[1] / 格X)),
                    int((b[1] + b[3] / 2) / (灰.shape[0] / 格Y)))
            条s.append(dict(框=tuple(b), 位置=位置, 补=补, 轮廓=轮, 码=码, 屏=屏号,
                            名=(名s[i] if 名s is not None and i < len(名s) else None)))
        self.库.extend(条s)
        return dict(记=True, 图标数=len(条s), 条s=条s)

    # 想象：从记忆里取第 i 条 → 按"生动度"保留轮廓像素 → 画成点线/虚影
    def 想象(self, 下标: int, 生动度_: float = 活力度, 位移=None,
             画布=None, 路径: str = None) -> dict:
        if not self.库:
            return dict(想象=False, 原因="库为空")
        条 = self.库[int(下标)]
        轮 = 条["轮廓"].astype(np.float32).ravel()
        稀疏 = self._稀疏(轮, 生动度_)
        x0, y0, w, h = 条["框"]
        if 位移:
            x0 += int(位移[0] * ((画布.shape[1] / 格X) if 画布 is not None else 16))
            y0 += int(位移[1] * ((画布.shape[0] / 格Y) if 画布 is not None else 16))
        if 画布 is None:
            raise ValueError("需要画布")
        # 生动度越低 → 越淡（"虚影"）：保留像素亮度乘 0.55~1.0
        淡 = 0.55 + 0.45 * float(生动度_)
        self._贴(画布, 稀疏.reshape(缩放, 缩放) * 淡, x0 - 框边距, y0 - 框边距, w + 2 * 框边距, h + 2 * 框边距)
        if 路径:
            写png(画布, 路径)
        return dict(想象=True, 路径=路径, 生动度=生动度_, 稀疏=稀疏.reshape(缩放, 缩放),
                    保留像素=int((稀疏 > 0).sum()), 总数=int((轮 > 0).sum()),
                    条=条, 画布=画布)

    def _稀疏(self, 轮: np.ndarray, 生动度_: float) -> np.ndarray:
        """按生动度保留轮廓像素：固定种子随机子集 ⇒ 散点/碎片（人想象的虚线感），确定性。
        生动度=1.0 全轮廓；0.4 碎片但骨架还在；0.1 只剩零星点。"""
        有效 = np.flatnonzero(轮 > 0)
        k = int(round(float(生动度_) * len(有效)))
        乱 = np.random.default_rng(5601).permutation(len(有效))   # 固定种子 ⇒ 同输入同输出
        留 = np.zeros(轮.shape[0], np.float32)
        留[有效[乱[:max(0, min(k, len(有效)))]]] = 1.0
        return 留 * 轮

    def _贴(self, 画布, 图64, x0, y0, w, h):
        H, W = 画布.shape[:2]
        ys = np.linspace(0, 缩放, max(h, 1) + 1).astype(int)
        xs = np.linspace(0, 缩放, max(w, 1) + 1).astype(int)
        for j in range(max(h, 1)):
            for i in range(max(w, 1)):
                区 = 图64[ys[j]:max(ys[j + 1], ys[j] + 1), xs[i]:max(xs[i + 1], xs[i] + 1)]
                if 区.size and 区.mean() > 0.02:
                    yy, xx = y0 + j, x0 + i
                    if 0 <= yy < H and 0 <= xx < W:
                        画布[yy, xx] = min(255, int(120 + 135 * float(区.mean())))

    # 识别：想象图（稀疏轮廓）→ 972 码 → 库最近邻（"这是什么"）
    def 识别(self, 稀疏图64, 反例种子: int = None) -> dict:
        if not self.库:
            return dict(认得=False, 原因="库为空")
        码 = 结构码(np.asarray(稀疏图64, np.float32) * 255.0)
        if 反例种子 is not None:
            码 = _乱序(码, 反例种子)
        拟 = [(float(码 @ 条["码"]), i) for i, 条 in enumerate(self.库)]
        拟.sort(key=lambda t: (-t[0], t[1]))
        最佳, i = 拟[0]
        条 = self.库[i]
        return dict(认得=True, 下标=i, 位置=条["位置"], 框=条["框"],
                    相似=round(float(最佳), 4),
                    次佳=round(float(拟[1][0]), 4) if len(拟) > 1 else None,
                    名=条.get("名"))


def 空画布(H: int = 144, W: int = 256, 底: float = 图标底) -> np.ndarray:
    return np.full((H, W), 底, np.float32)


def 贴真值屏的轮廓(帧, 路径: str = None) -> np.ndarray:
    """诊断用：把整屏亮线（阈值化）画出来——人眼对照"真值轮廓"。"""
    灰 = 灰度(帧)
    出 = np.full(灰.shape, 28.0, np.float32)
    出[灰 > float(np.median(灰)) + 30.0] = 235.0
    写png(出, 路径)
    return 出
