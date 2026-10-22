# -*- coding: utf-8 -*-
"""画面_稠密场.py —— meeting56：按 `会议_视觉` 的「保拓扑视场」修"画面"。

【为什么修】
  既有 [`内核_画面`](../meeting39/实施/内核_画面.py) 走的是**稀疏件列表**：每瞥最多 2 件、
  全屏上限 16 件，件的特征只是**中心 2×2（原始 4×4 px）光适应块**。实测（meeting56 诊断）：
    · 合成屏 7 个色块 → 16 件里 **9 件落在纯背景格**（σ≈2.5–3.2，纯传感器噪声）；
    · 件的特征只携带**色调**，携带不了形状（32×32 方块 / 横条 / 竖条在中心 4×4 上同色同结构）；
    · 成像把"归一化结构均值×60+128"映射成 RGB ⇒ 颜色是被去均值/归一化后的假色。
  这正是 [`会议_视觉/00_会议纪要.md`](../meeting38/会议_视觉/00_会议纪要.md) §0/§3 说的错位：
  **专家裁定"像人一样看"的产物是 P1 保拓扑视场 + 中央凹/外周分辨率差，不是离散件列表**；
  且 D3 裁定"原始像素块直接进 `特征码`、坐标由位置绑定承担"。

【本文件做什么（只新增、默认关、不改任何既有件）】
  · 视场 = 16×9 格**保拓扑稠密缓冲**：每格存**原始像素缩略块**（外周 8×8，中央凹 3×3 用 16×16）；
  · 认出/想象的身份路径另存**结构码**（灰度轮廓的二值池化，色无关；用于"认得出是什么形状"）；
  · 成像 = 把缓冲块按格拼回（8×8/16×16 → 16×16 → 放大），**不反演语义码**（专家：不做像素重建、
    颜色恒常不可——这里给的是"视网膜缓冲直读"，不冒充恒常/重建）；
  · 想象 = 只搬**结构件**、背景留原样（组合，不是整图平移）；碰撞按能量序确定性避让；
  · 全部零训练、零外挂模型、零 OCR、非 Transformer；固定种子 ⇒ C13 两趟逐位同。

【与专家的对齐（逐条）】
  P1 保拓扑场（会议纪要 §3.2）→ 稠密格缓冲，位置就是格键（保拓扑）。
  中央凹高分辨 + 外周低分辨（§0/§3.3）→ 外周 8×8、中央凹 3×3 为 16×16。
  D3 原始像素块直接进特征码（§2）→ 结构件仍从原始块算；缓冲保存原始块用于成像。
  诚实边界"颜色恒常不可/不做像素重建"（§6.5）→ 本文件不宣称恒常；成像只直读缓冲。
  不看物体列表（张力 B）→ 成像与想象都不依赖"件列表"；件列表只用于"搬哪些结构"。

铁律：C1 禁 Transformer ｜ C9 禁 OCR ｜ C10 禁外挂模型当器官 ｜ C13 确定性 ｜ C14 不破既有。
"""
from __future__ import annotations

import os
import struct
import zlib

import numpy as np

try:                                   # 与 形状束/器官_眼 同环境（项目 venv 已有 cv2）
    import cv2
except Exception:                      # 没有 cv2 时退化为最近邻（仍确定性）
    cv2 = None

格X, 格Y = 16, 9          # 与 视场前端/画面核 同网格
外周块 = 8                # 外周每格 8×8 RGB（低分辨）
中央凹块 = 16             # 中央凹每格 16×16 RGB（高分辨；单格全分辨率）
中央凹半径 = 1            # 3×3 中央凹
结构阈 = 0.03             # 块"是件"的门：max(块内标准差/255, 与**全屏背景中位色**的色差/255/√3)
                          #   —— 纯灰底噪声 ≈0.012 < 阈；彩色平块靠"块内跨通道差"过阈；
#                            白/黑平块没有块内差、但与背景有**颜色差** ⇒ 也过阈。
#                            用"背景中位色"而不是"4 邻域差"：否则物件**旁边**的背景格也会因
#                            与物件相邻而被判成件（实测会多搬一圈背景）。
件上限 = 32               # 结构件上限（只是"哪些要搬"，不限制成像）
来路_实感, 来路_回忆, 来路_想象 = "实感", "回忆", "想象"


def _池块(区, n: int) -> np.ndarray:
    """与 视场前端._池到 同律：把格区均值池化成 n×n（不依赖 cv2，确定性）。
    区尺寸不整除时先按最近邻裁到整倍数（格边最多差 1px）。"""
    a = np.asarray(区, dtype=np.float32)
    h0, w0 = a.shape[:2]
    h, w = (h0 // n) * n, (w0 // n) * n
    if h < n or w < n:
        return _缩放(a, n, n)
    a = a[:h, :w]
    return a.reshape(n, h // n, n, w // n, -1).mean(axis=(1, 3))


def _缩放(图, w: int, h: int) -> np.ndarray:
    """确定性缩放（优先 INTER_AREA；无 cv2 时按行/列均值分块）。"""
    a = np.asarray(图)
    if cv2 is not None:
        return cv2.resize(a, (int(w), int(h)), interpolation=cv2.INTER_AREA)
    h0, w0 = a.shape[:2]
    ys = (np.linspace(0, h0, int(h) + 1)).astype(int)
    xs = (np.linspace(0, w0, int(w) + 1)).astype(int)
    出 = np.zeros((int(h), int(w), a.shape[2]), a.dtype)
    for j in range(int(h)):
        for i in range(int(w)):
            区 = a[ys[j]:max(ys[j + 1], ys[j] + 1), xs[i]:max(xs[i + 1], xs[i] + 1)]
            出[j, i] = 区.reshape(-1, a.shape[2]).mean(0)
    return 出


def 形状描述(patch) -> np.ndarray:
    """色无关形状描述（972 维，与 `形状束.py` 同律：灰度→μ±0.6σ 三阈→16/8/2 三档池化→层内去共模）。

    与 `形状束.py` 的差别：本函数**不**再喂 `编码.特征码`（那是 D 维双极、不可逆），
    这里保留 972 维实数向量直接算余弦，用于验证"方块/横条/竖条可分"。
    输入应是**含背景边界的件补块**（框±6px），这样轮廓才在 patch 里；纯平色块 central 4×4
    是分不出形状的（这正是旧画面核的病灶）。
    """
    p = np.asarray(patch)
    if p.ndim == 3 and p.shape[2] >= 3:
        灰 = (0.299 * p[:, :, 0] + 0.587 * p[:, :, 1] + 0.114 * p[:, :, 2]).astype(np.float32)
    else:
        灰 = np.asarray(p, dtype=np.float32)
    if 灰.shape[0] < 12 or 灰.shape[1] < 12:
        return None
    灰 = _缩放(灰, 64, 64)
    μ, σ = float(灰.mean()), float(灰.std())
    if σ < 4.0:
        return None
    特征 = []
    for 阈 in (μ - 0.6 * σ, μ, μ + 0.6 * σ):
        二值 = (灰 > 阈).astype(np.float32)
        for n in (16, 8, 2):
            v = _池块(二值, n).reshape(-1).astype(np.float32)
            v = v - v.mean()
            特征 += v.ravel().tolist()
    v = np.asarray(特征, dtype=np.float32)
    return v / (float(np.linalg.norm(v)) + 1e-12)


def _结构度(块) -> float:
    """块**内部**结构度 = 块内跨通道+空间标准差 /255（色块有、纯平块无）。"""
    return float(np.asarray(块, dtype=np.float32).std() / 255.0)


def 结构度s(块s: dict, 格x: int = 格X, 格y: int = 格Y) -> dict:
    """每格"是件"的程度 = max(块内结构度, 与**全屏背景中位色**的色差/255/√3)。

    为什么要加"背景色差"：白/黑**平块**块内无差异（σ≈0），但相对背景是孤立物；
    只看块内差会把它们判成"无结构"（实测：白条/黑条搬家时没被搬）。
    为什么不用"与4邻域差"：那样物件**旁边**的背景格会因为与物件相邻而被判成件，
    想象时会把背景也搬走（实测）。背景 = 全屏块均值的逐通道中位数（确定性）。
    """
    均 = {k: np.asarray(v, dtype=np.float32).reshape(-1, 3).mean(0) for k, v in 块s.items()}
    背 = np.median(np.stack(list(均.values()), 0), axis=0) if 均 else np.zeros(3, np.float32)
    出 = {}
    for k, v in 块s.items():
        差 = float(np.linalg.norm(均[k] - 背)) / 255.0 / np.sqrt(3.0)
        出[k] = max(_结构度(v), 差)
    return 出


def _写png(图: np.ndarray, 路径: str = None) -> str:
    """最小 PNG 编码（zlib + 手写块；零第三方依赖，同 内核_画面._写png）。"""
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


class 稠密画面:
    """保拓扑视场缓冲 + 结构件 + 成像/想象（meeting56 修复件；默认不接任何主循环）。"""

    def __init__(self, 格x: int = 格X, 格y: int = 格Y, 外周: int = 外周块,
                 凹块: int = 中央凹块, 凹半径: int = 中央凹半径):
        self.格x, self.格y = int(格x), int(格y)
        self.外周, self.凹块 = int(外周), int(凹块)
        self.凹半径 = int(凹半径)
        self.库: list = []

    # ── 取格：外周低分辨 + 中央凹高分辨（保拓扑）──
    def 取格s(self, 帧, 注视=None) -> dict:
        F = np.asarray(帧)
        if F.ndim != 3 or F.shape[2] < 3:
            return {}
        H, W = F.shape[:2]
        gx, gy = (self.格x // 2, self.格y // 2) if 注视 is None else (int(注视[0]), int(注视[1]))
        块s = {}
        for by in range(self.格y):
            y0 = int(by * H / self.格y)
            y1 = max(int((by + 1) * H / self.格y), y0 + 1)
            for bx in range(self.格x):
                x0 = int(bx * W / self.格x)
                x1 = max(int((bx + 1) * W / self.格x), x0 + 1)
                区 = F[y0:y1, x0:x1, :3]
                凹 = abs(bx - gx) <= self.凹半径 and abs(by - gy) <= self.凹半径
                n = self.凹块 if 凹 else self.外周
                # 与 视场前端._池到 同律（均值池化）：原始像素块直接进"眼"，不经手工特征
                块s[(bx, by)] = _池块(区, n).clip(0, 255).astype(np.uint8)
        return 块s

    # ── 记一屏：缓冲 + 结构度 + 结构件（只有实感入库）──
    def 记一屏(self, 帧, 拍: int = -1, 注视=None, 来路: str = 来路_实感) -> dict:
        块s = self.取格s(帧, 注视)
        if not 块s:
            return dict(记=False, 原因="帧不可用")
        结构s = 结构度s(块s, self.格x, self.格y)
        件s = sorted([k for k, v in 结构s.items() if v >= 结构阈],
                     key=lambda k: (-结构s[k], k))[:int(件上限)]
        屏 = dict(拍=int(拍), 来路=str(来路), 注视=tuple(注视) if 注视 else None,
                  块s=块s, 结构s=结构s, 件s=件s,
                  结构格数=len(件s), 无结构格数=len(块s) - len(件s))
        入库 = (str(来路) == 来路_实感)
        if 入库:
            self.库.append(屏)
        return dict(记=True, 入库=入库, 屏=屏, 结构格数=len(件s),
                    无结构格数=len(块s) - len(件s), 件s=件s)

    # ── 成像：缓冲直读（外周块/凹块 → 每格 16×16 → 放大）──
    def 成像(self, 块s: dict, 放大: int = 2) -> np.ndarray:
        """缓冲直读：每格块（外周 8×8 / 凹 16×16）贴到 16×16 格位，再按 `放大` 整图放大。
        ★ 不做任何"码→图"反演：贴回去的就是当初池化下来的原始像素块（专家：颜色恒常不可，
          本文件不冒充重建）。放大用最近邻（np.repeat），避免把块内结构糊掉。"""
        格 = np.zeros((self.格y * 16, self.格x * 16, 3), np.uint8)
        for (bx, by), b in 块s.items():
            if not (0 <= bx < self.格x and 0 <= by < self.格y):
                continue
            b = np.asarray(b, dtype=np.uint8)
            if b.shape[0] == 16 and b.shape[1] == 16:
                p = b
            elif b.shape[0] > 0 and 16 % b.shape[0] == 0 and 16 % b.shape[1] == 0:
                p = np.repeat(np.repeat(b, 16 // b.shape[0], 0), 16 // b.shape[1], 1)
            else:
                p = _缩放(b, 16, 16).astype(np.uint8)
            格[by * 16:(by + 1) * 16, bx * 16:(bx + 1) * 16] = p
        if int(放大) > 1:
            格 = np.repeat(np.repeat(格, int(放大), 0), int(放大), 1)
        return 格

    def 回放(self, 屏下标: int = -1, 路径: str = None, 放大: int = 2) -> dict:
        if not self.库:
            return dict(回放=False, 原因="画面库为空")
        屏 = self.库[int(屏下标)]
        图 = self.成像(屏["块s"], 放大=放大)
        p = _写png(图, 路径)
        return dict(回放=True, 路径=p, 屏=屏, 图=图, 来路=来路_回忆)

    def _簇s(self, 件s) -> list:
        """把结构件按 4 邻域连通分成"物件"（确定性：格序 BFS）。"""
        件 = set(件s)
        见, 簇s = set(), []
        for k in sorted(件):
            if k in 见:
                continue
            队, 簇 = [k], []
            见.add(k)
            while 队:
                c = 队.pop(0)
                簇.append(c)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    j = (c[0] + dx, c[1] + dy)
                    if j in 件 and j not in 见:
                        见.add(j)
                        队.append(j)
            簇s.append(sorted(簇))
        return 簇s

    # ── 想象：只搬结构件、背景留原样（组合不是整图平移）──
    def 想象(self, 屏下标: int = -1, 位移=(3, 2), 路径: str = None, 放大: int = 2) -> dict:
        """把全部结构件**整组**搬到新格位（背景不动）：像"把记忆里的东西挪个位置"。

        为什么不逐件独立搬：独立搬会互相撞格，实测 7 个物件只能搬走 5 个（画面里两个没动，
        看起来像"想象失败"）；整组平移保持物件间相对布局，且**全部**物件都换格位。
        为什么不用环面：整图环面会把物件撕成两半贴在边上（人眼一看就"画面坏了"）；
        这里按**整组反弹**：原位移出界就试水平/垂直反向，取第一个整组不越界的位移（确定性）。
        """
        if not self.库:
            return dict(想象=False, 原因="画面库为空")
        屏 = self.库[int(屏下标)]
        dx, dy = int(位移[0]), int(位移[1])
        if dx == 0 and dy == 0:
            return dict(想象=False, 原因="零位移=回放")
        件 = list(屏["件s"])
        # 可行位移 = 整组都不越界；取"离要求位移最近"的那个（确定性：距离→|dx|→|dy|→dx→dy）
        可行 = []
        for ddx in range(-self.格x + 1, self.格x):
            for ddy in range(-self.格y + 1, self.格y):
                if (ddx, ddy) == (0, 0):
                    continue
                if all(0 <= c[0] + ddx < self.格x and 0 <= c[1] + ddy < self.格y for c in 件):
                    可行.append((ddx, ddy))
        if not 可行:
            return dict(想象=False, 原因="整组无处可搬")
        用 = min(可行, key=lambda d: ((d[0] - dx) ** 2 + (d[1] - dy) ** 2,
                                       abs(d[0]), abs(d[1]), d[0], d[1]))
        ddx, ddy = 用
        新件 = {(c[0] + ddx, c[1] + ddy): 屏["块s"][c] for c in 件}
        中位 = np.median(np.concatenate([b.reshape(-1, 3) for b in 屏["块s"].values()]), axis=0)
        新块s = dict(屏["块s"])
        for c in 件:
            原 = 屏["块s"][c]
            n = 原.shape[0]
            新块s[c] = np.tile(np.clip(中位, 0, 255).astype(np.uint8)[None, None, :], (n, n, 1))
        for t, b in 新件.items():
            新块s[t] = b
        新组合 = sum(1 for t in 新件 if t not in set(件))
        图 = self.成像(新块s, 放大=放大)
        p = _写png(图, 路径)
        return dict(想象=True, 路径=p, 图=图, 块s=新块s, 新件s=list(新件),
                    物件数=len(self._簇s(件)), 搬动物件数=len(self._簇s(件)),
                    位移实际=(ddx, ddy), 新组合格数=int(新组合),
                    位移=(dx, dy), 来路=来路_想象)

    # ── 比真值：成像忠实度（PSNR/MAE）──
    def 比对(self, 屏下标: int, 帧, 放大: int = 1) -> dict:
        真 = _缩放(np.asarray(帧)[:, :, :3], self.格x * 16, self.格y * 16)
        真 = np.repeat(np.repeat(真.astype(np.uint8), 放大, 0), 放大, 1) if 放大 > 1 else 真
        图 = self.成像(self.库[int(屏下标)]["块s"], 放大=放大)
        mse = float(np.mean((真.astype(np.float64) - 图.astype(np.float64)) ** 2))
        psnr = float(10.0 * np.log10(255.0 ** 2 / max(mse, 1e-12)))
        mae = float(np.mean(np.abs(真.astype(np.float64) - 图.astype(np.float64))))
        return dict(shape=tuple(图.shape), PSNR=round(psnr, 2), MAE=round(mae, 2))
