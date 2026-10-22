# -*- coding: utf-8 -*-
"""内核_画面.py —— meeting39 步8：视觉想象 / 画面（第六层内核；默认关的保护开关）。

【为什么有这个文件】所有者点名：「脑中想出画面」没有实现——内部独白已经有了
（`自省文()`/`_近期自语()` 可取），但**画面**属内核层新增，不是接线能解决的。

【它是什么】把「看见的一屏」压成**多瞥的纯 HDC 码**，再从这些码里**读出画面**——
读出来的不是"重新看一眼屏幕"，而是**从记忆里重新生成**的那一屏；
并且可以把记忆里的东西搬到**它从未待过的格位**上（组合想象）。
全部零梯度、零外挂模型、零 OCR、非 Transformer。

────────────────────────── 编码（纯 HDC） ──────────────────────────
  · 一屏 → 格网（16×9）→ 分 **12 个"瞥"**（4×3 分区；一次注视装不下整屏，就分多次注视）。
  · 每格取 2×2×3 = 12 维「光适应块」（减块均值/除块标准差：消全局明暗，只留局部结构）。
  · 每瞥取最显著的 ≤2 个格当"件"（同瞥内**去冗余**：与已选件码似 ≥ `件编码阈` 视为同一件）。
  · 物体码 S = sign(√F · P · x̂) ∈ {±1}^D，x̂ 是**单位化**后的 12 维块；
    P ∈ {±1/√D}^{D×F} 固定随机投影（不训练）⇒ **逆投影 = Pᵀ(·)/√D**。
    ★这条逆的尺度是我**量错过的**：写成 `Pᵀ/D` 会让重建小 √D≈32 倍，看起来像"保真只有 0.001"。
  · 件项 = 位置原子 ⊛ 物体码（位置原子 ∈ {±1}^D，格网预注册 ＝ 空间帧；格网外按 CRC32 派生）。
  · 每瞥一份场码：场 = Σ件项 / √件数（**不**归一到单位范数——否则搬家后注入系数变成 1，
    场里混进别的东西就被压死，实测"16 件搬家后 0 件认得出"）。
  · 画面库 = 内容寻址（每屏：多瞥场码 + 位置地址表 + 瞥号表 + 库物体码/外观）。
  · `W画`（场↔场 联想阵）**默认不写**：实测它不在读出路径上，且归一到单位 Frobenius 后
    联想读出失效（`v⊙Wv ≈ 0.002·v`，去噪负增益）⇒ 保留为实验臂，见模块头「为什么关」。

────────────────────────── 读出（位置当"地址"） ──────────────────────────
  对位置 p（并指定它该用哪一瞥的场）：残 = 场 ⊛ 丄_p；反投影 Pᵀ·残/√D = "那一块的样子"；
  再**重新编码**成码，与库码逐位比 → `码域相似 = mean(S_i ⊙ sign(P·重建)) ∈ [−1,1]`。
  · 为什么不用"归一化 cos"：残里的**噪声分量在归一化时被放大**，真屏上空位也能到 0.89
    （真位 0.98）⇒ 判据贴脸；码域用的是**未归一**的重建，噪声不放大。
  · ★位置**不从码里猜**（预检⑮：所有位置的 ‖残‖ 恒等 ⇒ 位置不在场的模里）：
    它由**存下来的地址表**给出，正如空间参考帧由身体给出、不靠视网膜自己发明。
  · 判据 `判据_检测 = 0.78`：真屏实测真位 均 0.889 / 最低 0.795；空位 最高 0.709 / 均 0.184。

────────────────────────── 想象（组合，不是回放） ──────────────────────────
  `想象码(位移)`：把屏里每个件**环面平移**到新格位（不 clip——clip 会让靠边件挤到同格），
  按**它当初被编码时所在的瞥**重组新场码，再用同一读出路径解出来。
  判定"这是想象不是回放"要**两条同时成立**：① 件还认得出来（码似 ≥ 判据）；
  ② 该件确实换了格位（新组合格数 > 0）。缺一即回放或幻觉。

────────────────────────── 铁律与自我指涉 ──────────────────────────
  · C1 禁 Transformer｜C9 禁 OCR｜C10 禁外挂模型｜C13 确定性（固定种子 PRNG，无全局随机）。
  · 本文件**不 import 任何器官**（眼/手/耳/嘴/语义/我 一律冻结）：眼睛由 V3 侧 `设帧源()` 接入。
    唯一覆盖点 = 综合体._引擎步（只加挂）。`开关默认=False` ⇒ 关时逐位恒等既有。
  · 记忆 ≠ 学习：画面库/迹与 `codec._段` **物理分离**（本文件不写记忆库）。
  · provenance：`来路 ∈ {实感, 回忆, 想象}`。**只有 实感 进画面库**——想象/回忆若入库，
    "我看见过的"就会被自己编的东西污染（与自我回路 J5「外源不可塑造我」同一条纪律）。
    想象喂自模型时也默认 `更新=False` ⇒ 自编的东西不塑造自己。

【实测（AHI3 第五轮；预检与实测脚本都在 `测试记录/`，可复跑）】
  | 量 | 结果 |
  |---|---|
  | 全网格逐格编码（144 格挤一份码） | 逐格 cos 0.45~0.65 ⇒ **粒度错**（每格预算仅 ~7 维） |
  | 物体级稀疏码 K=4 | 逐件保真 **0.94~0.99** ⇒ 正确的粒度 |
  | 残的**范数**当"哪里有东西" | 恒 = √D（双极码 ⇒ 零信息）——我第一版拿它当判据，**错**，已弃 |
  | 真屏（1080p）16 件 | 回放保真 **0.981**、身份 **16/16**、空位码似 ≤0.71（判据 0.78） |
  | 组合想象（环面搬家 +3,+2） | **16/16 仍认得出**（码似最低 0.83） |
  | `记一屏` 耗时 | 503ms → **21.6ms**（批量编码；并默认不再更新 1024² 的 W画） |

纪律：C1 禁 Transformer｜C9 禁 OCR｜C10 禁外挂模型当器官｜C13 确定性可复现｜C14 不破既有。
用法（测试）：见 `AHI3/meeting38/实施/test_模块六_画面.py`；演示：`V3主程序.py --画面`。
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from 内核_顿悟涌现 import 顶覆盖_综合体W                      # noqa: E402 （步6 宿主；六层链 H←W←Z←L←T←基）
from 内核_自我回路 import 来源标_自源, 来源标_外源              # noqa: E402
from 内核_自模型 import ETA_R                                  # noqa: E402

开关默认 = False            # ★保护开关默认关（C14/D1：默认关 ⇒ 既有输出逐位恒等）
η_画 = ETA_R                # Hebbian 步长 = 既有 readout_head ETA_R（#152 冻结；零新超参）
上限范数_画 = 1.0           # ★画面联想阵的范数上限（同 内核_自我回路.上限范数_self）
粗分辨率 = 2                # 每格 2×2×3 = 12 维光适应块（预检：t=2 时逐件保真 0.92~0.98）
维_特征 = 粗分辨率 * 粗分辨率 * 3
格X, 格Y = 16, 9            # 与 视场前端 同网格（144 格）
盯X, 盯Y = 4, 3             # ★分**瞥**（glimpse）网格：4×3=12 个瞥，每瞥一份独立场码
每瞥件 = 2                  # ★每瞥最多装 2 件（一次"注视"的容量）
件编码阈 = 0.70             # ★同一瞥内的**去冗余**门：与已选件的物体码相似 ≥ 此值 ⇒ 不重复装
#   ★为什么必须去冗余（真屏实测逼出来的）：真屏相邻格高度相关，一个 2×2 尺寸的"按钮"
#   会占满好几格，若把同瞥里最强的前 2 格都装进去，两个**几乎相同**的码束在一起会互相抵消
#   ⇒ 回放还勉强（保真 0.98），但**搬家后认不出来**（实测 16 件里 1 件）。判据取"相似 ≥ 0.7
#   视为同一件"：双极码相似 0.7 ≈ 同一实体在不同采样下的相似度（不同实体 ≈ 0）。
上限件 = 16                 # ★单屏总容量上限（预检⑯：K≤8 身份 8/8、保真 ≥0.86；K=16 保真 0.89）
# ★为什么必须分区：真屏**处处有结构**——"噪声阈"在真屏上完全失效（实测：σ 中位 0.098、
#   MAD 0.051 ⇒ 阈值 0.40 > 最大 0.37 ⇒ 选 0 件）。所以选件不能靠"绝对噪声底"，
#   只能靠**容量的硬约束**：一次注视装不下整屏，就分多次注视、每次装最显著的几件
#   （这正是"瞥"的定义）。分区是**架构**（视野被分成若干注视窗），不是人挑的语义阈值。
判据_检测 = 0.78            # ★"还认得出来"的门（**码域相似**口径，见 读出）：
                            #   真屏实测（1080p 桌面、16 件）：真位 均 0.889 / 最低 0.795；
                            #   空位 最高 0.709 / 均 0.184 ⇒ 0.78 落在两者之间，两侧都留余量。
                            #   （先前用"归一化 cos"当判据是错的：真位 0.98、空位 0.89 ⇒ 贴脸；见读出注释）
帧间隔s = 1.0               # 真帧采样间隔（默认 1 秒一屏；引擎思考拍也走 _引擎步，不设间隔会重复存同一屏）
帧变化阈 = 0.002            # 内容门：逐格绝对对比 σ 的**去共模**最大变化须超过它才算"变了一屏"
                            #   （2/255≈0.008 是 8bit 一个灰度级；0.002 ≪ 一格，取"非同一屏"下限）
写W画 = False               # ★是否维护画面联想阵 W画（第五轮**实测后关闭**，零新超参、零"背着不用"）
#   ——为什么关（`测试记录/_预检_画面联想阵.py` + `_诊断_画面联想阵.py` 可复跑）：
#     ① **它不在读出路径上**：画面是"episode 列表存场码"给的，W画 只是"场↔场"联想，回放/想象
#        都不经过它 ⇒ 每瞥 46.6ms × 9 瞥 ≈ **373ms/屏**（真屏实测记一屏 118ms→683ms），纯亏；
#     ② **它的联想读出被范数归一废掉**：W = Σ 场场ᵀ 归一到单位 Frobenius ⇒ ‖W·v‖ 比 ‖v‖ 小
#        23 倍 ⇒ v⊙(Wv) ≈ 0.0019·v，取符号后与真场 cos **0.002**；去噪实测"前 0.676 → 后 0.018"
#        （**负增益**，翻转 5%/15%/25% 全为负）；
#     ③ 若为了保联想而放弃范数归一，就回到"‖W‖ 随屏数无界"（第三轮 W_self 的同一条病）。
#   ⇒ 按纪律（不做"背着 1M 矩阵却没用"的设计）：**默认不写 W画**；`W画`/`更新W`/`联想去噪`/
#     `去噪自证` 保留为**实验臂**（`写W画=True` 可开，判据 N6-8 验"关时恒零"）。
来路_实感, 来路_回忆, 来路_想象 = "实感", "回忆", "想象"


def _双极(rng: np.random.Generator, n: int) -> np.ndarray:
    """确定性双极 ±1 向量（float64→float32）。"""
    return np.where(rng.random(int(n)) < 0.5, -1.0, 1.0).astype(np.float32)


def _cos(a, b) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def 光适应(块: np.ndarray) -> np.ndarray:
    """视网膜局部归一化：减块均值、除块标准差（消全局明暗，留局部结构）。
    与 `视场前端._光适应` 同律；全平块（σ≈0）返回全零（**不编码**——噪声不是画面）。"""
    b = np.asarray(块, dtype=np.float32) / 255.0
    b = b - float(b.mean())
    s = float(b.std())
    if s <= 1e-6:
        return np.zeros_like(b)
    return b / s


class 画面核:
    """视觉想象核：一屏 → 物体码 ⊛ 位置原子 → 场码；读出 = 地址解 × 库匹配。

    · 位置原子由**固定种子**PRNG 预注册（空间帧；同一性靠档案、不靠动力学）。
    · 画面库（episodes）内容寻址；W画 只在**实感**时 Hebbian 更新且有界（保方向、限幅度）。
    · 迹 = 独立画面痕迹存储（与 codec._段 物理分离；记忆≠学习）。
    """

    def __init__(self, D: int = 1024, 种子: int = 3800,
                 格x: int = 格X, 格y: int = 格Y, 写W画_: bool = None):
        self.D = int(D)
        self.种子 = int(种子)
        self.格x, self.格y = int(格x), int(格y)
        self.写W画 = bool(写W画 if 写W画_ is None else 写W画_)   # 默认关（见模块头）
        self.维_特征 = int(粗分辨率 * 粗分辨率 * 3)
        rng = np.random.default_rng(self.种子)
        # 空间帧：全部格位置原子（格序确定）
        self.位族 = {(bx, by): _双极(rng, self.D)
                     for by in range(self.格y) for bx in range(self.格x)}
        self.位序 = {k: i for i, k in enumerate(self.位族.keys())}
        # 固定随机投影 P ∈ {±1/√D}^{D×F}，列已单位化 ⇒ PᵀP = I ⇒ **逆投影 = Pᵀ(·)/√D**
        # （预检⑰ 实测：‖P·x‖≈1；把 "/D" 当逆是我的尺度错，会使重建小 √D≈32 倍 ⇒ "保真≈0" 假象）。
        # 不训练、不可变；这也是本文件不复用 `编码器.特征码` 的原因（后者按维度缓存、不可逆）。
        self.P = (_双极(np.random.default_rng(self.种子 + 1),
                        self.维_特征 * self.D) / np.sqrt(self.D)).reshape(self.D, self.维_特征)
        self.P逆 = self.P.T / np.sqrt(self.D)       # (F, D)：12 维"脑中那块" = P逆·残
        self.W画 = np.zeros((self.D, self.D), dtype=np.float64)   # 有界联想阵（仅实感写）
        self.库: list = []                    # 画面库：每屏 dict(场, 位s, 码s, 拍, 来路)
        self.迹: list = []                    # ★独立画面痕迹存储（读出/想象的账）
        self.实感次数 = 0
        self.想象次数 = 0

    # ══════════════════ 编码：一屏 → 物件（连通区）→ 场码 ══════════════════
    def 块s(self, 帧, 带对比: bool = False) -> dict:
        """帧 → {(bx,by): 12 维光适应块}（格网切块，确定性；全平块→全零）。

        `带对比=True` 时值 = (光适应特征, 绝对局部对比 σ)；σ 即 `视场前端.显著度场` 的同一量。
        """
        F = np.asarray(帧)
        if F.ndim != 3 or F.shape[2] < 3:
            return {}
        H, W = F.shape[0], F.shape[1]
        出 = {}
        for by in range(self.格y):
            y0, y1 = int(by * H / self.格y), int((by + 1) * H / self.格y)
            y1 = max(y1, y0 + 1)
            for bx in range(self.格x):
                x0, x1 = int(bx * W / self.格x), int((bx + 1) * W / self.格x)
                x1 = max(x1, x0 + 1)
                区 = F[y0:y1, x0:x1, :3].astype(np.float32)
                t = 粗分辨率
                h, w = 区.shape[:2]
                f = max(1, min(h // t, w // t))
                if f == 1:
                    p = 区[:t, :t]
                else:
                    h2, w2 = (h // f) * f, (w // f) * f
                    p = (区[:h2, :w2].reshape(h2 // f, f, w2 // f, f, 3)
                         .transpose(0, 2, 4, 1, 3).copy().mean(axis=(3, 4)))
                hh, ww = p.shape[:2]
                yy, xx = max(0, (hh - t) // 2), max(0, (ww - t) // 2)
                p = p[yy:yy + t, xx:xx + t]
                b = p / 255.0
                b = b - float(b.mean())
                σ = float(b.std())
                特 = (b / σ).reshape(-1) if σ > 1e-6 else np.zeros(t * t * 3, np.float32)
                出[(bx, by)] = (特, σ) if 带对比 else 特
        return 出

    def 特征码(self, 特征) -> np.ndarray:
        """12 维特征 → D 维双极物体码 sign(√F·P·x̂)（x̂ 先归一到单位范数）。

        ★ 归一化不是可选项：`P` 的列长为 1 但方向各向同性，sign(P·x) 的"随机程度"取决于
        ‖x‖——若不归一，亮块的码会比暗块"更随机"，同一个东西换个位置就认不出来。
        归一化后 sign(P·x̂) 是均匀随机双极码（与幅值无关 ⇒ 亮度不变性）。
        """
        x = np.asarray(特征, dtype=np.float32).ravel()
        if x.shape[0] != self.维_特征:
            return None
        n = float(np.linalg.norm(x))
        if n <= 1e-9:
            return None
        b = np.sign((self.P @ (x / n)) * np.sqrt(self.维_特征)).astype(np.float32)
        b[b == 0] = 1.0
        return b

    def 特征码批(self, 特征s) -> np.ndarray:
        """一批 12 维特征 → (N, D) 双极物体码。**一次 GEMM**（逐格调用会把 O(格) 次 matvec 变瓶颈）。"""
        X = np.asarray(特征s, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.维_特征:
            return np.zeros((0, self.D), dtype=np.float32)
        范 = np.linalg.norm(X, axis=1, keepdims=True)
        好 = (范.ravel() > 1e-9)
        if not bool(好.any()):
            return np.zeros((0, self.D), dtype=np.float32)
        Xn = X[好] / 范[好]
        B = np.sign((Xn @ self.P.T) * np.sqrt(self.维_特征)).astype(np.float32)
        B[B == 0] = 1.0
        return B

    def 特征值(self, 码) -> np.ndarray:
        """双极物体码 → 12 维特征（反投影 x̂ = Pᵀ·S/√D，即"脑中的那块"）。"""
        return (np.asarray(码, dtype=np.float32) @ self.P逆.T).astype(np.float32)

    def 分瞥(self) -> dict:
        """格键 → 瞥号（4×3 分区，确定性）。每个瞥是一份独立场码（一次"注视"）。"""
        出 = {}
        for by in range(self.格y):
            for bx in range(self.格x):
                出[(bx, by)] = (bx * 盯X // self.格x) + 盯X * (by * 盯Y // self.格y)
        return 出

    def 结构件s(self, 块带对比: dict) -> list:
        """一屏 → **物件**列表：分「瞥」后，每瞥取最显著的 `每瞥件` 个格当件。

        为什么这样切（真屏实测驱动的架构决定）：
          · 真屏处处有结构 ⇒ "绝对噪声阈"失效（实测阈值 0.40 > σ 最大 0.37 ⇒ 选 0 件）；
          · 容量是硬约束（D=1024 一次注视装不下整屏）⇒ 必须分多次注视、每次装几件；
          · 于是：**分区 = 架构**（视野分成 12 个注视窗），**每瞥取最显著者 = 容量分配**
            （每瞥 2 件 = 一次注视的清晰度上限），全局再按容量上限截断。
          件的**外观**仍取该格 12 维光适应块；件的**位置**=格键（空间帧里的一格）。
        """
        瞥s = self.分瞥()
        # ★性能（第五轮修正）：先把全屏 144 格**一次批量**编码（一次 GEMM），
        #   而不是在"选件 + 去冗余"里逐格调用 `特征码`（每格一次 matvec ⇒ 实测 503ms/屏）。
        格键s = [(bx, by) for by in range(self.格y) for bx in range(self.格x)]
        码表 = {}
        B = self.特征码批([块带对比[k][0] for k in 格键s])
        好 = [i for i in range(len(格键s))
              if float(np.linalg.norm(块带对比[格键s[i]][0])) > 1e-9]
        for j, i in enumerate(好):
            if j < B.shape[0]:
                码表[格键s[i]] = B[j]
        桶 = {}
        for k, (特, σ) in 块带对比.items():
            桶.setdefault(瞥s[k], []).append((k, float(σ)))
        件s = []
        for g in sorted(桶):
            桶[g].sort(key=lambda kv: (-kv[1], kv[0]))          # σ 降序、格序破平（确定性）
            选 = []
            for k, σ in 桶[g][:max(int(每瞥件) * 4, int(每瞥件))]:
                c = 码表.get(k)
                if c is None:
                    continue
                if any(float(np.mean(c * 已有)) >= float(件编码阈) for _k2, 已有 in 选):
                    continue                          # 与已选件"是同一件" ⇒ 不重复装
                选.append((k, c))
                if len(选) >= int(每瞥件):
                    break
            for k, c in 选:
                件s.append(dict(格s=[k], 位置=k, 能量=float(块带对比[k][1]),
                                外观=块带对比[k][0], 码=c, 瞥=g, 格数=1))
        # 全局截断：按能量降序、格序破平，取前 上限件
        件s.sort(key=lambda e: (-e["能量"], e["位置"]))
        return 件s[:int(上限件)]

    # ══════════════════ 记一屏（只有实感入库） ══════════════════
    def 记一屏(self, 帧, 拍: int = -1, 来路: str = 来路_实感) -> dict:
        """看一屏 → 选出物件 → 建物体码 / **实数场码** / 位置索引。**仅 来路=实感 入库并写 W画**。

        ★场码取**实数求和**（不 sign）：束的双极表决会把幅度信息丢掉，K>8 时身份崩
        （预检⑧ vs ⑯ 对照：K=16 双极保真 0.83 / 实数 0.89；K=32 双极 0.69 / 实数 0.77）。
        ★场码存前**归一到单位范数** ⇒ 「画面的强度」与"存了多少屏"无关（与 W画 同一条纪律）。
        """
        块 = self.块s(帧, 带对比=True)
        if not 块:
            return dict(记=False, 原因="帧不可用")
        件s = self.结构件s(块)
        if not 件s:
            return dict(记=False, 原因="无结构件")
        位s, 码s, 瞥s = [], [], []
        for e in 件s:
            c = e["码"] if e.get("码") is not None else self.特征码(e["外观"])
            if c is None:
                continue
            位s.append(e["位置"])
            码s.append(c)
            瞥s.append(int(e["瞥"]))
        if not 位s:
            return dict(记=False, 原因="无结构件")
        # ★多瞥场码：每瞥一份独立场。**幅度口径**：场 = Σ 项 / √件数
        #   （不是归一到单位范数）。两者对**回放**等价，但对**组合想象**不等价：
        #   注入项的系数始终是 1/√K（K=该瞥件数）⇒ 换背景后仍能解出来；
        #   若把场归一到单位范数，注入系数变成 1（="整个世界只有我看过的那一件"），
        #   一旦该瞥里混进别的东西就会被压下去（真屏实测：16 件搬家后 0 件认得出）。
        场s, 瞥号s = {}, []
        for g in sorted(set(瞥s)):
            号 = [i for i, gg in enumerate(瞥s) if gg == g]
            if not 号:
                continue
            v = np.stack([self.原子(位s[i]) * 码s[i] for i in 号], 0).sum(0).astype(np.float32)
            v = v / np.sqrt(float(len(号)))
            场s[g] = v
            瞥号s.extend([g] * len(号))
        屏 = dict(拍=int(拍), 来路=str(来路),
                  场s=场s, 瞥号s=瞥号s,
                  位s=[tuple(k) for k in 位s],
                  码s=np.stack(码s, 0),
                  外观s=np.stack([self.特征值(c) for c in 码s], 0),
                  物体数=len(位s), 件s=件s)
        入库 = (str(来路) == 来路_实感)
        if 入库:
            self.库.append(屏)
            self.实感次数 += 1
            if bool(self.写W画):                              # ★默认 False（见模块头「为什么关」）
                for g in sorted(场s):
                    self._更新W(场s[g])
        self.迹.append(dict(拍=int(拍), 来路=str(来路), 物体数=len(位s),
                            瞥数=len(场s), 入库=bool(入库), 库大小=len(self.库)))
        return dict(记=True, 入库=入库, 物体数=len(位s), 瞥数=len(场s),
                    库大小=len(self.库), 场s=场s, 位s=屏["位s"], 码s=屏["码s"],
                    外观s=屏["外观s"])

    # ══════════════════ 联想阵的**用途**：场码去噪（它凭什么存在） ══════════════════
    def 联想去噪(self, 场, 阈值: float = 0.0) -> dict:
        """用 W画 做**场码去噪/补全**：残 = 场 ⊙ (W画·场)，再符号化。

        W画 = Σ 屏 (场场ᵀ)（Hebbian，单位范数）。对**见过**的场，W画·场 ≈ 场（自联想）；
        对**加了噪**的场（位翻转），残里原本的位被多数票救回来 ⇒ 与真场更接近。
        返回 dict(去噪, 前cos, 后cos, 提升, 翻位数)。`阈值`：不设（0.0 = 纯自联想）。
        """
        v = np.asarray(场, dtype=np.float32).ravel()
        v = v / (float(np.linalg.norm(v)) + 1e-12)      # ★查询也要单位化（否则 v⊙Wv 被 v 的模主导）
        r = v * (self.W画 @ v.astype(np.float64)).astype(np.float32)
        b = np.sign(r).astype(np.float32)
        b[b == 0] = 1.0
        n = self.迹[-1] if self.迹 else {}
        return dict(去噪=b, 翻位数=int(np.sum(b != np.sign(v))), 键=n)

    def 去噪自证(self, 屏下标: int = -1, 翻转: float = 0.15, 种子: int = 5) -> dict:
        """判据用：取一屏已入库的场 → 随机翻转 `翻转` 比例的位当"带噪回忆" → 看 W画 能否救回。

        · 有增益 = 「提示不全时，记忆自己补全」——这就是 W画 存在的理由；
        · 无增益 = 这 1M 矩阵白背 ⇒ 应关掉 `写W画`（诚实取舍，见 开关）。
        """
        if not self.库:
            return dict(自证=False, 原因="画面库为空")
        屏 = self.库[int(屏下标)]
        真 = np.asarray(屏["场s"][sorted(屏["场s"])[0]], dtype=np.float32)
        rng = np.random.default_rng(int(种子))
        噪 = 真.copy()
        掩 = rng.random(真.shape[0]) < float(翻转)
        噪[掩] *= -1.0
        前 = _cos(噪, 真)
        out = self.联想去噪(噪)
        后 = _cos(out["去噪"], 真)
        return dict(自证=True, 前cos=round(float(前), 4), 后cos=round(float(后), 4),
                    提升=round(float(后 - 前), 4), 翻转位=int(掩.sum()), 维=int(真.shape[0]))

    # ══════════════════ 有界联想阵（同 自我回路 修法） ══════════════════
    def _更新W(self, 场) -> None:
        """Hebbian 写 + Frobenius 归一到 `上限范数_画`（保方向、限幅度、零新超参）。

        与 `内核_自我回路.更新W` 同一修法：不归一 ⇒ ‖W画‖ 随屏数无界增长，
        "画面的强度"就成了"存了多少屏"的函数（时间污染）。
        """
        v = np.asarray(场, dtype=np.float64)
        self.W画 = self.W画 + η_画 * np.outer(v, v)
        n = float(np.linalg.norm(self.W画))
        if n > 1e-12:
            self.W画 = self.W画 * (上限范数_画 / n)

    # ══════════════════ 位置原子（格内预注册 / 格网外按需派生，均确定性） ══════════════════
    def 原子(self, 键) -> np.ndarray:
        """取位置原子。键 = 格键 (bx,by) 或任意标签；**格网外**的键按 CRC32 确定性派生。

        ⚠ 不用 Python 内置 `hash(str)`——它按进程随机化（PYTHONHASHSEED）⇒ 破坏 C13 可复现。
        """
        import zlib
        a = self.位族.get(键)
        if a is not None:
            return a
        s = str(键)
        rng = np.random.default_rng(int(zlib.crc32(s.encode("utf-8"))) & 0xFFFFFFFF)
        a = _双极(rng, self.D)
        self.位族[键] = a                 # 缓存（同键同原子；派生本身是键的纯函数）
        return a

    # ══════════════════ 读出：位置地址 × 库匹配 ══════════════════
    def 读出(self, 场, 码库=None, 位置s=None, 外观库=None, 判据: float = 判据_检测,
             瞥号_位=None, 瞥号s=None) -> dict:
        """从画面码里"想出画面"。

        ★优先级：`瞥号s`（与位置逐项对齐，**想象必须用它**） > `瞥号_位`（格键→瞥号，
        即 `分瞥()` 的输出；由调用方给或本函数现算）。想象时物件搬家后落到的**格位**归属的
        瞥会变，但解码必须用它**当初被编码时所在的那份场**——否则解出来的是别瞥的码
        （实测：误用格位算瞥 ⇒ 16 件里只有 1 件认得出；用对的瞥 ⇒ 16/16 认得出）。
        返回 dict(命中位置s, 每格cos, 重建, 逐格物体, 命中数)。
        · 位置当**读地址**：残 = 场 ⊙ 丄_p（位置原子自逆）⇒ 反投影 Pᵀ·残/√D = "那一块的样子"。
        · **相似度 = 重建外观与库外观的余弦**（预检⑯：K≤8 时对角 0.92~0.98、身份 8/8）。
        · ★位置本身**不从场码里猜**（预检⑮：所有位置的 ‖残‖ 恒等 ⇒ 位置不在场的模里），
          它由**存下来的地址表**给出，正如空间参考帧由身体给出、不靠视网膜自己发明。
        """
        S = 码库
        if S is None:
            S = (self.库[-1]["码s"] if self.库 else None)
        if S is None:
            return dict(命中位置s=[], 每格cos={}, 重建={}, 逐格物体={}, 命中数=0)
        S = np.asarray(S, dtype=np.float32)
        if S.ndim == 1:
            S = S[None, :]
        A = 外观库
        if A is None:
            A = np.stack([self.特征值(c) for c in S], 0)
        A = np.asarray(A, dtype=np.float32)
        if A.ndim == 1:
            A = A[None, :]
        An = np.linalg.norm(A, axis=1) + 1e-12
        单场 = not isinstance(场, dict)
        v0 = None
        if 单场:
            v0 = np.asarray(场, dtype=np.float32).ravel()
        if not 单场 and 瞥号_位 is None and not 瞥号s:
            瞥号_位 = self.分瞥()
        格s = ([(bx, by) for by in range(self.格y) for bx in range(self.格x)]
               if 位置s is None else list(位置s))
        每格, 重建, 逐格物, 码似 = {}, {}, {}, {}
        for j, k in enumerate(格s):
            if 单场:
                v = v0
            else:
                if 瞥号s is not None and j < len(瞥号s):
                    g = 瞥号s[j]                      # ★与位置逐项对齐（想象/回放的主路）
                elif 瞥号_位 is not None:
                    g = 瞥号_位.get(tuple(k), None)
                else:
                    g = self.分瞥().get(tuple(k), list(场)[0])
                if g not in 场:
                    g = list(场)[0]
                v = np.asarray(场[g], dtype=np.float32).ravel()
            rec = (self.原子(k) * v) @ self.P逆.T
            rn = float(np.linalg.norm(rec))
            if rn <= 1e-12:
                每格[k], 逐格物[k], 重建[k], 码似[k] = 0.0, -1, (-1, rec), -1.0
                continue
            # ★判据用**码域相似**：重建特征→重新编码→与库码的逐位平均乘积（∈[-1,1]）。
            #   为什么不用"归一化 cos"：残里的**噪声分量在归一化时被放大**，空位也能到 0.89
            #   （真屏实测），而真位 0.98——判据会被噪声贴脸。码域用的是**未归一**的重建，
            #   噪声不放大；且与"库里到底是什么码"同一坐标系。见文件头「读出怎么定」。
            b = np.sign(self.P @ rec)
            b = np.where(b == 0, 1.0, b).astype(np.float32)
            似s = (S @ b) / float(self.D)
            i = int(np.argmax(似s))
            每格[k] = float(似s[i])
            逐格物[k] = i
            重建[k] = (i, rec)
            码似[k] = float(似s[i])
        命中 = [k for k in 格s if 每格[k] >= float(判据)]
        return dict(命中位置s=命中, 每格cos=每格, 每格码似=码似, 重建=重建,
                    逐格物体=逐格物, 命中数=len(命中))

    def 回放(self, 下标: int = -1, 判据: float = 判据_检测) -> dict:
        """按**存下来的地址表**读回一屏（回放）：给该屏位置索引处的解 + 检出的命中格。"""
        if not self.库:
            return dict(回放=False, 原因="画面库为空")
        屏 = self.库[int(下标)]
        r = self.读出(屏["场s"], 码库=屏["码s"], 位置s=屏["位s"],
                     外观库=屏.get("外观s"), 判据=判据)
        r.update(回放=True, 屏=屏, 来路=来路_回忆)
        self.迹.append(dict(拍=屏["拍"], 来路=来路_回忆, 物体数=屏["物体数"],
                            入库=False, 命中数=r["命中数"], 库大小=len(self.库)))
        return r

    def 新位置键(self, 标签: str = None) -> tuple:
        """一个**格网外**的新位置键（它从没在这个位置看见过任何东西）。"""
        return ("新", "想象位置|%s" % ("—" if 标签 is None else str(标签)))

    def 在格内(self, 键) -> bool:
        return (isinstance(键, tuple) and len(键) == 2
                and isinstance(键[0], int) and isinstance(键[1], int)
                and 0 <= 键[0] < self.格x and 0 <= 键[1] < self.格y)

    # ══════════════════ 想象：把记忆里的物体搬到它从没出现过的位置 ══════════════════
    def 想象(self, 物体下标: int = 0, 屏下标: int = -1, 新键=None,
             判据: float = 判据_检测) -> dict:
        """组合想象：取某屏里第 `物体下标` 个物体，绑到一个**新位置原子**上，按同一读出路径解出来。

        「这是想象不是回放」需要两条同时成立：
          ① 物体会**还认得出来**（解出的 cos ≥ `判据_检测`）；
          ② (该物体, 该位置) 这个组合**从未入库**（不是把看过的换个说法复述）。
        返回 dict(想象, 认得, 新组合, cos, 特征, 场, 键, 原键, …)。
        """
        if not self.库:
            return dict(想象=False, 原因="画面库为空")
        屏 = self.库[int(屏下标)]
        位s = [tuple(x) for x in 屏["位s"]]
        i = int(物体下标) % max(1, len(位s))
        原键, 原码 = 位s[i], np.asarray(屏["码s"][i], dtype=np.float32)
        外观 = np.asarray(屏["外观s"][i], dtype=np.float32)
        键 = self.新位置键() if 新键 is None else 新键
        场 = np.asarray(self.原子(键) * 原码, dtype=np.float32)
        r = self.读出(场, 码库=原码[None, :], 位置s=[键], 外观库=外观[None, :], 判据=判据)
        cos = float(r["每格cos"][键])
        认得 = bool(cos >= float(判据))
        新组合 = bool(键 not in 位s)                 # 该 (物体, 位置) 组合不在该屏的地址表里
        self.想象次数 += 1
        self.迹.append(dict(拍=屏.get("拍"), 来路=来路_想象, 物体数=1, 入库=False,
                            认得=认得, 新组合=新组合, cos=round(cos, 4),
                            库大小=len(self.库)))
        return dict(想象=True, 认得=认得, 新组合=新组合, cos=cos,
                    特征=r["重建"][键][1], 场=场, 键=键, 原键=原键,
                    物体下标=i, 屏下标=int(屏下标))

    # ══════════════════ 自指读出：它"想"的画面去喂它"预测自己"的那张表 ══════════════════
    def 想象码(self, 屏下标: int = -1, 位移: tuple = (3, 2)) -> dict:
        """**整体搬家**式组合想象：把某屏里每个物件绑到"原位置 + 位移"上，重组成**多瞥**新场码。

        为什么这样定义"新"：每个物件都出现在**它从未出现过的格位**（旧组合全部打散），
        但屏内布局（物件之间的相对位置）保留 ⇒ 新生的是"空间组合"，不是把看过的复述一遍。
        返回 dict(想象, 场s, 位s, 瞥号s, 源位s, 位移, 新组合格数)。
        """
        if not self.库:
            return dict(想象=False, 原因="画面库为空")
        屏 = self.库[int(屏下标)]
        dx, dy = int(位移[0]), int(位移[1])
        if dx == 0 and dy == 0:
            return dict(想象=False, 原因="零位移=回放，不叫想象")
        位s0 = [tuple(x) for x in 屏["位s"]]
        瞥号0 = list(屏["瞥号s"])
        位s, 源位s = [], []
        for k in 位s0:
            # ★环面平移（不回卷到边缘）：若用 clip，靠边的物件会**挤到同一格**上互相抵消
            #   （真屏实测：16 件搬家后 0 件认得出，根因就是边界碰撞）。
            #   环面平移保持"每个物件各占一格"，且新格位确实都是它没待过的位置。
            nx = int((k[0] + dx) % self.格x)
            ny = int((k[1] + dy) % self.格y)
            位s.append((nx, ny))
            源位s.append(k)
        场s, 瞥号s = {}, []
        for g in sorted(set(瞥号0)):
            号 = [i for i, gg in enumerate(瞥号0) if gg == g]
            v = np.stack([self.原子(位s[i]) * np.asarray(屏["码s"][i], dtype=np.float32)
                          for i in 号], 0).sum(0).astype(np.float32)
            v = v / np.sqrt(float(len(号)))              # 与入库同律（见 记一屏 幅度口径）
            场s[g] = v
            瞥号s.extend([g] * len(号))
        新组合数 = len(set(位s) - set(位s0))
        self.想象次数 += 1
        self.迹.append(dict(拍=屏.get("拍"), 来路=来路_想象, 物体数=len(位s), 入库=False,
                            新组合=bool(新组合数 > 0), 新组合格数=int(新组合数),
                            库大小=len(self.库)))
        return dict(想象=True, 场s=场s, 场=场s, 物体数=len(位s),
                    新组合=bool(新组合数 > 0), 新组合格数=int(新组合数),
                    位s=位s, 源位s=源位s, 瞥号s=瞥号s,
                    位移=(dx, dy), 源屏=int(屏下标))

    # ══════════════════ 想象的读出：逐件判"还认得出来吗" ══════════════════
    def 想象读出(self, 屏下标: int = -1, 位移: tuple = (3, 2),
                 判据: float = 判据_检测) -> dict:
        """想出**一幅新画面**（环面搬家）→ 按同一读出路径逐件解 → 逐件判"还认得出来吗"。

        返回 dict(想象, 场s, 位s, 源位s, 瞥号s, 认得件数, 总件数, 码似最低, 逐件码似, 位移)。
        判定"这是想象不是回放"要**两件同时成立**：
          ① 件还认得出来（码似 ≥ 判据）；② 该件确实换了格位（新组合格数 > 0）。
        """
        im = self.想象码(屏下标=屏下标, 位移=位移)
        if not im.get("想象"):
            return dict(想象=False, 原因=im.get("原因"))
        屏 = self.库[int(屏下标)]
        r = self.读出(im["场s"], 码库=屏["码s"], 位置s=list(im["位s"]),
                     外观库=屏.get("外观s"), 判据=判据, 瞥号s=list(im["瞥号s"]))
        似s = [float(r["每格码似"][k]) for k in im["位s"]]
        认 = sum(1 for x in 似s if x >= float(判据))
        out = dict(im)
        out.update(认得件数=int(认), 总件数=len(似s), 逐件码似=似s,
                   码似最低=(min(似s) if 似s else None),
                   认得=bool(len(似s) > 0 and 认 == len(似s)))
        return out

    # ══════════════════ 只读汇报 ══════════════════
    def 摘要(self) -> dict:
        return dict(库大小=len(self.库), 实感次数=self.实感次数, 想象次数=self.想象次数,
                    迹条数=len(self.迹), D=self.D, 格=(self.格x, self.格y),
                    瞥=(盯X, 盯Y), 每瞥件=每瞥件, 上限件=上限件,
                    维_特征=self.维_特征, 判据_检测=判据_检测,
                    写W画=bool(self.写W画),
                    W画范数=round(float(np.linalg.norm(self.W画)), 4))

    # ══════════════════ 画面的可视化（粗格 PNG；供人眼核对"脑子里那块"） ══════════════════
    @staticmethod
    def _格到rgb(特征) -> tuple:
        """12 维光适应特征 → 一个显示用 RGB（光适应是零均值，+128 归到中灰为底）。"""
        x = np.asarray(特征, dtype=np.float32).reshape(粗分辨率, 粗分辨率, 3)
        c = x.mean(axis=(0, 1)) * 60.0 + 128.0
        return tuple(int(v) for v in np.clip(c, 0, 255))

    def 画格(self, 键s, 重建s) -> np.ndarray:
        """按 (键 → 重建特征) 拼一张格图（16×9×3 uint8）。"""
        格 = np.full((self.格y, self.格x, 3), 24, np.uint8)
        for k in 键s:
            if not self.在格内(k):
                continue
            格[k[1], k[0]] = self._格到rgb(重建s[k][1])
        return 格

    def 造png(self, 屏下标: int = -1, 判据: float = 判据_检测, 放大: int = 24,
              路径: str = None) -> str:
        """把一屏**从记忆里读出来的画面**画成 PNG（回放口径）。"""
        if not self.库:
            return ""
        屏 = self.库[int(屏下标)]
        r = self.读出(屏["场s"], 码库=屏["码s"], 位置s=[tuple(x) for x in 屏["位s"]],
                     外观库=屏.get("外观s"), 判据=判据)
        格 = self.画格([tuple(x) for x in 屏["位s"]], r["重建"])
        图 = np.repeat(np.repeat(格, 放大, 0), 放大, 1)
        return _写png(图, 路径)

    def 造想象png(self, 屏下标: int = -1, 位移: tuple = (3, 2), 放大: int = 24,
                  路径: str = None) -> str:
        """把**想象出来的画面**画成 PNG：按想象场码读出，再把格位反移回原位显示（人眼可对照）。"""
        im = self.想象码(屏下标=屏下标, 位移=位移)
        if not im.get("想象"):
            return ""
        屏 = self.库[int(屏下标)]
        r = self.读出(im["场s"], 码库=屏["码s"], 位置s=list(im["位s"]),
                     外观库=屏.get("外观s"), 判据=判据_检测, 瞥号s=im["瞥号s"])
        # 反移回原格位显示（把"搬到新位置"折回原坐标系，便于与回放图并排比对）
        dx, dy = im["位移"]
        键反 = {k: (int(np.clip(k[0] - dx, 0, self.格x - 1)),
                    int(np.clip(k[1] - dy, 0, self.格y - 1))) for k in im["位s"]}
        格 = np.full((self.格y, self.格x, 3), 24, np.uint8)
        for k in im["位s"]:
            kk = 键反[k]
            格[kk[1], kk[0]] = self._格到rgb(r["重建"][k][1])
        图 = np.repeat(np.repeat(格, 放大, 0), 放大, 1)
        return _写png(图, 路径)



def _写png(图: np.ndarray, 路径: str = None) -> str:
    """最小 PNG 编码（zlib + 手写块；无第三方依赖 ⇒ 不受"外挂模型"红线影响）。"""
    import struct
    import zlib
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


# ────────────────────────── synthesis 顶覆盖（只加挂；基链一行不改） ──────────────────────────

class 顶覆盖_综合体H(顶覆盖_综合体W):
    """步8 宿主：在步6 宿主之上**再**加挂画面核（视觉想象）。

    ⚠ 继承顺序（重要）：必须继承 **步6 宿主 W**（而不是步5 的 Z），否则 `顿悟开关` 这一层
    就被跳过——链是线性的：基 ← T ← L ← Z ← W ← H。判据 N5-1 检的"五层齐全"依然成立。

    · 画面开关=False（默认）：_引擎步 与步6/基类逐位同构（D1 逐位恒等）。
    · 画面开关=True：每拍若拿到**真帧**（`设帧源(...)` 提供）→ 记一屏（来路=实感）；
      并可按判据回放/想象，把"想出来的画面码"喂给它自己的自模型（自指闭环）。
      **引擎状态推进行一行不改。**
    """

    def __init__(self, seed: int = 310, 私偏κ: float = 10.0, 画面开关=None, 档案seed=None,
                 自我开关=None, 学习开关=None, 思核开关=None, 自模型开关=None,
                 顿悟开关=None, 学习: bool = True, 状态投影=None, 思核参: dict = None,
                 画面种子=None, 自我提问开关=None):
        self.画面开关 = bool(开关默认 if 画面开关 is None else 画面开关)
        self.画面核 = None
        self._帧源 = None                     # 可调用：() -> RGB ndarray 或 None
        self._画面拍记 = 0
        self._画面上帧t = 0.0
        self._上帧σ = None
        self.帧间隔s = float(帧间隔s)
        self._画面种子 = int(seed if 画面种子 is None else 画面种子)
        super().__init__(seed=seed, 私偏κ=私偏κ, 自我开关=自我开关, 档案seed=档案seed,
                         学习开关=学习开关, 思核开关=思核开关, 自模型开关=自模型开关,
                         顿悟开关=顿悟开关, 学习=学习,
                         状态投影=状态投影, 思核参=思核参, 自我提问开关=自我提问开关)
        if self.画面开关:
            D = int(np.asarray(self.eng.hdc.Φ_act).shape[0])
            self.画面核 = 画面核(D=D, 种子=self._画面种子)

    # ── 真帧来源（由 V3 主程序接上 网页眼.截图；本文件不 import 任何器官）──
    def 设帧源(self, 源) -> None:
        """注册"真帧"来源（callable 或可截图对象）。None ⇒ 断开。"""
        self._帧源 = 源

    def _取真帧(self):
        s = self._帧源
        if s is None:
            return None
        try:
            if callable(s):
                return s()
            if hasattr(s, "截图"):
                return s.截图()
        except Exception:
            return None
        return None

    # ── 自指闭环：把**它自己想出来的画面**喂给「预测自己」的那张表（ĝ） ──
    def 喂想象给自模型(self, 屏下标: int = -1, 位移: tuple = (3, 2),
                       更新: bool = False) -> dict:
        """取**想象的画面**（整体搬家后的组合）当"下一帧"，送给步2 的 ĝ 去预测。

        · `更新=False`（默认）⇒ **只预测不写**：想象不算"我经历过的"，不写 ĝ 的 W
          （与「只有实感进画面库」同一条纪律：自己编的东西不塑造自己）。
        · `更新=True` ⇒ 允许它写 W（对照臂，用于判据比较"想象会不会污染自模型"）。
        返回 dict(喂, ε_self, Ŝ一致度, 位移, 新组合格数)。引擎状态推进行一行不改。
        """
        if self.画面核 is None or not self.画面核.库:
            return dict(喂=False, 原因="画面库为空或未开")
        zm = getattr(self, "自模型", None)
        if zm is None or getattr(zm, "W", None) is None:
            return dict(喂=False, 原因="自模型未就绪（需要先跑过至少一拍实感）")
        im = self.画面核.想象码(屏下标=屏下标, 位移=位移)
        if not im.get("想象"):
            return dict(喂=False, 原因=im.get("原因"))
        # ★"想象出来的那一帧" = 把多瞥场码**按瞥拼成一份 D 维场**（各瞥的单位化叠加，
        #   与引擎母语空间同维）。它喂的是 ĝ 的**输入键**，语义是"我自己想出来的下一帧"。
        S = np.zeros(self.画面核.D, dtype=np.float64)
        for g in sorted(im["场s"]):
            v = np.asarray(im["场s"][g], dtype=np.float64)
            S = S + v / (float(np.linalg.norm(v)) + 1e-12)
        S = S / (float(np.linalg.norm(S)) + 1e-12)
        S = np.sign(S)
        S[S == 0] = 1.0
        旧 = bool(getattr(zm, "更新", True))
        try:
            zm.更新 = bool(更新)
            ε = zm.学一拍(S, S, 拍=int(getattr(self, "拍", 0)))
        finally:
            zm.更新 = 旧
        Ŝ = None
        try:
            Ŝ = 1.0 - float(ε) if ε is not None else None      # ε = 1 − cos ⇒ 一致度 = 1 − ε
        except Exception:
            Ŝ = None
        return dict(喂=True, ε_self=(None if ε is None else float(ε)), 一致度=Ŝ,
                    位移=im["位移"], 新组合格数=im["新组合格数"], 更新W=bool(更新))

    def 画面一拍(self, 帧=None, 拍: int = None) -> dict:
        """显式记一屏 + 回放 + 组合想象（供测试/演示直接调；不依赖 _引擎步）。"""
        if self.画面核 is None:
            return dict(画面=False, 原因="画面开关未开")
        f = 帧 if 帧 is not None else self._取真帧()
        if f is None:
            return dict(画面=False, 原因="无真帧")
        记 = self.画面核.记一屏(f, 拍=(self.拍 if 拍 is None else 拍), 来路=来路_实感)
        if not 记.get("记"):
            return dict(画面=False, 原因=记.get("原因"))
        回 = self.画面核.回放() if self.画面核.库 else None
        im = self.画面核.想象(0) if self.画面核.库 else None
        return dict(画面=True, 记=记, 回放=(None if 回 is None else
                                        dict(命中数=回["命中数"], 物体数=记["物体数"])),
                    想象=(None if im is None else
                        dict(认得=im["认得"], 新组合=im["新组合"], cos=im["cos"])))

    def _画面旁挂(self, rec) -> None:
        """每拍旁挂：够间隔且**真帧变了**才记一屏（只增本核状态；不改 rec、不改引擎）。

        两条门（都为了"别把同一屏存 250 遍"）：
          ① 时间门：`帧间隔s`（默认 1.0s 一次）。引擎的思考/文件拍也走 _引擎步，
             屏幕内容在 1 秒里不会变 ⇒ 用真实采样率抽样，而不是用拍数设阈值（那不是人挑语义阈值）。
          ② **内容门**：新帧与上一帧的格对比图必须真的不同（逐格 σ 差 > 共模噪声；
             共模 = 全屏平均差不计——整屏亮度漂移不算"看到了新东西"）。
        返回 True 表示本拍真的记了一屏（判据用）。
        """
        if self.画面核 is None:
            return False
        import time as _t
        现 = _t.time()
        if (现 - float(getattr(self, "_画面上帧t", 0.0))) < float(self.帧间隔s):
            return False
        f = self._取真帧()
        if f is None:
            return False
        self._画面上帧t = 现
        try:
            块 = self.画面核.块s(f, 带对比=True)
            if not 块:
                return False
            格键s = [(bx, by) for by in range(self.画面核.格y) for bx in range(self.画面核.格x)]
            σv = np.array([块[k][1] for k in 格键s], dtype=np.float64)
            上 = getattr(self, "_上帧σ", None)
            if 上 is not None and 上.shape == σv.shape:
                差 = σv - 上
                共模 = float(差.mean())
                if float(np.max(np.abs(差 - 共模))) <= float(帧变化阈):
                    return False                     # 内容没变（只有整屏亮度漂移）⇒ 不记
            self._上帧σ = σv
            r = self.画面核.记一屏(f, 拍=int(getattr(self, "拍", 0)), 来路=来路_实感)
            if not r.get("记"):
                return False
            self._画面拍记 += 1
            if int(getattr(self, "拍", 0)) % 20 == 0 and len(self.账本) > 0:
                文本 = ("画面[第%d拍]：这一屏我看出了 %d 个结构件（画面库 %d 屏）。"
                        % (int(getattr(self, "拍", 0)), int(r.get("物体数", 0)),
                           int(r.get("库大小", 0))))
                self._自言("自评", 文本, (来源标_自源, "画面"), [len(self.账本) - 1])
            return True
        except Exception:
            return False

    def _引擎步(self, u):
        rec = super()._引擎步(u)                      # 基链状态推进 + 步2~6 旁挂，逐行不变
        if self.画面开关 and self.画面核 is not None:
            try:
                self._画面旁挂(rec)
            except Exception:
                pass                                  # 旁挂失败绝不打断既有路径
        return rec


__all__ = ["画面核", "顶覆盖_综合体H", "光适应", "开关默认", "判据_检测", "粗分辨率",
           "来路_实感", "来路_回忆", "来路_想象", "上限范数_画", "η_画"]
