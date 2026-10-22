# -*- coding: utf-8 -*-
"""meeting38/实施/fsw_kernel.py —— FSW 内核（步 1）：焦点束 + 长时库 + IVF 寻址 + E5 细码精排。

规格：meeting38/会议_容量悖论/06_内核规格_合并.md（含 §8/§9 实现期修正）。

§9 重要修正（2026-10-05，实测证伪记录）：
01 方案的随机桶原子目录树被冒烟实测证伪——200 条记录散成 191 个叶，5% 位翻转即使
下降路径与插入路径不一致（随机桶原子与内容无关，argmax 余量 ~1/√D 被噪声淹没；
且聚合计嵌套信号按 √(1/子树) 衰减，深必死）。故寻址层改用 **IVF 粗量化器**：
- 质心 = 内容派生（簇成员束的多数表决），对噪声查询局部性保真；
- 增量确定性 2-means 分裂（无梯度、无训练集概念，C1/C7 合规），簇大小 ≤ 2·簇目标；
- 检索 = 全质心打分 → top-nprobe 簇 → 簇内粗码粗选 → E5 细码精排。
这正是 03 专家 E5/E1（粗码常驻+多探+精排）路线的实现形态。

纪律（对照简报 C1–C8）：
- C1 无 Transformer：绑(按位乘)/束(多数表决)/余弦/argmax/2-means，无 attention/QKV/softmax/梯度；
- C4 D_C=1024 恒定：容量靠 IVF 簇数 ∝ M、因子化 O(R+F)、数轴多轴，不靠扩维；
- C6 预算：CPU numpy 口径（思维拍 ≤50ms / 手拍 ≤16ms，判据见 test_f1_f6.py）；
- C7 确定性：原子/投影/质心初始化/平票全部 sha256(种子) 或 (值降序,id 升序) 确定性；
- C8 复用：束/绑/相似 约定与 meeting38/共存态.py::编码器 逐位一致（平票→+1）；
  连续量复用 meeting32/声带.py::数轴（动态导入，不改其文件）。

三层 API：
- 检索地址(q束)   —— q ≈ 记录束/句柄副本（IVF 多探，延迟 ∝ nprobe·簇目标 + nlist·D）
- 检索内容(q束)   —— 部分线索（库直扫 M·D_c，工作点 M≤8192；更大规模走因子化）
- 解码(记录束,角色) —— 因子化解码（角色桶 ≤N_桶，绑(R,S) 后桶内找填充，两阶段精排）
"""

from __future__ import annotations

import hashlib

import numpy as np

# —— 规格常数（06 规格 §1/§2/§10；可在构造器覆盖，判据按默认值跑） ——
D_C = 1024          # 常驻粗码维度（C4：恒定，不扩）
簇目标 = 32          # IVF 每簇目标成员数（13000 条 → ~400 簇；F1 修正后 N_b 下调的配套）
簇分裂上限 = 64      # 超过即确定性 2-means 分裂
N_探 = 24           # 检索探查簇数（nprobe）
N_桶 = 1000         # 角色桶容量（解码候选上限）
D_F = 4096          # 细码维度（CPU 工作点；GPU 阶段按 03 可升 8192/16384）
K_焦点 = 7          # 焦点束默认工作点（02 口径）
K_上限 = 40         # 显著度门硬上限（01 口径）
种子默认 = 3800


def _哈希双极(键: str, D: int) -> np.ndarray:
    """键 → D 维双极原子。sha256 确定性导出（与 声带._基向量 同律，跨进程逐位一致）。"""
    h = hashlib.sha256(键.encode("utf-8")).digest()
    buf = h
    while len(buf) < max(D // 8, 1):
        buf += hashlib.sha256(buf + h).digest()
    bits = np.frombuffer(buf[: D // 8], dtype=np.uint8)
    return ((bits[:, None] >> np.arange(8)[None, :]) & 1).reshape(-1)[:D].astype(np.float32) * 2 - 1


def 束(码s, D: int = D_C) -> np.ndarray:
    """多数表决束；平票 → +1（与 共存态.编码器.束 逐位同约定）。"""
    码s = [np.asarray(c, dtype=np.float32).ravel() for c in 码s]
    if not 码s:
        return np.zeros(D, dtype=np.float32)
    b = np.sign(np.stack(码s, 0).sum(0)).astype(np.float32)
    b[b == 0] = 1.0
    return b


def 绑(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """绑定 role⊗filler：按位乘（自逆：R⊙R=1⃗，解绑 R⊙(R⊙F)=F）。约定同 编码器.绑(移位=0)。"""
    c = np.sign(np.asarray(a, np.float32) * np.asarray(b, np.float32)).astype(np.float32)
    c[c == 0] = 1.0
    return c


def 相似(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float32).ravel()
    b = np.asarray(b, np.float32).ravel()
    na = float(np.linalg.norm(a)) or 1.0
    nb = float(np.linalg.norm(b)) or 1.0
    return float(a @ b / (na * nb))


class FSW内核:
    """分层因子化稀疏工作区内核（纯新增，不依赖/不修改任何既有模块）。"""

    def __init__(self, 种子: int = 种子默认, D_C_: int = D_C,
                 簇目标_: int = 簇目标, 分裂上限: int = 簇分裂上限, N探: int = N_探,
                 N桶: int = N_桶, D_F_: int = D_F,
                 细码缓存条数: int = 20000, K焦点: int = K_焦点, K上限: int = K_上限):
        self.种子, self.D = int(种子), int(D_C_)
        self.簇目标, self.分裂上限, self.N探 = int(簇目标_), int(分裂上限), int(N探)
        self.N桶 = int(N桶)
        self.D_F, self.K焦点, self.K上限 = int(D_F_), int(K焦点), int(K上限)
        # E5 投影阵：sha256(种子) 派生 rng，确定性（C7）
        h = hashlib.sha256(("proj|%d" % self.种子).encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        self._P = (rng.standard_normal((self.D_F, self.D)) / np.sqrt(self.D)).astype(np.float32)
        # L0 原子表 + 角色桶索引
        self._原子表: dict = {}
        self._角色桶: dict = {}                     # 角色 → [填充键]（登记序，确定性）
        # L1 记录层（rec_id = 插入序 = 库行号）
        self.记录: dict = {}                       # rec_id → dict(句柄, 束, 字段)
        self._库: list = []                        # 粗码常驻
        self._库矩阵: np.ndarray = None
        self._下一id = 0
        # IVF 粗量化器（确定性增量 2-means）
        self._质心s: list = []                      # 簇id → 质心向量
        self._倒排: list = []                       # 簇id → [rec_id]
        self._质心矩阵: np.ndarray = None
        self._簇矩阵缓存: dict = {}                  # 簇id → (成员束矩阵)——查询热路径免逐条 stack
        # 细码表：**写入时生成**（CPU 阶段：RAM 充裕，细码 int8 常驻内存 ~4KB/条；
        # GPU 阶段（步 5+）改为按需生成+LRU，即 03 的显存方案。值确定性 ⇒ 何时生成不改结果，F6）
        self._记录细码表: dict = {}                 # rec_id → int8[D_F]
        self._填充细码表: dict = {}                 # 原子键 → int8[D_F]
        # 统计
        self.统计 = dict(写=0, 簇分裂=0, 细码生成=0, 细码命中=0, 探查记录=0)

    # ── 原子层 ──
    def 原子(self, 键: str) -> np.ndarray:
        v = self._原子表.get(键)
        if v is None:
            v = _哈希双极("原子|%d|%s" % (self.种子, 键), self.D)
            self._原子表[键] = v
            if 键.startswith("填充|"):
                角色 = 键[3:].split("=", 1)[0]
                self._角色桶.setdefault(角色, []).append(键)
        return v

    def 登记填充词表(self, 角色: str, 填充s) -> None:
        """预登记某角色的填充词表（建立角色桶，细码随登记生成）。"""
        for f in 填充s:
            键 = "填充|%s=%s" % (角色, f)
            self.原子(键)
            if 键 not in self._填充细码表:
                self._填充细码表[键] = self._细码(self._原子表[键]).astype(np.int8)

    # ── 基础算子 ──
    def _批相似(self, q: np.ndarray, 码本: np.ndarray) -> np.ndarray:
        qn = float(np.linalg.norm(q)) or 1.0
        n = np.linalg.norm(码本, axis=1)
        return (码本 @ q) / np.maximum(n * qn, 1e-9)

    def _批相似双极(self, q: np.ndarray, 码本: np.ndarray) -> np.ndarray:
        """双极快速路径：码本与 q 均为 ±1 时 ‖·‖=√D 精确成立，cos = (码本@q)/D。
        省去每次 axis=1 范数重算（768×1024 实测 2~3ms → <0.5ms）。仅限 ±1 输入。"""
        return (码本 @ np.asarray(q, np.float32)) / float(self.D)

    @staticmethod
    def _topk序(值: np.ndarray, k: int) -> np.ndarray:
        """确定性 top-k 索引：值降序，tie 按索引升序（stable）。"""
        k = min(max(int(k), 1), len(值))
        return np.argsort(-值, kind="stable")[:k]

    # ── 细码（E5）──
    def _细码(self, 粗码: np.ndarray) -> np.ndarray:
        """sign(P@粗码) → float32 ±1。确定性；缓存命中与否不改值。"""
        f = np.sign(self._P @ np.asarray(粗码, np.float32)).astype(np.float32)
        f[f == 0] = 1.0
        return f

    def _记录细码(self, rec_id: int) -> np.ndarray:
        c = self._记录细码表.get(rec_id)
        if c is not None:
            self.统计["细码命中"] += 1
            return c.astype(np.float32)
        self.统计["细码生成"] += 1
        f = self._细码(self.记录[rec_id]["束"])
        self._记录细码表[rec_id] = f.astype(np.int8)
        return f

    # ── IVF 粗量化器（确定性）──
    def _质心矩阵缓存(self) -> np.ndarray:
        if self._质心矩阵 is None or self._质心矩阵.shape[0] != len(self._质心s):
            self._质心矩阵 = np.stack(self._质心s, 0)
        return self._质心矩阵

    def _分配(self, s: np.ndarray) -> int:
        if not self._质心s:
            self._质心s.append(s.copy())
            self._倒排.append([])
            self._质心矩阵 = None
            return 0
        return int(np.argmax(self._批相似双极(s, self._质心矩阵缓存())))

    def _二分裂(self, 成员: list) -> tuple:
        """确定性 2-means：≤分裂上限 个成员束 → (质心A, 质心B, 分配表)。
        初始种子 = members[0] 与与其余弦最低者；8 轮 Lloyd；空侧保留旧质心。"""
        束s = [self.记录[i]["束"] for i in 成员]
        cA = 束s[0].copy()
        sims = [相似(cA, w) for w in 束s]
        cB = 束s[int(np.argmin(sims))].copy()
        归属 = [0] * len(束s)
        for _ in range(8):
            for i, w in enumerate(束s):
                归属[i] = 0 if 相似(w, cA) >= 相似(w, cB) else 1
            a侧 = [束s[i] for i in range(len(束s)) if 归属[i] == 0]
            b侧 = [束s[i] for i in range(len(束s)) if 归属[i] == 1]
            if a侧:
                cA = 束(a侧, self.D)
            if b侧:
                cB = 束(b侧, self.D)
            if not a侧 or not b侧:
                break
        return cA, cB, 归属

    def _可能分裂(self, cid: int):
        if len(self._倒排[cid]) <= self.分裂上限:
            return
        cA, cB, 归属 = self._二分裂(self._倒排[cid])
        a侧 = [self._倒排[cid][i] for i in range(len(归属)) if 归属[i] == 0]
        b侧 = [self._倒排[cid][i] for i in range(len(归属)) if 归属[i] == 1]
        self._质心s[cid] = cA
        self._倒排[cid] = a侧
        self._质心s.append(cB)
        self._倒排.append(b侧)
        self._质心矩阵 = None
        self._簇矩阵缓存.pop(cid, None)
        self.统计["簇分裂"] += 1
        # 分裂后两侧仍可能超限（罕见），继续（递归深度有界：簇大小减半）
        if len(a侧) > self.分裂上限:
            self._可能分裂(cid)
        if len(b侧) > self.分裂上限:
            self._可能分裂(len(self._质心s) - 1)

    # ── 写 ──
    def 写记录(self, 字段, 句柄=None) -> int:
        """字段: [(角色, 填充), ...]；记录 = 束([绑(R,F)...])；填充原子按角色作用域
        （键 = '填充|角色=填充'），与 解码/角色桶 同命名空间。"""
        vs = []
        for r, f in 字段:
            角色键, 填充键 = "角色|%s" % r, "填充|%s=%s" % (r, f)
            vs.append(绑(self.原子(角色键), self.原子(填充键)))
            if 填充键 not in self._填充细码表:      # 写入时生成细码（值确定，F6）
                self._填充细码表[填充键] = self._细码(self._原子表[填充键]).astype(np.int8)
        s = 束(vs, self.D)
        rid = self._下一id
        self._下一id += 1
        self.记录[rid] = dict(句柄=句柄, 束=s, 字段=list(字段))
        self._记录细码表[rid] = self._细码(s).astype(np.int8)   # 写入时生成（E5/CPU 工作点）
        self._库.append(s)
        self._库矩阵 = None
        cid = self._分配(s)
        self._倒排[cid].append(rid)
        self._簇矩阵缓存.pop(cid, None)
        self._可能分裂(cid)
        self.统计["写"] += 1
        return rid

    def _簇矩阵(self, cid: int) -> np.ndarray:
        m = self._簇矩阵缓存.get(cid)
        if m is None or m.shape[0] != len(self._倒排[cid]):
            m = np.stack([self.记录[i]["束"] for i in self._倒排[cid]], 0)
            self._簇矩阵缓存[cid] = m
        return m

    # ── 精排（两层共用）──
    def _精排(self, q束: np.ndarray, 候选id: list, k: int, 候选码本: np.ndarray = None) -> dict:
        if 候选码本 is None:
            候选码本 = np.stack([self.记录[i]["束"] for i in 候选id], 0)
        粗相似 = self._批相似双极(q束, 候选码本)
        序 = self._topk序(粗相似, k)
        候选k = [候选id[i] for i in 序]
        q_f = self._细码(q束)
        细相似 = np.stack([self._记录细码(i) for i in 候选k], 0) @ q_f / self.D_F
        星 = int(np.argmax(细相似))                # tie 取首个（= id 升序），确定性
        return dict(top1=候选k[星], 细相似=float(细相似[星]),
                    粗top1=候选k[0], 粗相似=float(粗相似[序[0]]),
                    候选k=候选k, 粗相似k=粗相似[序].tolist())

    # ── 读①：地址检索（IVF 多探；q ≈ 记录束/句柄副本）──
    def 检索地址(self, q束: np.ndarray, k: int = 32, N探: int = None) -> dict:
        if not self._质心s:
            return dict(top1=None, 候选k=[])
        n探 = self.N探 if N探 is None else int(N探)
        簇相似 = self._批相似双极(q束, self._质心矩阵缓存())
        簇序 = self._topk序(簇相似, min(n探, len(self._质心s)))
        候选: list = []
        矩阵s: list = []
        for c in 簇序:
            c = int(c)
            候选.extend(self._倒排[c])
            矩阵s.append(self._簇矩阵(c))
        self.统计["探查记录"] += len(候选)
        if not 候选:
            return dict(top1=None, 候选k=[])
        # 簇互斥 ⇒ 免去重；候选/码本按簇序拼接（簇序与倒排序确定性 ⇒ tie 确定性）
        候选码本 = np.concatenate(矩阵s, 0) if len(矩阵s) > 1 else 矩阵s[0]
        r = self._精排(q束, 候选, k, 候选码本)
        r["探查簇"] = [int(c) for c in 簇序]
        return r

    # ── 读②：内容检索（库直扫；部分线索；工作点 M≤8192）──
    def 检索内容(self, q束: np.ndarray, k: int = 32) -> dict:
        if not self._库:
            return dict(top1=None, 候选k=[])
        if self._库矩阵 is None or self._库矩阵.shape[0] != len(self._库):
            self._库矩阵 = np.stack(self._库, 0)
        全相似 = self._批相似双极(q束, self._库矩阵)
        序 = self._topk序(全相似, k)
        候选k = [int(i) for i in 序]
        q_f = self._细码(q束)
        细相似 = np.stack([self._记录细码(i) for i in 候选k], 0) @ q_f / self.D_F
        星 = int(np.argmax(细相似))
        return dict(top1=候选k[星], 细相似=float(细相似[星]),
                    粗top1=候选k[0], 粗相似=float(全相似[候选k[0]]),
                    候选k=候选k)

    # ── 读③：因子化解码（角色桶内找填充）──
    def 解码(self, 记录束: np.ndarray, 角色: str, k: int = 32) -> dict:
        raw = 绑(self.原子("角色|%s" % 角色), 记录束)
        桶 = self._角色桶.get(角色, [])
        if not 桶:
            return dict(top1=None)
        原子s = np.stack([self.原子(键) for 键 in 桶], 0)
        粗相似 = self._批相似双极(raw, 原子s)
        序 = self._topk序(粗相似, min(k, self.N桶))
        候选k = [桶[i] for i in 序]
        raw_f = self._细码(raw)
        细s = []
        for 键 in 候选k:
            c = self._填充细码表.get(键)
            if c is not None:
                self.统计["细码命中"] += 1
                细s.append(c.astype(np.float32))
            else:
                self.统计["细码生成"] += 1
                f = self._细码(self.原子(键))
                self._填充细码表[键] = f.astype(np.int8)
                细s.append(f)
        细相似 = np.stack(细s, 0) @ raw_f / self.D_F
        星 = int(np.argmax(细相似))
        return dict(top1=候选k[星], 细相似=float(细相似[星]),
                    粗top1=候选k[0], 候选k=候选k)

    # ── L2 焦点束 + 显著度门 ──
    def 刷新焦点(self, 候选, 阈值: float = 0.0, K: int = None):
        """候选: [(rec_id, 显著度), ...]；确定性排序 (显著度降序, rec_id 升序) → 取前 K(t)。"""
        K上限 = self.K上限 if K is None else int(K)
        过门 = [(int(i), float(s)) for i, s in 候选 if float(s) > 阈值]
        过门.sort(key=lambda t: (-t[1], t[0]))
        选中 = [i for i, _ in 过门[:K上限]]
        W = 束([self.记录[i]["束"] for i in 选中], self.D) if 选中 else np.zeros(self.D, np.float32)
        return W, 选中

    # ── 库统计 ──
    def 库大小(self) -> int:
        return len(self.记录)

    def 簇分布(self) -> list:
        return sorted(len(v) for v in self._倒排)
