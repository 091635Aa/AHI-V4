# -*- coding: utf-8 -*-
"""meeting28/lang_codec.py —— #216 AHI 对话体：语言↔HDC 编解码器（纯 HDC 零依赖）。

编码：字符基向量（sha256 确定性导出）+ 固定置换位置绑定 + 分块累积 → 1024 双极。
解码：近邻检索（对已注册文本段集合返回最相似段）。
确定性：同 seed 同文本逐位一致。
"""
from __future__ import annotations

import hashlib

import numpy as np

D = 1024
块数 = 8                       # 位置块（置换绑定）


def _基向量(字符: str, seed: int) -> np.ndarray:
    """字符→1024 双极基向量（sha256 确定性导出）。"""
    h = hashlib.sha256(("%d|%s" % (seed, 字符)).encode("utf-8")).digest()
    buf = b""
    while len(buf) < D // 8:
        buf += hashlib.sha256(buf + h).digest()
    bits = np.frombuffer(buf[:D // 8], dtype=np.uint8)
    return ((bits[:, None] >> np.arange(8)[None, :]) & 1).reshape(-1)[:D].astype(np.float32) * 2 - 1


def _置换(seed: int, 块: int) -> np.ndarray:
    h = hashlib.sha256(("%d|perm|%d" % (seed, 块)).encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    return rng.permutation(D)


class 语言编解码器:
    """文本↔1024 双极 HDC（#216）。encode 确定性；decode=注册段近邻检索。"""

    def __init__(self, seed: int = 216):
        self.seed = seed
        self._基: dict = {}                                 # 字符→原始 ±1 基向量（sha256 确定性种子）
        self._语义: dict = {}                               # 字符→实数语义累加器（单位向量；精化更新）
        self._生效: dict = {}                               # 字符→当前生效双极向量（=sign(语义累加器)）
        self._置换s = [_置换(seed, b) for b in range(块数)]
        self._段: list = []                                 # 注册文本段
        self._段向量: list = []                             # 段→向量缓存
        self._段字袋: list = []                             # 段→字袋缓存
        self._共现语料: list = []                           # 精化用共现语料（#222）

    def _原始基(self, c: str) -> np.ndarray:
        """字符→原始 ±1 基向量（sha256 确定性）。只作语义累加器的初值，不受精化影响。"""
        v = self._基.get(c)
        if v is None:
            v = _基向量(c, self.seed)
            self._基[c] = v
        return v

    def _字符向量(self, c: str) -> np.ndarray:
        """字符→**当前生效**双极向量（公开语义面）。

        未精化时 = 原始基（逐位与旧实现一致）；精化后 = sign(语义累加器)——共现诱导的分布式
        语义码。形状恒 (D,)、取值 ∈{±1}，故 编码/字袋/检索 的下游契约不变（C14）。"""
        v = self._生效.get(c)
        if v is None:
            a = self._语义.get(c)
            if a is None:
                a = self._原始基(c).astype(np.float64)
                self._语义[c] = a
            v = np.sign(a).astype(np.float32)
            v[v == 0.0] = 1.0                               # sign(0)→+1（确定性，与旧路径同律）
            self._生效[c] = v
        return v

    def 编码(self, 文本: str) -> np.ndarray:
        """文本→1024 双极：字符 bigram 位置绑定累积（块=位置//len×块数）。"""
        文本 = 文本.strip()
        if not 文本:
            return np.zeros(D, dtype=np.float32)
        累积 = np.zeros(D, dtype=np.float32)
        n = max(len(文本) - 1, 1)
        for i in range(len(文本) - 1):
            v1 = self._字符向量(文本[i])
            v2 = self._字符向量(文本[i + 1])
            块 = min(i * 块数 // max(n, 1), 块数 - 1)
            绑定 = v1 * v2[self._置换s[块]]                  # 位置绑定 bigram
            累积 += 绑定
        单字 = self._字符向量(文本[0]) if len(文本) == 1 else 0.0
        return np.sign(累积 + 单字).astype(np.float32)

    def 注册段(self, 文本: str) -> np.ndarray:
        """注册文本段（供解码双通道检索）；返回其向量。"""
        v = self.编码(文本)
        self._段.append(文本)
        self._段向量.append(v)
        self._段字袋.append(self._字袋(文本))
        return v

    def _字袋(self, 文本: str) -> np.ndarray:
        """字符袋向量（顺序无关——改述鲁棒通道）。"""
        累积 = np.zeros(D, dtype=np.float32)
        for ch in 文本.strip():
            if ch.strip():
                累积 += self._字符向量(ch)
        return np.sign(累积).astype(np.float32)

    def 解码(self, 文本: str) -> str:
        """双通道近邻检索：max(位置 bigram, 字袋)——改述鲁棒（#216 冒烟修订）。"""
        if not self._段:
            return ""
        v = self.编码(文本)
        vbag = self._字袋(文本)
        nv, nb = float(np.linalg.norm(v)) + 1e-9, float(np.linalg.norm(vbag)) + 1e-9
        最佳, 最佳分 = 0, -2.0
        for i, sv in enumerate(self._段向量):
            s1 = float(v @ sv) / (nv * float(np.linalg.norm(sv)) + 1e-9)
            s2 = float(vbag @ self._段字袋[i]) / (
                nb * float(np.linalg.norm(self._段字袋[i])) + 1e-9)
            s = max(s1, s2)
            if s > 最佳分:
                最佳分, 最佳 = s, i
        return self._段[最佳]

    def 相似度(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return float(v1 @ v2) / (float(np.linalg.norm(v1)) * float(np.linalg.norm(v2)) + 1e-9)

    # —— 语义精化（#222 v3）：字符向量向上下文袋捆绑（HDC 语义图，纯确定性零拟合） ——
    def 共现(self, 文本: str) -> None:
        """对话文本进共现统计（不进答案库）——在线语义学习的原料。"""
        self._共现语料.append(文本)

    def 字符相似(self, c1: str, c2: str) -> float:
        return self.相似度(self._字符向量(c1), self._字符向量(c2))

    def 精化(self, α: float = 0.3, 轮数: int = 2, 上限: float = 8.0) -> dict:
        """精化（#222 修复）：共现诱导的**双极语义码**——字符语义累加器按**IDF 加权**上下文累加。

        病（旧实现，实测空操作）：`v_c ← sign(v_c + α·sign(ctx))`，v_c∈{±1}、α=0.3<1
        ⇒ 每分量扰动 0.3 < 基线 |1| ⇒ 符号永不翻转（实测精化前后逐对 Δ=0.000000）；
        且每轮用 `新基 = dict(self._基)` 从原始 ±1 **重启** ⇒ 不跨轮累积。

        修（五点，纯 HDC/零梯度/无外挂模型）：
          ① **不对 ctx 取 sign**：用上下文向量的**加权和**，累计扰动可越过基线 1 ⇒ 符号可真实翻转；
          ② **跨轮累积**（acc 在轮间保持，只受 `上限` 有界化）⇒ 共现证据越多、语义漂移越大；
          ③ **同步更新**：每轮先冻结全部字符生效码快照，再据快照算 ctx 并统一赋值 ⇒ 顺序无关（C13）；
          ④ **上下文按 IDF 加权**（df 由共现语料自身统计，零参数）：公共字符（『的』『是』『(』等）
             df 高 ⇒ 权重≈0，只有**有区分度的字符**参与塑形；
          ⑤ **去公共分量**（每轮把各字符的 ctx 减去它们跨字符的均值）——★实测必需：大语料里所有
             字符的 ctx 共享一个巨大的「语料平均方向」，不减掉就会**全局坍缩**（实测 1046 行源码：
             段间均值 cos 0.31→1.00，任何两条记忆都变得近乎相同 ⇒ 检索失去区分度、系统开始瞎猜）；
             减掉后段间均值 cos 仅 0.31→0.48，且无关查询的最高相似度仍 0.055（远低于阈值 0.30）。
        完成后按新语义重建段缓存（段向量/字袋契约不变）。"""
        import math
        for c in list(self._基.keys()):                     # ① 已见字符累加器初始化（=原始 ±1 基）
            if c not in self._语义:
                self._语义[c] = self._原始基(c).astype(np.float64)
        语料 = [str(t) for t in self._共现语料]
        for t in 语料:                                       #   共现语料里的字符也要有语义态
            for ch in t.strip():
                if ch.strip() and ch not in self._语义:
                    self._语义[ch] = self._原始基(ch).astype(np.float64)
        df: dict = {}                                        # ④ 共现语料 df → idf（数据自定，零参数）
        for t in 语料:
            for ch in set(t):
                if ch.strip():
                    df[ch] = df.get(ch, 0) + 1
        N = max(len(语料), 1)
        idf = {ch: math.log((N + 1.0) / (c + 0.5)) for ch, c in df.items()}
        句表 = []                                            # ★加速(meeting43)：每句 (c集合, 贡献项表) 只建一次
        for t in 语料:
            cs = set(t)                                      #   该句出现的全部字符（**含空格**）——决定"哪些 c 要处理"
            L = [(ch, idf[ch]) for ch in t if ch.strip() and idf.get(ch, 0.0) > 0.0]
            句表.append((cs, L))
        for _ in range(轮数):
            快照 = {}                                        # ② 冻结本轮生效码（同步更新 ⇒ 顺序无关）
            for c, a in self._语义.items():
                v = np.sign(a).astype(np.float32)
                v[v == 0.0] = 1.0
                快照[c] = v
            ctxs = {c: np.zeros(D, dtype=np.float64) for c in self._语义}
            # ★加速(meeting43)：按句处理 + 「总 − 自身项」。ctx_c = Σ_{ch∈L, ch≠c} v_ch = (Σ_{ch∈L} v_ch) − (c 的项)。
            #   每句只需 O(|L|) 次向量加（旧实现是 O(|Cs|·|L|)）⇒ 实测 **5.7×**（1046 行语料 5.43s→0.96s/次）。
            #   数学上等价；实测符号级一致（3 次调用 × 81920 位编码 0 翻转，见 证据_精化加速.md + 判据_语义码 J6）。
            for cs, L in 句表:
                if not L:
                    continue
                Vl = [w * 快照.get(ch, self._原始基(ch)) for ch, w in L]
                tot = np.zeros(D, dtype=np.float64)
                for v in Vl:
                    np.add(tot, v, out=tot)
                出现 = {}
                for (ch, _w), v in zip(L, Vl):
                    出现.setdefault(ch, []).append(v)
                for c in cs:
                    ctx = ctxs.get(c)
                    if ctx is None:
                        continue
                    if c in 出现:
                        sub = np.zeros(D, dtype=np.float64)
                        for v in 出现[c]:
                            np.add(sub, v, out=sub)
                        np.add(ctx, tot, out=ctx)
                        np.subtract(ctx, sub, out=ctx)
                    else:
                        np.add(ctx, tot, out=ctx)
            均 = np.mean(list(ctxs.values()), axis=0)         # ⑤ 语料公共分量（跨字符均值）
            新 = {}
            for c, ctx in ctxs.items():
                if not np.any(ctx):
                    continue
                acc = self._语义[c] + α * (ctx - 均)          # 累加**去公共分量**的 IDF 加权上下文
                np.clip(acc, -abs(上限), abs(上限), out=acc)  # 有界化（确定性）
                新[c] = acc
            for c, a in 新.items():                          # ③ 统一赋值（本轮内互不可见）
                self._语义[c] = a
        self._生效.clear()                                   # ④ 生效向量失效 → 重建段缓存
        self._段向量 = [self.编码(t) for t in self._段]
        self._段字袋 = [self._字袋(t) for t in self._段]
        return dict(字符数=len(self._语义), 共现语料数=len(self._共现语料))

    # —— 持久化 ——
    def 存档(self) -> dict:
        """入档：段 + 段向量 + 语义累加器（新增可选键；旧读方忽略即得旧行为）。"""
        return dict(seed=self.seed, 段=list(self._段),
                    段向量=[v.tolist() for v in self._段向量],
                    语义={c: v.tolist() for c, v in self._语义.items()})

    def 载档(self, d: dict) -> None:
        """出档。**向后兼容**：旧档无 `语义` 键 → 语义态清空（按 seed 重建，行为同旧）；
        新档带 `语义` → 逐位还原（精化后的分布式语义可跨重启恢复）。"""
        self.seed = d["seed"]
        self._段 = list(d["段"])
        self._段向量 = [np.asarray(v, np.float32) for v in d["段向量"]]
        self._基 = {}
        self._生效 = {}
        语义 = d.get("语义")
        self._语义 = ({c: np.asarray(v, np.float64) for c, v in 语义.items()}
                      if 语义 else {})
        self._段字袋 = [self._字袋(t) for t in self._段]     # 与段向量同步重建（旧实现留空 = 潜在失配）
