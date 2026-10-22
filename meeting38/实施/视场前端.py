# -*- coding: utf-8 -*-
"""meeting38/实施/视场前端.py —— 视觉会议步2：注视–预测–眼跳前端（P1/P2/P3）。

依据：meeting38/会议_视觉/00_会议纪要.md §3（主持人合议版）。
铁律：C1 禁 Transformer ｜ C9 **禁 OCR** ｜ C10 禁外挂模型当器官 ｜ C13 确定性。
本模块**纯新增**，不 import / 不修改 器官_眼.py 与任何既有引擎文件（只读 态.编码、写 态 视场槽）。

「像人一样看」= 受注意调制的、注视限制的采样轨迹；每拍交出去
  (下一注视点, 惊讶/预测误差, 保拓扑视觉场)，而不是单帧全域解析出的离散框列表。

三个组织原语（全由现有算子合成，零新数学、零手工特征）：
  P1 视场码 = 束( [绑(位置原子_p, 粗码_p)]_全部格  +  [绑(位置原子_f, 细码_f)]_中央凹 )
  P2 误差场 = 逐格精确 绑(粗码_t, 粗码_上) → 意外；全局 束_p(绑(位置原子_p, Δ_p))
  P3 眼跳链 = argmax( w1·意外 + w2·显著度 + w3·意图偏置 − 抑制返回 )（tie 按格序，确定性）

★ 实现期修正（步2 判据实测，2026-10-05）：
  首版按"中央凹 1728 维 / 外周 108 维"两套分辨率喂养 `编码器.特征码`，实测**每次眼跳都会
  产生假意外**——因为 `特征码` 的投影阵**按输入维度缓存**，同一未变块在"进入中央凹"时
  换了投影 → 码改变 → 被判为变化。修法：**惊讶通道统一分辨率**（全部格用同一 8×8×3=192 维
  粗码、同一投影），中央凹改为**附加细节通道**（16×16×3=768 维细码，只进 P1 视场码，
  **不进误差场**）。既保留"中央凹看得更细"，又消除眼跳伪影。

★ 步3 接线期修正（2026-10-05，接线预检实测：单拍 70ms、热点=48 格池化被算两遍+轴序不优）：
  ① `粗码s` 与 `显著度场` 改为共用 `_粗块s`（一拍内只池化一次）；
  ② `_均值池` 归约轴转置到连续尾轴（快 ~1.4×）。
  **两处均逐位不变**——步2 五判据数值原样复现（对照 `步2_视场判据.json`，接线判据实测：
  `json` 深度相等为真）；前端单拍实测 70ms → 26.3ms（P95 33.9ms）。属性能修正非语义修正。

★ 步3.6 分辨率与扫视修正（2026-10-05，所有者复看演示录像后反馈"注视只在两三格间来回跳、像素太粗"）：
  预检量化根因（实施/预检_网格成本.py，可复跑）：
    ① **2-循环**：返回抑制只标记当前格，且 0.55^8≈0.008 在注视超时(8拍)前衰减归零
       → 换点时上一格已无抑制 → argmax 又回到对比度最高的同一格；实测唯一注视格
       `8×6/0.55` 仅 **5 格**（正是"E1↔E3 来回跳"）。
    ② **网格太粗**：8×6=48 格、每格 240×180px（1080p），结构落在同格内不可分。
  修法（全部经预检实测择优，非猜）：
    ① `抑制衰减 0.55 → 0.97`（半衰期 ~23 拍 ≈ 1.5s@15Hz，返回抑制跨越多次注视）：
       唯一注视格 5 → **25**（16×9 网格、400 拍实测；0.55=4、0.88=10、0.93=16）。
    ② `格X, 格Y = 8, 6 → 16, 9`（144 格；区域 1080×720 单拍 31.8ms/P95 37.6ms，仍守 15Hz 预算 67ms）。
    ③ `W邻近 = 0.05` 扫视邻近偏好（人眼幅度偏置）：平坦误差场下旧实现按"格序首个"破平，
       下一注视与注视历史无关（步2 判据 像人-1 在细网格下实测 Δ列=0 失败）；加邻近偏好后
       平坦场里"从当前注视就近选点"，下一注视真正依赖注视历史（Δ列=15 实测通过）。
  影响面（诚实登记）：粗码维度不变（仍 8×8×3=192），但格数变化 → 步2/步3 判据需按新网格重跑；
  行为变化属**预登记的修复目标**（"扫视要散开、要依赖历史"），不是无意漂移。

关键设计（对应被否路线的规避）：
  · **零手工特征**：块的编码直接交给引擎既有 `编码器.特征码`（随机投影=感受野），作用在原始像素块上。
  · **视网膜光适应**：每块先做局部归一化（减块均值、除块标准差）——全局明度渐变被消掉（视而不见），
    局部结构变化保留（判据 像人-2 的机制基础）。
  · **注视保持律**：无可测变化（max 逐格意外 < 在场阈值）时**不眼跳**（人眼对可预测世界的"视而不见"）；
    仅当出现超阈变化或注视超时才换点。
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# 数轴 动态导入（复用 meeting32/声带.py，不改其文件；照 情感因子层.py 的做法）
_AHI3 = os.path.dirname(os.path.dirname(HERE))               # AHI3 根（迁移：随文件定位）
_spec = importlib.util.spec_from_file_location(
    "声带mod", os.path.join(_AHI3, "meeting32", "声带.py"))
_声带mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_声带mod)
数轴 = _声带mod.数轴

格X, 格Y = 16, 9            # 网格 16×9 = 144 格（步3.6 加密；区域 1080×720 单拍 31.8ms 实测）
中央凹半径 = 1              # 中央凹 = 注视格 ±1 → 3×3
粗分辨率 = 8                # 惊讶通道：全部格统一 8×8×3 = 192 维（同投影，无眼跳伪影）
细分辨率 = 16               # 中央凹细节通道：16×16×3 = 768 维（只进视场码）
W意外, W显著, W意图 = 1.0, 0.3, 0.6
W邻近 = 0.05                # 步3.6：扫视邻近偏好（人眼幅度偏置：偏好近处落点；平坦场时据此破平，
                            # 使"下一注视"真正依赖注视历史——步2 判据 像人-1 的要求）
焦点σ = 1.5                 # 步5 §2.3：焦点加权带宽（格）。所有者「焦点是注意力放在那……把那区域
                            # 像素优先级抬高，从 1.0 变成 1.2」——只抬评分优先级，不减信息。
焦点增益 = 0.2              # 焦点处 1.0→1.2、远处→1.0（乘性增益；周边仍照常产生意外/候选，非裁剪）
抑制返回系数 = 0.6
抑制衰减 = 0.97             # 步3.6：0.55→0.97（跨多次注视存活；修"两格来回跳"2-循环，见文件头）
在场阈值 = 0.35             # 超阈变化才算"看到变化"（触发眼跳/在场候选）
注视超时 = 8                # 注视超过此拍数也允许换点


def _均值池(图: np.ndarray, 因子: int) -> np.ndarray:
    """均值池化（确定性；整数因子）。支持 HxW 与 HxWxC。

    ★ 步3 性能修正：归约轴转置到连续尾轴（reshape→transpose→copy→mean(尾两轴)），
    比原「reshape 后 mean(axis=(1,3))」快约 1.4×。**逐位不变**：输入为整数像素
    （uint8→float32，值≤255），部分和 ≤ f²·255 远小于 2²⁴（float32 精确整数域），
    故任意归约顺序得到同一精确和、同一除法舍入。接线判据实测：6 组形状逐位一致 +
    步2 五判据数值原样复现。"""
    图 = np.asarray(图, dtype=np.float32)
    f = max(int(因子), 1)
    if 图.ndim == 2:
        h, w = 图.shape
        h2, w2 = (h // f) * f, (w // f) * f
        return (图[:h2, :w2].reshape(h2 // f, f, w2 // f, f)
                .transpose(0, 2, 1, 3).copy().mean(axis=(2, 3)))
    h, w, c = 图.shape
    h2, w2 = (h // f) * f, (w // f) * f
    return (图[:h2, :w2].reshape(h2 // f, f, w2 // f, f, c)
            .transpose(0, 2, 4, 1, 3).copy().mean(axis=(3, 4)))


def _池到(区: np.ndarray, t: int) -> np.ndarray:
    """区域池化并居中裁剪到 t×t×C（自适应、确定性）。"""
    区 = np.asarray(区, dtype=np.float32)
    h, w = 区.shape[:2]
    f = max(1, min(h // t, w // t))
    p = _均值池(区, f)
    hh, ww = p.shape[:2]
    y0, x0 = max(0, (hh - t) // 2), max(0, (ww - t) // 2)
    return p[y0:y0 + t, x0:x0 + t]


def _光适应(块: np.ndarray) -> np.ndarray:
    """视网膜局部归一化（光适应/对比归一）：减块均值、除块标准差。
    全局明度渐变 → 归一化后不变 → 不产生意外（视而不见）；局部结构变化 → 保留。"""
    b = np.asarray(块, dtype=np.float32) / 255.0
    b = b - float(b.mean())
    s = float(b.std())
    return b / (s if s > 1e-6 else 1.0)


class 视场前端:
    """注视–预测–眼跳前端。眼单写视场槽；本类不做任何器官调用（只读写共享工作区）。"""

    def __init__(self, 态, 种子: int = 3800, 格x: int = 格X, 格y: int = 格Y):
        self.态 = 态
        self.格x, self.格y = int(格x), int(格y)
        self.种子 = int(种子)
        self.编码 = 态.编码
        self.轴x = 数轴("视场|x", self.编码.D, 16, 16, 0.0, max(self.格x - 1, 1))
        self.轴y = 数轴("视场|y", self.编码.D, 16, 16, 0.0, max(self.格y - 1, 1))
        self._位置原子: dict = {}
        for by in range(self.格y):
            for bx in range(self.格x):
                self._位置原子[(bx, by)] = self.编码.绑(self.轴x.编码向量(bx),
                                                        self.轴y.编码向量(by))
        self.注视 = (self.格x // 2, self.格y // 2)
        self.注视停留 = 0
        self.眼跳数 = 0
        self._上粗码: dict = {}
        self.抑制: np.ndarray = np.zeros(self.格x * self.格y, dtype=np.float32)
        self._屏幕 = (192, 144)
        self.统计 = dict(拍=0, 眼跳=0)

    # ── 块切分 ──
    def _块(self, 帧: np.ndarray, bx: int, by: int, t: int) -> np.ndarray:
        H, W = 帧.shape[:2]
        y0, y1 = int(by * H / self.格y), int((by + 1) * H / self.格y)
        x0, x1 = int(bx * W / self.格x), int((bx + 1) * W / self.格x)
        区 = 帧[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
        return _池到(区, t)

    def _粗块s(self, 帧: np.ndarray) -> dict:
        """全部格统一分辨率的粗块（惊讶通道与显著度共用）。
        ★ 步3 性能修正（2026-10-05，接线预检实测：单拍 70ms，热点=分块池化被算了两遍）：
        `粗码s` 与 `显著度场` 原先各自做一次 48 格池化；本函数让一拍内只算一次。
        **逐位不变**：同帧同块同一算子，分段值一字不差（步2 五判据数值原样复现为准）。"""
        return {(bx, by): self._块(帧, bx, by, 粗分辨率)
                for by in range(self.格y) for bx in range(self.格x)}

    def _在中央凹(self, bx: int, by: int) -> bool:
        gx, gy = self.注视
        return abs(bx - gx) <= 中央凹半径 and abs(by - gy) <= 中央凹半径

    def 粗码s(self, 帧: np.ndarray, 块s: dict = None) -> dict:
        """惊讶通道：**全部格统一分辨率与投影**（无眼跳伪影）。
        块s：可选预算分块（一拍内与 显著度场 共用；同帧同值，逐位不变）。"""
        if 块s is None:
            块s = self._粗块s(帧)
        return {(bx, by): self.编码.特征码(
            _光适应(块s[(bx, by)]).reshape(-1))
            for (bx, by) in 块s}

    def 细码s(self, 帧: np.ndarray) -> dict:
        """中央凹细节通道：仅注视邻域 3×3，高分辨率（只进视场码，不进误差）。"""
        return {(bx, by): self.编码.特征码(
            _光适应(self._块(帧, bx, by, 细分辨率)).reshape(-1))
            for by in range(self.格y) for bx in range(self.格x)
            if self._在中央凹(bx, by)}

    # ── P1 视场码 ──
    def 视场码(self, 粗码: dict, 细码: dict = None) -> np.ndarray:
        """束( [绑(位置,粗码)]_全部 + [绑(位置,细码)]_中央凹 )——保拓扑连续场。"""
        项s = [self.编码.绑(self._位置原子[(bx, by)], 粗码[(bx, by)]) for (bx, by) in 粗码]
        if 细码:
            项s += [self.编码.绑(self._位置原子[(bx, by)], 细码[(bx, by)]) for (bx, by) in 细码]
        return self.编码.束(项s)

    # ── P2 误差场 ──
    def 误差场(self, 粗码: dict) -> tuple:
        """(误差场, 逐格意外[48], 全局意外度)。逐格精确：Δ=绑(粗码_t, 粗码_上)。

        意外 = clip(1 − mean(Δ), 0, 1)：**恒等(全 +1)→0；完全无关(mean≈0)→1（上限）**。
        ★ 公式修正（步2 判据实测，2026-10-05）：01 号专家给的是 `0.5·(1−mean)`，其数学上限
        仅为 **0.5**（两张无关码 mean≈0），却同时把判据写成"惊讶峰 ≥0.5"——公式与判据自相矛盾。
        本实现取 `1−mean`（上限 1.0，0=无变化/1=完全无关），使语义与预登记判据自洽；
        预登记判据阈值**未改**（这是推导勘误，不是放宽门槛）。"""
        逐格 = np.zeros(self.格x * self.格y, dtype=np.float32)
        Δs = {}
        for (bx, by), c in 粗码.items():
            上 = self._上粗码.get((bx, by))
            if 上 is None:
                意外, Δ = 0.0, c
            else:
                Δ = self.编码.绑(c, 上)
                意外 = 1.0 - float(np.mean(Δ))
            逐格[by * self.格x + bx] = max(0.0, min(1.0, 意外))
            Δs[(bx, by)] = Δ
        场 = self.编码.束([self.编码.绑(self._位置原子[(bx, by)], Δs[(bx, by)])
                          for (bx, by) in Δs])
        return 场, 逐格, float(逐格.mean())

    # ── P3 眼跳链 ──
    def 邻近场(self) -> np.ndarray:
        """扫视邻近偏好场（0..1，当前注视处=1、最远角=0）。人眼扫视幅度偏置的极小化实现：
        平坦显著场时不至于"按格序乱跳"，而是就近选点——同时使下一注视依赖注视历史。"""
        gx, gy = self.注视
        X = np.arange(self.格x, dtype=np.float32) - float(gx)
        Y = np.arange(self.格y, dtype=np.float32) - float(gy)
        D = np.sqrt(X[None, :] ** 2 + Y[:, None] ** 2)
        最大 = float(np.hypot(max(self.格x - 1, 1), max(self.格y - 1, 1)))
        return (1.0 - D / 最大).astype(np.float32).ravel()

    def 焦点增益场(self) -> np.ndarray:
        """焦点=乘性权重（步5 §2.3；所有者 2026-10-06「焦点是注意力放在那……把那区域像素
        优先级抬高，从 1.0 变成 1.2；旁边发生什么我还是能看到」）：
          增益 = 1.0 + 焦点增益·exp(−d² / 2σ²)   # d=格到**焦点**(当前注视)的距离
        焦点处 1.2、远处 → 1.0。**只抬评分优先级、不减信息**：周边格照常产生意外、照常进
        候选与元素表（非裁剪）——这正是"焦点 ≠ 全局只看那块"的实现。"""
        gx, gy = self.注视
        X = np.arange(self.格x, dtype=np.float32) - float(gx)
        Y = np.arange(self.格y, dtype=np.float32) - float(gy)
        D2 = X[None, :] ** 2 + Y[:, None] ** 2
        return (1.0 + 焦点增益 * np.exp(-D2 / (2.0 * 焦点σ ** 2))).astype(np.float32).ravel()

    def 下一注视(self, 逐格意外: np.ndarray, 显著度: np.ndarray = None,
                 意图偏置: np.ndarray = None) -> tuple:
        s = W意外 * 逐格意外.astype(np.float32)
        if 显著度 is not None:
            s = s + W显著 * 显著度.astype(np.float32)
        if 意图偏置 is not None:
            s = s + W意图 * 意图偏置.astype(np.float32)
        s = s - 抑制返回系数 * self.抑制
        s = s + W邻近 * self.邻近场()             # 步3.6：邻近偏好（平坦场破平靠注视历史）
        s = s * self.焦点增益场()                  # 步5 §2.3：焦点乘性加权（焦点 1.2×；不减信息）
        i = int(np.argmax(s))                     # 首个最大 → tie 按格序（确定性）
        return (i % self.格x, i // self.格x)

    def 显著度场(self, 帧: np.ndarray, 块s: dict = None) -> np.ndarray:
        """局部对比（粗块标准差/255）——纯算术，非手工特征。
        块s：可选预算分块（与 粗码s 共用同一次池化；逐位不变）。"""
        if 块s is None:
            块s = self._粗块s(帧)
        return np.array([float((np.asarray(块s[(bx, by)], dtype=np.float32) / 255.0).std())
                         for by in range(self.格y) for bx in range(self.格x)],
                        dtype=np.float32)

    def 意图偏置场(self) -> np.ndarray:
        """自上而下偏置场（加性双源）：
          ① 意图槽.当前.目标点（手要去哪，眼看哪）权重 1.0；
          ② 目标槽.心（步8 教学指认：我在找那里）权重 0.8 —— **加性常驻**：探索意图不断时
             目光仍会扫向被指认处（否则死锁：图标不进场→注意选不到→永远不看它）。
        ★ 步8 修复潜伏断线：意图字段是「目标点」，旧代码读「心/点」→ 联动从未生效。"""
        出 = np.zeros(self.格x * self.格y, dtype=np.float32)
        try:
            W, H = self._屏幕

            def _点入(点, 权):
                if 点:
                    bx = int(min(max(点[0] / max(W, 1) * self.格x, 0), self.格x - 1))
                    by = int(min(max(点[1] / max(H, 1) * self.格y, 0), self.格y - 1))
                    出[by * self.格x + bx] += 权

            意 = self.态.取意图()
            当 = 意.get("当前") or {}
            _点入(当.get("目标点") or 当.get("心") or 当.get("点"), 1.0)
            目 = self.态.取目标()
            _点入(目.get("心"), 0.8)
        except Exception:
            pass
        return 出

    # ── 一拍 ──
    def 一拍(self, 帧: np.ndarray, 屏幕=None) -> dict:
        self._屏幕 = 屏幕 or (帧.shape[1], 帧.shape[0])
        粗块s = self._粗块s(帧)                    # 一拍内 48 格池化只算一次（两通道共用）
        粗 = self.粗码s(帧, 粗块s)
        细 = self.细码s(帧)
        场 = self.视场码(粗, 细)
        误差场, 逐格, 意外度 = self.误差场(粗)
        显著 = self.显著度场(帧, 粗块s)
        偏置 = self.意图偏置场()
        self.注视停留 += 1
        # 注视保持律：无可测变化（低于在场阈值）且未超时 → 不眼跳（"视而不见"）
        最意外 = float(逐格.max())
        if (最意外 >= 在场阈值) or (self.注视停留 >= 注视超时):
            新 = self.下一注视(逐格, 显著, 偏置)
            if 新 != self.注视:
                self.注视 = 新
                self.眼跳数 += 1
                self.统计["眼跳"] += 1
                self.注视停留 = 0
        self.统计["拍"] += 1
        self.抑制 *= 抑制衰减
        gx, gy = self.注视
        self.抑制[gy * self.格x + gx] = 1.0
        self._上粗码 = 粗
        self.态.写视场(视场=场, 注视=(gx, gy, self.注视停留), 误差场=误差场,
                       意外度=意外度, 逐格意外=逐格, 眼跳数=self.眼跳数)
        return dict(视场=场, 注视=self.注视, 误差场=误差场, 意外度=意外度,
                    逐格意外=逐格, 最意外=最意外, 眼跳数=self.眼跳数)

    # ── 在场者涌现（步3 接器官_眼 时用；纯函数，无副作用）──
    def 在场者(self, 逐格意外: np.ndarray, 阈值: float = 在场阈值) -> list:
        """逐格意外>阈值 → 4-邻域 BFS 连通 → 区域。48 格小图上做，非 cv2。"""
        命中 = 逐格意外.reshape(self.格y, self.格x) > 阈值
        已访 = np.zeros_like(命中, dtype=bool)
        区s = []
        for by in range(self.格y):
            for bx in range(self.格x):
                if not 命中[by][bx] or 已访[by][bx]:
                    continue
                栈, 集合 = [(bx, by)], []
                已访[by][bx] = True
                while 栈:
                    cx, cy = 栈.pop()
                    集合.append((cx, cy))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = cx + dx, cy + dy
                        if (0 <= nx < self.格x and 0 <= ny < self.格y
                                and 命中[ny][nx] and not 已访[ny][nx]):
                            已访[ny][nx] = True
                            栈.append((nx, ny))
                xs = [c[0] for c in 集合]; ys = [c[1] for c in 集合]
                格 = (min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)
                区s.append(dict(格=格, 中心=((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0),
                                意外=float(max(逐格意外[cy * self.格x + cx]
                                              for cx, cy in 集合))))
        区s.sort(key=lambda r: (-r["意外"], r["格"]))     # 确定性
        return 区s