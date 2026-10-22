# -*- coding: utf-8 -*-
"""AHI4/meeting51/实施/算子_千层骨.py —— ★meeting51 实施：千层（四轴张量）骨架。

━━━ 这一件解决什么 ━━━
会议裁决 2/3/4/5/6 规定：千层 = **四轴张量积**（不是串行堆层），零新算子、零总控、零路由器，
全部新增旁挂默认关，既有 16/10/8 臂口径一字不改。本件把规格落成可跑代码：

  ① **单元参数容器**（规格 §2）：`g_u / τ_u / P_u / l_u / c_u / pro_u` + 慢态 A / 快态 B；
     U = 12 × 6 × 3 × 6 = **1296** 个层单元，共享同一 D 维状态空间（单元只持 g 与稀疏入边 m=4）。
  ② **入边拓扑生成器**（规格 §4）：柱内阶段链 / 前馈 / 反馈 / 横向 四类边，全部由**索引规则**
     确定性生成；**不存在 dict[分区名]→功能 查表**（己3 自扫），分区名字不进入计算（甲4）。
  ③ **前向一拍**（规格 §3，实测修正见下）：六层档**逐档序贯推进**（层档 b 读 b−1）；逐单元重合
     汇集（第八算子「符」）+ 丘脑增益（并入 g）+ 三区室整合（顶树/胞体/输出门，第三次调用既有
     `门`）→ 竞争归一 + 全局抑制 → 慢态 A → 输出 = 门(门控, 符(A,B), B)。
  ④ **耦合实时化**（裁决 4）：α_{u←v} 每拍由 `符` 实时算出（归一化重合强度加静息基线），
     **不查静态权重表**；乙5 反例臂把它换成同分布固定随机 α ⇒ 迁移必须下降，否则撤回本设计。
  ⑤ **三档记忆 + 重放容器**（裁决 5）：回波（泄漏痕）/ 冻结槽位（容量 K、阈值 ζ 解冻、冻结期
     不参与广播）/ 慢态 A / 离线重放（不接输出）；provenance 参考系 `η_u = η_0·(1−pro_u)`。

━━━ 实测修正（相对 `规格_千层张量_四轴.md` §3，逐条登记在 `实施台账_千层.md`）━━━
  R1 汇集式：规格只写 `Σ 符(A_v,B_v)`。实测：纯符合支路会把线性信息在 2~3 档内绞成高阶乘积，
     线性读出无法从深层解出单通道输入（上界≈0.58）。⇒ 改为**双支路**：`源快态 B_v`（直通，
     保信息可达）+ `符(A_v,B_v)`（重合，仍保留规格那条支路）；`符` 另在**耦合门**与**输出门**
     两次承重 ⇒ 第八算子仍是 8 个，不新增算子。
  R2 逐档序贯：规格 §3 的伪码是“同拍全档一次算完”。实测这样深度只等于延迟管线（1 档/拍）。
     ⇒ 按裁决 3“柱内多阶段推进”改为**同一拍内 b=0→5 序贯推进**；环路轮数 = 整链重复次数。
  R3 慢态更新式：规格写 `A ← 软限(A + λ(g⊙B) − 抑制)`。按“自带衰减”改为泄漏积分
     `A ← 软限((1−λ)A + λ(g⊙B) − 抑制)`，λ = 1/τ档；否则快档（λ=1）退化为纯积分不可跟踪。
  R4 逐单元定点归一：每档 B 除以本单元均值幅度（丘脑增益的稳态标定），否则逐档幅度指数衰减。
  R5 拓扑方向：前馈/反馈用**路径制**（a+1 有界 / a−1 有界，不绕环），使 乙4“去反馈”可证伪。

━━━ 铁律对齐 ━━━
C1  零注意力：无全局归一化加权、无查询-键-值三元、无坐标嵌入；耦合只在 m=4 条**稀疏入边**上逐元素算，
    记忆钥匙走「迭代补全 + 阈值门控 + 稀疏存储」而非一步检索；本文件被 `禁词列` 自扫须命中 0。
C9  零 OCR；C10 只 import numpy/hashlib/importlib/os/json（不引任何外挂模型）。
C13 确定性：随机只来自 `default_rng(显式种子)`；状态/参数全走 2^-20 定点格（`_归`/`_快归`）；
    两趟逐位同；`_快归` 与继承 `_归` 在判据里逐位比对。
C14 不破既有：只读 import meeting41 能力核 / meeting49 两区室核（第八算子「符」走继承登记）；
    本件纯新增，既有 16/10/8 臂口径一字不改。

━━━ 直跑 ━━━
    python 算子_千层骨.py          # 冒烟：建 1296 单元 + T1 任务快速训/测 + 计费/指纹
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))                       # …/AHI4/meeting51/实施
AHI4 = os.path.dirname(os.path.dirname(HERE))                           # …/AHI4
_能力核py = os.path.join(AHI4, "meeting41", "实施", "算子_能力核.py")
_两区室py = os.path.join(AHI4, "meeting49", "实施", "算子_两区室核.py")
_环境py = os.path.join(HERE, "环境_千层任务.py")


def _载(路径, 名):
    spec = importlib.util.spec_from_file_location(名, 路径)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_能力 = _载(_能力核py, "能力核_只读_千层")
_V4 = _载(_两区室py, "两区室核_只读_千层")
_环境 = _载(_环境py, "环境_千层任务_只读")

软限 = _能力.软限
_归 = _能力._归
定点 = _能力.定点
界幅 = _能力.界幅
禁词列 = _能力.禁词列
两区室核 = _V4.两区室核
算子_符 = _V4.算子_符


def _快归(v):
    """热路径用的 `_归` 同式快速实现（限幅 → 2^-20 定点格量化）；判据里与继承 `_归` 逐位比对。"""
    a = np.clip(np.asarray(v, dtype=np.float64), -界幅, 界幅)   # clip 同时处理 ±inf
    if not np.isfinite(a).all():                                # 只在出现 NaN 时走慢路径
        a = np.nan_to_num(a, nan=0.0, posinf=界幅, neginf=-界幅)
    return np.rint(a * 定点) / 定点


# ═══════════════════════════════ 冻结常量（判据不据结果改） ═══════════════════════════════
D默认 = 64                            # 行为探针的额定维数（代价臂另按 D=256 解析上报）
轴默认 = dict(A=12, B=6, C=3, D轴=6)   # 四轴：分区×层档×区室×时间尺度 ⇒ U=1296 ≥ 1024
τ档列 = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)   # 轴 D 的 6 档（扫描冻结：见台账 §扫描；非生物测量值）
入边数 = 4                            # m=4（规格 §2 的稀疏投影 P_u）
边类名 = ("柱内", "前馈", "反馈", "横向")
边权初值 = (0.30, 0.60, 0.60, 0.20)   # 四类边权重剖面初值（扫描冻结；见 台账 §扫描）
种子默认 = 51001

# 三档记忆 + 重放（裁决 5）
λ_e默认 = 0.90        # 回波泄漏衰减
槽容量K默认 = 8        # 冻结槽位容量
ζ默认 = 0.35          # 解冻阈值（归一化重合强度）
N_replay默认 = 32     # 离线重放拍数（离线账，不进在线计费）

# 竞争归一（基底节式竞争 + 全局抑制）
β竞争默认 = 0.0       # 阈值 = 均值 + β·标准差
γ_g默认 = 0.0         # 全局抑制系数
竞争模式列 = ("全局", "单元", "关")

# 学习率（慢可塑由 provenance 调制：η_u = η_0·(1−pro_u)）
η读出默认 = 0.01
η_g默认 = 0.002
η_pro默认 = 0.01
g下界, g上界 = 0.05, 4.00
输入维默认 = _环境.输入维

# 开关默认（全部新增**默认关**：主路只开阅读所需；拆件/反例臂显式开）
开关默认 = dict(
    竞争模式="全局",
    β=β竞争默认,
    γ_g=γ_g默认,
    环路=1,              # 每拍整链重复轮数（乙3：1 vs 多）
    柱内链=True,         # 拆轴B：False ⇒ 断柱内阶段链（各档改接外部）
    边类=(True, True, True),        # 前馈 / 反馈 / 横向
    动态耦合=True,       # 乙5 反例：False ⇒ 固定同分布 α
    槽位=True,           # 丙2/丙5：冻结槽位
    回波=True,           # 三档记忆之回波
    槽广播=False,        # 丙2 反例：冻结槽位参与广播（“高激活保持”变体）
    重放=False,          # 丙3：离线重放
    重放接输出=False,    # 丙3 反例臂
    读全局池=False,      # 拆轴A：readout 不再按分区隔离
    合并区室=False,      # 拆轴C：三区室规则统一
    统一时档=False,      # 拆轴D：单一时间常数
    τ档=None,            # 时间档取值（None ⇒ 模块级 τ档列；扫描冻结用）
    外部全档=True,       # 外部输入直灌所有层档（多档并联特征库）；拆轴B 时关掉
    读出层档=(0,),       # 读出取哪些层档（默认=接收层；按层档拼接可扩多尺度特征库）
    读出区室=(2,),       # 读出取哪些区室（默认=输出门）
    读出时档=(0,),       # 读出取哪些时间档（默认=最快档；池=均值）
    η读出=None,          # 读出学习率（None ⇒ 模块级 η读出默认；扫描冻结用）
    g学=False,           # 主任务默认不学 g（避免特征非平稳）；丙4 漂移探针显式开
    pro学=True,
    读出学=True,
)


# ═══════════════════════════════ 单元参数容器（规格 §2） ═══════════════════════════════
class 层单元容器:
    """唯一新增数据结构：g/τ/P/l/c/pro + 两递推态 A/B（规格 §2，“容器不是算子”）。

    · 全部参数按**单元号 u** 一维展开；数值一律走 `_归` 2^-20 定点格（C13）。
    · `P` 用稀疏表示：`入源[u, k]` = 第 k 条入边的源单元号（−1 表示无边），`入类[u, k]` = 边类。
    · 四轴坐标 `分区/层/区室/时档` 只是参数寻址，**名字/标签不进入计算**（甲4/己3）。
    """

    def __init__(self, D, 轴, 种子, 开关):
        self.D = int(D)
        self.轴 = dict(轴)
        A, B, C, Dd = self.轴["A"], self.轴["B"], self.轴["C"], self.轴["D轴"]
        self.U = int(A * B * C * Dd)
        self.开关 = dict(开关)
        网格 = np.indices((A, B, C, Dd)).reshape(4, -1)
        self.分区 = 网格[0].astype(np.int64)
        self.层 = 网格[1].astype(np.int64)
        self.区室 = 网格[2].astype(np.int64)
        self.时档 = 网格[3].astype(np.int64)
        rng = np.random.default_rng(int(种子))
        # ① 逐元素增益 g（含丘脑增益档 κ_th），可塑、慢
        self.g = _归(1.0 + 0.05 * rng.standard_normal((self.U, self.D)))
        # ② 边类权重剖面（复刻“连接统计”：四类边各有一条逐元素权重；不是 U×U 表）
        self.边权 = _归(np.asarray(边权初值)[:, None]
                       * (1.0 + 0.10 * rng.standard_normal((入边数, self.D))))
        # ③ 稀疏入边拓扑（P_u）：见 _建拓扑
        self.入源 = -np.ones((self.U, 入边数), dtype=np.int64)
        self.入类 = np.zeros((self.U, 入边数), dtype=np.int64)
        self._建拓扑()
        # ④ 外部输入投影（单元参数，不是算子）：层档 0 的“接收”单元必接外部；
        #    `外部全档` 时每个层档的单元也有自己的外部投影（多尺度并联特征库，见 R6）。
        self.外单元 = np.where(self.层 == 0)[0]
        self.外分区 = self.分区[self.外单元]
        self.输入维 = int(输入维默认)
        # ON 型感受野：投影取 |·|（极性别一致）⇒ 逐档*乘积特征的单项式系数在各维上同号，
        # 线性读出才能把它们累加起来（实测 R6：混合号时 2 阶乘积上界 0.80，ON 型 1.00）。
        self.W_in = _归(np.abs(rng.standard_normal((self.外单元.size, self.D, self.输入维)))
                        / np.sqrt(float(self.输入维)))
        self.W_in全 = _归(np.abs(rng.standard_normal((self.U, self.D, self.输入维)))
                          / np.sqrt(float(self.输入维)))
        self.外分区全 = self.分区.copy()
        # ⑤ 读出单元（分区 × 读出层档 × 读出区室 × 读出时档），池=均值，再按层档拼接
        读层0 = tuple(self.开关["读出层档"])
        读层 = tuple(sorted(set(min(int(b), self.轴["B"] - 1) for b in 读层0)))   # 结构消融下自动夹取
        读区 = tuple(self.开关["读出区室"])
        读时 = tuple(self.开关["读出时档"])
        self.读出层 = 读层
        self.读单元 = np.array([[[[self._号(a, b, cc, dd) for dd in 读时] for cc in 读区]
                                 for b in 读层] for a in range(A)], dtype=np.int64)   # (A, L, C, T)
        self.读单元平 = self.读单元.reshape(A, -1)
        # ⑥ provenance ∈ [0,1]（自源经历累积占比；初值 0 —— 由经历长出来，不按名字分配）
        self.pro = np.zeros(self.U, dtype=np.float64)
        # ⑦ 两递推态
        self.A慢 = np.zeros((self.U, self.D), dtype=np.float64)
        self.B快 = np.zeros((self.U, self.D), dtype=np.float64)
        self.输出 = np.zeros((self.U, self.D), dtype=np.float64)
        # ⑧ 投影（记忆任务：把外部模式投影到共享 D 空间；纯随机、非学习）
        self.P_入 = _归(rng.standard_normal((self.D, self.轴["A"] * self.输入维))
                        / np.sqrt(float(self.轴["A"] * self.输入维)))

    def _号(self, a, b, c, d):
        B, C, Dd = self.轴["B"], self.轴["C"], self.轴["D轴"]
        return int(((a * B + b) * C + c) * Dd + d)

    def _号向量(self, a, b, c, d):
        B, C, Dd = self.轴["B"], self.轴["C"], self.轴["D轴"]
        return np.asarray(((a * B + b) * C + c) * Dd + d, dtype=np.int64)

    def _建拓扑(self):
        """四类边（规格 §4）：柱内阶段链 / 前馈 / 反馈 / 横向。全部由索引规则确定性生成。"""
        A, B, C, Dd = self.轴["A"], self.轴["B"], self.轴["C"], self.轴["D轴"]
        a, b, c, d = self.分区, self.层, self.区室, self.时档
        # slot0：柱内阶段链（同分区/区室/时间档，层档 −1）——轴 B 的“多阶段推进”
        if self.开关["柱内链"] and B > 1:
            有 = b > 0
            self.入源[有, 0] = self._号向量(a[有], b[有] - 1, c[有], d[有])
            self.入类[有, 0] = 0
        # slot1：前馈（下标更大的分区、前一层档 → 本档；L2/3 → L4 的不对称规则）
        if self.开关["边类"][0] and A > 1:
            有前 = a + 1 < A
            self.入源[有前, 1] = self._号向量(a[有前] + 1, np.maximum(0, b[有前] - 1),
                                          c[有前], d[有前])
            self.入类[有前, 1] = 1
        # slot2：反馈（下标更小的分区、前一层档 → 本档；L5/6 → 低区 L1/L2/3 的不对称规则）
        if self.开关["边类"][1] and A > 1:
            有反 = a - 1 >= 0
            self.入源[有反, 2] = self._号向量(a[有反] - 1, np.maximum(0, b[有反] - 1),
                                          c[有反], d[有反])
            self.入类[有反, 2] = 2
        # slot3：横向（同分区：跨区室 + 跨时间档的侧向竞争，提供 top-k 稀疏与多尺度混合）
        if self.开关["边类"][2]:
            源 = self._号向量(a, b, (c + 1) % C, (d + 1) % Dd)
            不同 = 源 != np.arange(self.U)
            self.入源[不同, 3] = 源[不同]
            self.入类[不同, 3] = 3
        self.入源 = np.where(self.入源 < 0, -1, self.入源)
        # 预计算：逐（层档, 边槽）的 目标行/源/边类（拓扑静态，省热路径掩码开销）
        self.层槽 = []
        for b_ in range(B):
            行b = np.where(self.层 == b_)[0]
            self.层槽.append([(np.where(self.入源[行b, k] >= 0)[0],
                              self.入源[行b, k][self.入源[行b, k] >= 0],
                              self.入类[行b, k][self.入源[行b, k] >= 0])
                             for k in range(入边数)])
        self.层行 = [np.where(self.层 == b_)[0] for b_ in range(B)]
        self.掩区0 = self.区室 == 0
        self.掩区1 = self.区室 == 1
        self.掩区2 = self.区室 == 2

    def 摘要(self):
        return dict(U=int(self.U), D=int(self.D), 轴=dict(self.轴),
                    参数总量=int(self.U * self.D + 入边数 * self.D + self.W_in.size + self.W_in全.size),
                    定点存储MB=round(float(self.U * self.D + 入边数 * self.D + self.W_in.size
                                          + self.W_in全.size) * 4 / 1e6, 3),
                    边数=int(np.sum(self.入源 >= 0)), 入边数=入边数,
                    外部单元数=int(self.外单元.size), 读出单元数=int(self.读单元平.shape[1]),
                    读出层档=[int(x) for x in self.读出层],
                    层档数=int(self.轴["B"]), 区室数=int(self.轴["C"]), 时间档数=int(self.轴["D轴"]),
                    分区数=int(self.轴["A"]))


# ═══════════════════════════════ 三档记忆 + 重放容器（规格 §5） ═══════════════════════════════
class 记忆容器:
    """回波（泄漏痕）/ 冻结槽位（容量 K、阈值 ζ 解冻、冻结期不广播）/ 慢态（在单元容器里）。"""

    def __init__(self, D, K=槽容量K默认, λ_e=λ_e默认, ζ=ζ默认):
        self.D = int(D)
        self.K = int(K)
        self.λ_e = float(λ_e)
        self.ζ = float(ζ)
        self.痕 = np.zeros(self.D, dtype=np.float64)            # 回波
        self.槽 = np.zeros((self.K, self.D), dtype=np.float64)  # 冻结槽位
        self.槽有效 = np.zeros(self.K, dtype=bool)
        self.槽用 = np.zeros(self.K, dtype=np.int64)
        self.槽重 = np.zeros(self.K, dtype=np.float64)
        self.广播 = np.zeros(self.D, dtype=np.float64)          # 本拍广播激活（冻结期不写入）

    def 回波(self, 激活):
        self.痕 = _快归(self.λ_e * self.痕 + np.asarray(激活, dtype=np.float64))
        return self.痕

    def 冻结(self, 快照):
        快照 = _快归(np.asarray(快照, dtype=np.float64))
        空 = np.where(~self.槽有效)[0]
        k = int(空[0]) if 空.size else int(np.argmin(self.槽用))
        self.槽[k] = _快归(0.999 * self.槽[k] + 快照)
        self.槽有效[k] = True
        self.槽用[k] = 0
        self.槽重[k] = 0.0
        return int(k)

    def 重合强度(self, 线索, 符):
        线 = _快归(np.asarray(线索, dtype=np.float64))
        线尺 = float(np.mean(np.abs(线))) + 1e-9
        强 = np.full(self.K, -1e9, dtype=np.float64)
        for k in range(self.K):
            if not self.槽有效[k]:
                continue
            槽尺 = float(np.mean(np.abs(self.槽[k]))) + 1e-9
            强[k] = float(np.mean(符(线 / 线尺, self.槽[k] / 槽尺)))
        return 强

    def 解冻(self, 线索, 符, 迭代=1, ζ=None):
        """迭代补全（C1 红线：迭代 + 阈值门控 + 稀疏存储，不做一步全局加权检索）。

        每轮：算归一化重合 → 阈值门控选出激活槽 → 用**最强槽**补全线索引（top-1 稀疏补全）→ 下一轮。
        """
        ζ = self.ζ if ζ is None else float(ζ)
        x = _快归(np.asarray(线索, dtype=np.float64))
        for _ in range(max(1, int(迭代))):
            强 = self.重合强度(x, 符)
            选 = np.where(强 >= ζ)[0]
            if 选.size == 0:
                选 = np.array([int(np.argmax(强))], dtype=np.int64)
            self.槽重[选] = 强[选]
            self.槽用[选] += 1
            k = int(选[int(np.argmax(强[选]))])                   # top-1 补全（不动别的槽）
            x = _快归(0.5 * x + 0.5 * self.槽[k])
        return x, self.重合强度(x, 符)

    def 快照(self):
        return dict(痕=self.痕.copy(), 槽=self.槽.copy(), 槽有效=self.槽有效.copy(),
                    槽用=self.槽用.copy(), 槽重=self.槽重.copy(), λ_e=self.λ_e, K=self.K, ζ=self.ζ)


# ═══════════════════════════════ 千层核（四轴张量积） ═══════════════════════════════
class 千层核(两区室核):
    """★千层骨架：继承 两区室核（七原语 + 第八算子「符」），把层单元容器接进主路。"""

    def __init__(self, D=D默认, 种子=种子默认, 轴=None, 开关=None):
        轴 = dict(轴默认) if 轴 is None else dict(轴默认, **dict(轴))
        开关 = dict(开关默认) if 开关 is None else dict(开关默认, **dict(开关))
        super().__init__(D=int(D), 种子=int(种子), 模式="两区室", 慢率=0.12, 学模式="credit")
        self.轴 = 轴
        self.开关 = 开关
        self.模式 = "千层"
        # 第八算子「符」：沿用继承登记的单位标度（A/B 逐单元归一后量级 ~1 ⇒ 乘积项与直通项同量级，
        # 否则高阶乘积支路被 1/16 压成噪声 —— 实测修正 R1 的配套标定）
        self.登记算子(算子_符, np.ones(self.D))
        self.容器 = 层单元容器(self.D, self.轴, int(种子), self.开关)
        rng = np.random.default_rng(int(种子) + 977)
        读维 = int(len(self.容器.读出层)) * self.D
        self.读出 = _快归(rng.standard_normal((self.轴["A"], 2, 读维)) / np.sqrt(float(读维)))
        self.读出偏 = np.zeros((self.轴["A"], 2), dtype=np.float64)   # 读出偏置（线性分类截距）
        # 读出前稳态归一（R7）：逐特征维的运行均值/方差（确定性、无随机；不是算子）
        self.均 = np.zeros((self.轴["A"], 读维), dtype=np.float64)
        self.方 = np.ones((self.轴["A"], 读维), dtype=np.float64)
        self.记忆 = 记忆容器(self.D)
        self.最近池 = np.zeros((self.轴["A"], self.D), dtype=np.float64)
        self.最近快 = np.zeros((self.容器.U, self.D), dtype=np.float64)
        self.最近输出 = np.zeros((self.容器.U, self.D), dtype=np.float64)
        self.最近活性 = np.zeros(self.D, dtype=np.float64)
        self.最近抑制 = 0.0
        self.最近汇 = np.zeros((self.容器.U, self.D), dtype=np.float64)
        self.α固定 = None
        self.重放计数 = 0

    # ── 便捷属性 ──
    @property
    def U(self):
        return self.容器.U

    def 单元坐标(self, u):
        c = self.容器
        return int(c.分区[u]), int(c.层[u]), int(c.区室[u]), int(c.时档[u])

    # ── 第八算子「符」的热路径同式实现（语义与继承 `符` 同：归(W符⊙a⊙b)；判据逐位比对）──
    def 符(self, a, b):
        return _快归(self.W表[算子_符] * np.asarray(a, dtype=np.float64)
                     * np.asarray(b, dtype=np.float64))

    # ── 既有 `门` 的批量化同式（g⊙x + (1−g)⊙y）；判据里逐行真调用继承的 `门` 复核 ──
    @staticmethod
    def 门批量(门控, x, y):
        g = np.clip(np.asarray(门控, dtype=np.float64), 0.0, 1.0)
        return _快归(g * np.asarray(x, dtype=np.float64)
                     + (1.0 - g) * np.asarray(y, dtype=np.float64))

    # ── 时间档 → 慢态更新率 λ ──
    def _λ(self):
        列 = τ档列 if self.开关.get("τ档") is None else tuple(self.开关["τ档"])
        if self.开关["统一时档"]:
            return np.full(self.容器.U, 1.0 / float(列[2]))   # 拆轴D：单一 τ
        return 1.0 / np.asarray(列, dtype=np.float64)[self.容器.时档]

    # ── 耦合权：α = 静息基线 + 归一化重合强度（逐元素、每拍实时算；裁决 4）──
    def _耦合(self, 目标A, 源B):
        门 = self.符(目标A, 源B)
        尺 = np.mean(np.abs(门), axis=1, keepdims=True) + 1e-9
        return _快归(1.0 + 0.5 * 软限(门 / 尺, 1.0))            # ∈(0.5,1.5)

    def 物化固定α(self, x列, 拍数=32, 种子偏=0):
        """乙5 反例臂：采样真实 α 的**边际分布**，用同分布固定随机值替换（同种子可复现）。"""
        rng = np.random.default_rng(int(self.种子) + 7701 + int(种子偏))
        样本 = [self._前向内部(x列[t])["α样本"] for t in range(int(拍数))]
        α样本 = np.concatenate(样本, axis=0)
        均值 = np.mean(α样本, axis=0)
        标度 = np.std(α样本, axis=0) + 1e-9
        c = self.容器
        self.α固定 = _快归(均值 + 标度 * rng.standard_normal((c.U, 入边数, self.D)))
        return dict(均值=均值.tolist(), 标度=标度.tolist(), 形状=[int(c.U), 入边数, int(self.D)])

    # ── 竞争归一（基底节式：top-k + 全局抑制）──
    def _竞争(self, B快):
        模 = self.开关["竞争模式"]
        if 模 == "关":
            return B快, np.mean(np.abs(B快), axis=0), 0.0
        if 模 == "全局":
            活性 = np.mean(np.abs(B快), axis=0)                 # (D,)
            阈 = float(np.mean(活性) + self.开关["β"] * np.std(活性))
            掩 = 活性 >= 阈
            抑制 = float(self.开关["γ_g"] * np.mean(活性))
            return _快归(B快 * 掩), 活性, 抑制
        活 = np.abs(B快)
        阈 = np.mean(活, axis=1, keepdims=True) + self.开关["β"] * np.std(活, axis=1, keepdims=True)
        掩 = 活 >= 阈
        return _快归(B快 * 掩), np.mean(活, axis=0), float(self.开关["γ_g"] * np.mean(活))

    # ── 三区室 B 规则（顶树-情境 / 胞体-感觉 / 输出门）──
    def _算B层(self, 汇, 外部, 旧慢, 区室, g行):
        """对一个层档的行（n,D）算快态 B；区室决定整合式（第三次调用既有 `门`）。"""
        if self.开关["合并区室"]:                                # 拆轴C：统一用顶树规则
            B = 软限(外部 + 0.5 * 汇)
        else:
            B = np.zeros_like(汇)
            m0 = 区室 == 0
            m1 = 区室 == 1
            m2 = 区室 == 2
            if np.any(m0):
                B[m0] = 软限(外部[m0] + 0.5 * 汇[m0])
            if np.any(m1):
                B[m1] = 软限(外部[m1] + 汇[m1])
            if np.any(m2):
                门控 = np.clip(0.5 + 0.5 * 软限(旧慢[m2]), 0.0, 1.0)
                B[m2] = self.门批量(门控, 软限(外部[m2] + 汇[m2]), 软限(汇[m2]))
        # 逐单元定点归一（丘脑增益稳态标定）⇒ 信号跨层档不衰减（R4）
        B /= (np.mean(np.abs(B), axis=1, keepdims=True) + 1e-9)
        return _快归(g行 * B)

    # ── 前向内部：逐层档序贯推进（R2）──
    def _前向内部(self, x列=None):
        c = self.容器
        外部项 = np.zeros((c.U, self.D), dtype=np.float64)
        if x列 is not None and c.外单元.size:
            x = np.asarray(x列, dtype=np.float64)
            if x.ndim == 1:
                x = x.reshape(-1, c.输入维)
            if self.开关["外部全档"]:
                外部项 = 软限((c.W_in全 * x[c.外分区全][:, None, :]).sum(axis=2), 界幅)
            else:
                外 = 软限((c.W_in * x[c.外分区][:, None, :]).sum(axis=2), 界幅)
                外部项[c.外单元] = 外
        B = c.B快
        A = c.A慢
        g = c.g
        汇全 = self.最近汇
        α样本 = []
        for _ in range(max(1, int(self.开关["环路"]))):
            for b_ in range(self.轴["B"]):
                行 = c.层行[b_]
                if 行.size == 0:
                    continue
                汇 = np.zeros((行.size, self.D), dtype=np.float64)
                for k in range(入边数):
                    本行, 源, 类 = c.层槽[b_][k]
                    if 本行.size == 0:
                        continue
                    源快 = B[源]
                    符v = self.符(A[源], 源快)                # 重合检测支路（第八算子）
                    if self.开关["动态耦合"]:
                        α = self._耦合(A[行[本行]], 源快)     # 耦合权每拍实时算
                        α样本.append(α[::32])
                    else:
                        if self.α固定 is None:
                            raise RuntimeError("动态耦合=False 需先 物化固定α()")
                        α = self.α固定[行[本行], k]
                    汇[本行] += c.边权[类] * α * (源快 + 符v)   # 双支路：直通 + 重合（R1）
                B[行] = self._算B层(汇, 外部项[行], A[行], c.区室[行], g[行])
                汇全[行] = _快归(汇)
        B, 活性, 抑制 = self._竞争(B)
        λ = self._λ()
        A = _快归(软限((1.0 - λ)[:, None] * A + λ[:, None] * (g * B) - 抑制))
        c.A慢 = A
        c.B快 = B
        符出 = self.符(A, B)
        # ★ 输出门（末区室）：对“重合输出”与“原始驱动”再做一次 `门`（第三次调用，非新算子）
        输出 = np.array(符出, dtype=np.float64, copy=True)
        m2 = c.掩区2 if not self.开关["合并区室"] else np.zeros(c.U, dtype=bool)
        if np.any(m2):
            门控2 = np.clip(0.5 + 0.5 * 软限(A[m2]), 0.0, 1.0)
            输出[m2] = self.门批量(门控2, 符出[m2], B[m2])
        c.输出 = _快归(输出)
        self.最近快 = B
        self.最近输出 = c.输出
        self.最近活性 = 活性
        self.最近抑制 = 抑制
        return dict(α样本=np.concatenate(α样本, axis=0) if α样本 else np.zeros((0, self.D)))

    # ── 读出：逐层档池（分区 × 区室 × 时档 的输出均值）→ 按层档拼接 ──
    def _读池(self):
        c = self.容器
        段 = []
        for i in range(len(c.读出层)):
            段.append(np.mean(c.输出[c.读单元[:, i, :, :].reshape(self.轴["A"], -1)], axis=1))
        池 = np.concatenate(段, axis=1)                          # (A, L·D)
        if self.开关["读全局池"]:
            池 = np.tile(np.mean(池, axis=0)[None, :], (self.轴["A"], 1))
        # 逐层段归一（每 D 维一段），保证多层拼接量级一致
        for i in range(len(c.读出层)):
            块 = 池[:, i * self.D:(i + 1) * self.D]
            池[:, i * self.D:(i + 1) * self.D] = 块 / (np.mean(np.abs(块), axis=1, keepdims=True) + 1e-9)
        池 = _快归(池)
        # 稳态归一（R7，运行统计；感知机的可学性靠这一步——实测未归一时训练内一直停在 0.51）
        偏 = 池 - self.均
        self.均 = _快归(self.均 + 0.02 * 偏)
        self.方 = _快归(self.方 + 0.02 * (偏 * 偏 - self.方))
        return _快归(偏 / (np.sqrt(np.abs(self.方)) + 1e-6))

    # ── 前向一拍（对外接口）──
    def 前向一拍(self, x列=None, 更新记忆=True):
        self._前向内部(x列)
        池 = self._读池()
        if self.开关["槽广播"] and np.any(self.记忆.槽有效):
            池 = _快归(池 + 1.0 * np.mean(self.记忆.槽[self.记忆.槽有效], axis=0)[None, :])
        self.最近池 = 池
        得分 = (self.读出 * 池[:, None, :]).sum(axis=2) + self.读出偏
        预测 = np.argmax(得分, axis=1)
        if 更新记忆 and self.开关["回波"]:
            广播 = _快归(np.mean(self.最近快, axis=0))
            self.记忆.广播 = 广播
            self.记忆.回波(广播)
        return dict(预测=预测, 池=池, 得分=得分, 输出=self.最近输出, 快态=self.最近快,
                    慢态=self.容器.A慢, 活性=self.最近活性, 抑制=self.最近抑制)

    # ── 学习一拍：读出（判错才动）+ 三因子慢可塑（η 由 pro 调制）+ provenance ──
    def 学一拍(self, 标签列):
        c = self.容器
        A_ = self.轴["A"]
        池 = self.最近池
        得分 = (self.读出 * 池[:, None, :]).sum(axis=2) + self.读出偏
        预测 = np.argmax(得分, axis=1)
        正确 = np.zeros(A_, dtype=bool)
        if self.开关["读出学"]:
            # 误差驱动的最小二乘随机梯度（LMS / δ 规则）：目标 one-hot，误差 = 目标 − 得分。
            # 只用内部预测误差与当前特征，不用外部模型（C10）；定点归一保确定（C13）。
            目标 = np.zeros_like(得分)
            目标[np.arange(A_), np.asarray(标签列, dtype=np.int64)] = 1.0
            误 = 目标 - 得分
            η = η读出默认 if self.开关.get("η读出") is None else float(self.开关["η读出"])
            读出 = self.读出 + η * 误[:, :, None] * 池[:, None, :]
            偏 = self.读出偏 + η * 误
            self.读出 = _快归(读出)
            self.读出偏 = _快归(偏)
        正确[:] = (预测 == np.asarray(标签列, dtype=np.int64))
        绝 = np.abs(self.最近输出)
        均 = float(np.mean(绝)) + 1e-9
        用 = np.clip(np.mean(绝, axis=1) / 均, 0.0, 4.0)
        正确扩 = 正确[c.分区]
        if self.开关["pro学"]:
            # provenance = 自源经历累积占比的**泄漏估计**：答对时按“被用到多少”累积，答错时回落。
            # 用泄漏（而非纯累加）才可能有单元间差异（纯累加会全部饱和到 1，参考系消失）。
            c.pro = np.clip((1.0 - η_pro默认) * c.pro
                            + η_pro默认 * (正确扩 * 用 - 0.5 * (~正确扩) * 用), 0.0, 1.0)
        if self.开关["g学"]:
            δ = np.where(正确扩, 1.0, -1.0)
            迹 = (np.abs(self.最近快) - np.mean(np.abs(self.最近快), axis=1, keepdims=True)) \
                / (np.mean(np.abs(self.最近快), axis=1, keepdims=True) + 1e-9)
            η_u = η_g默认 * (1.0 - c.pro)
            c.g = _快归(np.clip(c.g + η_u[:, None] * δ[:, None] * 迹, g下界, g上界))
        return dict(预测=预测, 正确=正确)

    # ── 训练/测试（任务胶水；判据与自测同一口径）──
    def 训测(self, 序列, 训拍, 测拍, 更新记忆=True, 学=True, 预热=0):
        """预热（只前向，等慢态收敛）→ 训拍 学习 → 测拍 **冻结读出** 评估（慢态/记忆仍在线）。"""
        输入, 目标 = 序列["输入"], 序列["目标"]
        n = min(int(序列["长度"]) - 1, int(预热) + int(训拍) + int(测拍))
        总 = 对 = 训总 = 训对 = 0
        起 = int(预热)
        for t in range(n):
            果 = self.前向一拍(输入[t], 更新记忆=更新记忆)
            if t < 起:
                continue
            if t < 起 + int(训拍):
                if 学:
                    预 = self.学一拍(目标[t])["预测"]
                    训对 += int(np.sum(预 == 目标[t]))
                    训总 += int(预.size)
                continue
            对 += int(np.sum(果["预测"] == 目标[t]))
            总 += int(果["预测"].size)
        return dict(训练准确率=float(训对) / max(1, 训总), 测试准确率=float(对) / max(1, 总),
                    训练拍=int(训拍), 测试拍=int(总 // max(1, self.轴["A"])), 预热拍=int(预热))

    def 跑序列(self, 序列, 学习=True, 从=0, 到=None):
        输入, 目标 = 序列["输入"], 序列["目标"]
        n = min(int(序列["长度"]) - 1, int(到 if 到 is not None else 序列["长度"] - 1))
        预列 = np.zeros((n - int(从), self.轴["A"]), dtype=np.int64)
        对 = 0
        for i, t in enumerate(range(int(从), n)):
            果 = self.前向一拍(输入[t])
            if 学习:
                果 = self.学一拍(目标[t])
            预列[i] = 果["预测"]
            对 += int(np.sum(果["预测"] == 目标[t]))
        return dict(预测=预列, 目标=目标[int(从):n], 准确率=float(对) / max(1, 预列.size))

    # ── 记忆任务：投影 / 冻结 / 解冻 ──
    def 投影模式(self, 模式行):
        c = self.容器
        平 = np.asarray(模式行, dtype=np.float64).ravel()
        if 平.size != c.轴["A"] * c.输入维:
            raise ValueError("模式长度须为 %d，收到 %d" % (c.轴["A"] * c.输入维, 平.size))
        return _快归(c.P_入 @ 平)

    def 冻结模式(self, 模式行):
        return self.记忆.冻结(self.投影模式(模式行))

    def 解冻召回(self, 模式行, 迭代=1, ζ=None):
        return self.记忆.解冻(self.投影模式(模式行), self.符, 迭代=迭代, ζ=ζ)

    def 召回模式(self, 模式行, 迭代=1, ζ=None):
        _, 强 = self.解冻召回(模式行, 迭代=迭代, ζ=ζ)
        return int(np.argmax(强)), 强

    # ── 竞争胜出探针（乙2）：两路同时进入竞争层，微弱偏置 Δ 可复现切换胜者 ──
    def 竞争胜出(self, 左, 右, 偏置=0.0):
        左 = _快归(np.asarray(左, dtype=np.float64).ravel())
        右 = _快归(np.asarray(右, dtype=np.float64).ravel() + float(偏置))
        Bm, 活性, 抑制 = self._竞争(np.stack([左, 右], axis=0))
        左活 = float(np.mean(np.abs(Bm[0])))
        右活 = float(np.mean(np.abs(Bm[1])))
        return dict(胜="左" if 左活 >= 右活 else "右", 左=左活, 右=右活,
                    差=abs(左活 - 右活), 掩维=int(np.sum(活性 >= np.mean(活性))), 抑制=float(抑制))

    def 召回判定(self, 模式行, 迭代=1, 门控=ζ默认):
        """阈值门控召回：最强重合 < 门控 ⇒ 判“未召回”（−1，计数错）。"""
        选, 强 = self.召回模式(模式行, 迭代=迭代, ζ=门控)
        return (-1 if float(强[选]) < float(门控) else int(选)), float(强[选])

    # ── 单单元冻结（甲1）：把某单元的逐元素增益 g_u 置零（该单元失活，其余不动）──
    def 冻结单元(self, u):
        u = int(u)
        self.容器.g[u] = 0.0
        return u

    def 解冻单元(self, u, g旧):
        self.容器.g[int(u)] = _快归(np.asarray(g旧, dtype=np.float64))

    # ── 置零单元入边（乙1 无总控检验）：把某单元全部入边切断并重建缓存 ──
    def 置零入边(self, u):
        u = int(u)
        self.容器.入源[u] = -1
        self._重建拓扑缓存()
        return int(np.sum(self.容器.入源[u] >= 0))

    def _重建拓扑缓存(self):
        c = self.容器
        B = self.轴["B"]
        c.层槽 = []
        for b_ in range(B):
            行b = np.where(c.层 == b_)[0]
            c.层槽.append([(np.where(c.入源[行b, k] >= 0)[0],
                            c.入源[行b, k][c.入源[行b, k] >= 0],
                            c.入类[行b, k][c.入源[行b, k] >= 0])
                           for k in range(入边数)])

    # ── 名字无关性（甲4）：分区标签只是显示用；重命名不得改变任何数值 ──
    def 分区标签(self):
        if not hasattr(self, "_分区标签"):
            self._分区标签 = ["分区%02d" % (i + 1) for i in range(self.轴["A"])]
        return list(self._分区标签)

    def 重命名分区(self, 种子偏=3):
        rng = np.random.default_rng(int(self.种子) + 8801 + int(种子偏))
        旧 = self.分区标签()
        self._分区标签 = [str(旧[i]) for i in rng.permutation(self.轴["A"])]
        return list(self._分区标签)

    # ── 拆件回滚（戊2）：保存/恢复开关，避免拆件污染后续臂 ──
    def 存开关(self):
        return dict(self.开关)

    def 复开关(self, 快照):
        self.开关.clear()
        self.开关.update(快照)
        return self.开关

    # ── 离线重放（丙3）：按优先级抽槽快照重跑主循环，只更新慢系统，不接输出 ──
    def 重放(self, 序列, 拍数=N_replay默认, 接输出=False):
        if not self.开关["重放"]:
            return dict(已重放=0, 接输出=False)
        槽 = [k for k in range(self.记忆.K) if self.记忆.槽有效[k]]
        if not 槽:
            return dict(已重放=0, 接输出=False)
        rng = np.random.default_rng(int(self.种子) + 3301)
        输入, 目标 = 序列["输入"], 序列["目标"]
        已 = 0
        for _ in range(int(拍数)):
            k = int(槽[int(rng.integers(0, len(槽)))])
            self.记忆.槽用[k] += 1
            t = int(rng.integers(0, max(1, int(序列["长度"]) - 1)))
            self.前向一拍(输入[t], 更新记忆=False)
            self.学一拍(目标[t])
            已 += 1
        self.重放计数 += 已
        return dict(已重放=int(已), 接输出=bool(接输出))

    # ── 拆件（戊2）：功能级拆轴（保留已学参数，只断该轴的功能通道）──
    拆件名 = ("轴A分区", "轴B层档", "轴C区室", "轴D时档")

    def 拆件(self, 名):
        名 = str(名)
        if 名 == "轴A分区":
            self.开关["读全局池"] = True          # 分区隔离失效：readout 读全域池
        elif 名 == "轴B层档":
            self.开关["柱内链"] = False           # 柱内阶段链断开
            self.开关["外部全档"] = False         # 且深层失去外部直灌 ⇒ 只剩接收层（单档）
        elif 名 == "轴C区室":
            self.开关["合并区室"] = True          # 三区室规则统一
        elif 名 == "轴D时档":
            self.开关["统一时档"] = True          # 单一时间常数
        else:
            raise KeyError("未知拆件：%r（可用：%s）" % (名, "、".join(self.拆件名)))
        return self.开关

    def _给各档接外部(self):
        """拆轴B 的配套：柱内链断开后，给所有层档的单元直灌外部输入（阶段组合→并联）。"""
        c = self.容器
        if not hasattr(c, "W_in全"):
            rng = np.random.default_rng(int(self.种子) + 6607)
            c.W_in全 = _快归(rng.standard_normal((c.U, c.D, c.输入维))
                             / np.sqrt(float(c.输入维)))
            c.外分区全 = c.分区

    # ── 反焊死件：分区映射置乱（戊1 的“同参数量、无分区结构”基线）──
    def 置乱分区映射(self, 种子偏=1):
        """只打乱“谁能读谁”的分区映射（参数/拓扑不变）⇒ 同参数量、无分区结构基线。"""
        c = self.容器
        rng = np.random.default_rng(int(self.种子) + 4401 + int(种子偏))
        c.分区 = rng.permutation(c.分区).astype(np.int64)
        c.外分区 = c.分区[c.外单元]
        c.外分区全 = c.分区.copy()
        return c.分区.copy()

    # ── 确定性指纹（C13 丁2）──
    def 指纹(self):
        h = hashlib.sha256()
        for 名 in ("g", "边权", "W_in", "W_in全", "pro", "A慢", "B快", "输出", "P_入"):
            h.update(np.ascontiguousarray(getattr(self.容器, 名)).tobytes())
        h.update(np.ascontiguousarray(self.读出).tobytes())
        h.update(np.ascontiguousarray(self.读出偏).tobytes())
        h.update(np.ascontiguousarray(self.均).tobytes())
        h.update(np.ascontiguousarray(self.方).tobytes())
        h.update(np.ascontiguousarray(self.记忆.槽).tobytes())
        return h.hexdigest()

    # ── 代价（丁1）：O(U·D)，不含 U² ──
    def 计费千层(self, 拍数=1, 单元数=None, D=None, 环路=None):
        U = int(self.容器.U) if 单元数 is None else int(单元数)
        D = int(self.D) if D is None else int(D)
        H = max(1, int(self.开关["环路"])) if 环路 is None else max(1, int(环路))
        m = 入边数
        每拍 = U * D * (2 * m + 11) * H + 4 * D
        return dict(乘加数=int(int(拍数) * 每拍 + 4 * D), 每拍乘加=int(每拍), 单元数=U, 维数=D,
                    环路=H, 阶=1, 口径="乘加数 = 拍数·U·D·(2m+11)·H + 4D；不含 U²")

    def 参数总量(self):
        return int(self.容器.U * self.D + 入边数 * self.D + self.容器.W_in.size
                   + self.容器.W_in全.size + self.读出.size + self.读出偏.size + self.容器.P_入.size)


# ═══════════════════════════════ 自测（直跑观测用） ═══════════════════════════════
def 自测():
    核 = 千层核(D=D默认, 种子=种子默认)
    序 = _环境.生成序列_乘积(_环境.种子环境默认, 长度=2600, 阶=2, 噪率=0.02)
    果 = 核.训测(序, 训拍=1600, 测拍=600)
    return dict(容器=核.容器.摘要(), 主臂=果, 算子表=核.算子表(), 参数总量=核.参数总量(),
                计费=核.计费千层(拍数=1), 指纹=核.指纹()[:16])


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    np.seterr(all="ignore")
    print(json.dumps(自测(), ensure_ascii=False, indent=1, default=str)[:4000])
