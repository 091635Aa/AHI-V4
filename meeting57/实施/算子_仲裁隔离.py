# -*- coding: utf-8 -*-
"""AHI4/meeting57/实施/算子_仲裁隔离.py —— P1：内部仲裁隔离（A′ / B′ / S′ 三臂）。

━━━ 这一件回答什么 ━━━
同一连续 VSA 状态（D=2048）、同一 P1 合成任务，只改「仲裁方式」：

  A′ 同一连续状态连续自省：事件直接写入（不审权限）；每轮自省把当前读回再写回
     （自指回路；读回=解绑+硬阈值迭代吸引子，无独立审判）；
  B′ 事件触发：只读快照（np.copy 状态）→ 独立审判器（只读快照+当前事件；
     权限优先、同权限取新、冲突时低权限被拒）→ 受控写回（先擦除旧值的绑定再写新值）；
     无新证据的自省轮不写回；
  S′ B′ 架构 + 惰性/sham 审判：不看权限、按新近接受（其余同 B′）——零假设对照。

三臂共享**同一个** `连续状态` 类与同一 `码本` / `吸引子清理` 回读；
判据另行计算：语义漂移率、震荡幅度、冲突后错误率、错误传播率、更正延迟、复合仲裁分。

━━━ 铁律对齐 ━━━
C1  无 S 字头全局归一化、无 QK^T、无全局注意力：回读 = 解绑 + **硬阈值迭代吸引子**
    （Hopfield 式 sign 定点迭代，固定码本，非全局归一化检索）；判据另做源码级禁词自扫。
C13 随机只来自 np.random.default_rng(显式种子)；所有写入经 `_归`（2^-20 定点格）量化；
    同种子两趟逐位同。
C14 纯新增；只读 import meeting41 的 `_归`（不修改既有内核）；不 import P0 私有类。
本件不含学习、不含外挂模型、不含 OCR；只做合成任务上的存储/仲裁机制对照。

━━━ 直跑（冒烟；dev seed 100，不是判据/留出集）━━━
    python 算子_仲裁隔离.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AHI4 = os.path.dirname(os.path.dirname(HERE))
_能力核py = os.path.join(AHI4, "meeting41", "实施", "算子_能力核.py")


def _载(路径, 名):
    spec = importlib.util.spec_from_file_location(名, 路径)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_能力 = _载(_能力核py, "能力核_只读_仲裁隔离")
_归 = _能力._归

# ── 冻结常量（dev 100–103 校准；判据集/留出集只判不调）──
D默认 = 2048
ρ默认 = 0.995
状态幅 = 8.0
吸引子步数 = 2
回读阈值 = 0.10
事件写权 = 1.0
自省强度 = 1.0
关键键 = ("goal", "constraint", "entity")

默认参数 = dict(D=D默认, ρ=ρ默认, 状态幅=状态幅, 吸引子步数=吸引子步数,
                回读阈值=回读阈值, 事件写权=事件写权, 自省强度=自省强度)


def 探测键列(任务) -> list:
    """探测键 = 关键键 + 稳定细节键 + 矛盾细节键（顺序固定）。"""
    out = list(任务["关键键列"])
    for k in 任务["稳定键列"]:
        if str(k) not in out:
            out.append(str(k))
    for k in 任务["矛盾键列"]:
        if str(k) not in out:
            out.append(str(k))
    return out


def 全键列(任务) -> list:
    keys = []
    for 列 in (任务["细节键列"], 任务["关键键列"], 任务["干扰键列"]):
        for k in 列:
            if str(k) not in keys:
                keys.append(str(k))
    return keys


def 按步(任务) -> tuple:
    """把事件列/查询列按步分组（保持原有顺序）。"""
    事 = {}
    查 = {}
    for e in 任务["事件列"]:
        事.setdefault(int(e["步"]), []).append(e)
    for q in 任务["查询列"]:
        查.setdefault(int(q["步"]), []).append(q)
    return 事, 查


# ═════════════════════════════ 码本（自包含；同 P0 口径，不 import P0 私有类） ═════════════════════════════
class 码本:
    """确定性双极码本：键码 / 值码（值码固定 0..15）。"""

    def __init__(self, 键列, D=D默认, 种子=0):
        self.D = int(D)
        self.种子 = int(种子)
        self.键列 = sorted({str(k) for k in 键列})
        rng = np.random.default_rng(self.种子 + 7717)
        self.键码 = {}
        for k in self.键列:
            self.键码[k] = (np.where(rng.integers(0, 2, size=self.D) == 1, 1.0, -1.0)
                          / np.sqrt(float(self.D)))
        self.值码 = (np.where(rng.integers(0, 2, size=(16, self.D)) == 1, 1.0, -1.0)
                   / np.sqrt(float(self.D)))

    def 取键码(self, 键):
        return self.键码[str(键)]

    def 绑(self, 键, 值):
        """k ⊗ v，乘 D 后逐元素 ∈ {−1,+1}；不是注意力/不是外积矩阵。"""
        return self.取键码(键) * self.值码[int(值)] * float(self.D)


def 吸引子清理(u, 值码, 步数=吸引子步数):
    """硬阈值迭代吸引子（Hopfield 式 sign 迭代；无 S 字头归一化、无全局注意力）。

    对解绑向量做 x_{t+1}=sign(Σ_i (v_i·x_t) v_i) 的定点迭代；
    最后把 ‖x‖ 归一到 √D（Rademacher 尺度），便于与值码做内积读出。
    """
    x = np.asarray(u, dtype=np.float64).copy()
    V = 值码.shape[0]
    for _ in range(max(1, int(步数))):
        s = 值码 @ x
        x = 值码.T @ s / float(V)
        x = np.where(x >= 0.0, 1.0, -1.0)                  # 硬阈值，非 S 字头归一化
    n = float(np.linalg.norm(x)) / np.sqrt(float(x.shape[0]))
    if not np.isfinite(n) or n < 1e-12:
        return np.zeros_like(x)
    return x / n


# ═════════════════════════════ 三臂共享的连续状态 ═════════════════════════════
class 连续状态:
    """三臂共享的连续 VSA 状态（D=2048）+ 同一码本/吸引子回读。

    只提供「怎么存/怎么读」；**不提供**「该不该写」——仲裁策略由各臂决定。
    """

    def __init__(self, 键列, D=D默认, 种子=0, ρ=ρ默认, 状态幅=状态幅,
                 吸引子步数=吸引子步数, 回读阈值=回读阈值):
        self.D = int(D)
        self.种子 = int(种子)
        self.ρ = float(ρ)
        self.状态幅 = float(状态幅)
        self.清理步数 = int(吸引子步数)
        self.阈值 = float(回读阈值)
        self.码 = 码本(键列, D=self.D, 种子=self.种子)
        self.h = np.zeros(self.D, dtype=np.float64)

    # ── 快照（审判器的只读输入；用 np.copy，审判器不得改原状态）──
    def 快照(self):
        return np.copy(self.h)

    # ── 叠加写：事件/自省直接累加（不擦除旧值）──
    def 叠加写(self, 键, 值, 权=事件写权):
        k = str(键)
        if k not in self.码.键码:
            return
        self.h = _归(self.h + float(权) * self.码.绑(k, 值), 幅=64.0)

    # ── 受控写回：先擦除旧值的绑定，再写新值（旧值=None 时只写）──
    def 受控写(self, 键, 新值, 旧值=None, 权=事件写权):
        k = str(键)
        if k not in self.码.键码:
            return
        w = float(权)
        if 旧值 is not None and int(旧值) != int(新值):
            self.h = _归(self.h - w * self.码.绑(k, int(旧值)), 幅=64.0)
        self.h = _归(self.h + w * self.码.绑(k, int(新值)), 幅=64.0)

    # ── 自省回写：把当前读回按强度写回（自指回路）──
    def 自省回写(self, 键, 值, 强度=自省强度):
        self.叠加写(键, 值, 权=float(强度))

    def 步进(self):
        self.h = _归(self.ρ * self.h, 幅=64.0)
        m = float(np.max(np.abs(self.h))) if self.h.size else 0.0
        if m > self.状态幅:
            self.h = _归(self.h * (self.状态幅 / m), 幅=64.0)

    def 读(self, 键, 快照=None):
        k = str(键)
        if k not in self.码.键码:
            return None
        h = self.h if 快照 is None else 快照
        u = self.码.取键码(k) * h
        if self.清理步数 <= 0:
            s = self.码.值码 @ u
        else:
            x = 吸引子清理(u, self.码.值码, self.清理步数)
            s = self.码.值码 @ x
        i = int(np.argmax(s))
        if float(s[i]) < self.阈值:
            return None
        return int(i)

    def 指纹(self):
        return "%.10f|%.10f" % (float(np.sum(self.h)), float(np.sum(self.h * self.h)))


# ═════════════════════════════ 审判器（B′/S′ 的仲裁核） ═════════════════════════════
class 独立审判器:
    """B′ 的独立审判器：只读快照 + 当前事件；权限优先、同权限取新、冲突时低权限被拒。

    审判器自己的状态只有「已受秩表」：键 → (权限, 步, 序号)。这是仲裁元数据，
    不复制记忆内容（记忆内容仍在共享连续状态里）。
    """

    名 = "独立审判器"

    def __init__(self):
        self.已受 = {}

    def 秩(self, 事件):
        return (int(事件["权限"]), int(事件["步"]), int(事件.get("序号", 0)))

    def 判定(self, 快照, 状态, 事件):
        """返回 None（拒写）或 dict(键, 新值, 旧值, 快照值, 秩)。"""
        if not bool(事件.get("任务相关", True)):
            return None                                   # 干扰：非证据，不入记忆
        k = str(事件["键"])
        if k not in 状态.码.键码:
            return None
        秩新 = self.秩(事件)
        if k in self.已受:
            秩旧 = self.已受[k][0]
            if 秩新 <= 秩旧:
                return None                               # 低权限被拒 / 同权限但不更新
            旧值 = self.已受[k][1]
        else:
            旧值 = None
        快照值 = 状态.读(k, 快照=快照)                     # 只读快照确认当前记忆
        return dict(键=k, 新值=int(事件["值"]), 旧值=旧值,
                    快照值=快照值, 秩=秩新)

    def 登记(self, 结果):
        self.已受[结果["键"]] = (tuple(结果["秩"]), int(结果["新值"]))


class Sham审判器(独立审判器):
    """S′ 的 sham 审判器：不看权限，按新近接受（惰性跳过：同拍或不更新则跳过）。"""

    名 = "Sham审判器"

    def 秩(self, 事件):
        return (int(事件["步"]), int(事件.get("序号", 0)))   # 故意丢掉权限位

    def 判定(self, 快照, 状态, 事件):
        if not bool(事件.get("任务相关", True)):
            return None
        k = str(事件["键"])
        if k not in 状态.码.键码:
            return None
        秩新 = self.秩(事件)
        if k in self.已受:
            秩旧 = self.已受[k][0]
            if 秩新 <= 秩旧:
                return None
            旧值 = self.已受[k][1]
        else:
            旧值 = None
        快照值 = 状态.读(k, 快照=快照)
        if 快照值 is not None and int(快照值) == int(事件["值"]):
            return None                                   # 惰性：与当前记忆相同则不写
        return dict(键=k, 新值=int(事件["值"]), 旧值=旧值,
                    快照值=快照值, 秩=秩新)


# ═════════════════════════════ 三臂 ═════════════════════════════
class _臂基:
    名 = "?"

    def __init__(self, 任务, 种子=0, 参数=None):
        self.任务 = 任务
        self.种子 = int(种子)
        self.参数 = dict(默认参数)
        self.参数.update({k: v for k, v in (参数 or {}).items() if k in 默认参数})
        self.状态 = 连续状态(全键列(任务), D=self.参数["D"], 种子=self.种子,
                            ρ=self.参数["ρ"], 状态幅=self.参数["状态幅"],
                            吸引子步数=self.参数["吸引子步数"],
                            回读阈值=self.参数["回读阈值"])

    def 收事件(self, 事件列):
        raise NotImplementedError

    def 步进(self, 步):
        self.状态.步进()

    def 自省轮(self, 轮, 探测键):
        raise NotImplementedError

    def 读(self, 键):
        return self.状态.读(键)

    def 指纹(self):
        return self.状态.指纹()


class A自指臂(_臂基):
    """A′：同一连续状态连续自省；事件直接写入；每轮自省把读回再写回；无独立审判。"""

    名 = "A′"

    def 收事件(self, 事件列):
        for e in 事件列:
            if not bool(e.get("任务相关", True)):
                权 = float(e.get("关注度", 0.25)) * float(self.参数["事件写权"])
            else:
                权 = float(e.get("关注度", 1.0)) * float(self.参数["事件写权"])
            self.状态.叠加写(e["键"], e["值"], 权)

    def 自省轮(self, 轮, 探测键):
        for k in 探测键:
            v = self.状态.读(k)
            if v is not None:
                self.状态.自省回写(k, v, 强度=float(self.参数["自省强度"]))


class B事件触发臂(_臂基):
    """B′：只读快照 → 独立审判器 → 受控写回；自省轮无新证据时不写回。"""

    名 = "B′"

    def __init__(self, 任务, 种子=0, 参数=None):
        super().__init__(任务, 种子=种子, 参数=参数)
        self.审判 = 独立审判器()

    def 收事件(self, 事件列):
        for e in 事件列:
            快照 = self.状态.快照()
            结果 = self.审判.判定(快照, self.状态, e)
            if 结果 is None:
                continue
            self.状态.受控写(结果["键"], 结果["新值"],
                            旧值=结果["旧值"], 权=float(self.参数["事件写权"]))
            self.审判.登记(结果)

    def 自省轮(self, 轮, 探测键):
        return                                    # 无新证据：不写回


class S惰性审判臂(B事件触发臂):
    """S′：B′ 架构 + 惰性/sham 审判（不看权限、按新近接受）；其余同 B′。"""

    名 = "S′"

    def __init__(self, 任务, 种子=0, 参数=None):
        super().__init__(任务, 种子=种子, 参数=参数)
        self.审判 = Sham审判器()


_臂表 = {"A′": A自指臂, "B′": B事件触发臂, "S′": S惰性审判臂}


def 建(臂, 任务, 种子=0, 参数=None):
    if 臂 not in _臂表:
        raise KeyError("未知臂：%r" % (臂,))
    return _臂表[臂](任务, 种子=种子, 参数=参数)


def 跑(臂, 任务, 种子=0, 参数=None) -> dict:
    """跑一个臂，返回可 JSON 化的原始记录（基线读 / 每轮自省后读 / 查询回答 / 指纹）。"""
    体 = 建(臂, 任务, 种子=种子, 参数=参数)
    探测键 = 探测键列(任务)
    T = int(任务["T"])
    R = int(任务.get("自省轮", 0))
    事, 查 = 按步(任务)
    基线 = []
    轮报 = []
    回答 = {}
    for t in range(T):
        体.收事件(事.get(t, ()))
        体.步进(t)
        基线.append({k: 体.读(k) for k in 探测键})
        rs = []
        for r in range(R):
            体.自省轮(r, 探测键)
            rs.append({k: 体.读(k) for k in 探测键})
        轮报.append(rs)
        for q in 查.get(t, ()):
            回答[str(q["编号"])] = 体.读(q["键"])
    return dict(臂=str(臂), 种子=int(种子), T=T, D=int(体.参数["D"]), 自省轮=R,
                探测键列=探测键, 基线=基线, 轮报=轮报, 查询回答=回答,
                状态指纹=体.指纹(), 参数=dict(体.参数))


if __name__ == "__main__":
    # 冒烟：只跑 dev seed 100（不是判据/留出集）
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _环境 = _载(os.path.join(HERE, "环境_长程合成任务.py"), "环境_仲裁冒烟")
    任务 = _环境.生成(100, "P1")
    表 = {}
    for 臂 in ("A′", "B′", "S′"):
        记 = 跑(臂, 任务, 种子=100)
        表[臂] = dict(指纹=记["状态指纹"],
                      基线样例={k: 记["基线"][8][k] for k in 记["探测键列"]},
                      轮报样例={k: 记["轮报"][8][2][k] for k in 记["探测键列"]},
                      查询数=len(记["查询回答"]))
    print(json.dumps(表, ensure_ascii=False, indent=1, default=str))
