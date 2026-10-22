# -*- coding: utf-8 -*-
"""meeting41/实施/判据_效率.py —— L4 性能对标判据（E1 线性时间 / E2 长程依赖 / E3 任意培训）。

契约依据：`接口契约.py` 第五节「L4 判据契约」（只读，不改）。
实现者D 的交付件；只依赖契约签名，**不重复实现能力核**。

━━━ 三条硬判据（sham 必须真的不过，否则判据无效）━━━
  E1 线性时间   : 计费(4096) / 计费(1024) < 4.5，且计费随 n 严格增长。
                  ★一律用**解析式计费**（`能力核.计费()["乘加数"]`），**不用墙钟秒做判据**；
                    墙钟只进 `墙钟诊断`（顶层字段，且不进任何特征码/判定）。
                  sham = 故意二次实现（比值 ≈16）⇒ 必须判不过。
                  恒真探针 = 常数计费（比值 =1.0，不随 n 增长）⇒ 必须判不过
                             （证明本判据不是「只看上界」的恒真式）。
  E2 长程依赖   : 长度 L 的序列（默认档 L=1024；长程档 L=4096）上做 index 检索，
                  命中率 ≥ 0.9。每次查询：词条 = 绑(键_i, 令牌_i) → 探针 = 解绑(键_i, 词条)
                  → 读(全序列码本, 探针) 必须查回**位置 i**（不是"查回了个东西"）。
                  sham = 只读**局部窗**（末 8 项）⇒ 窗外查询必失手，必须判不过。
                  恒真探针 = 读恒返回同一索引 ⇒ 必须判不过。
  E3 任意培训   : 未定义令牌（码本外输入）在两个读法下都成立——
                  A 同一未定义令牌 × 3 个不同上下文 ⇒ ≥2 个非零且两两不同；
                  B 3 个不同未定义令牌 × 同一上下文 ⇒ 两两不同且都有限；
                  且同上下文两遍**逐位同**（C13）。
                  sham = 忽略上下文（输出与上下文无关）⇒ 必须判不过。
                  恒真探针 = 全零输出 ⇒ 必须判不过。

━━━ 口径 / 纪律 ━━━
  · 测量台口径：`测量台.注册(名, 主臂, sham, 门槛)` + `.跑双集()`；双集 = 判据集 seed 0–5、
    留出集 seed 6–11；**阈值只在判据集口径下定死**（本文件顶部 `门槛`），不看留出集来调。
  · 随机性只来自 `numpy.random.default_rng(显式种子)`；判据**禁用 `time.time()`**（只用 `时基`）。
  · 臂签名 `臂(种子) -> dict(通过=bool, ...)`；臂内部兜底异常 ⇒ 记 `通过=False` + `异常`。
  · 臂结果**不含任何墙钟量**，故 `测量台` 的同 seed 两遍特征码必须逐位同。

━━━ 真件 / 桩臂（诚实边界）━━━
  · 真件 = `算子_能力核.能力核`（实现者A）。`取核()` 优先用真件。
  · 真件此刻不存在时，退化为本文件内的 `桩_能力核`（★**桩臂**，只为把判据自身跑通、
    证明判据可失败用；不是能力核实现）。**所有结果里都带 `臂来源` / `桩臂` 标注**，
    绝不用桩臂读数冒充真件读数。
  · 真件到位后复跑本文件即自动改用真件（无需改代码）。

诚实边界（本判据**不主张**的）：
  · E1 量的是**解析式乘加账**（机制是线性规模），**不是**实测墙钟速度；墙钟只作诊断。
  · E2 量的是「任意位置的 index 查回」（键→位置），**不是**「叠加式联想记忆容量」——
    纯叠加式（叠 L 个绑定向量的 crosstalk ≈ 1/√L）在 L=4096 时容量不足，故本判据不采用
    叠加式构造；这一点在结果里如实标注。
  · 三条判据都只证「机制成立且可失败」，不证「能力已等价于任何外部架构」。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AHI3 = os.path.dirname(os.path.dirname(HERE))          # → AHI3
for _p in (HERE, os.path.join(AHI3, "meeting40", "实施")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                   # noqa: E402

import 时基                                                          # noqa: E402
import 测量台                                                        # noqa: E402
from 接口契约 import 禁词                                             # noqa: E402

产物目录 = os.path.join(HERE, "产物")
真件路径 = os.path.join(HERE, "算子_能力核.py")
结果路径 = os.path.join(产物目录, "效率_结果.json")

# ── 阈值（判据集口径；改这里必须同步改证据文档，且不得看留出集来定）──
门槛 = dict(
    E1_比值上限=4.5,        # 契约：计费(4096)/计费(1024) < 4.5（线性≈4.0；二次≈16）
    E1_比值下限=1.5,        # 反恒真：计费必须随 n 增长（常数计费 ≈1.0 ⇒ 判不过）
    E2_命中率下限=0.9,      # 契约：index 检索命中率 ≥ 0.9
    E3_上下文数=3,
    E3_最少非零=2,
)
L_默认 = 1024               # E2 默认档
L_长程 = 4096               # E2 长程档（契约第五节的 4096）
查询数 = 10                 # 每种子查询 10 个位置（含首/尾），≥0.9 允许错 1 个
局部窗 = 8                  # sham：只读末 8 项
E3候选数 = 8                # 未定义令牌候选（±1 与连续各一半）


# ═══════════════════════════════════════════════════════════════════════
# 零、桩臂（★不是能力核实现；只为跑通判据自身、证明可失败）
# ═══════════════════════════════════════════════════════════════════════
class 桩_能力核:
    """★桩臂：契约签名的**最小替身**。真件 `算子_能力核.能力核` 到位后本类不再被取用。

    它只做「签名对得上、读数可复现」的最低限度工作：
      绑=逐元素乘积 ｜ 解绑=再乘一次键 ｜ 门=逐元素线性混合 ｜ 叠=求和后归一
      归=按 ‖·‖∞ 归一到 1 ｜ 读=码本最近邻（点积）｜ 迭代=半步混合的定点推进
      前向=归(绑(令牌, 归(上下文)))，并对**码本外**输入如实上报 未定义令牌=True
      计费=解析式 n·D·步数·3（线性）
    **这不是能力核的实现**，任何「能力」结论都不许引用桩臂读数。
    """

    D = 1024

    def __init__(self, D: int = 1024, 种子: int = 0) -> None:
        self.D = int(D)
        self.种子 = int(种子)
        自 = np.random.default_rng(int(种子) * 104729 + 11)
        self.内部码本 = (自.integers(0, 2, size=(64, self.D), dtype=np.int8) * 2 - 1)   # ±1，判定"定义域"
        self.W = {}
        for 名 in ("绑", "解绑", "门", "叠", "归", "读", "迭代"):
            self.登记算子(名, np.ones(self.D, dtype=np.float64))

    # ── 形状工具 ──
    def _列(self, 令牌列) -> np.ndarray:
        if isinstance(令牌列, np.ndarray):
            列 = 令牌列.astype(np.float64)
            return 列.reshape(1, -1) if 列.ndim == 1 else 列
        return np.stack([np.asarray(x, dtype=np.float64).ravel() for x in 令牌列])

    def _在码本(self, v: np.ndarray) -> bool:
        v = np.asarray(v, dtype=np.float64).ravel()
        if v.shape[0] != self.D or not np.all(np.abs(v) == 1.0):
            return False
        return bool(np.any(np.all(self.内部码本 == v.astype(np.int8), axis=1)))

    # ── 七原语 ──
    def 绑(self, 键, 值):
        return np.asarray(键, dtype=np.float64).ravel() * np.asarray(值, dtype=np.float64).ravel()

    def 解绑(self, 键, 捆绑):
        return np.asarray(键, dtype=np.float64).ravel() * np.asarray(捆绑, dtype=np.float64).ravel()

    def 门(self, 门控, x, y):
        g = np.asarray(门控, dtype=np.float64).ravel()
        return g * np.asarray(x, dtype=np.float64).ravel() + (1.0 - g) * np.asarray(y, dtype=np.float64).ravel()

    def 叠(self, 向量列):
        return self.归(np.sum(self._列(向量列), axis=0))

    def 归(self, 向量):
        v = np.asarray(向量, dtype=np.float64).ravel()
        峰 = float(np.max(np.abs(v))) if v.size else 0.0
        return v if 峰 <= 1.0 else v / 峰

    def 读(self, 码本, 探针, k: int = 1) -> dict:
        列 = self._列(码本)
        探 = np.asarray(探针, dtype=np.float64).ravel()
        模 = np.linalg.norm(列, axis=1) * max(1e-12, float(np.linalg.norm(探)))
        分 = (列 @ 探) / np.maximum(模, 1e-12)
        序 = np.argsort(-分, kind="stable")[:max(1, int(k))]
        return dict(k=int(k), 索引=[int(x) for x in 序], 相似=[float(分[x]) for x in 序],
                    向量=列[序], 命中=True)

    def 迭代(self, 态, 驱动力, 步数: int = 1) -> np.ndarray:
        态 = np.asarray(态, dtype=np.float64).ravel()
        驱 = np.asarray(驱动力, dtype=np.float64).ravel()
        for _ in range(max(1, int(步数))):
            态 = self.归(0.5 * 态 + 0.5 * 驱)
        return 态

    # ── 注册表 ──
    def 登记算子(self, 名: str, W) -> None:
        a = np.asarray(W, dtype=np.float64).ravel()
        self.W[str(名)] = a if a.size == self.D else np.full(self.D, float(a[0]) if a.size else 1.0)

    def 取算子(self, 名: str) -> dict:
        W = self.W[str(名)]
        return dict(名=str(名), W=W, 形状=tuple(W.shape), 可训练=True, 是注意力=False, 维数=self.D)

    def 算子表(self) -> list:
        return sorted(self.W)

    # ── 前向（上下文条件计算）──
    def 前向(self, 令牌列, 上下文=None, 迭代步数: int = 1) -> dict:
        列 = self._列(令牌列)
        未定义 = False
        出 = []
        ctx = None if 上下文 is None else self.归(np.asarray(上下文, dtype=np.float64).ravel())
        for i in range(列.shape[0]):
            v = 列[i]
            if not self._在码本(v):
                未定义 = True
            u = v if ctx is None else self.绑(v, ctx)
            出.append(self.迭代(self.归(u), 驱动力=u, 步数=int(迭代步数)))
        输 = np.stack(出)
        return dict(输出=输, 终态=出[-1], 迭代数=int(迭代步数), 未定义令牌=bool(未定义))

    # ── 参数基线 ──
    def 参数基线(self) -> dict:
        规范 = {k: np.round(v, 6).tolist() for k, v in sorted(self.W.items())}
        return dict(维数=self.D, 算子数=len(self.W),
                    参数总量=int(sum(v.size for v in self.W.values())),
                    W码=时基.特征码(规范))

    # ── 解析式计费（线性：n·D·步数·3）──
    def 计费(self, n: int, 迭代步数: int = 1) -> dict:
        乘加 = int(n) * self.D * max(1, int(迭代步数)) * 3
        return dict(乘加数=int(乘加), 时长估计=int(乘加 * 0.02))

    # ── 更新（只动 W）──
    def 更新(self, 名: str, ΔW) -> None:
        if str(名) not in self.W:
            return
        self.W[str(名)] = self.归(self.W[str(名)] + np.asarray(ΔW, dtype=np.float64).ravel() * 0.1)


def _构造(类, D, 种子):
    """按契约签名构造（kwargs → 位置 → 无参，逐级回退）。"""
    试 = [lambda: 类(D=D, 种子=int(种子)), lambda: 类(D, int(种子)), lambda: 类(int(D)),
          lambda: 类(D=D), lambda: 类()]
    错 = []
    for f in 试:
        try:
            return f(), None
        except Exception as ex:                                       # noqa: BLE001
            错.append("%s: %s" % (type(ex).__name__, ex))
    return None, " | ".join(错[:3])


def 取核(种子: int = 0, D: int = 1024, 强制桩: bool = False):
    """→ (核, 信息)。优先真件 `算子_能力核.能力核`；不可用则退化为 ★桩臂。"""
    信息 = dict(种子=int(种子), 维数=int(D), 真件路径=真件路径,
                真件存在=bool(os.path.exists(真件路径)), 桩臂=False, 来源="", 错误=None)
    if not 强制桩 and 信息["真件存在"]:
        try:
            import 算子_能力核 as m                                     # noqa: PLC0415
            if hasattr(m, "能力核"):
                核, 错 = _构造(m.能力核, D, 种子)
                if 核 is not None:
                    信息.update(来源="真件:算子_能力核.能力核", 桩臂=False)
                    return 核, 信息
                信息["错误"] = "真件构造失败: %s" % 错
            else:
                信息["错误"] = "真件模块里没有 `能力核` 类"
        except Exception as ex:                                       # noqa: BLE001
            信息["错误"] = "真件导入失败: %s: %s" % (type(ex).__name__, ex)
    信息.update(来源="桩臂:判据_效率.桩_能力核" + ("（真件不可用）" if 信息["错误"] else "（真件未到位）"),
                桩臂=True)
    return 桩_能力核(D=D, 种子=种子), 信息


# ═══════════════════════════════════════════════════════════════════════
# 一、确定性造数（只来自 default_rng；纯数据，可缓存）
# ═══════════════════════════════════════════════════════════════════════
_造数缓存: dict = {}


def 造令牌列(L: int, 种子: int, D: int = 1024, 流: int = 0) -> np.ndarray:
    """长度 L 的 ±1 令牌序列（键要用 流=1 另起一条流）。"""
    键 = (int(L), int(种子), int(D), int(流))
    if 键 not in _造数缓存:
        自 = np.random.default_rng(0x51 * 1000003 + int(种子) * 7919 + int(L) + int(流) * 104729)
        _造数缓存[键] = (自.integers(0, 2, size=(int(L), int(D)), dtype=np.int8) * 2 - 1).astype(np.float32)
    return _造数缓存[键]


def 查询位置列(L: int, K: int = 查询数) -> list:
    """等距查询位置（含首 0 与尾 L-1），覆盖 首/中/尾。"""
    K = max(2, int(K))
    return sorted({int(round(k * (L - 1) / float(K - 1))) for k in range(K)})


def 造上下文列(种子: int, 个数: int, D: int = 1024) -> list:
    """个数 个互不相同的 ±1 上下文（确定性）。"""
    自 = np.random.default_rng(0x77 * 1000003 + int(种子) * 33 + int(个数))
    ctxs = []
    while len(ctxs) < int(个数):
        v = (自.integers(0, 2, size=(int(D),), dtype=np.int8) * 2 - 1).astype(np.float64)
        if not any(np.array_equal(v, x) for x in ctxs):
            ctxs.append(v)
    return ctxs


def 造未定义候选(种子: int, 个数: int, D: int = 1024) -> list:
    """未定义令牌候选：前一半 ±1、后一半连续值（哪个能被核判为"码本外"由核自己上报）。"""
    自 = np.random.default_rng(0x99 * 1000003 + int(种子) * 613 + int(个数))
    out = []
    for i in range(int(个数)):
        if i % 2 == 0:
            out.append((自.integers(0, 2, size=(int(D),), dtype=np.int8) * 2 - 1).astype(np.float64))
        else:
            out.append(自.normal(0.0, 1.0, size=(int(D),)))
    return out


# ═══════════════════════════════════════════════════════════════════════
# 二、适配层：解析 `读()` / `前向()` 的返回（只解析，不重实现核）
# ═══════════════════════════════════════════════════════════════════════
_索引键 = ("索引", "index", "位置", "名次", "序", "idx", "indices", "id")
_向量键 = ("向量", "向量s", "最近", "值", "条目", "码", "vec", "vectors", "v", "输出")


def _取整数(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return None
        return _取整数(x[0])
    try:
        return int(x)
    except Exception:                                                 # noqa: BLE001
        return None


def 解析命中索引(果, 码本列, 探针) -> tuple:
    """从 `读()` 的返回里解析出命中条目号 → (索引|None, 依据)。**不重实现核**。

    `码本列` 只在需要向量匹配时才转 float64（避免大码本的无谓拷贝）。
    """
    def _宽():
        return int(np.asarray(码本列).shape[1])

    try:
        if isinstance(果, dict):
            for k in _索引键:
                if k in 果:
                    i = _取整数(果[k])
                    if i is not None and 0 <= i < len(码本列):
                        return i, "键:%s" % k
            for k in _向量键:                                  # 向量精确匹配 → 行号
                if k in 果:
                    a = np.asarray(果[k], dtype=np.float64)
                    if a.ndim == 1 and a.size == _宽():
                        a = a.reshape(1, -1)
                    if a.ndim == 2 and a.shape[1] == _宽():
                        A = np.asarray(码本列, dtype=np.float64)
                        同 = np.where(np.all(np.isclose(a[0], A), axis=1))[0]
                        if 同.size:
                            return int(同[0]), "键:%s(向量精确匹配)" % k
                        return None, "键:%s(向量在码本里找不到)" % k
            for k, v in 果.items():                            # 通用扫描
                if isinstance(v, str) or v is None:
                    continue
                a = np.asarray(v, dtype=np.float64)
                if a.size and a.shape[-1] == _宽():
                    if a.ndim == 1:
                        a = a.reshape(1, -1)
                    A = np.asarray(码本列, dtype=np.float64)
                    同 = np.where(np.all(np.isclose(a[0], A), axis=1))[0]
                    if 同.size:
                        return int(同[0]), "扫描:%s" % k
            return None, "返回里没有可解析的索引/向量（键=%s）" % sorted(map(str, 果.keys()))
        if isinstance(果, (list, tuple)) and 果:
            return 解析命中索引(果[0], 码本列, 探针)
        if hasattr(果, "shape") and np.asarray(果).size:
            return 解析命中索引(dict(向量=np.asarray(果)), 码本列, 探针)
    except Exception as ex:                                           # noqa: BLE001
        return None, "解析异常: %s: %s" % (type(ex).__name__, ex)
    return None, "返回类型不可解析: %s" % type(果).__name__


class 读适配:
    """`读(码本, 探针, k)` 的形态适配（ndarray → list → dict），记住先成功的那一种。"""

    def __init__(self, 核):
        self.核 = 核
        self.形态 = None
        self.错误: list = []

    def _候选码本(self, 令牌列: np.ndarray, 形态: str):
        if 形态 == "ndarray":
            return 令牌列
        if 形态 == "list":
            return [令牌列[i] for i in range(令牌列.shape[0])]
        return {"条目%d" % i: 令牌列[i] for i in range(令牌列.shape[0])}

    def 读(self, 令牌列: np.ndarray, 探针, k: int = 1):
        for 形态 in ([self.形态] if self.形态 else ["ndarray", "list", "dict"]):
            try:
                果 = self.核.读(self._候选码本(令牌列, 形态), 探针, k=k)
                self.形态 = 形态
                return 果
            except Exception as ex:                                   # noqa: BLE001
                记 = "%s: %s: %s" % (形态, type(ex).__name__, ex)
                if 记 not in self.错误:
                    self.错误.append(记)
        return None


class 前向适配:
    """`前向(令牌列, 上下文)` 的形态适配（上下文 (D,) → (1,D) → list），记住先成功的那一种。"""

    def __init__(self, 核):
        self.核 = 核
        self.形态 = None
        self.错误: list = []

    def _候选上下文(self, 上下文, 形态: str):
        if 上下文 is None:
            return None
        if 形态 == "一维":
            return 上下文
        if 形态 == "二维":
            return np.asarray(上下文, dtype=np.float64).reshape(1, -1)
        return [np.asarray(上下文, dtype=np.float64)]

    def 前向(self, 令牌列, 上下文=None, 迭代步数: int = 1):
        for 形态 in ([self.形态] if self.形态 else ["一维", "二维", "列表"]):
            try:
                果 = self.核.前向(令牌列, 上下文=self._候选上下文(上下文, 形态),
                                 迭代步数=int(迭代步数))
                self.形态 = 形态
                return 果
            except TypeError:
                try:
                    果 = self.核.前向(令牌列, self._候选上下文(上下文, 形态), int(迭代步数))
                    self.形态 = 形态 + "(位置参数)"
                    return 果
                except Exception as ex:                               # noqa: BLE001
                    记 = "%s: %s: %s" % (形态, type(ex).__name__, ex)
            except Exception as ex:                                   # noqa: BLE001
                记 = "%s: %s: %s" % (形态, type(ex).__name__, ex)
            if 记 not in self.错误:
                self.错误.append(记)
        return None


def _输出向量(果, 序: int = 0):
    """从 `前向()` 的返回里取第 序 行的输出向量。"""
    if not isinstance(果, dict):
        return None, "返回不是 dict: %s" % type(果).__name__
    出 = 果.get("输出", 果.get("终态"))
    if 出 is None:
        return None, "返回里没有 输出/终态（键=%s）" % sorted(map(str, 果.keys()))
    a = np.asarray(出, dtype=np.float64)
    if a.ndim == 1:
        return a, None
    if a.ndim == 2 and a.shape[0] > int(序):
        return a[int(序)], None
    return None, "输出形状不可用: %s" % (a.shape,)


# ═══════════════════════════════════════════════════════════════════════
# 三、臂内兜底（臂不许把异常抛出 测量台）
# ═══════════════════════════════════════════════════════════════════════
def _兜底(fn):
    def _包(种子, **kw):
        try:
            return fn(种子, **kw)
        except Exception as ex:                                       # noqa: BLE001
            return dict(通过=False, 异常="%s: %s" % (type(ex).__name__, ex), 种子=int(种子))
    _包.__name__ = fn.__name__
    return _包


# ═══════════════════════════════════════════════════════════════════════
# 四、E1 线性时间（解析式计费）
# ═══════════════════════════════════════════════════════════════════════
class _计费壳:
    """把内核算子的 `计费()` 换成指定解析式，其余转发（sham / 恒真探针用）。"""

    def __init__(self, 内, 计费函数):
        self._内 = 内
        self._计费 = 计费函数

    def __getattr__(self, 名):
        return getattr(self._内, 名)

    def 计费(self, n, 迭代步数: int = 1) -> dict:
        return self._计费(int(n), max(1, int(迭代步数)))


def 读数_线性(核, 基准: int = 1024, 步: int = 1) -> dict:
    """解析式计费读数（判据只用 乘加数；时长估计只作同式佐证）。"""
    def _一(n):
        d = dict(核.计费(int(n), 迭代步数=int(步)))
        return int(d.get("乘加数", 0)), int(d.get("时长估计", 0))

    c1, c2, c3, c4 = _一(基准), _一(基准 * 2), _一(基准 * 4), _一(基准 * 8)
    比值乘加 = c3[0] / float(max(1, c1[0]))
    比值时长 = c3[1] / float(max(1, c1[1]))
    单调 = bool(c1[0] < c2[0] < c3[0])
    通过 = bool(门槛["E1_比值下限"] <= 比值乘加 < 门槛["E1_比值上限"] and 单调
               and (c1[1] <= 0 or 比值时长 < 门槛["E1_比值上限"]))
    return dict(通过=通过,
                计费={str(基准): c1, str(基准 * 2): c2, str(基准 * 4): c3, str(基准 * 8): c4},
                乘加数_1024=c1[0], 乘加数_4096=c3[0],
                比值_乘加=round(比值乘加, 6), 比值_时长估计=round(比值时长, 6),
                比值_8192比2048=round(c4[0] / float(max(1, c2[0])), 6),
                严格增长=单调,
                门槛=dict(下限=门槛["E1_比值下限"], 上限=门槛["E1_比值上限"]),
                口径="契约：计费(4096)/计费(1024) < 4.5（线性≈4.0、二次≈16）；解析式，非墙钟")


def _E1臂(种子, 强制桩=False):
    核, 信息 = 取核(种子, 强制桩=强制桩)
    读 = 读数_线性(核)
    return dict(读, 种子=int(种子), 臂来源=信息["来源"], 桩臂=bool(信息["桩臂"]),
                真件存在=bool(信息["真件存在"]), 真件错误=信息["错误"])


@_兜底
def 判E1(种子):
    return _E1臂(种子)


@_兜底
def 判E1_sham二次(种子):
    """sham：故意二次实现（乘加数 = n²·3）⇒ 比值 ≈16 ⇒ 必须判不过。"""
    核, 信息 = 取核(种子)
    二次 = _计费壳(核, lambda n, s: dict(乘加数=int(n) * int(n) * s * 3,
                                     时长估计=int(n) * int(n) * s * 3))
    读 = 读数_线性(二次)
    return dict(读, 种子=int(种子), 臂来源="sham:二次计费壳(%s)" % 信息["来源"],
                桩臂=bool(信息["桩臂"]), 说明="故意二次实现（≈16）——判据必须判它不过")


@_兜底
def 判E1_常数臂(种子):
    """恒真探针：计费不随 n 增长（比值 =1.0）⇒ 必须判不过（证明判据不是只看上界）。"""
    核, 信息 = 取核(种子)
    固定 = int(1024) * 1024 * 3
    常数 = _计费壳(核, lambda n, s: dict(乘加数=固定, 时长估计=固定))
    读 = 读数_线性(常数)
    return dict(读, 种子=int(种子), 臂来源="恒真探针:常数计费壳(%s)" % 信息["来源"],
                桩臂=bool(信息["桩臂"]), 说明="常数计费（比值 1.0）——判据必须判它不过")


# ═══════════════════════════════════════════════════════════════════════
# 五、E2 长程依赖（长序列 index 查回）
# ═══════════════════════════════════════════════════════════════════════
def _E2一次(核, 种子, L, 窗=None, 读壳=None):
    """一次 E2 检索：词条=绑(键_i,令牌_i) → 探针=解绑(键_i,词条) → 读(码本,探针) 是否查回位置 i。"""
    D = int(getattr(核, "D", 1024))
    令牌列 = 造令牌列(L, 种子, D, 流=0)
    键列 = 造令牌列(L, 种子, D, 流=1)
    起点 = 0 if 窗 is None else max(0, int(L) - int(窗))
    码本 = 令牌列[起点:]                                     # sham：只给末窗
    适配 = 读适配(核) if 读壳 is None else 读壳
    命中 = 0
    逐 = []
    for i in 查询位置列(L, 查询数):
        词条 = 核.绑(键列[i], 令牌列[i])
        探针 = 核.解绑(键列[i], 词条)
        余弦 = float(np.dot(np.asarray(探针, dtype=np.float64).ravel(),
                           np.asarray(令牌列[i], dtype=np.float64).ravel())
                   / max(1e-12, float(np.linalg.norm(探针)) * float(np.linalg.norm(令牌列[i]))))
        果 = 适配.读(码本, 探针, k=1)
        if 果 is None:
            局, 依据 = None, "调用失败"
        else:
            局, 依据 = 解析命中索引(果, 码本, 探针)
        if 局 is None:
            全 = None
        else:
            全 = int(局) + int(起点)
        中 = bool(全 == int(i))
        命中 += 1 if 中 else 0
        逐.append(dict(位置=int(i), 命中=中, 返回条目=局, 全局条目=全,
                       探针余弦=round(余弦, 6), 解析依据=str(依据)[:120]))
    return dict(通过=bool(命中 / float(len(查询位置列(L, 查询数))) >= 门槛["E2_命中率下限"]),
                命中率=round(命中 / float(len(查询位置列(L, 查询数))), 6),
                命中数=int(命中), 查询数=len(查询位置列(L, 查询数)),
                序列长度=int(L), 码本条目数=int(码本.shape[0]),
                窗口=窗, 首查在窗内=(窗 is not None and 查询位置列(L, 查询数)[-1] >= 起点),
                逐查询=逐, 读形态=getattr(适配, "形态", None),
                读错误=getattr(适配, "错误", []),
                门槛=门槛["E2_命中率下限"],
                口径="词条=绑(键_i,令牌_i)；探针=解绑(键_i,词条)；读(码本,探针) 必须查回位置 i")


@_兜底
def 判E2(种子, L=L_默认):
    核, 信息 = 取核(种子)
    果 = _E2一次(核, 种子, L)
    return dict(果, 种子=int(种子), 臂来源=信息["来源"], 桩臂=bool(信息["桩臂"]),
                真件存在=bool(信息["真件存在"]))


@_兜底
def 判E2_sham局部窗(种子, L=L_默认):
    """sham：只读末 8 项（局部窗）⇒ 窗外查询必失手 ⇒ 必须判不过。"""
    核, 信息 = 取核(种子)
    果 = _E2一次(核, 种子, L, 窗=局部窗)
    在窗内 = [x for x in 果["逐查询"] if x["位置"] >= int(L) - 局部窗]
    窗外 = [x for x in 果["逐查询"] if x["位置"] < int(L) - 局部窗]
    return dict(果, 种子=int(种子), 臂来源="sham:局部窗(%d/%s)" % (局部窗, 信息["来源"]),
                桩臂=bool(信息["桩臂"]),
                窗内命中数=sum(1 for x in 在窗内 if x["命中"]),
                窗内查询数=len(在窗内), 窗外命中数=sum(1 for x in 窗外 if x["命中"]),
                窗外查询数=len(窗外),
                说明="sham 只看得见末 %d 项：窗内该中、窗外必失手（证明 sham 是「看不见远处」而非「坏掉」）"
                     % 局部窗)


@_兜底
def 判E2_恒定命中臂(种子, L=L_默认):
    """恒真探针：读恒返回同一条目 ⇒ 必须判不过（证明命中判据不是"读回了个东西就算中"）。"""
    核, 信息 = 取核(种子)

    class _恒读:
        def __init__(self):
            self.形态 = "恒定"
            self.错误 = []

        def 读(self, 码本, 探针, k: int = 1):
            return dict(索引=0, 相似=1.0, 命中=True)

    果 = _E2一次(核, 种子, L, 读壳=_恒读())
    return dict(果, 种子=int(种子), 臂来源="恒真探针:恒定索引壳(%s)" % 信息["来源"],
                桩臂=bool(信息["桩臂"]),
                说明="读恒返回条目 0 ⇒ 只有位置 0 那次可能中 ⇒ 必须判不过")


# ═══════════════════════════════════════════════════════════════════════
# 六、E3 任意培训（未定义令牌）
# ═══════════════════════════════════════════════════════════════════════
class _前向壳:
    """忽略令牌与上下文、恒定输出的 sham（其余转发）。"""

    def __init__(self, 内, 常数: np.ndarray):
        self._内 = 内
        self._常数 = np.asarray(常数, dtype=np.float64).ravel()

    def __getattr__(self, 名):
        return getattr(self._内, 名)

    def 前向(self, 令牌列, 上下文=None, 迭代步数: int = 1) -> dict:
        n = len(令牌列) if not isinstance(令牌列, np.ndarray) else (
            1 if 令牌列.ndim == 1 else 令牌列.shape[0])
        return dict(输出=np.tile(self._常数, (n, 1)), 终态=self._常数.copy(),
                    迭代数=int(迭代步数), 未定义令牌=True)


def _取输出(果):
    return _输出向量(果, 0)


def _E3臂(种子, 核, 来源, 桩臂):
    D = int(getattr(核, "D", 1024))
    适配 = 前向适配(核)
    ctxs = 造上下文列(种子, 门槛["E3_上下文数"], D)
    候选 = 造未定义候选(种子, E3候选数, D)

    未定义 = []
    探测 = []
    for idx, v in enumerate(候选):
        果 = 适配.前向([v], 上下文=ctxs[0], 迭代步数=1)
        if 果 is None:
            探测.append(dict(候选=idx, 上报=False, 异常="前向调用失败"))
            continue
        f = bool(果.get("未定义令牌")) if isinstance(果, dict) else False
        探测.append(dict(候选=idx, 上报=f, 形态=适配.形态))
        if f:
            未定义.append(v)

    甲 = dict(说明="A 同一未定义令牌 × 3 个上下文", 上下文数=len(ctxs), 输出列=[], 非零数=0,
              两两不同对数=0, 对数总数=len(ctxs) * (len(ctxs) - 1) // 2, 逐位同=True)
    乙 = dict(说明="B 3 个不同未定义令牌 × 同一上下文", 令牌数=0, 输出列=[], 非零数=0,
              两两不同对数=0, 对数总数=0, 都有限=True)
    组 = None
    if 未定义:
        首 = 未定义[0]
        出列 = []
        for k in range(len(ctxs)):
            果1 = 适配.前向([首], 上下文=ctxs[k], 迭代步数=1)
            果2 = 适配.前向([首], 上下文=ctxs[k], 迭代步数=1)
            v1, 错 = _取输出(果1)
            v2, _ = _取输出(果2)
            if v1 is None or v2 is None:
                出列.append(None)
                continue
            逐位同 = bool(np.array_equal(v1, v2))
            出列.append(dict(上下文=k, 有限=bool(np.all(np.isfinite(v1))),
                             非零=bool(float(np.linalg.norm(v1)) > 0.0),
                             逐位同=逐位同, 范数=round(float(np.linalg.norm(v1)), 6),
                             首4=[round(float(x), 6) for x in v1[:4]]))
            甲["逐位同"] = 甲["逐位同"] and 逐位同
        向量列 = [np.asarray(x["首4"], dtype=np.float64) for x in 出列 if x is not None]
        完整列 = []
        for k in range(len(ctxs)):
            果 = 适配.前向([首], 上下文=ctxs[k], 迭代步数=1)
            v, _ = _取输出(果)
            完整列.append(v)
        甲["输出列"] = 出列
        甲["非零数"] = sum(1 for x in 出列 if x and x["非零"])
        对 = [(a, b) for a in range(len(ctxs)) for b in range(a + 1, len(ctxs))]
        甲["两两不同对数"] = sum(
            1 for a, b in 对
            if 完整列[a] is not None and 完整列[b] is not None
            and not np.array_equal(完整列[a], 完整列[b]))
        甲["通过"] = bool(甲["非零数"] >= 门槛["E3_最少非零"]
                        and 甲["两两不同对数"] == 甲["对数总数"] and 甲["逐位同"])
        # B：3 个不同未定义令牌 × 同一上下文
        用 = 未定义[:3]
        乙["令牌数"] = len(用)
        乙列 = []
        for v in 用:
            果 = 适配.前向([v], 上下文=ctxs[0], 迭代步数=1)
            x, _ = _取输出(果)
            乙列.append(x)
            乙["输出列"].append(None if x is None else dict(
                有限=bool(np.all(np.isfinite(x))), 非零=bool(float(np.linalg.norm(x)) > 0.0),
                范数=round(float(np.linalg.norm(x)), 6),
                首4=[round(float(t), 6) for t in x[:4]]))
        对乙 = [(a, b) for a in range(len(用)) for b in range(a + 1, len(用))]
        乙["对数总数"] = len(对乙)
        乙["两两不同对数"] = sum(1 for a, b in 对乙
                            if 乙列[a] is not None and 乙列[b] is not None
                            and not np.array_equal(乙列[a], 乙列[b]))
        乙["都有限"] = bool(all(x is not None and np.all(np.isfinite(x)) for x in 乙列))
        乙["非零数"] = sum(1 for x in 乙列 if x is not None and float(np.linalg.norm(x)) > 0.0)
        乙["通过"] = bool(乙["令牌数"] == 3 and 乙["两两不同对数"] == 3 and 乙["都有限"])
        组 = bool(甲["通过"] and 乙["通过"])
    通过 = bool(组)
    return dict(通过=通过, 甲_同一令牌三上下文=甲, 乙_三令牌同上下文=乙,
                未定义数=len(未定义), 候选探测=探测,
                上下文形态=适配.形态, 前向错误=适配.错误,
                种子=int(种子), 臂来源=来源, 桩臂=bool(桩臂),
                门槛=dict(上下文数=门槛["E3_上下文数"], 最少非零=门槛["E3_最少非零"]),
                口径="契约：未定义令牌 3 个不同上下文 ⇒ ≥2 非零且两两不同；同上下文两遍逐位同；"
                     "码本外 3 个向量 ⇒ 输出两两不同且都有限",
                诚实边界=("若未定义数=0 ⇒ 核未上报「码本外输入」，本判据如实判不过（不伪造通过）"
                        if not 未定义 else ""))


@_兜底
def 判E3(种子):
    核, 信息 = 取核(种子)
    return _E3臂(种子, 核, 信息["来源"], 信息["桩臂"])


@_兜底
def 判E3_sham忽略上下文(种子):
    """sham：输出与上下文无关（恒定）⇒ 三上下文输出全同 ⇒ 必须判不过。"""
    核, 信息 = 取核(种子)
    D = int(getattr(核, "D", 1024))
    常 = np.linspace(0.25, 0.75, D)
    壳 = _前向壳(核, 常)
    return dict(_E3臂(种子, 壳, "sham:忽略上下文壳(%s)" % 信息["来源"], 信息["桩臂"]),
                说明="输出与上下文无关 ⇒ 两两不同=0 ⇒ 必须判不过")


@_兜底
def 判E3_sham全零(种子):
    """恒真探针：输出恒为全零 ⇒ 非零数=0 ⇒ 必须判不过。"""
    核, 信息 = 取核(种子)
    D = int(getattr(核, "D", 1024))
    壳 = _前向壳(核, np.zeros(D))
    return dict(_E3臂(种子, 壳, "恒真探针:全零输出壳(%s)" % 信息["来源"], 信息["桩臂"]),
                说明="输出恒全零 ⇒ 非零数=0 ⇒ 必须判不过")


# ═══════════════════════════════════════════════════════════════════════
# 七、C1 自检（禁词命中；只读文本，不改）
# ═══════════════════════════════════════════════════════════════════════
def C1自检(路径列=None) -> dict:
    路径列 = 路径列 or [os.path.join(HERE, "判据_效率.py"), os.path.join(HERE, "真机探头.py")]
    逐 = {}
    for p in 路径列:
        if not os.path.exists(p):
            逐[os.path.basename(p)] = dict(存在=False, 命中=[])
            continue
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            文 = f.read().lower()
        逐[os.path.basename(p)] = dict(存在=True,
                                    命中=sorted({w for w in 禁词 if w.lower() in 文}))
    return dict(通过=all(not v["命中"] for v in 逐.values()), 逐文件=逐,
                口径="子串级、大小写不敏感的禁词扫描（C1 审计同口径）")


# ═══════════════════════════════════════════════════════════════════════
# 八、主：注册 → 双集 → 读数摘要 → 落盘
# ═══════════════════════════════════════════════════════════════════════
def _必须不过(主, 零, 种子):
    """恒真探针条目的门槛：**该臂本该不过**，它不过才算这条探针成立。"""
    return (not bool(主.get("通过")))


def 建台() -> 测量台.测量台:
    台 = 测量台.测量台("L4_效率")
    台.注册("E1_线性时间", 判E1, 判E1_sham二次, None,
            "计费(4096)/计费(1024)<4.5（解析式）；sham=二次实现≈16")
    台.注册("E1_恒真探针_常数计费", 判E1_常数臂, None, _必须不过,
            "常数计费（比值1.0）本该判不过 ⇒ 证明判据不是只看上界的恒真式")
    台.注册("E2_长程依赖_L1024", lambda s: 判E2(s, L=L_默认),
            lambda s: 判E2_sham局部窗(s, L=L_默认), None,
            "L=1024 序列 index 检索命中率≥0.9；sham=只读末%d项" % 局部窗)
    台.注册("E2_长程依赖_L4096", lambda s: 判E2(s, L=L_长程),
            lambda s: 判E2_sham局部窗(s, L=L_长程), None,
            "L=4096 长程档（契约第五节）；sham=只读末%d项" % 局部窗)
    台.注册("E2_恒真探针_恒定命中", lambda s: 判E2_恒定命中臂(s, L=L_默认), None, _必须不过,
            "读恒返回同一条目本该判不过 ⇒ 证明命中=「查回正确位置」")
    台.注册("E3_任意培训", 判E3, 判E3_sham忽略上下文, None,
            "未定义令牌：3 上下文⇒≥2 非零且两两不同 + 3 令牌⇒两两不同 + 同上下文逐位同")
    台.注册("E3_恒真探针_全零", 判E3_sham全零, None, _必须不过,
            "全零输出本该判不过 ⇒ 证明判据不是恒真")
    return 台


def _臂字段(报, 名, 字段):
    return [v["主"][字段] for v in 报["集"]["判据集"][名]["逐种子"]]


def 读数摘要(报) -> dict:
    判 = 报["集"]["判据集"]
    留 = 报["集"]["留出集"]
    E1 = 判["E1_线性时间"]["逐种子"]
    E2a = 判["E2_长程依赖_L1024"]["逐种子"]
    E2b = 判["E2_长程依赖_L4096"]["逐种子"]
    E3 = 判["E3_任意培训"]["逐种子"]
    return dict(
        E1=dict(比值乘加=_臂字段(报, "E1_线性时间", "比值_乘加"),
                比值乘加_区间=[min(_臂字段(报, "E1_线性时间", "比值_乘加")),
                            max(_臂字段(报, "E1_线性时间", "比值_乘加"))],
                sham二次比值=[v["零"]["比值_乘加"] for v in E1],
                sham二次全不过=all(not v["零"]["通过"] for v in E1),
                恒真探针常数比值=[v["主"]["比值_乘加"]
                            for v in 判["E1_恒真探针_常数计费"]["逐种子"]],
                留出集比值=[v["主"]["比值_乘加"] for v in 留["E1_线性时间"]["逐种子"]]),
        E2=dict(L1024_命中率=[v["主"]["命中率"] for v in E2a],
                L1024_sham命中率=[v["零"]["命中率"] for v in E2a],
                L1024_sham全不过=all(not v["零"]["通过"] for v in E2a),
                L4096_命中率=[v["主"]["命中率"] for v in E2b],
                L4096_sham命中率=[v["零"]["命中率"] for v in E2b],
                L4096_sham全不过=all(not v["零"]["通过"] for v in E2b),
                留出集_L4096命中率=[v["主"]["命中率"] for v in 留["E2_长程依赖_L4096"]["逐种子"]],
                恒真探针命中率=[v["主"]["命中率"]
                           for v in 判["E2_恒真探针_恒定命中"]["逐种子"]]),
        E3=dict(A_非零数=[v["主"]["甲_同一令牌三上下文"]["非零数"] for v in E3],
                A_两两不同对数=[v["主"]["甲_同一令牌三上下文"]["两两不同对数"] for v in E3],
                A_对数总数=[v["主"]["甲_同一令牌三上下文"]["对数总数"] for v in E3],
                A_逐位同=[v["主"]["甲_同一令牌三上下文"]["逐位同"] for v in E3],
                B_两两不同对数=[v["主"]["乙_三令牌同上下文"]["两两不同对数"] for v in E3],
                B_都有限=[v["主"]["乙_三令牌同上下文"]["都有限"] for v in E3],
                未定义数=[v["主"]["未定义数"] for v in E3],
                sham两两不同对数=[v["零"]["甲_同一令牌三上下文"]["两两不同对数"] for v in E3],
                sham全不过=all(not v["零"]["通过"] for v in E3),
                留出集A两两不同对数=[v["主"]["甲_同一令牌三上下文"]["两两不同对数"]
                              for v in 留["E3_任意培训"]["逐种子"]]),
    )


def 主(存: bool = True, 长程: bool = True) -> dict:
    台 = 测量台.测量台("L4_效率") if not 长程 else 建台()
    if not 长程:                                            # 短档：去掉 L4096 一条（诊断用）
        台 = 测量台.测量台("L4_效率")
        台.注册("E1_线性时间", 判E1, 判E1_sham二次, None, "计费比值<4.5；sham=二次")
        台.注册("E2_长程依赖_L1024", lambda s: 判E2(s, L=L_默认),
                lambda s: 判E2_sham局部窗(s, L=L_默认), None, "L=1024 index 命中率≥0.9")
        台.注册("E3_任意培训", 判E3, 判E3_sham忽略上下文, None, "未定义令牌三读法")
    报净 = 台.跑双集()
    摘要 = 读数摘要(报净) if 长程 else {}
    C1 = C1自检()
    核信息 = 取核(0)[1]
    报净 = dict(报净, 读数=摘要, C1自检=C1, 核信息=核信息,
                门槛=门槛, 口径=dict(判据集="seed 0–5", 留出集="seed 6–11（只判不调）",
                                  E1="解析式计费（非墙钟）", E2="index 查回位置",
                                  E3="未定义令牌×上下文", 局部窗=局部窗, 查询数=查询数),
                诚实边界=[
                    "E1 量的是解析式乘加账（线性规模），不是实测墙钟速度；墙钟只进 `墙钟诊断`。",
                    "E2 量的是「任意位置的 index 查回」，不是叠加式联想记忆容量；"
                    "纯叠加式在 L=4096 时 crosstalk≈1/√L，故本判据不采用叠加式构造。",
                    "桩臂读数只用于证明判据可失败；能力结论一律以 `臂来源=真件:*` 的读数为准。",
                    ("真件 `算子_能力核.py` 未到位 ⇒ 本轮读数为 **桩臂**，真件到位后必须复跑替换。"
                     if 核信息["桩臂"] else "真件在读。"),
                ])
    存报 = 报净
    if 存:
        os.makedirs(产物目录, exist_ok=True)
        报存 = 台.存(结果路径, 报净)                       # 含 本台特征码（= 净报的 sha256）
        钟 = 时基.拍钟()
        钟.打点("τ_q")
        核, _ = 取核(0)
        核.计费(1024)
        核.计费(4096)
        钟.打点("τ_arb")
        存报 = dict(报存, 墙钟诊断=dict(
            用途="仅诊断（不参与任何判定、不进任何特征码）",
            计费两次调用毫秒=钟.延迟("τ_q", "τ_arb")["毫秒"],
            说明="判据本体一律用解析式计费；此处墙钟只为让审计看到真实调用成本"))
        with open(结果路径, "w", encoding="utf-8") as f:
            json.dump(存报, f, ensure_ascii=False, indent=1, default=str)
    return 存报


def _打印摘要(报) -> None:
    读 = 报.get("读数") or {}
    E1, E2, E3 = 读.get("E1", {}), 读.get("E2", {}), 读.get("E3", {})
    print("─" * 78)
    print("E1 线性时间（解析式计费）")
    print("  主臂 计费(4096)/计费(1024) 判据集 = %s" % (E1.get("比值乘加"),))
    print("  sham 二次实现 比值               = %s  ⇒ sham 全不过 = %s"
          % (E1.get("sham二次比值"), E1.get("sham二次全不过")))
    print("  恒真探针 常数计费 比值           = %s" % (E1.get("恒真探针常数比值"),))
    print("  留出集 比值                      = %s" % (E1.get("留出集比值"),))
    print("E2 长程依赖（index 查回）")
    print("  L=1024 主臂命中率 = %s ｜ sham(末%d项) = %s ｜ sham 全不过 = %s"
          % (E2.get("L1024_命中率"), 局部窗, E2.get("L1024_sham命中率"), E2.get("L1024_sham全不过")))
    print("  L=4096 主臂命中率 = %s ｜ sham(末%d项) = %s ｜ sham 全不过 = %s"
          % (E2.get("L4096_命中率"), 局部窗, E2.get("L4096_sham命中率"), E2.get("L4096_sham全不过")))
    print("  恒真探针 恒定命中率 = %s ｜ 留出集 L4096 命中率 = %s"
          % (E2.get("恒真探针命中率"), E2.get("留出集_L4096命中率")))
    print("E3 任意培训（未定义令牌）")
    print("  A 同一令牌×3上下文: 非零数=%s 两两不同=%s/%s 两遍逐位同=%s"
          % (E3.get("A_非零数"), E3.get("A_两两不同对数"), E3.get("A_对数总数"), E3.get("A_逐位同")))
    print("  B 3令牌×同上下文  : 两两不同=%s/3 都有限=%s ｜ 未定义数=%s"
          % (E3.get("B_两两不同对数"), E3.get("B_都有限"), E3.get("未定义数")))
    print("  sham 忽略上下文 两两不同=%s ⇒ sham 全不过=%s"
          % (E3.get("sham两两不同对数"), E3.get("sham全不过")))
    print("─" * 78)
    print("判据集/留出集 逐条：")
    for 集名, 集 in 报["集"].items():
        for 名, v in sorted(集.items()):
            print("  [%s] %-22s 全过=%s 门槛全过=%s 确定性全过=%s"
                  % (集名, 名, v["全过"], v["门槛全过"], v["确定性全过"]))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                                 # noqa: BLE001
        pass
    报 = 主(存=True)
    _打印摘要(报)
    核信息 = 报["核信息"]
    print("核来源：%s ｜ 桩臂=%s ｜ 真件存在=%s ｜ 真件错误=%s"
          % (核信息["来源"], 核信息["桩臂"], 核信息["真件存在"], 核信息["错误"]))
    print("C1 自检（禁词命中=0）：%s" % (报["C1自检"]["通过"],))
    print("总过=%s ｜ 判据数=%s ｜ 结果=%s" % (报["总过"], 报["判据数"], 结果路径))
    if 核信息["桩臂"]:
        print("★诚实标注：本轮为**桩臂**读数（真件 `算子_能力核.py` 未到位）；真件到位后必须复跑。")
    print("[判据_效率] %s" % ("PASS" if 报["总过"] else "FAIL"))
    raise SystemExit(0 if 报["总过"] else 1)
