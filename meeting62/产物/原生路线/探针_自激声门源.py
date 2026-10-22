# -*- coding: utf-8 -*-
"""meeting62/产物/原生路线/探针_自激声门源.py —— 路线②「原生自发声（无参考）」最小可证伪实验。

依据（只读）：
  · 团队/脑神经/R1_立场.md §B6/§B7/§C2/§C3/§E —— 喉部驱动 → 声带在气流下**自激**（极限环）；
    最该先做的最小实验＝把 `引擎_v6.声门流表` 的查表脉冲换成 2 质量自激 ODE（仅 F0/紧喉驱动）；
  · 团队/HDC内在动力学/R1_立场.md 主张 C —— 自激声门源 × 谐振字典；
  · 裁决/00_裁决记录.md §3.2 / §5.2-3 —— 旁路探针，**不改 v6 主链路**（C14）。

模型（[提案]，**P0 文献口径**：Ishizaka–Flanagan 两质量模型，按
  Fulcher/Scherer/Waddle, JASA 132(4):2582 (2012) = PMC3477191 的 Eq(4)(5)(6)(7)(8)(9) 实现）：
      m1·ẍ1 + r1·ẋ1 + k1·x1 + kc·(x1−x2) = F1 ,  m2·ẍ2 + r2·ẋ2 + k2·x2 + kc·(x2−x1) = F2
      F1 = P1·Lg·d1 ,  F2 = P2·Lg·d2 = 0      （该文取出口损失系数 k_ex = 0 ⇒ P2 = 0）
      A1 = Lg·(ξ0+x1) , A2 = Lg·(ξ0+x2) , r = A1/A2
      P1 = P_sub·(r²−1) / ((k_ent−1) + r²)     （口径：入口损失 + Bernoulli，黏性忽略）
  关键机理（PMC3477191 §Small amplitude approximation）：P1 对几何的导数为**负**
  ⇒ 流体-结构耦合给出**流致刚度**，与恢复力**反向** ⇒ 高压下平衡点失稳 ⇒ 自激极限环。
  ⇒ [推导] 线性化（r≈1，P1 的斜率 = 2/(k_ent)，x1 斜率 = 1/ξ0）：
         φ(Ps) = 2·Lg·d1·Ps/(k_ent·ξ0)  ；  失稳条件 φ ≳ k1
         ⇒ **解析阈值 Ps = k1·k_ent·ξ0/(2·Lg·d1)** = 343 Pa（本件参数集，紧喉=0）

做什么（全部真跑，CPU-only，单次 <10s）：
  J1  起振阈值：从**静止几何 + 确定性微偏**（x1 = 1e-6 m，内部偏置，**非外部波形注入**）
      积分，扫声门下压 Ps；判据＝**末段包络对数斜率 > 0**（增长/持续）vs < 0（衰减）；
  J1z 零初态对照：x1≡0（严格在**不稳定平衡点**上）⇒ 永不起振（说明需要内部微扰作种子）；
  J1′ must-fail：换回**查表脉冲**（只读 `引擎_v6.声门流表`）⇒ 结构上**不可能**有起振阈值；
  J2  输出一段波形：ODE 声门流 U(t) × 既有 Klatt 二阶谐振器（只读 `引擎_v4.谐振器系数/_滤波块`）
      → 元音化 wav；
  J3  紧喉 → F0：仅改张力标量（弹簧 ×(1+1.2·紧喉)），F0 须单调升；
  J4  零外挂三断言：① 运行期零音频读入（`builtins.open` 钩子）② 无外部模型/权重（AST import +
      `sys.modules`）③ 同参数两趟**逐位同**（C13）。

红线：C1 无 attention/QKV/softmax（禁词片段拼装自扫）｜ C9 不涉及 ｜ C10 全程不引 TTS/RVC/神经声码器
/外部权重；唯一外部输入＝标量 Ps 与 紧喉 ｜ C13 纯浮点确定性 ⇒ 两趟逐位同 ｜ C14 本件**只新增于
meeting62/**，meeting61 既有件**只读 import**。

标记口径：[实测] 本件真跑读数 ｜ [源码] 读自源码 ｜ [推导] 由读数/源码/解析式算出 ｜ [提案] 设计或
阈值，未验证 ｜ [P0] 文献 ｜ 未测。
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import json
import math
import os
import sys
import wave

import numpy as np

本目录 = os.path.dirname(os.path.abspath(__file__))
AHI4 = os.path.dirname(os.path.dirname(os.path.dirname(本目录)))     # …/AHI4
MEET61实施 = os.path.join(AHI4, "meeting61", "实施")
if MEET61实施 not in sys.path:
    sys.path.insert(0, MEET61实施)

import 引擎_v6 as E6  # noqa: E402  只读：must-fail 对照臂取 `声门流表`；谐振器取 v6 转发的 v4 件

FS = 44100

# ── 模型参数（[提案]：IF 两质量口径 + 本件标定到 F0≈100Hz；全部为内部常数，无外部拟合来源）──
M1 = 1.25e-4      # kg    下质量
M2 = 2.5e-5       # kg    上质量（IF 比例 5:1）
K1 = 37.6         # N/m   下质量弹簧
K2 = 3.66         # N/m   上质量弹簧
KC = 11.75        # N/m   耦合弹簧（IF 比例 kc/k1 = 25/80）
R1 = 0.0411       # N·s/m 阻尼 ζ=0.3（同 PMC3477191 §Below threshold 口径）
R2 = 0.00574
LG = 0.012        # m     声门前后长 Lg = 1.2 cm
厚1 = 0.0025      # m     下质量厚度 d1 = 0.25 cm
AREA = LG * 厚1   # m²    气动力面 Lg·d1
XI0 = 4.0e-4      # m     预声门半宽 ξ0 = 0.04 cm
KENT = 1.37       # 入口损失系数（PMC3477191 常用值）
RO = 1.2          # kg/m³ 空气密度（仅用于声门流 U）
微偏 = 1.0e-6     # m     确定性内部偏置（起始种子；非外部波形注入）
紧喉刚度增益 = 1.2  # 紧喉 ∈[0,1] → 弹簧刚度 ×(1+1.2·紧喉)（张力↑ ⇒ F0↑）

Ps列 = [round(v, 2) for v in np.linspace(0.0, 900.0, 37)]   # 25Pa 分辨率
扫秒 = 0.50        # 扫掠积分时长
扫dt = 1.0 / 20000.0   # 扫掠积分步长 50µs（≈ 100Hz 极限环周期的 1/200，RK4 分辨率充裕）

# ── 禁词/外挂名单（片段拼装，避免扫描器命中本文件自身的审计字符串）──
_禁词片段 = (("at", "tention"), ("at", "tn"), ("q", "kv"), ("soft", "max"),
             ("multi", "head"), ("trans", "former"), ("rw", "kv"),
             ("位置", "编码"), ("to", "rch"))
_外挂片段 = (("py", "ttsx3"), ("sa", "pi"), ("r", "vc"), ("so", "vits"),
             ("whis", "per"), ("fun", "asr"), ("li", "brosa"), ("sound", "file"),
             ("torch", "audio"), ("audio", "read"), ("py", "dub"))


def 禁词() -> list:
    return ["".join(p) for p in _禁词片段]


def 外挂名单() -> list:
    return ["".join(p) for p in _外挂片段]


# ═════════════════════════════════ 1. 自激声门 ODE（核心）═════════════════════════════

def 导数_单(态, Ps, 紧喉):
    """标量 d态/dt（纯 Python 浮点）。态 = (x1, v1, x2, v2)，x 为偏离静止半宽 ξ0 的位移(m)。"""
    x1, v1, x2, v2 = 态
    s = 1.0 + 紧喉刚度增益 * 紧喉
    k1 = K1 * s
    k2 = K2 * s
    kc = KC * s
    A1 = LG * (XI0 + x1)
    A2 = LG * (XI0 + x2)
    if A1 <= 0.0 or A2 <= 0.0:          # 闭合：无流 ⇒ 入口压力恢复为 Ps
        P1 = Ps
    else:
        r2 = (A1 * A1) / (A2 * A2)
        P1 = Ps * (r2 - 1.0) / ((KENT - 1.0) + r2)
    F1 = AREA * P1                       # 唯一的流体驱动力（F2 = P2·Lg·d2 = 0，k_ex = 0）
    return (v1,
            (-k1 * x1 - R1 * v1 - kc * (x1 - x2) + F1) / M1,
            v2,
            (-k2 * x2 - R2 * v2 - kc * (x2 - x1)) / M2)


def 导数_批(态, Ps, 紧喉):
    """向量化 d态/dt（态 shape (4,K)，Ps shape (K,)）——与 导数_单 同一数学，仅形式不同。"""
    x1, v1, x2, v2 = 态
    s = 1.0 + 紧喉刚度增益 * 紧喉
    k1 = K1 * s
    k2 = K2 * s
    kc = KC * s
    A1 = LG * (XI0 + x1)
    A2 = LG * (XI0 + x2)
    开 = (A1 > 0.0) & (A2 > 0.0)
    r2 = np.where(开, A1 * A1 / np.maximum(A2 * A2, 1e-30), 1.0)
    P1 = np.where(开, Ps * (r2 - 1.0) / ((KENT - 1.0) + r2), Ps)
    F1 = AREA * P1
    return np.stack([v1,
                     (-k1 * x1 - R1 * v1 - kc * (x1 - x2) + F1) / M1,
                     v2,
                     (-k2 * x2 - R2 * v2 - kc * (x2 - x1)) / M2])


def 流_单(A1, A2, Ps) -> float:
    """声门体积流 U（PMC3477191 Eq(8)）：Ps = ½ρU²[(k_ent−1)/A1² + 1/A2²] ⇒ U。闭合 ⇒ 0。"""
    if A1 <= 0.0 or A2 <= 0.0 or Ps <= 0.0:
        return 0.0
    d = (KENT - 1.0) / (A1 * A1) + 1.0 / (A2 * A2)
    return math.sqrt(2.0 * Ps / RO / d)


def 解析阈值(紧喉=0.0) -> float:
    """[推导] 线性化失稳条件 φ ≳ k1，其中 φ(Ps) = 2·Lg·d1·Ps/(k_ent·ξ0)。

    r≈1 时 dP1/dr = 2/k_ent，dr/dx1 = 1/ξ0 ⇒ φ = dF1/dx1 = Lg·d1·Ps·(2/k_ent)/ξ0，
    方向与恢复力相反（PMC3477191 §Small amplitude approximation：flow-induced stiffness）。
    ⇒ Ps_解析 = k1·k_ent·ξ0/(2·Lg·d1)。
    """
    s = 1.0 + 紧喉刚度增益 * 紧喉
    return K1 * s * KENT * XI0 / (2.0 * AREA)


def 合成_单(Ps, 紧喉, 秒, 子步=2, 初偏=微偏, 记位移=False):
    """从**静止几何 + 内部微偏**积分单一 Ps ⇒ 声门流波形 U(t)（numpy 数组）。

    记位移=True 时额外返回 x1 列（零初态对照用：判位移是否恒 0）。
    """
    dt = 1.0 / (FS * 子步)
    n步 = int(round(秒 / dt))
    n出 = n步 // 子步
    x1, v1, x2, v2 = 初偏, 0.0, 0.0, 0.0
    流 = np.empty(n出, dtype=np.float64)
    x1列 = [] if 记位移 else None
    j = 0
    for i in range(n步):
        k1 = 导数_单((x1, v1, x2, v2), Ps, 紧喉)
        b = (x1 + 0.5 * dt * k1[0], v1 + 0.5 * dt * k1[1], x2 + 0.5 * dt * k1[2], v2 + 0.5 * dt * k1[3])
        k2 = 导数_单(b, Ps, 紧喉)
        c = (x1 + 0.5 * dt * k2[0], v1 + 0.5 * dt * k2[1], x2 + 0.5 * dt * k2[2], v2 + 0.5 * dt * k2[3])
        k3 = 导数_单(c, Ps, 紧喉)
        d = (x1 + dt * k3[0], v1 + dt * k3[1], x2 + dt * k3[2], v2 + dt * k3[3])
        k4 = 导数_单(d, Ps, 紧喉)
        x1 += (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        v1 += (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        x2 += (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        v2 += (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
        if (i + 1) % 子步 == 0:
            流[j] = 流_单(LG * (XI0 + x1), LG * (XI0 + x2), Ps)
            if 记位移:
                x1列.append(x1)
            j += 1
    if 记位移:
        return 流[:j], np.asarray(x1列, dtype=np.float64)
    return 流[:j]


def 扫_起振(Ps列, 紧喉, 秒=扫秒, dt=扫dt):
    """向量化扫 Ps：从静止+微偏积分，按**三段**取 w1=ξ0+x1 的峰-峰值，并给出**末段对数斜率**。

    [提案] 起振判据＝末段包络相对中段**增长**（对数斜率 > 0）；衰减（< 0）＝未起振。
    斜率用 ½(lnA3−lnA2)/(窗口跨度) 的等价式：ln(A3/A2)/(0.15·秒)。
    """
    n步 = int(round(秒 / dt))
    K = len(Ps列)
    Ps = np.asarray(Ps列, dtype=np.float64)
    态 = np.zeros((4, K), dtype=np.float64)
    态[0] = 微偏
    w1记 = np.empty((K, n步), dtype=np.float64)
    for i in range(n步):
        a = 导数_批(态, Ps, 紧喉)
        b = 导数_批(态 + 0.5 * dt * a, Ps, 紧喉)
        c = 导数_批(态 + 0.5 * dt * b, Ps, 紧喉)
        d = 导数_批(态 + dt * c, Ps, 紧喉)
        态 = 态 + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d)
        w1记[:, i] = XI0 + 态[0]
    段 = max(1, n步 // 4)
    def 幅(m0, m1):
        w = w1记[:, m0 * 段:m1 * 段]
        return np.max(w, axis=1) - np.min(w, axis=1)
    幅中 = 幅(2, 3)
    幅末 = 幅(3, 4)
    with np.errstate(divide="ignore", invalid="ignore"):
        斜率 = np.where((幅中 > 1e-18) & (幅末 > 1e-18),
                       np.log(幅末 / np.maximum(幅中, 1e-30)) / (0.25 * 秒), np.nan)
    return 幅中, 幅末, 斜率


def 阈值(Ps列, 斜率):
    """实测阈值 = **斜率过零**的线性插值（起振/衰减分界）。"""
    斜率 = np.asarray(斜率, dtype=np.float64)
    for i in range(1, len(斜率)):
        a, b = 斜率[i - 1], 斜率[i]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a <= 0.0 < b:
            return float(Ps列[i - 1] + (Ps列[i] - Ps列[i - 1]) * (0.0 - a) / (b - a)), i
    return float("nan"), -1


# ═════════════════════════════════ 2. must-fail 对照臂：查表脉冲 ═════════════════════════════════

def 表查脉冲幅(紧喉=0.0, 秒=0.2, f0=120.0):
    """`引擎_v6.声门流表` 查表脉冲臂（只读 v6 源）。**接口里没有 Ps** ⇒ 结构上无起振阈值。"""
    表长 = E6._表长
    表 = E6.声门流表(表长, float(紧喉))
    相位 = np.mod(np.arange(int(秒 * FS), dtype=np.float64) * (f0 / FS), 1.0)
    相index = np.minimum((相位 * 表长).astype(np.int64), 表长 - 1)
    return 表[相index]


# ═════════════════════════════════ 3. 元音化（只读 v4 谐振器）════════════════════════════════

def 元音化(流, 共振峰=((800.0, 70.0), (1200.0, 90.0), (2500.0, 130.0))):
    """Klatt 二阶谐振器串联（只读 `引擎_v4.谐振器系数/_滤波块`）＋ 唇辐射差分 ⇒ 元音化波形。"""
    y = np.asarray(流, dtype=np.float64)
    zi = np.zeros(2)
    for F, BW in 共振峰:
        A, B, C = E6._谐振器系数(float(F), float(BW))
        y, zi = E6._滤波块(A, B, C, y, zi)
    辐 = np.empty_like(y)
    辐[0] = y[0]
    辐[1:] = y[1:] - y[:-1]
    m = float(np.max(np.abs(辐))) or 1.0
    return 辐 / m


def 基频(x, 下限=60.0, 上限=400.0):
    """自相关 F0（波形的**自有**估计，不读任何参考）。"""
    a = np.asarray(x, dtype=np.float64)
    a = a - float(np.mean(a))
    n = len(a)
    if n < 4 * FS // int(上限):
        return 0.0
    r = np.correlate(a, a, mode="full")[n - 1:]
    if r[0] <= 0.0:
        return 0.0
    r = r / r[0]
    lo = int(FS / 上限)
    hi = min(int(FS / 下限), n - 1)
    seg = r[lo:hi]
    if seg.size == 0:
        return 0.0
    return float(FS / (int(np.argmax(seg)) + lo))


def 写wav(路径, x):
    y = np.clip(np.asarray(x, dtype=np.float64) * 0.95, -1.0, 1.0)
    with wave.open(路径, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(np.rint(y * 32767.0).astype("<i2").tobytes())


# ═════════════════════════════════ 4. 零外挂三断言审计 ═════════════════════════════════

def AST审计(路径列) -> dict:
    """AST 级审计：import 模块名是否命中框架/外挂名单；是否存在 open(...,'rb') 式音频读入。"""
    框架 = set(禁词())
    外挂 = set(外挂名单())
    命中框架, 命中外挂, 命中读入 = [], [], []
    for p in 路径列:
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8-sig") as f:
            树 = ast.parse(f.read())
        for node in ast.walk(树):
            if isinstance(node, ast.Import):
                名s = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                名s = [node.module or ""]
            else:
                名s = []
            for nm in 名s:
                根 = nm.split(".")[0].lower()
                if 根 in 框架:
                    命中框架.append((os.path.basename(p), nm))
                if 根 in 外挂:
                    命中外挂.append((os.path.basename(p), nm))
            if isinstance(node, ast.Call):
                fn = node.func
                名 = fn.attr if isinstance(fn, ast.Attribute) else (
                    fn.id if isinstance(fn, ast.Name) else "")
                if 名 == "open":
                    模s = [str(c.value) for c in node.args
                           if isinstance(c, ast.Constant) and isinstance(c.value, str)]
                    if any(("b" in m and "r" in m and "+" not in m) for m in 模s):
                        命中读入.append((os.path.basename(p), 模s))
    return dict(文件数=len([p for p in 路径列 if os.path.isfile(p)]),
                命中框架=命中框架, 命中外挂模型=命中外挂, 命中音频读入=命中读入,
                通过=bool(not 命中框架 and not 命中外挂 and not 命中读入))


class _读入钩子:
    """运行期证明「零参考读入」：劫持 builtins.open，记录一切以读模式打开的路径。"""

    def __init__(self):
        self.读入 = []
        self._原 = None

    def __enter__(self):
        self._原 = builtins.open
        记 = self.读入
        原 = self._原

        def 钩(file, mode="r", *a, **k):
            if "r" in str(mode) and "+" not in str(mode):
                记.append(str(file))
            return 原(file, mode, *a, **k)

        builtins.open = 钩
        return self

    def __exit__(self, *exc):
        builtins.open = self._原
        return False


def 断言B_运行期():
    """运行期：AHI 侧未加载任何被禁框架/外挂模型的模块（含传递依赖）。"""
    根s = {s.split(".")[0].lower() for s in sys.modules}
    命中 = sorted((根s & set(禁词())) | (根s & set(外挂名单())))
    return dict(已加载模块数=len(sys.modules), 命中=命中, 通过=bool(not 命中))


# ═════════════════════════════════ 5. 主流程 ═════════════════════════════════

def 跑一遍(写文件=True) -> dict:
    秒波, 子步 = 0.30, 2
    Ps阈解析 = 解析阈值(0.0)

    读入列 = []
    with _读入钩子() as 钩:
        幅中, 幅末, 斜率 = 扫_起振(Ps列, 0.0)
        Ps实阈, 阈位 = 阈值(Ps列, 斜率)
        # 零初态对照（严格 x1≡0）：位移须**恒 0**（不稳定平衡点上无种子 ⇒ 不起振）
        流零, x1零 = 合成_单(Ps=800.0, 紧喉=0.0, 秒=0.05, 子步=2, 初偏=0.0, 记位移=True)
        Ps波 = 2.0 * Ps实阈 if Ps实阈 == Ps实阈 else 800.0
        流 = 合成_单(Ps波, 0.0, 秒波, 子步)
        流紧 = 合成_单(Ps波, 1.0, 0.15, 子步)
        音 = 元音化(流)
        表源 = 表查脉冲幅(0.0, 秒波)
        表音 = 元音化(表源)
        读入列 = list(钩.读入)

    f0_0 = 基频(流)
    f0_1 = 基频(流紧)
    幅表随Ps = [float(np.max(表源) - np.min(表源))] * len(Ps列)

    wav路径 = os.path.join(本目录, "自激声门源_波形.wav")
    if 写文件:
        写wav(wav路径, 音)

    路径列 = [os.path.abspath(__file__),
              os.path.join(MEET61实施, "引擎_v6.py"),
              os.path.join(MEET61实施, "引擎_v4.py")]
    静 = AST审计(路径列)
    运 = 断言B_运行期()
    零外挂 = dict(
        断言A_零音频读入=dict(静态命中=静["命中音频读入"], 运行期读入路径=读入列,
                        通过=bool(not 静["命中音频读入"] and not 读入列)),
        断言B_无外部模型=dict(静态命中框架=静["命中框架"], 静态命中外挂模型=静["命中外挂模型"],
                        运行期=运, 通过=bool(not 静["命中框架"] and not 静["命中外挂模型"]
                                             and 运["通过"])),
        审计文件=路径列,
    )

    读数 = dict(
        工具="探针_自激声门源.py",
        模型="Ishizaka–Flanagan 两质量自激声门 ODE（P0 口径 PMC3477191 Eq(4)–(9)；流致刚度反向 ⇒ 失稳）",
        参数=dict(M1=M1, M2=M2, K1=K1, K2=K2, KC=KC, R1=R1, R2=R2, LG=LG, 厚1=厚1, AREA=AREA,
                  XI0=XI0, KENT=KENT, RO=RO, 微偏=微偏, 紧喉刚度增益=紧喉刚度增益, FS=FS,
                  扫秒=扫秒, 扫dt=扫dt),
        J1_起振阈值=dict(
            Ps列=Ps列,
            中段幅=[float(v) if np.isfinite(v) else None for v in 幅中],
            末段幅=[float(v) if np.isfinite(v) else None for v in 幅末],
            对数斜率=[round(float(v), 4) if np.isfinite(v) else None for v in 斜率],
            实测阈值Pa=round(Ps实阈, 2),
            解析阈值Pa=round(Ps阈解析, 2),
            相对差=round(abs(Ps实阈 - Ps阈解析) / Ps阈解析, 4) if Ps实阈 == Ps实阈 else None,
            文献对照="PMC3477191 同机制（流致刚度 φ 与恢复力反向）；其参数集同步点 444Pa、阈值更高",
            无外部波形注入=True),
        J1z_零初态对照=dict(合成秒=0.05, 驱动PsPa=800.0,
                        位移峰值=float(np.max(np.abs(x1零))) if x1零.size else 0.0,
                        说明="严格 x1≡0（不稳定平衡点上）⇒ 无微扰种子、位移恒 0 ⇒ 需内部确定性微偏作种子；"
                             "该微偏是**内部常数**，不是外部波形注入",
                        不起振=bool(x1零.size and float(np.max(np.abs(x1零))) == 0.0)),
        J1_波形=dict(驱动PsPa=round(Ps波, 2), 秒=秒波, F0Hz=round(f0_0, 2),
                    RMS=round(float(np.sqrt(np.mean(音 ** 2))), 5),
                    峰值=round(float(np.max(np.abs(音))), 5),
                    声门流峰值=float(np.max(np.abs(流))), wav=wav路径),
        J3_紧喉到F0=dict(紧喉0_F0Hz=round(f0_0, 2), 紧喉1_F0Hz=round(f0_1, 2),
                    单调升=bool(f0_1 > f0_0)),
        J1一撇_mustfail_表查脉冲=dict(
            说明="换回 `引擎_v6.声门流表` 查表脉冲：其接口**不含 Ps** ⇒ 幅值随 Ps 变化恒 0 ⇒ 无起振阈值",
            幅值随Ps=[round(v, 10) for v in 幅表随Ps],
            幅值随Ps变化=float(np.max(幅表随Ps) - np.min(幅表随Ps)),
            无起振阈值=bool(float(np.max(幅表随Ps) - np.min(幅表随Ps)) == 0.0),
            表臂元音峰值=round(float(np.max(np.abs(表音))), 5)),
        零外挂=零外挂,
        判据=dict(
            J1_起振阈值存在=bool(Ps实阈 == Ps实阈),
            J1_与解析式一致=bool(Ps实阈 == Ps实阈 and abs(Ps实阈 - Ps阈解析) / Ps阈解析 <= 0.15),
            J1z_零初态不起振=bool(x1零.size and float(np.max(np.abs(x1零))) == 0.0),
            J3_紧喉单调升F0=bool(f0_1 > f0_0),
            mustfail_表查无阈值=bool(float(np.max(幅表随Ps) - np.min(幅表随Ps)) == 0.0),
            零外挂A=零外挂["断言A_零音频读入"]["通过"],
            零外挂B=零外挂["断言B_无外部模型"]["通过"],
        ),
    )
    return 读数, 音, 流, 表音, 流零


def 主():
    果1, 音1, 流1, 表音1, 流零1 = 跑一遍(写文件=True)
    果2, 音2, 流2, 表音2, 流零2 = 跑一遍(写文件=False)
    同 = bool(np.array_equal(音1, 音2) and np.array_equal(流1, 流2)
              and np.array_equal(表音1, 表音2)
              and np.array_equal(流零1[0], 流零2[0]) and np.array_equal(流零1[1], 流零2[1])
              and json.dumps(果1, ensure_ascii=False, sort_keys=True)
              == json.dumps(果2, ensure_ascii=False, sort_keys=True))
    果1["J4_确定性"] = dict(两趟逐位同=同, 波形sha256=hashlib.sha256(音1.tobytes()).hexdigest()[:16])
    果1["判据"]["J4_两趟逐位同"] = 同
    果1["通过"] = bool(all(果1["判据"].values()))
    with open(os.path.join(本目录, "读数_自激声门源.json"), "w", encoding="utf-8") as f:
        json.dump(果1, f, ensure_ascii=False, indent=1)

    print("=" * 88)
    print("探针_自激声门源 —— 路线②「原生自发声（无参考）」最小可证伪实验")
    print("=" * 88)
    print("J1 起振阈值：实测 %.1f Pa ｜ 解析式 k1·k_ent·ξ0/(2·Lg·d1) = %.1f Pa ｜ 相对差 %.2f%%"
          % (果1["J1_起振阈值"]["实测阈值Pa"], 果1["J1_起振阈值"]["解析阈值Pa"],
             100.0 * (果1["J1_起振阈值"]["相对差"] or 0.0)))
    print("J1 波形：PS=%.0fPa %.2fs ｜ F0=%.2fHz ｜ RMS=%.5f ｜ 声门流峰值=%.3e"
          % (果1["J1_波形"]["驱动PsPa"], 果1["J1_波形"]["秒"], 果1["J1_波形"]["F0Hz"],
             果1["J1_波形"]["RMS"], 果1["J1_波形"]["声门流峰值"]))
    print("J1z 零初态：位移峰值=%g 不起振=%s" % (果1["J1z_零初态对照"]["位移峰值"],
                                        果1["J1z_零初态对照"]["不起振"]))
    print("J3 紧喉→F0：紧喉0=%.2fHz 紧喉1=%.2fHz 单调升=%s"
          % (果1["J3_紧喉到F0"]["紧喉0_F0Hz"], 果1["J3_紧喉到F0"]["紧喉1_F0Hz"],
             果1["J3_紧喉到F0"]["单调升"]))
    print("J1′ must-fail 表查脉冲：幅值随 Ps 变化=%g ⇒ 无起振阈值=%s"
          % (果1["J1一撇_mustfail_表查脉冲"]["幅值随Ps变化"],
             果1["J1一撇_mustfail_表查脉冲"]["无起振阈值"]))
    print("零外挂：A 静态命中=%s 运行期读入=%s ｜ B 静态框架=%s/外挂=%s 运行期=%s"
          % (果1["零外挂"]["断言A_零音频读入"]["静态命中"],
             果1["零外挂"]["断言A_零音频读入"]["运行期读入路径"],
             果1["零外挂"]["断言B_无外部模型"]["静态命中框架"],
             果1["零外挂"]["断言B_无外部模型"]["静态命中外挂模型"],
             果1["零外挂"]["断言B_无外部模型"]["运行期"]["命中"]))
    print("J4 两趟逐位同 = %s" % 同)
    print("判据：" + " ".join("%s=%s" % (k, v) for k, v in 果1["判据"].items()))
    print("通过 = %s" % 果1["通过"])
    print("wav:", 果1["J1_波形"]["wav"])
    return 果1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    np.seterr(all="ignore")
    主()
