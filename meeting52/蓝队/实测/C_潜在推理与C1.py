# -*- coding: utf-8 -*-
"""C_潜在推理与C1.py —— 蓝队 worker C（Level-2）· C 组四臂真跑（2026-10-17）。

仅本文件 + `C_结果.json`（同目录）。**不改任何既有件**：所有既有件一律 importlib 只读加载。
纪律：`python.exe -B`（不落 __pycache__）；零 git、零装包；阈值**写死在文件头**，不据结果改。

四臂
────
C1 执行覆盖：`贯通_多核心.造双区室(seed=310)` → 120 拍；monkeypatch 计 `桥.思核.步` 调用数；
   报 调用次数 / 迹条数 / 造桥墙钟 / 120 拍墙钟。（已知 0/120，本件为独立复测）
C2 接线原型（纯脚本内，不改既有件）：包装两区室核
       B_t = 前向_双区室(...)["世界态"]；z_t = 思核.迭代(z0,K)[-1]；
       o_t = 归((1+软限(z_t)) ⊙ B_t)；读出用 核.读出。
   干预 ①替换 z（另一条轨迹）②扰动 z←单位(z+δu)，δ∈{0.01,0.05,0.10} ③冻结 z≡z0
        ④decoy：同分布随机 z 轨迹（同 K 同步数）。
   指标：TV（|Δo| 均值 + 直方图 TVD 两口径）、argmax 改变率、Acc 掉幅（n=200 决策）。
C3 C1 数值签名：对被判"是"的 `_相对门` 算 Σw / S1（输入整体缩放 0.5 后 |Σw 变化|）/
   S2（逐键 ε 差分 offdiag/diag）/ min w；再对 5 个整改候选离线复算。
C4 候选代价（解析）：J=4, D=1024，每拍乘加/exp 变化。

判定规则（预注册，见 阈值 字典；不得事后改）
────────────────
C2 替换/扰动：TV均值 ≥0.10 ∧ TVD ≥0.10 ∧ 改变率 ≥0.10 ⇒ 通过。
C2 冻结：Acc 掉 ≥0.05 ⇒ 通过。
C2 decoy：|Acc_decoy − Acc_真| < 1SE(0.029) ⇒ **必须写"不可判定"**（不得写"无影响"/"z 无信息"）。
C3：S1 通过（Σw 尺度不变 ≈1）∧ S2>0.05 ⇒ 是；S1 通过 ∧ S2≈0 ⇒ 条件；¬S1 ⇒ 否。
C3 阳性对照：dummy softmax 必须"是"；逐维门控必须"否"。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np

# ══════════════════════════ 预注册阈值（写死；不据结果改）══════════════════════════
阈值 = dict(
    TV均值=0.10,          # C2：替换/扰动 |Δo| 均值 下界
    TVD=0.10,             # C2：替换/扰动 直方图 TVD 下界（固定分箱 [-16,16]×128）
    改变率=0.10,          # C2：argmax 改变率 下界
    冻结Acc掉=0.05,       # C2：冻结 z 的 Acc 掉幅下界
    decoy_1SE=0.029,      # C2：decoy 与真 z 的 Acc 差 < 1SE ⇒ 不可判定
    S1容差=1e-6,          # C3：|Σw−1| 与 |Σw(0.5a)−Σw(a)| 的"≈0"容差
    S2=0.05,              # C3：S2 下界（>0.05 ⇒ 有跨键连续耦合）
    ε差分=1e-6,           # C3：逐键 ε 差分步长
    n决策=200,            # C2：评测决策数（≥200）
    n训练=200,            # C2：读出拟合试次数
    K=4,                  # C2：思核 迭代步数（z_t = 迭代(z0,K)[-1]）
    J=4,                  # C4：竞争键数（口径）
    D=1024,               # C4：维数（口径）
    分箱=np.linspace(-16.0, 16.0, 129),
)

HERE = os.path.dirname(os.path.abspath(__file__))                     # …/meeting52/蓝队/实测
AHI4 = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))                      # …/AHI4
结果路径 = os.path.join(HERE, "C_结果.json")

_M49 = os.path.join(AHI4, "meeting49", "实施")
_M41 = os.path.join(AHI4, "meeting41", "实施")
_M39 = os.path.join(AHI4, "meeting39", "实施")
_M38 = os.path.join(AHI4, "meeting38", "实施")
for _p in (_M38, _M39, _M41, _M49):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def 载(名: str, 路径: str):
    """只读 importlib 加载（不写字节码：配合 python -B）。"""
    spec = importlib.util.spec_from_file_location(名, 路径)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ── 只读加载既有件 ──
核m = 载("C_两区室核_只读", os.path.join(_M49, "算子_两区室核.py"))          # 软限/_归/生成序列/两区室核
思m = 载("C_相空间算子T_只读", os.path.join(_M39, "内核_相空间算子T.py"))      # 思核
贯m = 载("C_贯通多核心_只读", os.path.join(_M49, "贯通_多核心.py"))           # 造双区室/跑

软限 = 核m.软限
归 = 核m._归
生成序列 = 核m.生成序列
思核类 = 思m.思核
T2参数 = dict(思m.T2参数)


def 单位(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(x))
    return x / n if n > 1e-12 else x


def 直方TV(a, b) -> float:
    p, _ = np.histogram(np.asarray(a, dtype=np.float64), bins=阈值["分箱"])
    q, _ = np.histogram(np.asarray(b, dtype=np.float64), bins=阈值["分箱"])
    p = p / max(1, int(p.sum()))
    q = q / max(1, int(q.sum()))
    return float(0.5 * np.abs(p - q).sum())


# ══════════════════════════════ C1 执行覆盖 ══════════════════════════════
def 臂C1(拍数: int = 120, seed: int = 310) -> dict:
    """造双区室(seed=310) → 120 拍；monkeypatch 计 `桥.思核.步` 调用数与迹条数。"""
    t0 = time.perf_counter()
    双 = 贯m.造双区室(seed=int(seed))
    t造 = time.perf_counter() - t0
    桥 = 双["桥"]
    sx = getattr(桥, "思核", None)
    if sx is None:
        return dict(异常="思核未挂载（造桥未起思核）", 造桥秒=t造)
    计 = {"步": 0}

    def 计步(*a, **k):
        计["步"] += 1
        return 原步(*a, **k)

    原步 = sx.步
    迹前 = len(sx.迹)
    sx.步 = 计步                       # 实例属性遮蔽：外部 self.步 与 内部 self.步(...) 都被计到
    t0 = time.perf_counter()
    果 = 贯m.跑(双, int(拍数))
    t跑 = time.perf_counter() - t0
    try:
        sx.步 = 原步                   # 还原（不改类、不改文件）
    except Exception:
        pass
    迹后 = len(sx.迹)
    宿 = 果.get("内核态") or {}
    出 = dict(
        拍数=int(拍数), seed=int(seed), 已跑=int(果.get("已跑", -1)), 异常=果.get("异常"),
        造桥秒=round(t造, 3), 跑120拍秒=round(t跑, 3),
        思核步调用数=int(计["步"]), 迹条数=迹后, 迹增量=int(迹后 - 迹前),
        内核态_思核=宿.get("思核"), 内核态_自模型=宿.get("自模型"),
        内核态_真学习=宿.get("真学习"),
        自我态有限=bool(果.get("自我态有限")), 自我态界=float(果.get("自我态界", float("nan"))),
    )
    出["判定"] = "0 次调用 / 0 条迹 ⇒ 主路径 120 拍**执行覆盖 0/120**（与已知一致）" \
        if (出["思核步调用数"] == 0 and 出["迹条数"] == 0) else "有执行（与已知 0/120 不符，需复核）"
    出["注"] = "计步包的是实例属性，类未被改；迭代() 内部若调 self.步 也计入 ⇒ 计数是上界不是下界"
    return dict(结果=出, _双=双)


# ══════════════════════════════ C2 接线原型 ══════════════════════════════
_背景缓存: dict = {}


def _背景(D: int) -> np.ndarray:
    """固定双极背景（种子 7000，**全试次相同**）——C2 主臂的感觉基底。"""
    if int(D) not in _背景缓存:
        _背景缓存[int(D)] = np.asarray(贯m.输入向量(0, int(D)), dtype=np.float64).copy()
    return _背景缓存[int(D)].copy()


def _输入D1(x: float, t: int, D: int) -> np.ndarray:
    """仪器校准设计 D1：背景 = 贯通.输入向量(t)（逐试次随机 ±1），前两维承载音高 x。

    缺点（本件实测）：1024 维逐试次随机背景使 B 的跨试次方差由**噪声**主导
    ⇒ 200 样本线性读出无法定位音高方向，基线 Acc≈0.55（≈随机）⇒ 干预指标无分辨率。
    """
    u = np.asarray(贯m.输入向量(int(t), int(D)), dtype=np.float64).copy()
    u[0] = 2.0 * float(x) - 1.0
    u[1] = 1.0 - 2.0 * float(x)
    return u


def _输入D2(x: float, t: int, D: int) -> np.ndarray:
    """C2 主臂设计 D2：**固定**双极背景 + 16 维音高片 = 2x−1（跨试次差异只来自 x）。"""
    u = _背景(D)
    u[:16] = 2.0 * float(x) - 1.0
    return u


def _读出拟合(o列: np.ndarray, c列: np.ndarray) -> np.ndarray:
    """用系统自己的读出参数 核.读出（(2,D)）：方向 = 类均值差（闭式、确定性、无外挂优化器）。"""
    o1 = o列[c列 == 1]
    o0 = o列[c列 == 0]
    d = o1.mean(axis=0) - o0.mean(axis=0)
    尺 = float(np.max(np.abs(d))) if d.size else 0.0
    d = d / 尺 * 8.0 if 尺 > 1e-12 else d
    return 归(np.stack([-0.5 * d, 0.5 * d]))          # (2,D)，行 0/1 = 两个类的分


def 跑C2(双: dict, 输入f, 设计名: str) -> dict:
    桥, 核, 宿主 = 双["桥"], 双["核"], 双["宿主"]
    sx = getattr(桥, "思核", None)
    if sx is None:
        return dict(异常="思核未挂载")
    D = int(核.D)
    J = int(sx.维度)
    if J != D:
        return dict(异常="思核维度(%d) != 核.D(%d)：z 无法逐元素乘到 B 上" % (J, D))
    K = int(阈值["K"])
    n训, n测 = int(阈值["n训练"]), int(阈值["n决策"])
    n = n训 + n测
    序列 = 生成序列(52031, n, (0.35, 0.65))
    xs, bs = 序列["x"], 序列["边界"]

    样 = []
    for t in range(n):
        x = float(xs[t])
        u = 输入f(x, t, D)
        宿主.设来源(桥, True)
        桥.步_向量(u.astype(np.float32), 原文=None)
        R = 核.前向_双区室(u.reshape(1, -1), None, 1)
        B = np.asarray(R["世界态"][0], dtype=np.float64)
        H = np.asarray(桥.eng.hdc.H, dtype=np.float64)
        z0 = 单位(np.sign(H))
        轨 = sx.迭代(z0, K)                        # [z0, z1, …, zK]
        z真, z冻 = np.asarray(轨[-1], float), np.asarray(轨[0], float)
        z骗 = np.asarray(sx.迭代(单位(np.random.default_rng(77000 + t).standard_normal(D)), K)[-1], float)
        o真 = 归((1.0 + 软限(z真)) * B)
        核.设自我态向量(归(软限(核.自我态 + np.asarray(R["输出"][0], dtype=np.float64))))
        样.append(dict(t=t, x=x, b=float(bs[t]), B=B, z真=z真, z冻=z冻, z骗=z骗, o真=o真,
                       c=int(1 if x > float(bs[t]) else 0)))

    训 = 样[:n训]
    测 = 样[n训:]
    核.读出 = _读出拟合(np.stack([s["o真"] for s in 训]),
                       np.array([s["c"] for s in 训], dtype=np.int64))
    读出 = 核.读出

    def 判(o):
        return int(np.argmax(读出 @ np.asarray(o, dtype=np.float64)))

    def Acc(选):
        选 = np.asarray(选, dtype=np.int64)
        真 = np.array([s["c"] for s in 测], dtype=np.int64)
        if 选.size == 0:
            return float("nan")
        return float((选 == 真).mean())

    原 = [判(s["o真"]) for s in 测]
    Acc原 = Acc(原)
    o原列 = np.concatenate([s["o真"] for s in 测])

    def 跑臂(名, 生成z, 判据键):
        os列, 选 = [], []
        for i, s in enumerate(测):
            z = 生成z(i, s)
            o = 归((1.0 + 软限(z)) * s["B"])
            os列.append(o)
            选.append(判(o))
        o列 = np.concatenate(os列)
        改变率 = float(np.mean(np.asarray(选, np.int64) != np.asarray(原, np.int64)))
        Acc臂 = Acc(选)
        行 = dict(臂=名, TV均值=round(float(np.mean(np.abs(o列 - o原列))), 6),
                 TVD=round(直方TV(o列, o原列), 6),
                 改变率=round(改变率, 4), Acc=round(Acc臂, 4),
                 Acc掉=round(Acc原 - Acc臂, 4),
                 裁剪比=round(float(np.mean(np.abs(o列) >= 16.0)), 6),
                 判据=判据键)
        if 判据键 == "TV+改变率":
            行["通过"] = bool(行["TV均值"] >= 阈值["TV均值"] and 行["TVD"] >= 阈值["TVD"]
                            and 行["改变率"] >= 阈值["改变率"])
            行["逐项达标"] = dict(TV均值=bool(行["TV均值"] >= 阈值["TV均值"]),
                              TVD=bool(行["TVD"] >= 阈值["TVD"]),
                              改变率=bool(行["改变率"] >= 阈值["改变率"]))
        elif 判据键 == "Acc掉":
            行["通过"] = bool(行["Acc掉"] >= 阈值["冻结Acc掉"])
        elif 判据键 == "decoy":
            SE = float(np.sqrt(max(Acc原 * (1 - Acc原), 1e-9) / n测 + max(Acc臂 * (1 - Acc臂), 1e-9) / n测))
            差 = float(Acc臂 - Acc原)
            行["Acc差(decoy−真)"] = round(差, 4)
            行["SE_差_理论"] = round(SE, 4)
            行["1SE阈值"] = 阈值["decoy_1SE"]
            行["通过"] = None
            行["判定"] = "不可判定" if abs(差) < 阈值["decoy_1SE"] else (
                "decoy 显著低于真 z ⇒ 真 z 含决策相关信息" if 差 < 0 else "decoy 反高 ⇒ 异常")
        else:
            行["通过"] = None
        return 行

    def z替换(i, s):
        return np.asarray(测[(i + 1) % len(测)]["z真"], dtype=np.float64)

    def z扰动(δ):
        def f(i, s):
            r = 单位(np.random.default_rng(88000 + i).standard_normal(D))
            return 单位(np.asarray(s["z真"], float) + δ * r)
        return f

    def z冻结(i, s):
        return np.asarray(s["z冻"], dtype=np.float64)

    def z骗(i, s):
        return np.asarray(s["z骗"], dtype=np.float64)

    臂列 = [跑臂("替换z(另一条轨迹)", z替换, "TV+改变率")]
    for δ in (0.01, 0.05, 0.10):
        臂列.append(跑臂("扰动z δ=%.2f" % δ, z扰动(δ), "TV+改变率"))
    臂列.append(跑臂("冻结z≡z0", z冻结, "Acc掉"))
    臂列.append(跑臂("decoy(同分布随机z,同K)", z骗, "decoy"))

    # ── 诊断（非承重；不参与判定）──
    # ①z 单独解码 x 的能力（门控前的信息侧，与门控幅度无关）
    c训 = np.array([s["c"] for s in 训], np.int64)
    真c = np.array([s["c"] for s in 测], np.int64)
    d_z = _读出拟合(np.stack([s["z真"] for s in 训]), c训)
    Acc_z = float((np.array([int(np.argmax(d_z @ s["z真"])) for s in 测]) == 真c).mean())
    d_B = _读出拟合(np.stack([s["B"] for s in 训]), c训)
    Acc_B = float((np.array([int(np.argmax(d_B @ s["B"])) for s in 测]) == 真c).mean())
    # ②z 的跨试次变异（若 z 几乎不随试次变，置换/冻结类干预在构造上就无效应）
    Z = np.stack([s["z真"] for s in 测])
    邻距 = float(np.mean(np.linalg.norm(Z[1:] - Z[:-1], axis=1)))
    跨距 = float(np.mean(np.linalg.norm(Z - Z.mean(axis=0), axis=1)))
    # ③仅凭 x 的最佳 1D 阈值准确率 = 任何"只看 x"的决策器的上界（仪器分辨率天花板）
    上界 = 0.0
    测x = np.array([s["x"] for s in 测])
    for th in np.linspace(0.0, 1.0, 201):
        上界 = max(上界, float(np.mean((测x > th).astype(np.int64) == 真c)))
    # ④放大臂：z×16（软限 仍在近似线性区：16×0.025≈0.4 ≪ 幅16），看"z 的相位"能否穿透；
    #   纯诊断，不是候选整改 —— 用来区分"幅度不足"与"信息为空"这两种解释。
    原读出 = 核.读出
    放大 = []
    for 名, gz in (("替换@放大16", z替换), ("decoy@放大16", z骗)):
        os列, 选 = [], []
        for i, s in enumerate(测):
            o = 归((1.0 + 软限(16.0 * np.asarray(gz(i, s), float))) * s["B"])
            os列.append(o)
            选.append(int(np.argmax(原读出 @ o)))
        o列 = np.concatenate(os列)
        放大.append(dict(臂=名,
                      TV均值=round(float(np.mean(np.abs(o列 - o原列))), 6),
                      TVD=round(直方TV(o列, o原列), 6),
                      改变率=round(float(np.mean(np.asarray(选, np.int64) != np.asarray(原, np.int64))), 4),
                      Acc=round(Acc(选), 4), Acc掉=round(Acc原 - Acc(选), 4)))

    诊断 = dict(
        Acc原=round(Acc原, 4),
        z单独解码Acc=round(Acc_z, 4),
        B单独解码Acc=round(Acc_B, 4),
        x阈值上界=round(上界, 4),
        仪器可用=bool(Acc原 >= 上界 - 0.05),
        z的幅值=dict(均值=round(float(np.mean([np.mean(np.abs(s["z真"])) for s in 测])), 6),
                   极大=round(float(np.max([np.max(np.abs(s["z真"])) for s in 测])), 6)),
        门控偏离1的均值=round(float(np.mean([np.mean(np.abs(软限(s["z真"]))) for s in 测])), 6),
        z跨试次=dict(相邻试次距离=round(邻距, 6), 距均值距离=round(跨距, 6)),
        放大臂=放大,
        注="放大臂仅诊断『门控幅度是否锁死』，不构成任何整改候选；不得据此改写 C2 判定",
    )
    return dict(
        设计=设计名,
        配置=dict(D=D, 思核维度=J, 竞争键数=int(len(sx.竞争表)), K=K, n训练=n训, n决策=n测,
                序列种子=52031, 边界=(0.35, 0.65),
                读出拟合="核.读出 ← 类均值差方向(闭式)", 自我态更新="同 贯通.一拍 步骤③",
                接线="o=归((1+软限(z))⊙B)，B=前向_双区室 世界态"),
        Acc原=round(Acc原, 4),
        Acc原_前半=round(float(np.mean(np.asarray(原[: n测 // 2], np.int64)
                                   == np.array([s["c"] for s in 测[: n测 // 2]], np.int64))), 4),
        臂=臂列, decoy=臂列[-1], 诊断=诊断,
        注="z 与 B 的维度必须相等（%d==%d）；本原型在脚本内，既有系统主路径**没有**这条接线（见 C1 0/120）" % (J, D),
    )


def 臂C2(双校准: dict, 双主: dict) -> dict:
    """C2：先跑仪器校准设计 D1（随机背景），再跑主臂 D2（固定背景+16 维音高片）。"""
    校 = 跑C2(双校准, _输入D1, "D1 仪器校准：逐试次随机 ±1 背景 + 2 维音高")
    主 = 跑C2(双主, _输入D2, "D2 主臂：固定双极背景 + 16 维音高片(2x−1)")
    return dict(
        主=主, 仪器校准=校,
        口径=("主臂判定只取 D2。D1 一并如实上报：它的基线 Acc≈0.55（≈随机）说明"
             "『1024 维逐试次随机背景 + 200 样本线性读出』这个仪器在 D1 下**没有分辨率**，"
             "因此 D1 的干预读数（TV/改变率）只能当『仪器下限』看，不能当『z 无因果』的证据。"),
        换设计声明=("阈值、干预集合、K、n 全部未动；只把**感觉输入的背景**由逐试次随机改成固定 —— "
                "理由是仪器分辨率（诊断 x阈值上界 vs Acc原），不是因为干预结果不显著。"),
    )


# ══════════════════════════════ C3 式级签名 ══════════════════════════════
def 签名(f, 样本列, ε=None) -> dict:
    """对权重生成器 f(a→w 向量) 算 Σw / S1 / S2 / min w。"""
    ε = 阈值["ε差分"] if ε is None else float(ε)
    Σw列, minw列, S2列, S1列 = [], [], [], []
    for a in 样本列:
        a = np.asarray(a, dtype=np.float64)
        w = np.asarray(f(a), dtype=np.float64)
        Σw列.append(float(w.sum()))
        minw列.append(float(w.min()))
        S1列.append(abs(float(np.asarray(f(0.5 * a), dtype=np.float64).sum()) - float(w.sum())))
        J = a.size
        for j in range(J):
            a2 = a.copy()
            a2[j] += ε
            Δ = np.asarray(f(a2), dtype=np.float64) - w
            diag = abs(float(Δ[j]))
            off = max([abs(float(Δ[i])) for i in range(J) if i != j] or [0.0])
            S2列.append(off / diag if diag > 1e-18 else 0.0)
    偏离 = max(abs(x - 1.0) for x in Σw列)
    S1值 = max(S1列)
    S1通过 = bool(偏离 <= 阈值["S1容差"] and S1值 <= 阈值["S1容差"])
    S2值 = float(np.mean(S2列)) if S2列 else 0.0
    判定 = "否" if not S1通过 else ("是" if S2值 > 阈值["S2"] else "条件")
    return dict(Σw均值=round(float(np.mean(Σw列)), 6), Σw偏离1=float("%.3e" % 偏离),
                S1_收缩后Σw变化=float("%.3e" % S1值), S1通过=S1通过,
                S2=round(S2值, 4), minw=round(float(np.min(minw列)), 6),
                判定=判定)


def 臂C3(双: dict) -> dict:
    sx = 双["桥"].思核
    D = int(sx.维度)                     # 相空间维（生成样本 z 用）
    β, κ = float(sx.β门控), float(sx.κ门)
    竞争表 = {k: np.asarray(v, float) for k, v in sx.竞争表.items()}
    名表 = sorted(竞争表)
    J = int(len(名表))                   # 竞争键数（签名作用在 J 维驱动向量上）
    样本列 = []
    for t in range(12):
        z = 单位(np.asarray(贯m.输入向量(41000 + t, D), dtype=np.float64))
        a = np.array([float(np.dot(z, 竞争表[k])) for k in 名表])   # 逐键驱动 = cos(z, 竞争表[k])
        样本列.append(a)
    h = {k: 0.0 for k in 名表}                                    # 首拍习惯化 h=0（最保守）
    θ分位 = np.array([float(np.quantile([s[j] for s in 样本列], 0.9)) for j in range(J)])
    η能量 = 0.5

    def f真实(a):
        """M4 `_相对门`：softmax_j(β(a_j − ā) − κ h_j)（内核_相空间算子T.py:90-103，原式）。"""
        ā = float(np.mean(a))
        e = β * (a - ā)
        e = e - np.array([κ * h[k] for k in 名表])
        e = e - float(np.max(e))
        w = np.exp(e)
        return w / max(float(w.sum()), 1e-300)

    def f_softmax(a):                            # 阳性对照 d1：dummy softmax(β对照·a)，独立温度
        e = 1.0 * np.asarray(a, float)            # β对照=1.0（另一套温度，避免与 M4 数值同）
        e = e - float(np.max(e))
        w = np.exp(e)
        return w / max(float(w.sum()), 1e-300)

    def f逐维(a, θ):                             # 逐维门控：w_j = 软限(θ_j − a_j)（Σ≠1，逐键独立）
        return 软限(np.asarray(θ, float) - np.asarray(a, float))

    def f候1(a):                                   # 候选① 逐维门控 per-key θ=历史分位
        return f逐维(a, θ分位)

    def f候2(a):                                   # 候选② 逐维门控 θ=mean(全场)（伪造交叉耦合）
        return f逐维(a, np.full(J, float(np.mean(a))))

    def f候3(a):                                   # 候选③ 可分能量下降 V(c)=½c²（一步梯度 + ReLU）
        return np.maximum((1.0 - η能量) * np.asarray(a, float), 0.0)

    def f候4(a):                                   # 候选④ 硬竞争 top-1（one-hot）
        w = np.zeros(J)
        w[int(np.argmax(np.asarray(a, float)))] = 1.0
        return w

    def f候5(a):                                   # 候选⑤ top-2 保留全局归一（top-k 重归一）
        idx = np.argsort(-np.asarray(a, float), kind="stable")[:2]
        e = β * np.asarray(a, float)[idx]
        e = e - float(np.max(e))
        w = np.exp(e)
        w = w / max(float(w.sum()), 1e-300)
        out = np.zeros(J)
        out[idx] = w
        return out

    表 = [
        ("M4 真实 `_相对门`（思核 · softmax β=%.1f κ=%.1f）" % (β, κ), 签名(f真实, 样本列), "被审机制"),
        ("阳性对照 d1：dummy softmax", 签名(f_softmax, 样本列), "必须判『是』"),
        ("阴性对照 d5：逐维门控(per-key θ)", 签名(f候1, 样本列), "必须判『否』"),
        ("候选① 逐维门控 per-key θ=历史分位", 签名(f候1, 样本列), "整改候选"),
        ("候选② 逐维门控 θ=mean(全场)", 签名(f候2, 样本列), "整改候选（禁止参数区：θ 用全场均值）"),
        ("候选③ 可分能量下降 V(c)=½c²", 签名(f候3, 样本列), "整改候选"),
        ("候选④ 硬竞争 top-1", 签名(f候4, 样本列), "整改候选"),
        ("候选⑤ top-2 保留全局归一", 签名(f候5, 样本列), "整改候选（禁止参数区：top-k 重归一）"),
    ]
    行 = [dict(机制=名, **签, 角色=角) for 名, 签, 角 in 表]
    对照 = dict(d1判=行[1]["判定"], d5判=行[2]["判定"],
               阳性对照通过=bool(行[1]["判定"] == "是"), 阴性对照通过=bool(行[2]["判定"] == "否"))
    return dict(
        配置=dict(J=J, β=β, κ门=κ, 样本数=len(样本列), 驱动="a_j=cos(z, 竞争表[j])",
                h="首拍 0（最保守）", ε差分=阈值["ε差分"], 能量步长=η能量,
                θ分位="样本集内每键 0.9 分位", 判据="¬S1⇒否；S1∧S2>0.05⇒是；S1∧S2≈0⇒条件"),
        样本示例=[[round(float(v), 6) for v in s] for s in 样本列[:3]],
        表=行,
        对照=对照,
        判定汇总={r["机制"]: r["判定"] for r in 行},
    )


# ══════════════════════════════ C4 候选代价（解析）══════════════════════════
def 臂C4(J: int = 阈值["J"], D: int = 阈值["D"]) -> dict:
    J, D = int(J), int(D)
    cos乘加 = J * D                    # _相对门 的 J 次余弦
    共振乘加 = J * D                   # _共振场 的 Σ_j g·门·c·v
    基乘加 = cos乘加 + 共振乘加
    基exp = J
    行 = [
        dict(候选="基线 M4 一拍（cos J·D + 共振 J·D）", 乘加=基乘加, Δ乘加=0,
             exp=基exp, Δexp=0, 占比="1.00×", 说明="当前 `_相对门`+`_共振场`"),
        dict(候选="(ii) 逐维门控（exp→软限）", 乘加=基乘加 + 4 * J - 4 * J, Δ乘加=0,
             exp=0, Δexp=-J, 占比="1.00×",
             说明="软限=幅x/(幅+|x|)：4 乘加/键 ×J=+%d，与被替换的 exp 前处理(%d)相抵 ⇒ 净 0；per-key θ 另需 +J 标量读取" % (4 * J, 4 * J)),
        dict(候选="(iv) 可分能量下降 V(c)=½c²", 乘加=基乘加 + J * D, Δ乘加=J * D,
             exp=基exp, Δexp=0, 占比="%.2f×" % ((基乘加 + J * D) / 基乘加),
             说明="+J·D = +%d 乘加/拍（相对 %d ⇒ **+50%%**）" % (J * D, 基乘加)),
        dict(候选="(i) 硬竞争 top-1", 乘加=基乘加, Δ乘加=0, exp=0, Δexp=-J, 占比="1.00×",
             说明="−J exp = −%d exp/拍；改为 %d 次比较（argmax）" % (J, J - 1)),
        dict(候选="(iii) 移出链", 乘加=0, Δ乘加=-基乘加, exp=0, Δexp=-基exp, 占比="0",
             说明="0（少一路执行）；子集为空 ⇒ 只能作门禁，不能算整改（N5/K3）"),
        dict(候选="(v) top-2 保留全局归一（附加行，非 C4 原列）", 乘加=基乘加, Δ乘加=0,
             exp=2, Δexp=2 - J, 占比="1.00×",
             说明="排序 +2 exp；但 C3 判『是』⇒ 仍属注意力族，不得作为整改"),
    ]
    return dict(J=J, D=D, 基线=dict(乘加=基乘加, exp=基exp),
                口径="解析估计（式本身），不含内存/缓存；墙钟**未测**（§10 项2）",
                表=行)


# ══════════════════════════════ 主流程 ══════════════════════════════
def 快检():
    s = 造与维度()
    print(json.dumps(s, ensure_ascii=False, indent=1, default=str))


def 造与维度() -> dict:
    t0 = time.perf_counter()
    双 = 贯m.造双区室(seed=310)
    t = time.perf_counter() - t0
    桥 = 双["桥"]
    Φ = getattr(桥.eng.hdc, "Φ_act", None)
    H = np.asarray(桥.eng.hdc.H, dtype=np.float64)
    sx = 桥.思核
    return dict(造桥秒=round(t, 3), 核D=int(双["核"].D), 探针shape=list(np.shape(桥.探针)),
                Φ_act_shape=list(np.shape(Φ)) if Φ is not None else None,
                H形状=list(H.shape), 思核维度=(int(sx.维度) if sx is not None else None),
                竞争键=sorted(sx.竞争表) if sx is not None else None,
                思核β=(float(sx.β门控) if sx is not None else None), 迹条数=(len(sx.迹) if sx is not None else None),
                自模型迹=(len(桥.自模型.迹) if getattr(桥, "自模型", None) is not None else None))


def 主(拍数: int = 120):
    t全 = time.perf_counter()
    果 = dict(元=dict(件="C_潜在推理与C1.py", 角色="蓝队 worker C（Level-2）",
                     时间=time.strftime("%Y-%m-%dT%H:%M:%S"), python=sys.version.split()[0],
                     numpy=np.__version__, DSH="python -B；零 git；零装包；只读 importlib 加载",
                     口径="阈值写死在本文件头 `阈值` 字典"),
              阈值={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in 阈值.items()},
              未测=["C2 每拍墙钟（本件只报代价解析，见 C4 口径）",
                   "C2 的『真接线』：既有主路径 0/120，本件接线在脚本内 ⇒ 不是既有系统行为",
                   "跨进程/跨机确定性（不在 C 组范围）",
                   "C3 的 θ 分位在真实长程历史上的取值（本件只用 12 样本的历史分位）"])

    # ── C1 ──
    try:
        r1 = 臂C1(拍数)
        双 = r1["_双"]
        果["C1"] = r1["结果"]
        果["维度探针"] = dict(核D=int(双["核"].D), 思核维度=int(双["桥"].思核.维度),
                            H形状=list(np.shape(双["桥"].eng.hdc.H)),
                            Φ_act=list(np.shape(双["桥"].eng.hdc.Φ_act)))
    except Exception as ex:                                    # noqa: BLE001
        果["C1"] = dict(异常=repr(ex))
        双 = None

    # ── C2 ──
    try:
        if 双 is None:
            双 = 贯m.造双区室(seed=310)
        # 主臂用**新造的桥**（H 从初态起，避免复用 C1 已跑 120 拍后的历史）；
        # 校准臂（D1）复用 C1 同一桥 ⇒ 与 D1 的"逐试次随机背景"性质一致（桥龄不构成信号）。
        t0 = time.perf_counter()
        双主 = 贯m.造双区室(seed=310)
        造桥秒2 = time.perf_counter() - t0
        果["C2"] = 臂C2(双, 双主)
        果["C2"]["主臂造桥秒"] = round(造桥秒2, 3)
        双 = 双主
    except Exception as ex:                                    # noqa: BLE001
        果["C2"] = dict(异常=repr(ex))

    # ── C3 ──
    try:
        if 双 is None:
            双 = 贯m.造双区室(seed=310)
        果["C3"] = 臂C3(双)
    except Exception as ex:                                    # noqa: BLE001
        果["C3"] = dict(异常=repr(ex))

    # ── C4 ──
    try:
        J实 = int(len(双["桥"].思核.竞争表)) if 双 is not None else 阈值["J"]
    except Exception:                                          # noqa: BLE001
        J实 = 阈值["J"]
    果["C4"] = 臂C4(阈值["J"], 阈值["D"])
    果["C4"]["实测J"] = J实
    if J实 != 阈值["J"]:
        果["C4"]["警告"] = "实测 J=%d ≠ 口径 J=%d ⇒ C4 表按口径 J=4 计，须换算" % (J实, 阈值["J"])

    果["总墙钟秒"] = round(time.perf_counter() - t全, 2)
    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump(果, f, ensure_ascii=False, indent=1, default=str)
    return 果


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if "--快检" in sys.argv:
        快检()
    else:
        g = 主()
        c2 = g.get("C2") or {}
        主臂 = c2.get("主") or {}
        print(json.dumps(dict(
            C1=dict(步调用=(g.get("C1") or {}).get("思核步调用数"),
                    迹=(g.get("C1") or {}).get("迹条数"),
                    造桥秒=(g.get("C1") or {}).get("造桥秒"),
                    跑拍秒=(g.get("C1") or {}).get("跑120拍秒")),
            C2=dict(异常=主臂.get("异常"), Acc原=主臂.get("Acc原"),
                    臂=[(r["臂"], r.get("TV均值"), r.get("TVD"), r.get("改变率"), r.get("Acc"),
                         r.get("Acc掉"), r.get("通过"), r.get("判定")) for r in (主臂.get("臂") or [])],
                    诊断=主臂.get("诊断"),
                    校准D1=dict(Acc原=(c2.get("仪器校准") or {}).get("Acc原"),
                              Acc_B=((c2.get("仪器校准") or {}).get("诊断") or {}).get("B单独解码Acc"))),
            C3=(g.get("C3") or {}).get("判定汇总"),
            C3对照=(g.get("C3") or {}).get("对照"),
            C4=(g.get("C4") or {}).get("基线"),
            墙钟=g.get("总墙钟秒"), 落盘=结果路径), ensure_ascii=False, indent=1, default=str))
