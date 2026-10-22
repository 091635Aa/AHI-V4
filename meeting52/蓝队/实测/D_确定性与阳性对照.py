# -*- coding: utf-8 -*-
"""D 组臂：三级指纹确定性（4 配置 × 2 趟，独立进程）+ 阳性对照 7 项 + 读序审计。

━━━ 纪律 ━━━
只读 importlib 加载 `算子_两区室核.py` / `算子_能力核.py`；不改任何既有件；不跑任何 `判据_*.py`；
`python -B`（不建 __pycache__）；本文件自带主程序 + subprocess worker 两种模式。

━━━ 用法 ━━━
    python -B D_确定性与阳性对照.py            # 主程序：起 8 个 D1 worker + 7 个对照 worker，落 D_结果.json
    python -B D_确定性与阳性对照.py --worker '{...}'   # 由主程序调用（单个臂、单进程）

━━━ 冻结阈值（预注册；跑后不改）━━━
  · 配置矩阵：PYTHONHASHSEED ∈ {0,12345} × OMP_NUM_THREADS ∈ {1,4}，每配置 2 趟独立进程 ⇒ 8 趟
  · 通过线   ：L1 ∧ L2 4/4 配置全同（且每配置两趟全同）；L3 仅记录（同机回归守卫）
  · 格断言   ：每趟 `x·2^20 == rint(x·2^20)` 全格；违例 > 0 ⇒ 该趟 INVALID
  · 模型/训/测种子：47001 / 47101 / 47102；慢率 0.20；D=256；训 400 + 测 400
  · 变体种子：47002（换种子对照）
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True

import numpy as np

# ══════════════════════════ 预注册阈值 / 常量（写死） ══════════════════════════
D默认 = 256
慢率默认 = 0.20
训长 = 400
测长 = 400
种子模型 = 47001
种子训 = 47101
种子测 = 47102
种子变体 = 47002          # 换种子对照
配置列 = ((0, 1), (0, 4), (12345, 1), (12345, 4))
每配置趟数 = 2
格步 = 2.0 ** -20         # 定点格步 = 1/2^20
亚格扰动 = 1e-9
通过线 = dict(配置数=len(配置列), 每配置趟数=每配置趟数,
              要求="L1 ∧ L2 全同（4 配置 × 2 趟）", 格违例上限=0)
未测清单 = [
    "跨硬件（第二台 CPU）逐位同：未测，不主张",
    "跨 numpy 版本 / 跨 BLAS 构建逐位同：未测",
    "L3 跨机比较：未测（本轮只作同机守卫）",
    "断点续跑 / 逐数组扰动之外的状态完整性（续跑 20 拍）：未做（D2③ 只做逐数组 1 格步）",
    "规则随机模式（_rng 非 None）的状态指纹：未测",
]

HERE = os.path.dirname(os.path.abspath(__file__))
AHI4 = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))       # …/AHI4
P两区室 = os.path.join(AHI4, "meeting49", "实施", "算子_两区室核.py")
P能力核 = os.path.join(AHI4, "meeting41", "实施", "算子_能力核.py")
落件 = os.path.join(HERE, "D_结果.json")

定点 = 1 << 20
标记 = "@@JSON@@"


# ══════════════════════════ 只读加载 ══════════════════════════
def 载模块(路径, 名):
    spec = importlib.util.spec_from_file_location(名, 路径)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha文件(路径):
    h = hashlib.sha256()
    with open(路径, "rb") as f:
        for 块 in iter(lambda: f.read(1 << 20), b""):
            h.update(块)
    return h.hexdigest()


# ══════════════════════════ 状态清单 / 三级指纹 ══════════════════════════
# 清单 = P,读出,感W,自我W,自我态,最近门,_最近B,_o,自我标量 + W表[8 算子]（共 17 个数组 / 20,993 元素）
状态名列 = ("P", "读出", "感W", "自我W", "自我态", "最近门", "_最近B", "_o", "自我标量")


def 取状态(模型):
    """按**冻结清单**取状态；W表 一律 sorted 名字序（与 dict 插入序无关）。"""
    状态 = {}
    for 名 in 状态名列:
        状态[名] = getattr(模型, 名)
    for 名 in sorted(模型.W表.keys()):                 # ← sorted：读序不依赖插入序
        状态["W表." + str(名)] = 模型.W表[名]
    return 状态


def 规范数组(v):
    a = np.asarray(v, dtype=np.float64)
    if a.ndim == 0:
        a = a.reshape(1)                               # 标量统一成 (1,)
    return np.ascontiguousarray(a)


def 三纹(状态):
    """L2 = sha256(按 sorted 名字序：名|dtype.str|shape|int64(格数 tobytes))；L3 = 同清单原始 float64 bytes。"""
    名序 = sorted(状态.keys())
    h2 = hashlib.sha256()
    h3 = hashlib.sha256()
    违例 = 0
    元素 = 0
    字节 = 0
    清单 = []
    for 名 in 名序:
        a = 规范数组(状态[名])
        形状 = "x".join(str(s) for s in a.shape)
        前缀 = (名 + "|" + a.dtype.str + "|" + 形状 + "|").encode("utf-8")
        格 = np.rint(a * 定点)
        违例 += int(np.sum((a * 定点) != 格))          # 格断言：x·2^20 == rint(x·2^20)
        h2.update(前缀)
        h2.update(格.astype(np.int64).tobytes())       # L2：int64 规范序列化
        h3.update(前缀)
        h3.update(a.tobytes())                         # L3：原始 float64 位型
        清单.append(dict(名=名, dtype=a.dtype.str, 形状=形状, 元素=int(a.size)))
        元素 += int(a.size)
        字节 += int(a.nbytes)
    return dict(L2=h2.hexdigest(), L3=h3.hexdigest(), 格违例=违例, 元素=元素, 字节=字节, 清单=清单)


def 决策指纹(训, 测):
    """L1 = sha256(训选‖训奖‖测选‖测奖) 决策 bytes（int64 规范序列化）。"""
    h = hashlib.sha256()
    for 标, a in (("训选", 训["选"]), ("训奖", 训["奖"]), ("测选", 测["选"]), ("测奖", 测["奖"])):
        h.update(标.encode("utf-8"))
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.int64)).tobytes())
    return h.hexdigest()


def 输出指纹(训, 测, 模型):
    """输出指纹（死系统臂用）：只看**模型自己的输出**（选序列 + 末拍输出），不含奖赏（奖赏来自输入标签）。"""
    h = hashlib.sha256()
    for 标, a in (("训选", 训["选"]), ("测选", 测["选"])):
        h.update(标.encode("utf-8"))
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.int64)).tobytes())
    o = 规范数组(模型._o)
    h.update("o末".encode("utf-8"))
    h.update(np.rint(o * 定点).astype(np.int64).tobytes())
    return h.hexdigest()


def 奖指纹(训, 测):
    h = hashlib.sha256()
    for 标, a in (("训奖", 训["奖"]), ("测奖", 测["奖"])):
        h.update(标.encode("utf-8"))
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.int64)).tobytes())
    return h.hexdigest()


# ══════════════════════════ 造模型 + 跑 ══════════════════════════
def 造跑(模型种子=种子模型, 注入=None, 序列种子=(种子训, 种子测)):
    """新建模块/序列/模型 → （可选注入）→ 训 400 + 测 400。返回 (M2, 模型, 训, 测, 元)。"""
    M2 = 载模块(P两区室, "两区室核_只读_D")
    序列训 = M2.生成序列(int(序列种子[0]), 训长, M2.边界训练)
    序列测 = M2.生成序列(int(序列种子[1]), 测长, M2.边界训练)
    模型 = M2.建模型("两区室", D默认, M2.投影P(D默认, int(模型种子)),
                    M2.读出初值(D默认, int(模型种子)),
                    种子=int(模型种子), 慢率=慢率默认, 学模式="credit")
    元 = dict(模型种子=int(模型种子), 序列种子=[int(序列种子[0]), int(序列种子[1])],
              慢率=慢率默认, D=D默认, 训长=训长, 测长=测长, 注入=注入 or "无")

    if 注入 == "float32":
        # ★ 量化前 float32 计算注入：替换模块级 `_归`（先 float32 计算、再量化），仅内存 monkeypatch
        基模块 = M2._m

        def _归_f32(v, 幅=16.0):
            a = np.asarray(v, dtype=np.float64)
            a = a.astype(np.float32).astype(np.float64)     # ★ 注入点：计算链里插一次 float32 往返
            a = np.nan_to_num(a, nan=0.0, posinf=幅, neginf=-幅)
            a = np.clip(a, -幅, 幅)
            return np.rint(a * 定点) / 定点

        M2._归 = _归_f32
        基模块._归 = _归_f32
        元["补丁"] = "算子_两区室核._归 + 算子_能力核._归（量化前 float32 计算注入，内存替换）"

    elif 注入 == "亚格":
        P0 = 模型.P.copy()
        未量化 = P0 * (1.0 + 亚格扰动)
        模型.P = M2._归(未量化)                       # 回格（量化器吸收）
        元["亚格_未量化位移max"] = float(np.max(np.abs(未量化 - P0)))
        元["亚格_未量化位型改变元素"] = int(np.sum(未量化 != P0))
        元["亚格_回格后改变元素"] = int(np.sum(模型.P != P0))
        模型._刷新自我态()

    训 = M2.跑(模型, 序列训, 学读出=True)
    测 = M2.跑(模型, 序列测, 学读出=False)
    return M2, 模型, 训, 测, 元


class 死系统核:
    """常量输出模型：输出与输入无关、学习也不动 ⇒ 若"两输入指纹不同"仍判它过，则该检测器失效。"""

    def __init__(self, D):
        self.D = int(D)
        o = np.full(self.D, 1.0 / np.sqrt(float(self.D)), dtype=np.float64)
        self.输出 = np.rint(o * 定点) / 定点
        self._o = self.输出.copy()
        # 固定读出（不学）：得分恒定 ⇒ 选序列恒定、与输入无关（真正的"不看输入"）
        r = np.zeros((2, self.D), dtype=np.float64)
        r[0, :] = 1.0 / np.sqrt(float(self.D))
        r[1, :] = -1.0 / np.sqrt(float(self.D))
        self.读出 = np.rint(r * 定点) / 定点

    def 前向_试次(self, x):
        self._o = self.输出.copy()
        return self._o

    def 学读出(self, ŷ, 奖, 更新读出=True):
        return None

    def 学自我(self, ŷ, 奖):
        return None


def 跑死系统(序列种子=(种子训, 种子测)):
    M2 = 载模块(P两区室, "两区室核_只读_D")
    序列训 = M2.生成序列(int(序列种子[0]), 训长, M2.边界训练)
    序列测 = M2.生成序列(int(序列种子[1]), 测长, M2.边界训练)
    模型 = 死系统核(D默认)
    训 = M2.跑(模型, 序列训, 学读出=True)
    测 = M2.跑(模型, 序列测, 学读出=False)
    元 = dict(模型种子="常量输出(无种子)", 序列种子=[int(序列种子[0]), int(序列种子[1])],
              慢率=慢率默认, D=D默认, 训长=训长, 测长=测长, 注入="死系统(常量输出)")
    return M2, 模型, 训, 测, 元


def 采集(M2, 模型, 训, 测, 元):
    状态 = 取状态(模型)
    纹 = 三纹(状态)
    out = dict(元)
    out.update(L1=决策指纹(训, 测), L2=纹["L2"], L3=纹["L3"],
               格违例=纹["格违例"], 元素=纹["元素"], 字节=纹["字节"], 清单=纹["清单"],
               清单sha256=hashlib.sha256(json.dumps(纹["清单"], ensure_ascii=False,
                                                   sort_keys=True).encode("utf-8")).hexdigest(),
               Acc_A=float(测["Acc_A"]), Acc_B=float(测["Acc_B"]), Acc=float(测["Acc"]))
    return out, 状态


def 基线参照():
    """同进程内再建一份干净的基线实例（独立模块实例），用于给对照臂算"决策差异元素数"。"""
    M2, 模型, 训, 测, 元 = 造跑(种子模型)
    out, _ = 采集(M2, 模型, 训, 测, 元)
    return out, 训, 测


def 补差异(out, 训, 测):
    """补：同进程基线三纹（用于 PC-8 逐位同 + 与独立进程基线互校）与决策/奖赏差异元素数。"""
    基out, 基训, 基测 = 基线参照()
    out["同进程基线"] = dict(L1=基out["L1"], L2=基out["L2"], L3=基out["L3"])
    out["选差异数"] = int(np.sum(np.asarray(基训["选"]) != np.asarray(训["选"]))
                        + np.sum(np.asarray(基测["选"]) != np.asarray(测["选"])))
    out["奖差异数"] = int(np.sum(np.asarray(基训["奖"]) != np.asarray(训["奖"]))
                        + np.sum(np.asarray(基测["奖"]) != np.asarray(测["奖"])))
    out["总决策数"] = int(len(训["选"]) + len(测["选"]))
    return out


# ══════════════════════════ 单个 worker（一臂一进程） ══════════════════════════
def 跑worker(规格):
    臂 = str(规格.get("臂", "D1"))
    起 = time.perf_counter()

    if 臂 == "D1":
        M2, 模型, 训, 测, 元 = 造跑(种子模型)
        out, _ = 采集(M2, 模型, 训, 测, 元)
        out.update(臂="D1", PYTHONHASHSEED=os.environ.get("PYTHONHASHSEED"),
                   OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"),
                   MKL_NUM_THREADS=os.environ.get("MKL_NUM_THREADS"),
                   OPENBLAS_NUM_THREADS=os.environ.get("OPENBLAS_NUM_THREADS"),
                   pid=os.getpid())

    elif 臂 == "换种子":
        M2, 模型, 训, 测, 元 = 造跑(种子变体)
        out, _ = 采集(M2, 模型, 训, 测, 元)
        补差异(out, 训, 测)
        out["臂"] = 臂

    elif 臂 == "urandom":
        s = int.from_bytes(os.urandom(4), "big")
        M2, 模型, 训, 测, 元 = 造跑(s)
        out, _ = 采集(M2, 模型, 训, 测, 元)
        补差异(out, 训, 测)
        out["臂"] = 臂
        out["urandom种子"] = int(s)
        out["随机源"] = "os.urandom(4)"

    elif 臂 == "float32":
        M2, 模型, 训, 测, 元 = 造跑(种子模型, 注入="float32")
        out, _ = 采集(M2, 模型, 训, 测, 元)
        补差异(out, 训, 测)
        out["臂"] = 臂

    elif 臂 == "亚格":
        M2, 模型, 训, 测, 元 = 造跑(种子模型, 注入="亚格")
        out, _ = 采集(M2, 模型, 训, 测, 元)
        补差异(out, 训, 测)
        out["臂"] = 臂

    elif 臂 == "格步":
        M2, 模型, 训, 测, 元 = 造跑(种子模型)
        out, 状态 = 采集(M2, 模型, 训, 测, 元)
        out["臂"] = 臂
        基 = out["L2"]
        逐数组 = {}
        for 名 in sorted(状态.keys()):
            副本 = {}
            for k, v in 状态.items():
                副本[k] = np.array(v, dtype=np.float64, copy=True)
            a = 副本[名]
            if a.ndim == 0:
                a = a.reshape(1)
                a.flat[0] = a.flat[0] + 格步
                副本[名] = a
            else:
                a.flat[0] = a.flat[0] + 格步
                副本[名] = a
            L2n = 三纹(副本)["L2"]
            逐数组[名] = dict(L2=L2n, 变=bool(L2n != 基))
        out["逐数组"] = 逐数组
        out["数组数"] = int(len(逐数组))
        out["漏项数组"] = sorted([k for k, v in 逐数组.items() if not v["变"]])
        out["全变"] = bool(all(v["变"] for v in 逐数组.values()))
        # 单格步（P[0,0]）也在其中：P 一行
        out["P单格步L2"] = 逐数组.get("P", {}).get("L2")

    elif 臂 == "ULP":
        M2, 模型, 训, 测, 元 = 造跑(种子模型)
        基out, 状态 = 采集(M2, 模型, 训, 测, 元)
        副本 = {}
        for k, v in 状态.items():
            副本[k] = np.array(v, dtype=np.float64, copy=True)
        a = 副本["读出"]
        a[0, 0] = float(np.nextafter(a[0, 0], np.inf))     # 1 ULP
        副本["读出"] = a
        纹n = 三纹(副本)
        out = dict(元)
        out.update(臂=臂, 读出错元素=0.0)
        out["基"] = dict(L1=基out["L1"], L2=基out["L2"], L3=基out["L3"], 格违例=基out["格违例"])
        out["注入后"] = dict(L2=纹n["L2"], L3=纹n["L3"], 格违例=纹n["格违例"])
        out["L2同"] = bool(纹n["L2"] == 基out["L2"])
        out["L3同"] = bool(纹n["L3"] == 基out["L3"])
        out["读出UU"] = [float(副本["读出"][0, 0]), float(状态["读出"][0, 0])]

    elif 臂 in ("两输入", "死系统"):
        输入列 = [(种子训, 种子测), (种子训 + 1000, 种子测 + 1000)]
        指纹 = []
        奖纹 = []
        accs = []
        for s1, s2 in 输入列:
            if 臂 == "两输入":
                M2, 模型, 训, 测, 元 = 造跑(种子模型, 序列种子=(s1, s2))
            else:
                M2, 模型, 训, 测, 元 = 跑死系统(序列种子=(s1, s2))
            指纹.append(输出指纹(训, 测, 模型))
            奖纹.append(奖指纹(训, 测))
            accs.append([float(测["Acc_A"]), float(测["Acc_B"]), float(测["Acc"])])
        out = dict(元)
        out.update(臂=臂, 输入列=[list(x) for x in 输入列], 输出指纹=指纹, 奖指纹=奖纹,
                   输出指纹不同=bool(指纹[0] != 指纹[1]), 奖指纹不同=bool(奖纹[0] != 奖纹[1]),
                   Acc列=accs)
    else:
        raise ValueError("未知臂：%r" % (臂,))

    out["臂内墙钟"] = round(time.perf_counter() - 起, 3)
    return out


# ══════════════════════════ 读序审计（静态） ══════════════════════════
def 读序审计():
    规则 = re.compile(r"^\s*(?:for|while)\s+(.+?)\s+in\s+(.+?)\s*:\s*$")
    危险词 = (".items(", ".keys(", ".values(", "set(", "dict(", "vars(", "{", "自W表", "W表[", ".__dict__")
    全部 = []
    危险 = []
    for 名, 路径 in (("算子_两区室核.py", P两区室), ("算子_能力核.py", P能力核)):
        with open(路径, "r", encoding="utf-8") as f:
            行列 = f.read().splitlines()
        for i, 行 in enumerate(行列, 1):
            m = 规则.match(行)
            if not m:
                continue
            项 = dict(文件=名, 行=i, 变量=m.group(1).strip(), 迭代体=m.group(2).strip(), 原文=行.strip())
            全部.append(项)
            if any(w in m.group(2) for w in 危险词):
                危险.append(项)
    人工结论 = (
        "8 算子数值链上**无 set/dict 遍历进入数值累积**："
        "前向/叠/读/迭代/符/世界态 的循环源全为 range(n)/list/ndarray，"
        "`叠` 用 int64 精确求和（与顺序无关）；"
        "`W表` 只在 `算子表()` 内以 sorted 遍历（且仅列名字、不进数值累积），其余为单键下标存取（W表[算子_读] 等）；"
        "唯一 dict 迭代在 `_取上下文` 的 4 键白名单（tuple 字面量定序）；"
        "`禁词自检/扫目录` 的 os.walk 已 sorted(文件列)，且属审计代码、不进决策链。"
        "判定：代码层无顺序依赖；HASHSEED 0 vs 12345 实测同 ⇒ 与静态审计一致（合规记录，不是检测器命中）。"
    )
    return dict(扫描文件=2, for行数=len(全部), 危险行=危险, 全部行=全部, 人工结论=人工结论,
                含sorted="`算子表()`/`sorted(文件列)`/`sorted(状态.keys())` 已在数值路径与指纹序列化上使用")


# ══════════════════════════ 主程序 ══════════════════════════
def 起worker(规格, hashseed, omp, 超时=900):
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(hashseed)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = str(omp)
    cmd = [sys.executable, "-B", os.path.abspath(__file__), "--worker",
           json.dumps(规格, ensure_ascii=False)]
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, env=env, timeout=超时)
    dt = time.perf_counter() - t
    so = p.stdout.decode("utf-8", errors="replace")
    se = p.stderr.decode("utf-8", errors="replace")
    行 = [x for x in so.splitlines() if x.startswith(标记)]
    if p.returncode != 0 or not 行:
        raise RuntimeError("worker 失败 rc=%r 规格=%r\nstdout=%s\nstderr=%s"
                           % (p.returncode, 规格, so[-2000:], se[-3000:]))
    r = json.loads(行[-1][len(标记):])
    r["墙钟含启动"] = round(dt, 3)
    r["rc"] = int(p.returncode)
    return r


def 环境快照():
    快 = dict(系统=platform.platform(), 机器=platform.machine(), 处理器=platform.processor(),
              Python=sys.version.replace("\n", " "), numpy=np.__version__,
              逻辑核=os.cpu_count(), 工作目录=os.getcwd(), 脚本sha256=sha文件(os.path.abspath(__file__)),
              算子_两区室核_sha256=sha文件(P两区室), 算子_能力核_sha256=sha文件(P能力核),
              配置矩阵=[list(x) for x in 配置列], 通过线=通过线,
              采样时间=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    try:
        cfg = np.show_config(mode="dicts")
        快["BLAS"] = json.loads(json.dumps(cfg.get("Build Dependencies", {}).get("blas", {}), default=str))
    except Exception as e:
        快["BLAS"] = "取配置失败：%s" % (e,)
    return 快


def main():
    t总 = time.perf_counter()
    元信息 = 环境快照()

    # ───────── D1：4 配置 × 2 趟，独立进程 ─────────
    D1行 = []
    D1全 = []
    for hs, omp in 配置列:
        for 趟 in range(1, 每配置趟数 + 1):
            r = 起worker(dict(臂="D1"), hs, omp)
            D1全.append(r)
            D1行.append(dict(配置="H%d_T%d" % (hs, omp), hashseed=hs, omp=omp, 趟=趟,
                             L1=r["L1"][:16], L2=r["L2"][:16], L3=r["L3"][:16],
                             格违例=r["格违例"], 元素=r["元素"], 字节=r["字节"],
                             Acc_A=round(r["Acc_A"], 7), Acc_B=round(r["Acc_B"], 7),
                             墙钟含启动=r["墙钟含启动"], 臂内墙钟=r["臂内墙钟"],
                             pid=r.get("pid"), 环境回读=[r.get("PYTHONHASHSEED"), r.get("OMP_NUM_THREADS"),
                                                   r.get("MKL_NUM_THREADS"), r.get("OPENBLAS_NUM_THREADS")]))

    L1集 = sorted({r["L1"] for r in D1全})
    L2集 = sorted({r["L2"] for r in D1全})
    L3集 = sorted({r["L3"] for r in D1全})
    配置级 = {}
    for hs, omp in 配置列:
        子 = [r for r in D1全 if r.get("PYTHONHASHSEED") == str(hs) and r.get("OMP_NUM_THREADS") == str(omp)]
        配置级["H%d_T%d" % (hs, omp)] = dict(
            趟数=len(子), L1同=len({x["L1"] for x in 子}) == 1,
            L2同=len({x["L2"] for x in 子}) == 1, L3同=len({x["L3"] for x in 子}) == 1)
    格违例总 = int(sum(r["格违例"] for r in D1全))
    D1通过 = bool(len(L1集) == 1 and len(L2集) == 1 and len(L3集) == 1 and 格违例总 == 0
                and all(v["L1同"] and v["L2同"] and v["L3同"] for v in 配置级.values()))
    基线 = D1全[0]

    # ───────── D2：阳性对照 7 项（每项独立进程） ─────────
    对 = {}
    对["①换种子"] = 起worker(dict(臂="换种子"), 0, 1)
    对["②urandom"] = 起worker(dict(臂="urandom"), 0, 1)
    对["③1格步"] = 起worker(dict(臂="格步"), 0, 1)
    对["④量化前float32"] = 起worker(dict(臂="float32"), 0, 1)
    对["⑤亚格1e-9"] = 起worker(dict(臂="亚格"), 0, 1)
    对["⑥死系统"] = dict(健康=起worker(dict(臂="两输入"), 0, 1), 死=起worker(dict(臂="死系统"), 0, 1))
    对["⑦1ULP"] = 起worker(dict(臂="ULP"), 0, 1)

    def 差(r):
        基 = r.get("同进程基线", {}) or {}
        return dict(L1不同=bool(r["L1"] != 基线["L1"]), L2不同=bool(r["L2"] != 基线["L2"]),
                    L3不同=bool(r["L3"] != 基线["L3"]), 格违例=r["格违例"],
                    选差异数=r.get("选差异数"), 奖差异数=r.get("奖差异数"),
                    总决策数=r.get("总决策数"),
                    同进程基线逐位同=bool(基 and 基.get("L1") == 基线["L1"]
                                    and 基.get("L2") == 基线["L2"] and 基.get("L3") == 基线["L3"]),
                    Acc_A=round(r["Acc_A"], 7), Acc_B=round(r["Acc_B"], 7),
                    墙钟含启动=r["墙钟含启动"])

    D2 = []
    r1 = 对["①换种子"]
    d1 = 差(r1)
    D2.append(dict(项="①换种子 47001→47002", 期望="三纹必不同",
                   实测=d1, 通过=bool(d1["L1不同"] and d1["L2不同"] and d1["L3不同"]), 必过=True))

    r2 = 对["②urandom"]
    d2 = 差(r2)
    d2["种子"] = r2.get("urandom种子")
    D2.append(dict(项="②os.urandom 派生种子", 期望="三纹必不同",
                   实测=d2, 通过=bool(d2["L1不同"] and d2["L2不同"] and d2["L3不同"]), 必过=True))

    r3 = 对["③1格步"]
    逐 = r3["逐数组"]
    d3 = dict(数组数=r3["数组数"], 全变=r3["全变"], 漏项数组=r3["漏项数组"],
              P单格步L2变=bool(r3.get("P单格步L2") != 基线["L2"]),
              格违例=r3["格违例"], 墙钟含启动=r3["墙钟含启动"],
              逐数组={k: bool(v["变"]) for k, v in 逐.items()})
    D2.append(dict(项="③1 格步扰动（+2^-20）· 逐数组全覆盖", 期望="L2 必不同（每个数组都要能改变 L2）",
                   实测=d3, 通过=bool(r3["全变"] and len(r3["漏项数组"]) == 0), 必过=True))

    r4 = 对["④量化前float32"]
    d4 = 差(r4)
    d4["补丁"] = r4.get("补丁")
    D2.append(dict(项="④量化前 float32 计算注入", 期望="L2/L3 必不同；L1 允许同",
                   实测=d4, 通过=bool(d4["L2不同"] and d4["L3不同"]), 必过=True))

    r5 = 对["⑤亚格1e-9"]
    d5 = 差(r5)
    d5.update(未量化位移max=r5.get("亚格_未量化位移max"),
              未量化位型改变元素=r5.get("亚格_未量化位型改变元素"),
              回格后改变元素=r5.get("亚格_回格后改变元素"))
    D2.append(dict(项="⑤亚格 1e-9 扰动（应被量化吸收）", 期望="三纹必同",
                   实测=d5, 通过=bool(not d5["L1不同"] and not d5["L2不同"] and not d5["L3不同"]), 必过=True))

    r6 = 对["⑥死系统"]
    健 = r6["健康"]
    死 = r6["死"]
    d6 = dict(健康_输出指纹不同=健["输出指纹不同"], 健康_奖指纹不同=健["奖指纹不同"],
              健康_Acc=[round(x, 7) for x in 健["Acc列"][0]],
              死_输出指纹不同=死["输出指纹不同"], 死_奖指纹不同=死["奖指纹不同"],
              死_Acc=[round(x, 7) for x in 死["Acc列"][0]],
              说明="输出指纹=选序列+末拍输出（不含奖赏，奖赏来自输入标签）；健康臂必不同、常量模型必同 ⇒ 检测器能抓死系统",
              墙钟=[健["墙钟含启动"], 死["墙钟含启动"]])
    D2.append(dict(项="⑥死系统臂（常量输出模型）", 期望="健康臂两输入输出指纹必不同；常量模型必同（被抓）",
                   实测=d6, 通过=bool(健["输出指纹不同"] and not 死["输出指纹不同"]), 必过=True))

    r7 = 对["⑦1ULP"]
    d7 = dict(L2同=r7["L2同"], L3同=r7["L3同"], 格违例_注入后=r7["注入后"]["格违例"],
              说明="1 ULP 注入使该元素离开 2^-20 格（格违例=1，属注入本身、非数值策略破坏）；rint 后 L2 同、原始位型 L3 不同",
              墙钟含启动=r7["墙钟含启动"])
    D2.append(dict(项="⑦1 ULP 扰动", 期望="L3 不同、L2 同（记录项）",
                   实测=d7, 通过=bool(d7["L2同"] and (not d7["L3同"])),
                   必过=False))

    D2必过 = [x for x in D2 if x["必过"]]
    D2通过 = bool(all(x["通过"] for x in D2必过))
    # 补充（PC-8）：各对照臂**同进程内重建的基线实例**必须与独立进程基线逐位同（排除隐藏态/缓存污染）
    重建 = [差(对[k]) for k in ("①换种子", "②urandom", "④量化前float32", "⑤亚格1e-9")]
    D2补充 = dict(同进程重建基线逐位同=bool(all(y["同进程基线逐位同"] for y in 重建)),
                  项数=len(重建),
                  说明="PC-8 同配置重复：同进程重建 + 跨 8 个独立进程（D1）三纹均逐位同 ⇒ 无隐藏态/缓存污染")

    # ───────── D3：读序审计 ─────────
    审 = 读序审计()
    hs0 = [r for r in D1全 if r.get("PYTHONHASHSEED") == "0"]
    hs1 = [r for r in D1全 if r.get("PYTHONHASHSEED") == "12345"]
    D3 = dict(静态=dict(扫描文件=审["扫描文件"], for行数=审["for行数"], 危险行=审["危险行"],
                       人工结论=审["人工结论"], sorted使用=审["含sorted"]),
              动态=dict(HASHSEED0指纹=[hs0[0]["L1"][:16], hs0[0]["L2"][:16]] if hs0 else None,
                        HASHSEED12345指纹=[hs1[0]["L1"][:16], hs1[0]["L2"][:16]] if hs1 else None,
                        L1同=bool(len({r["L1"] for r in hs0 + hs1}) == 1),
                        L2同=bool(len({r["L2"] for r in hs0 + hs1}) == 1)),
              结论="动态同 + 静态无危险遍历 ⇒ 读序合规（记录项，按修正 C 不要求'必须不同'）",
              通过=bool(len({r["L1"] for r in hs0 + hs1}) == 1 and len({r["L2"] for r in hs0 + hs1}) == 1)
                   and len(审["危险行"]) == 0)

    # ───────── 汇总 ─────────
    墙钟 = dict(D1八趟含启动秒=round(sum(r["墙钟含启动"] for r in D1全), 2),
                D1单趟臂内秒=[r["臂内墙钟"] for r in D1全],
                对照含启动秒=round(sum(v["墙钟含启动"] for k, v in 对.items() if k != "⑥死系统"), 2),
                对照死系统两项=round(对["⑥死系统"]["健康"]["墙钟含启动"] + 对["⑥死系统"]["死"]["墙钟含启动"], 2),
                总墙钟秒=round(time.perf_counter() - t总, 2))

    判定 = "PASS" if (D1通过 and D2通过 and D3["通过"]) else ("INVALID" if not D2通过 else "FAIL")

    结果 = dict(臂="D · 三级指纹确定性 + 阳性对照 + 读序审计", 判定=判定,
                D1=dict(通过=D1通过, 通过线=通过线, 全同=dict(L1=len(L1集) == 1, L2=len(L2集) == 1,
                                                          L3=len(L3集) == 1),
                        L1=基线["L1"], L2=基线["L2"], L3=基线["L3"],
                        不同L1数=len(L1集), 不同L2数=len(L2集), 不同L3数=len(L3集),
                        格违例总=格违例总, 元素=基线["元素"], 字节=基线["字节"],
                        清单sha256=基线["清单sha256"], 清单=基线["清单"],
                        配置级=配置级, 行=D1行),
                D2=dict(通过=D2通过, 必过项数=len(D2必过), 项=D2, 补充=D2补充),
                D3=D3, 环境=元信息, 墙钟=墙钟, 未测=未测清单,
                落件=落件, 生成时间=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    with open(落件, "w", encoding="utf-8") as f:
        json.dump(结果, f, ensure_ascii=False, indent=1, default=str)

    print(json.dumps(dict(判定=判定, D1通过=D1通过, D2通过=D2通过, D3通过=D3["通过"],
                          L1=基线["L1"][:16], L2=基线["L2"][:16], L3=基线["L3"][:16],
                          格违例总=格违例总, 墙钟=墙钟, 落件=落件),
                     ensure_ascii=False, indent=1))
    return 结果


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        np.seterr(all="ignore")
    except Exception:
        pass
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        规格 = json.loads(sys.argv[2])
        果 = 跑worker(规格)
        print(标记 + json.dumps(果, ensure_ascii=False, sort_keys=True, default=str))
    else:
        main()
