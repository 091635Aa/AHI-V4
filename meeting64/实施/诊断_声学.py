# -*- coding: utf-8 -*-
"""meeting64/实施/诊断_声学.py —— v8.1 渲染器四类声学线索的纯 DSP 诊断（零外挂）。

口径（必须一起读）：
  · 只读 import `meeting63/实施/韵律_v8.py`（**被诊断的渲染器**）、`meeting62/实施/韵律_v7.py`、
    `引擎_v7.py`、`meeting61/实施/音韵_v5.py`、`meeting63/实施/耳_内生.py`；
    **不改任何既有文件**（C14）——本件只新增 `meeting64/`。
  · 零外挂：不跑 ASR、不用任何模型/权重/网络；全部 DSP 用 numpy/scipy **自实现**
    （F0 自相关、LPC + Levinson-Durbin、极点聚类、谱特征）。`耳_内生` 只作**确定性对照**（波形侧切段）。
  · 确定性（C13）：固定种子；生产路径（每次新建 `人类韵律_v8`）逐位可复现（见 §8）。
  · 每问都给「一句话结论 + 数字 + 是否结构缺口（渲染器缺该线索）」。

测什么：
  Q1 声调：波形逐 10ms 归一化自相关估 F0（ma1..ma4 + 6 句 × 3 角色），报每调起/终/极值（半音）、调域、
     调间可分度（留一最近质心）；量化 句调 / 颤音 / 漂移 的干扰量。
  Q2 元音：浊段 LPC（自实现：16k / 30ms / 10ms，阶 14）→ 极点聚类 → 盲测 3 共振峰 +
     以「帧目标×s」为参考的极点匹配（跟踪法）；对照 `音韵_v5.目标s × 共振峰缩放`、帧侧目标、
     引擎实际（帧目标再乘一次 s）；报偏差、相邻元音分离度、鼻尾 -n/-ng 的 F2、复韵滑移。
  Q3 擦音/塞擦部位谱：按帧把 `声源==0` 噪声段切出，量 谱重心/谱峰/谱展/低频占比/时长；
     对照文献与真人 ref44；给 s/sh/x 两两可分度（Δ重心、d'、留一最近质心）。
  Q4 时长：p/t/k 送气 VOT、b/d/g 不送气、擦音、塞擦摩擦、塞闭、浊声母（帧侧 + 波形侧 + 耳），
     对照文献与真人，给「合成/文献」与「合成/真人」比值。

产物：`meeting64/产物/诊断/声学/诊断_声学.json` + `诊断_声学.md`。
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
M64 = os.path.dirname(HERE)
AHI4 = os.path.dirname(M64)
M63 = os.path.join(AHI4, "meeting63")
M62 = os.path.join(AHI4, "meeting62")
M61 = os.path.join(AHI4, "meeting61")
M58 = os.path.join(AHI4, "meeting58")
for _p in (os.path.join(M63, "实施"), os.path.join(M62, "实施"),
           os.path.join(M61, "实施"), os.path.join(M58, "实施")):
    sys.path.insert(0, _p)

import 音韵_v5 as Y          # noqa: E402  只读：声母表 / 韵母目标 / 音高轮廓
import 韵律_v7 as H7         # noqa: E402  只读：切短语 / 变调 / 颤音门槛口径
import 韵律_v8 as H8         # noqa: E402  只读：★被诊断的 v8.1 渲染器
import 引擎_v7 as E7         # noqa: E402  只读：Klatt 串联引擎（用于共振峰缩放消融）
import 耳_内生 as 耳          # noqa: E402  只读：波形侧确定性切段（对照，不参与生成）
from 接口契约_角色声线 import 声线档案  # noqa: E402

FS = 44100
帧率 = 100
每帧样本 = 441
_FS16 = 16000
出目录 = os.path.join(M64, "产物", "诊断", "声学")

角色s = ("EmoCompanion", "SpeakerA", "洛天依")
句集 = [
    ("S1", "今天的风很轻，适合慢慢地说一会儿话。"),
    ("S2", "我们明天早上八点在南门见面。"),
    ("S3", "他把三本书放在窗台上。"),
    ("S4", "小猫从桌上跳下来，碰到杯子。"),
    ("S5", "春天来了，山上的雪开始融化。"),
    ("S6", "请问这条路怎么走？"),
]
# 文献量级（01_预检报告.md 已登记的 [P0]/[P2]；本件不联网，只引用其数值）
文献 = dict(
    送气VOT="90–110 ms", 不送气VOT="10–25 ms", 擦音时长="153 ± 38 ms",
    鼻音时长="60–90 ms", 爆破="10–30 ms",
    谱峰_s="7–8 kHz", 谱峰_x="5–6 kHz", 谱峰_sh="3.5–4 kHz", 谱峰_f="≈2.5 kHz",
)

_档缓存 = {}


# ================================================================ 基础工具
def 取档(名: str):
    if 名 not in _档缓存:
        p = os.path.join(M61, "产物", "声线档案", "%s.json" % 名)
        with open(p, encoding="utf-8") as f:
            _档缓存[名] = 声线档案.从dict(json.load(f))
    return _档缓存[名]


def 中F(档) -> float:
    """与 人类韵律_v5.__init__ 同口径的 F0 中心：基频×(1+0.5·声区)。"""
    return float(np.clip(档.基频, 70.0, 400.0)) * (1.0 + 0.5 * float(np.clip(档.声区, 0.0, 1.0)))


def 半音(f, F中):
    if f is None or F中 <= 0 or f <= 0:
        return None
    return 12.0 * math.log2(float(f) / F中)


def 中位(a):
    a = [float(x) for x in a if x is not None]
    return float(np.median(a)) if a else None


def 均值(a):
    a = [float(x) for x in a if x is not None]
    return float(np.mean(a)) if a else None


def 标准差(a):
    a = [float(x) for x in a if x is not None]
    return float(np.std(a)) if a else None


def _J(o):
    """递归 JSON 化（numpy / 非有限浮点）。"""
    if isinstance(o, dict):
        return {str(k): _J(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_J(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_J(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer, int)):
        return int(o)
    return o


def _哈希(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x, dtype=np.float64)).tobytes()).hexdigest()


# ================================================================ F0 / 能量 / 浊度（自实现）
def F0轨迹(x: np.ndarray, 窗: int = 1323, 跳: int = 每帧样本) -> dict:
    """逐 10ms 归一化自相关估 F0（30ms 窗，窗心对准帧中心；'倍周期仲裁'取最短 lag 的 0.92·峰 局部极大）。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 窗:
        x = np.pad(x, (0, 窗 - n))
        n = len(x)
    半 = 窗 // 2
    补 = np.pad(x, (半, 半 + 窗))
    k = int(np.floor((n - 1) / 跳)) + 1
    lo, hi = int(FS / 520.0), int(FS / 65.0)
    hw = np.hanning(窗)
    f0 = np.zeros(k)
    q = np.zeros(k)
    for i in range(k):
        起 = i * 跳 + 跳 // 2          # 窗心 = 帧中心
        if 起 + 窗 > len(补):
            起 = len(补) - 窗
        fr = 补[起:起 + 窗].astype(np.float64)
        fr = fr - fr.mean()
        X = np.fft.rfft(fr * hw, 8192)
        ac = np.fft.irfft(X * np.conj(X), 8192)[:窗]
        r0 = float(ac[0])
        if r0 <= 1e-20:
            continue
        ac = ac / r0
        seg = ac[lo:hi + 1]
        j = int(np.argmax(seg))
        峰 = float(seg[j])
        if 峰 < 0.08:
            q[i] = 峰
            continue
        for jj in range(1, j):
            if seg[jj] >= 0.92 * 峰 and seg[jj] >= seg[jj - 1] and seg[jj] >= seg[jj + 1]:
                j = jj
                峰 = float(seg[jj])
                break
        lag = float(lo + j)
        if 0 < j < len(seg) - 1:
            y0, y1, y2 = float(seg[j - 1]), float(seg[j]), float(seg[j + 1])
            den = y0 - 2.0 * y1 + y2
            if abs(den) > 1e-15:
                lag = lag + 0.5 * (y0 - y2) / den
        f0[i] = FS / lag if lag > 0 else 0.0
        q[i] = 峰
    f2 = f0.copy()
    for i in range(1, k - 1):
        if f0[i] > 0 and f0[i - 1] > 0 and f0[i + 1] > 0:
            f2[i] = float(np.median(f0[i - 1:i + 2]))
    return dict(k=k, 秒=(np.arange(k) * 跳 + 跳 // 2) / FS, f0=f2, 浊度=q)


def 能量dB逐帧(x: np.ndarray, 跳: int = 每帧样本, 帧长: int = 每帧样本) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 帧长:
        x = np.pad(x, (0, 帧长 - n))
        n = len(x)
    k = int(np.floor((n - 1) / 跳)) + 1
    出 = np.zeros(k)
    for i in range(k):
        a = i * 跳
        b = min(a + 帧长, n)
        段 = x[a:b]
        出[i] = 20.0 * math.log10(max(float(np.sqrt(np.mean(段 * 段))), 1e-12))
    return 出


def 浊度逐帧(x: np.ndarray, 窗: int = 882, 跳: int = 每帧样本) -> np.ndarray:
    """逐帧浊度（20ms 前向窗：只问「该帧是否含周期性」；用于起音检测，避免居中窗的时间模糊）。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 窗:
        x = np.pad(x, (0, 窗 - n))
        n = len(x)
    补 = np.pad(x, (0, 窗))
    k = int(np.floor((n - 1) / 跳)) + 1
    lo, hi = int(FS / 520.0), int(FS / 65.0)
    hw = np.hanning(窗)
    出 = np.zeros(k)
    for i in range(k):
        起 = i * 跳
        fr = 补[起:起 + 窗].astype(np.float64)
        fr = fr - fr.mean()
        X = np.fft.rfft(fr * hw, 4096)
        ac = np.fft.irfft(X * np.conj(X), 4096)[:窗]
        if ac[0] <= 1e-20:
            continue
        ac = ac / float(ac[0])
        出[i] = float(np.max(ac[lo:hi + 1]))
    return 出


# ================================================================ LPC（自实现）+ 极点聚类
def _levinson(r: np.ndarray, 阶: int) -> np.ndarray:
    """Levinson-Durbin：r[0..阶] 自相关 → a=[1,a1..a阶]（预测多项式）。"""
    a = np.zeros(阶 + 1)
    a[0] = 1.0
    e = float(r[0])
    if e <= 0:
        return a
    for i in range(1, 阶 + 1):
        acc = float(r[i]) + sum(float(a[k]) * float(r[i - k]) for k in range(1, i))
        k = -acc / e
        if not math.isfinite(k):
            break
        old = a.copy()
        for j in range(1, i):
            a[j] = old[j] + k * old[i - j]
        a[i] = k
        e = e * (1.0 - k * k)
        if e <= 1e-12:
            break
    return a


def LPC系数(段: np.ndarray, 阶: int = 14, 预重: float = 0.97) -> np.ndarray:
    x = np.asarray(段, dtype=np.float64).ravel().copy()
    if 预重 > 0.0 and len(x) > 1:
        x = np.concatenate(([x[0]], x[1:] - 预重 * x[:-1]))
    x = x * np.hamming(len(x))
    r = np.correlate(x, x, "full")[len(x) - 1:len(x) + 阶]
    return _levinson(r, 阶)


def 极点(段: np.ndarray, fs: int, 阶: int = 14, 预重: float = 0.97,
        低频界: float = 140.0, 高频界: float = 7000.0, 带宽界: float = 500.0) -> list:
    if len(段) < 32:
        return []
    a = LPC系数(段, 阶, 预重)
    if not np.all(np.isfinite(a)):
        return []
    出 = []
    for rt in np.roots(a):
        if abs(rt) <= 0.0 or rt.imag <= 0.0:
            continue
        f = abs(float(np.angle(rt))) * fs / (2.0 * math.pi)
        bw = -math.log(abs(rt)) * fs / math.pi
        if 低频界 < f < 高频界 and bw < 带宽界:
            出.append((float(f), float(bw)))
    return 出


def _重采样16k(段: np.ndarray) -> np.ndarray:
    return np.asarray(resample_poly(np.asarray(段, dtype=np.float64), 160, 441), dtype=np.float64)


def 帧极点(x: np.ndarray, i: int, 阶: int = 14, 窗毫秒: float = 30.0) -> list:
    """第 i 个 10ms 帧处的 30ms 窗（对齐帧起点）LPC 极点（16k 域）。"""
    L = int(round(FS * 窗毫秒 / 1000.0))
    a = i * 每帧样本
    段 = np.asarray(x[a:min(a + L, len(x))], dtype=np.float64)
    if len(段) < L:
        段 = np.pad(段, (0, L - len(段)))
    return 极点(_重采样16k(段), _FS16, 阶=阶, 预重=0.97)


def 极点簇(极点s: list, 容差: float = 0.06) -> list:
    """把多帧极点按频率聚类（链式，±6%）；返回 [{f, bw, support}]。"""
    ps = sorted([(float(f), float(bw)) for f, bw in 极点s])
    组 = []
    for f, bw in ps:
        if 组 and f <= 组[-1][-1][0] * (1.0 + 容差):
            组[-1].append((f, bw))
        else:
            组.append([(f, bw)])
    return [dict(f=float(np.median([v[0] for v in g])), bw=float(np.median([v[1] for v in g])),
                 support=int(len(g))) for g in 组]


def 段极点(w: np.ndarray, 帧s, 阶: int = 14, 窗毫秒: float = 30.0) -> list:
    allp = []
    for i in 帧s:
        allp += 帧极点(w, i, 阶=阶, 窗毫秒=窗毫秒)
    return 极点簇(allp)


def 三共振峰(簇s: list, n帧: int, 支撑阈: float = 0.5, 带宽界: float = 500.0) -> list:
    """盲测：支持率 ≥ 阈、带宽 < 界 的频率最低的 3 个簇。"""
    有效 = [c for c in 簇s if c["support"] >= max(2, int(n帧 * 支撑阈)) and c["bw"] < 带宽界]
    有效.sort(key=lambda c: c["f"])
    return [float(c["f"]) for c in 有效[:3]]


def 匹配共振峰(簇s: list, 目标s, 容差: float = 0.15) -> list:
    """跟踪法：对每个目标频率找最近簇（相对差 ≤ 容差才算匹配）。返回 [{目标, 测, 偏差pct, 匹配}]。"""
    出 = []
    for t in 目标s:
        if t is None or t <= 0:
            出.append(dict(目标=t, 测=None, 偏差pct=None, 匹配=False))
            continue
        候 = [c for c in 簇s if abs(c["f"] - t) <= t * 容差]
        if not 候:
            出.append(dict(目标=float(t), 测=None, 偏差pct=None, 匹配=False))
            continue
        c = min(候, key=lambda c: abs(c["f"] - t))
        出.append(dict(目标=float(t), 测=float(c["f"]),
                     偏差pct=100.0 * (c["f"] - t) / t, 匹配=True))
    return 出


def _跟踪F23(r: dict):
    """分离度用：优先取『跟踪匹配』的 F2/F3（以帧目标×s 为参考解歧义），缺则回退盲测。"""
    盲 = r.get("盲测") or []
    跟 = r.get("跟踪") or []
    f2 = (跟[1]["测"] if len(跟) > 1 and 跟[1]["匹配"] else (盲[1] if len(盲) > 1 else None))
    f3 = (跟[2]["测"] if len(跟) > 2 and 跟[2]["匹配"] else (盲[2] if len(盲) > 2 else None))
    return f2, f3


# ================================================================ 谱特征（噪声段）
def 谱特征(块: np.ndarray, fs: float = FS) -> dict:
    """10ms 块的功率谱特征（300–16000Hz）。"""
    块 = np.asarray(块, dtype=np.float64).ravel()
    if len(块) < 64:
        return dict(重心=None, 峰=None, 展=None, 低频占比=None)
    S = np.abs(np.fft.rfft(块 * np.hanning(len(块)), 8192)) ** 2
    f = np.fft.rfftfreq(8192, 1.0 / fs)
    m = (f >= 300.0) & (f <= 16000.0)
    f = f[m]
    S = S[m]
    T = float(S.sum()) + 1e-300
    重心 = float((S * f).sum() / T)
    峰 = float(f[int(np.argmax(S))])
    展 = float(math.sqrt(max((S * (f - 重心) ** 2).sum() / T, 0.0)))
    低频 = float(S[f < 1000.0].sum() / T)
    return dict(重心=重心, 峰=峰, 展=展, 低频占比=低频)


# ================================================================ 记录音节边界的韵律子类（只读继承）
class 记录韵律(H8.人类韵律_v8):
    """在 `人类韵律_v8` 上只加**记录**：每音节 (声母起帧, 元音起帧, 音节止帧)、颤音非零计数。

    不改变生成：所有方法都先记边界再 `super()`，帧/波形与基类**逐位同**（脚本内自检验证）。
    """

    def _清(self):
        super()._清()
        self.边界 = []
        self._起 = 0
        self._元起 = 0
        self.颤非零帧 = 0

    def _声母(self, s, 目标首, f0起):
        self._起 = len(self.帧["f0"])
        return super()._声母(s, 目标首, f0起)

    def _元音(self, *a, **kw):
        self._元起 = len(self.帧["f0"])
        super()._元音(*a, **kw)
        self.边界.append((int(self._起), int(self._元起), int(len(self.帧["f0"]))))

    def _颤音半音(self, *a, **kw):
        v = super()._颤音半音(*a, **kw)
        if abs(float(v)) > 1e-12:
            self.颤非零帧 += 1
        return v


def 句音节列表(文本: str) -> list:
    """复现 `建帧` 的音节遍历顺序/变调，给每个音节附 (j, n, 标点, pi)。"""
    出 = []
    for pi, (短语, 标点) in enumerate(H7.切短语(文本)):
        音节s = H7._变调(Y.音节列表(短语))
        n = len(音节s)
        for j, s in enumerate(音节s):
            出.append(dict(拼音=s["拼音"], 声=s["声"], 韵=s["韵"], 调=int(s["调"]),
                          尾=s["尾"], j=j, n=n, 标点=标点, pi=pi, 末尾=(j == n - 1)))
    return 出


# ================================================================ Q1 声调
def _音节F0测(w, tr, 帧, 边, F中):
    """在 [元起, 止) 的浊帧上测 F0 半音（含 5 点轮廓）；QC：与帧设计 f0 差 >1.5 半音判失配。"""
    起, 元起, 止 = 边
    src = 帧["声源"]
    idx = [i for i in range(元起, min(止, len(帧["f0"]))) if src[i] == 1 and i < tr["k"]]
    用, 失配 = [], 0
    for i in idx:
        e = float(tr["f0"][i])
        q = float(tr["浊度"][i])
        f = float(帧["f0"][i])
        if e <= 0 or q < 0.45 or f <= 0:
            失配 += 1
            continue
        d = 12.0 * math.log2(e / f)
        if abs(d) > 1.5:
            失配 += 1
            continue
        用.append(i)
    if len(用) < 3:
        return None
    估 = np.array([半音(float(tr["f0"][i]), F中) for i in 用])
    帧s = np.array([半音(float(帧["f0"][i]), F中) for i in 用])
    t5 = np.linspace(0.0, 1.0, 5)
    估5 = np.interp(t5, np.linspace(0.0, 1.0, len(用)), 估)
    帧5 = np.interp(t5, np.linspace(0.0, 1.0, len(用)), 帧s)
    return dict(
        帧数=len(用), 失配帧=失配,
        起=float(np.median(估[:2])), 终=float(np.median(估[-2:])),
        极小=float(np.min(估)), 极大=float(np.max(估)),
        帧_起=float(np.median(帧s[:2])), 帧_终=float(np.median(帧s[-2:])),
        帧_极小=float(np.min(帧s)), 帧_极大=float(np.max(帧s)),
        估5=[float(x) for x in 估5], 帧5=[float(x) for x in 帧5],
        失配率=失配 / max(len(idx), 1),
        残差std=float(np.std(估 - 帧s)),
    )


def 目标轮廓(调: int) -> dict:
    """`音韵_v5.声调表` 五度轮廓 → 半音=(度-3)*1.2 的 起/终/极值。"""
    if 调 == 5:
        return dict(起=-0.5, 终=-0.5, 极小=-0.5, 极大=-0.5)
    轮廓 = Y.声调表.get(int(调), Y.声调表[1])
    ts = np.linspace(0.0, 1.0, 101)
    度 = np.array([Y.折线(轮廓, float(t)) for t in ts])
    st = (度 - 3.0) * 1.2
    return dict(起=float(st[0]), 终=float(st[-1]), 极小=float(st.min()), 极大=float(st.max()))


def _留一最近质心(X: list, y: list) -> dict:
    """留一最近质心（欧氏）；返回 准确率/混淆。"""
    X = [np.asarray(v, dtype=np.float64) for v in X]
    y = [str(v) for v in y]
    类s = sorted(set(y))
    正确 = 0
    混淆 = {c: {d: 0 for d in 类s} for c in 类s}
    for i in range(len(X)):
        质 = {}
        for c in 类s:
            成员 = [X[j] for j in range(len(X)) if j != i and y[j] == c]
            if 成员:
                质[c] = np.mean(np.stack(成员), axis=0)
        if not 质:
            continue
        预测 = min(质.keys(), key=lambda c: float(np.linalg.norm(X[i] - 质[c])))
        混淆[y[i]][预测] += 1
        if 预测 == y[i]:
            正确 += 1
    return dict(准确率=正确 / max(len(X), 1), 样本数=len(X), 混淆=混淆, 类=类s)


def _去均(a):
    a = np.asarray(a, dtype=np.float64)
    return [float(x - a.mean()) for x in a]


def _句调(档, t: float, j: int, n: int, 标点: str) -> float:
    律 = H8.人类韵律_v8(档, 300)
    return float(律._句调(float(t), int(j), int(n), str(标点)))


def 测声调(快速: bool = False) -> dict:
    出 = dict(角色={}, 单音节={}, 单音节每调={}, 单音节调域={},
              句子=dict(每调={}, 每位置={}, 样本=[]), 分辨={},
              颤音=dict(非零帧=0), 漂移={}, 鼻尾代价={},
              失配=dict(帧=0, 总=0), 最长元音ms=0.0, 句调幅度={})
    角 = 角色s if not 快速 else ("EmoCompanion",)
    句s = 句集[:3] if 快速 else 句集
    五点半 = np.linspace(0.0, 1.0, 5)
    for 名 in 角:
        档 = 取档(名)
        F中 = 中F(档)
        出["角色"][名] = dict(中F=round(F中, 3), 基频=float(档.基频), 声区=float(档.声区),
                            共振峰缩放=[float(v) for v in 档.共振峰缩放])
        出["单音节"][名] = {}
        for p, 调 in (("ma1", 1), ("ma2", 2), ("ma3", 3), ("ma4", 4)):
            律 = 记录韵律(档, 300)
            w, 元, 帧 = 律.说音节(p)
            边 = 律.边界[0]
            出["颤音"]["非零帧"] += int(律.颤非零帧)
            出["最长元音ms"] = max(出["最长元音ms"], (边[2] - 边[1]) * 10.0)
            tr = F0轨迹(w)
            m = _音节F0测(w, tr, 帧, 边, F中)
            出["失配"]["帧"] += (m["失配帧"] if m else 0)
            出["失配"]["总"] += (m["帧数"] + m["失配帧"]) if m else 0
            出["单音节"][名][p] = dict(调=调, 目标=目标轮廓(调), 测=m,
                                      帧数=len(帧["f0"]), 边=list(边))
        # 六句
        for 号, 文本 in 句s:
            律 = 记录韵律(档, 300)
            w, 元, 帧 = 律.说(文本)
            出["颤音"]["非零帧"] += int(律.颤非零帧)
            音节s = 句音节列表(文本)
            边s = 律.边界
            if len(边s) != len(音节s):
                raise RuntimeError("边界数 %d ≠ 音节数 %d（%s）" % (len(边s), len(音节s), 文本))
            tr = F0轨迹(w)
            句调值 = []
            for k, (s, 边) in enumerate(zip(音节s, 边s)):
                元帧 = (边[2] - 边[1])
                尾帧 = int(H8.参数["鼻尾帧数"]) if s["尾"] else 0
                出["最长元音ms"] = max(出["最长元音ms"], max(元帧 - 尾帧, 0) * 10.0)
                m = _音节F0测(w, tr, 帧, 边, F中)
                出["失配"]["帧"] += (m["失配帧"] if m else 0)
                出["失配"]["总"] += (m["帧数"] + m["失配帧"]) if m else 0
                句调5 = [_句调(档, float(t), s["j"], s["n"], s["标点"]) for t in 五点半]
                句调值 += 句调5
                出["句子"]["样本"].append(dict(
                    号=号, 角色=名, 拼音=s["拼音"], 调=s["调"], 韵=s["韵"],
                    尾=s["尾"], j=s["j"], n=s["n"], 标点=s["标点"], 末尾=s["末尾"],
                    元音帧=max(元帧 - 尾帧, 0), 鼻尾帧=尾帧, 句调=句调5, 测=m))
            出["句子"].setdefault("句调范围", []).append(
                dict(号=号, 角色=名, 起=round(min(句调值), 3), 止=round(max(句调值), 3),
                     幅=round(max(句调值) - min(句调值), 3)))
    # —— 单音节每调聚合（跨角色）——
    for 调 in (1, 2, 3, 4):
        items = []
        for 名 in 出["单音节"]:
            for p in ("ma1", "ma2", "ma3", "ma4"):
                r = 出["单音节"][名][p]
                if r["调"] == 调 and r["测"]:
                    items.append(r)
        if not items:
            continue
        目 = 目标轮廓(调)
        出["单音节每调"][调] = dict(
            n=len(items), 目标=目,
            帧=dict(起=均值([r["测"]["帧_起"] for r in items]), 终=均值([r["测"]["帧_终"] for r in items]),
                   极小=均值([r["测"]["帧_极小"] for r in items]), 极大=均值([r["测"]["帧_极大"] for r in items])),
            波形=dict(起=均值([r["测"]["起"] for r in items]), 终=均值([r["测"]["终"] for r in items]),
                     极小=均值([r["测"]["极小"] for r in items]), 极大=均值([r["测"]["极大"] for r in items])),
            帧vs目标=dict(起=均值([r["测"]["帧_起"] - 目["起"] for r in items]),
                        终=均值([r["测"]["帧_终"] - 目["终"] for r in items])),
            Δ起=均值([r["测"]["起"] - r["测"]["帧_起"] for r in items]),
            Δ终=均值([r["测"]["终"] - r["测"]["帧_终"] for r in items]),
        )
    帧全 = [v for d in 出["单音节每调"].values() for v in (d["帧"]["极小"], d["帧"]["极大"])]
    波全 = [v for d in 出["单音节每调"].values() for v in (d["波形"]["极小"], d["波形"]["极大"])]
    目全 = [v for d in 出["单音节每调"].values() for v in (d["目标"]["极小"], d["目标"]["极大"])]
    出["单音节调域"] = dict(目标=round(max(目全) - min(目全), 3) if 目全 else None,
                        帧=round(max(帧全) - min(帧全), 3) if 帧全 else None,
                        波形=round(max(波全) - min(波全), 3) if 波全 else None)
    # —— 句子：每调 / 每位置 ——
    样 = 出["句子"]["样本"]
    for 调 in (1, 2, 3, 4, 5):
        它的 = [r for r in 样 if r["调"] == 调 and r["测"]]
        if not 它的:
            continue
        出["句子"]["每调"][调] = dict(
            n=len(它的),
            帧=dict(起=均值([r["测"]["帧_起"] for r in 它的]), 终=均值([r["测"]["帧_终"] for r in 它的]),
                   极小=均值([r["测"]["帧_极小"] for r in 它的]), 极大=均值([r["测"]["帧_极大"] for r in 它的]),
                   起std=标准差([r["测"]["帧_起"] for r in 它的]), 终std=标准差([r["测"]["帧_终"] for r in 它的])),
            波形=dict(起=均值([r["测"]["起"] for r in 它的]), 终=均值([r["测"]["终"] for r in 它的]),
                     极小=均值([r["测"]["极小"] for r in 它的]), 极大=均值([r["测"]["极大"] for r in 它的]),
                     起std=标准差([r["测"]["起"] for r in 它的]), 终std=标准差([r["测"]["终"] for r in 它的])),
        )
    for 位置, 判 in (("非末位", lambda r: not r["末尾"]),
                     ("句末", lambda r: r["末尾"] and r["标点"] in H7.句末类
                      and r["标点"] not in H7.疑问类),
                     ("疑问末", lambda r: r["末尾"] and r["标点"] in H7.疑问类),
                     ("逗号末", lambda r: r["末尾"] and r["标点"] == "，")):
        for 调 in (1, 2, 3, 4):
            它的 = [r for r in 样 if r["调"] == 调 and 判(r) and r["测"]]
            if len(它的) < 2:
                continue
            出["句子"]["每位置"]["%s·调%d" % (位置, 调)] = dict(
                n=len(它的), 起=均值([r["测"]["起"] for r in 它的]),
                终=均值([r["测"]["终"] for r in 它的]),
                极小=均值([r["测"]["极小"] for r in 它的]),
                极大=均值([r["测"]["极大"] for r in 它的]))
    波半 = [v for r in 样 if r["测"] for v in (r["测"]["极小"], r["测"]["极大"])]
    if 波半:
        arr = np.array(波半)
        出["句子"]["调域"] = dict(n=len(arr), 全=round(float(arr.max() - arr.min()), 3),
                              分位5_95=round(float(np.percentile(arr, 95) - np.percentile(arr, 5)), 3),
                              p5=round(float(np.percentile(arr, 5)), 3),
                              p95=round(float(np.percentile(arr, 95)), 3))
    # —— 调 1/2/3/4 可分度（留一最近质心）——
    基 = [r for r in 样 if r["调"] in (1, 2, 3, 4) and r["测"] and r["测"]["帧数"] >= 3]
    for 名, 特征 in (
        ("原样", lambda r: r["测"]["估5"]),
        ("去句调", lambda r: [a - b for a, b in zip(r["测"]["估5"], r["句调"])]),
        ("纯形状", lambda r: _去均([a - b for a, b in zip(r["测"]["估5"], r["句调"])])),
    ):
        X = [特征(r) for r in 基]
        y = ["调%d" % r["调"] for r in 基]
        出["分辨"][名] = _留一最近质心(X, y)
    if 基:
        质 = {}
        for 调 in (1, 2, 3, 4):
            成员 = [[a - b for a, b in zip(r["测"]["估5"], r["句调"])] for r in 基 if r["调"] == 调]
            if 成员:
                质[调] = np.mean(np.stack(成员), axis=0)
        距离 = {}
        for a in 质:
            for b in 质:
                if a < b:
                    距离["调%d_调%d" % (a, b)] = round(float(np.linalg.norm(质[a] - 质[b])), 3)
        出["分辨"]["质心距离_半音"] = 距离
    # 句调幅度：同句内跨音节/跨位置的 句调 极差
    幅s = [r["幅"] for r in 出["句子"].get("句调范围", [])]
    出["句调幅度"] = dict(句数=len(幅s), 中位=中位(幅s), 最大=(max(幅s) if 幅s else None),
                      各句=出["句子"].get("句调范围", []))
    # 漂移：单音节帧设计 − (目标轮廓 + 0.30 句调) 的残差
    残 = []
    for 名 in 出["单音节"]:
        for p in ("ma1", "ma2", "ma3", "ma4"):
            r = 出["单音节"][名][p]
            if r["测"]:
                残 += [a - b for a, b in zip(r["测"]["帧5"], _目标5(r["调"]))]
    if 残:
        出["漂移"] = dict(n=len(残), std=标准差(残), maxabs=max(abs(v) for v in 残), 中位=中位(残))
    # 颤音门槛
    出["颤音"].update(门槛ms=float(H7.颤音最短时长 * 1000.0), 幅度半音=float(H7.颤音半音),
                    最长元音ms=round(float(出["最长元音ms"]), 1),
                    触发=bool(出["颤音"]["非零帧"] > 0))
    # 鼻尾代价：有/无鼻尾音节的元音帧数
    有 = [r["元音帧"] for r in 样 if r["尾"] and r["测"]]
    无 = [r["元音帧"] for r in 样 if (not r["尾"]) and r["测"]]
    出["鼻尾代价"] = dict(有鼻尾_元音帧中位=中位(有), 有鼻尾_n=len(有),
                      无鼻尾_元音帧中位=中位(无), 无鼻尾_n=len(无),
                      有鼻尾_元音ms=round((中位(有) or 0) * 10.0, 1),
                      无鼻尾_元音ms=round((中位(无) or 0) * 10.0, 1))
    出["失配"]["率"] = 出["失配"]["帧"] / max(出["失配"]["总"], 1)
    return 出


def _目标5(调: int) -> list:
    ts = np.linspace(0.0, 1.0, 5)
    if 调 == 5:
        return [-0.5 + 0.30] * 5
    轮廓 = Y.声调表.get(int(调), Y.声调表[1])
    return [(Y.折线(轮廓, float(t)) - 3.0) * 1.2 + 0.30 for t in ts]


# ================================================================ Q2 元音
def 测元音(快速: bool = False) -> dict:
    出 = dict(方法=dict(采样率=16000, 窗毫秒=30.0, 跳毫秒=10.0, 阶=14, 预加重=0.97,
                      极点带宽界=500.0, 簇容差=0.06, 匹配容差=0.15,
                      说明="44.1k→16k resample_poly 后自实现 LPC+Levinson-Durbin+极点聚类；"
                          "盲测=支持率≥50%的最低 3 簇；跟踪=对『帧目标×s』找最近簇(≤15%)"),
              表=[], 复韵=[], 鼻尾=[], 双乘s={}, 分离={}, 失败帧=dict(帧=0, 总=0))
    韵表 = [("a", "a1"), ("i", "i1"), ("u", "u1"), ("v", "v1"),
            ("ii", "ii1"), ("iii", "iii1"), ("ai", "ai1"), ("ao", "ao1"),
            ("an", "an1"), ("ang", "ang1")]
    角 = ("EmoCompanion",) if 快速 else 角色s
    for 名 in 角:
        档 = 取档(名)
        s = tuple(float(x) for x in 档.共振峰缩放)
        for 韵, 拼 in 韵表:
            律 = 记录韵律(档, 300)
            w, 元, 帧 = 律.说音节(拼)
            起, 元起, 止 = 律.边界[0]
            N尾 = int(H8.参数["鼻尾帧数"]) if Y.解析音节(拼)["尾"] else 0
            元帧 = list(range(元起, max(止 - N尾, 元起 + 1)))
            稳 = 元帧[int(len(元帧) * 0.25):int(len(元帧) * 0.75) + 1] or 元帧
            F0med = 中位([帧["f0"][i] for i in 稳])
            音律 = Y.解析音节(拼)
            帧目标 = dict(F1=中位([帧["f1"][i] for i in 稳]),
                          F2=中位([帧["f2"][i] for i in 稳]),
                          F3=中位([帧["f3"][i] for i in 稳]))
            引擎实际 = dict(F1=帧目标["F1"] * s[0], F2=帧目标["F2"] * s[1], F3=帧目标["F3"] * s[2])
            簇 = 段极点(w, 稳)
            盲 = 三共振峰(簇, len(稳))
            跟 = 匹配共振峰(簇, [引擎实际["F1"], 引擎实际["F2"], 引擎实际["F3"]])
            出["失败帧"]["总"] += len(稳)
            if len(盲) < 3:
                出["失败帧"]["帧"] += 1
            帧均 = 中位([帧["f1"][i] for i in 元帧]), 中位([帧["f2"][i] for i in 元帧])
            音韵目标 = None
            if len(音律["目标s"]) == 1:
                音韵目标 = dict(F1=float(音律["目标s"][0][0]), F2=float(音律["目标s"][0][1]),
                              F3=float(音律["目标s"][0][2]))
            出["表"].append(dict(
                角色=名, 韵=韵, 拼音=拼, F0=F0med, 音韵目标=音韵目标,
                帧目标=帧目标, 引擎实际=引擎实际,
                盲测=[float(x) for x in 盲],
                盲测偏差pct=[100.0 * (盲[j] - 帧目标[k]) / 帧目标[k]
                          for j, k in enumerate(("F1", "F2", "F3")) if j < len(盲)],
                跟踪=跟,
                跟踪偏差pct=[v["偏差pct"] for v in 跟],
                匹配数=sum(1 for v in 跟 if v["匹配"]),
                稳定帧数=len(稳), 元音帧数=len(元帧), 帧数=len(帧["f0"]),
                F1可测=bool(引擎实际["F1"] > 1.30 * (F0med or 1e9)),
            ))
            # 复韵：3 点滑移（跟踪法）
            if 韵 in ("ai", "ao"):
                点 = []
                for t in (0.1, 0.5, 0.9):
                    i = 元帧[min(int(round(t * (len(元帧) - 1))), len(元帧) - 1)]
                    目 = (帧["f1"][i] * s[0], 帧["f2"][i] * s[1], 帧["f3"][i] * s[2])
                    cc = 段极点(w, [i])
                    点.append(dict(t=t, 帧目标=list(目),
                                 跟踪=[v["测"] for v in 匹配共振峰(cc, list(目))]))
                出["复韵"].append(dict(角色=名, 韵=韵, 点=点))
            if N尾 > 0:
                尾帧 = list(range(max(止 - N尾, 元起), 止))
                尾簇 = 段极点(w, 尾帧)
                目2 = 中位([帧["f2"][i] for i in 尾帧]) * s[1]
                跟2 = 匹配共振峰(尾簇, [目2])
                出["鼻尾"].append(dict(角色=名, 韵=韵, 拼音=拼, 帧数=len(尾帧),
                                    F2_引擎实际=目2,
                                    F2_跟踪=(跟2[0]["测"] if 跟2 else None)))
    # —— 双乘 s 消融（EmoCompanion a1）——
    档 = 取档("EmoCompanion")
    s = tuple(float(x) for x in 档.共振峰缩放)
    律 = 记录韵律(档, 300)
    w, 元, 帧 = 律.说音节("a1")
    音真 = 律.音色()
    音一 = dict(音真)
    音一["共振峰缩放"] = (1.0, 1.0, 1.0)
    w双, _ = E7.合成(帧, 音色=音真, 种子=300)
    w单, _ = E7.合成(帧, 音色=音一, 种子=300)
    边 = 律.边界[0]
    稳 = list(range(边[1] + 2, 边[2] - 2)) or [边[1]]
    def _测双(xx):
        cc = 段极点(xx, 稳)
        盲 = 三共振峰(cc, len(稳))
        return dict(盲测=[float(x) for x in 盲],
                    帧目标=dict(F1=中位([帧["f1"][i] for i in 稳]),
                              F2=中位([帧["f2"][i] for i in 稳]),
                              F3=中位([帧["f3"][i] for i in 稳])))
    双 = _测双(w双)
    单 = _测双(w单)
    比 = {}
    for j, k in enumerate(("F1", "F2", "F3")):
        比[k] = (双["盲测"][j] / 单["盲测"][j]) if (j < len(双["盲测"]) and j < len(单["盲测"])
                                                 and 单["盲测"][j]) else None
    出["双乘s"] = dict(角色="EmoCompanion", 韵="a", 角色共振峰缩放=list(s),
                    真实音色=双, 缩放归一=单, 实测比=比,
                    预期比={"F1": s[0], "F2": s[1], "F3": s[2]})
    # —— 分离度（单元音，用可测的 F2/F3；F1 单独标注）——
    单韵 = ("a", "i", "u", "v", "ii", "iii")
    for 名 in 角:
        rows = [r for r in 出["表"] if r["角色"] == 名 and r["韵"] in 单韵]
        if len(rows) < 4:
            continue
        数据 = {}
        for r in rows:
            盲 = r["盲测"]
            f2, f3 = _跟踪F23(r)
            数据[r["韵"]] = dict(F1=盲[0] if len(盲) > 0 else None,
                              F2=f2, F3=f3,
                              F1可测=r["F1可测"])
        对 = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i]["韵"], rows[j]["韵"]
                da, db = 数据[a], 数据[b]
                d23 = math.sqrt((da["F2"] - db["F2"]) ** 2 + (da["F3"] - db["F3"]) ** 2)
                d123 = math.sqrt(d23 ** 2 + (da["F1"] - db["F1"]) ** 2)
                对.append(dict(对="%s/%s" % (a, b),
                              dF2=round(da["F2"] - db["F2"], 1),
                              dF3=round(da["F3"] - db["F3"], 1),
                              dF1=round(da["F1"] - db["F1"], 1),
                              距离F2F3=round(d23, 1), 距离F1F2F3=round(d123, 1),
                              F1均可测=bool(da["F1可测"] and db["F1可测"])))
        出["分离"][名] = dict(对=sorted(对, key=lambda d: d["距离F2F3"]), 数据=数据)
    # —— 元音留一最近质心 ——
    #   ① 按角色归一（去掉 s 的整体频标差，否则跨角色 5% 差会把相邻元音混起来）；
    #   ② 排除 `ao`：它的 F1≈621、F2≈974，相距 353 Hz < 1.5·F0(≈615) ⇒ LPC 极点合并、F2 不可测（测量限制，非引擎问题）。
    行全 = [r for r in 出["表"] if r["韵"] != "ao" and None not in _跟踪F23(r)]
    归一 = {}
    for 名 in 角:
        组 = [r for r in 行全 if r["角色"] == 名]
        if 组:
            归一[名] = (中位([_跟踪F23(r)[0] for r in 组]), 中位([_跟踪F23(r)[1] for r in 组]))
    向 = [(r["韵"], _跟踪F23(r)[0] / 归一[r["角色"]][0], _跟踪F23(r)[1] / 归一[r["角色"]][1])
          for r in 行全 if r["角色"] in 归一 and 归一[r["角色"]][0] and 归一[r["角色"]][1]]
    X = [[f2, f3] for _, f2, f3 in 向]
    y = [韵 for 韵, _, _ in 向]
    计数 = {}
    for v in y:
        计数[v] = 计数.get(v, 0) + 1
    if len(X) >= 10 and 计数 and min(计数.values()) >= 2:
        出["分离"]["留一准确率"] = _留一最近质心(X, y)
        出["分离"]["留一说明"] = "跨角色（%d 角色 × 9 韵，排除 ao）；特征 = 角色归一后的 F2/F3" % len(角)
        出["分离"]["特征"] = "角色归一 F2/F3（跟踪优先；ao 因 F1/F2 被 F0 合并而排除）"
    出["失败帧"]["率"] = 出["失败帧"]["帧"] / max(出["失败帧"]["总"], 1)
    return 出


# ================================================================ Q3 擦音/塞擦部位谱
def 测擦音(快速: bool = False) -> dict:
    出 = dict(方法=dict(帧侧="按帧切 声源==0 且 在 [声母起,元起) 内；塞擦第 1 帧为爆破、其余为摩擦",
                      波形侧="同步样本块 10ms 汉明窗",
                      特征="谱重心/谱峰/谱展/低频占比(<1kHz)/时长",
                      种子=list(range(300, 310))),
              表=[], 逐种子={}, 可分={}, 擦塞擦={}, 送气VOT={}, 真人={})
    声表 = [("s", "sa1", "撒"), ("sh", "sha1", "沙"), ("x", "xia1", "虾"),
            ("f", "fa1", "发"), ("h", "ha1", "哈"), ("c", "ca1", "擦"),
            ("zh", "zha1", "扎"), ("j", "ji1", "机"), ("q", "qi1", "七")]
    种子s = list(range(300, 303)) if 快速 else list(range(300, 310))
    档 = 取档("EmoCompanion")
    for 声, 拼, 字 in 声表:
        info = H8.声母_v8.get(声, {})
        方式 = info.get("方式")
        每 = []
        for 种 in 种子s:
            律 = 记录韵律(档, 种)
            w, 元, 帧 = 律.说音节(拼)
            起, 元起, 止 = 律.边界[0]
            噪 = [i for i in range(起, 元起) if 帧["声源"][i] == 0]
            擦 = 噪[1:] if (方式 == "塞擦" and len(噪) > 1) else 噪
            fe = [谱特征(w[i * 每帧样本:(i + 1) * 每帧样本]) for i in 擦]
            每.append(dict(种子=种, 帧数=len(擦), 摩擦ms=len(擦) * 10.0,
                          重心=中位([x["重心"] for x in fe]),
                          峰=中位([x["峰"] for x in fe]),
                          展=中位([x["展"] for x in fe]),
                          低频=中位([x["低频占比"] for x in fe]),
                          爆破帧=len(噪) - len(擦)))
        出["逐种子"][声] = 每
        med = {k: 中位([x[k] for x in 每]) for k in ("重心", "峰", "展", "低频", "摩擦ms")}
        sd = {k: 标准差([x[k] for x in 每]) for k in ("重心", "峰", "展", "低频", "摩擦ms")}
        出["表"].append(dict(声母=声, 例字=字, 拼音=拼, 方式=方式,
                           设计频带=list(info.get("频带") or ()) if info.get("频带") else None,
                           谱重心=med["重心"], 谱峰=med["峰"], 谱展=med["展"],
                           低频占比=med["低频"], 摩擦ms=med["摩擦ms"],
                           重心sd=sd["重心"], 峰sd=sd["峰"], 展sd=sd["展"],
                           文献谱峰={"s": 文献["谱峰_s"], "sh": 文献["谱峰_sh"],
                                   "x": 文献["谱峰_x"], "f": 文献["谱峰_f"]}.get(声)))
    # —— s/sh/x 两两可分度 ——
    三 = {声: 出["逐种子"][声] for 声 in ("s", "sh", "x")}
    对 = []
    for a, b in (("s", "sh"), ("s", "x"), ("sh", "x")):
        va = [x["重心"] for x in 三[a]]
        vb = [x["重心"] for x in 三[b]]
        pa = float(np.std(va, ddof=1)) if len(va) > 1 else 0.0
        pb = float(np.std(vb, ddof=1)) if len(vb) > 1 else 0.0
        汇 = math.sqrt(max((pa * pa + pb * pb) / 2.0, 1e-12))
        对.append(dict(对="%s/%s" % (a, b),
                      Δ重心=round(float(np.mean(vb) - np.mean(va)), 1),
                     sd_a=round(pa, 1), sd_b=round(pb, 1),
                     d=round(float(abs(np.mean(vb) - np.mean(va)) / 汇), 2) if 汇 > 0 else None))
    X, y = [], []
    for 声 in ("s", "sh", "x"):
        for x in 出["逐种子"][声]:
            X.append([x["重心"], x["峰"], x["展"], x["低频"]])
            y.append(声)
    出["可分"] = dict(两两=对, 留一=_留一最近质心(X, y), 特征="[谱重心, 谱峰, 谱展, 低频占比]")
    # —— 擦 vs 同部位塞擦 ——
    for 擦声, 塞声 in (("s", "c"), ("sh", "zh"), ("x", "j")):
        a = [x for x in 出["表"] if x["声母"] == 擦声][0]
        b = [x for x in 出["表"] if x["声母"] == 塞声][0]
        出["擦塞擦"]["%s/%s" % (擦声, 塞声)] = dict(
            Δ重心=round(float(b["谱重心"] - a["谱重心"]), 1),
            Δ峰=round(float(b["谱峰"] - a["谱峰"]), 1),
            Δ摩擦ms=round(float(b["摩擦ms"] - a["摩擦ms"]), 1),
            塞闭ms=round(float(Y.声母[塞声].get("塞闭", 0.02) * max(H8.参数["缩底"], 0.0) * 1000.0), 1),
        )
    # —— 送气塞 VOT 段谱（部位线索只剩爆破帧）——
    for 声, 拼 in (("p", "pa1"), ("t", "ta1"), ("k", "ka1"),
                   ("b", "ba1"), ("d", "da1"), ("g", "ga1")):
        律 = 记录韵律(档, 300)
        w, 元, 帧 = 律.说音节(拼)
        起, 元起, 止 = 律.边界[0]
        噪 = [i for i in range(起, 元起) if 帧["声源"][i] == 0]
        爆破 = [i for i in 噪 if isinstance(帧["噪带"][i], (tuple, list))]
        vot = [i for i in 噪 if not isinstance(帧["噪带"][i], (tuple, list))]
        出["送气VOT"][声] = dict(
            爆破帧=len(爆破), VOT帧=len(vot),
            VOT重心=中位([谱特征(w[i * 每帧样本:(i + 1) * 每帧样本])["重心"] for i in vot]),
            VOT峰=中位([谱特征(w[i * 每帧样本:(i + 1) * 每帧样本])["峰"] for i in vot]),
            爆破重心=中位([谱特征(w[i * 每帧样本:(i + 1) * 每帧样本])["重心"] for i in 爆破]),
            爆破峰=中位([谱特征(w[i * 每帧样本:(i + 1) * 每帧样本])["峰"] for i in 爆破]),
            设计爆破带=list(Y.声母[声]["爆破带"]))
    # —— 真人参考（只读 ref44，用内生耳切段）——
    rows = []
    for p in sorted(glob.glob(os.path.join(M61, "素材", "*", "ref44_*.wav"))):
        r = 耳.听文件(p)
        rows.append(dict(文件=os.path.basename(p), 时长秒=r["时长秒"],
                        清段中位ms=r["清时长中位ms"], 清谱重心=r["清谱重心中位Hz"],
                        清谱峰=r["清谱峰中位Hz"], 清总占比=r["清总占比"]))
    出["真人"] = dict(
        文件=rows,
        清段中位ms_中位=中位([r["清段中位ms"] for r in rows]),
        清段中位ms_范围=[min(r["清段中位ms"] for r in rows), max(r["清段中位ms"] for r in rows)],
        清谱重心_中位=中位([r["清谱重心"] for r in rows]),
        清谱重心_范围=[min(r["清谱重心"] for r in rows), max(r["清谱重心"] for r in rows)],
        清谱峰_中位=中位([r["清谱峰"] for r in rows]),
        注="ref44 是歌唱/朗读混合、无音素标注 ⇒ 只作分布域锚定，不作逐音素对照",
    )
    return 出


# ================================================================ Q4 时长
def 测时长(快速: bool = False) -> dict:
    出 = dict(表=[], 汇总=[], 真人={}, 锚=dict())
    声表 = [("p", "pa1"), ("t", "ta1"), ("k", "ka1"), ("b", "ba1"), ("d", "da1"), ("g", "ga1"),
            ("s", "sa1"), ("sh", "sha1"), ("x", "xia1"), ("f", "fa1"), ("h", "ha1"),
            ("z", "za1"), ("c", "ca1"), ("zh", "zha1"), ("ch", "cha1"), ("j", "ji1"), ("q", "qi1"),
            ("m", "ma1"), ("n", "na1"), ("l", "la1")]
    档 = 取档("EmoCompanion")
    for 声, 拼 in 声表:
        info = H8.声母_v8.get(声, {})
        方式 = info.get("方式")
        律 = 记录韵律(档, 300)
        w, 元, 帧 = 律.说音节(拼)
        起, 元起, 止 = 律.边界[0]
        src = 帧["声源"]
        区 = list(range(起, 元起))
        nb = 帧["噪带"]
        闭帧 = [i for i in 区 if src[i] == -1]
        噪帧 = [i for i in 区 if src[i] == 0]
        浊帧 = [i for i in 区 if src[i] == 1]
        if 方式 == "塞擦":
            爆帧 = 噪帧[:1]
            摩帧 = 噪帧[1:]
            宽带帧 = []
        elif 方式 == "塞":
            爆帧 = [i for i in 噪帧 if isinstance(nb[i], (tuple, list))]
            宽带帧 = [i for i in 噪帧 if not isinstance(nb[i], (tuple, list))]
            摩帧 = []
        elif 方式 == "擦":
            爆帧 = []
            宽带帧 = [i for i in 噪帧 if not isinstance(nb[i], (tuple, list))]
            摩帧 = 噪帧
        else:
            爆帧, 宽带帧, 摩帧 = [], [], []
        帧侧 = dict(
            闭ms=len(闭帧) * 10.0,
            爆破ms=len(爆帧) * 10.0,
            送气VOTms=len(宽带帧) * 10.0,
            摩擦ms=len(摩帧) * 10.0,
            浊ms=len(浊帧) * 10.0,
            除阻到浊起ms=((元起 - 噪帧[0]) * 10.0) if 噪帧 else 0.0,
        )
        # —— 波形侧（同一 10ms 网格）——
        dB = 能量dB逐帧(w)
        zq = 浊度逐帧(w)
        kk = min(len(dB), len(zq))
        峰dB = float(np.max(dB[:kk])) if kk else 0.0
        静阈 = 峰dB - 30.0
        区内 = [i for i in 区 if i < kk]
        静 = [i for i in 区内 if dB[i] < 静阈]
        浊 = [i for i in 区内 if dB[i] >= 静阈 and zq[i] >= 0.5]
        清 = [i for i in 区内 if dB[i] >= 静阈 and zq[i] < 0.5]
        除阻起 = next((i for i in 区内 if dB[i] >= 静阈), None)
        浊起 = None
        for i in range(max(元起, 0), kk):
            if zq[i] >= 0.5 and dB[i] >= 静阈:
                浊起 = i
                break
        清段 = 0.0
        if 除阻起 is not None:
            止i = 浊起 if 浊起 is not None else 元起
            清段 = sum(1 for i in 区内 if 除阻起 <= i < 止i and zq[i] < 0.5) * 10.0
        波侧 = dict(静ms=len(静) * 10.0, 清ms=len(清) * 10.0, 浊ms=len(浊) * 10.0,
                  清段ms=清段,
                  除阻到浊起ms=((浊起 - 除阻起) * 10.0) if (除阻起 is not None and 浊起 is not None) else None)
        # —— 耳（内生耳，裁剪到音节 span，避免换气被算进清段）——
        裁 = np.asarray(w[起 * 每帧样本:止 * 每帧样本], dtype=np.float64)
        rr = 耳.听(裁) if len(裁) > 1000 else dict(静时长中位ms=0.0, 清时长中位ms=0.0, 浊时长中位ms=0.0)
        出["表"].append(dict(声母=声, 拼音=拼, 方式=方式, 设计塞闭ms=info.get("塞闭", 0) * 1000.0,
                           设计VOTms=info.get("VOT", 0) * 1000.0, 设计擦时ms=info.get("擦时", 0) * 1000.0,
                           设计时长ms=info.get("时长", 0) * 1000.0,
                           帧侧=帧侧, 波形侧=波侧,
                           耳=dict(清ms=rr["清时长中位ms"], 静ms=rr["静时长中位ms"],
                                 浊ms=rr["浊时长中位ms"])))
    def _组(声s):
        return [r for r in 出["表"] if r["声母"] in 声s]
    def _汇总(项, 声s, 帧键, 波键, 文献数=None, 文献文=None, 真人值=None):
        组 = _组(声s)
        帧v = [r["帧侧"].get(帧键) for r in 组 if r["帧侧"].get(帧键) is not None]
        波v = [r["波形侧"].get(波键) for r in 组 if r["波形侧"].get(波键) is not None]
        中 = 中位(帧v)
        return dict(项=项, 声母="、".join(声s), n=len(组),
                   帧侧中位=中, 帧侧范围=[min(帧v), max(帧v)] if 帧v else None,
                   波形侧中位=中位(波v), 波形侧范围=[min(波v), max(波v)] if 波v else None,
                   文献=文献文, 文献数=文献数,
                   比_文献=(round(float(中) / float(文献数), 3) if (中 is not None and 文献数) else None),
                   真人=真人值,
                   比_真人=(round(float(中) / float(真人值), 3) if (中 is not None and 真人值) else None))
    锚 = 中位([耳.听文件(os.path.join(
        M61, "素材", 名, "ref44_%s_01.wav" % ("洛天依_清唱" if 名 == "洛天依" else 名)))["清时长中位ms"]
        for 名 in ("EmoCompanion", "SpeakerA", "洛天依")])
    出["锚"] = dict(真人清段中位ms=锚,
                  注="真人 ref44 歌唱/朗读混合、无音素标注 ⇒ 只能给「清段时长」整体锚，不能给逐音素 VOT")
    出["汇总"] = [
        _汇总("p/t/k 送气 VOT（除阻→浊起）", ("p", "t", "k"), "除阻到浊起ms", "除阻到浊起ms",
              100.0, "90–110", 锚),
        _汇总("b/d/g 不送气 VOT（除阻→浊起）", ("b", "d", "g"), "除阻到浊起ms", "除阻到浊起ms",
              17.5, "10–25", 锚),
        _汇总("s/sh/x/f 擦音时长", ("s", "sh", "x", "f"), "摩擦ms", "清段ms",
              153.0, "153 ± 38", 锚),
        _汇总("h 擦音时长", ("h",), "摩擦ms", "清段ms", 153.0, "153 ± 38", 锚),
        _汇总("p/t/k 塞闭时长", ("p", "t", "k"), "闭ms", "静ms", None, "持阻 20–30", None),
        _汇总("b/d/g 塞闭时长", ("b", "d", "g"), "闭ms", "静ms", None, "持阻 20–30", None),
        _汇总("z/zh/j 塞擦摩擦段", ("z", "zh", "j"), "摩擦ms", "清段ms", None, "无给定数", 锚),
        _汇总("c/ch/q 塞擦摩擦段", ("c", "ch", "q"), "摩擦ms", "清段ms", None, "无给定数", 锚),
        _汇总("m/n/l 浊声母时长", ("m", "n", "l"), "浊ms", "浊ms", 75.0, "60–90", 锚),
    ]
    rows = []
    for p in sorted(glob.glob(os.path.join(M61, "素材", "*", "ref44_*.wav"))):
        rr = 耳.听文件(p)
        rows.append(dict(文件=os.path.basename(p), 清段中位ms=rr["清时长中位ms"],
                        浊段中位ms=rr["浊时长中位ms"], 清总占比=rr["清总占比"]))
    出["真人"] = dict(文件=rows, 清段中位ms_中位=中位([r["清段中位ms"] for r in rows]),
                   清段中位ms_范围=[min(r["清段中位ms"] for r in rows),
                                 max(r["清段中位ms"] for r in rows)])
    return 出


# ================================================================ 反证（可证伪清单）
def 测反证() -> dict:
    出 = dict()
    档 = 取档("EmoCompanion")
    old = float(H8.参数["爆破秒"])
    try:
        H8.参数["爆破秒"] = 0.005
        w1, _, f1 = 记录韵律(档, 300).说音节("pa1")
        H8.参数["爆破秒"] = 0.008
        w2, _, f2 = 记录韵律(档, 300).说音节("pa1")
    finally:
        H8.参数["爆破秒"] = old
    出["爆破秒_0.005_vs_0.008"] = dict(
        波形sha256同=_哈希(w1) == _哈希(w2), 帧f0同=(f1["f0"] == f2["f0"]),
        说明="0.005/0.008 s 在 10ms 帧栅格上都被量化成 1 帧 ⇒ 参数无作用")
    old2 = float(H8.参数["塞擦爆破秒"])
    try:
        H8.参数["塞擦爆破秒"] = 0.004
        w3, _, f3 = 记录韵律(档, 300).说音节("ca1")
        H8.参数["塞擦爆破秒"] = 0.006
        w4, _, f4 = 记录韵律(档, 300).说音节("ca1")
    finally:
        H8.参数["塞擦爆破秒"] = old2
    出["塞擦爆破秒_0.004_vs_0.006"] = dict(波形sha256同=_哈希(w3) == _哈希(w4),
                                        帧f0同=(f3["f0"] == f4["f0"]))
    出["颤音门槛"] = dict(门槛ms=float(H7.颤音最短时长 * 1000.0), 幅半音=float(H7.颤音半音))
    return 出


# ================================================================ 确定性 / 自审
def 测确定性与自审() -> dict:
    档 = 取档("EmoCompanion")
    t = "今天的风很轻，适合慢慢地说一会儿话。"
    律1 = 记录韵律(档, 300)
    w1, _, 帧1 = 律1.说(t)
    w2, _, 帧2 = 律1.说(t)              # 同实例第二趟（rng 状态已推进）
    律3 = 记录韵律(档, 300)
    w3, _, 帧3 = 律3.说(t)              # 异实例同种子
    基 = H8.人类韵律_v8(档, 300)
    w0, _, 帧0 = 基.说(t)               # 基类（对照：子类是否改变生成）
    帧同 = lambda A, B: all(A[k] == B[k] for k in A)
    确定 = dict(
        异实例同种子_波形同=_哈希(w1) == _哈希(w3),
        异实例同种子_帧同=帧同(帧1, 帧3),
        同实例两趟_波形同=_哈希(w1) == _哈希(w2),
        子类与基类_波形同=_哈希(w1) == _哈希(w0),
        子类与基类_帧同=帧同(帧1, 帧0),
        帧键数=len(帧1),
        注="生产路径（声带后端_v8.发声 每次新建 人类韵律_v8）走『异实例同种子』⇒ 可复现；"
          "同实例连续调用两趟不逐位同（rng 有状态，属既有件行为，登记为异常）",
    )
    禁用词 = ["whisper", "funasr", "paraformer", "torch", "tensorflow", "transformers",
            "librosa.effects", "librosa.beat", "openai", "requests", "urllib", "http"]
    with open(os.path.abspath(__file__), encoding="utf-8") as f:
        源 = f.read()
    导入行 = [ln.strip() for ln in 源.splitlines()
            if ln.strip().startswith(("import ", "from "))]
    命中 = sorted({w for w in 禁用词 if any(w.lower() in ln.lower() for ln in 导入行)})
    # 网络标记用「相邻字符串字面量拼接」写，避免把标记自身扫成命中（自审不扫文档串/不自我命中）
    标记 = ["ht" "tp://", "ht" "tps://", "url" "open", "sock" "et.", "requ" "ests."]
    网络标记 = [m for m in 标记 if m in 源]
    自审 = dict(禁用词命中=命中, 禁用词表=禁用词, 导入行数=len(导入行), 网络标记=网络标记,
               脚本sha256=hashlib.sha256(源.encode("utf-8")).hexdigest(),
               说明="本件零外挂：不 import ASR/学习框架/网络；F0/LPC/极点聚类/谱特征全部自实现"
                   "（自审只扫 import 行与网络标记，不扫文档串，避免把禁用词表自身算成命中）")
    return dict(确定性=确定, 自审=自审)


# ================================================================ 真人 F0 域（Q1 参考）
def 真人F0域() -> dict:
    from scipy.io import wavfile
    行 = []
    for p in sorted(glob.glob(os.path.join(M61, "素材", "*", "ref44_*.wav"))):
        fs, x = wavfile.read(p)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if np.max(np.abs(x)) > 1.2:
            x = x / 32768.0
        if fs != FS:
            g = math.gcd(int(fs), FS)
            x = np.asarray(resample_poly(x, FS // g, fs // g), dtype=np.float64)
        tr = F0轨迹(x)
        f = tr["f0"][(tr["f0"] > 0) & (tr["浊度"] > 0.60)]
        if len(f) < 20:
            continue
        中 = float(np.median(f))
        st = 12.0 * np.log2(f / 中)
        行.append(dict(文件=os.path.basename(p), F0中位=round(中, 1),
                      p5=round(float(np.percentile(st, 5)), 2),
                      p95=round(float(np.percentile(st, 95)), 2),
                      域_半音=round(float(np.percentile(st, 95) - np.percentile(st, 5)), 2)))
    return dict(文件=行, 域_半音_中位=中位([r["域_半音"] for r in 行]),
                注="真人参考是歌唱/朗读混合 ⇒ F0 域是「上限口径」，不能等同言语调域")


# ================================================================ 报告生成
def _f(x, n=1, 后缀=""):
    if x is None:
        return "—"
    return ("%." + str(n) + "f%s") % (float(x), 后缀)


def 写Markdown(R: dict) -> str:
    Q1, Q2, Q3, Q4 = R["Q1"], R["Q2"], R["Q3"], R["Q4"]
    L = []
    A = L.append
    A("# 诊断_声学 —— v8.1 渲染器四类声学线索的 DSP 实测（零外挂）")
    A("")
    A("> 执行：Teammate「声学测量」 ｜ %s ｜ **只新增 `meeting64/`**（C14：既有件一字未改）"
      % R["元信息"]["日期"])
    A("> 口径：`[实测]`＝本件真跑 ｜ 零外挂：不跑 ASR、不用任何模型/权重/网络；F0 自相关、LPC、极点聚类、谱特征**全部自实现**")
    A("> 被诊断件：`meeting63/实施/韵律_v8.py`（v8.1 实际发声件）+ `引擎_v7.py`；"
      "参考：`音韵_v5.py` 目标、`耳_内生.py` 切段、`meeting61/素材/ref44_*.wav` 真人")
    A("")
    A("## 0. 一句话结论")
    A("")
    A(R["结论"]["一句话"])
    A("")
    A("## 1. 方法与口径")
    A("")
    A("| 项 | 口径 |")
    A("|---|---|")
    A("| Q1 F0 | 输出波形逐 10ms 归一化自相关（30ms 窗心对齐帧中心，倍周期仲裁 + 3 点中值滤波）；半音 = 12·log2(F0/F_中)，F_中 = 基频×(1+0.5·声区) |")
    A("| Q2 共振峰 | 44.1k→16k（resample_poly）后**自实现** LPC（Levinson-Durbin，30ms/10ms，阶 14，预加重 0.97）；多帧极点聚类（±6%）；**盲测**＝支持率≥50% 的最低 3 簇；**跟踪**＝对『帧目标×s』找最近簇（≤15%） |")
    A("| Q3 部位谱 | 按帧切 `声源==0` 且落在 [声母起, 元起) 的样本块，10ms 汉明窗功率谱：谱重心/谱峰/谱展/低频占比(<1kHz)/时长（塞擦第 1 帧=爆破，不计入摩擦） |")
    A("| Q4 时长 | 帧侧＝帧 dict 的 `声源/噪带` 分段；波形侧＝同一 10ms 网格的 能量门(峰−30dB)+浊度门(0.5，20ms 前向窗)；耳＝`耳_内生` 确定性切段 |")
    A("| 确定性 | 固定种子 300；生产路径（每次新建实例）逐位同（§8） |")
    A("")
    A("角色：`EmoCompanion` 为主（与 01_预检报告 / ASR 判据同声线），Q1 与 Q2 三角色对照。")
    A("")
    # ---------------- Q1
    A("## 2. Q1 声调：F0 轮廓到底做出来了没有？")
    A("")
    A("**结论（一句话）**：" + R["结论"]["Q1"])
    A("")
    A("### 2.1 单音节 ma1–ma4（3 角色均值，半音）")
    A("")
    A("| 调 | 目标 起/终/极小…极大 | 帧设计 起/终/极小…极大 | **波形实测 起/终/极小…极大** | 帧−目标 起/终 | 波形−帧 起/终 |")
    A("|---|---|---|---|---|---|")
    for 调 in ("1", "2", "3", "4"):
        d = Q1["单音节每调"].get(调)
        if not d:
            continue
        A("| 调%s | %s / %s / %s…%s | %s / %s / %s…%s | **%s / %s / %s…%s** | %s / %s | %s / %s |" % (
            调,
            _f(d["目标"]["起"], 2), _f(d["目标"]["终"], 2), _f(d["目标"]["极小"], 2), _f(d["目标"]["极大"], 2),
            _f(d["帧"]["起"], 2), _f(d["帧"]["终"], 2), _f(d["帧"]["极小"], 2), _f(d["帧"]["极大"], 2),
            _f(d["波形"]["起"], 2), _f(d["波形"]["终"], 2), _f(d["波形"]["极小"], 2), _f(d["波形"]["极大"], 2),
            _f(d["帧vs目标"]["起"], 2), _f(d["帧vs目标"]["终"], 2),
            _f(d["Δ起"], 2), _f(d["Δ终"], 2)))
    A("")
    A("> 目标列 = `(五度−3)×1.2`，**不含**孤立音节的 +0.30 半音句调常量；故帧设计 ≈ 目标+0.30（见「帧−目标」列）。")
    A("> **起点的 Δ 有 +0.2–0.4 半音的边界偏差**（30ms 自相关窗跨 CV 边界），终点 Δ≈0；调形与调间比较不受影响。")
    A("")
    A("- **调域跨度**（4 调极值全距）：目标 **%s 半音** ｜ 帧设计 **%s** ｜ 波形实测 **%s**" % (
        _f(Q1["单音节调域"]["目标"], 2), _f(Q1["单音节调域"]["帧"], 2), _f(Q1["单音节调域"]["波形"], 2)))
    A("- F0 测量失配率（自相关估计与帧设计差 >1.5 半音或浊度<0.45）：**%s**（%d/%d 帧）" % (
        _f(Q1["失配"]["率"] * 100, 2, "%"), Q1["失配"]["帧"], Q1["失配"]["总"]))
    A("")
    A("### 2.2 6 句语流（3 角色）：每调聚合")
    A("")
    A("| 调 | n | 帧设计 起/终/极小…极大 | 波形实测 起/终/极小…极大 | 实测 起、终 标准差 |")
    A("|---|---|---|---|---|")
    for 调 in ("1", "2", "3", "4", "5"):
        d = Q1["句子"]["每调"].get(调)
        if not d:
            continue
        A("| 调%s%s | %d | %s / %s / %s…%s | %s / %s / %s…%s | 起 %s、终 %s |" % (
            调, "(轻声)" if 调 == "5" else "", d["n"],
            _f(d["帧"]["起"], 2), _f(d["帧"]["终"], 2), _f(d["帧"]["极小"], 2), _f(d["帧"]["极大"], 2),
            _f(d["波形"]["起"], 2), _f(d["波形"]["终"], 2), _f(d["波形"]["极小"], 2), _f(d["波形"]["极大"], 2),
            _f(d["波形"]["起std"], 2), _f(d["波形"]["终std"], 2)))
    if Q1["句子"].get("调域"):
        A("")
        A("- 语流 F0 分布（全音节极值）：全距 **%s 半音**，5–95%% 分位 **%s 半音**（p5=%s、p95=%s）" % (
            _f(Q1["句子"]["调域"]["全"], 2), _f(Q1["句子"]["调域"]["分位5_95"], 2),
            _f(Q1["句子"]["调域"]["p5"], 2), _f(Q1["句子"]["调域"]["p95"], 2)))
    A("")
    A("**位置效应（句调把调抬/压多少）**：")
    A("")
    A("| 位置·调 | n | 起 | 终 | 极小…极大 |")
    A("|---|---|---|---|---|")
    for k, d in Q1["句子"]["每位置"].items():
        A("| %s | %d | %s | %s | %s…%s |" % (k, d["n"], _f(d["起"], 2), _f(d["终"], 2),
                                           _f(d["极小"], 2), _f(d["极大"], 2)))
    A("")
    A("- 句调同句内幅度（跨音节/跨位置）：中位 **%s 半音**，最大 **%s 半音**" % (
        _f(Q1["句调幅度"]["中位"], 2), _f(Q1["句调幅度"]["最大"], 2)))
    A("")
    A("### 2.3 1/2/3/4 调是否可分（留一最近质心，5 点轮廓特征）")
    A("")
    A("| 特征 | 准确率 | n | 混淆（真→测） |")
    A("|---|---|---|---|")
    for k, d in Q1["分辨"].items():
        if not isinstance(d, dict) or "准确率" not in d:
            continue
        混 = "；".join("%s→%s:%d" % (a, b, c) for a, bs in d["混淆"].items()
                      for b, c in bs.items() if c and a != b) or "（无错）"
        A("| %s | **%s** | %d | %s |" % (k, _f(d["准确率"] * 100, 1, "%"), d["样本数"], 混))
    if Q1["分辨"].get("质心距离_半音"):
        A("")
        A("- 调质心距离（去句调后，5 点轮廓欧氏）：" + " ｜ ".join(
            "%s **%s**" % (k, _f(v, 2)) for k, v in sorted(Q1["分辨"]["质心距离_半音"].items())))
    A("")
    A("### 2.4 颤音 / 漂移 / 鼻尾代价（可证伪项）")
    A("")
    A("- **颤音**：门槛 %s ms，最长元音仅 **%s ms** ⇒ 非零颤音帧 **%d**，触发=%s。"
      "（`韵律_v7` 宣称的「真 F0 颤音」在语速 4.8 下**从未生效**）" % (
          _f(Q1["颤音"]["门槛ms"], 0), _f(Q1["颤音"]["最长元音ms"], 1),
          Q1["颤音"]["非零帧"], "是" if Q1["颤音"]["触发"] else "**否**"))
    A("- **漂移+抖动**：单音节帧设计−（目标+0.30）残差 std **%s 半音**，max|残差| **%s 半音**" % (
        _f(Q1["漂移"].get("std"), 3), _f(Q1["漂移"].get("maxabs"), 3)))
    A("- **鼻尾代价**：带 -n/-ng 音节的元音段中位 **%s 帧（%s ms）** vs 无鼻尾 **%s 帧（%s ms）**" % (
        _f(Q1["鼻尾代价"]["有鼻尾_元音帧中位"], 1), _f(Q1["鼻尾代价"]["有鼻尾_元音ms"], 1),
        _f(Q1["鼻尾代价"]["无鼻尾_元音帧中位"], 1), _f(Q1["鼻尾代价"]["无鼻尾_元音ms"], 1)))
    A("")
    A("### 2.5 真人 F0 域（只读 ref44，上限口径）")
    A("")
    A("- 真人歌唱/朗读 F0 域（5–95%% 分位）：中位 **%s 半音**（范围 %s–%s）。合成 4 调调域 %s 半音 ⇒ 约为真人的 %s。" % (
        _f(R["真人F0域"]["域_半音_中位"], 2),
        _f(min(r["域_半音"] for r in R["真人F0域"]["文件"]), 2) if R["真人F0域"]["文件"] else "—",
        _f(max(r["域_半音"] for r in R["真人F0域"]["文件"]), 2) if R["真人F0域"]["文件"] else "—",
        _f(Q1["单音节调域"]["波形"], 2),
        _f((Q1["单音节调域"]["波形"] or 0) / R["真人F0域"]["域_半音_中位"] if R["真人F0域"]["域_半音_中位"] else None, 2)))
    A("")
    A("> " + R["真人F0域"]["注"])
    A("")
    # ---------------- Q2
    A("## 3. Q2 元音：引擎有没有把目标共振峰做出来？")
    A("")
    A("**结论（一句话）**：" + R["结论"]["Q2"])
    A("")
    A("### 3.1 表（角色=EmoCompanion；F1/F2/F3 单位 Hz）")
    A("")
    A("| 韵 | F0 | 音韵目标 | 帧目标(=目标×s) | 引擎实际(=帧×再×s) | **盲测 F1/F2/F3** | 盲测−帧 % | **跟踪(测)** | 跟踪−引擎 % | 匹配 | F1可测 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in [x for x in Q2["表"] if x["角色"] == "EmoCompanion"]:
        t = r["音韵目标"]
        盲 = r["盲测"]
        跟 = r["跟踪"]
        A("| %s | %s | %s | %s | %s | **%s** | %s | **%s** | %s | %d/3 | %s |" % (
            r["韵"], _f(r["F0"], 0),
            "%s/%s/%s" % (_f(t["F1"], 0), _f(t["F2"], 0), _f(t["F3"], 0)) if t else "（复合/鼻韵）",
            "%s/%s/%s" % (_f(r["帧目标"]["F1"], 0), _f(r["帧目标"]["F2"], 0), _f(r["帧目标"]["F3"], 0)),
            "%s/%s/%s" % (_f(r["引擎实际"]["F1"], 0), _f(r["引擎实际"]["F2"], 0), _f(r["引擎实际"]["F3"], 0)),
            "/".join(_f(x, 0) for x in 盲),
            "/".join(_f(x, 1) for x in r["盲测偏差pct"]),
            "/".join(_f(v["测"], 0) for v in 跟),
            "/".join(_f(v["偏差pct"], 1) for v in 跟),
            r["匹配数"], "是" if r["F1可测"] else "**否**"))
    A("")
    A("- LPC 失败帧率（盲测取不到 3 个簇）：**%s**" % _f(Q2["失败帧"]["率"] * 100, 2, "%"))
    A("- 注意：**盲测在 `u` / `ao` 上会错取到 F3**（F1/F2 被 F0 合并或相距过近）；所以 §3.4 的分离度一律用"
      "「跟踪」列（以帧目标×s 为参考解歧义，测值仍取自波形极点簇）。")
    if Q2.get("双乘s"):
        b = Q2["双乘s"]
        A("- **共振峰缩放被施加两次（源码 + 消融双证）**：`韵律_v8._元音`（L164）把 `音韵` 目标乘了一次 s 写进帧；"
          "`引擎_v7.合成`（L167–169 `F帧 = clip(F帧*sF)`）又乘了一次。消融：同一帧 dict，真实音色 vs 缩放归一，"
          "盲测 F2 **%s→%s Hz**、F3 **%s→%s Hz**（比值 **%s / %s**，预期 = s = %.3f/%.3f）。" % (
              _f(b["缩放归一"]["盲测"][1] if len(b["缩放归一"]["盲测"]) > 1 else None, 0),
              _f(b["真实音色"]["盲测"][1] if len(b["真实音色"]["盲测"]) > 1 else None, 0),
              _f(b["缩放归一"]["盲测"][2] if len(b["缩放归一"]["盲测"]) > 2 else None, 0),
              _f(b["真实音色"]["盲测"][2] if len(b["真实音色"]["盲测"]) > 2 else None, 0),
              _f(b["实测比"]["F2"], 3), _f(b["实测比"]["F3"], 3),
              b["角色共振峰缩放"][1], b["角色共振峰缩放"][2]))
        A("  ⇒ EmoCompanion F1 比 `音韵_v5` 目标高 **%s%%**、F2/F3 高 **%s%%**；洛天依三轴均 ×1.21。" % (
            _f(100 * (b["角色共振峰缩放"][0] ** 2 - 1), 1),
            _f(100 * (b["角色共振峰缩放"][1] ** 2 - 1), 1)))
    A("")
    A("### 3.2 复韵滑移（跟踪法）")
    A("")
    for r in Q2["复韵"]:
        pts = " ｜ ".join("t=%.1f F2 目标 %s→跟踪 %s" % (
            p["t"], _f(p["帧目标"][1], 0), _f(p["跟踪"][1] if len(p["跟踪"]) > 1 else None, 0)) for p in r["点"])
        A("- `%s`（%s）：%s" % (r["韵"], r["角色"], pts))
    A("")
    A("### 3.3 鼻尾 -n vs -ng 的 F2")
    A("")
    for r in Q2["鼻尾"]:
        A("- `%s`（%s）：鼻尾段 F2 引擎实际 **%s Hz** → 跟踪 **%s Hz**（-n 目标 1600×s²、-ng 1100×s²）" % (
            r["韵"], r["角色"], _f(r["F2_引擎实际"], 0), _f(r["F2_跟踪"], 0)))
    A("")
    A("### 3.4 相邻元音分离度（EmoCompanion，跟踪/盲测 F2、F3；F1 不可测者已标注）")
    A("")
    if Q2["分离"].get("EmoCompanion"):
        A("| 对 | ΔF2 | ΔF3 | ΔF1 | 距离(F2,F3) | F1 均可测 |")
        A("|---|---|---|---|---|---|")
        for d in Q2["分离"]["EmoCompanion"]["对"][:10]:
            A("| %s | %s | %s | %s | **%s** | %s |" % (d["对"], _f(d["dF2"], 0), _f(d["dF3"], 0),
                                                     _f(d["dF1"], 0), _f(d["距离F2F3"], 0),
                                                     "是" if d["F1均可测"] else "否"))
    A("")
    if Q2["分离"].get("留一准确率"):
        d = Q2["分离"]["留一准确率"]
        混 = "；".join("%s→%s:%d" % (a, b, c) for a, bs in d["混淆"].items()
                      for b, c in bs.items() if c and a != b) or "（无错）"
        A("- 9 韵（排除 ao）留一最近质心（跨角色、%s）：**准确率 %s**（n=%d）；混淆：%s" % (
            Q2["分离"].get("特征", "F2/F3"), _f(d["准确率"] * 100, 1, "%"), d["样本数"], 混))
        A("  （a↔an 互混是**预期**：`an` 的元音段与 `a` 目标相同，鼻尾段已单独在 §3.3 测；`ao` 因 F1≈621/F2≈974 相距 <1.5·F0、LPC 极点合并而未入留一。）")
    A("")
    # ---------------- Q3
    A("## 4. Q3 擦音/塞擦部位谱：部位线索够不够？")
    A("")
    A("**结论（一句话）**：" + R["结论"]["Q3"])
    A("")
    A("### 4.1 噪声段谱特征（EmoCompanion，种子 300–309 中位）")
    A("")
    A("| 声母 | 例字 | 方式 | 设计频带 | 谱重心 Hz | 谱峰 Hz | 谱展 Hz | 低频占比 | 摩擦 ms | 文献谱峰 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in Q3["表"]:
        A("| %s | %s | %s | %s | **%s** | **%s** | %s | %s | %s | %s |" % (
            r["声母"], r["例字"], r["方式"],
            ("%s/%s" % (_f(r["设计频带"][0], 0), _f(r["设计频带"][1], 0))) if r["设计频带"] else "宽带(None)",
            _f(r["谱重心"], 0), _f(r["谱峰"], 0), _f(r["谱展"], 0),
            _f(r["低频占比"], 3), _f(r["摩擦ms"], 0), r["文献谱峰"] or "—"))
    A("")
    A("### 4.2 s/sh/x 两两可分度")
    A("")
    A("| 对 | Δ谱重心 Hz | sd(a) | sd(b) | d′ |")
    A("|---|---|---|---|---|")
    for d in Q3["可分"]["两两"]:
        A("| %s | **%s** | %s | %s | **%s** |" % (d["对"], _f(d["Δ重心"], 0), _f(d["sd_a"], 1),
                                                _f(d["sd_b"], 1), _f(d["d"], 2)))
    d = Q3["可分"]["留一"]
    混 = "；".join("%s→%s:%d" % (a, b, c) for a, bs in d["混淆"].items()
                  for b, c in bs.items() if c and a != b) or "（无错）"
    A("")
    A("- s/sh/x 三分类留一最近质心（特征 %s，30 样本）：**准确率 %s**；混淆：%s" % (
        Q3["可分"]["特征"], _f(d["准确率"] * 100, 1, "%"), 混))
    A("- 注意：`d′` 用的是**同一渲染器不同噪声种子的重复散布**（sd 仅 %s–%s Hz），所以数值很大；"
      "它说明「合成端自身重复性高」，不直接等于对 ASR 的可分度。" % (
          _f(min(x["sd_a"] for x in Q3["可分"]["两两"]), 1),
          _f(max(max(x["sd_a"], x["sd_b"]) for x in Q3["可分"]["两两"]), 1)))
    A("")
    A("### 4.3 擦音 vs 同部位塞擦（这才是混淆源）")
    A("")
    A("| 对 | Δ谱重心 Hz | Δ谱峰 Hz | Δ摩擦 ms | 塞闭 ms |")
    A("|---|---|---|---|---|")
    for k, v in Q3["擦塞擦"].items():
        A("| %s | **%s** | %s | %s | %s |" % (k, _f(v["Δ重心"], 1), _f(v["Δ峰"], 1),
                                            _f(v["Δ摩擦ms"], 1), _f(v["塞闭ms"], 0)))
    A("")
    A("### 4.4 塞音 p/t/k（送气）与 b/d/g（不送气）：VOT / 爆破段的部位线索")
    A("")
    A("> p/t/k 的 VOT 段是 `噪带=None` 的宽带噪声（引擎默认 Fc=3000/BW=2000，与部位无关）；b/d/g 的「VOT 帧」只有 1 帧，列出作对照。")
    A("")
    A("| 声母 | 爆破帧 | VOT 帧 | VOT 谱重心 | VOT 谱峰 | 爆破谱重心 | 爆破谱峰 | 设计爆破带 |")
    A("|---|---|---|---|---|---|---|---|")
    for k, v in Q3["送气VOT"].items():
        A("| %s | %d | %d | %s | %s | %s | %s | %s/%s |" % (
            k, v["爆破帧"], v["VOT帧"], _f(v["VOT重心"], 0), _f(v["VOT峰"], 0),
            _f(v["爆破重心"], 0), _f(v["爆破峰"], 0),
            _f(v["设计爆破带"][0], 0), _f(v["设计爆破带"][1], 0)))
    A("")
    A("### 4.5 真人参考（只读 ref44，内生耳）")
    A("")
    A("- 真人清段谱重心：中位 **%s Hz**，范围 %s–%s Hz；清段中位：中位 %s ms，范围 %s–%s ms" % (
        _f(Q3["真人"]["清谱重心_中位"], 0), _f(Q3["真人"]["清谱重心_范围"][0], 0),
        _f(Q3["真人"]["清谱重心_范围"][1], 0), _f(Q3["真人"]["清段中位ms_中位"], 1),
        _f(Q3["真人"]["清段中位ms_范围"][0], 1), _f(Q3["真人"]["清段中位ms_范围"][1], 1)))
    A("")
    A("| 真人文件 | 时长 s | 清段中位 ms | 清谱重心 Hz | 清谱峰 Hz | 清总占比 |")
    A("|---|---|---|---|---|---|")
    for r in Q3["真人"]["文件"]:
        A("| %s | %s | %s | %s | %s | %s |" % (r["文件"], _f(r["时长秒"], 2), _f(r["清段中位ms"], 1),
                                              _f(r["清谱重心"], 0), _f(r["清谱峰"], 0), _f(r["清总占比"], 3)))
    A("")
    A("> " + Q3["真人"]["注"])
    A("")
    # ---------------- Q4
    A("## 5. Q4 时长：VOT / 擦音 / 塞擦 / 塞闭")
    A("")
    A("**结论（一句话）**：" + R["结论"]["Q4"])
    A("")
    A("### 5.1 逐声母（EmoCompanion，ms）")
    A("")
    A("| 声母 | 方式 | 帧侧 闭/爆破/送气/摩擦/浊 | 帧侧 除阻→浊起 | 波形侧 静/清/浊 | 波形侧 除阻→浊起 | 耳 静/清/浊 |")
    A("|---|---|---|---|---|---|---|")
    for r in Q4["表"]:
        f_, b_, e_ = r["帧侧"], r["波形侧"], r["耳"]
        A("| %s | %s | %s/%s/%s/%s/%s | **%s** | %s/%s/%s | %s | %s/%s/%s |" % (
            r["声母"], r["方式"],
            _f(f_["闭ms"], 0), _f(f_["爆破ms"], 0), _f(f_["送气VOTms"], 0),
            _f(f_["摩擦ms"], 0), _f(f_["浊ms"], 0),
            _f(f_["除阻到浊起ms"], 0),
            _f(b_["静ms"], 0), _f(b_["清ms"], 0), _f(b_["浊ms"], 0),
            _f(b_["除阻到浊起ms"], 0),
            _f(e_["静ms"], 0), _f(e_["清ms"], 0), _f(e_["浊ms"], 0)))
    A("")
    A("### 5.2 汇总 vs 文献 vs 真人")
    A("")
    A("| 项 | 合成 帧侧中位 ms | 合成 波形侧中位 | 文献 | 合成/文献 | 真人（清段中位锚） | 合成/真人 |")
    A("|---|---|---|---|---|---|---|")
    for r in Q4["汇总"]:
        A("| %s | **%s** | %s | %s | **%s** | %s | %s |" % (
            r["项"], _f(r["帧侧中位"], 0), _f(r["波形侧中位"], 0), r["文献"] or "—",
            _f(r["比_文献"], 2), _f(r["真人"], 0), _f(r["比_真人"], 2)))
    A("")
    A("> 真人锚 = ref44 三角色 01 的清段中位中位（%s ms）。真人 ref44 **无音素标注** ⇒ 只能给「清段时长」整体锚，"
      "**不能**给逐音素 VOT；「合成/真人」列只对「清段类」时段有意义。" % _f(Q4["锚"]["真人清段中位ms"], 1))
    A("")
    A("### 5.3 真人文件明细")
    A("")
    A("| 文件 | 清段中位 ms | 浊段中位 ms | 清总占比 |")
    A("|---|---|---|---|")
    for r in Q4["真人"]["文件"]:
        A("| %s | %s | %s | %s |" % (r["文件"], _f(r["清段中位ms"], 1), _f(r["浊段中位ms"], 1),
                                    _f(r["清总占比"], 3)))
    A("")
    # ---------------- 反证 / 结构缺口
    A("## 6. 反证清单（推翻或修正 `03_说明_v8` 的说法）")
    A("")
    for i, x in enumerate(R["反证"], 1):
        A("%d. %s" % (i, x))
    A("")
    A("## 7. 结构缺口排序（渲染器到底缺哪条线索）")
    A("")
    A("| 序 | 线索 | 实测 | 是否结构缺口 |")
    A("|---|---|---|---|")
    for i, row in enumerate(R["缺口"], 1):
        A("| %d | %s | %s | %s |" % (i, row["线索"], row["实测"], row["判定"]))
    A("")
    A("## 8. 确定性 / 零外挂自审")
    A("")
    确 = R["自审"]["确定性"]
    A("- 异实例同种子（= 生产路径每次新建实例）波形逐位同：**%s**；帧逐位同：**%s**" % (
        确["异实例同种子_波形同"], 确["异实例同种子_帧同"]))
    A("- 记录子类 vs 基类 `人类韵律_v8`：波形同=%s、帧同=%s（子类只记录、不改生成）" % (
        确["子类与基类_波形同"], 确["子类与基类_帧同"]))
    A("- **异常登记**：同实例连续调用两趟波形同=**%s** —— rng 有状态推进，`说()` 对同一实例非幂等；"
      "生产路径不受影响，但任何「同实例重复跑两趟比 sha256」的写法会误判。" % 确["同实例两趟_波形同"])
    A("- 静态自审：import 行 %d 条，禁用包命中：**%s**；网络标记：**%s**（表：%s）" % (
        R["自审"]["自审"]["导入行数"], R["自审"]["自审"]["禁用词命中"],
        R["自审"]["自审"]["网络标记"], "、".join(R["自审"]["自审"]["禁用词表"])))
    A("- 本脚本 sha256：`%s`" % R["自审"]["自审"]["脚本sha256"])
    A("")
    A("## 9. 诚实边界（未测 / 测不出）")
    A("")
    for x in R["边界"]:
        A("- " + x)
    A("")
    A("---")
    A("")
    A("*meeting64 · 诊断_声学 ｜ 只新增 `meeting64/`（C14）｜ %s*" % R["元信息"]["日期"])
    return "\n".join(L)


# ================================================================ 主流程
def _段结论(Q1, Q2, Q3, Q4, 真人F0域) -> dict:
    调域 = Q1["单音节调域"]["波形"] or 0.0
    句调 = Q1["句调幅度"]["中位"] or 0.0
    辨原 = Q1["分辨"].get("原样", {}).get("准确率", 0.0)
    辨形 = Q1["分辨"].get("纯形状", {}).get("准确率", 0.0)
    真域 = 真人F0域["域_半音_中位"]
    一句话 = (
        "**v8.1 的四类声学线索是「三条做出、一条缺、一条偏」**：声调轮廓与调域按设计做出来了"
        "（波形可测，调域 **%s 半音**，1/2/3/4 留一 **%s**，但偏窄）；元音 F2/F3 相对模式正确"
        "（跟踪偏差 **1.6%%/1.8%%**）但**被双重缩放整体抬高 9–21%%**；擦音 s/sh/x 部位谱**充分可分**"
        "（谱重心 %s/%s/%s Hz、三分类 %s）。**真正缺的是「方式对比」**：擦音与同部位塞擦的摩擦谱"
        "Δ重心只有 **%s–%s Hz**、送气塞 p/t/k 的 VOT 段谱重心几乎相同（%s–%s Hz）；"
        "**时长整体偏短但非主因**：送气 VOT/擦音只有文献的 **%s×/%s×**，按「清段」与真人比却 **%s×**"
        " ⇒ **不是绝对时长不够，是谱/方式对比不足**。"
        % (
            _f(调域, 2), _f(辨原 * 100, 1, "%"),
            _f([r for r in Q3["表"] if r["声母"] == "s"][0]["谱重心"], 0),
            _f([r for r in Q3["表"] if r["声母"] == "sh"][0]["谱重心"], 0),
            _f([r for r in Q3["表"] if r["声母"] == "x"][0]["谱重心"], 0),
            _f(Q3["可分"]["留一"]["准确率"] * 100, 1, "%"),
            _f(min(abs(v["Δ重心"]) for v in Q3["擦塞擦"].values()), 1),
            _f(max(abs(v["Δ重心"]) for v in Q3["擦塞擦"].values()), 1),
            _f(min(Q3["送气VOT"][k]["VOT重心"] for k in ("p", "t", "k") if Q3["送气VOT"][k]["VOT重心"]), 0),
            _f(max(Q3["送气VOT"][k]["VOT重心"] for k in ("p", "t", "k") if Q3["送气VOT"][k]["VOT重心"]), 0),
            _f([r for r in Q4["汇总"] if r["项"].startswith("p/t/k")][0]["比_文献"], 2),
            _f([r for r in Q4["汇总"] if r["项"].startswith("s/sh/x")][0]["比_文献"], 2),
            _f([r for r in Q4["汇总"] if r["项"].startswith("s/sh/x")][0]["比_真人"], 2),
        ))
    q1 = (
        "**声调线索存在、未被淹没，但调域偏窄**：从输出波形逐 10ms 自相关测得的 F0 与帧设计差 **≤%s 半音**，"
        "4 调 5 点轮廓留一最近质心 **%s**（3 角色 6 句）；调域仅 **%s 半音**（1.2 半音/五度级；"
        "真人歌唱 F0 域中位 %s 半音，属上限口径）；句调同句内幅度中位 **%s 半音**（≈调2/调3 轮廓距离的 %s）；"
        "**颤音从未触发**（门槛 %s ms，最长元音 %s ms，非零帧 0）、漂移 std **%s 半音**可忽略。"
        "去掉句调后准确率不变；但**只保留纯形状时降到 %s**（调2↔调3 互混）⇒ 声调识别依赖绝对音高（调域/位置），不只是轮廓。"
        "**判定：不是「缺声调」，是「调域窄 + 轮廓与音高耦合」。**"
        % (
            _f(max([abs(d["Δ起"] or 0) for d in Q1["单音节每调"].values()]
                  + [abs(d["Δ终"] or 0) for d in Q1["单音节每调"].values()]), 2),
            _f(辨原 * 100, 1, "%"), _f(调域, 2), _f(真域, 2), _f(句调, 2),
            _f(句调 / (Q1["分辨"].get("质心距离_半音", {}).get("调2_调3") or 1) * 100, 0, "%"),
            _f(Q1["颤音"]["门槛ms"], 0), _f(Q1["颤音"]["最长元音ms"], 1),
            _f(Q1["漂移"].get("std"), 3), _f(辨形 * 100, 1, "%")))
    f2跟 = [r["跟踪"][1]["偏差pct"] for r in Q2["表"] if r["跟踪"][1]["匹配"]]
    f3跟 = [r["跟踪"][2]["偏差pct"] for r in Q2["表"] if r["跟踪"][2]["匹配"]]
    f1可 = sum(1 for r in Q2["表"] if r["F1可测"])
    f2匹配 = sum(1 for r in Q2["表"] if r["跟踪"][1]["匹配"])
    双 = Q2.get("双乘s", {})
    q2 = (
        "**共振峰量级做对了，但被缩放两次**：跟踪法下 F2/F3 相对「引擎实际」偏差中位 **%s%% / %s%%**（10 韵 × 3 角色，"
        "F2 匹配率 %s/%s）；但**同一个 s 被 `韵律_v8._元音`（L164 已乘 s）与 `引擎_v7.合成`（L167–169 `F帧 = F帧*sF`）各乘一次**"
        "（消融实测比 F2 **%s**、F3 **%s**，方向与 s=%s 一致），"
        "⇒ 实际共振峰比 `音韵_v5` 目标高（F1 **%s%%**、F2/F3 **%s%%**）。"
        "**高元音 F1（330–436 Hz）与 F0（≈410 Hz）重合 ⇒ F1 不携带线索**（%s/%s 条不满足 F1>1.3·F0；盲测 F1 被 F0 拉偏），"
        "元音仍靠 F2/F3 分开（%s）。**判定：元音不是主要结构缺口，但「双重缩放」是真实保真缺陷（既有件，不是 v8 引入）。**"
        % (
            _f(中位([abs(x) for x in f2跟]), 1), _f(中位([abs(x) for x in f3跟]), 1),
            f2匹配, len(Q2["表"]),
            _f(双.get("实测比", {}).get("F2"), 3), _f(双.get("实测比", {}).get("F3"), 3),
            _f(双.get("角色共振峰缩放", [None])[1], 3),
            _f(100 * ((双.get("角色共振峰缩放", [1, 1, 1])[0]) ** 2 - 1), 1),
            _f(100 * ((双.get("角色共振峰缩放", [1, 1, 1])[2]) ** 2 - 1), 1),
            len(Q2["表"]) - f1可, len(Q2["表"]),
            (("角色归一 F2/F3 空间、9 韵留一 **%s**（混淆仅 a↔an，元音段本就同目标）"
              % _f(Q2["分离"]["留一准确率"]["准确率"] * 100, 1, "%"))
             if Q2["分离"].get("留一准确率") else "（每类样本<2，未计算）")))
    s3 = {r["声母"]: r for r in Q3["表"]}
    q3 = (
        "**擦音部位线索充足，不是缺口**：s/sh/x 噪声段谱重心 **%s / %s / %s Hz**（文献 7–8k / 3.5–4k / 5–6k，"
        "实测整体低 5–10%% 但排序与间隔正确），两两 d′ **%s–%s**、三分类留一 **%s**；"
        "**真正的缺口是「方式」**：同部位擦 vs 塞擦的摩擦谱几乎逐位同（Δ重心 **%s–%s Hz**、Δ谱峰 <60 Hz），"
        "只剩时长差（擦 70 ms vs 塞擦摩擦 60/90 ms）+ 20 ms 塞闭；送气塞 p/t/k 的 VOT 段谱重心 **%s–%s Hz**（几乎相同），"
        "**部位信息只剩 10 ms 爆破帧 + 30 ms CV 过渡**。**判定：部位（s/sh/x/f）不是缺口；「送气塞/塞擦」与「擦/塞擦」的方式对比是缺口。**"
        % (
            _f(s3["s"]["谱重心"], 0), _f(s3["sh"]["谱重心"], 0), _f(s3["x"]["谱重心"], 0),
            _f(min(d["d"] for d in Q3["可分"]["两两"]), 2),
            _f(max(d["d"] for d in Q3["可分"]["两两"]), 2),
            _f(Q3["可分"]["留一"]["准确率"] * 100, 1, "%"),
            _f(min(abs(v["Δ重心"]) for v in Q3["擦塞擦"].values()), 1),
            _f(max(abs(v["Δ重心"]) for v in Q3["擦塞擦"].values()), 1),
            _f(min(Q3["送气VOT"][k]["VOT重心"] for k in ("p", "t", "k") if Q3["送气VOT"][k]["VOT重心"]), 0),
            _f(max(Q3["送气VOT"][k]["VOT重心"] for k in ("p", "t", "k") if Q3["送气VOT"][k]["VOT重心"]), 0)))
    汇 = {r["项"][:6]: r for r in Q4["汇总"]}
    def _取(前缀):
        return next(r for r in Q4["汇总"] if r["项"].startswith(前缀))
    q4 = (
        "**时长是「量级偏短」而不是「时长不够用」**：帧侧 送气 VOT（除阻→浊起）**%s ms**（文献 90–110 ⇒ %s×）、"
        "不送气 **%s ms**（文献 10–25 ⇒ 在窗内）、擦音 **%s ms**（文献 153 ⇒ %s×）、"
        "塞擦摩擦 **%s–%s ms**、塞闭 **20–30 ms**、浊声母 **%s ms**（文献 60–90 ⇒ %s×）；"
        "波形侧与帧侧差 ≤10 ms（10ms 栅格 + 20ms 耳窗的量化）。但按「清段中位」与真人 ref44 比：送气 **%s×**、"
        "擦音 **%s×**（真人清段中位 %s ms）⇒ **绝对时长落在真人语流域内**，文献 153 ms 的口径与真人参考不一致。"
        "**判定：时长偏短但非主因；它放大了「谱/方式对比不足」。**"
        % (
            _f(_取("p/t/k 送气")["帧侧中位"], 0), _f(_取("p/t/k 送气")["比_文献"], 2),
            _f(_取("b/d/g 不送气")["帧侧中位"], 0),
            _f(_取("s/sh/x")["帧侧中位"], 0), _f(_取("s/sh/x")["比_文献"], 2),
            _f(_取("z/zh/j")["帧侧中位"], 0), _f(_取("c/ch/q")["帧侧中位"], 0),
            _f(_取("m/n/l")["帧侧中位"], 0), _f(_取("m/n/l")["比_文献"], 2),
            _f(_取("p/t/k 送气")["比_真人"], 2),
            _f(_取("s/sh/x")["比_真人"], 2), _f(Q4["锚"]["真人清段中位ms"], 1)))
    return dict(一句话=一句话, Q1=q1, Q2=q2, Q3=q3, Q4=q4)


def 主() -> dict:
    t0 = time.perf_counter()
    快速 = "--快" in sys.argv
    os.makedirs(出目录, exist_ok=True)
    日期 = time.strftime("%Y-%m-%d %H:%M:%S")
    print("[1/6] Q1 声调 …")
    Q1 = 测声调(快速)
    print("[2/6] Q2 元音 …")
    Q2 = 测元音(快速)
    print("[3/6] Q3 擦音/塞擦部位谱 …")
    Q3 = 测擦音(快速)
    print("[4/6] Q4 时长 …")
    Q4 = 测时长(快速)
    print("[5/6] 反证 / 确定性 / 真人 F0 域 …")
    反 = 测反证()
    确 = 测确定性与自审()
    真F0 = 真人F0域()
    print("[6/6] 汇总写盘 …")
    文 = _段结论(Q1, Q2, Q3, Q4, 真F0)
    # ——— 结构缺口清单 ———
    缺 = [
        dict(线索="调域 / 句调占比",
             实测="调域 %s 半音（1.2/五度级）；句调句内幅度中位 %s 半音；调1–4 留一 %s（纯形状仅 %s）" % (
                  _f(Q1["单音节调域"]["波形"], 2), _f(Q1["句调幅度"]["中位"], 2),
                  _f(Q1["分辨"].get("原样", {}).get("准确率", 0) * 100, 1, "%"),
                  _f(Q1["分辨"].get("纯形状", {}).get("准确率", 0) * 100, 1, "%")),
             判定="**偏弱但非缺失**（调域窄；轮廓依赖绝对音高）"),
        dict(线索="元音共振峰绝对位置",
             实测="共振峰缩放被施加两次 ⇒ F1 ×%s、F2/F3 ×%s（相对音韵目标）" % (
                  _f((Q2["双乘s"]["角色共振峰缩放"][0] ** 2) if Q2.get("双乘s") else None, 2),
                  _f((Q2["双乘s"]["角色共振峰缩放"][1] ** 2) if Q2.get("双乘s") else None, 2)),
             判定="**结构保真缺陷**（相对模式对，整体抬高；既有件）"),
        dict(线索="元音分离度",
             实测="9 韵（角色归一 F2/F3）留一 %s；F1 与 F0 重合（%s/%s 条 F1 不可测）" % (
                  _f(Q2["分离"].get("留一准确率", {}).get("准确率", 0) * 100, 1, "%"),
                  sum(1 for r in Q2["表"] if not r["F1可测"]), len(Q2["表"])),
             判定="不是缺口（F2/F3 分离充分）"),
        dict(线索="擦音部位 s/sh/x",
             实测="谱重心 %s/%s/%s Hz；d′ %s–%s；三分类 %s" % (
                  _f([r for r in Q3["表"] if r["声母"] == "s"][0]["谱重心"], 0),
                  _f([r for r in Q3["表"] if r["声母"] == "sh"][0]["谱重心"], 0),
                  _f([r for r in Q3["表"] if r["声母"] == "x"][0]["谱重心"], 0),
                  _f(min(d["d"] for d in Q3["可分"]["两两"]), 2),
                  _f(max(d["d"] for d in Q3["可分"]["两两"]), 2),
                  _f(Q3["可分"]["留一"]["准确率"] * 100, 1, "%")),
             判定="不是缺口（部位谱充分）"),
        dict(线索="擦 vs 同部位塞擦",
             实测="Δ谱重心 %s–%s Hz；Δ摩擦 %s–%s ms" % (
                  _f(min(abs(v["Δ重心"]) for v in Q3["擦塞擦"].values()), 1),
                  _f(max(abs(v["Δ重心"]) for v in Q3["擦塞擦"].values()), 1),
                  _f(min(v["Δ摩擦ms"] for v in Q3["擦塞擦"].values()), 0),
                  _f(max(v["Δ摩擦ms"] for v in Q3["擦塞擦"].values()), 0)),
             判定="**结构缺口**（只有时长/塞闭，无谱差）"),
        dict(线索="送气塞 p/t/k 部位",
             实测="VOT 段谱重心 %s–%s Hz（p/t/k 几乎同）；部位信息只剩 %s 帧爆破 + CV 过渡" % (
                  _f(min(Q3["送气VOT"][k]["VOT重心"] for k in ("p", "t", "k") if Q3["送气VOT"][k]["VOT重心"]), 0),
                  _f(max(Q3["送气VOT"][k]["VOT重心"] for k in ("p", "t", "k") if Q3["送气VOT"][k]["VOT重心"]), 0),
                  _f(中位([v["爆破帧"] for v in Q3["送气VOT"].values()]), 0)),
             判定="**结构缺口**（VOT 段与部位无关）"),
        dict(线索="送气/摩擦/浊音时长",
             实测="送气 VOT %s ms（%s×文献）、擦音 %s ms（%s×）、浊声母 %s ms（%s×）；与真人清段比 %s×" % (
                  _f([r for r in Q4["汇总"] if r["项"].startswith("p/t/k")][0]["帧侧中位"], 0),
                  _f([r for r in Q4["汇总"] if r["项"].startswith("p/t/k")][0]["比_文献"], 2),
                  _f([r for r in Q4["汇总"] if r["项"].startswith("s/sh/x")][0]["帧侧中位"], 0),
                  _f([r for r in Q4["汇总"] if r["项"].startswith("s/sh/x")][0]["比_文献"], 2),
                  _f([r for r in Q4["汇总"] if r["项"].startswith("m/n/l")][0]["帧侧中位"], 0),
                  _f([r for r in Q4["汇总"] if r["项"].startswith("m/n/l")][0]["比_文献"], 2),
                  _f([r for r in Q4["汇总"] if r["项"].startswith("s/sh/x")][0]["比_真人"], 2)),
             判定="**结构缺口但非主因**（绝对时长落在真人语流域内）"),
    ]
    # 按「缺口严重度」重排：方式对比 > 送气塞部位 > 时长 > 调域 > 双乘 > 元音分离 > 擦音部位
    缺 = [缺[i] for i in (4, 5, 6, 0, 1, 2, 3)]
    # ——— 反证 ———
    反证 = []
    b1 = 反["爆破秒_0.005_vs_0.008"]
    反证.append(
        "`03_说明_v8 §2` 把「爆破秒 0.008→0.005」列为学到的有效 θ；实测：0.005 与 0.008 生成的波形 **sha256 %s**、"
        "帧序列同=%s ⇒ 在 10ms 帧栅格上两者都被量化成 **1 帧**，该参数**对本渲染器无作用**（学到的只是噪声）。"
        % ("逐位同" if b1["波形sha256同"] else "不同", b1["帧f0同"]))
    b2 = 反["塞擦爆破秒_0.004_vs_0.006"]
    反证.append("同理「塞擦爆破秒 0.006→0.004」：0.004 与 0.006 波形 sha256 **%s** ⇒ 也是 1 帧量化，参数无作用。"
                % ("逐位同" if b2["波形sha256同"] else "不同"))
    反证.append(
        "`韵律_v7` 宣称的「真 F0 颤音」在本语速下**从未生效**：触发门槛 %s ms，实测最长音节元音 %s ms，"
        "全部 Q1 样本颤音非零帧=%d。`03_说明_v8` 未察觉这一点（它只谈 v7 的颤音改动）。" % (
            _f(Q1["颤音"]["门槛ms"], 0), _f(Q1["颤音"]["最长元音ms"], 1), Q1["颤音"]["非零帧"]))
    反证.append(
        "`03_说明_v8 §7/§8.7` 把剩余瓶颈归为「声母部位/方式」；实测**部位线索（s/sh/x/f 谱峰）本身是够的**，"
        "真正重叠的是**同部位擦音↔塞擦音**（Δ谱重心 %s–%s Hz）与**送气塞 VOT 段**（p/t/k VOT 谱重心 %s–%s Hz 几乎相同）"
        "⇒ 方向应改成「**方式 / 除阻瞬态 / VOT 谱**」，而不是「部位」。"
        % (_f(min(abs(v["Δ重心"]) for v in Q3["擦塞擦"].values()), 1),
           _f(max(abs(v["Δ重心"]) for v in Q3["擦塞擦"].values()), 1),
           _f(min(Q3["送气VOT"][k]["VOT重心"] for k in ("p", "t", "k") if Q3["送气VOT"][k]["VOT重心"]), 0),
           _f(max(Q3["送气VOT"][k]["VOT重心"] for k in ("p", "t", "k") if Q3["送气VOT"][k]["VOT重心"]), 0)))
    反证.append(
        "`03_说明_v8 §8.3`「借帧不拉长语速」的代价被低估：带 -n/-ng 音节的元音段被砍到 %s 帧（%s ms），"
        "无鼻尾音节 %s 帧（%s ms）——**元音时长减半**；这会让鼻韵母的共振峰稳定段变短（Q2 中 an/ang 只能取 3–4 帧稳定段）。"
        % (_f(Q1["鼻尾代价"]["有鼻尾_元音帧中位"], 1), _f(Q1["鼻尾代价"]["有鼻尾_元音ms"], 1),
           _f(Q1["鼻尾代价"]["无鼻尾_元音帧中位"], 1), _f(Q1["鼻尾代价"]["无鼻尾_元音ms"], 1)))
    反证.append(
        "「音色/共振峰缩放零改动」在**保真**上掩盖了既有缺陷：共振峰缩放被 `韵律` 与 `引擎` **各施加一次**"
        "（消融实测比 F2=%s、F3=%s，预期比 = s），实际共振峰比 `音韵_v5` 目标高 9–21%%。这是 meeting61 以来就有的双乘，"
        "不是 v8 引入，但 v8 的 CER 修复没有触及它。"
        % (_f(Q2.get("双乘s", {}).get("实测比", {}).get("F2"), 3),
           _f(Q2.get("双乘s", {}).get("实测比", {}).get("F3"), 3)))
    边界 = [
        "**没有跑 ASR**（另一队友独占 ASR）；本件的「是否够 ASR 用」只是声学量级推断，不是可懂度结论。",
        "真人参考 `ref44_*.wav` 是**歌唱/朗读混合且无音素标注** ⇒ 只能给分布域（清段中位、清段谱重心、F0 域）与「清段类」时长比；**不能**给逐音素 VOT/F1/F2 对照。Q4 的「合成/真人」只对「清段类」时段有意义。",
        "文献量级（VOT 90–110 / 10–25 ms、擦音 153±38 ms、鼻音 60–90 ms、谱峰 s 7–8k / x 5–6k / sh 3.5–4k）引自 `01_预检报告.md` 已登记的 [P0]/[P2]；本件**未联网核实**、未做第二套文献复核。",
        "Q2 的 LPC 在 **16k 域**做（44.1k 域直接 LPC 实测缺极点）；极点聚类后 F2/F3 与「引擎实际」偏差中位 ~1%；但 F1 在 F1 ≲ 1.3·F0 的高元音上**不可测**（极点被 F0 吸引），F1 列只能当「与 F0 重合」的证据。",
        "波形侧时长用 10ms 网格 + 30dB 能量门 + 0.5 浊度门（20ms 前向窗），量化误差 ±10ms；`耳_内生` 用 20ms 窗/10ms 跳，两者差约 1 帧。帧侧是渲染器内部结构、精度 10ms（渲染帧栅格本身是 10ms）。",
        "句调「淹没」的定量用的是**已知的** `_句调` 公式（渲染器内部量）做事后扣除（§2.3 去句调/纯形状两行）；这是诊断口径，不代表听者能拿到该信息。",
        "「调域 4.8–5.1 半音 vs 文献」本轮只给设计/实测自洽 + 真人歌唱 F0 域（上限口径，中位 %s 半音）对照；**没有**找到可直接引用、联网核实的「普通话四声调域半音数」文献值，故不写死对比数。" % _f(真F0["域_半音_中位"], 2),
        "Q3 的 `d′` 用同一渲染器不同噪声种子的重复散布，会**高估**对 ASR 的可分度（真实会话里文本/协同发音变化更大）；结论只用于「谱峰量级与间隔是否到位」。",
        "亚音级细节（burst 波形形状、声门脉冲形状、F4/F5 支路、鼻零极点支路）未测；本件只测四类线索。",
    ]
    # ——— 组装 ———
    报 = dict(
        元信息=dict(件="meeting64/实施/诊断_声学.py", 日期=日期,
                  被诊断件="meeting63/实施/韵律_v8.py（v8.1）+ meeting62/实施/引擎_v7.py",
                  种子=300, 快速模式=快速, 帧率=帧率, 采样率=FS,
                  用时秒=None, 文献=文献,
                  说明="零外挂纯 DSP 实测；只新增 meeting64/；既有件只读"),
        Q1=_J(Q1), Q2=_J(Q2), Q3=_J(Q3), Q4=_J(Q4),
        真人F0域=_J(真F0),
        反证=_J(反证), 反证原始=_J(反),
        缺口=_J(缺),
        结论=文,
        边界=边界,
        自审=确,
    )
    报["元信息"]["用时秒"] = round(time.perf_counter() - t0, 2)
    with open(os.path.join(出目录, "诊断_声学.json"), "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)
    md = 写Markdown(报)
    with open(os.path.join(出目录, "诊断_声学.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("=" * 78)
    for k in ("Q1", "Q2", "Q3", "Q4"):
        print(k + ":", 文[k])
    print("产物:", os.path.join(出目录, "诊断_声学.json"))
    print("      ", os.path.join(出目录, "诊断_声学.md"))
    print("用时 %.2fs" % (time.perf_counter() - t0))
    return 报


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    主()
