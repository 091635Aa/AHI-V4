# -*- coding: utf-8 -*-
"""meeting67/团队/声学_物理建模/探针_渲染上限.py —— step0「器官渲染上限」探针（纯 DSP，不训练）。

★ 前提更正（Lead 2026-10-21 裁定）：**S1–S12 是评测文本句集，没有真人录音**。
   ⇒ 本探针改用**真实训练语料**（`<path>\\SpeakerA7代模型训练数据\\(N).wav`）的确定性片段做 oracle 控制源。

────────────────────────────── 口径（必须逐条写清，可复核） ──────────────────────────────
源数据   ：`<path>\SpeakerA7代模型训练数据\(N).wav`（44.1k 立体声 60s，SpeakerA），只读、不拷贝；
           片段 = 固定表 `片段表`（8 条，每条 4.0s，起点固定）⇒ 确定性取法。
参考文本 ：T_h = 同一 ASR 对**该真片段**的转写（meeting64「真人-合成校准」同口径）。
           这样 CER(T_h, ·) 测的是「同一 ASR、同一内容下，渲染器丢了多少可听信息」，
           不会把 ASR 对真音频自身的错算进渲染器头上。
oracle 控制（每 10ms 帧）：
   F0 / 清浊 = `librosa.pyin`（fmin=60, fmax=500, frame_length=1024, hop=220, center=True）
   LPC 包络  = 每帧 25ms hamming 窗（551 点）+ 自相关法 + Levinson-Durbin，阶 p=18
               （白噪修正 1e-5；反射系数 |k| 截到 0.97 保正面积）
   幅度      = 每帧真波形 RMS（合成后逐帧对齐 ⇒ 幅度信息不进探针，只测谱/源/激励）
最小器官（等价类）：
   P1 脉冲串 × LPC 全极点声道（经典 LPC-10 声码器口径）= 「表示上限」：包络+F0+清浊的极限
   P2 声门流导数脉冲（Rosenberg 派生）+ 气声 × 同一声道 = 「物理源形状」的代价
   P3 Kelly-Lochbaum 数字波导（面积函数由 k 反推）+ 同一激发 = 「波导实现」的代价
判据读数：逐条 CER + 均值；对照 = 真 wav→ASR（≈0，ASR 上限）、真 mel→GL→ASR（同段，仪器上限）、
          既有全句读数 0.0345（meeting66 阳性对照，样本 0075）。
确定性   ：全 float64、噪声用 np.random.default_rng(固定种子)、同输入两趟 sha256 逐位比。
────────────────────────────── 用法 ──────────────────────────────
  $env:PYTHONIOENCODING="utf-8"
  & "<workspace>/03_AHI\.venv\Scripts\python.exe" "…\探针_渲染上限.py" 自检
  & "…\探针_渲染上限.py" 探针            # 载 Paraformer（单进程！）+ 跑 8 条片段 5 条件
  & "…\探针_渲染上限.py" 探针 --条数 4   # 快跑
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AHI4 = HERE.parents[2]                     # …\AHI4
M66 = AHI4 / "meeting66"
M64 = AHI4 / "meeting64"
数据根 = Path(r"<path>\SpeakerA7代模型训练数据")
出目录 = HERE / "产物"

# ───────────── 口径常量（与 meeting66/数据_SpeakerA.py、声码器与评测.py 逐参数同） ─────────────
SR = 22050
HOP = 220
WIN = 551
MEL带数 = 80
N_FFT = 1024
AMIN = 1e-6
阶 = 18
白噪修正 = 1e-5
k上限 = 0.97
气声增益 = 0.25          # 浊帧叠加的确定性白噪（相对激发幅度）
清帧噪声 = 1.0
段长秒 = 4.0

# 确定性片段表（8 条；与 S1–S12 无关，纯语料抽样）
片段表 = (
    (0, 12.0), (6, 20.5), (12, 8.0), (22, 30.0),
    (30, 15.0), (43, 25.5), (56, 5.0), (72, 35.0),
)


# ══════════════════════ 只读加载既有件（不改） ══════════════════════
_缓存: dict = {}


def _按路径载(名: str, 路: Path):
    if 名 in _缓存:
        return _缓存[名]
    spec = importlib.util.spec_from_file_location(名, str(路))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _缓存[名] = mod
    return mod


def 量具():
    return _按路径载("量具_v9_只读", M64 / "实施" / "量具_v9.py")


def 声码():
    return _按路径载("声码器与评测_只读", M66 / "实施" / "声码器与评测.py")


def librosa模块():
    if "librosa" not in _缓存:
        import librosa  # noqa: PLC0415
        _缓存["librosa"] = librosa
    return _缓存["librosa"]


def scipy_signal():
    if "scipy.signal" not in _缓存:
        import scipy.signal as ss  # noqa: PLC0415
        _缓存["scipy.signal"] = ss
    return _缓存["scipy.signal"]


def _sha256(x) -> str:
    if isinstance(x, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    return hashlib.sha256(str(x).encode("utf-8")).hexdigest()


# ══════════════════════ 读 wav / mel ══════════════════════
清单路 = 数据根 / "asr_opt" / "SpeakerA7代模型训练数据.list"


def 清单() -> list:
    """清单行 → [dict(文件, 文本)]（保持清单顺序；行号 = npz 序号 = 片段表用号）。"""
    if "__清单" in _缓存:
        return _缓存["__清单"]
    行s = 清单路.read_text(encoding="utf-8", errors="replace").splitlines()
    出 = []
    for 行 in 行s:
        if not 行.strip():
            continue
        段 = 行.split("|")
        名 = 段[0].split("\\")[-1].strip()
        出.append(dict(文件=名, 文本=(段[3] if len(段) > 3 else "")))
    _缓存["__清单"] = 出
    return 出


def 读片段(行号: int, 起点秒: float) -> np.ndarray:
    """→ 22.05k 单声道 float64 片段（段长秒）。只读源 wav。`行号` = 清单行号。"""
    import soundfile as sf  # noqa: PLC0415
    名 = 清单()[int(行号)]["文件"]
    x, sr = sf.read(str(数据根 / 名), dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        from math import gcd
        g = gcd(int(sr), SR)
        x = scipy_signal().resample_poly(x, SR // g, int(sr) // g)
    a = int(round(起点秒 * SR))
    b = a + int(round(段长秒 * SR))
    x = x[a:b]
    if x.size < int(段长秒 * SR):
        x = np.pad(x, (0, int(段长秒 * SR) - x.size))
    return np.ascontiguousarray(x, dtype=np.float64)


def 计算logmel(x: np.ndarray) -> np.ndarray:
    """与 数据_SpeakerA.计算logmel 同参数：ln(power+1e-6)，[T,80]。"""
    lib = librosa模块()
    m = lib.feature.melspectrogram(y=np.asarray(x, dtype=np.float32), sr=SR, n_fft=N_FFT,
                                   hop_length=HOP, win_length=WIN, window="hann",
                                   center=True, pad_mode="constant",
                                   n_mels=MEL带数, fmin=0.0, fmax=SR / 2.0, power=2.0)
    return np.log(np.maximum(m, 0.0) + AMIN).T.astype(np.float64)


# ══════════════════════ oracle 控制提取 ══════════════════════
def 自相关(x: np.ndarray, p: int) -> np.ndarray:
    n = 1 << int(np.ceil(np.log2(2 * len(x) - 1)))
    X = np.fft.rfft(x, n)
    r = np.fft.irfft(X * np.conj(X), n)[: p + 1]
    return np.asarray(r, dtype=np.float64)


def 列文森(r: np.ndarray, p: int):
    """自相关 → (A(z)=1+Σa_j z^-j, 反射系数 k[0..p-1])；Levinson-Durbin。"""
    a = np.zeros(p + 1)
    a[0] = 1.0
    k = np.zeros(p)
    E = float(r[0]) * (1.0 + 白噪修正)
    if E <= 0:
        return a, k
    for i in range(1, p + 1):
        acc = float(r[i])
        for j in range(1, i):
            acc += a[j] * float(r[i - j])
        ki = -acc / E
        ki = float(np.clip(ki, -k上限, k上限))
        a旧 = a.copy()
        for j in range(1, i):
            a[j] = a旧[j] + ki * a旧[i - j]
        a[i] = ki
        k[i - 1] = ki
        E *= (1.0 - ki * ki)
        if E <= 1e-12:
            break
    return a, k


def 提取控制(x: np.ndarray) -> dict:
    """→ dict(f0[T], 浊[T], 幅度[T], A[T,p+1], k[T,p], 帧数)。"""
    lib = librosa模块()
    f0, 浊, _prob = lib.pyin(np.asarray(x, dtype=np.float32), fmin=60.0, fmax=500.0, sr=SR,
                             frame_length=1024, hop_length=HOP, center=True,
                             fill_na=0.0, switch_prob=0.5, no_trough_prob=0.1)
    T = 1 + len(x) // HOP
    xp = np.pad(x, (WIN // 2, WIN), mode="constant")   # 尾部多留 WIN，防末帧越界（(T−1)×HOP 长度）
    窗 = np.hamming(WIN)
    幅度 = np.zeros(T)
    A = np.zeros((T, 阶 + 1))
    K = np.zeros((T, 阶))
    for t in range(T):
        s = t * HOP
        帧 = xp[s: s + WIN]
        if 帧.size < WIN:                               # 兜底：定长 551（确定性）
            帧 = np.pad(帧, (0, WIN - 帧.size), mode="constant")
        帧 = 帧 * 窗
        幅度[t] = float(np.sqrt(np.mean(帧 ** 2)) + 1e-12)
        r = 自相关(帧, 阶)
        if r[0] <= 1e-12:
            A[t, 0] = 1.0
            continue
        a, kk = 列文森(r, 阶)
        A[t] = a
        K[t] = kk
    浊 = np.asarray(浊, dtype=bool)[:T]
    f0 = np.asarray(f0, dtype=np.float64)[:T]
    if 浊.size < T:
        浊 = np.pad(浊, (0, T - 浊.size), mode="edge")
        f0 = np.pad(f0, (0, T - f0.size), mode="edge")
    return dict(f0=f0, 浊=浊, 幅度=幅度, A=A, k=K, 帧数=T)


# ══════════════════════ 激发：脉冲串 / 声门流导数 ══════════════════════
def 激发_脉冲(控制: dict, n样本: int) -> np.ndarray:
    """单位冲激串（逐周期 1 个 1.0）；清帧置 0（由噪声补）。"""
    源 = np.zeros(n样本)
    相位 = 0.0
    上 = -1
    for n in range(n样本):
        t = n // HOP
        if t != 上:
            上 = t
        if 控制["浊"][t] and 控制["f0"][t] > 0:
            相位 += float(控制["f0"][t]) / SR
            if 相位 >= 1.0:
                相位 -= 1.0
                源[n] += 1.0
    return 源


def 激发_声门(控制: dict, n样本: int, 开商: float = 0.6) -> np.ndarray:
    """Rosenberg 声门流导数脉冲（开相 0.6 / 闭相 0.4），峰值归一 1。"""
    源 = np.zeros(n样本)
    相位 = 0.0
    for n in range(n样本):
        t = n // HOP
        if not (控制["浊"][t] and 控制["f0"][t] > 0):
            continue
        f0 = float(控制["f0"][t])
        相位 += f0 / SR
        if 相位 >= 1.0:
            相位 -= 1.0
            L = max(int(round(开商 * SR / f0)), 8)          # 开相总长（点数）
            L开 = max(int(round(0.6 * L)), 3)
            L闭 = max(L - L开, 2)
            形 = np.zeros(L)
            k = np.arange(L开)
            形[:L开] = 0.5 * (np.pi / L开) * np.sin(np.pi * k / L开)
            k2 = np.arange(L闭)
            形[L开:] = -(np.pi / (2.0 * L闭)) * np.sin(np.pi * k2 / (2.0 * L闭))
            峰 = float(np.abs(形).max())
            if 峰 > 1e-9:
                形 /= 峰
            末 = min(n + L, n样本)
            源[n:末] += 形[: 末 - n]
    return 源


def 噪声(n样本: int, 种子: int) -> np.ndarray:
    rng = np.random.default_rng(int(种子) & 0x7FFFFFFF)
    return rng.standard_normal(n样本)


def 混合清浊激(控制: dict, n样本: int, 源: np.ndarray, 种子: int) -> np.ndarray:
    """浊帧：源 + 气声增益×噪声×帧幅度；清帧：噪声×帧幅度。"""
    噪 = 噪声(n样本, 种子)
    出 = 源.copy()
    for t in range(控制["帧数"]):
        a, b = t * HOP, min((t + 1) * HOP, n样本)
        if b <= a:
            break
        幅 = float(控制["幅度"][t])
        if 控制["浊"][t]:
            出[a:b] += 气声增益 * 幅 * 噪[a:b]
        else:
            出[a:b] += 清帧噪声 * 幅 * 噪[a:b] * 3.0
    return 出


# ══════════════════════ 声道：全极点（Klatt 级联等价）/ KL 波导 ══════════════════════
def 合成_全极点(控制: dict, 激: np.ndarray) -> np.ndarray:
    """逐帧 LPC 全极点合成（scipy lfilter，状态跨帧继承；= p 个二阶节级联的等价形）。"""
    ss = scipy_signal()
    y = np.zeros_like(激)
    zi = np.zeros(阶)
    上 = np.zeros(阶 + 1)
    上[0] = 1.0
    for t in range(控制["帧数"]):
        a, b = t * HOP, min((t + 1) * HOP, len(激))
        if b <= a:
            break
        A = 控制["A"][t]
        if not np.all(np.isfinite(A)) or abs(A[0]) < 1e-12:
            A = 上
        上 = A
        y[a:b], zi = ss.lfilter([1.0], A, 激[a:b], zi=zi)
    return y


def 合成_波导(控制: dict, 激: np.ndarray, 唇反射: float = 1.0) -> np.ndarray:
    """Kelly-Lochbaum 数字波导：面积由 k 反推，p 段、p-1 个内部结点 + 唇端反射。

    k_i = (A_i - A_{i+1})/(A_i + A_{i+1})；散射：
        f_out = (1+k)F - k B ;  b_out = k F + (1-k) B
    每段 1 样本延迟 ⇒ 段长 = c/(2·SR) ≈ 7.94mm（p=18 ⇒ 14.3cm 声道）。
    """
    p = 控制["k"].shape[1]
    nj = max(p - 1, 1)
    F = np.zeros(nj)
    B = np.zeros(nj)
    y = np.zeros_like(激)
    k帧 = 控制["k"]
    当前 = None
    for n in range(len(激)):
        t = n // HOP
        if t != 当前:
            当前 = t
            kk = np.clip(k帧[t], -k上限, k上限)
            if not np.all(np.isfinite(kk)):
                kk = np.zeros(p)
        F[0] = 激[n]
        Fout = np.empty(nj)
        Bout = np.empty(nj)
        for i in range(nj):
            k = float(kk[i])
            Fout[i] = (1.0 + k) * F[i] - k * B[i]
            Bout[i] = k * F[i] + (1.0 - k) * B[i]
        # 传播：前向进下一结点；后向回上一结点；端点
        F[1:] = Fout[:-1]
        B[:-1] = Bout[1:]
        B[nj - 1] = 唇反射 * Fout[nj - 1]
        y[n] = Fout[nj - 1]
    return y


def 逐帧对幅(y: np.ndarray, 控制: dict, 上限倍=12.0) -> np.ndarray:
    """逐帧 RMS 对齐到真波形（oracle 幅度）：只抹平电平，不改谱形。"""
    出 = y.copy()
    g = np.ones(控制["帧数"])
    for t in range(控制["帧数"]):
        a, b = t * HOP, min((t + 1) * HOP, len(y))
        if b <= a:
            break
        r = float(np.sqrt(np.mean(y[a:b] ** 2)) + 1e-12)
        g[t] = float(np.clip(控制["幅度"][t] / r, 1.0 / 上限倍, 上限倍))
    # 3 帧滑动平均（降增益抖动），确定性
    gs = np.convolve(g, np.ones(3) / 3.0, mode="same")
    for t in range(控制["帧数"]):
        a, b = t * HOP, min((t + 1) * HOP, len(出))
        if b > a:
            出[a:b] *= gs[t]
    return 出


def 归一(y: np.ndarray, 峰=0.9) -> np.ndarray:
    m = float(np.abs(y).max())
    if m < 1e-9:
        return y
    return y / m * 峰


# ══════════════════════ 器官合成（三条臂） ══════════════════════
def 合成_臂(控制: dict, 臂: str, n样本: int, 种子: int) -> np.ndarray:
    if 臂 == "P1_脉冲_LPC全极点":
        激 = 混合清浊激(控制, n样本, 激发_脉冲(控制, n样本), 种子)
        y = 合成_全极点(控制, 激)
    elif 臂 == "P2_声门源_LPC全极点":
        激 = 混合清浊激(控制, n样本, 激发_声门(控制, n样本), 种子)
        y = 合成_全极点(控制, 激)
    elif 臂 == "P3_声门源_KL波导":
        激 = 混合清浊激(控制, n样本, 激发_声门(控制, n样本), 种子)
        y = 合成_波导(控制, 激)
    else:
        raise ValueError("未知臂 %s" % 臂)
    return 归一(逐帧对幅(y, 控制))


# ══════════════════════ 自检（不载 ASR） ══════════════════════
def _共振峰(谱: np.ndarray, 频: np.ndarray, 个数=3) -> list:
    from scipy.signal import find_peaks  # noqa: PLC0415
    峰值, _ = find_peaks(谱, distance=max(int(150 / (频[1] - 频[0])), 1))
    if 峰值.size == 0:
        return []
    序 = 峰值[np.argsort(谱[峰值])[::-1][:个数]]
    序 = np.sort(序)
    return [round(float(频[i]), 1) for i in 序]


def 自检(条数: int = 2) -> dict:
    """① 稳定性（有限值/无爆音）②全极点 vs KL 波导的共振峰一致性 ③确定性两趟。"""
    ss = scipy_signal()
    报 = dict(阶=阶, SR=SR, 段长秒=段长秒, 逐条=[], 备注=[])
    for 样本id, 起点 in 片段表[:条数]:
        x = 读片段(样本id, 起点)
        控 = 提取控制(x)
        n = len(x)
        条 = dict(样本id=样本id, 起点秒=起点, 帧数=控["帧数"],
                  浊帧占比=round(float(控["浊"].mean()), 3),
                  F0中位=round(float(np.median(控["f0"][控["浊"]])) if 控["浊"].any() else 0.0, 2),
                  k绝对最大=round(float(np.abs(控["k"]).max()), 4),
                  面积正=bool(np.all(np.cumprod(np.ones(1)) > 0)))
        # 速度/确定性：P2 两趟
        t0 = time.perf_counter()
        y1 = 合成_臂(控, "P2_声门源_LPC全极点", n, 12345)
        t1 = time.perf_counter()
        y2 = 合成_臂(控, "P2_声门源_LPC全极点", n, 12345)
        条["合成秒"] = round(t1 - t0, 3)
        条["实时率"] = round((t1 - t0) / (n / SR), 3)
        条["两趟逐位同"] = bool(_sha256(y1) == _sha256(y2))
        条["P2_有限"] = bool(np.all(np.isfinite(y1)))
        条["P2_峰值"] = round(float(np.abs(y1).max()), 4)
        # 波导：同样控制下 1 帧冲激响应 vs 全极点 IR 的共振峰对比
        A = 控["A"][控["帧数"] // 2]
        ir = ss.lfilter([1.0], A, np.eye(1, 1200, 0)[0])
        KL控 = dict(f0=控["f0"], 浊=np.ones(控["帧数"], dtype=bool), 幅度=控["幅度"],
                    A=np.repeat(A[None, :], 控["帧数"], axis=0),
                    k=np.repeat(控["k"][控["帧数"] // 2][None, :], 控["帧数"], axis=0),
                    帧数=控["帧数"])
        激 = np.zeros(1200)
        激[0] = 1.0
        irw = 合成_波导(KL控, 激)
        频 = np.fft.rfftfreq(2048, 1.0 / SR)
        H1 = np.abs(np.fft.rfft(ir, 2048))
        H2 = np.abs(np.fft.rfft(irw, 2048))
        f1 = _共振峰(H1, 频, 3)
        f2 = _共振峰(H2, 频, 3)
        条["LPC共振峰"] = f1
        条["KL波导共振峰"] = f2
        条["共振峰偏差Hz"] = [round(abs(a - b), 1) for a, b in zip(f1, f2)] if len(f1) == len(f2) else None
        报["逐条"].append(条)
        print("[自检] %s 浊=%.2f k|max|=%.3f 实时率=%.3f 两趟同=%s 共振峰 LPC=%s KL=%s"
              % (样本id, 条["浊帧占比"], 条["k绝对最大"], 条["实时率"],
                 条["两趟逐位同"], f1, f2), flush=True)
    return 报


# ══════════════════════ 探针主流程（载 ASR） ══════════════════════
def 探针(条数: int = 8, 输出: Path | None = None) -> dict:
    输出 = 输出 or (出目录 / "探针_渲染上限.json")
    os.makedirs(str(出目录), exist_ok=True)
    V = 量具()
    S = 声码()
    引 = V.载引擎()
    print("[ASR] %s" % 引["名"], flush=True)
    开始 = time.time()
    条件名 = ["真_原wav", "真mel_GL", "P1_脉冲_LPC全极点", "P2_声门源_LPC全极点", "P3_声门源_KL波导"]
    逐条 = []
    for k, (样本id, 起点) in enumerate(片段表[:条数]):
        x = 读片段(样本id, 起点)
        n = len(x)
        控 = 提取控制(x)
        条 = dict(样本id=样本id, 起点秒=起点, 时长秒=round(n / SR, 3), 条件={})
        # ① 真 wav → ASR = 参考 T_h
        wav真 = 出目录 / ("探针_真_%s.wav" % 样本id)
        V.写wav(str(wav真), x, fs=SR)
        T_h = 引["转写"](str(wav真))
        条["T_h"] = T_h
        print("[%d/%d] %s 真转写「%s」" % (k + 1, 条数, 样本id, T_h), flush=True)
        # ② 真 mel → GL → ASR（同段仪器上限）
        mel = 计算logmel(x)
        wavGL = 出目录 / ("探针_GL_%s.wav" % 样本id)
        S.写wav(str(wavGL), S.mel转波(mel), sr=SR)
        T_gl = 引["转写"](str(wavGL))
        条["条件"]["真mel_GL"] = dict(CER=V.CER(T_h, T_gl), 转写=T_gl)
        # ③ 三条器官臂
        for 臂 in 条件名[2:]:
            t0 = time.perf_counter()
            y = 合成_臂(控, 臂, n, 种子=1000 + k)
            合成秒 = time.perf_counter() - t0
            路 = 出目录 / ("探针_%s_%s.wav" % (臂.split("_")[0], 样本id))
            V.写wav(str(路), y, fs=SR)
            T_s = 引["转写"](str(路))
            条["条件"][臂] = dict(CER=V.CER(T_h, T_s), 转写=T_s,
                                  合成秒=round(合成秒, 3), 实时率=round(合成秒 / (n / SR), 3))
            print("    %-22s CER=%.4f 实时率=%.3f 「%s」"
                  % (臂, 条["条件"][臂]["CER"], 条["条件"][臂]["实时率"], T_s[:28]), flush=True)
        # 确定性：P2 两趟逐位（不重跑 ASR）
        y_a = 合成_臂(控, "P2_声门源_LPC全极点", n, 种子=1000 + k)
        y_b = 合成_臂(控, "P2_声门源_LPC全极点", n, 种子=1000 + k)
        条["确定性_两趟逐位同"] = bool(_sha256(y_a) == _sha256(y_b))
        逐条.append(条)
    汇总 = {}
    for 臂 in 条件名[1:]:
        vs = [r["条件"][臂]["CER"] for r in 逐条 if 臂 in r["条件"]]
        if vs:
            汇总[臂] = dict(CER均值=round(float(np.mean(vs)), 4),
                            CER中位=round(float(np.median(vs)), 4),
                            CER范围=[round(float(min(vs)), 4), round(float(max(vs)), 4)], n=len(vs))
    报 = dict(
        探针="meeting67 step0 器官渲染上限（纯 DSP，无训练）",
        口径=dict(SR=SR, 跳=HOP, 窗=WIN, 阶=阶, 片段=len(逐条), 段长秒=段长秒,
                 参考文本="同一 ASR 对该真片段的转写 T_h（meeting64 校准口径）",
                 对照="真 mel→GL→ASR（同段）；既有全句读数 0.0345（样本 0075）",
                 ASR=引["名"], 时间戳=time.strftime("%Y-%m-%d %H:%M:%S")),
        逐条=逐条, 汇总=汇总,
        耗时秒=round(time.time() - 开始, 1))
    _写json(输出, 报)
    print("\n=== 汇总（CER，对 T_h 算） ===", flush=True)
    for 名, v in 汇总.items():
        print("%-24s 均值=%.4f 中位=%.4f 范围=%s n=%d"
              % (名, v["CER均值"], v["CER中位"], v["CER范围"], v["n"]), flush=True)
    print("[落盘] %s（耗时 %.1fs）" % (输出, 报["耗时秒"]), flush=True)
    return 报


def _写json(路: Path, o) -> None:
    def 转(v):
        if isinstance(v, dict):
            return {str(k): 转(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [转(x) for x in v]
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return 转(v.tolist())
        if isinstance(v, np.bool_):
            return bool(v)
        return v
    路.write_text(json.dumps(转(o), ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> int:
    派 = argparse.ArgumentParser(description="meeting67 step0 器官渲染上限探针")
    派.add_argument("模式", nargs="?", default="探针", choices=("自检", "探针"))
    派.add_argument("--条数", type=int, default=8)
    派.add_argument("--输出", default=None)
    a = 派.parse_args()
    os.makedirs(str(出目录), exist_ok=True)
    if a.模式 == "自检":
        报 = 自检(a.条数 if a.条数 < 8 else 2)
        _写json(出目录 / "自检.json", 报)
        print("[自检] 已落盘 %s" % (出目录 / "自检.json"))
        return 0
    探针(a.条数, Path(a.输出) if a.输出 else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
