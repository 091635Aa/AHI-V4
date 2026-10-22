# -*- coding: utf-8 -*-
"""meeting67 · P2-2：谐波声码器（自写、确定性、无外部模型/预训练权重；C1/C10 合规）。

输入：80 维 log-mel（10 ms 帧，22.05 kHz 口径，ln(power+1e-6)）+ 帧级 F0/浊音；
输出：22.05 kHz mono 波形（长度 = (T−1)×跳，与 Griffin-Lim 同长度约定）。

实现（预注册 `11` §2.3）：
  1. log-mel → 线性频率幅度谱估计：power = max(exp(mel)−1e-6, 0) → librosa `mel_to_stft`
     （mel 滤波器伪逆，NNLS；与 GL 声码器同一反变换，保证输入口径一致）；
  2. 浊音帧：谐波叠加。第 k 次谐波频率 = k·F0，幅度 = 该帧线性幅度包络在 k·F0 处的插值，
     再按帧能量（Σ幅度²）逐帧重标定，使谐波梳的总能量与 mel 包络帧能量一致；
     相位 = 2π·k·cum[i]（cum = Σ_{j<i} F0_j·hop/SR，固定初相 0）⇒ 帧间相位连续；
  3. 清音帧：带限噪声。幅度谱 = mel 包络幅度，相位由固定种子（种子, 帧号）确定性伪随机；
  4. 全段 STFT → librosa.istft（hann / center=True / 同 n_fft/窗长/跳长）→ 峰值归一 0.9；
  5. 确定性（C13）：同输入 + 同种子 ⇒ 同输出逐位同（sha256 对比）。

定位：本项目自研 DSP 声码器（纯 numpy/librosa 信号处理，非外部模型）；**F0/浊音由调用方给**，
本件不 import 任何 F0 提取工具（提取在 `提取_F0浊音.py`，只作离线信号）。

用法：
  & <venv python> meeting67/实施/声码器_谐波.py            # 自检 + 真实 mel 保真诊断
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M67 = HERE.parent
AHI4 = M67.parent
M66 = AHI4 / "meeting66"
出目录 = M67 / "产物" / "P2" / "音频"
自检目录 = M67 / "产物" / "P2" / "自检"
日志目录 = M67 / "团队" / "P2_实现"

# 与 meeting66 `声码器与评测.py` / `数据_SpeakerA.py` 逐参数同（mel 口径）
SR = 22050
MEL带数 = 80
N_FFT = 1024
窗长 = 551
跳长 = 220
FMIN = 0.0
FMAX = SR / 2.0
AMIN = 1e-6
MEL下限 = -40.0
MEL上限 = 40.0
目标峰值 = 0.9
F0下限 = 60.0
F0上限 = 500.0
噪声种子 = 6702
最大谐波数 = 200


def _sha256数组(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def _sha256文件(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for 块 in iter(lambda: f.read(1 << 20), b""):
            h.update(块)
    return h.hexdigest()


_LIB = {}


def librosa模块():
    if "librosa" not in _LIB:
        import librosa  # noqa: PLC0415
        _LIB["librosa"] = librosa
    return _LIB["librosa"]


def _按路径载(名: str, 路径: Path):
    if 名 in _LIB:
        return _LIB[名]
    spec = importlib.util.spec_from_file_location(名, str(路径))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _LIB[名] = mod
    return mod


def 写wav(路径, 波形, sr: int = SR) -> None:
    """22.05k/16bit/单声道（复用 meeting64 量具_v9.写wav 同一量化规则；只读引用）。"""
    量 = _按路径载("量具_v9_只读", AHI4 / "meeting64" / "实施" / "量具_v9.py")
    量.写wav(str(路径), np.asarray(波形, dtype=np.float64), fs=int(sr))


def _读回核对(路径, 期望采样率=SR) -> dict:
    with wave.open(str(路径), "rb") as w:
        return dict(声道=w.getnchannels(), 位深字节=w.getsampwidth(),
                    采样率=w.getframerate(), 帧数=w.getnframes(),
                    达标=bool(w.getnchannels() == 1 and w.getsampwidth() == 2
                              and w.getframerate() == int(期望采样率)))


def 期望长度(帧数: int, 跳: int = 跳长) -> int:
    return max(int(帧数) - 1, 0) * int(跳)


def 幅度谱估计(mel) -> np.ndarray:
    """log-mel [T,80] → 线性频率幅度估计 [513,T]（与 GL 声码器同一 mel_to_stft 反变换）。"""
    lib = librosa模块()
    m = np.asarray(mel, dtype=np.float64)
    if m.ndim != 2 or m.shape[1] != MEL带数:
        raise ValueError("mel 形状须为 [T,%d]，实际 %s" % (MEL带数, (m.shape,)))
    if m.shape[0] < 2:
        raise ValueError("mel 帧数须 ≥2，实际 %d" % m.shape[0])
    m = np.nan_to_num(m, nan=MEL下限, posinf=MEL上限, neginf=MEL下限)
    m = np.clip(m, MEL下限, MEL上限)
    功率 = np.maximum(np.exp(m) - AMIN, 0.0)
    幅度 = lib.feature.inverse.mel_to_stft(功率.T, sr=SR, n_fft=N_FFT, power=2.0,
                                          fmin=FMIN, fmax=FMAX)
    幅度 = np.maximum(np.nan_to_num(np.asarray(幅度, dtype=np.float64)), 0.0)
    return 幅度                                        # [513, T]


def _相位累计(f0: np.ndarray) -> np.ndarray:
    """帧级 F0 [T] → 累计周期数 cum[i] = Σ_{j<i} F0_j·hop/SR（rad/2π）。"""
    f = np.nan_to_num(np.asarray(f0, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    f = np.clip(f, 0.0, F0上限)
    return np.concatenate([[0.0], np.cumsum(f[:-1] * 跳长 / SR)])


def 谐波声码器(mel, f0, voicing=None, 种子: int = 噪声种子,
               峰值: float = 目标峰值) -> np.ndarray:
    """log-mel [T,80] + 帧级 F0 [T]/浊音 [T] → 波形 float32（T 同长；长度=(T−1)×220）。"""
    lib = librosa模块()
    mel = np.asarray(mel, dtype=np.float64)
    T = int(mel.shape[0])
    f0 = np.nan_to_num(np.asarray(f0, dtype=np.float64).reshape(-1), nan=0.0,
                       posinf=0.0, neginf=0.0)
    if f0.size != T:
        raise ValueError("F0 长度 %d ≠ mel 帧数 %d" % (f0.size, T))
    if voicing is None:
        浊 = f0 > 0.0
    else:
        浊 = np.asarray(voicing).reshape(-1).astype(bool) & (f0 > 0.0)
    if 浊.size != T:
        raise ValueError("voicing 长度 %d ≠ mel 帧数 %d" % (浊.size, T))
    f0 = np.where(浊, np.clip(f0, F0下限, F0上限), 0.0)

    幅度 = 幅度谱估计(mel)                             # [513, T]
    帧能量 = (幅度 ** 2).sum(axis=0)                    # [T]
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)            # [513]
    df = float(SR) / float(N_FFT)
    谱 = np.zeros((幅度.shape[0], T), dtype=np.complex128)
    cum = _相位累计(f0)

    # —— 浊音帧：谐波梳（帧能量重标定 + 连续相位） ——
    浊帧号 = np.nonzero(浊)[0]
    for i in 浊帧号:
        i = int(i)
        f = float(f0[i])
        H = int(min(最大谐波数, np.floor(FMAX / f)))
        if H < 1:
            continue
        k = np.arange(1, H + 1, dtype=np.float64)
        fk = k * f
        h幅度 = np.interp(fk, freqs, 幅度[:, i])
        能 = float((h幅度 ** 2).sum())
        if 能 <= 1e-20:
            continue
        s = np.sqrt(max(float(帧能量[i]), 0.0) / 能)
        h幅度 = h幅度 * s
        相位 = 2.0 * np.pi * k * cum[i]            # 固定初相 0
        bin号 = np.clip(np.round(fk / df).astype(np.int64), 0, 幅度.shape[0] - 1)
        np.add.at(谱[:, i], bin号, h幅度 * np.exp(1j * 相位))

    # —— 清音帧：带限噪声（谱形 = mel 包络；相位 = 固定种子确定性伪随机） ——
    清帧号 = np.nonzero(~浊)[0]
    for i in 清帧号:
        i = int(i)
        if float(帧能量[i]) <= 1e-20:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([int(种子), i]))
        ph = rng.uniform(0.0, 2.0 * np.pi, size=幅度.shape[0])
        谱[:, i] = 幅度[:, i] * np.exp(1j * ph)

    波形 = lib.istft(谱, hop_length=跳长, win_length=窗长, n_fft=N_FFT,
                     window="hann", center=True, length=期望长度(T))
    波形 = np.nan_to_num(np.asarray(波形, dtype=np.float64), nan=0.0,
                         posinf=0.0, neginf=0.0)
    if 峰值:
        现峰 = float(np.abs(波形).max()) if 波形.size else 0.0
        if 现峰 > 1e-8:
            波形 = 波形 / 现峰 * float(峰值)
    return 波形.astype(np.float32)


def 诊断_mel保真(波形: np.ndarray, mel: np.ndarray) -> dict:
    """合成波形 → 同一 mel 口径回算 → 与输入 mel 的余弦/MAE（客观包络保真自检）。"""
    lib = librosa模块()
    功率 = lib.feature.melspectrogram(
        y=np.asarray(波形, dtype=np.float64), sr=SR, n_fft=N_FFT,
        hop_length=跳长, win_length=窗长, window="hann", center=True,
        n_mels=MEL带数, fmin=FMIN, fmax=FMAX, power=2.0)
    回 = np.log(np.maximum(功率, 1e-10)).T                   # [T,80]
    真 = np.clip(np.asarray(mel, dtype=np.float64), MEL下限, MEL上限)
    n = min(回.shape[0], 真.shape[0])
    回, 真 = 回[:n], 真[:n]
    a = 回.reshape(-1)
    b = 真.reshape(-1)
    余弦 = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return dict(帧数=int(n), mel余弦=round(余弦, 4),
                mel_MAE=round(float(np.abs(a - b).mean()), 4),
                回算均值=round(float(回.mean()), 3), 输入均值=round(float(真.mean()), 3))


def 自检(保存: bool = True) -> dict:
    """随机 mel + 交替 F0 → 波形：长度/两趟 sha256/全 finite/峰值/wav 读回/mel 保真。"""
    lib = librosa模块()
    rng = np.random.default_rng(6701)
    T = 200
    mel = rng.normal(-9.0, 3.0, size=(T, MEL带数)).astype(np.float64)
    mel[:, :8] -= 6.0                                   # 低频稍强，接近语音
    f0 = np.where((np.arange(T) // 10) % 2 == 0, 220.0, 0.0)   # 浊 100ms / 清 100ms
    f0[:5] = 0.0
    voicing = (f0 > 0).astype(np.int8)

    t0 = time.time()
    w1 = 谐波声码器(mel, f0, voicing)
    耗1 = time.time() - t0
    t0 = time.time()
    w2 = 谐波声码器(mel, f0, voicing)
    耗2 = time.time() - t0
    s1, s2 = _sha256数组(w1), _sha256数组(w2)
    期望 = 期望长度(T)
    报 = dict(
        方法="谐波声码器（浊音谐波梳 + 清音带限噪声；mel_to_stft 反变换；istft hann/center）",
        输入=dict(帧数=T, mel带数=MEL带数, f0模式="浊100ms/清100ms 交替（220Hz）",
                 浊音帧=int(voicing.sum())),
        波形=dict(样本数=int(w1.size), 期望样本数=期望,
                 长度正确=bool(w1.size == 期望),
                 时长秒=round(float(w1.size) / SR, 3),
                 峰值=round(float(np.abs(w1).max()) if w1.size else 0.0, 4),
                 全finite=bool(np.isfinite(w1).all())),
        两趟=dict(sha256_1=s1, sha256_2=s2, 两趟逐位同=bool(s1 == s2),
                 耗时秒=dict(首趟=round(耗1, 3), 次趟=round(耗2, 3))),
        mel保真=诊断_mel保真(w1, mel),
    )
    # 边界：全清音 / 4 帧 / 全静音 mel
    边 = {}
    try:
        w清 = 谐波声码器(np.full((60, MEL带数), -9.0), np.zeros(60))
        边["全清音"] = dict(样本数=int(w清.size), 全finite=bool(np.isfinite(w清).all()),
                           峰值=round(float(np.abs(w清).max()), 4))
    except Exception as e:  # noqa: BLE001
        边["全清音"] = dict(错误="%s: %s" % (type(e).__name__, str(e)[:120]))
    try:
        w4 = 谐波声码器(np.full((4, MEL带数), -9.0), np.array([0.0, 200.0, 0.0, 300.0]))
        边["4帧"] = dict(样本数=int(w4.size), 期望=期望长度(4),
                        长度正确=bool(w4.size == 期望长度(4)))
    except Exception as e:  # noqa: BLE001
        边["4帧"] = dict(错误="%s: %s" % (type(e).__name__, str(e)[:120]))
    try:
        w静 = 谐波声码器(np.full((40, MEL带数), -40.0), np.zeros(40))
        边["全静音mel"] = dict(样本数=int(w静.size), 全finite=bool(np.isfinite(w静).all()),
                             峰值=round(float(np.abs(w静).max()), 4))
    except Exception as e:  # noqa: BLE001
        边["全静音mel"] = dict(错误="%s: %s" % (type(e).__name__, str(e)[:120]))
    报["边界"] = 边
    报["确定性口径"] = dict(种子=噪声种子, 相位="浊音按 F0 累积（固定初相 0）；清音按 (种子,帧号) 确定性伪随机")

    if 保存:
        自检目录.mkdir(parents=True, exist_ok=True)
        路 = 自检目录 / "自检_谐波声码器.wav"
        写wav(路, w1)
        报["wav"] = dict(路径=str(路), sha256=_sha256文件(路), 读回=_读回核对(路))
    报["PASS"] = bool(报["波形"]["长度正确"] and 报["两趟"]["两趟逐位同"]
                      and 报["波形"]["全finite"] and (报.get("wav") or {}).get("读回", {}).get("达标", True)
                      and 报["mel保真"]["mel余弦"] > 0.5
                      and 边.get("全清音", {}).get("全finite", False)
                      and 边.get("4帧", {}).get("长度正确", False))
    return _json可写(报)


def _json可写(o):
    if isinstance(o, dict):
        return {str(k): _json可写(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json可写(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def 真实mel保真(样本id: int = 75) -> dict:
    """真实SpeakerA样本：真 mel + 真 F0（离线提取产物）→ 谐波声码器 → mel 回算保真（不出 ASR）。"""
    f0路 = M67 / "产物" / "P2" / "f0" / ("%04d.npz" % int(样本id))
    样本路 = M66 / "产物" / "数据_对齐asr" / "样本" / ("%04d.npz" % int(样本id))
    if not f0路.exists() or not 样本路.exists():
        return dict(错误="缺 f0 或 mel 产物（先跑 提取_F0浊音.py）", 样本=int(样本id))
    z = np.load(str(样本路))
    mel = np.asarray(z["mel"], dtype=np.float64)
    f = np.load(str(f0路))
    f0 = np.asarray(f["f0"], dtype=np.float64)
    w = 谐波声码器(mel, f0)
    报 = dict(样本=int(样本id), 帧数=int(mel.shape[0]),
              谐波帧=int((f0 > 0).sum()), 清音帧=int((f0 <= 0).sum()),
              波形样本数=int(w.size), 期望样本数=期望长度(mel.shape[0]),
              长度正确=bool(w.size == 期望长度(mel.shape[0])),
              峰值=round(float(np.abs(w).max()), 4),
              全finite=bool(np.isfinite(w).all()),
              两趟逐位同=bool(_sha256数组(w) == _sha256数组(谐波声码器(mel, f0))),
              波形sha256=_sha256数组(w),
              mel保真=诊断_mel保真(w, np.asarray(z["mel"], dtype=np.float64)))
    return _json可写(报)


def 主(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    自检目录.mkdir(parents=True, exist_ok=True)
    日志目录.mkdir(parents=True, exist_ok=True)
    报 = dict(工具="声码器_谐波.py", 版本="1.0", 任务="meeting67 P2-2 谐波声码器自检",
              时间=time.strftime("%Y-%m-%d %H:%M:%S"),
              脚本sha256=_sha256文件(Path(__file__).resolve()),
              参数=dict(SR=SR, MEL带数=MEL带数, N_FFT=N_FFT, 窗长=窗长, 跳长=跳长,
                       AMIN=AMIN, 目标峰值=目标峰值, F0范围=[F0下限, F0上限],
                       噪声种子=噪声种子, 最大谐波数=最大谐波数),
              自检=自检(保存=True))
    报["真实mel保真_0075"] = 真实mel保真(75)
    (自检目录 / "自检_谐波声码器.json").write_text(
        json.dumps(报, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(报, ensure_ascii=False, indent=2), flush=True)
    print("[谐波声码器] PASS=%s 产物=%s"
          % (报["自检"]["PASS"], 自检目录 / "自检_谐波声码器.json"), flush=True)
    return 0 if 报["自检"]["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(主())
