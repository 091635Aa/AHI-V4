# -*- coding: utf-8 -*-
"""meeting66/实施/声码器与评测.py —— T-F：log-mel → 波形（Griffin-Lim）+ 双集 ASR CER 复评。

★ 定位：本件是「生成路径的声码器 + 外部评测驱动」。**不训练、不改任何既有件**（C14）。
   · 声学模型 = `meeting66/实施/模型_学说话.py` 从零训练的非 Transformer 卷积网络
     （只读 import；ckpt = `meeting66/产物/模型/ckpt.pt`）；
   · 声码器 = **Griffin-Lim**（首版，确定性；不用神经声码器、不联网）；
   · 评测 = `meeting64/实施/量具_v9.py` 的 `载引擎 / CER / 对齐`（**同源复用**，
     ASR = Paraformer-seaco-large 只作离线仪器；单进程、引载一次复用）。

──────────────────────────────── 口径（必须与数据侧逐参数同） ────────────────────────────────
mel（`meeting66/实施/数据_SpeakerA.py::计算logmel`）：
  sr=22050 ｜ 25ms 窗（551 点）｜ 10ms 跳（220 点）｜ n_fft=1024 ｜ 80 带
  fmin=0 ｜ fmax=11025 ｜ hann(periodic) ｜ center=True pad_mode=constant ｜ power=2.0
  log 口径 = **ln(mel_power + 1e-6)**  （AMIN=1e-6）
反变换（本件 `mel转波`）：
  ① 功率 = max(exp(log-mel) − AMIN, 0)                     ← 与 log 口径互逆
  ② 幅度 = `librosa.feature.inverse.mel_to_stft(功率, sr=22050, n_fft=1024,
             power=2.0, fmin=0, fmax=11025)`                ← mel 滤波器伪逆（NNLS）
  ③ 波形 = `librosa.griffinlim(幅度, n_iter=64, hop_length=220, win_length=551,
             n_fft=1024, window='hann', center=True, pad_mode='constant',
             random_state=0)`                              ← 固定迭代 64 + 固定种子 ⇒ C13
  ④ 峰值归一化到 0.9（确定性；便于 ASR 电平稳定）
  ★ 注：librosa 0.11 起 Griffin-Lim 位于 `librosa.griffinlim`（旧的
    `librosa.effects.griffinlim` 已移除）；两者算法/参数同源，仅入口改名。
  ★ 长度：T 帧 → (T−1)×220 采样点（center=True 的 STFT/ISTFT 约定）；
    自检按 |实际 − (T−1)×220| ≤ 220 判「长度正确」。

──────────────────────────────── 双集与对照 ────────────────────────────────
历史集 S1–S6 / 留出集 S7–S12 逐字取自
`meeting64/产物/诊断/上限/_预注册.json`（预注册先于任何 A/B 写入）。
角色 = SpeakerA（训练语料即SpeakerA；模型无角色输入）。
对照（v8.1，`meeting64/产物/诊断/上限/实验_渲染上限.json` 的 c00，种子 300，量具_v9 口径）：
  历史 S1–S6 · SpeakerA = 0.5853  ｜ 历史全角色 = 0.5292
  留出 S7–S12 · 全角色 = 0.6063 ｜ 留出 · SpeakerA = 0.6893
  逐句 SpeakerA CER 见 `对照逐句`（同 JSON 的来源字段）。

──────────────────────────────── 用法 ────────────────────────────────
  $env:PYTHONIOENCODING="utf-8"
  & "<workspace>/\03_AHI\\.venv\\Scripts\\python.exe" "…\\meeting66\\实施\\声码器与评测.py" 自检
  & "…\\声码器与评测.py" 评测              # 等 ckpt 训练完（训练摘要.md 出现）再跑全流程
  & "…\\声码器与评测.py" 全部              # 自检 + 评测（默认）
  # 选项：--ckpt 路径 ｜ --每音节帧数 20 ｜ --不等待 ｜ --最多轮询 20 ｜ --轮询秒 60
  #       ｜ --输出 JSON路径 ｜ --不做阳性对照
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
import warnings
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                        # …\meeting66\实施
M66 = HERE.parent                                             # …\meeting66
AHI4 = M66.parent                                             # …\AHI4
M64 = AHI4 / "meeting64"
产物 = M66 / "产物"
音频目录 = 产物 / "音频"
判据目录 = 产物 / "判据"
默认ckpt = 产物 / "模型" / "ckpt.pt"
默认输出 = 判据目录 / "判据_学说话.json"

# ───────────────────────── mel / 声码器口径（与 数据_SpeakerA.py 逐参数同） ─────────────────────────
SR = 22050
MEL带数 = 80
N_FFT = 1024
窗长 = 551
跳长 = 220
FMIN = 0.0
FMAX = SR / 2.0
AMIN = 1e-6
每音节帧数缺省 = 20
GL迭代 = 64
GL种子 = 0
目标峰值 = 0.9
MEL下限 = -40.0
MEL上限 = 40.0

# ───────────────────────── 对照（v8.1，只读引用；不重跑） ─────────────────────────
对照 = dict(
    历史_SpeakerA=0.5853,
    历史_全角色=0.5292,
    留出_全角色=0.6063,
    留出_SpeakerA=0.6893,
    来源="meeting64/产物/诊断/上限/实验_渲染上限.json → c00"
         "（v8.1 韵律_v8 + 声线覆盖层_v7，种子 300，量具_v9 口径；"
         "历史集=c00.筛选、留出集=c00.留出；逐角色均值/逐样本 CER）",
)
对照逐句 = {
    "历史": {"S1": 0.625, "S2": 0.1538, "S3": 0.9, "S4": 0.3333, "S5": 0.5, "S6": 1.0},
    "留出": {"S7": 0.6, "S8": 0.5833, "S9": 0.8333, "S10": 0.8462,
             "S11": 0.3636, "S12": 0.9091},
}
# 对照集口径说明：上面逐句值 = c00 逐样本里「SpeakerA」单角色的 CER（与 0.5853/0.6893 均值同源）

# ───────────────────────── 句集（逐字取自 _预注册.json） ─────────────────────────
句集历史 = (
    ("S1", "今天的风很轻，适合慢慢地说一会儿话。"),
    ("S2", "我们明天早上八点在南门见面。"),
    ("S3", "他把三本书放在窗台上。"),
    ("S4", "小猫从桌上跳下来，碰到杯子。"),
    ("S5", "春天来了，山上的雪开始融化。"),
    ("S6", "请问这条路怎么走？"),
)
句集留出 = (
    ("S7", "他昨天下午去了图书馆。"),
    ("S8", "外面下着小雨，我们打伞走吧。"),
    ("S9", "请把窗户关上，房间里有点冷。"),
    ("S10", "这个周末我们一起去爬山，好吗？"),
    ("S11", "他从口袋里拿出一个本子。"),
    ("S12", "春天的花开了，风也很温柔。"),
)
角色 = "SpeakerA"
# 仪器阳性对照：最短的真实SpeakerA样本（真 mel → 本件声码器 → ASR，证明 22.05k 管线与仪器可用）
阳性对照样本 = 产物 / "数据" / "样本" / "0075.npz"

训练目录 = 产物 / "模型"
训练摘要 = 训练目录 / "训练摘要.md"


# ════════════════════════════════════════════════════════════════════════
# 0. 小工具 / 只读模块加载
# ════════════════════════════════════════════════════════════════════════
_模块缓存: dict = {}


def _按路径载(名: str, 路径: Path):
    """按文件路径只读 import（不落 __pycache__ 到既有件目录之外无副作用）。"""
    if 名 in _模块缓存:
        return _模块缓存[名]
    spec = importlib.util.spec_from_file_location(名, str(路径))
    if spec is None or spec.loader is None:
        raise ImportError("无法按路径加载 %s" % 路径)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _模块缓存[名] = mod
    return mod


def 模型模块():
    return _按路径载("模型_学说话_只读", HERE / "模型_学说话.py")


def 量具模块():
    """meeting64 量具（CER/对齐/写wav/载引擎/新run_id；与 meeting63/64 同源口径）。"""
    return _按路径载("量具_v9_只读", M64 / "实施" / "量具_v9.py")


def librosa模块():
    if "librosa" not in _模块缓存:
        import librosa  # noqa: PLC0415
        _模块缓存["librosa"] = librosa
    return _模块缓存["librosa"]


def _sha256文件(路径) -> str:
    h = hashlib.sha256()
    with open(路径, "rb") as f:
        for 块 in iter(lambda: f.read(1 << 20), b""):
            h.update(块)
    return h.hexdigest()


def _sha256数组(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def _新run_id() -> str:
    return "%s-%d-%s" % (time.strftime("%Y%m%d-%H%M%S"), os.getpid(),
                         hashlib.sha256(os.urandom(8)).hexdigest()[:4])


def _确保目录(*路径s) -> None:
    for p in 路径s:
        os.makedirs(str(p), exist_ok=True)


def _json可写(o):
    """numpy → python（JSON 落盘前统一转换）。"""
    if isinstance(o, dict):
        return {str(k): _json可写(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json可写(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return _json可写(o.tolist())
    return o


def _读回核对(路径, 期望采样率=SR) -> dict:
    """wav 落盘后读回：采样率/位深/单声道/帧数（16bit = sampwidth 2）。"""
    with wave.open(str(路径), "rb") as w:
        return dict(声道=w.getnchannels(), 位深字节=w.getsampwidth(),
                    采样率=w.getframerate(), 帧数=w.getnframes(),
                    达标=bool(w.getnchannels() == 1 and w.getsampwidth() == 2
                              and w.getframerate() == int(期望采样率)))


# ════════════════════════════════════════════════════════════════════════
# 1. 声码器：log-mel → 波形（Griffin-Lim，确定性）
# ════════════════════════════════════════════════════════════════════════
def mel转波(mel, sr: int = SR, 迭代: int = GL迭代, 种子: int = GL种子,
            峰值: float = 目标峰值) -> np.ndarray:
    """log-mel [T,80]（ln(power+1e-6)，22.05kHz 口径）→ 波形 float32。

    实现：功率 = max(exp(mel) − AMIN, 0) → librosa mel 滤波器伪逆（NNLS）→
    Griffin-Lim（固定 迭代=64、random_state=固定种子、hann/center/constant）。
    确定性（C13）：同输入 ⇒ 同输出逐位同。长度 = (T−1)×220 采样点。
    `峰值` 给定时把波形峰值归一化到该值（0 = 不归一化）。
    """
    lib = librosa模块()
    m = np.asarray(mel, dtype=np.float64)
    if m.ndim != 2 or m.shape[1] != MEL带数:
        raise ValueError("mel 形状须为 [T,%d]，实际 %s" % (MEL带数, (m.shape,)))
    if m.shape[0] == 0:
        raise ValueError("mel 帧数为 0")
    m = np.nan_to_num(m, nan=MEL下限, posinf=MEL上限, neginf=MEL下限)
    m = np.clip(m, MEL下限, MEL上限)

    功率 = np.maximum(np.exp(m) - AMIN, 0.0)                       # [T,80] 功率谱
    幅度 = lib.feature.inverse.mel_to_stft(
        功率.T, sr=int(sr), n_fft=N_FFT, power=2.0, fmin=FMIN, fmax=sr / 2.0)
    幅度 = np.maximum(np.asarray(幅度, dtype=np.float64), 0.0)      # [1+n_fft/2, T]
    入口 = getattr(lib, "griffinlim", None) or lib.effects.griffinlim
    波形 = 入口(幅度, n_iter=int(迭代), hop_length=跳长, win_length=窗长, n_fft=N_FFT,
                window="hann", center=True, pad_mode="constant",
                random_state=int(种子))
    波形 = np.asarray(波形, dtype=np.float64)
    波形 = np.nan_to_num(波形, nan=0.0, posinf=0.0, neginf=0.0)
    if 峰值 and 波形.size:
        现峰 = float(np.abs(波形).max())
        if 现峰 > 1e-8:
            波形 = 波形 / 现峰 * float(峰值)
    return 波形.astype(np.float32)


def 期望长度(帧数: int, 跳: int = 跳长) -> int:
    """T 帧 → 波形采样点数（center=True 的 ISTFT 约定：(T−1)×跳）。"""
    return max(int(帧数) - 1, 0) * int(跳)


def 写wav(路径, 波形, sr: int = SR) -> None:
    """22.05kHz / 16bit / 单声道（复用 量具_v9.写wav 同一量化规则）。"""
    量具模块().写wav(str(路径), np.asarray(波形, dtype=np.float64), fs=int(sr))


# ════════════════════════════════════════════════════════════════════════
# 2. 合成：文本 → mel → 波形 → wav
# ════════════════════════════════════════════════════════════════════════
def 载ckpt(ckpt路径=None, 设备: str = "cpu"):
    """ckpt → 模型（eval 模式；默认 CPU ⇒ 确定性 + 不占显存）。"""
    路 = Path(ckpt路径) if ckpt路径 else 默认ckpt
    if not 路.exists():
        raise FileNotFoundError("ckpt 不存在：%s" % 路)
    return 模型模块().加载(路, 设备=设备)


def 推理mel(模型, 文本: str, 每音节帧数: int = 每音节帧数缺省):
    """文本 → (mel float32[T,80], 有效音节数, 跳过的音节)。"""
    模 = 模型模块()
    mel = 模.合成mel(模型, str(文本), 每音节帧数=int(每音节帧数))
    音符 = getattr(模型, "词表", None)
    有效音节数 = None
    跳过 = []
    if isinstance(音符, dict):
        _tok, 有效, 跳 = 模.文本到token(音符, str(文本))
        有效音节数 = int(len(有效))
        跳过 = [dict(项) for 项 in 跳[:8]]
    return np.asarray(mel, dtype=np.float32), 有效音节数, 跳过


def _mel统计(mel: np.ndarray) -> dict:
    m = np.asarray(mel, dtype=np.float64)
    return dict(形状=list(m.shape), 均值=round(float(m.mean()), 4),
                标准差=round(float(m.std()), 4),
                最小=round(float(m.min()), 4), 最大=round(float(m.max()), 4))


def 合成(文本: str, ckpt=None, 模型=None, 每音节帧数: int = 每音节帧数缺省,
         写文件: bool = True, 标签: str = "合成", 保存目录=None,
         run_id: str | None = None, 序号: int = 0) -> dict:
    """文本 → mel → Griffin-Lim 波形 → `meeting66/产物/音频/<唯一名>.wav`。

    `ckpt`：ckpt 路径或已加载的模型（也可直接传 `模型=`）。
    → dict(mel, 波形, 路径, sha256, 时长秒, 音节率, 期望长度, 长度正确, ...)
    """
    run_id = run_id or _新run_id()
    保存目录 = Path(保存目录) if 保存目录 else 音频目录
    if 写文件:
        _确保目录(保存目录)
    if 模型 is None:
        模型 = 载ckpt(ckpt)
    mel, 有效音节数, 跳过 = 推理mel(模型, 文本, 每音节帧数)
    波形 = mel转波(mel)
    字 = _正规字数(文本)
    出 = dict(
        文本=str(文本), 标签=str(标签), 角色=角色,
        有效音节数=有效音节数, 跳过音节=跳过, 字数=字,
        mel=_mel统计(mel),
        mel数组sha256=_sha256数组(mel),
        波形样本数=int(波形.size),
        期望样本数=期望长度(mel.shape[0]),
        长度正确=bool(abs(int(波形.size) - 期望长度(mel.shape[0])) <= 跳长),
        时长秒=round(float(波形.size) / float(SR), 3),
        峰值=round(float(np.abs(波形).max()) if 波形.size else 0.0, 4),
        路径=None, sha256=None, 读回=None,
    )
    出["音节率"] = (round(float(字) / 出["时长秒"], 3) if 出["时长秒"] > 0 else None)
    if 写文件:
        名 = "合成_%s_%s_%02d_%s.wav" % (_安全(标签), _安全(run_id), int(序号), _安全(文本[:10]))
        路 = 保存目录 / 名
        写wav(路, 波形)
        出["路径"] = str(路)
        出["文件"] = 名
        出["sha256"] = _sha256文件(路)
        出["读回"] = _读回核对(路)
    出["波形"] = 波形
    return 出


def _正规字数(文本: str) -> int:
    """字数口径与量具 `_正规` 一致（只留汉字/字母数字）。"""
    return sum(1 for ch in str(文本)
               if ("\u4e00" <= ch <= "\u9fff") or ch.isalnum())


def _安全(s: str) -> str:
    """文件名安全化（保留中文；只替换路径分隔符/冒号等）。"""
    坏 = '\\/:*?"<>| \t\r\n'
    return "".join("_" if ch in 坏 else ch for ch in str(s))[:32]


# ════════════════════════════════════════════════════════════════════════
# 3. 自检（不需要 ASR、不占大内存）
# ════════════════════════════════════════════════════════════════════════
def 自检(模型=None, ckpt=None, 保存目录=None) -> dict:
    """随机 mel → 波形自检：长度正确 / 两趟逐位同（sha256）/ 常数/静音边界。

    不依赖 ckpt（给了模型就加测「模型合成两趟 mel 逐位同」）。
    """
    保存目录 = Path(保存目录) if 保存目录 else 音频目录
    _确保目录(保存目录)
    run = _新run_id()
    异常 = []
    rng = np.random.default_rng(66)        # 固定种子 ⇒ 自检本身可复现
    T = 200
    # 随机 log-mel：覆盖训练统计的合理范围（|Δ| ≤ 2σ），避免造出物理不可达谱
    均值 = np.full(MEL带数, -8.0)
    标准差 = np.full(MEL带数, 2.0)
    if 模型 is not None:
        阈值 = np.asarray(getattr(模型, "mel均值", 均值), dtype=np.float64).reshape(-1)
        标准差_模型 = np.asarray(getattr(模型, "mel标准差", 标准差), dtype=np.float64).reshape(-1)
        if 阈值.size == MEL带数:
            均值 = 阈值
        if 标准差_模型.size == MEL带数:
            标准差 = 标准差_模型
    mel = (均值 + 标准差 * rng.standard_normal((T, MEL带数))).astype(np.float32)

    t0 = time.perf_counter()
    w1 = mel转波(mel)
    t1 = time.perf_counter()
    w2 = mel转波(mel)                     # 第二趟（同输入）
    t2 = time.perf_counter()

    路1 = 保存目录 / ("自检_随机mel_a_%s.wav" % run)
    路2 = 保存目录 / ("自检_随机mel_b_%s.wav" % run)
    写wav(路1, w1)
    写wav(路2, w2)
    sha1, sha2 = _sha256文件(路1), _sha256文件(路2)
    期长 = 期望长度(T)
    长正确 = bool(abs(int(w1.size) - 期长) <= 跳长)
    两趟同 = bool(sha1 == sha2 and np.array_equal(w1, w2))
    读回 = _读回核对(路1)
    if not 长正确:
        异常.append("随机 mel 波形长度不符：实际 %d，期望 %d（±%d）"
                    % (w1.size, 期长, 跳长))
    if not 两趟同:
        异常.append("同输入两趟波形/sha256 不同")
    if not 读回["达标"]:
        异常.append("wav 读回格式不符：%s" % 读回)

    # 边界：全静音 mel（极小功率）与单帧 mel
    边 = {}
    try:
        w静 = mel转波(np.full((T, MEL带数), MEL下限, np.float32))
        边["全静音mel"] = dict(样本数=int(w静.size), 峰值=round(float(np.abs(w静).max()), 6),
                               长度正确=bool(abs(int(w静.size) - 期长) <= 跳长))
    except Exception as e:  # noqa: BLE001
        边["全静音mel"] = dict(错误="%s: %s" % (type(e).__name__, str(e)[:120]))
        异常.append("全静音 mel 反变换失败")
    try:
        w短 = mel转波(np.zeros((4, MEL带数), np.float32))
        边["4帧mel"] = dict(样本数=int(w短.size), 期望=int(期望长度(4)),
                            长度正确=bool(w短.size == 期望长度(4)))
    except Exception as e:  # noqa: BLE001
        边["4帧mel"] = dict(错误="%s: %s" % (type(e).__name__, str(e)[:120]))
        异常.append("4 帧 mel 反变换失败")

    # 模型侧确定性：同一句两趟 mel 逐位同（C13；仅当给了模型）
    模型两趟 = None
    if 模型 is not None:
        try:
            m1, _n1, _s1 = 推理mel(模型, "今天天气不错，我们出去走走吧。", 每音节帧数缺省)
            m2, _n2, _s2 = 推理mel(模型, "今天天气不错，我们出去走走吧。", 每音节帧数缺省)
            模型两趟 = dict(两趟mel逐位同=bool(np.array_equal(m1, m2)),
                            形状=list(m1.shape), mel标准差=round(float(m1.std()), 4))
            if not 模型两趟["两趟mel逐位同"]:
                异常.append("模型两趟推理 mel 不同（C13 不达标）")
        except Exception as e:  # noqa: BLE001
            模型两趟 = dict(错误="%s: %s" % (type(e).__name__, str(e)[:160]))
            异常.append("模型两趟推理自检失败")

    报 = dict(
        随机mel=dict(帧数=T, 形状=list(mel.shape), 均值=round(float(mel.mean()), 3),
                    标准差=round(float(mel.std()), 3)),
        波形=dict(样本数=int(w1.size), 期望样本数=int(期长), 长度正确=长正确,
                  时长秒=round(float(w1.size) / SR, 3),
                  峰值=round(float(np.abs(w1).max()), 4)),
        两趟sha256同=两趟同, sha256=[sha1, sha2],
        耗时秒=dict(mel转波首趟=round(t1 - t0, 2), 次趟=round(t2 - t1, 2)),
        读回=读回, 边界=边, 模型两趟=模型两趟,
        文件=[str(路1), str(路2)],
        PASS=bool(长正确 and 两趟同 and 读回["达标"] and not 异常),
        异常=异常,
    )
    print("[自检] 长度=%s（%d 点，期望 %d）；两趟 sha256 同=%s；耗时 %.2fs/%.2fs；PASS=%s"
          % (长正确, w1.size, 期长, 两趟同, t1 - t0, t2 - t1, 报["PASS"]), flush=True)
    return 报


# ════════════════════════════════════════════════════════════════════════
# 4. ckpt 等待（训练在跑时不抢内存：先等训练摘要.md / 完成标记）
# ════════════════════════════════════════════════════════════════════════
def _ckpt状态(ckpt路径: Path) -> dict:
    出 = dict(存在=ckpt路径.exists(), 完成=None, 训练信息=None, sha256=None,
              字节=None, 修改时间=None)
    if not 出["存在"]:
        return 出
    st = ckpt路径.stat()
    出["字节"] = int(st.st_size)
    出["修改时间"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
    try:
        import torch  # noqa: PLC0415
        ck = torch.load(str(ckpt路径), map_location="cpu", weights_only=False)
        训 = ck.get("训练信息") or {}
        出["训练信息"] = dict(训)
        出["完成"] = bool(训.get("完成"))
        del ck
    except Exception as e:  # noqa: BLE001  （训练正在写 ckpt 时可能读到半截）
        出["错误"] = "%s: %s" % (type(e).__name__, str(e)[:120])
    return 出


def 等待ckpt(ckpt路径=None, 最多轮询: int = 20, 轮询秒: float = 60.0,
            不等待: bool = False) -> dict:
    """轮询 ckpt：训练摘要.md 出现 或 ckpt 内 训练信息.完成=True ⇒ 可用。

    训练写入期间 torch.load 可能失败；本函数只在下述条件下返回：
      · 训练摘要.md 存在（写在最后一次 ckpt 保存之后） 或 ckpt 完成标记为 True；
      · 且 ckpt 修改时间距今 ≥ 20s（防写到一半）。
    返回状态 dict；超时则 `可用=False`（由调用方决定是否继续用最新 ckpt）。
    """
    路 = Path(ckpt路径) if ckpt路径 else 默认ckpt
    if 不等待:
        return dict(可用=路.exists(), 轮询=0, 说明="--不等待：直接使用当前 ckpt")
    说明 = []
    for 轮 in range(1, int(最多轮询) + 1):
        状 = _ckpt状态(路)
        摘要在 = 训练摘要.exists()
        静默秒 = None
        if 状["存在"]:
            静默秒 = time.time() - 路.stat().st_mtime
        行 = ("[等ckpt] 轮%d/%d 存在=%s 摘要=%s 完成=%s 静默=%s"
              % (轮, 最多轮询, 状["存在"], 摘要在, 状["完成"],
                 ("%.0fs" % 静默秒) if 静默秒 is not None else "-"))
        print(行, flush=True)
        说明.append(行)
        if 状["存在"] and (摘要在 or 状["完成"] is True) and (静默秒 is None or 静默秒 >= 20):
            return dict(可用=True, 轮询=轮, 说明=说明, 摘要存在=摘要在,
                        完成标记=状["完成"], ckpt=状)
        if 轮 < int(最多轮询):
            time.sleep(float(轮询秒))
    状 = _ckpt状态(路)
    return dict(可用=状["存在"], 轮询=int(最多轮询), 说明=说明,
                超时=True, ckpt=状)


# ════════════════════════════════════════════════════════════════════════
# 5. 评测（双集 × SpeakerA × 两趟转写 + sham + 阳性对照）
# ════════════════════════════════════════════════════════════════════════
def _sham(引, 目录, run_id) -> dict:
    """must-fail 对照：白噪 / 静音 → CER 须 ≥ 0.9（与 量具_v9._sham 同内容同种子）。"""
    量 = 量具模块()
    路白 = Path(目录) / ("sham_白噪_%s.wav" % run_id)
    路静 = Path(目录) / ("sham_静音_%s.wav" % run_id)
    rng = np.random.default_rng(63051)                     # 与 meeting63/64 同种子
    写wav(路白, rng.standard_normal(int(3.4 * SR)) * 0.2)
    写wav(路静, np.zeros(int(3.4 * SR)))
    白 = 量.CER(量.真值, 引["转写"](str(路白)))
    静 = 量.CER(量.真值, 引["转写"](str(路静)))
    print("[sham] 白噪 CER=%.3f ｜ 静音 CER=%.3f" % (白, 静), flush=True)
    return dict(白噪=白, 静音=静, 门槛=0.9,
                双达标=bool(白 >= 0.9 and 静 >= 0.9),
                文件=[str(路白), str(路静)])


def _评一句(引, 样本: dict, 集合名: str) -> dict:
    """一份 wav：两趟转写（C13）→ CER + 对齐；附 v8.1 逐句对照。"""
    量 = 量具模块()
    路 = 样本["路径"]
    t1 = 引["转写"](路)
    t2 = 引["转写"](路)
    真 = 样本["文本"]
    cer = 量.CER(真, t1)
    对 = 量.对齐(真, t1)
    对v8 = 对照逐句.get(集合名, {}).get(样本["句号"])
    出 = dict(
        句号=样本.get("句号"), 句=真, 角色=角色,
        文件=样本.get("文件"), 路径=路, sha256=样本["sha256"],
        时长秒=样本["时长秒"], 字数=样本["字数"], 有效音节数=样本["有效音节数"],
        音节率=样本["音节率"], 峰值=样本.get("峰值"), mel=样本["mel"],
        转写=t1, 转写各趟=[t1, t2], 趟一致=bool(t1 == t2),
        CER=cer, 对齐=dict(S=对["S"], D=对["D"], I=对["I"],
                          错误位=对["错误位"], 插入=对["插入"]),
        测文=对["测文"], 真文=对["真文"],
        v8_1_SpeakerACER=对v8,
        Δvs_v8_1=(round(cer - 对v8, 4) if 对v8 is not None else None),
    )
    return 出


def _评集合(逐句: list, 集合名: str) -> dict:
    """集合汇总：CER 均值 / v8.1 逐句对照 / 两趟一致 / 音节率 / 对齐计数。"""
    cers = np.asarray([r["CER"] for r in 逐句], dtype=np.float64)
    v8s = [r["v8_1_SpeakerACER"] for r in 逐句 if r.get("v8_1_SpeakerACER") is not None]
    均 = round(float(cers.mean()), 4) if cers.size else None
    均v8 = round(float(np.mean(v8s)), 4) if v8s else None
    总字数 = int(sum(r["字数"] for r in 逐句))
    总音节 = int(sum((r["有效音节数"] or 0) for r in 逐句))
    总时长 = round(float(sum(r["时长秒"] for r in 逐句)), 3)
    音节率 = round(float(总字数) / 总时长, 3) if 总时长 else None
    S = int(sum(r["对齐"]["S"] for r in 逐句))
    D = int(sum(r["对齐"]["D"] for r in 逐句))
    I = int(sum(r["对齐"]["I"] for r in 逐句))
    return dict(
        集合=集合名, 句数=len(逐句), 逐句=逐句,
        CER均值=均,
        对照v8_1_SpeakerA逐句均值=均v8,
        Δvs_v8_1_SpeakerA=(round(均 - 均v8, 4) if (均 is not None and 均v8 is not None) else None),
        两趟一致=bool(all(r["趟一致"] for r in 逐句)),
        总字数=总字数, 有效音节数总和=总音节, 总时长秒=总时长,
        音节率=音节率, 音节率门槛=3.0,
        音节率达标=bool(音节率 is not None and 音节率 >= 3.0),
        峰值=round(float(max((r.get("峰值") or 0.0) for r in 逐句)), 4),
        对齐合计=dict(S=S, D=D, I=I, 错误数=S + D + I),
    )


def _读meta() -> dict:
    p = 产物 / "数据" / "meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _阳性对照_合成(run_id: str, meta: dict):
    """真实SpeakerA样本 mel → 本件声码器 → wav（**仪器阳性对照**，不走生成路径）。

    用途：证明「22.05k wav 经 Paraformer 转写」链路本身可用（真 mel 反变换应可懂），
    从而把「合成句 CER 高」归因到声学模型/声码器质量，而不是采样率/文件格式错误。
    """
    if not 阳性对照样本.exists():
        return None
    try:
        z = np.load(str(阳性对照样本))
        mel = np.asarray(z["mel"], dtype=np.float32)
        波形 = mel转波(mel)
        路 = 音频目录 / ("阳性对照_真mel_%s_%s.wav" % (阳性对照样本.stem, run_id))
        写wav(路, 波形)
        文本 = None
        for r in meta.get("样本", []):
            if str(r.get("文件")) == 阳性对照样本.name:
                文本 = r.get("文本")
                break
        出 = dict(源=str(阳性对照样本), 帧数=int(mel.shape[0]),
                  期望文本=文本, 时长秒=round(float(波形.size) / SR, 3),
                  路径=str(路), sha256=_sha256文件(路),
                  读回=_读回核对(路), mel=_mel统计(mel))
        print("[阳性对照] 真 mel %d 帧 → %.2fs wav（%s）"
              % (mel.shape[0], 出["时长秒"], 路.name), flush=True)
        return 出
    except Exception as e:  # noqa: BLE001
        print("[阳性对照] 合成失败：%s: %s" % (type(e).__name__, str(e)[:160]), flush=True)
        return None


def _环境() -> dict:
    出 = dict(python=sys.version.split()[0], numpy=np.__version__,
               sr=SR, n_fft=N_FFT, 窗长=窗长, 跳长=跳长, mel带数=MEL带数, AMIN=AMIN)
    try:
        出["librosa"] = librosa模块().__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch  # noqa: PLC0415
        出["torch"] = torch.__version__
    except Exception:  # noqa: BLE001
        pass
    return 出


# ════════════════════════════════════════════════════════════════════════
# 6. 评测主流程
# ════════════════════════════════════════════════════════════════════════
def 评测(ckpt=None, 每音节帧数: int = 每音节帧数缺省, 不等待: bool = False,
         最多轮询: int = 20, 轮询秒: float = 60.0, 输出=None,
         做阳性对照: bool = True) -> dict:
    """双集（S1–S6 历史 / S7–S12 留出）× SpeakerA × 两趟转写 → CER + sham → 判据 JSON。"""
    开始 = time.time()
    输出 = Path(输出) if 输出 else 默认输出
    _确保目录(音频目录, 判据目录)
    异常: list = []
    ckpt路 = Path(ckpt) if ckpt else 默认ckpt

    等 = 等待ckpt(ckpt路, 最多轮询=最多轮询, 轮询秒=轮询秒, 不等待=不等待)
    if not ckpt路.exists():
        raise RuntimeError("ckpt 不可用：%s（等ckpt=%s）" % (ckpt路, 等))
    if not 等.get("可用"):
        异常.append("等ckpt 超时（%s），仍用当前 ckpt 继续" % 等.get("轮询"))
    print("[ckpt] %s（%s）" % (ckpt路, 等.get("ckpt", {}).get("修改时间")), flush=True)

    模型 = 载ckpt(ckpt路)
    训练信息 = dict(getattr(模型, "训练信息", {}) or {})
    ckpt信息 = dict(路径=str(ckpt路), sha256=_sha256文件(ckpt路),
                    字节=int(ckpt路.stat().st_size),
                    修改时间=time.strftime("%Y-%m-%d %H:%M:%S",
                                          time.localtime(ckpt路.stat().st_mtime)),
                    训练信息=训练信息,
                    训练完成=bool(训练信息.get("完成")))
    run = _新run_id()
    meta = _读meta()
    print("[模型] 训练信息：%s" % json.dumps(
        {k: 训练信息.get(k) for k in ("轮", "最佳轮", "最佳验证损失", "提前停", "完成", "两趟一致")},
        ensure_ascii=False), flush=True)

    # —— ① 自检 ——
    print("[自检] 随机 mel → GL（含两趟 sha256 / 模型两趟 mel）…", flush=True)
    自检报 = 自检(模型=模型)
    if not 自检报["PASS"]:
        异常.extend(自检报.get("异常") or [])

    # —— ② 合成（先出全部波形并释放模型，再载 ASR，避免内存峰值叠加） ——
    合成结果: dict = {}
    序 = 0
    for 集合名, 句s in (("历史", 句集历史), ("留出", 句集留出)):
        for 句号, 句 in 句s:
            序 += 1
            样本 = 合成(句, 模型=模型, 每音节帧数=每音节帧数, 标签=集合名,
                        run_id=run, 序号=序)
            样本["句号"] = 句号
            合成结果[(集合名, 句号)] = 样本
            print("[合成] %-2s %-4s 字数=%d 音节=%s 帧=%d 时长=%.2fs 峰=%.2f sha=%s"
                  % (集合名, 句号, 样本["字数"], 样本["有效音节数"],
                     样本["mel"]["形状"][0], 样本["时长秒"], 样本["峰值"],
                     (样本["sha256"] or "")[:12]), flush=True)
            if not 样本["长度正确"]:
                异常.append("%s·%s 波形长度不符（%d vs %d）"
                            % (集合名, 句号, 样本["波形样本数"], 样本["期望样本数"]))
            if not (样本.get("读回") or {}).get("达标", False):
                异常.append("%s·%s wav 读回格式不符" % (集合名, 句号))

    阳 = _阳性对照_合成(run, meta) if 做阳性对照 else None

    峰值前 = float(max((v["峰值"] or 0.0) for v in 合成结果.values()))
    del 模型
    gc.collect()
    try:
        import torch  # noqa: PLC0415
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    print("[内存] 模型已释放，载 ASR 前 峰值=%.2f（合成波形）" % 峰值前, flush=True)

    # —— ③ 载 ASR（单进程、引载一次） ——
    print("[ASR] 载引擎（Paraformer-seaco-large，CPU；全机只许一个 ASR 进程）…", flush=True)
    引 = 量具模块().载引擎()
    print("[ASR] 引擎 = %s" % 引.get("名"), flush=True)

    # —— ④ 双集 × 两趟转写 ——
    全逐句: dict = {"历史": [], "留出": []}
    for 集合名, 句s in (("历史", 句集历史), ("留出", 句集留出)):
        for 句号, _句 in 句s:
            样本 = 合成结果[(集合名, 句号)]
            r = _评一句(引, 样本, 集合名)
            全逐句[集合名].append(r)
            print("[ASR] %-2s %-4s CER=%.4f 两趟一致=%s 对照=%.4f 转写=%s"
                  % (集合名, 句号, r["CER"], r["趟一致"], r["v8_1_SpeakerACER"],
                     r["转写"][:44]), flush=True)
            if not r["趟一致"]:
                异常.append("%s·%s 两趟转写不一致" % (集合名, 句号))
    汇总 = {名: _评集合(全逐句[名], 名) for 名 in ("历史", "留出")}

    # —— ⑤ sham（must-fail） ——
    sham = _sham(引, 音频目录, run)
    if not sham["双达标"]:
        异常.append("sham 未达标（白噪=%.3f 静音=%.3f，门槛 0.9）"
                    % (sham["白噪"], sham["静音"]))

    # —— ⑥ 仪器阳性对照转写 ——
    if 阳:
        try:
            转 = 引["转写"](阳["路径"])
            转2 = 引["转写"](阳["路径"])
            阳["转写"] = 转
            阳["转写各趟"] = [转, 转2]
            阳["趟一致"] = bool(转 == 转2)
            阳["CER"] = (量具模块().CER(阳["期望文本"], 转)
                         if 阳.get("期望文本") else None)
            print("[阳性对照] CER=%s 两趟一致=%s 转写=%s"
                  % (阳["CER"], 阳["趟一致"], 转[:60]), flush=True)
        except Exception as e:  # noqa: BLE001
            阳["错误"] = "%s: %s" % (type(e).__name__, str(e)[:160])
            异常.append("阳性对照转写失败")

    # —— ⑦ 判定 ——
    历史均 = 汇总["历史"]["CER均值"]
    留出均 = 汇总["留出"]["CER均值"]
    历史Δ = round(历史均 - 对照["历史_SpeakerA"], 4)
    留出Δ全 = round(留出均 - 对照["留出_全角色"], 4)
    留出Δ登 = round(留出均 - 对照["留出_SpeakerA"], 4)
    条款 = [
        dict(条款="C13：每句两趟转写逐字一致",
             通过=bool(汇总["历史"]["两趟一致"] and 汇总["留出"]["两趟一致"]),
             实测="历史=%s 留出=%s" % (汇总["历史"]["两趟一致"], 汇总["留出"]["两趟一致"])),
        dict(条款="sham（白噪/静音）CER ≥ 0.9",
             通过=bool(sham["双达标"]),
             实测="白噪=%.3f 静音=%.3f" % (sham["白噪"], sham["静音"])),
        dict(条款="声码器自检（长度正确 / 两趟 sha256 同 / 读回格式）",
             通过=bool(自检报["PASS"]),
             实测="长度=%s 两趟sha256同=%s 读回=%s"
                  % (自检报["波形"]["长度正确"], 自检报["两趟sha256同"],
                     自检报["读回"]["达标"])),
        dict(条款="音节率 ≥ 3.0/s（双集）",
             通过=bool(汇总["历史"]["音节率达标"] and 汇总["留出"]["音节率达标"]),
             实测="历史=%.3f 留出=%.3f"
                  % (汇总["历史"]["音节率"], 汇总["留出"]["音节率"])),
        dict(条款="可懂度优于 v8.1（历史<0.5853 且 留出<0.6063）",
             通过=bool(历史均 < 对照["历史_SpeakerA"] and 留出均 < 对照["留出_全角色"]),
             实测="历史=%.4fΔ%+.4f ｜ 留出=%.4fΔ%+.4f（vs 全角色）"
                  % (历史均, 历史Δ, 留出均, 留出Δ全)),
    ]
    总通过 = all(bool(c["通过"]) for c in 条款)
    判定 = dict(
        条款=条款, 全部通过=总通过,
        历史=dict(CER均值=历史均, 对照v8_1_SpeakerA=对照["历史_SpeakerA"], Δ=历史Δ,
                 优于v8_1=bool(历史均 < 对照["历史_SpeakerA"])),
        留出=dict(CER均值=留出均, 对照v8_1_全角色=对照["留出_全角色"],
                 对照v8_1_SpeakerA=对照["留出_SpeakerA"],
                 Δvs_v8_1_全角色=留出Δ全, Δvs_v8_1_SpeakerA=留出Δ登,
                 优于v8_1_全角色=bool(留出均 < 对照["留出_全角色"]),
                 优于v8_1_SpeakerA=bool(留出均 < 对照["留出_SpeakerA"])),
        结论=("PASS：工程口径全过，且双集 CER 均优于 v8.1"
               if 总通过 else
               "FAIL：未全部通过（逐条见条款；可懂度以 CER 实测为准，不修饰）"),
    )

    报 = dict(
        工具="声码器与评测.py", 版本="1.0", 任务="meeting66 T-F（声码器 Griffin-Lim + 双集 ASR 复评）",
        开始时间=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(开始)),
        run_id=run, 角色=角色,
        角色说明="训练语料即SpeakerA；模型输入只有音节音素，无角色条件",
        ckpt=ckpt信息, 等ckpt=等,
        声码器=dict(
            方法="log-mel → 功率=exp(mel)−1e-6 → librosa mel 滤波器伪逆(NNLS, mel_to_stft)"
                 " → librosa.griffinlim（固定迭代/固定种子）→ 峰值归一",
            入口="librosa.griffinlim（librosa>=0.11；旧 librosa.effects.griffinlim 已移除）",
            sr=SR, n_fft=N_FFT, 窗长=窗长, 跳长=跳长, mel带数=MEL带数,
            fmin=FMIN, fmax=FMAX, log口径="ln(mel_power + %.0e)" % AMIN,
            GL迭代=GL迭代, GL种子=GL种子, 相位初始化="random_state=%d" % GL种子,
            峰值归一=目标峰值,
            长度约定="T 帧 → (T−1)×跳采样点（center=True ISTFT）",
        ),
        每音节帧数=int(每音节帧数),
        自检=自检报,
        双集=汇总,
        对照=dict(**对照, 逐句=对照逐句,
                 逐句说明="c00 逐样本中「SpeakerA」单角色 CER"),
        对照集=dict(历史=[list(x) for x in 句集历史], 留出=[list(x) for x in 句集留出],
                  来源="meeting64/产物/诊断/上限/_预注册.json（预注册先于 A/B）"),
        sham=sham, 仪器阳性对照=阳,
        判定=判定, 环境=_环境(),
        异常=异常,
        耗时秒=round(time.time() - 开始, 1),
        结束时间=time.strftime("%Y-%m-%d %H:%M:%S"),
        产物=dict(音频目录=str(音频目录), 判据=str(输出)),
        复现命令=("$env:PYTHONIOENCODING=\"utf-8\"; & \"<workspace>/\03_AHI\\.venv\\Scripts\\python.exe\" "
                "\"<workspace>/\03_AHI\\AHI4\\meeting66\\实施\\声码器与评测.py\" 评测"),
    )
    报 = _json可写(报)
    输出.write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")

    # —— ⑧ 打印 ——
    print("=" * 96)
    for 集合名 in ("历史", "留出"):
        s = 汇总[集合名]
        print("[%s集] CER均值=%.4f ｜ v8.1SpeakerA逐句均值=%.4f ｜ Δ=%+.4f ｜ 两趟一致=%s ｜ 音节率=%.3f"
              % (集合名, s["CER均值"], s["对照v8_1_SpeakerA逐句均值"],
                 s["Δvs_v8_1_SpeakerA"], s["两趟一致"], s["音节率"]))
        for r in s["逐句"]:
            print("   %-4s CER=%.4f (v8.1 %+.4f) 两趟一致=%s [%s] %s"
                  % (r["句号"], r["CER"], r["Δvs_v8_1"], r["趟一致"],
                     r["文件"], r["转写"][:44]))
    print("-" * 96)
    for c in 条款:
        print("  [%s] %-44s %s" % ("PASS" if c["通过"] else "FAIL", c["条款"], c["实测"]))
    print("=" * 96)
    print("总判定：%s ｜ 耗时 %.0fs ｜ 异常 %d 条 ｜ 判据 JSON：%s"
          % (判定["结论"], 报["耗时秒"], len(异常), 输出))
    if 异常:
        for 项 in 异常:
            print("  ! " + str(项))
    return 报


# ════════════════════════════════════════════════════════════════════════
# 7. 诊断：合成 mel 的结构 vs 真 mel 基准（不载 ASR；定位可懂度失败成因）
# ════════════════════════════════════════════════════════════════════════
def _mel结构统计(mel) -> dict:
    """一帧一帧看结构：帧间变化 / 带内时间起伏 / 能量包络起伏 / 谱倾斜 / 带均值轮廓。"""
    m = np.asarray(mel, dtype=np.float64)
    帧间 = float(np.mean(np.abs(np.diff(m, axis=0)))) if m.shape[0] > 1 else 0.0
    带内时间标准差 = float(np.mean(np.std(m, axis=0)))
    轮廓 = np.mean(m, axis=0)
    能 = np.log(np.sum(np.exp(m), axis=1) + 1e-12)
    能std = float(np.std(能))
    倾斜 = float(np.mean(m[:, :20]) - np.mean(m[:, 60:]))
    return dict(帧数=int(m.shape[0]), 帧间变化=round(帧间, 4),
                带内时间标准差=round(带内时间标准差, 4),
                能量包络std=round(能std, 4), 谱倾斜_低频减高频=round(倾斜, 4),
                带均值轮廓=[round(float(x), 2) for x in 轮廓])


def _真实mel基准(样本数: int = 4, 每段帧: int = 320) -> dict:
    """数据集真 mel（只读）作结构基准；同口径统计 ⇒ 与模型合成 mel 可比。"""
    段s, 用了 = [], []
    for i in range(12):
        p = 产物 / "数据" / "样本" / ("%04d.npz" % i)
        if not p.exists():
            continue
        try:
            mel = np.asarray(np.load(str(p))["mel"], dtype=np.float32)
        except Exception:  # noqa: BLE001
            continue
        if mel.ndim == 2 and mel.shape[1] == MEL带数:
            段s.append(mel[:每段帧])
            用了.append(p.name)
        if len(段s) >= 样本数:
            break
    if not 段s:
        return {}
    st = _mel结构统计(np.concatenate(段s, axis=0))
    st["来源"] = 用了
    st["每段帧"] = 每段帧
    return st


def 诊断(ckpt=None, 输出=None) -> dict:
    """只诊断「模型合成 mel 的结构」（不载 ASR、不写音频）：与数据集真 mel 同口径对照。

    用途：CER=1.0 时区分「声码器坏了」还是「模型 mel 退化」。真 mel 的
    mel→GL→ASR 已由 评测 的仪器阳性对照给出（CER 很低 ⇒ 声码器/仪器可用），
    本动作再给出「模型 mel 是否塌缩成常数/无音节结构」的数值证据。
    """
    _确保目录(判据目录)
    路 = Path(ckpt) if ckpt else 默认ckpt
    输出 = Path(输出) if 输出 else 判据目录 / "诊断_mel结构.json"
    模型 = 载ckpt(路)
    mmean = np.asarray(getattr(模型, "mel均值"), dtype=np.float64).reshape(-1)
    msd = np.asarray(getattr(模型, "mel标准差"), dtype=np.float64).reshape(-1)
    真基准 = _真实mel基准()
    报 = dict(工具="声码器与评测.py", 动作="诊断_mel结构",
              时间=time.strftime("%Y-%m-%d %H:%M:%S"),
              说明="只诊断模型合成 mel 的结构（不载 ASR、不写音频）；真 mel 基准取自 产物/数据/样本/*.npz",
              ckpt=dict(路径=str(路), sha256=_sha256文件(路),
                        训练信息=dict(getattr(模型, "训练信息", {}) or {})),
              训练mel统计=dict(均值=[round(float(x), 2) for x in mmean],
                            标准差均值=round(float(msd.mean()), 4) if msd.size else None),
              真mel基准=真基准, 逐句=[])
    for 句号, 句 in 句集历史:
        mel, _n, _s = 推理mel(模型, 句)
        st = _mel结构统计(mel)
        k = int(min(mmean.size, msd.size, mel.shape[1]))
        st["相对训练均值_σ单位"] = round(float(np.mean(
            np.abs(mel[:, :k] - mmean[:k]) / np.maximum(msd[:k], 1e-6))), 3)
        if 真基准.get("带均值轮廓"):
            st["轮廓相关_真mel"] = round(float(np.corrcoef(
                st["带均值轮廓"], 真基准["带均值轮廓"])[0, 1]), 3)
        if mmean.size == mel.shape[1]:
            st["轮廓相关_训练均值"] = round(float(np.corrcoef(
                st["带均值轮廓"], mmean)[0, 1]), 3)
        报["逐句"].append(dict(句号=句号, 句=句, **st))
    帧间 = [x["帧间变化"] for x in 报["逐句"]]
    时std = [x["带内时间标准差"] for x in 报["逐句"]]
    能std = [x["能量包络std"] for x in 报["逐句"]]
    均 = lambda v: round(float(np.mean(v)), 4) if v else None  # noqa: E731
    报["汇总"] = dict(
        模型_帧间变化均值=均(帧间),
        真mel_帧间变化=真基准.get("帧间变化"),
        帧间变化比=(round(均(帧间) / 真基准["帧间变化"], 3)
                if v_ok(帧间) and 真基准.get("帧间变化") else None),
        模型_带内时间标准差均值=均(时std),
        真mel_带内时间标准差=真基准.get("带内时间标准差"),
        带内时间标准差比=(round(均(时std) / 真基准["带内时间标准差"], 3)
                  if v_ok(时std) and 真基准.get("带内时间标准差") else None),
        模型_能量包络std均值=均(能std),
        真mel_能量包络std=真基准.get("能量包络std"),
    )
    # 一句话判读（不修饰）
    比 = 报["汇总"]["帧间变化比"]
    if 比 is not None and 比 < 0.25:
        报["汇总"]["判读"] = ("模型合成 mel 的帧间变化仅为真 mel 的 %.0f%%（塌缩/近乎常值）"
                          "⇒ 音节级变化缺失，声码器只能输出稳态嗡鸣；"
                          "可懂度失败根因在声学模型，不在 Griffin-Lim。" % (100 * 比))
    else:
        报["汇总"]["判读"] = "模型 mel 有明显时间变化；可懂度问题需结合逐句对齐进一步定位。"
    报 = _json可写(报)
    输出.write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[诊断] 模型 帧间变化均值=%s（真 mel=%s，比=%s）｜ 带内时间标准差 模型=%s 真=%s"
          % (报["汇总"]["模型_帧间变化均值"], 报["汇总"]["真mel_帧间变化"],
             报["汇总"]["帧间变化比"], 报["汇总"]["模型_带内时间标准差均值"],
             报["汇总"]["真mel_带内时间标准差"]))
    print("[诊断] 判读：%s" % 报["汇总"]["判读"])
    for r in 报["逐句"]:
        print("   %-3s 帧间变化=%.3f 带内std=%.3f 能量std=%.3f 倾斜=%.2f "
              "σ偏差=%.2f 轮廓相关真mel=%s"
              % (r["句号"], r["帧间变化"], r["带内时间标准差"], r["能量包络std"],
                 r["谱倾斜_低频减高频"], r["相对训练均值_σ单位"], r.get("轮廓相关_真mel")))
    print("[诊断] JSON：%s" % 输出)
    return 报


def v_ok(v) -> bool:
    return bool(v) and len(v) > 0


# ════════════════════════════════════════════════════════════════════════
# 8. CLI
# ════════════════════════════════════════════════════════════════════════
def 主(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
    warnings.filterwarnings("ignore", category=FutureWarning)
    np.seterr(all="ignore")
    ap = argparse.ArgumentParser(description="meeting66 T-F：声码器（Griffin-Lim）+ 双集 ASR 复评")
    ap.add_argument("动作", nargs="?", default="全部",
                    choices=["自检", "评测", "全部", "诊断"],
                    help="自检=只测声码器；评测=双集 ASR 全流程；全部=同评测（默认）；诊断=只诊断合成 mel 结构")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--每音节帧数", type=int, default=每音节帧数缺省)
    ap.add_argument("--不等待", action="store_true", help="不等训练完成，直接用当前 ckpt")
    ap.add_argument("--最多轮询", type=int, default=20)
    ap.add_argument("--轮询秒", type=float, default=60.0)
    ap.add_argument("--输出", default=None)
    ap.add_argument("--不做阳性对照", action="store_true")
    参 = ap.parse_args(argv)
    _确保目录(音频目录, 判据目录)
    t0 = time.time()
    if 参.动作 == "诊断":
        诊断(ckpt=参.ckpt, 输出=参.输出)
        print("诊断完成 ｜ 耗时 %.1fs" % (time.time() - t0))
        return 0
    if 参.动作 == "自检":
        模型 = None
        路 = Path(参.ckpt) if 参.ckpt else 默认ckpt
        if 路.exists():
            try:
                模型 = 载ckpt(路)
            except Exception as e:  # noqa: BLE001
                print("[自检] ckpt 暂不可用（%s），只做纯声码器自检" % type(e).__name__)
        报 = 自检(模型=模型)
        print(json.dumps(_json可写({k: v for k, v in 报.items() if k != "边界"}),
                         ensure_ascii=False, indent=1))
        print("自检 PASS=%s ｜ 耗时 %.1fs" % (报["PASS"], time.time() - t0))
        return 0 if 报["PASS"] else 1
    报 = 评测(ckpt=参.ckpt, 每音节帧数=参.每音节帧数, 不等待=参.不等待,
               最多轮询=参.最多轮询, 轮询秒=参.轮询秒, 输出=参.输出,
               做阳性对照=not 参.不做阳性对照)
    return 0 if 报["判定"]["全部通过"] else 2


if __name__ == "__main__":
    sys.exit(主())
