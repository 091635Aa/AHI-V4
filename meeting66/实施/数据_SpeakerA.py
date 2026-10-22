# -*- coding: utf-8 -*-
"""meeting66/实施/数据_SpeakerA.py —— SpeakerA 167×60s 标注语料 → 音素 token + log-mel 数据集。

任务（meeting66 · T-D）
---------------------
读 ``<path>\\SpeakerA7代模型训练数据\\asr_opt\\SpeakerA7代模型训练数据.list``（``wav|speaker|ZH|文本``，
167 行；list 内路径前缀是 ``<data-root>/\...\\  (1).wav``，真实文件在
``<path>\\SpeakerA7代模型训练数据\\(1).wav``（文件名无前导空格），脚本做路径映射并校验存在）。

单条处理链（逐文件流式，RAM 紧）::

    44.1k 立体声 wav → 单声道 → resample_poly 22.05k
      → 能量 VAD（25ms 窗/10ms 跳，阈 = 峰值 −35 dB）→ 裁掉首尾静音
      → log-mel（80 带 / 25ms 窗 / 10ms 跳 / n_fft=1024；librosa）
      → 文本 → 音节（去标点/非汉字；pypinyin TONE3；声/韵/调 由
        meeting61/实施/音韵_v5.解析音节 给出）
      → 音节→帧：VAD 语音段内按音节数均分（round 后余数给末音节），
        每音节 25% 声母 / 75% 韵母（声母至少 2 帧）
      → meeting66/产物/数据/样本/NNNN.npz + meta.json

产出（只写 meeting66/产物/数据/）
-------------------------------
* ``样本/NNNN.npz``：``mel`` float32 [T,80]、``声母``/``韵母``/``调`` int32 [N]、
  ``音节帧数`` int32 [N]（另存 ``声母帧数``/``韵母帧数`` 以便训练对齐）
* ``meta.json``：vocab / 参数 / 样本清单 / train-val 划分（种子 300）/ 全局 mel 均值方差 / 统计
* ``自检.txt``：自检打印留档

约束
----
只读数据源（不复制原始音频）；不跑 ASR；不联网；确定性（同输入两趟 meta sha256 逐位同，
所有随机只用固定种子 300）；不改既有件（C14）。

用法
----
::

    $py = "<workspace>/\03_AHI\\.venv\\Scripts\\python.exe"
    & $py meeting66\\实施\\数据_SpeakerA.py --限 5                # 先跑 5 条自检（写 产物\\数据\\自检_前5）
    & $py meeting66\\实施\\数据_SpeakerA.py                       # 全 167 条 → 产物\\数据
    & $py meeting66\\实施\\数据_SpeakerA.py --基准meta-sha <sha>   # 第二趟：确定性对照
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import librosa
import numpy as np
import scipy.signal as sps
import soundfile as sf
from pypinyin import Style, lazy_pinyin

# ---------------------------------------------------------------- 路径 / 常量
根 = Path(__file__).resolve().parents[2]                    # ...\03_AHI\AHI4
音韵路 = 根 / "meeting61" / "实施" / "音韵_v5.py"           # 只读 import
产物根 = 根 / "meeting66" / "产物" / "数据"
清单路 = Path(r"<path>\SpeakerA7代模型训练数据\asr_opt\SpeakerA7代模型训练数据.list")
音频根 = Path(r"<path>\SpeakerA7代模型训练数据")

SR = 22050                      # 重采样目标
窗毫秒 = 25.0
跳毫秒 = 10.0
窗长 = int(round(SR * 窗毫秒 / 1000.0))                     # 551 采样点（25ms）
跳长 = int(round(SR * 跳毫秒 / 1000.0))                     # 220 采样点（≈10ms）
N_FFT = 1024
MEL带数 = 80
FMIN = 0.0
FMAX = SR / 2.0
AMIN = 1e-6                     # log-mel 下限（log(power + AMIN)）
VAD_低于峰值dB = 35.0           # 能量阈 = 峰值 −35 dB
种子 = 300
VAL比例 = 0.05

# VAD / mel 共用同一帧栅格（25ms 窗、10ms 跳）


# ---------------------------------------------------------------- 小工具
def 印(行: str = "", 记录: list | None = None) -> None:
    print(行)
    if 记录 is not None:
        记录.append(行)


def _载入音韵():
    """按文件路径导入只读的 音韵_v5.py（不修改、不安装）。"""
    spec = importlib.util.spec_from_file_location("音韵_v5_只读", 音韵路)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def 是汉字(c: str) -> bool:
    o = ord(c)
    return 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF


def 取汉字(文本: str) -> str:
    """只保留汉字（去标点/英文/数字/空白）。"""
    return "".join(c for c in 文本 if 是汉字(c))


# ---------------------------------------------------------------- 清单 / 路径映射
def 读清单():
    """返回条目列表；同时做 <data-root>/\... → <path>\\SpeakerA7代模型训练数据\\ 的路径映射。"""
    行s = 清单路.read_text(encoding="utf-8").splitlines()
    条目s = []
    for i, 行 in enumerate(行s):
        行 = 行.strip()
        if not 行:
            continue
        段 = 行.split("|", 3)
        if len(段) != 4:
            raise ValueError("清单第 %d 行字段数=%d，期望 4：%s" % (i + 1, len(段), 行[:120]))
        原路, 说话人, 语言, 文本 = 段
        名 = Path(原路.strip()).name.strip()       # 去文件名前导空格
        if not 名.lower().endswith(".wav"):
            raise ValueError("清单第 %d 行不是 wav：%s" % (i + 1, 原路))
        条目s.append(dict(
            序号=len(条目s), 原路=原路.strip(), 名=名,
            wav=音频根 / 名, 说话人=说话人.strip(), 语言=语言.strip(), 文本=文本.strip(),
        ))
    return 条目s


# ---------------------------------------------------------------- 音频 / VAD
def 读音频(wav: Path):
    """44.1k 立体声 → 单声道 float32 → 22.05k（resample_poly）。"""
    x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    单 = np.ascontiguousarray(x.mean(axis=1, dtype=np.float32))
    if sr != SR:
        g = math.gcd(int(sr), SR)
        单 = np.ascontiguousarray(sps.resample_poly(单, SR // g, int(sr) // g).astype(np.float32))
    return 单, int(sr)


def 找语音段(x: np.ndarray):
    """能量 VAD：返回 (起, 止, 异常)。在 25ms 窗 / 10ms 跳 栅格上算帧能量，
    阈 = 全段峰值 −35 dB，取首末浊帧，裁掉首尾静音。"""
    n = int(x.size)
    if n < 窗长:
        return 0, n, ["音频过短(%d 采样)" % n]
    视 = np.lib.stride_tricks.sliding_window_view(x, 窗长)[::跳长]
    能 = np.mean(np.square(视.astype(np.float64)), axis=1)
    能db = 10.0 * np.log10(能 + 1e-12)
    浊 = 能db > (float(能db.max()) - VAD_低于峰值dB)
    下标 = np.flatnonzero(浊)
    异常 = []
    if 下标.size == 0:
        return 0, n, ["无浊帧(整段按静音处理)"]
    首 = int(下标[0])
    末 = int(下标[-1])
    起 = 首 * 跳长
    止 = min(n, 末 * 跳长 + 窗长)
    if 末 - 首 < 3:
        异常.append("语音段极短(%d 帧)" % (末 - 首 + 1))
    if 起 == 0 and 止 == n:
        异常.append("首尾静音未裁掉")
    return 起, 止, 异常


# ---------------------------------------------------------------- log-mel
_梅尔滤波器 = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=MEL带数,
                                   fmin=FMIN, fmax=FMAX, norm="slaney", htk=False)
_汉宁窗 = sps.windows.hann(窗长, sym=False)      # 与 librosa 的 periodic hann 一致


def 计算logmel(x: np.ndarray) -> np.ndarray:
    """librosa 主线：log(power + AMIN)，float32 [T,80]，T = 1 + len(x)//跳长。"""
    S = librosa.feature.melspectrogram(
        y=np.ascontiguousarray(x, dtype=np.float32), sr=SR, n_fft=N_FFT,
        hop_length=跳长, win_length=窗长, window="hann", center=True,
        pad_mode="constant", power=2.0, n_mels=MEL带数, fmin=FMIN, fmax=FMAX,
        norm="slaney", htk=False,
    )
    return np.log(S + AMIN).astype(np.float32).T      # librosa 返回 [80, T] → 统一成 [T, 80]


def 自实现logmel(x: np.ndarray) -> np.ndarray:
    """对照实现（手工 STFT + librosa 的 mel 滤波器组），用于校验帧化/加窗/功率口径。"""
    y = np.pad(np.asarray(x, dtype=np.float64), (N_FFT // 2, N_FFT // 2), mode="constant")
    帧数 = 1 + len(x) // 跳长
    帧 = np.lib.stride_tricks.sliding_window_view(y, N_FFT)[::跳长][:帧数]
    加窗 = 帧 * np.pad(_汉宁窗, ((N_FFT - 窗长) // 2, N_FFT - 窗长 - (N_FFT - 窗长) // 2))[None, :]
    谱 = np.fft.rfft(加窗, n=N_FFT, axis=1)
    功 = 谱.real ** 2 + 谱.imag ** 2
    mel = 功 @ _梅尔滤波器.T
    return np.log(mel + AMIN).astype(np.float32)


# ---------------------------------------------------------------- 文本 → 音节
def 解析文本(文本: str, 音韵):
    """汉字串 → ([解析音节 dict], 汉字串, 去掉的非汉字数)。"""
    cn = 取汉字(文本)
    去数 = len(文本) - len(cn)
    if not cn:
        return [], cn, 去数
    拼s = lazy_pinyin(cn, style=Style.TONE3, neutral_tone_with_five=True)
    音节s = [音韵.解析音节(p) for p in 拼s if p and p[0].isascii()]
    return 音节s, cn, 去数


def 建词表(解析s列表, 音韵):
    """声母 22（含 ∅）、韵母（音韵_v5 韵母表按定义序 ∩ 语料出现，外加兜底）、调 5。"""
    声母表 = ["∅"] + list(音韵.声母序)
    出现韵 = sorted({s["韵"] for ss in 解析s列表 for s in ss})
    韵母表 = [y for y in 音韵.韵母表 if y in set(出现韵)]
    韵母表 += [y for y in 出现韵 if y not in set(韵母表)]
    调表 = [1, 2, 3, 4, 5]
    return dict(
        声母=声母表, 韵母=韵母表, 调=调表,
        声母id={s: i for i, s in enumerate(声母表)},
        韵母id={y: i for i, y in enumerate(韵母表)},
        调id={t: i for i, t in enumerate(调表)},
    )


# ---------------------------------------------------------------- 音节 → 帧
def 分配帧数(总帧: int, 音节数: int):
    """VAD 语音段内按音节数均分：per = round(T/N)，余数给末音节；
    round 溢出使末音节非正时退回 floor 分配。返回 (每音节帧数 list, 是否退回)。"""
    if 音节数 <= 0:
        raise ValueError("音节数为 0")
    if 总帧 < 音节数:
        raise ValueError("帧数(%d) < 音节数(%d)" % (总帧, 音节数))
    每 = int(math.floor(总帧 / 音节数 + 0.5))
    帧s = [每] * 音节数
    帧s[-1] += 总帧 - 每 * 音节数
    if 帧s[-1] <= 0:
        每 = 总帧 // 音节数
        帧s = [每] * 音节数
        帧s[-1] += 总帧 - 每 * 音节数
        return 帧s, True
    return 帧s, False


def 声韵帧数(音节帧: int):
    """每音节内部 25% 声母 / 75% 韵母，声母至少 2 帧（帧数过少时保底各 ≥1）。"""
    if 音节帧 <= 1:
        return 1, 0
    if 音节帧 == 2:
        return 1, 1
    声帧 = max(2, int(math.floor(音节帧 * 0.25 + 0.5)))
    if 声帧 >= 音节帧:
        声帧 = 音节帧 - 1
    return 声帧, 音节帧 - 声帧


# ---------------------------------------------------------------- npz 读写（确定性：固定 zip 时间戳）
def 写npz(路径: Path, 数组s: dict) -> None:
    缓 = io.BytesIO()
    with zipfile.ZipFile(缓, "w", zipfile.ZIP_DEFLATED) as z:
        for 键 in sorted(数组s):
            数组 = np.ascontiguousarray(数组s[键])
            b = io.BytesIO()
            np.lib.format.write_array(b, 数组, allow_pickle=False)
            zi = zipfile.ZipInfo(键 + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o600 << 16
            z.writestr(zi, b.getvalue())
    临时 = 路径.with_name(路径.name + ".tmp")
    临时.write_bytes(缓.getvalue())
    临时.replace(路径)


def 数组sha(数组s: dict) -> str:
    h = hashlib.sha256()
    for 键 in sorted(数组s):
        数组 = np.ascontiguousarray(数组s[键])
        h.update(键.encode("utf-8"))
        h.update(str(数组.dtype).encode("ascii"))
        h.update(str(数组.shape).encode("ascii"))
        h.update(数组.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- 单条处理
def 处理一条(条目, 音节s, 词表, 音韵, 做对照: bool):
    """返回 (记录 dict, 数组 dict, 对照最大差 or None, 异常 list)。"""
    x, 源sr = 读音频(条目["wav"])
    起, 止, 异常 = 找语音段(x)
    x段 = np.ascontiguousarray(x[起:止])
    mel = 计算logmel(x段)
    帧数 = int(mel.shape[0])

    音节数 = len(音节s)
    if 音节数 == 0:
        raise ValueError("%s 无汉字音节" % 条目["名"])
    帧s, 退回 = 分配帧数(帧数, 音节数)
    if 退回:
        异常.append("帧分配 round 溢出→floor 兜底")
    声帧s, 韵帧s = [], []
    for 帧 in 帧s:
        声帧, 韵帧 = 声韵帧数(帧)
        声帧s.append(声帧)
        韵帧s.append(韵帧)

    声id = np.array([词表["声母id"][s["声"] or "∅"] for s in 音节s], dtype=np.int32)
    韵id = np.array([词表["韵母id"][s["韵"]] for s in 音节s], dtype=np.int32)
    调id = np.array([词表["调id"][s["调"]] for s in 音节s], dtype=np.int32)
    数组s = dict(
        mel=mel,
        声母=声id, 韵母=韵id, 调=调id,
        音节帧数=np.array(帧s, dtype=np.int32),
        声母帧数=np.array(声帧s, dtype=np.int32),
        韵母帧数=np.array(韵帧s, dtype=np.int32),
    )
    assert int(np.sum(帧s)) == 帧数
    assert 帧数 == 1 + len(x段) // 跳长

    对照差 = None
    if 做对照:
        ref = 自实现logmel(x段)
        if ref.shape != mel.shape:
            异常.append("对照实现形状不符 %s vs %s" % (ref.shape, mel.shape))
        else:
            对照差 = float(np.max(np.abs(ref.astype(np.float64) - mel.astype(np.float64))))
            if 对照差 > 2e-4:
                异常.append("mel 对照差超阈 %.3e" % 对照差)

    记录 = dict(
        id=int(条目["序号"]), 文件="%04d.npz" % 条目["序号"],
        wav=条目["名"], 说话人=条目["说话人"],
        源采样率=源sr,
        帧数=帧数,
        音节数=音节数,
        音素token数=3 * 音节数,
        原时长秒=round(len(x) / SR, 4),
        语音时长秒=round(len(x段) / SR, 4),
        起点采样=int(起), 终点采样=int(止),
        裁掉首ms=round(1000.0 * 起 / SR, 1),
        裁掉尾ms=round(1000.0 * (len(x) - 止) / SR, 1),
        非汉字字数=int(条目["非汉字数"]),
        文本=条目["文本"],
        数组sha256=数组sha(数组s),
    )
    return 记录, 数组s, 对照差, 异常


# ---------------------------------------------------------------- 自检 / 输出
def 打印对照(条目s, 解析s列表, 记录s, 印记录: list) -> None:
    印("\n[自检] 文本 ↔ 音节 token ↔ 帧数（前 5 条）", 印记录)
    词表 = 词表_缓存
    for k in range(min(5, len(条目s))):
        e = 条目s[k]
        r = 记录s[k]
        音节s = 解析s列表[k]
        帧s = 音节帧表_缓存[k]
        印("  id=%04d wav=%-12s 帧=%d 音节=%d 语音=%.2fs 裁首/尾=%.0f/%.0fms" % (
            e["序号"], e["名"], r["帧数"], r["音节数"], r["语音时长秒"],
            r["裁掉首ms"], r["裁掉尾ms"]), 印记录)
        印("    文本: %s" % (e["文本"][:60] + ("…" if len(e["文本"]) > 60 else "")), 印记录)
        片段 = []
        for ch, s in list(zip(取汉字(e["文本"]), 音节s))[:8]:
            片段.append("%s=%s(%s,%s,%d)" % (ch, s["拼音"], s["声"] or "∅", s["韵"], s["调"]))
        印("    音节: " + " ".join(片段), 印记录)
        声id = [词表["声母id"][s["声"] or "∅"] for s in 音节s]
        韵id = [词表["韵母id"][s["韵"]] for s in 音节s]
        调id = [词表["调id"][s["调"]] for s in 音节s]
        印("    token 前 12 组 [声,韵,调]: " + " ".join(
            "[%d,%d,%d]" % t for t in list(zip(声id, 韵id, 调id))[:12]), 印记录)
        印("    音节帧数 前 12: " + str(帧s[:12]) + "  合计=%d" % int(np.sum(帧s)), 印记录)


词表_缓存: dict = {}
音节帧表_缓存: list = []


def 自检回读(样本目录: Path, 记录s: list, 数量: int, 印记录: list) -> dict:
    """至少 20 个样本可读：逐数组校验形状/类型/有限/对齐/词表区间，并核对 meta 里的数组 sha。"""
    n = len(记录s)
    if n == 0:
        return {"通过": False, "原因": "无样本"}
    取 = sorted({int(round(i * (n - 1) / max(1, 数量 - 1))) for i in range(数量)})
    检查 = 0
    问题 = []
    for i in 取:
        r = 记录s[i]
        p = 样本目录 / r["文件"]
        if not p.exists():
            问题.append("%s 不存在" % r["文件"])
            continue
        with np.load(p, allow_pickle=False) as z:
            键 = set(z.files)
            缺少 = {"mel", "声母", "韵母", "调", "音节帧数"} - 键
            if 缺少:
                问题.append("%s 缺键 %s" % (r["文件"], 缺少))
                continue
            mel = z["mel"]
            声 = z["声母"]; 韵 = z["韵母"]; 调 = z["调"]
            帧 = z["音节帧数"]; 声帧 = z["声母帧数"]; 韵帧 = z["韵母帧数"]
        if mel.dtype != np.float32 or mel.ndim != 2 or mel.shape[1] != MEL带数:
            问题.append("%s mel 形状/类型 %s %s" % (r["文件"], mel.dtype, mel.shape))
        if not np.all(np.isfinite(mel)):
            问题.append("%s mel 含非有限值" % r["文件"])
        if mel.shape[0] != r["帧数"] != int(帧.sum()):
            问题.append("%s 帧数不一致 m=%d 记=%d Σ音节帧=%d" % (
                r["文件"], mel.shape[0], r["帧数"], int(帧.sum())))
        if not (声.shape == 韵.shape == 调.shape == 帧.shape == (r["音节数"],)):
            问题.append("%s token 形状不符" % r["文件"])
        if int(np.sum(声帧 + 韵帧)) != int(帧.sum()):
            问题.append("%s 声/韵帧和 ≠ 音节帧和" % r["文件"])
        if np.any(帧 < 1) or np.any(声帧 < 1):
            问题.append("%s 有非正帧数" % r["文件"])
        if 声.size and (声.min() < 0 or 声.max() >= len(词表_缓存["声母"])
                        or 韵.max() >= len(词表_缓存["韵母"]) or 调.max() >= len(词表_缓存["调"])):
            问题.append("%s token id 越界" % r["文件"])
        # 数组 sha 与 meta 记录一致
        h = hashlib.sha256()
        for 键名, 数组 in sorted(dict(mel=mel, 声母=声, 韵母=韵, 调=调,
                                     音节帧数=帧, 声母帧数=声帧, 韵母帧数=韵帧).items()):
            数组 = np.ascontiguousarray(数组)
            h.update(键名.encode("utf-8"))
            h.update(str(数组.dtype).encode("ascii"))
            h.update(str(数组.shape).encode("ascii"))
            h.update(数组.tobytes())
        if h.hexdigest() != r["数组sha256"]:
            问题.append("%s 数组 sha 不符" % r["文件"])
        检查 += 1
    通过 = not 问题
    印("  回读自检: %d/%d 个样本 OK，问题 %d 条%s" % (
        检查, len(取), len(问题), "" if 通过 else "：" + "; ".join(问题[:5])), 印记录)
    for q in 问题[:20]:
        印("    [问题] " + q, 印记录)
    return {"通过": 通过, "检查数": 检查, "问题数": len(问题), "问题": 问题[:20]}


def 划分(n: int):
    rng = np.random.RandomState(种子)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * VAL比例)))
    val = sorted(int(i) for i in perm[:n_val])
    train = sorted(int(i) for i in perm[n_val:])
    return train, val


# ---------------------------------------------------------------- 主流程
def 主(argv=None) -> int:
    global 词表_缓存, 音节帧表_缓存
    ap = argparse.ArgumentParser(description="SpeakerA语料 → 音素 token + log-mel 数据集")
    ap.add_argument("--限", type=int, default=0, help="只处理前 N 条（自检用；输出到 自检_前N）")
    ap.add_argument("--基准meta-sha", default="", help="第二趟用：与该 sha256 比较 meta 确定性")
    ap.add_argument("--对照样本数", type=int, default=20, help="做 librosa/自实现 mel 对照的样本数")
    参 = ap.parse_args(argv)

    if not 清单路.exists():
        print("清单不存在：%s" % 清单路)
        return 2
    if not 音韵路.exists():
        print("音韵_v5.py 不存在：%s" % 音韵路)
        return 2
    音韵 = _载入音韵()

    输出根 = 产物根 / ("自检_前%d" % 参.限 if 参.限 > 0 else "")
    样本目录 = 输出根 / "样本"
    样本目录.mkdir(parents=True, exist_ok=True)
    清掉 = 0
    for p in 样本目录.glob("*.npz"):
        p.unlink()
        清掉 += 1

    印记录: list[str] = []
    印("=" * 78, 印记录)
    印("数据_SpeakerA —— SpeakerA语料 → 音素 token + log-mel 数据集", 印记录)
    印("清单  : %s" % 清单路, 印记录)
    印("音频根: %s" % 音频根, 印记录)
    印("输出  : %s" % 输出根, 印记录)
    印("参数  : sr=%d 窗=%.0fms(%d点) 跳=%.0fms(%d点) n_fft=%d mel=%d fmin=%.0f fmax=%.0f log=ln(power+%.0e)" % (
        SR, 窗毫秒, 窗长, 跳毫秒, 跳长, N_FFT, MEL带数, FMIN, FMAX, AMIN), 印记录)

    条目s = 读清单()
    总数 = len(条目s)
    用 = 条目s[:参.限] if 参.限 > 0 else 条目s

    # —— 路径映射校验 ——
    缺 = [e for e in 条目s if not e["wav"].exists()]
    印("\n[路径映射] 清单 %d 行；存在 %d；缺失 %d" % (总数, 总数 - len(缺), len(缺)), 印记录)
    if 缺:
        for e in 缺[:10]:
            印("  缺失: %s → %s" % (e["原路"], e["wav"]), 印记录)
        print("路径映射失败，停止（先把映射表报给 lead）")
        return 3
    印("  映射规则: basename(原路).strip() → %s\\<name>" % 音频根, 印记录)
    印("  例: %s → %s" % (条目s[0]["原路"], 条目s[0]["wav"]), 印记录)

    # —— 文本 → 音节（全量先解析，用于建词表；纯文本，开销小）——
    解析s列表, 汉字串s, 非汉字s, 无汉字 = [], [], [], []
    for e in 条目s:
        音节s, cn, 去数 = 解析文本(e["文本"], 音韵)
        if not 音节s:
            无汉字.append(e["序号"])
        解析s列表.append(音节s)
        汉字串s.append(cn)
        非汉字s.append(去数)
        e["非汉字数"] = 去数

    词表 = 建词表(解析s列表, 音韵)
    词表_缓存 = 词表
    总音节文本 = sum(len(s) for s in 解析s列表)
    非汉字总 = sum(非汉字s)
    印("\n[文本/词表] 汉字音节总数=%d；符/英文/数字等非汉字字符总数=%d；无汉字样本=%d" % (
        总音节文本, 非汉字总, len(无汉字)), 印记录)
    印("  vocab: 声母=%d 韵母=%d 调=%d" % (len(词表["声母"]), len(词表["韵母"]), len(词表["调"])), 印记录)
    印("  声母: " + " ".join(词表["声母"]), 印记录)
    印("  韵母: " + " ".join(词表["韵母"]), 印记录)
    印("  调  : " + " ".join(str(t) for t in 词表["调"]), 印记录)
    未命中 = [s for ss in 解析s列表 for s in ss if not s["命中"]]
    印("  解析未命中（声韵查表失败）音节数=%d" % len(未命中), 印记录)

    # —— 逐条处理 ——
    记录s, 帧数表, 音节帧表 = [], [], []
    mel和 = np.zeros(MEL带数, dtype=np.float64)
    mel平方和 = np.zeros(MEL带数, dtype=np.float64)
    总采样点 = 0
    总语音采样点 = 0
    对照差s = []
    异常汇总: list[dict] = []
    for k, e in enumerate(用):
        做对照 = k < 参.对照样本数
        记录, 数组s, 对照差, 异常 = 处理一条(e, 解析s列表[e["序号"]], 词表, 音韵, 做对照)
        写npz(样本目录 / 记录["文件"], 数组s)
        记录s.append(记录)
        帧数表.append(记录["帧数"])
        音节帧表.append(数组s["音节帧数"].tolist())
        总采样点 += int(round(记录["原时长秒"] * SR))
        总语音采样点 += int(round(记录["语音时长秒"] * SR))
        mel = 数组s["mel"].astype(np.float64)
        mel和 += mel.sum(axis=0)
        mel平方和 += np.square(mel).sum(axis=0)
        if 对照差 is not None:
            对照差s.append(对照差)
        if 异常:
            异常汇总.append({"id": 记录["id"], "wav": 记录["wav"], "异常": 异常})
        if (k + 1) % 10 == 0 or k + 1 == len(用):
            print("  … %d/%d 条（帧=%d 音节=%d）" % (k + 1, len(用), 记录["帧数"], 记录["音节数"]))

    总帧 = int(sum(帧数表))
    总音节 = int(sum(int(r["音节数"]) for r in 记录s))
    总语音秒 = 总语音采样点 / SR
    总原秒 = 总采样点 / SR
    mel均值 = mel和 / 总帧
    mel方差 = np.maximum(mel平方和 / 总帧 - np.square(mel均值), 0.0)
    mel标准差 = np.sqrt(mel方差)

    # —— 打印对照（前 5 条：文本 ↔ 音节 token ↔ 帧数）——
    音节帧表_缓存 = 音节帧表
    打印对照(用, [解析s列表[e["序号"]] for e in 用], 记录s, 印记录)

    # —— mel 对照 ——
    if 对照差s:
        印("\n[mel 对照] librosa vs 自实现（%.0f%% 帧偏移/加窗/功率口径）: 样本 %d 个，最大绝对差=%.3e" % (
            100, len(对照差s), max(对照差s)), 印记录)
    else:
        印("\n[mel 对照] 未做（--对照样本数=0）", 印记录)

    # —— 回读自检 ——
    印("\n[回读自检]", 印记录)
    回读 = 自检回读(样本目录, 记录s, min(20, len(记录s)), 印记录)

    # —— 统计 / meta ——
    train, val = 划分(总数) if 参.限 == 0 else 划分(len(用))
    if 参.限 == 0:
        train = [i for i in train if i < 总数]
        val = [i for i in val if i < 总数]
    else:
        train = [i for i in train if i < len(用)]
        val = [i for i in val if i < len(用)]

    异常列表 = list(异常汇总)
    if 无汉字:
        异常列表.append({"id": 无汉字, "异常": ["文本无汉字"]})
    if len(未命中):
        异常列表.append({"id": "全语料", "异常": ["解析未命中音节数=%d，例：%s" % (
            len(未命中), [(s["拼音"], s["声"], s["韵"]) for s in 未命中[:10]])]})

    meta = dict(
        版本="数据_SpeakerA v1",
        任务="meeting66 T-D：SpeakerA 167×60s 标注语料 → 音素 token + log-mel 数据集",
        来源=dict(清单=str(清单路), 音频根=str(音频根),
                  条数=总数, 本趟条数=len(用),
                  映射="basename(原路).strip() → 音频根\\<name>"),
        参数=dict(sr=SR, hop_ms=跳毫秒, hop_samples=跳长, win_ms=窗毫秒, win_samples=窗长,
                  n_fft=N_FFT, mel_bins=MEL带数, fmin=FMIN, fmax=FMAX,
                  logmel="ln(mel_power + %.0e)" % AMIN, window="hann(periodic)",
                  stft="librosa center=True pad_mode=constant", power=2.0,
                  vad="帧能量(25ms/10ms) 阈=全段峰值−%.0fdB，取首末浊帧裁首尾静音" % VAD_低于峰值dB),
        vocab=dict(声母=词表["声母"], 韵母=词表["韵母"], 调=[int(t) for t in 词表["调"]],
                   大小=dict(声母=len(词表["声母"]), 韵母=len(词表["韵母"]), 调=len(词表["调"]))),
        id映射=dict(声母={s: i for i, s in enumerate(词表["声母"])},
                    韵母={y: i for i, y in enumerate(词表["韵母"])},
                    调={str(t): i for i, t in enumerate(词表["调"])}),
        token格式="每音节 [声母_id, 韵母_id, 调_id]；平铺后长度=3×音节数；npz 另存 声母/韵母/调 各 [N]",
        音节帧规则="VAD 语音段内 T 帧按 N 音节均分：per=round(T/N)，余数给末音节；每音节内部 声母 25% / 韵母 75%（声母≥2帧，帧数极少时保底各≥1）",
        npz键=dict(mel="float32 [T,80] log-mel", 声母="int32 [N] 声母 id",
                   韵母="int32 [N] 韵母 id", 调="int32 [N] 调 id",
                   音节帧数="int32 [N] 每音节总帧数", 声母帧数="int32 [N] 每音节声母帧数",
                   韵母帧数="int32 [N] 每音节韵母帧数"),
        划分=dict(种子=种子, 比例="95/5", train=train, val=val,
                  train数=len(train), val数=len(val),
                  说明="样本 id 列表；RandomState(300).permutation 前 5%% 为 val"),
        统计=dict(总条数=len(记录s), 总时长秒=round(总原秒, 3), 总语音秒=round(总语音秒, 3),
                  总帧=总帧, 总音节=总音节,
                  音节率_音节每秒=round(总音节 / 总语音秒, 4) if 总语音秒 > 0 else 0.0,
                  平均每样本帧=round(总帧 / max(1, len(记录s)), 2),
                  平均每样本音节=round(总音节 / max(1, len(记录s)), 2),
                  汉字音节文本数=总音节文本, 非汉字字符数=非汉字总,
                  mel均值=[round(float(v), 6) for v in mel均值],
                  mel方差=[round(float(v), 6) for v in mel方差],
                  mel标准差=[round(float(v), 6) for v in mel标准差],
                  平均语音段占比=round(总语音秒 / 总原秒, 4) if 总原秒 > 0 else 0.0),
        样本=[dict(id=r["id"], 文件=r["文件"], wav=r["wav"], 帧数=r["帧数"], 音节数=r["音节数"],
                   原时长秒=r["原时长秒"], 语音时长秒=r["语音时长秒"],
                   裁掉首ms=r["裁掉首ms"], 裁掉尾ms=r["裁掉尾ms"],
                   文本=r["文本"], 数组sha256=r["数组sha256"]) for r in 记录s],
        异常=异常列表,
        自检=dict(librosa_vs_自实现mel最大绝对差=(max(对照差s) if 对照差s else None),
                  对照样本数=len(对照差s), 回读=回读),
    )
    文本 = json.dumps(meta, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    meta路 = 输出根 / "meta.json"
    meta路.write_text(文本, encoding="utf-8", newline="\n")
    meta_sha = hashlib.sha256(meta路.read_bytes()).hexdigest()

    # —— 汇总打印 ——
    印("\n" + "=" * 78, 印记录)
    印("[统计] 样本=%d  总时长=%.1fs(%.3fh)  语音段=%.1fs  总帧=%d  总音节=%d  音节率=%.2f 音节/s  平均语音段占比=%.1f%%" % (
        len(记录s), 总原秒, 总原秒 / 3600.0, 总语音秒, 总帧, 总音节,
        总音节 / 总语音秒 if 总语音秒 > 0 else 0.0,
        100.0 * 总语音秒 / 总原秒 if 总原秒 > 0 else 0.0), 印记录)
    印("[mel] 均值范围 [%.3f, %.3f]（80 带）方差范围 [%.3f, %.3f] 全局均值=%.4f 全局标准差=%.4f" % (
        float(mel均值.min()), float(mel均值.max()), float(mel方差.min()), float(mel方差.max()),
        float(mel均值.mean()), float(mel标准差.mean())), 印记录)
    印("[划分] train=%d val=%d（种子 %d）  val ids=%s" % (
        len(train), len(val), 种子, val), 印记录)
    印("[异常] %d 条" % len(异常列表), 印记录)
    for a in 异常列表[:20]:
        印("  " + str(a)[:300], 印记录)
    印("[路径] %s" % meta路, 印记录)
    印("[meta sha256] %s" % meta_sha, 印记录)
    基准 = (参.基准meta_sha or "").strip()
    if 基准:
        同 = meta_sha == 基准
        印("[确定性] 两趟 meta sha256 %s（本趟 %s，基准 %s）" % (
            "PASS" if 同 else "FAIL", meta_sha, 基准), 印记录)
        if not 同:
            印("  !! 两趟产出不一致，需要排查", 印记录)

    (输出根 / "自检.txt").write_text("\n".join(印记录) + "\n", encoding="utf-8", newline="\n")
    print("完成：%d 条 → %s" % (len(记录s), 输出根))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(主())
