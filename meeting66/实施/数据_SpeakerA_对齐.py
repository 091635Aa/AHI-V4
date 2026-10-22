# -*- coding: utf-8 -*-
"""meeting66 / 数据_SpeakerA_对齐.py —— 用 **ASR 字符级时间戳** 给SpeakerA语料做真实对齐，
重建 TTS 数据集（与 数据_SpeakerA.py 同契约，但每音节帧数来自时间戳，而不是均分）。

※ 红线登记（meeting66/00_裁决）：ASR = **离线外部仪器**，只用于（a）训练数据的对齐时间戳，
   **不进生成路径**；标签文本仍用所有者自备清单的转写（ASR 只提供时间，不提供文本标签）。
   对齐算法：ASR 字符序列 ↔ 清单转写字符序列 做编辑距离对齐；匹配上的字符取 ASR 时间戳，
   未匹配的按邻字插值；再按「声母 25% / 韵母 75%」切分到音节内。

输出：`meeting66/产物/数据_对齐/`（meta.json + 样本/*.npz），契约与 数据_SpeakerA.py 完全一致。
只读 L: 语料；只新增 meeting66/ 下文件（C14）。
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
import wave
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal as sps

HERE = Path(__file__).resolve().parent
M66 = HERE.parent
AHI4 = M66.parent
for _p in (HERE, str(AHI4 / "meeting61" / "实施"), str(AHI4 / "meeting64" / "实施")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SR = 22050
源SR = 44100
N_FFT = 1024
窗长 = 551
跳长 = 220
MEL带数 = 80
FMIN = 0.0
FMAX = SR / 2.0
AMIN = 1e-6
种子 = 300

清单 = Path(r"<path>\SpeakerA7代模型训练数据\asr_opt\SpeakerA7代模型训练数据.list")
音频根 = Path(r"<path>\SpeakerA7代模型训练数据")
输出根 = M66 / "产物" / "数据_对齐"

import librosa  # noqa: E402
import pypinyin  # noqa: E402
from pypinyin import Style, lazy_pinyin  # noqa: E402


def 汉字(c: str) -> bool:
    return "\u4e00" <= c <= "\u9fff"


def 取汉字(s: str) -> str:
    return "".join(c for c in str(s) if 汉字(c))


def 读清单():
    out = []
    for i, line in enumerate(清单.read_text(encoding="utf-8", errors="replace").splitlines()):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        name = Path(parts[0]).name.strip()
        wav = 音频根 / name
        if not wav.exists():
            continue
        out.append(dict(序号=i, 名=name, wav=str(wav), 文本=parts[3], 说话人=parts[1] if len(parts) > 1 else "?"))
    return out


def 读音频16k(wav: str):
    x, sr = sf.read(wav, dtype="float32", always_2d=True)
    x = x.mean(axis=1, dtype=np.float32)
    if sr != 16000:
        g = math.gcd(int(sr), 16000)
        x = sps.resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
    return np.ascontiguousarray(x)


def 读音频22k(wav: str):
    x, sr = sf.read(wav, dtype="float32", always_2d=True)
    x = x.mean(axis=1, dtype=np.float32)
    if sr != SR:
        g = math.gcd(int(sr), SR)
        x = sps.resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return np.ascontiguousarray(x)


_梅尔 = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=MEL带数, fmin=FMIN, fmax=FMAX,
                            norm="slaney", htk=False)


def 算mel(x: np.ndarray) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=N_FFT, hop_length=跳长, win_length=窗长,
                                       window="hann", center=True, pad_mode="constant",
                                       power=2.0, n_mels=MEL带数, fmin=FMIN, fmax=FMAX,
                                       norm="slaney", htk=False)
    return np.log(S + AMIN).astype(np.float32).T


def 写wav16k(路径: Path, x: np.ndarray):
    with wave.open(str(路径), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2").tobytes())


def 编辑对齐(a: str, b: str):
    """返回 [(i, j)] 匹配对（a[i] ↔ b[j]），用 Levenshtein 回溯。"""
    n, m = len(a), len(b)
    距 = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        距[i][0] = i
    for j in range(1, m + 1):
        距[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            距[i][j] = min(距[i - 1][j] + 1, 距[i][j - 1] + 1,
                          距[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    i, j, 对 = n, m, []
    while i > 0 and j > 0:
        if 距[i][j] == 距[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] == b[j - 1]:
                对.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif 距[i][j] == 距[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    对.reverse()
    return 对


def 载ASR(设备: str):
    md = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "models",
                      "iic--speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
    方 = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    import funasr
    return funasr.AutoModel(model=方, disable_update=True, trust_remote_code=True,
                            device=设备, cache_dir=os.path.dirname(md))


def 一条(条目, 引, 音韵, 词表, tmp: Path, 用asr: bool = False):
    x16 = 读音频16k(条目["wav"])
    p16 = tmp / "a16.wav"
    写wav16k(p16, x16)
    r = 引.generate(input=str(p16), batch_size_s=300, return_timestamp=True)
    it = r[0] if r else {}
    块s = str(it.get("text", "")).split()
    ts = it.get("timestamp") or []
    if not 块s or len(ts) != len(块s):
        return None, ["ASR 无时间戳/长度不符 toks=%d ts=%d" % (len(块s), len(ts))]
    对s = [(c, ts[i]) for i, c in enumerate(块s) if len(c) == 1 and 汉字(c)]
    if not 对s:
        return None, ["ASR 无汉字 token"]
    文本_asr = "".join(c for c, _ in 对s)
    ts = [t for _, t in 对s]
    if 用asr:
        # N1：文本与时间戳同源（ASR），不存在“对齐错位”；标签=ASR 文本
        真文 = 文本_asr
        时间 = [(float(t[0]), float(t[1])) for t in ts]
    else:
        真文 = 取汉字(条目["文本"])
        if not 真文:
            return None, ["清单无汉字"]
        对 = 编辑对齐(真文, 文本_asr)
        时间 = [None] * len(真文)
        for i, j in 对:
            时间[i] = (float(ts[j][0]), float(ts[j][1]))
        已 = [k for k, t in enumerate(时间) if t is not None]
        if len(已) < max(3, len(真文) // 2):
            return None, ["匹配字符过少 %d/%d" % (len(已), len(真文))]
        if not 已:
            return None, ["无匹配字符"]
        # 未匹配字符：用相邻匹配字插值
        for k in range(len(时间)):
            if 时间[k] is not None:
                continue
            前 = next((x for x in reversed(已) if x < k), None)
            后 = next((x for x in 已 if x > k), None)
            if 前 is not None and 后 is not None:
                a, b = 时间[前][1], 时间[后][0]
                时间[k] = (a, max(a + 20.0, b - 20.0))
            elif 前 is not None:
                时间[k] = (时间[前][1], 时间[前][1] + 200.0)
            else:
                时间[k] = (时间[后][0] - 200.0, 时间[后][0])
    # 首末修剪到语音段；mel 用 22.05k
    x22 = 读音频22k(条目["wav"])
    起s = max(0.0, 时间[0][0] / 1000.0 - 0.05)
    止s = min(len(x22) / SR, 时间[-1][1] / 1000.0 + 0.05)
    x段 = np.ascontiguousarray(x22[int(起s * SR):int(止s * SR)])
    mel = 算mel(x段)
    T = int(mel.shape[0])
    音节s = [音韵.解析音节(p) for p in lazy_pinyin(真文, style=Style.TONE3,
                                               neutral_tone_with_five=True) if p and p[0].isascii()]
    if len(音节s) != len(真文):
        return None, ["音节数(%d)≠字数(%d)" % (len(音节s), len(真文))]
    帧起 = []
    for (a, b) in 时间:
        fa = int(round((a / 1000.0 - 起s) * 1000.0 / 10.0))
        fb = int(round((b / 1000.0 - 起s) * 1000.0 / 10.0))
        帧起.append((max(0, fa), max(0, fb)))
    # 逐音节帧数 = 下一字起点 − 本字起点（用中点法避免缝隙）
    边界 = [帧起[0][0]]
    for k in range(1, len(帧起)):
        边界.append((帧起[k - 1][1] + 帧起[k][0]) // 2)
    边界.append(T)
    边界 = [min(max(0, b), T) for b in 边界]
    for k in range(1, len(边界)):
        if 边界[k] <= 边界[k - 1]:
            边界[k] = min(T, 边界[k - 1] + 1)
    帧s = np.array([边界[k + 1] - 边界[k] for k in range(len(帧起))], np.int32)
    if int(帧s.sum()) != T or (帧s <= 0).any():
        帧s[-1] += T - int(帧s.sum())
        if (帧s <= 0).any():
            return None, ["帧分配非法"]
    声帧s, 韵帧s = [], []
    for 帧 in 帧s:
        if 帧 <= 1:
            声帧s.append(1); 韵帧s.append(0)
        elif 帧 == 2:
            声帧s.append(1); 韵帧s.append(1)
        else:
            声 = max(2, int(math.floor(帧 * 0.25 + 0.5)))
            声 = min(声, 帧 - 1)
            声帧s.append(声); 韵帧s.append(帧 - 声)
    声id = np.array([词表["声母id"][s["声"] or "∅"] for s in 音节s], np.int32)
    韵id = np.array([词表["韵母id"][s["韵"]] for s in 音节s], np.int32)
    调id = np.array([词表["调id"][s["调"]] for s in 音节s], np.int32)
    数组 = dict(mel=mel, 声母=声id, 韵母=韵id, 调=调id,
               音节帧数=帧s, 声母帧数=np.array(声帧s, np.int32), 韵母帧数=np.array(韵帧s, np.int32))
    return dict(条目=条目, 真文=真文, asr=文本_asr, 帧s=帧s, 数组=数组, T=T), []


def 写npz(路径: Path, 数组s: dict):
    缓 = io.BytesIO()
    with zipfile.ZipFile(缓, "w", zipfile.ZIP_DEFLATED) as z:
        for 键 in sorted(数组s):
            b = io.BytesIO()
            np.lib.format.write_array(b, np.ascontiguousarray(数组s[键]), allow_pickle=False)
            zi = zipfile.ZipInfo(键 + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, b.getvalue())
    p = 路径.with_name(路径.name + ".tmp")
    p.write_bytes(缓.getvalue())
    p.replace(路径)


def 主(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--前N", type=int, default=0)
    ap.add_argument("--设备", default="cuda")
    ap.add_argument("--出", default=str(输出根))
    ap.add_argument("--用ASR文本", action="store_true",
                    help="N1：标签直接用 ASR 文本（与时间戳同源，消除对齐错位）")
    args = ap.parse_args(argv)
    条目s = 读清单()
    if args.前N:
        条目s = 条目s[:args.前N]
    print("清单 %d 条；ASR 设备=%s" % (len(条目s), args.设备))
    import 音韵_v5 as 音韵
    from 数据_SpeakerA import 建词表
    拼s = [音韵.解析音节(p) for e in 条目s for p in lazy_pinyin(取汉字(e["文本"]), style=Style.TONE3,
                                                       neutral_tone_with_five=True) if p and p[0].isascii()]
    词表 = 建词表([拼s], 音韵)
    print("词表: 声母%d 韵母%d 调%d" % (len(词表["声母"]), len(词表["韵母"]), len(词表["调"])))
    引 = 载ASR(args.设备)
    出 = Path(args.出)
    (出 / "样本").mkdir(parents=True, exist_ok=True)
    tmp = 出 / "_tmp"
    tmp.mkdir(exist_ok=True)
    记录, 异常 = [], []
    总帧 = 总音节 = 0
    时长 = 0.0
    for k, e in enumerate(条目s):
        try:
            果, 异 = 一条(e, 引, 音韵, 词表, tmp, 用asr=args.用ASR文本)
        except Exception as exc:  # noqa: BLE001
            果, 异 = None, ["异常 %s: %s" % (type(exc).__name__, str(exc)[:120])]
        if 果 is None:
            异常.append(dict(id=e["序号"], wav=e["名"], 异常=异))
            continue
        写npz(出 / "样本" / ("%04d.npz" % len(记录)), 果["数组"])
        记录.append(dict(id=len(记录), 文件="%04d.npz" % len(记录), wav=e["名"],
                          文=果["真文"], asr文=果["asr"], 帧数=int(果["T"]), 音节数=len(果["帧s"]),
                          时长秒=round(int(果["T"]) * 0.01, 3)))
        总帧 += int(果["T"]); 总音节 += len(果["帧s"]); 时长 += int(果["T"]) * 0.01
        if (k + 1) % 20 == 0 or k == len(条目s) - 1:
            print("  %d/%d 已处理；有效 %d；总时长 %.1fs" % (k + 1, len(条目s), len(记录), 时长), flush=True)
    # mel 统计
    统计n = 0; 和 = np.zeros(MEL带数, np.float64); 平和 = np.zeros(MEL带数, np.float64)
    for r in 记录:
        z = np.load(出 / "样本" / r["文件"])
        m = z["mel"].astype(np.float64)
        和 += m.sum(0); 平和 += (m * m).sum(0); 统计n += m.shape[0]
    均值 = 和 / max(统计n, 1)
    方差 = np.maximum(平和 / max(统计n, 1) - 均值 ** 2, 1e-6)
    rng = np.random.default_rng(种子)
    序 = np.arange(len(记录)); rng.shuffle(序)
    val = sorted(int(x) for x in 序[:max(1, len(记录) // 20)])
    meta = dict(版本="数据_SpeakerA_对齐 v1", 任务="ASR 字符时间戳对齐重建（离线仪器，只为时间；标签=所有者清单转写）",
                来源=dict(清单=str(清单), 音频根=str(音频根), 本趟条数=len(记录), 异常=异常[:30], 异常数=len(异常), 文本来源=("ASR文本" if args.用ASR文本 else "所有者清单转写")),
                参数=dict(sr=SR, hop_ms=10, hop_samples=跳长, n_fft=N_FFT, 窗长=窗长, mel_bins=MEL带数,
                          fmin=FMIN, fmax=FMAX, logmel="ln(power+1e-06)", ASR设备=args.设备),
                词表=dict(声母=词表["声母"], 韵母=词表["韵母"], 调=词表["调"],
                        声母id=词表["声母id"], 韵母id=词表["韵母id"], 调id=词表["调id"]),
                id映射=dict(声母=词表["声母id"], 韵母=词表["韵母id"], 调=词表["调id"]),
                npz键=["mel", "声母", "韵母", "调", "音节帧数", "声母帧数", "韵母帧数"],
                token格式="每音节 [声母_id, 韵母_id, 调_id]；帧数=ASR 字符时间戳切分",
                划分=dict(train=[i for i in range(len(记录)) if i not in val], val=val,
                         train数=len(记录) - len(val), val数=len(val), 比例="95/5", 种子=种子),
                样本=记录, 统计=dict(总时长秒=round(时长, 2), 总帧=总帧, 总音节=总音节,
                                   音节率=round(总音节 / max(时长, 1e-9), 3),
                                   mel均值=[round(float(x), 6) for x in 均值],
                                   mel标准差=[round(float(x), 6) for x in np.sqrt(方差)]),
                mel统计=dict(mel均值=[round(float(x), 6) for x in 均值],
                            mel标准差=[round(float(x), 6) for x in np.sqrt(方差)],
                            mel方差=[round(float(x), 6) for x in 方差]))
    (出 / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("完成：%d 条；总时长 %.1fs；异常 %d 条；→ %s" % (len(记录), 时长, len(异常), 出))
    return 0


if __name__ == "__main__":
    raise SystemExit(主())
