# -*- coding: utf-8 -*-
"""meeting60 / refs-hi · 从原始高采样源重切 44.1kHz 单声道 PCM16 参考片段。

只做：ffmpeg 基础 seek/解码/重采样/单声道/量化 + scipy 核对（scipy 只用于
      resample_poly 核对与包络互相关，不参与生成）。
不跑任何模型（无 TTS/RVC/神经声码器/降噪）。

输入（只读）：
  EmoCompanion   <path>\\...\\数据集\\full_24k_mono.wav（24k / 1ch / s16）
  SpeakerA   meeting59/素材/SpeakerA/manifest.json 里 4 段 source_path（40k Vocals 切片）
  洛天依 meeting59/素材/洛天依/raw_干声_一梦千宵.m4s（AAC 44.1k 立体声，第三方清唱干声原始流）

输出（只写本任务目录）：
  meeting60/素材/{EmoCompanion,SpeakerA,洛天依}/ref44_*.wav
  meeting60/素材/{EmoCompanion,SpeakerA,洛天依}/manifest44.json

确定性：ffmpeg 参数固定；sha256 全量登记；两趟运行逐字节同。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

py_io = sys.stdout
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(r"<workspace>/03_AHI\AHI4")
M59 = ROOT / "meeting59" / "素材"
M60 = ROOT / "meeting60" / "素材"
FFMPEG = r"<path>\cmd\ffmpeg\bin\ffmpeg.exe"
FFPROBE = r"<path>\cmd\ffmpeg\bin\ffprobe.exe"
OUT_SR = 44100


# ---------------------------------------------------------------- 基础工具
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ffprobe_json(p: Path) -> dict:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(p)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe 失败: {p}\n{out.stderr}")
    return json.loads(out.stdout)


def ffmpeg_ver() -> str:
    out = subprocess.run([FFMPEG, "-version"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    return out.stdout.splitlines()[0].strip()


def run_ffmpeg(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError("ffmpeg 失败:\n" + " ".join(cmd) + "\n--- stderr ---\n" + r.stderr[-3000:])


def 读wav(p: Path):
    with wave.open(str(p), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        n = w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    else:
        raise RuntimeError(f"非 s16: {p} sw={sw}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def 基本读数(x: np.ndarray, sr: int) -> dict:
    if x.size == 0:
        return {"n_samples": 0}
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x ** 2)))
    return {
        "n_samples": int(x.size),
        "duration_s": round(x.size / sr, 6),
        "peak_dbfs": round(20 * np.log10(max(peak, 1e-12)), 3),
        "rms_dbfs": round(20 * np.log10(max(rms, 1e-12)), 3),
        "clip_frac_ge_32700": round(float(np.mean(np.abs(x) >= 32700 / 32768.0)), 8),
    }


def 包络核对(x44: np.ndarray, sr44: int, ref16: np.ndarray, sr16: int,
             hop=0.010, win=0.020) -> dict:
    """44.1k 片段 vs meeting59 的 16k 参考片段：能量包络零延迟互相关。

    ref44 用 scipy resample_poly 降到 16k 后与 ref16k 比包络（帧 20ms/跳 10ms），
    在 ±50ms 内找最佳延迟；正确切窗应 best_lag≈0 且 corr 高。
    """
    from scipy.signal import resample_poly

    n16 = int(round(ref16.size * sr16 / sr16))
    ref16 = ref16[:n16]
    y = resample_poly(x44, 160, 441).astype(np.float64)  # 44100 -> 16000
    L = min(y.size, ref16.size)

    def env(z):
        nf = max(1, 1 + (z.size - int(win * sr16)) // int(hop * sr16))
        fr = np.lib.stride_tricks.sliding_window_view(z, int(win * sr16))[:: int(hop * sr16)]
        e = np.sqrt(np.mean(fr[:nf] ** 2, axis=1) + 1e-12)
        return 20 * np.log10(e)

    ea, eb = env(y[:L]), env(ref16[:L])
    nf = min(ea.size, eb.size)
    ea, eb = ea[:nf], eb[:nf]
    max_lag = int(round(0.050 / hop))
    best = (-2.0, 0)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = ea[-lag:], eb[:nf + lag]
        elif lag > 0:
            a, b = ea[:nf - lag], eb[lag:]
        else:
            a, b = ea, eb
        if a.size < 20:
            continue
        a = a - a.mean()
        b = b - b.mean()
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        if c > best[0]:
            best = (c, lag)
    # 零延迟（不搜索）的原始相关，便于判断内容是否同窗
    a0 = ea - ea.mean()
    b0 = eb - eb.mean()
    c0 = float(np.dot(a0, b0) / (np.linalg.norm(a0) * np.linalg.norm(b0) + 1e-12))
    rms16 = float(np.sqrt(np.mean(ref16[:L] ** 2)))
    rms44 = float(np.sqrt(np.mean(x44 ** 2)))
    return {
        "对照文件": None,  # 调用方填
        "包络相关_零延迟": round(c0, 4),
        "包络相关_最佳": round(best[0], 4),
        "最佳延迟_ms": round(best[1] * hop * 1000, 1),
        "RMS差_dB_44k减16k": round(20 * np.log10(max(rms44, 1e-12) / max(rms16, 1e-12)), 3),
    }


def 命令串(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd)


# ---------------------------------------------------------------- 任务定义
def 读meeting59角色manifest(角色: str) -> dict:
    return json.loads((M59 / 角色 / "manifest.json").read_text(encoding="utf-8"))


def EmoCompanion任务() -> list[dict]:
    m = 读meeting59角色manifest("EmoCompanion")
    tasks = []
    for seg in m["片段"]:
        tasks.append({
            "角色": "EmoCompanion",
            "file": "ref44_EmoCompanion_" + seg["file"].split("_")[-1].replace(".wav", "") + ".wav",
            "source_path": seg["source_path"],
            "source_start_s": seg["source_start_s"],
            "source_end_s": seg["source_end_s"],
            "expected_duration_s": seg["duration_s"],
            "ref16k": M59 / "EmoCompanion" / seg["file"],
            "源带宽上界_hz": 24000 / 2,
            "源采样率": seg["source_sample_rate"],
            "源声道": seg["source_channels"],
            "seek_mode": "input",  # -ss 在 -i 前（wav 可精确定位）
        })
    return tasks


def SpeakerA任务() -> list[dict]:
    m = 读meeting59角色manifest("SpeakerA")
    tasks = []
    for seg in m["片段"]:
        tasks.append({
            "角色": "SpeakerA",
            "file": "ref44_SpeakerA_" + seg["file"].split("_")[-1].replace(".wav", "") + ".wav",
            "source_path": seg["source_path"],
            "source_start_s": seg["source_start_s"],
            "source_end_s": seg["source_end_s"],
            "expected_duration_s": seg["duration_s"],
            "ref16k": M59 / "SpeakerA" / seg["file"],
            "源带宽上界_hz": 40000 / 2,
            "源采样率": seg["source_sample_rate"],
            "源声道": seg["source_channels"],
            "seek_mode": "input",
        })
    return tasks


def 洛天依任务() -> list[dict]:
    m = json.loads((M59 / "洛天依" / "manifest.json").read_text(encoding="utf-8"))
    src = next(s for s in m["sources"] if s["source_id"] == "L-C")
    raw = ROOT / src["raw_file"]["path"].replace("03_AHI/AHI4/", "")
    tasks = []
    for seg in m["segments"]:
        if "清唱" not in seg["file"]:
            continue
        idx = seg["file"].split("_")[-1].replace(".wav", "")
        tasks.append({
            "角色": "洛天依",
            "file": f"ref44_洛天依_清唱_{idx}.wav",
            "source_path": str(raw),
            "source_start_s": seg["t0_s_in_source"],
            "source_end_s": seg["t1_s_in_source"],
            "expected_duration_s": round(seg["t1_s_in_source"] - seg["t0_s_in_source"], 6),
            "ref16k": M59 / "洛天依" / seg["file"],
            "源带宽上界_hz": 44100 / 2,  # 原始流 44.1k AAC ⇒ 22.05k 上界
            "源采样率": 44100,
            "源声道": 2,
            "seek_mode": "input",
        })
    return tasks


def 跑一个任务(t: dict, role_dir: Path) -> dict:
    out = role_dir / t["file"]
    src = Path(t["source_path"])
    t0 = float(t["source_start_s"])
    dur = float(t["expected_duration_s"])
    cmd = [
        FFMPEG, "-y", "-v", "error",
        "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}",
        "-i", str(src),
        "-ac", "1", "-ar", str(OUT_SR), "-c:a", "pcm_s16le",
        str(out),
    ]
    命令串完整 = 命令串(cmd)
    run_ffmpeg(cmd)

    x, sr = 读wav(out)
    probe = ffprobe_json(out)
    st = probe["streams"][0]
    fmt = probe["format"]
    expected_n = int(round(OUT_SR * dur))
    读数 = 基本读数(x, sr)

    条目 = {
        "file": t["file"],
        "source_path": t["source_path"],
        "source_sha256": t["source_sha256"],
        "source_sample_rate": t["源采样率"],
        "source_channels": t["源声道"],
        "source_start_s": t0,
        "source_end_s": float(t["source_end_s"]),
        "source_bandwidth_upper_hz": t["源带宽上界_hz"],
        "out_sample_rate": st["sample_rate"],
        "out_channels": st["channels"],
        "out_sample_format": st["codec_name"],
        "out_sha256": sha256_file(out),
        "out_size_bytes": out.stat().st_size,
        "duration_expected_s": round(dur, 6),
        "duration_readout_s": round(float(fmt["duration"]), 6),
        "n_samples_expected": expected_n,
        "n_samples_readout": 读数["n_samples"],
        "n_samples_diff": 读数["n_samples"] - expected_n,
        "sample_rate_ok": int(st["sample_rate"]) == OUT_SR,
        "channels_ok": int(st["channels"]) == 1,
        "format_ok": st["codec_name"] == "pcm_s16le",
        "peak_dbfs": 读数["peak_dbfs"],
        "rms_dbfs": 读数["rms_dbfs"],
        "clip_frac_ge_32700": 读数["clip_frac_ge_32700"],
        "ffmpeg_cmd": 命令串完整,
    }
    # 与 meeting59 的 16k 参考同窗核对
    if t.get("ref16k") and Path(t["ref16k"]).exists():
        rx, rsr = 读wav(Path(t["ref16k"]))
        chk = 包络核对(x, sr, rx, rsr)
        chk["对照文件"] = str(t["ref16k"])
        条目["对照_meeting59_16k"] = chk
    return 条目


def main() -> int:
    print("=" * 78)
    print("meeting60 · refs-hi · 44.1k 高采样参考重切")
    print("ffmpeg:", ffmpeg_ver())
    print("=" * 78)

    specs = [
        ("EmoCompanion", EmoCompanion任务(), "24k 源（12k 带宽上界）"),
        ("SpeakerA", SpeakerA任务(), "40k Vocals 切片源（20k 带宽上界）"),
        ("洛天依", 洛天依任务(), "44.1k AAC 原始流（22.05k 带宽上界）"),
    ]

    # 源文件 sha256 缓存（EmoCompanion 78MB 只算一次）
    sha_cache: dict[str, str] = {}
    for 角色, tasks, 说明 in specs:
        for t in tasks:
            if t["source_path"] not in sha_cache:
                print(f"  sha256 源文件: {t['source_path']} ...", flush=True)
                sha_cache[t["source_path"]] = sha256_file(Path(t["source_path"]))
            t["source_sha256"] = sha_cache[t["source_path"]]

    for 角色, tasks, 说明 in specs:
        role_dir = M60 / 角色
        role_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {角色}：{len(tasks)} 段 · {说明} ---")
        片段 = []
        for t in tasks:
            print(f"  切 {t['file']}  [{t['source_start_s']:.3f}, {t['source_end_s']:.3f}] ...",
                  flush=True)
            条目 = 跑一个任务(t, role_dir)
            flag = "OK" if (条目["sample_rate_ok"] and 条目["channels_ok"]
                            and 条目["format_ok"] and abs(条目["n_samples_diff"]) <= 2) else "!!"
            print(f"    [{flag}] sr={条目['out_sample_rate']} ch={条目['out_channels']} "
                  f"n={条目['n_samples_readout']} (期望 {条目['n_samples_expected']}, "
                  f"差 {条目['n_samples_diff']}) rms={条目['rms_dbfs']} dBFS")
            if "对照_meeting59_16k" in 条目:
                c = 条目["对照_meeting59_16k"]
                print(f"    对照16k: 包络相关零延迟={c['包络相关_零延迟']} "
                      f"最佳={c['包络相关_最佳']}@{c['最佳延迟_ms']}ms "
                      f"ΔRMS={c['RMS差_dB_44k减16k']}dB")
            片段.append(条目)

        manifest = {
            "角色": 角色,
            "生成时间": datetime.now().astimezone().isoformat(timespec="seconds"),
            "written_by": "refs-hi (team meeting60)",
            "任务": "从原始高采样源重切 44.1kHz/单声道/PCM16 参考片段；与 meeting59 已用片段同时间窗",
            "口径": ("参考音频仅本机声学参数估计用；参数化模仿、不是声克隆/角色本人；"
                     "不用于训练外部模型；参考音频不再分发。"),
            "处理链": ("ffmpeg 基础 seek（-ss 在 -i 前，输入定位）/ -t 截长 / 解码 / "
                       "重采样到 44.1k（libswresample 默认）/ 降为单声道 / pcm_s16le 量化；"
                       "无降噪/无归一化/无EQ/无TTS/无RVC/无神经声码器"),
            "ffmpeg": FFMPEG,
            "ffmpeg_版本": ffmpeg_ver(),
            "重采样说明": {
                "EmoCompanion": "24000→44100 上采样（源带宽上界 12000 Hz，重采样不产生 12k 以上新内容）",
                "SpeakerA": "40000→44100 上采样（源带宽上界 20000 Hz）",
                "洛天依": "44100→44100 无需重采样；AAC 立体声→单声道（libswresample 默认下混）",
            }[角色],
            "源带宽上界_hz": {"EmoCompanion": 12000, "SpeakerA": 20000, "洛天依": 22050}[角色],
            "输出": {
                "out_sample_rate": OUT_SR,
                "out_channels": 1,
                "out_sample_format": "pcm_s16le",
            },
            "片段": 片段,
            "合计时长_s": round(sum(p["duration_readout_s"] for p in 片段), 6),
            "核对汇总": {
                "全部sr=44100": all(p["sample_rate_ok"] for p in 片段),
                "全部单声道": all(p["channels_ok"] for p in 片段),
                "全部pcm_s16le": all(p["format_ok"] for p in 片段),
                "全部样本数误差≤2": all(abs(p["n_samples_diff"]) <= 2 for p in 片段),
                "全部无削波": all(p["peak_dbfs"] < -0.1 for p in 片段),
                "与16k参考同窗核对_零延迟最小包络相关": round(min(
                    p["对照_meeting59_16k"]["包络相关_零延迟"]
                    for p in 片段 if "对照_meeting59_16k" in p), 4) if any(
                    "对照_meeting59_16k" in p for p in 片段) else None,
            },
        }
        (role_dir / "manifest44.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  → {role_dir / 'manifest44.json'}")

    print("\n完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
