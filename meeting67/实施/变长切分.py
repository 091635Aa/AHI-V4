# -*- coding: utf-8 -*-
"""meeting67/实施/变长切分.py —— 变长录音 → 训练可用的 2–60 s 片段（确定性；离线）。

按 `02_方案` §8.1 规范：
  · 片段为唯一单元（2–60 s）；
  · 25/10 ms 能量 VAD（librosa.effects.split，top_db 固定）；
  · ±50 ms 边界；
  · sha256 + 能量指纹去重；
  · 失败台账 jsonl。

输入：任意 ffmpeg 可解的长录音（mp3/wav/flac/...）。
输出：片段 wav（44.1 kHz / 2ch / 16bit，与既有 167 条同格式）+ manifest.jsonl + 失败.jsonl。

纪律：不改既有件（C14）；音频解码/重采样由 ffmpeg 完成（确定性）；同参数两趟逐位同。

用法：
  $py 变长切分.py --源 "<path>\...\长录音.mp3" --出 meeting67\数据_长录音\切分_MP3 --起秒 0 --长秒 600
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

FFMPEG = Path(r"<path>\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe")
SR16 = 16000
SR出 = 44100
块秒 = 120.0
块重叠 = 2.0


def 解码流(源: Path, 起秒: float, 长秒: float, 块回调):
    """ffmpeg 解码为 16k mono s16le，按 ~块秒 分块回调 (frame_offset, x_f32)。"""
    命 = [str(FFMPEG), "-v", "error"]
    if 起秒 > 0:
        命 += ["-ss", f"{起秒:.3f}"]
    if 长秒 > 0:
        命 += ["-t", f"{长秒:.3f}"]
    命 += ["-i", str(源), "-ac", "1", "-ar", str(SR16), "-f", "s16le", "-"]
    p = subprocess.Popen(命, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    n块 = int(块秒 * SR16)
    已收 = 0
    try:
        while True:
            raw = p.stdout.read(n块 * 2)
            if not raw:
                break
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            块回调(已收, x)
            已收 += x.size
    finally:
        p.stdout.close()
        码 = p.wait()
        if 码 != 0:
            err = p.stderr.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"ffmpeg 解码失败 {码}: {err}")


def 切块(源: Path, 起秒: float, 长秒: float, top_db: float):
    """流式 VAD → [(起, 止)]（秒，绝对时间）。"""
    import librosa
    区间 = []          # 已合并的绝对区间（秒）
    已处理 = 0.0       # 已回调的音频末尾（秒）
    重叠区 = 块重叠

    def 回调(帧偏移, x):
        nonlocal 区间, 已处理
        t0 = 帧偏移 / SR16
        if x.size < int(0.2 * SR16):
            return
        for a, b in librosa.effects.split(x, top_db=top_db, frame_length=400, hop_length=160):
            s = t0 + a / SR16
            e = t0 + b / SR16
            if s < 已处理 - 重叠区 - 0.5:   # 已处理区的重复（块间重叠），丢弃
                continue
            if 区间 and s <= 区间[-1][1] + 0.30:
                区间[-1] = (区间[-1][0], max(区间[-1][1], e))
            else:
                区间.append((s, e))
        已处理 = t0 + x.size / SR16

    解码流(源, 起秒, 长秒, 回调)
    return 区间


def 细分长段(源: Path, 区: list, 最小: float, 最大: float):
    """把 >最大 的语音区间在低能量点处切开（≤最大）。"""
    out = []
    for s, e in 区:
        if e - s <= 最大:
            out.append((s, e))
            continue
        段长 = e - s
        # 每段目标 ~ (最大*0.8) 秒；在该区间内按 10 ms 帧能量找局部最低点切
        目标 = 最大 * 0.8
        切点 = s
        while e - 切点 > 最大:
            win_s = 切点 + 目标 * 0.6
            win_e = min(切点 + 目标 * 1.4, e - 最小)
            if win_e <= win_s:
                break
            # 解码该窗（16k mono）找最小能量点
            x = _解一小段(源, win_s, win_e - win_s)
            if x.size < SR16:
                cut = 切点 + 目标
            else:
                hop = 160; fl = 400
                幅 = librosa_stft_energy(x, fl, hop)
                idx = int(np.argmin(幅))
                cut = win_s + idx * hop / SR16
            if cut - 切点 < 最小:
                cut = 切点 + 最小
            out.append((切点, cut))
            切点 = cut
        if e - 切点 >= 最小 * 0.5:
            out.append((切点, e))
    return out


def librosa_stft_energy(x, fl, hop):
    import librosa
    S = np.abs(librosa.stft(x, n_fft=fl, hop_length=hop))
    return S.sum(axis=0)


def _解一小段(源: Path, 起: float, 长: float):
    命 = [str(FFMPEG), "-v", "error", "-ss", f"{起:.3f}", "-t", f"{长:.3f}",
          "-i", str(源), "-ac", "1", "-ar", str(SR16), "-f", "s16le", "-"]
    r = subprocess.run(命, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def 提取片段(源: Path, s: float, e: float, 出wav: Path):
    命 = [str(FFMPEG), "-v", "error", "-y", "-ss", f"{s:.3f}", "-t", f"{(e-s):.3f}",
          "-i", str(源), "-ac", "2", "-ar", str(SR出), "-c:a", "pcm_s16le", str(出wav)]
    r = subprocess.run(命, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf-8", "replace")[:300])


def 指纹(出wav: Path):
    import wave
    with wave.open(str(出wav), "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
    声 = hashlib.sha256(raw).hexdigest()
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if x.size:
        每秒 = SR出 * 2  # 2ch 交错
        块s = [float(np.sqrt(np.mean(x[i:i + 每秒] ** 2) + 1e-12)) for i in range(0, x.size, 每秒)]
        能 = ",".join("%.3f" % v for v in 块s[:8])
    else:
        能 = ""
    return 声, 能


def 主(argv=None):
    ap = argparse.ArgumentParser(description="变长录音 → 2–60 s 片段（确定性）")
    ap.add_argument("--源", required=True)
    ap.add_argument("--出", required=True)
    ap.add_argument("--起秒", type=float, default=0.0)
    ap.add_argument("--长秒", type=float, default=0.0, help="0=到结尾")
    ap.add_argument("--最小", type=float, default=2.0)
    ap.add_argument("--最大", type=float, default=60.0)
    ap.add_argument("--top_db", type=float, default=30.0)
    ap.add_argument("--去重", action="store_true", default=True)
    参 = ap.parse_args(argv)
    源 = Path(参.源); 出 = Path(参.出)
    段目录 = 出 / "片段"
    段目录.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[切分] 源={源} 起={参.起秒}s 长={参.长秒 or '全部'}")
    区 = 切块(源, 参.起秒, 参.长秒, 参.top_db)
    print(f"[切分] VAD 区间 {len(区)} 个，用时 {time.time()-t0:.1f}s")
    区2 = 细分长段(源, 区, 参.最小, 参.最大)
    print(f"[切分] 细分后 {len(区2)} 段")
    manifest, failures = [], []
    见过 = set()
    for i, (s, e) in enumerate(区2):
        时长 = e - s
        if 时长 < 参.最小 * 0.9 or 时长 > 参.最大 * 1.05:
            failures.append({"序": i, "起": round(s, 3), "止": round(e, 3), "原因": "时长越界"})
            continue
        s2, e2 = max(0.0, s - 0.05), e + 0.05
        名 = f"{i:05d}_{s2:010.2f}-{e2:010.2f}.wav"
        路 = 段目录 / 名
        try:
            提取片段(源, s2, e2, 路)
            声, 能 = 指纹(路)
            if 参.去重 and 声 in 见过:
                failures.append({"序": i, "起": round(s2, 3), "止": round(e2, 3), "原因": "sha256重复"})
                路.unlink(missing_ok=True)
                continue
            见过.add(声)
            manifest.append({"序": i, "文件": str(路), "文件名": 名, "源": str(源),
                             "起秒": round(s2, 3), "止秒": round(e2, 3),
                             "时长秒": round(e2 - s2, 3), "sha256": 声, "能量指纹": 能})
        except Exception as ex:  # noqa: BLE001
            failures.append({"序": i, "起": round(s2, 3), "止": round(e2, 3),
                             "原因": f"{type(ex).__name__}: {ex}"})
    (出 / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in manifest) + ("\n" if manifest else ""),
        encoding="utf-8")
    (出 / "失败.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in failures) + ("\n" if failures else ""),
        encoding="utf-8")
    总 = sum(r["时长秒"] for r in manifest)
    报 = {"工具": "变长切分.py", "源": str(源), "起秒": 参.起秒, "长秒": 参.长秒,
          "参数": {"最小": 参.最小, "最大": 参.最大, "top_db": 参.top_db},
          "段数": len(manifest), "失败数": len(failures), "总语音秒": round(总, 1),
          "总语音小时": round(总 / 3600, 3), "耗时秒": round(time.time() - t0, 1),
          "时间": time.strftime("%Y-%m-%d %H:%M:%S")}
    (出 / "切分报告.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(报, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(主())
