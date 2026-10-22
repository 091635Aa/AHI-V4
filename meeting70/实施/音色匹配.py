# -*- coding: utf-8 -*-
"""meeting70/实施/音色匹配.py —— T1/T2：闭式静态 EQ 把器官输出往真人 LTAS 推（零训练）。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M70 = HERE.parent
M66 = M70.parent / "meeting66"
M64 = M70.parent / "meeting64" / "实施"
出 = M70 / "产物" / "音色匹配"
SR = 22050


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


器 = 载("器官v2", HERE / "器官v2.py")
演 = 载("示范模仿", HERE / "示范模仿.py")
量 = 载("量具_v9", M64 / "量具_v9.py")
Lib = 器.耳 and None
import librosa

段 = [(15, 10.0), (28, 10.0), (42, 10.0), (57, 10.0)]


def 写(p, a):
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())


def ltas(x):
    S = np.abs(librosa.stft(np.asarray(x, np.float64), n_fft=1024, hop_length=220,
                            win_length=551, window="hann", center=True))
    return np.mean(S, axis=1) + 1e-9


def eq(器官, 真人, 迭代=1):
    x = np.asarray(器官, np.float64)
    S = librosa.stft(x, n_fft=1024, hop_length=220, win_length=551, window="hann", center=True)
    g = ltas(真人) / ltas(x)
    # 5 点平滑 + 钳位
    k = np.ones(5) / 5
    g = np.convolve(g, k, mode="same")
    g = np.clip(g, 0.25, 4.0)
    y = x
    for _ in range(int(迭代)):
        S = librosa.stft(y, n_fft=1024, hop_length=220, win_length=551, window="hann", center=True)
        y = librosa.istft(S * g[:, None], hop_length=220, win_length=551, n_fft=1024,
                          window="hann", center=True)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        p = float(np.sqrt(np.mean(x ** 2))) + 1e-9
        y = y / (float(np.sqrt(np.mean(y ** 2))) + 1e-9) * p
        if 迭代 > 1:
            g = 0.5 * g + 0.5 * (ltas(真人) / ltas(y))
            g = np.clip(np.convolve(g, np.ones(5) / 5, mode="same"), 0.25, 4.0)
    return y.astype(np.float32)


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    出.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    meta = json.loads((M66 / "产物" / "数据_对齐asr" / "meta.json").read_text(encoding="utf-8"))
    样本 = {r["id"]: r for r in meta["样本"]}
    音频根 = Path(r"<path>\SpeakerA7代模型训练数据")
    真目录 = M70.parent / "meeting69" / "产物" / "S1" / "真实"
    引 = 量.载引擎()
    from funasr import AutoModel
    cam = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common", disable_update=True,
                    trust_remote_code=True, device="cpu")

    def 嵌(p):
        r = cam.generate(input=str(p))
        e = r[0].get("spk_embedding")
        return np.asarray(e.detach().cpu().numpy() if hasattr(e, "detach") else e).reshape(-1)

    def 余(a, b):
        return round(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))), 4)

    嵌真 = [嵌(真目录 / "(5)@10.wav"), 嵌(真目录 / "(64)@10.wav")]
    报 = dict(工具="音色匹配.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 项=[])
    for sid, 起 in 段:
        x = 器.耳.读wav(音频根 / 样本[sid]["wav"])
        a = int(起 * SR); b = min(len(x), a + 4 * SR)
        真 = x[a:b]
        p真 = 出 / f"T_{sid:04d}_真人.wav"; 写(p真, 真)
        porg = 出 / f"T_{sid:04d}_器官.wav"
        演.模仿(p真, porg)
        org = 器.耳.读wav(porg)
        row = dict(sid=sid, 真人ASR=引["转写"](str(p真)), 项={})
        for 名, 迭代 in (("T1", 1), ("T2", 2)):
            y = eq(org, 真, 迭代)
            py = 出 / f"T_{sid:04d}_{名}.wav"; 写(py, y)
            sha1 = hashlib.sha256((y * 32767).astype(np.int16).tobytes()).hexdigest()[:16]
            y2 = eq(org, 真, 迭代)
            sha2 = hashlib.sha256((y2 * 32767).astype(np.int16).tobytes()).hexdigest()[:16]
            r = 引["转写"](str(py))
            camv = [余(嵌真[0], 嵌(py)), 余(嵌真[1], 嵌(py))]
            row["项"][名] = dict(CAM=camv, CAM均值=round(sum(camv) / 2, 4),
                               ASR=r, CER=量.CER(row["真人ASR"], r), 两趟同=(sha1 == sha2))
            print("|", sid, 名, "CAM=%.3f" % row["项"][名]["CAM均值"], "CER=%.3f" % row["项"][名]["CER"],
                  "同", sha1 == sha2, flush=True)
        # 基线（未 EQ）
        r0 = 引["转写"](str(porg))
        c0 = [余(嵌真[0], 嵌(porg)), 余(嵌真[1], 嵌(porg))]
        row["基线"] = dict(CAM=c0, CAM均值=round(sum(c0) / 2, 4), CER=量.CER(row["真人ASR"], r0))
        报["项"].append(row)
    for 名 in ("基线", "T1", "T2"):
        m = float(np.mean([r["项"][名]["CAM均值"] if 名 != "基线" else r["基线"]["CAM均值"] for r in 报["项"]]))
        报[f"均值_CAM_{名}"] = round(m, 4)
    报["耗时秒"] = round(time.time() - t0, 1)
    (M70 / "产物" / "音色匹配_读数.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in 报.items() if k.startswith("均值")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(主())
