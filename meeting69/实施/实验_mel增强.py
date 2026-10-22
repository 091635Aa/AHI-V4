# -*- coding: utf-8 -*-
"""meeting69/实施/实验_mel增强.py —— 直接修模型 mel 的过平滑（诊断已证明：声码器无罪，模型 mel 才是病灶）。

诊断（`诊断_声码器保真.json`）：真人 mel→GL 的 CAM++ 相似度 0.9285、ASR CER 0.0；而生成 wav 与真人
相似度仅 0.13–0.23、生成 mel 帧间变化仅为真人 0.376、时间方差仅 0.184 ⇒ 过平滑是主因。
本件只做**推理端** mel 对比度恢复（不训练、不改通道）：逐带统计匹配 + 时间锐化。
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M69 = HERE.parent
M64 = M69.parent / "meeting64" / "实施"
出 = M69 / "产物" / "听审"
SR = 22050


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


精 = 载("音频线_v1_精修", HERE / "音频线_v1_精修.py")
耳 = 精.耳
D = 载("数据_SpeakerA", M69.parent / "meeting66" / "实施" / "数据_SpeakerA.py")
量 = 载("量具_v9", M64 / "量具_v9.py")
协 = 载("合成_S0刺激", HERE / "合成_S0刺激.py")

真实 = M69 / "产物" / "S1" / "真实"
文 = "今天天气很好我听见了自己的声音我想再说一遍"


def 移动均值(m, k=3):
    pad = k // 2
    p = np.pad(m, ((pad, pad), (0, 0)), mode="edge")
    ker = np.ones(k) / k
    out = np.empty_like(m)
    for b in range(m.shape[1]):
        out[:, b] = np.convolve(p[:, b], ker, mode="valid")[: m.shape[0]]
    return out


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    出.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    真 = [耳.读wav(真实 / "(5)@10.wav"), 耳.读wav(真实 / "(64)@10.wav")]
    真mel = np.concatenate([D.计算logmel(x) for x in 真], axis=0)
    μ真 = 真mel.mean(axis=0); σ真 = np.maximum(真mel.std(axis=0), 1e-6)
    δ真 = np.mean(np.abs(np.diff(真mel, axis=0)))
    μ真谱 = 真mel.mean(axis=1).mean() if False else 0.0

    melG, _, _ = 精.推理mel(文, 24)
    μG = melG.mean(axis=0); σG = np.maximum(melG.std(axis=0), 1e-6)
    缩放 = np.clip(σ真 / σG, 1.0, 4.0)

    def 统计匹配(m):
        return np.clip((m - μG[None, :]) * 缩放[None, :] + μ真[None, :], -40, 40)

    def 时间锐化(m, α=0.5):
        return np.clip(m + α * (m - 移动均值(m, 3)), -40, 40)

    变体 = {
        "G0_原始": melG,
        "G1_统计匹配": 统计匹配(melG),
        "G2_匹配加锐化": 时间锐化(统计匹配(melG)),
        "G3_仅锐化": 时间锐化(melG),
    }

    from funasr import AutoModel
    cam = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common", disable_update=True,
                    trust_remote_code=True, device="cpu")

    def 嵌(p):
        r = cam.generate(input=str(p))
        e = r[0].get("spk_embedding")
        return np.asarray(e.detach().cpu().numpy() if hasattr(e, "detach") else e).reshape(-1)

    def 余(a, b):
        return round(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))), 4)

    嵌真 = [嵌(真实 / "(5)@10.wav"), 嵌(真实 / "(64)@10.wav")]
    引 = 量.载引擎()

    报 = dict(工具="实验_mel增强.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
              真mel统计=dict(均值=round(float(μ真.mean()), 3), 标准差=round(float(σ真.mean()), 3),
                           帧间变化=round(float(δ真), 4)),
              缩放=dict(均值=round(float(缩放.mean()), 3), 最小=round(float(缩放.min()), 3), 最大=round(float(缩放.max()), 3)),
              项=[])
    for 名, m in 变体.items():
        x = 精.mel转波_精修(m, 迭代=256, momentum=0.99, 时间倍率=1, nfft=2048, 窗=1024)
        x = 精.收尾(x)
        路 = 出 / f"增强_{名}.wav"
        协.写wav(路, x)
        转 = 引["转写"](str(路))
        项 = dict(变体=名, 路径=str(路), CER=量.CER(文, 转), 转写=转,
                  CAM_真1=余(嵌真[0], 嵌(路)), CAM_真2=余(嵌真[1], 嵌(路)),
                  帧间=round(float(np.mean(np.abs(np.diff(m, axis=0)))), 4))
        报["项"].append(项)
        print("|", 名, "CER=%.3f" % 项["CER"], "CAM", 项["CAM_真1"], 项["CAM_真2"], "帧间", 项["帧间"], "|", 转[:30], flush=True)
    报["耗时秒"] = round(time.time() - t0, 1)
    (出 / "实验_mel增强.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(主())
