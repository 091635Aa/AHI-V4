# -*- coding: utf-8 -*-
"""meeting69/实施/诊断_声码器保真.py —— 判断「外星人」出在声码器还是出在模型 mel。

关键对照：
  A. 真人 wav → 真人 mel（与 v13 训练同口径 `数据_SpeakerA.计算logmel`）→ GL（64 / A4 2048）→ wav
     · 若 CAM++ 相似度仍高、ASR 仍可懂 ⇒ 声码器无罪，问题在模型 mel；
     · 若相似度崩 ⇒ 声码器毁音色。
  B. 生成 mel vs 真人 mel 的结构差异（带均值谱差、帧间变化比、时间方差比）。
  C. 生成 wav 与真人 wav 的 CAM++ 相似度（跨说话人判定）。
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
M66 = M69.parent / "meeting66"
M66实 = M66 / "实施"
M64 = M69.parent / "meeting64" / "实施"
出 = M69 / "产物" / "听审"
SR = 22050


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


精 = 载("音频线_v1_精修", HERE / "音频线_v1_精修.py")
声 = 精.声
耳 = 精.耳
谐 = 载("谐波重合成_v1", HERE / "谐波重合成_v1.py")
协 = 载("合成_S0刺激", HERE / "合成_S0刺激.py")
量 = 载("量具_v9", M64 / "量具_v9.py")
D = 载("数据_SpeakerA", M66实 / "数据_SpeakerA.py")


def 写(名, x):
    路 = 出 / 名
    协.写wav(路, x)
    return str(路)


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    出.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    报 = dict(工具="诊断_声码器保真.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"))

    真实 = M69 / "产物" / "S1" / "真实"
    真1 = 耳.读wav(真实 / "(5)@10.wav")
    真2 = 耳.读wav(真实 / "(64)@10.wav")

    # A. 真人 mel → GL
    mel1 = D.计算logmel(真1)
    a64 = 声.mel转波(mel1)
    a4 = 精.mel转波_精修(mel1, 迭代=256, momentum=0.99, 时间倍率=1, nfft=2048, 窗=1024)
    a64 = 精.收尾(np.asarray(a64, np.float32)); a4 = 精.收尾(np.asarray(a4, np.float32))
    路64 = 写("真人mel_GL64.wav", a64); 路A4 = 写("真人mel_GL2048.wav", a4)
    报["A_真人mel转波"] = dict(
        路径=dict(GL64=路64, GL2048=路A4),
        波形相关=dict(GL64=round(float(np.corrcoef(真1[:min(len(真1), len(a64))], a64[:min(len(真1), len(a64))])[0, 1]), 4)),
    )

    # CAM++ 相似度
    from funasr import AutoModel
    cam = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common", disable_update=True,
                    trust_remote_code=True, device="cpu")

    def 嵌(p):
        r = cam.generate(input=str(p))
        e = r[0].get("spk_embedding")
        return np.asarray(e.detach().cpu().numpy() if hasattr(e, "detach") else e).reshape(-1)

    def 余(a, b):
        return round(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))), 4)

    嵌R = [嵌(真实 / "(5)@10.wav"), 嵌(真实 / "(64)@10.wav")]
    报["A_真人mel转波"]["CAM_sim_vs_真1"] = dict(GL64=余(嵌R[0], 嵌(路64)), GL2048=余(嵌R[0], 嵌(路A4)))
    报["A_真人mel转波"]["CAM_sim_两真人"] = 余(嵌R[0], 嵌R[1])

    # B. 生成 mel vs 真人 mel 结构
    gen = 出.parent / "S2A"
    melG, _, _ = 精.推理mel("今天天气很好我听见了自己的声音我想再说一遍", 24)

    def 比(m, ref):
        m = np.asarray(m, np.float64); ref = np.asarray(ref, np.float64)
        n = min(len(m), len(ref))
        m, ref = m[:n], ref[:n]
        谱差 = (m.mean(axis=0) - ref.mean(axis=0))
        帧间 = float(np.mean(np.abs(np.diff(m, axis=0)))) / max(float(np.mean(np.abs(np.diff(ref, axis=0)))), 1e-9)
        时间方差 = float(np.mean(np.var(m, axis=0))) / max(float(np.mean(np.var(ref, axis=0))), 1e-9)
        return dict(带均值谱差dB_均值=round(float(谱差.mean()), 3), 带均值谱差dB_低频=round(float(np.mean(谱差[:20])), 3),
                    带均值谱差dB_高频=round(float(np.mean(谱差[-20:])), 3),
                    帧间变化比=round(帧间, 4), 时间方差比=round(时间方差, 4))
    报["B_结构"] = dict(生成vs真人=比(melG, mel1), 真人自身=比(mel1, mel1))

    # C. 生成 wav 相似度（跨说话人）
    生成表 = {
        "v0平静": M69 / "产物" / "S2A" / "自声" / "平静.wav",
        "v0高兴": M69 / "产物" / "S2A" / "自声" / "高兴.wav",
        "v1A4平静": M69 / "产物" / "精修v1" / "句0_A4.wav",
        "v1A4高兴P2": M69 / "产物" / "精修v1" / "高兴_A4_P2.wav",
    }
    报["C_生成wav"] = {}
    for k, p in 生成表.items():
        if p.exists():
            报["C_生成wav"][k] = dict(路径=str(p), CAM_sim_真1=余(嵌R[0], 嵌(p)), CAM_sim_真2=余(嵌R[1], 嵌(p)))

    # ASR 交叉：真人 / 真人mel转波
    引 = 量.载引擎()
    for 名, p in (("真人原声", 真实 / "(5)@10.wav"), ("真人mel_GL64", 路64), ("真人mel_GL2048", 路A4)):
        报.setdefault("D_ASR", {})[名] = 引["转写"](str(p))
    报["D_ASR"]["真人mel_GL64_CER_vs_原声"] = 量.CER(报["D_ASR"]["真人原声"], 报["D_ASR"]["真人mel_GL64"])
    报["D_ASR"]["真人mel_GL2048_CER_vs_原声"] = 量.CER(报["D_ASR"]["真人原声"], 报["D_ASR"]["真人mel_GL2048"])

    报["耗时秒"] = round(time.time() - t0, 1)
    (出 / "诊断_声码器保真.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(报, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(主())
