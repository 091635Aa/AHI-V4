# -*- coding: utf-8 -*-
"""meeting69/实施/判据_拼接声.py —— T1 M1/M2 判据读数（CAM++ / 双 ASR / 确定性）。

零训练；只读产物。ASR 与 CAM++ 均为离线仪器（C10）；
Qwen3-Omni 的听感由 `跑_Omni听审.ps1` 单独产出（llama.cpp 本地）。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M69 = HERE.parent
M64 = M69.parent / "meeting64" / "实施"
出 = M69 / "产物" / "拼接v1"
真实 = M69 / "产物" / "S1" / "真实"


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


量 = 载("量具_v9", M64 / "量具_v9.py")

项 = {
    "主句": ("今天天气很好我听见了自己的声音我想再说一遍", 出 / "拼接_主句.wav"),
    "M2a": ("今天的风很轻我慢慢说给你听", 出 / "拼接_M2a.wav"),
    "M2b": ("窗外的雨停了我该休息一会", 出 / "拼接_M2b.wav"),
    "M2c": ("你先别走我还想再说两句", 出 / "拼接_M2c.wav"),
}


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()
    报 = dict(工具="判据_拼接声.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 项=[])
    # CAM++
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
    # Paraformer
    引 = 量.载引擎()
    # Whisper（faster-whisper，CPU）
    import glob
    snap = glob.glob(str(Path.home() / ".cache" / "huggingface" / "hub" / "models--Systran--faster-whisper-medium" / "snapshots" / "*"))[0]
    from faster_whisper import WhisperModel
    wm = WhisperModel(snap, device="cpu", compute_type="int8")

    def 耳语(p):
        segs, _ = wm.transcribe(str(p), language="zh", beam_size=5)
        return "".join(s.text for s in segs).strip()

    for 名, (文, 路) in 项.items():
        if not 路.exists():
            报["项"].append(dict(样本=名, 缺="wav 不存在")); continue
        pf = 引["转写"](str(路))
        wh = 耳语(路)
        项r = dict(样本=名, 文本=文, 路径=str(路), sha256=hashlib.sha256(路.read_bytes()).hexdigest(),
                  Paraformer=pf, Paraformer_CER=量.CER(文, pf),
                  Whisper=wh, Whisper_CER=量.CER(文, wh) if wh else 1.0,
                  CAM_真1=余(嵌真[0], 嵌(路)), CAM_真2=余(嵌真[1], 嵌(路)))
        项r["CAM_均值"] = round((项r["CAM_真1"] + 项r["CAM_真2"]) / 2, 4)
        报["项"].append(项r)
        print("|", 名, "P-CER=%.3f" % 项r["Paraformer_CER"], "W-CER=%.3f" % 项r["Whisper_CER"],
              "CAM=%.3f/%.3f" % (项r["CAM_真1"], 项r["CAM_真2"]), "|", pf[:24], "|", wh[:24], flush=True)
    报["耗时秒"] = round(time.time() - t0, 1)
    (出 / "读数_拼接声.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(主())
