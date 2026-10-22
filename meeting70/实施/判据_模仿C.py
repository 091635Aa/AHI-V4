# -*- coding: utf-8 -*-
"""meeting70/实施/判据_模仿C.py —— C 信息臂读数：控制域模仿的 CAM/双 ASR（不声称过门）。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M70 = HERE.parent
M64 = M70.parent / "meeting64" / "实施"
出 = M70 / "产物"
真实 = M70.parent / "meeting69" / "产物" / "S1" / "真实"


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


量 = 载("量具_v9", M64 / "量具_v9.py")
句表 = {
    "C主句": "今天天气很好我听见了自己的声音我想再说一遍",
    "C新句1": "今天的风很轻我慢慢说给你听",
    "C新句2": "窗外的雨停了我该休息一会",
    "C新句3": "你先别走我还想再说两句",
}


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()
    报 = dict(工具="判据_模仿C.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
              说明="信息臂；A 门未过，C 不声称通过任何门", 项=[])
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
    import glob
    snap = glob.glob(str(Path.home() / ".cache" / "huggingface" / "hub" / "models--Systran--faster-whisper-medium" / "snapshots" / "*"))[0]
    from faster_whisper import WhisperModel
    wm = WhisperModel(snap, device="cpu", compute_type="int8")

    def 耳语(p):
        segs, _ = wm.transcribe(str(p), language="zh", beam_size=5)
        return "".join(s.text for s in segs).strip()

    for 名, 文 in 句表.items():
        p = 出 / f"{名}.wav"
        if not p.exists():
            报["项"].append(dict(样本=名, 缺="wav 不存在")); continue
        pf = 引["转写"](str(p)); wh = 耳语(p)
        r = dict(样本=名, 文本=文, 路径=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()[:16],
                 Paraformer=pf, Paraformer_CER=量.CER(文, pf),
                 Whisper=wh, Whisper_CER=量.CER(文, wh) if wh else 1.0,
                 CAM_真1=余(嵌真[0], 嵌(p)), CAM_真2=余(嵌真[1], 嵌(p)))
        r["CAM_均值"] = round((r["CAM_真1"] + r["CAM_真2"]) / 2, 4)
        报["项"].append(r)
        print("|", 名, "P-CER=%.3f" % r["Paraformer_CER"], "W-CER=%.3f" % r["Whisper_CER"],
              "CAM=%.3f/%.3f" % (r["CAM_真1"], r["CAM_真2"]), "|", pf[:24], flush=True)
    报["耗时秒"] = round(time.time() - t0, 1)
    (出 / "C_读数_面板.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(主())
