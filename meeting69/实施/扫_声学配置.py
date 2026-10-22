# -*- coding: utf-8 -*-
"""meeting69/实施/扫_声学配置.py —— 换 ckpt / 换迭代轮数，看是否有能听的配置（CAM++ 相似度 + ASR）。"""
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
量 = 载("量具_v9", M64 / "量具_v9.py")
协 = 载("合成_S0刺激", HERE / "合成_S0刺激.py")

真实 = M69 / "产物" / "S1" / "真实"
文 = "今天天气很好我听见了自己的声音我想再说一遍"

候选 = {
    "v13_e0050": M66 / "产物" / "模型_v13_K4_320" / "ckpt_e0050.pt",
    "v13_e0100": M66 / "产物" / "模型_v13_K4_320" / "ckpt_e0100.pt",
    "v13_e0150": M66 / "产物" / "模型_v13_K4_320" / "ckpt_e0150.pt",
    "v13_e0200": M66 / "产物" / "模型_v13_K4_320" / "ckpt_e0200.pt",
    "v13_e0350": M66 / "产物" / "模型_v13_K4_320" / "ckpt_e0350.pt",
    "v13_末": M66 / "产物" / "模型_v13_K4_320" / "ckpt_末.pt",
    "v12_ramp周期": M66 / "产物" / "模型_v12_ramp周期" / "ckpt.pt",
    "v6_计划采样": M66 / "产物" / "模型_v6_计划采样" / "ckpt.pt",
}


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    出.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
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
    报 = dict(工具="扫_声学配置.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 项=[])
    for 名, 路 in 候选.items():
        if not Path(路).exists():
            print("skip", 名); continue
        try:
            mdl = 声.载ckpt(Path(路), 设备="cpu")
            mel, _, _ = 声.推理mel(mdl, 文, 24)
            x = 精.mel转波_精修(mel, 迭代=256, momentum=0.99, 时间倍率=1, nfft=2048, 窗=1024)
            x = 精.收尾(x)
            p = 出 / f"扫_{名}.wav"
            协.写wav(p, x)
            转 = 引["转写"](str(p))
            项 = dict(配置=名, ckpt=str(路), 路径=str(p), CER=量.CER(文, 转), 转写=转,
                      CAM_真1=余(嵌真[0], 嵌(p)), CAM_真2=余(嵌真[1], 嵌(p)),
                      帧间=round(float(np.mean(np.abs(np.diff(mel, axis=0)))), 4))
        except Exception as e:  # noqa: BLE001
            项 = dict(配置=名, 失败=f"{type(e).__name__}: {str(e)[:120]}")
        报["项"].append(项)
        print("|", 名, 项.get("CER"), 项.get("CAM_真1"), 项.get("CAM_真2"), 项.get("转写", "")[:24] if 项.get("转写") else 项.get("失败"), flush=True)
    报["耗时秒"] = round(time.time() - t0, 1)
    (出 / "扫_声学配置.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(主())
