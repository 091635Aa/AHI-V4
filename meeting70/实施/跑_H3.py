# -*- coding: utf-8 -*-
"""meeting70/实施/跑_H3.py —— H3 示范模仿：4 段真人 → 器官照着说（读数 + 听测包）。"""
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
出 = M70 / "产物" / "H3听测包"
SR = 22050
段 = [(15, 10.0), (28, 10.0), (42, 10.0), (57, 10.0)]
时长 = 4.0


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


器 = 载("器官v2", HERE / "器官v2.py")
演 = 载("示范模仿", HERE / "示范模仿.py")
量 = 载("量具_v9", M64 / "量具_v9.py")


def 写(p, a):
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())


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

    嵌真 = [嵌(M70.parent / "meeting69" / "产物" / "S1" / "真实" / "(5)@10.wav"),
            嵌(M70.parent / "meeting69" / "产物" / "S1" / "真实" / "(64)@10.wav")]
    报 = dict(工具="跑_H3.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 项=[])
    for sid, 起 in 段:
        x = 器.耳.读wav(音频根 / 样本[sid]["wav"])
        a = int(起 * SR); b = min(len(x), a + int(时长 * SR))
        真 = x[a:b]
        p真 = 出 / f"H3_{sid:04d}_真人.wav"; p器 = 出 / f"H3_{sid:04d}_器官.wav"
        写(p真, 真)
        r = 演.模仿(p真, p器)
        # 确定性：再跑一次
        r2 = 演.模仿(p真, 出 / f"_tmp_{sid}.wav")
        同 = r["sha"] == r2["sha"]
        q真 = 引["转写"](str(p真)); q器 = 引["转写"](str(p器))
        cer = 量.CER(q真, q器) if q真 else 1.0
        报["项"].append(dict(sid=sid, 起=起, 真人=q真, 器官=q器, CER=round(cer, 4),
                            空=bool(not q器.strip()), 两趟同=bool(同),
                            CAM=余(嵌真[0], 嵌(p器))))
        print("|", sid, "CER=%.3f" % cer, "CAM=%.3f" % 报["项"][-1]["CAM"], "同", 同, "|", q器[:24], flush=True)
    通过 = sum(1 for r in 报["项"] if r["CER"] <= 0.35)
    报.update(过段=通过, 总段=len(报["项"]),
              判=("PASS（H3-1 ≥3/4）" if 通过 >= 3 else "未达标（H3-1 <3/4）"),
              耗时秒=round(time.time() - t0, 1))
    (M70 / "产物" / "H3_读数.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in 报.items() if k != "项"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(主())
