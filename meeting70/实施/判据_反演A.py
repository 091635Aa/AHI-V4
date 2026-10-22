# -*- coding: utf-8 -*-
"""meeting70/实施/判据_反演A.py —— A 门：≤5 min（D0）真人 → 解析控制 → 器官重放（自重建）。

判据（meeting70 D7）：自重建 CER ≤0.25 才进 C。
参照 = 同段真人 ASR 转写；器官参数取 04 扫参最优臂（若未生成则用基线）。
"""
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
出 = M70 / "产物"
SR = 22050
D0目录 = M70.parent / "meeting67" / "数据" / "子集_D0_le5min" / "样本"


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


器 = 载("器官v2", HERE / "器官v2.py")
量 = 载("量具_v9", M64 / "量具_v9.py")


def 写(p, a):
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())


def 取最优参数():
    p = 出 / "G0v3R1_扫参.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        臂 = sorted(d["臂"], key=lambda r: r["均值CER"])
        return 臂[0]["参数"], 臂[0]["臂"]
    return dict(OQ=0.62, OQ1=0.42, 噪声尺度=0.25, 去加重=0.97), "A_基线"


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()
    参数, 臂名 = 取最优参数()
    meta = json.loads((M66 / "产物" / "数据_对齐asr" / "meta.json").read_text(encoding="utf-8"))
    样本 = {r["id"]: r for r in meta["样本"]}
    音频根 = Path(r"<path>\SpeakerA7代模型训练数据")
    D0 = sorted([int(f.stem) for f in D0目录.glob("*.npz")])   # 26/37/139/161/164
    段 = [(sid, 起) for sid in D0 for 起 in (10.0, 30.0)]      # 5×2 = 10 段 ×4 s
    引 = 量.载引擎()
    报 = dict(工具="判据_反演A.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
              参数=参数, 参数来源=臂名, D0=D0, 段=段, 项=[])
    cers = []
    for sid, 起 in 段:
        x = 器.耳.读wav(音频根 / 样本[sid]["wav"])
        a = int(起 * SR); b = min(len(x), a + 4 * SR)
        真 = x[a:b]
        c = 器.分析控制(真)
        y = 器.合成(c, 模式="全极点", **参数)
        p真 = 出 / f"A_{sid:04d}_{int(起)}_真人.wav"; p器 = 出 / f"A_{sid:04d}_{int(起)}_器官.wav"
        写(p真, 真); 写(p器, y)
        r真 = 引["转写"](str(p真)); r器 = 引["转写"](str(p器))
        cer = 量.CER(r真, r器) if r真 else 1.0
        cers.append(cer)
        报["项"].append(dict(sid=sid, 起=起, 真人=r真, 器官=r器, CER=round(cer, 4)))
        print("|", sid, 起, "CER=%.3f" % cer, "|", r器[:26], flush=True)
    均 = float(np.mean(cers)) if cers else 1.0
    报.update(均值CER=round(均, 4), 判=("PASS" if 均 <= 0.25 else "FAIL（≤0.25 才进 C）"),
              耗时秒=round(time.time() - t0, 1))
    (出 / "A_读数.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in 报.items() if k != "项"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(主())
