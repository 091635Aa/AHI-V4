# -*- coding: utf-8 -*-
"""meeting70/实施/扫_器官参数.py —— 预注册 04：器官 v2.1 限一轮改进（7 臂，同一 8 段）。"""
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


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


器 = 载("器官v2", HERE / "器官v2.py")
量 = 载("量具_v9", M64 / "量具_v9.py")

段 = [(5, 10.0), (15, 10.0), (28, 10.0), (42, 10.0), (57, 10.0), (71, 10.0), (86, 10.0), (106, 10.0)]
候选 = [
    ("A_基线", dict(OQ=0.62, OQ1=0.42, 噪声尺度=0.25, 去加重=0.97)),
    ("B_长开相", dict(OQ=0.75, OQ1=0.50, 噪声尺度=0.25, 去加重=0.97)),
    ("C_短开相", dict(OQ=0.50, OQ1=0.35, 噪声尺度=0.25, 去加重=0.97)),
    ("D_弱噪声", dict(OQ=0.62, OQ1=0.42, 噪声尺度=0.15, 去加重=0.97)),
    ("E_强噪声", dict(OQ=0.62, OQ1=0.42, 噪声尺度=0.40, 去加重=0.97)),
    ("F_去加重090", dict(OQ=0.62, OQ1=0.42, 噪声尺度=0.25, 去加重=0.90)),
    ("G_去加重099", dict(OQ=0.62, OQ1=0.42, 噪声尺度=0.25, 去加重=0.99)),
]


def 写(p, a):
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()
    meta = json.loads((M66 / "产物" / "数据_对齐asr" / "meta.json").read_text(encoding="utf-8"))
    样本 = {r["id"]: r for r in meta["样本"]}
    音频根 = Path(r"<path>\SpeakerA7代模型训练数据")
    引 = 量.载引擎()
    报 = dict(工具="扫_器官参数.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
              预注册="meeting70/04_预注册_器官v2.1_限一轮.md", 臂=[])
    # 先缓存控制 + 真人转写（一次）
    控制 = []
    for sid, 起 in 段:
        x = 器.耳.读wav(音频根 / 样本[sid]["wav"])
        a = int(起 * SR); b = min(len(x), a + 4 * SR)
        真 = x[a:b]
        c = 器.分析控制(真)
        p真 = 出 / f"G0v3R1_{sid:04d}_真人.wav"; 写(p真, 真)
        r真 = 引["转写"](str(p真))
        控制.append((sid, c, r真))
    for 名, kw in 候选:
        cers = []; 空 = 0
        for sid, c, r真 in 控制:
            y = 器.合成(c, 模式="全极点", **kw)
            y2 = 器.合成(c, 模式="全极点", **kw)
            同 = hashlib.sha256((y * 32767).astype(np.int16).tobytes()).hexdigest() == \
                 hashlib.sha256((y2 * 32767).astype(np.int16).tobytes()).hexdigest()
            p = 出 / f"G0v3R1_{名}_{sid:04d}.wav"
            写(p, y)
            r = 引["转写"](str(p))
            if not r.strip():
                空 += 1
            cers.append(量.CER(r真, r) if r真 else 1.0)
        m = float(np.mean(cers))
        报["臂"].append(dict(臂=名, 参数=kw, 均值CER=round(m, 4), 空输出=空,
                           逐段=[round(v, 3) for v in cers]))
        print("|", 名, "均值CER=%.4f" % m, "空", 空, flush=True)
    报["耗时秒"] = round(time.time() - t0, 1)
    (出 / "G0v3R1_扫参.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(主())
