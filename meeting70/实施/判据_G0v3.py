# -*- coding: utf-8 -*-
"""meeting70/实施/判据_G0v3.py —— 预注册 03 的 G0v3 判据（先写后跑）。

oracle 控制（从真人解析）→ 器官 v2 → ASR；参照 = 同一段真人的 ASR 转写。
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


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


器 = 载("器官v2", HERE / "器官v2.py")
量 = 载("量具_v9", M64 / "量具_v9.py")

# 固定 8 段（确定性）：样本 id + 起始秒 + 时长 4 s
段 = [(5, 10.0), (15, 10.0), (28, 10.0), (42, 10.0), (57, 10.0), (71, 10.0), (86, 10.0), (106, 10.0)]
时长 = 4.0


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
    模式 = sys.argv[1] if len(sys.argv) > 1 else "共振峰"
    报 = dict(工具="判据_G0v3.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 模式=模式,
              预注册="meeting70/03_预注册_G0v3.md", 段=段, 时长秒=时长, 项=[])
    cers = []
    for sid, 起 in 段:
        x = 器.耳.读wav(音频根 / 样本[sid]["wav"])
        a = int(起 * SR); b = min(len(x), a + int(时长 * SR))
        真 = x[a:b]
        c = 器.分析控制(真)
        y = 器.合成(c, 模式=模式)
        y2 = 器.合成(c, 模式=模式)
        同 = hashlib.sha256((y * 32767).astype(np.int16).tobytes()).hexdigest() == \
             hashlib.sha256((y2 * 32767).astype(np.int16).tobytes()).hexdigest()
        p真 = 出 / f"G0v3_{sid:04d}_真人.wav"; p器 = 出 / f"G0v3_{sid:04d}_器官.wav"
        写(p真, 真); 写(p器, y)
        r真 = 引["转写"](str(p真)); r器 = 引["转写"](str(p器))
        cer = 量.CER(r真, r器) if r真 else 1.0
        cers.append(cer)
        报["项"].append(dict(sid=sid, 起=起, 真人转写=r真, 器官转写=r器, CER=cer,
                           空输出=bool(not r器.strip()), 两趟同=bool(同),
                           F0中位=round(float(np.median(c["f0"][c["f0"] > 0])), 1) if (c["f0"] > 0).any() else None))
        print("|", sid, "CER=%.3f" % cer, "空" if not r器.strip() else "", "|", r真[:24], "|", r器[:24], flush=True)
    均 = float(np.mean(cers)) if cers else 1.0
    空数 = sum(1 for r in 报["项"] if r["空输出"])
    同 = all(r["两趟同"] for r in 报["项"])
    if 均 <= 0.20:
        判 = "PASS"
    elif 均 <= 0.35:
        判 = "限一轮"
    else:
        判 = "FAIL（>0.35 ⇒ 器官 v2 判死）"
    报.update(均值CER=round(均, 4), 空输出数=空数, 两趟全同=bool(同),
              判=判, 判死=(判.startswith("FAIL")), 耗时秒=round(time.time() - t0, 1))
    (出 / f"G0v3_读数_{模式}.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in 报.items() if k != "项"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(主())
