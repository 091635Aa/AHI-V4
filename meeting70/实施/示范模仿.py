# -*- coding: utf-8 -*-
"""meeting70/实施/示范模仿.py —— H3：给一段真人录音，器官照着说（零训练）。

用法：
  & $py 示范模仿.py <输入wav> <输出wav>
参数固定 = 预注册 04 的最优臂 E（噪声尺度 0.40）。
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M70 = HERE.parent
出 = M70 / "产物"
SR = 22050
参数 = dict(OQ=0.62, OQ1=0.42, 噪声尺度=0.40, 去加重=0.97)


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


器 = 载("器官v2", HERE / "器官v2.py")


def 写(p, a):
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())


def 模仿(输入, 输出):
    x = 器.耳.读wav(输入)
    c = 器.分析控制(x)
    y = 器.合成(c, 模式="全极点", **参数)
    写(输出, y)
    sha = hashlib.sha256((y * 32767).astype(np.int16).tobytes()).hexdigest()
    return dict(输入=str(输入), 输出=str(输出), 秒=round(len(y) / SR, 3),
                帧=c["T"], sha=sha,
                F0中位=round(float(np.median(c["f0"][c["f0"] > 0])), 1) if (c["f0"] > 0).any() else None)


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    出.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) < 3:
        print("用法: 示范模仿.py <输入wav> <输出wav>")
        return 2
    import json
    print(json.dumps(模仿(sys.argv[1], sys.argv[2]), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(主())
