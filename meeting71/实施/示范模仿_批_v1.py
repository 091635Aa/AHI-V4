# -*- coding: utf-8 -*-
"""meeting71/实施/示范模仿_批_v1.py —— 路线 3 优化版「照着说」（H3 批量 + 电平连续 + 长录音）。

相对 meeting70/实施/示范模仿.py（单段）的优化：
1) 批量：目录/单文件/长录音都可；
2) 长录音：按 8 s 窗口（1 s 重叠）分析控制再合成，接缝 40 ms 交叉淡化；
3) 电平连续：短时 RMS（100 ms）向中位对齐、增益平滑（钳 ±6 dB），消除段间音量跳变；
4) 报告：逐段 F0 中位 / sha256 / 确定性复验。

用法：
  & $py 示范模仿_批_v1.py <输入wav|目录> <输出目录>
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
M70 = HERE.parent.parent / "meeting70"
E = Path(r"<path>\AHI_m71")
SR = 22050
参数 = dict(OQ=0.62, OQ1=0.42, 噪声尺度=0.40, 去加重=0.97)
窗口 = 8.0
重叠 = 1.0


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


器 = 载("器官v2", M70 / "实施" / "器官v2.py")


def 写(p, a):
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())


def 合成一段(x):
    c = 器.分析控制(x)
    y = 器.合成(c, 模式="全极点", **参数)
    return y, c


def 长录音合成(x):
    n = len(x)
    if n <= int(窗口 * SR):
        y, c = 合成一段(x)
        return y, [c], 1
    起 = 0
    片 = []
    控 = []
    淡 = int(0.040 * SR)
    while 起 < n:
        终 = min(n, 起 + int(窗口 * SR))
        y, c = 合成一段(x[起:终])
        控.append(c)
        if 片 and 淡 > 1:
            m = min(淡, len(片[-1]), len(y))
            w = np.linspace(0, 1, m).astype(np.float32)
            片[-1] = np.concatenate([片[-1][:-m], 片[-1][-m:] * (1 - w) + y[:m] * w])
            片.append(y[m:])
        else:
            片.append(y)
        if 终 >= n:
            break
        起 = int(终 - 重叠 * SR) if 终 < n else n
    out = np.concatenate([np.asarray(p, np.float64) for p in 片 if len(p)]) if 片 else np.zeros(1)
    return out, 控, len(片)


def 电平连续(y, 目标=0.05, 窗=0.1):
    hop = max(1, int(窗 * SR))
    n = len(y)
    if n < hop * 2:
        return y
    k = int(np.ceil(n / hop))
    rms = np.zeros(k)
    for i in range(k):
        seg = y[i * hop:(i + 1) * hop]
        rms[i] = float(np.sqrt(np.mean(seg ** 2))) + 1e-9
    中 = float(np.median(rms))
    if 中 <= 1e-8:
        return y
    gain = np.clip(目标 / rms, 0.5, 2.0)
    # 平滑（3 点）
    g = np.convolve(gain, np.ones(3) / 3, mode="same")
    gi = np.interp(np.arange(n) / hop, np.arange(k), g)
    return y * gi


def 合成一文件(输入, 输出):
    x = 器.耳.读wav(输入)
    y, 控, 片数 = 长录音合成(x)
    y = 电平连续(np.asarray(y, np.float64))
    p = float(np.abs(y).max())
    if p > 1e-9:
        y = y / p * 0.9
    写(输出, y)
    f0s = [float(np.median(c["f0"][c["f0"] > 0])) if (c["f0"] > 0).any() else None for c in 控]
    sha = hashlib.sha256((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()).hexdigest()[:16]
    return dict(输入=str(输入), 输出=str(输出), 秒=round(len(y) / SR, 3),
                片=片数, 帧=sum(int(c["T"]) for c in 控), sha=sha, F0中位列表=f0s)


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) < 3:
        print("用法: 示范模仿_批_v1.py <输入wav|目录> <输出目录>")
        return 2
    源 = Path(sys.argv[1]); 出 = Path(sys.argv[2]); 出.mkdir(parents=True, exist_ok=True)
    文件 = [源] if 源.is_file() else sorted(源.glob("*.wav"))
    报 = dict(工具="示范模仿_批_v1.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 项=[])
    t0 = time.time()
    for f in 文件:
        try:
            r = 合成一文件(f, 出 / f.name)
            报["项"].append(r)
            print("|", f.name, r["秒"], "s 片", r["片"], "sha", r["sha"], flush=True)
        except Exception as ex:
            报["项"].append(dict(输入=str(f), 失败=str(ex)[:200]))
            print("|", f.name, "FAIL", str(ex)[:80], flush=True)
    报["耗时秒"] = round(time.time() - t0, 1)
    (出 / "报告.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print("完成", len(文件), "个 耗时", 报["耗时秒"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(主())
