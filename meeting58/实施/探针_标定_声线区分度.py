# -*- coding: utf-8 -*-
"""meeting58/实施/探针_标定_声线区分度.py —— 标定（不是最终判据）：声线区分度与音量有效性。

用途：在冻结判据阈值前，先量出本实现的实际区分度（共振峰缩放/谱倾斜/气声/紧喉是否真的
把声纹代理拉开），并检查 `音量倍` 是否线性可用。最终判据见 `判据_接口与解耦.py`。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from 接口契约_角色声线 import 状态帧  # noqa: E402
import 声线库  # noqa: E402
import 协议_音色渲染 as R  # noqa: E402

FS = 16000
产物 = os.path.join(HERE, "产物", "标定")
os.makedirs(产物, exist_ok=True)
文本 = "你好，我是你的角色。今天想听我说话吗？"
文本_元音 = "啊啊啊啊啊"
带 = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 3000),
      (3000, 4000), (4000, 8000)]


def 包络7(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    频 = np.fft.rfftfreq(len(x), 1.0 / FS)
    e = []
    for lo, hi in 带:
        m = (频 >= lo) & (频 < hi)
        e.append(np.log(np.mean(spec[m] ** 2) + 1e-12))
    e = np.asarray(e)
    return e - e.mean()


def F0中位(x: np.ndarray) -> float:
    f0 = []
    n = max(1, (len(x) - int(0.04 * FS)) // int(0.01 * FS))
    for i in range(n):
        seg = x[i * int(0.01 * FS): i * int(0.01 * FS) + int(0.04 * FS)]
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
        lo, hi = int(FS / 400), int(FS / 70)
        if hi >= len(ac):
            continue
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] / (ac[0] + 1e-12) > 0.30:
            f0.append(FS / lag)
    return float(np.median(f0)) if f0 else 0.0


def 倒谱共振峰(x: np.ndarray) -> list:
    """倒谱 lifter(30) 包络峰 → F1/F2/F3 近似。"""
    seg = x[int(0.2 * FS):int(0.2 * FS) + int(0.08 * FS)]
    if len(seg) < 256:
        return []
    w = seg * np.hanning(len(seg))
    logmag = np.log(np.abs(np.fft.rfft(w, 2048)) + 1e-9)
    cep = np.fft.irfft(logmag)
    cep[30:-30] = 0.0
    env = np.fft.rfft(cep).real
    频 = np.fft.rfftfreq(2048, 1.0 / FS)
    峰 = []
    for i in range(2, len(env) - 2):
        if env[i] > env[i - 1] and env[i] > env[i + 1] and 200 < 频[i] < 4000:
            峰.append(float(频[i]))
    return 峰[:3]


def 余弦(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def 跑() -> dict:
    器 = R.声带渲染器(目录=产物)
    行 = {}
    for 名 in "洛天依 初音未来 MOSS EmoCompanion 中性".split():
        a = 声线库.取(名)
        o = 器.渲染(文本, 声线=a, 种子=300, 写wav=False)
        x = o["波形"]
        行[名] = dict(F0=round(F0中位(x), 2), 包络=包络7(x).round(3).tolist(),
                      RMS=round(float(np.sqrt(np.mean(x ** 2))), 5),
                      时长=o["时长秒"], 通过=o["通过"])
    # 元音共振峰
    for 名 in "洛天依 初音未来 MOSS EmoCompanion".split():
        a = 声线库.取(名)
        o = 器.渲染(文本_元音, 声线=a, 种子=300, 写wav=False)
        行[名]["共振峰"] = [round(v, 1) for v in 倒谱共振峰(o["波形"])]
        行[名]["元音F0"] = round(F0中位(o["波形"]), 2)
    # 两两
    对 = {}
    名列 = list(行)
    for i in range(len(名列)):
        for j in range(i + 1, len(名列)):
            a, b = 名列[i], 名列[j]
            对["%s|%s" % (a, b)] = dict(
                包络余弦=round(余弦(np.array(行[a]["包络"]), np.array(行[b]["包络"])), 4),
                F0半音=round(abs(12 * np.log2(max(行[a]["F0"], 1) / max(行[b]["F0"], 1))), 2))
    # 音量倍有效性
    a = 声线库.取("EmoCompanion")
    lo = 器.渲染(文本, 声线=a, 种子=300, 音量倍=0.5, 写wav=False)
    hi = 器.渲染(文本, 声线=a, 种子=300, 音量倍=1.0, 写wav=False)
    rms = lambda o: float(np.sqrt(np.mean(np.asarray(o["波形"]) ** 2)))
    音量 = dict(低=round(rms(lo), 5), 基准=round(rms(hi), 5),
                比=round(rms(lo) / max(rms(hi), 1e-12), 4))
    # 状态→波形差异（固定文本/声线）
    st1 = 状态帧.建(拍=1, 观测=1, o用=np.linspace(0.1, 0.9, 64),
                    内心=np.linspace(0.9, 0.1, 64), 自我标量=0.3)
    st2 = 状态帧.建(拍=2, 观测=2, o用=np.linspace(0.9, 0.1, 64),
                    内心=np.linspace(0.1, 0.9, 64), 自我标量=0.7)
    o1 = 器.渲染(文本, 声线=a, 状态=st1, 种子=300, 写wav=False)
    o2 = 器.渲染(文本, 声线=a, 状态=st2, 种子=300, 写wav=False)
    x1, x2 = np.asarray(o1["波形"]), np.asarray(o2["波形"])
    状态差 = dict(长度=[len(x1), len(x2)],
                  同长=bool(len(x1) == len(x2)),
                  波形相关=round(float(np.corrcoef(x1[:min(len(x1), len(x2))],
                                                    x2[:min(len(x1), len(x2))])[0, 1]), 5),
                  F0差半音=round(abs(12 * np.log2(max(F0中位(x1), 1) / max(F0中位(x2), 1))), 3),
                  逐位同=bool(len(x1) == len(x2) and np.array_equal(x1, x2)))
    # 确定性
    d1 = 器.渲染(文本, 声线=a, 种子=300, 写wav=False)
    d2 = 器.渲染(文本, 声线=a, 种子=300, 写wav=False)
    return dict(档案=行, 两两=对, 音量倍=音量, 状态差=状态差,
                确定性=bool(np.array_equal(np.asarray(d1["波形"]), np.asarray(d2["波形"]))))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    结 = 跑()
    p = os.path.join(产物, "标定结果.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(结, f, ensure_ascii=False, indent=1)
    print("档案：")
    for k, v in 结["档案"].items():
        print("  %-5s F0=%6.1f 共振峰=%s RMS=%.4f 时长=%.2f 通过=%s"
              % (k, v["F0"], v.get("共振峰"), v["RMS"], v["时长"], v["通过"]))
    print("两两：")
    for k, v in 结["两两"].items():
        print("  %-18s 包络余弦=%.4f F0半音=%.2f" % (k, v["包络余弦"], v["F0半音"]))
    print("音量倍：", 结["音量倍"])
    print("状态差：", 结["状态差"])
    print("确定性：", 结["确定性"])
    print("落盘：", p)
