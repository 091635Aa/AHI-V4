# -*- coding: utf-8 -*-
"""meeting69/实施/对比_精修v1.py —— 候选 A0/A1/A2/A3/H 的不纯度 + ASR 可懂代理（信息臂）。

ASR 只作离线仪器（C10）；本件不设门，只给所有者挑选用的对照表。
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M69 = HERE.parent
出 = M69 / "产物" / "精修v1"
M64 = M69.parent / "meeting64" / "实施"
SR = 22050


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


谐 = 载("谐波重合成_v1", HERE / "谐波重合成_v1.py")
耳 = 谐.耳
量 = 载("量具_v9", M64 / "量具_v9.py")

句子 = {
    "句0": "今天天气很好我听见了自己的声音我想再说一遍",
    "高兴": "今天天气很好我听见了自己的声音我想再说一遍",
    "低落": "今天天气很好我听见了自己的声音我想再说一遍",
    "句1": "早上起来我先看了看窗外的光",
    "句2": "我不太确定这句话你听清了没有",
    "句3": "如果听不清也没关系我再说一遍",
}


def 指标(路):
    x = 耳.读wav(str(路))
    t, f0 = 耳.F0轨迹(x)
    v = [float(f) for f in f0 if f > 0]
    F0 = float(np.median(v)) if v else float("nan")
    不 = 谐.不纯度(x)
    # 浊帧能量占比（近似）
    W = int(0.04 * SR)
    能量 = []
    for ti, fi in zip(t, f0):
        c = int(round(float(ti) * SR)); a = c - W // 2
        if a < 0 or a + W > len(x):
            continue
        能量.append(float(np.sqrt(np.mean(np.asarray(x[a:a + W], np.float64) ** 2))))
    return dict(时长秒=round(len(x) / SR, 3), F0中位Hz=round(F0, 1), 浊帧比=round(len(v) / max(len(f0), 1), 3),
                jitter=不["jitter"], shimmer=不["shimmer"],
                能量中位=round(float(np.median(能量)), 4) if 能量 else None)


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()
    # 组名 → 文本
    def 组(名):
        for k in ("高兴", "低落", "句1", "句2", "句3"):
            if 名.startswith(k):
                return k
        return "句0"
    文件 = sorted(出.glob("*.wav"))
    报 = dict(工具="对比_精修v1.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 项=[])
    for 路 in 文件:
        报["项"].append(dict(文件=路.name, **指标(路)))
    print("[指标]", json.dumps(报["项"], ensure_ascii=False)[:400], flush=True)
    引 = 量.载引擎()
    for 项 in 报["项"]:
        真 = 句子[组(项["文件"])]
        转 = 引["转写"](str(出 / 项["文件"]))
        项["ASR_CER"] = 量.CER(真, 转)
        项["转写"] = 转
        print("%-22s CER=%.4f F0=%.0f jit=%.4f shim=%.3f %s" %
              (项["文件"], 项["ASR_CER"], 项["F0中位Hz"], 项["jitter"], 项["shimmer"], 转[:26]), flush=True)
    报["耗时秒"] = round(time.time() - t0, 1)
    (出 / "对比_精修v1.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(主())
