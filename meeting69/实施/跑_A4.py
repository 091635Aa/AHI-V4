# -*- coding: utf-8 -*-
"""meeting69/实施/跑_A4.py —— A4 臂（n_fft 2048）生成 + F5 对拍 + ASR 可懂代理（信息臂）。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
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


精 = 载("音频线_v1_精修", HERE / "音频线_v1_精修.py")
谐 = 载("谐波重合成_v1", HERE / "谐波重合成_v1.py")
协 = 载("合成_S0刺激", HERE / "合成_S0刺激.py")
量 = 载("量具_v9", M64 / "量具_v9.py")

句子 = [
    "今天天气很好我听见了自己的声音我想再说一遍",
    "早上起来我先看了看窗外的光",
    "我不太确定这句话你听清了没有",
    "如果听不清也没关系我再说一遍",
]


def 写(名, x):
    路 = 出 / 名
    协.写wav(路, x)
    return str(路), hashlib.sha256(路.read_bytes()).hexdigest()


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()
    报 = dict(工具="跑_A4.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 项=[])
    # F5 对拍：A0 是否逐位复现 v0 平静
    r = 精.合成_精修(句子[0], "平静", "正常", 臂="A0")
    路, sha = 写("句0_A0_v0复现.wav", r["波形"])
    v0 = M69 / "产物" / "S2A" / "自声" / "平静.wav"
    报["F5"] = dict(v0_sha=hashlib.sha256(v0.read_bytes()).hexdigest(), A0_sha=sha,
                   同=bool(hashlib.sha256(v0.read_bytes()).hexdigest() == sha))
    # A4：4 句 + 高兴/低落
    项表 = []
    for i, 文 in enumerate(句子):
        r = 精.合成_精修(文, "平静", "正常", 臂="A4")
        路, sha = 写(f"句{i}_A4.wav", r["波形"])
        项表.append((f"句{i}_A4.wav", 路, 文))
    for 情 in ("高兴", "低落"):
        for 移 in ("P1", "P2"):
            r = 精.合成_精修(句子[0], 情, "正常", 臂="A4", 移调=移)
            路, sha = 写(f"{情}_A4_{移}.wav", r["波形"])
            项表.append((f"{情}_A4_{移}.wav", 路, 句子[0]))
    引 = 量.载引擎()
    for 名, 路, 文 in 项表:
        x = 谐.耳.读wav(路)
        t, f0 = 谐.耳.F0轨迹(x)
        v = [float(f) for f in f0 if f > 0]
        不 = 谐.不纯度(x)
        转 = 引["转写"](路)
        报["项"].append(dict(文件=名, 路径=路, CER=量.CER(文, 转), 转写=转,
                            F0中位Hz=round(float(np.median(v)), 1) if v else None,
                            jitter=不["jitter"], shimmer=不["shimmer"], 时长秒=round(len(x) / SR, 3)))
        print("%-20s CER=%.3f F0=%.0f jit=%.4f 时长=%.2f | %s" %
              (名, 报["项"][-1]["CER"], 报["项"][-1]["F0中位Hz"], 不["jitter"], len(x) / SR, 转[:24]), flush=True)
    报["耗时秒"] = round(time.time() - t0, 1)
    (出 / "读数_A4.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print("F5:", 报["F5"]["同"], "耗时", 报["耗时秒"])
    return 0


if __name__ == "__main__":
    sys.exit(主())
