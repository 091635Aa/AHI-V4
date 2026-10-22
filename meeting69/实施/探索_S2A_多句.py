# -*- coding: utf-8 -*-
"""meeting69/实施/探索_S2A_多句.py —— 设计外·探索性（不进门）：多句「完整的话」+ ASR 仪器转写。

目的：单句 CER 0.14 可能是运气；用 3 条**新句子**（同一音频线/同一 ckpt/同一控制）看是否成立。
标注：**晚于**预注册与主判读的探索性信息臂；ASR 仅离线仪器（C10）；不宣称可懂度、不进任何门。
落盘：产物/S2A/多句_可懂代理.json + 自声/多句_*.wav
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
M69 = HERE.parent
S2A = M69 / "产物" / "S2A"
M68 = M69.parent / "meeting68" / "实施"
M64 = M69.parent / "meeting64" / "实施"

句子 = [
    "早上起来我先看了看窗外的光",
    "我不太确定这句话你听清了没有",
    "如果听不清也没关系我再说一遍",
]


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


音频线 = 载("音频线_v0", M68 / "音频线_v0.py")
量 = 载("量具_v9", M64 / "量具_v9.py")


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()
    出 = dict(工具="探索_S2A_多句.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
              说明="设计外探索（晚于预注册与主判读）；ASR 仅离线仪器；不进门、不宣称可懂",
              声源="meeting68/音频线_v0（平静/正常/4轮/阻尼0.5）", 项=[])
    路表 = []
    for i, 文 in enumerate(句子):
        路 = S2A / "自声" / f"多句_{i}.wav"
        r = 音频线.合成(文, 情绪="平静", 风格="正常", 迭代=4, 阻尼=0.5, 保存=str(路), 采样嵌入=False)
        r.pop("波形", None)
        路表.append((文, 路, r))
    引 = 量.载引擎()
    for i, (文, 路, r) in enumerate(路表):
        转 = 引["转写"](str(路))
        cer = 量.CER(文, 转)
        出["项"].append(dict(序=i, 文本=文, 路径=str(路), 时长秒=r["时长秒"],
                            sha256=r["sha256"], 转写=转, CER=cer))
        print(i, "CER=%.4f" % cer, repr(转), flush=True)
    cers = [x["CER"] for x in 出["项"]]
    出["汇总"] = dict(n=len(cers), 均值=round(sum(cers) / len(cers), 4), 最小=min(cers),
                      最大=max(cers), 耗时秒=round(time.time() - t0, 1))
    (S2A / "多句_可懂代理.json").write_text(json.dumps(出, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(出["汇总"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(主())
