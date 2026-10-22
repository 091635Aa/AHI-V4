# -*- coding: utf-8 -*-
"""meeting69/实施/检测_S2A_可懂代理.py —— 设计外补充（信息臂·仅仪器，不进门）。

对 S2-A 的 3 条自声做 **离线 ASR 转写 + CER**（Paraformer-seaco-large，`meeting64/实施/量具_v9.载引擎`）。
用途：给所有者一个「这句话听起来是什么」的可懂度代理；**不进门、不宣称可懂**（C10：ASR 只作仪器）。
标注：本件**晚于** S2-A 主判读，属设计外补充，读数单独落盘 `产物/S2A/可懂代理.json`。
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
M69 = HERE.parent
S2A = M69 / "产物" / "S2A"
M64 = M69.parent / "meeting64" / "实施"


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


量 = 载("量具_v9", M64 / "量具_v9.py")
文本 = "今天天气很好我听见了自己的声音我想再说一遍"


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()
    引 = 量.载引擎()
    出 = dict(工具="检测_S2A_可懂代理.py", 引擎=引["名"], 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
              文本=文本, 说明="设计外补充·信息臂·ASR 仅离线仪器（C10）；不进门、不宣称可懂", 项=[])
    for 名 in ("平静", "高兴", "低落", "平静_噪5dB"):
        路 = S2A / "自声" / f"{名}.wav"
        转 = 引["转写"](str(路))
        cer = 量.CER(文本, 转)
        出["项"].append(dict(项=名, 转写=转, CER=cer))
        print(名, "CER=%.4f" % cer, repr(转), flush=True)
    出["耗时秒"] = round(time.time() - t0, 1)
    (S2A / "可懂代理.json").write_text(json.dumps(出, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(主())
