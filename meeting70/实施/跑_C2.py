# -*- coding: utf-8 -*-
"""meeting70/实施/跑_C2.py —— C2 变体：逐音节过器官 + 波形交叉淡化（信息臂）。"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M70 = HERE.parent
M64 = M70.parent / "meeting64" / "实施"
出 = M70 / "产物"


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


C = 载("控制模仿_C", HERE / "控制模仿_C.py")
量 = 载("量具_v9", M64 / "量具_v9.py")
协 = 载("合成_S0刺激", M70.parent / "meeting69" / "实施" / "合成_S0刺激.py")
句表 = {
    "C2主句": "今天天气很好我听见了自己的声音我想再说一遍",
    "C2新句1": "今天的风很轻我慢慢说给你听",
    "C2新句2": "窗外的雨停了我该休息一会",
    "C2新句3": "你先别走我还想再说两句",
}


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()
    库, 样本, 文 = C.建库_控制()
    字表 = C.建字表(库, 文)
    参数 = dict(OQ=0.62, OQ1=0.42, 噪声尺度=0.40, 去加重=0.97)
    p = 出 / "G0v3R1_扫参.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        参数 = sorted(d["臂"], key=lambda r: r["均值CER"])[0]["参数"]
    引 = 量.载引擎()
    报 = dict(工具="跑_C2.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 参数=参数, 项=[])
    for 名, 文本 in 句表.items():
        y, 状态 = C.合成句_控制(文本, 参数, 库, 样本, 文, 字表, 逐音节=True)
        if y is None:
            报["项"].append(dict(样本=名, 失败="无段")); continue
        pp = 出 / f"{名}.wav"
        协.写wav(pp, y)
        r = 引["转写"](str(pp))
        报["项"].append(dict(样本=名, 文本=文本, 路径=str(pp), 秒=round(len(y) / 22050, 3),
                           缺=sum(1 for _, s in 状态 if s == "缺"), 转写=r, CER=量.CER(文本, r)))
        print("|", 名, "CER=%.3f" % 报["项"][-1]["CER"], "|", r[:26], flush=True)
    报["均值CER"] = round(float(np.mean([r["CER"] for r in 报["项"] if "CER" in r])), 4)
    报["耗时秒"] = round(time.time() - t0, 1)
    (出 / "C2_读数.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print("均值CER", 报["均值CER"])
    return 0


if __name__ == "__main__":
    sys.exit(主())
