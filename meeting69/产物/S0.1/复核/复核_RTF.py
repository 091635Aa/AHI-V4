# -*- coding: utf-8 -*-
"""附加探针 3：RTF 抖动（同一 60 s 真实音频，S0 与 S0.1 交替各 2 趟）。"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

M69 = Path(r"<workspace>/03_AHI\AHI4\meeting69")
复核 = M69 / "产物" / "S0.1" / "复核"
实 = Path(r"<path>\SpeakerA7代模型训练数据\(1).wav")


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


前0 = 载("f0", M69 / "实施" / "耳蜗前端_S0.py")
前1 = 载("f1", M69 / "实施" / "耳蜗前端_S0_1.py")
行 = []
for k in range(2):
    a = 前0.分析(实)
    行.append(dict(趟=f"S0_{k}", RTF=a["实时率"], sha=a["sha256"][:12]))
    b = 前1.分析(实, 变体="V3")
    行.append(dict(趟=f"S01_{k}", RTF=b["实时率"], sha=b["sha256"][:12]))
出 = dict(逐趟=行,
          S0最小=min(x["RTF"] for x in 行 if x["趟"].startswith("S0_")),
          S01最小=min(x["RTF"] for x in 行 if x["趟"].startswith("S01_")),
          S0与S01_sha同=bool([x["sha"] for x in 行 if x["趟"].startswith("S0_")][0] !=
                          [x["sha"] for x in 行 if x["趟"].startswith("S01_")][0]),
          说明="sha 不同属预期（适应组合规则不同→场不同）；RTF 为墙钟，受机器负载影响")
(复核 / "复核_RTF.json").write_text(json.dumps(出, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(出, ensure_ascii=False, indent=1))
