# -*- coding: utf-8 -*-
"""AHI4/挂载/跑_挂载演示.py —— 生成三个角色的独白（洛天依 / MOSS / 打标日记）并落盘。

只写 产物/；不改既有件。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "meeting53", "实施"))
import V5_双通道 as V5  # noqa: E402
import 角色包 as 包模块  # noqa: E402
import 挂载_角色 as 挂载  # noqa: E402

np.seterr(all="ignore")
产物 = os.path.join(HERE, "产物")
os.makedirs(产物, exist_ok=True)
训练拍, 演示拍 = 120, 30


def main():
    器 = 挂载.挂载器()
    状态 = 器.状态列(训练拍 + 演示拍, 种子=V5.种子默认)
    行总 = ["【角色挂载演示 · 同一状态序列换三套角色包】",
            "身体：D=64 两区室核（V5）；挂载层只读 o用/内心；不加载外部权重。",
            "说明：正文=角色词；「｜想起：」=该角色包记忆条目（外源，经归属门只读）。", ""]
    汇总 = {}
    for 名, 包 in 包模块.全部包().items():
        r = 器.跑(包, 状态, 最多词=4)
        行 = r["行"][训练拍:]
        块 = ["=" * 24 + " 角色：%s（%s）" % (r["角色"], 名),
              "包来源：%s" % r["来源"],
              "包指纹：%s ｜ 头指纹：%s" % (hashlib16(包), r["头指纹"]), ""]
        for x in 行:
            文 = x["正文"]
            if x["回忆"]:
                文 += " ｜想起：" + str(x["回忆"].get("文本", ""))[:80]
            块.append("  第%02d拍 ｜%s" % (x["拍"], 文))
        块.append("")
        行总 += 块
        汇总[名] = dict(角色=r["角色"], 头指纹=r["头指纹"], 包指纹=hashlib16(包),
                        召回数=int(r["台账"]["召回数"]),
                        样例=[x["正文"] for x in 行[:10]])
    最优 = os.path.join(产物, "挂载_三角色独白.txt")
    with open(最优, "w", encoding="utf-8") as f:
        f.write("\n".join(行总))
    with open(os.path.join(产物, "挂载_汇总.json"), "w", encoding="utf-8") as f:
        json.dump(汇总, f, ensure_ascii=False, indent=1)
    return 行总, 汇总


def hashlib16(包):
    import hashlib
    return hashlib.sha256(json.dumps(包, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    行, 汇 = main()
    print("\n".join(行[:60]))
    print("…… 全文 →", os.path.join(产物, "挂载_三角色独白.txt"))
    print("汇总 →", json.dumps({k: {kk: v[kk] for kk in ("角色", "头指纹", "召回数")} for k, v in 汇.items()},
                              ensure_ascii=False))
