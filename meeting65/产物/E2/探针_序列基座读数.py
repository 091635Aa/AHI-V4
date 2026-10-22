# -*- coding: utf-8 -*-
"""meeting65/产物/E2/探针_序列基座读数.py —— E2 席只读探针：量「AHI4 现有序列基座」的长度与冗余。

为什么需要：LST 的机制前提是「语音 token 序列远长于文本序列」，靠 patch 聚合把 AR 单位变短。
要判断这条前提在 AHI4 上是否成立，必须先量 AHI4 的实际序列长度：
  · 帧计划（韵律 θ 侧）= 生成侧单位；帧率 100Hz；
  · 耳切段（波形侧）= 感知侧单位；
  · 文本音节 = 文本单位。
本探针只做三件事：
  1) 建帧（**不合成**，最轻）→ 帧数 / 音节数 / 静音帧；
  2) 对三句真跑合成（3 句 ≈ 0.5s，实时率 0.045）→ 耳切段 → 段数；
  3) 帧序列的「精确重复」游程编码（RLE）→ 冗余度（静音/长音是否可折叠）。
零外挂、零写产物（JSON 由 __main__ 显式写本目录）。C14：不改任何既有件。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M65 = os.path.dirname(os.path.dirname(HERE))
AHI4 = os.path.dirname(M65)
M61 = os.path.join(AHI4, "meeting61")
M62 = os.path.join(AHI4, "meeting62")
M63 = os.path.join(AHI4, "meeting63")
M58 = os.path.join(AHI4, "meeting58")
for _p in (os.path.join(M63, "实施"), os.path.join(M62, "实施"),
           os.path.join(M61, "实施"), os.path.join(M58, "实施")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import 音韵_v5 as Y            # noqa: E402
import 韵律_v8 as H8           # noqa: E402
import 耳_内生 as 耳           # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402

句s = ("你好，我是EmoCompanion。",
       "今天的风很轻，适合慢慢地说一会儿话。",
       "语言不只是声音，它把一个人的温度，送到另一个人的心里。")
帧键 = ("f0", "f1", "f2", "f3", "f4", "f5", "幅", "气声", "声源",
        "送气", "紧喉", "粗糙", "噪形")


def 帧元组(帧: dict, i: int) -> tuple:
    """一帧的**逐字段**值（噪带单独处理：None 记 -1.0/-1.0）。"""
    nb = 帧["噪带"][i]
    if isinstance(nb, (tuple, list)):
        nbv = (float(nb[0]), float(nb[1]))
    elif nb is None:
        nbv = (-1.0, -1.0)
    else:
        nbv = (float(nb), -1.0)
    return tuple(float(帧[k][i]) for k in 帧键) + nbv


def 游程(列: list) -> list:
    """精确相等的游程编码：[(值, 连续个数)]。"""
    out = []
    for v in 列:
        if out and out[-1][0] == v:
            out[-1] = (v, out[-1][1] + 1)
        else:
            out.append((v, 1))
    return out


def 量一句(句: str, 档, 种子: int = 300) -> dict:
    h = H8.人类韵律_v8(档, 种子)
    帧 = h.建帧(句)
    N = len(帧["f0"])
    音节数 = len(Y.音节列表(句))
    声源 = [int(x) for x in 帧["声源"]]
    静帧 = sum(1 for t in 声源 if t == -1)
    浊帧 = sum(1 for t in 声源 if t == 1)
    清帧 = sum(1 for t in 声源 if t == 0)
    列 = [帧元组(帧, i) for i in range(N)]
    rle = 游程(列)
    # 只按「发声类别+噪形」的粗游程（静音/清/浊块）——这是最保守的「可折叠」下界
    粗列 = [(t, int(帧["噪形"][i])) for i, t in enumerate(声源)]
    粗rle = 游程(粗列)
    # 波形侧：真跑合成（轻）→ 耳切段
    波, 元, _ = h.说(句)
    耳r = 耳.听(波, 含段=True)
    段s = 耳r["段"]
    return dict(
        句=句, 时长秒=round(len(波) / 44100.0, 3), 音节数=音节数,
        帧数=N, 帧率=100.0,
        帧每音节=round(N / max(音节数, 1), 2),
        静帧=静帧, 清帧=清帧, 浊帧=浊帧,
        静帧占比=round(静帧 / max(N, 1), 4),
        帧序列精确RLE段数=len(rle),
        帧RLE压缩比=round(len(rle) / max(N, 1), 4),
        粗游程段数=len(粗rle),
        粗游程压缩比=round(len(粗rle) / max(N, 1), 4),
        耳段数=len(段s),
        耳段表=段s,
        耳段每句压缩比=round(len(段s) / max(N, 1), 4),
        耳清段数=耳r["清段数"], 耳浊段数=耳r["浊段数"], 耳静段数=耳r["静段数"],
        耳清总占比=耳r["清总占比"], 耳静总占比=耳r["静总占比"],
    )


def main() -> dict:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    np.seterr(all="ignore")
    with open(os.path.join(M61, "产物", "声线档案", "EmoCompanion.json"), encoding="utf-8") as f:
        档 = 声线档案.从dict(json.load(f))
    行 = [量一句(s, 档) for s in 句s]
    # 确定性：同一句建帧两次，逐位同
    h1 = H8.人类韵律_v8(档, 300); f1 = h1.建帧(句s[0])
    h2 = H8.人类韵律_v8(档, 300); f2 = h2.建帧(句s[0])
    同 = bool(f1["f0"] == f2["f0"] and f1["f1"] == f2["f1"] and f1["声源"] == f2["声源"])
    汇 = dict(
        工具="探针_序列基座读数.py", 版本="meeting65-E2",
        口径="帧计划=韵律 θ 侧单位（100Hz）；耳段=波形侧感知单位；音节=文本单位",
        句=行,
        汇=dict(
            总帧数=sum(r["帧数"] for r in 行),
            总音节数=sum(r["音节数"] for r in 行),
            总耳段数=sum(r["耳段数"] for r in 行),
            帧每音节_均值=round(sum(r["帧每音节"] for r in 行) / len(行), 2),
            帧每耳段_均值=round(sum(r["帧数"] / max(r["耳段数"], 1) for r in 行) / len(行), 2),
            帧精确RLE压缩比_均值=round(sum(r["帧RLE压缩比"] for r in 行) / len(行), 4),
            粗游程压缩比_均值=round(sum(r["粗游程压缩比"] for r in 行) / len(行), 4),
        ),
        确定性=dict(建帧两趟逐位同=同),
    )
    out = os.path.join(HERE, "读数_序列基座.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(汇, f, ensure_ascii=False, indent=1)
    print(json.dumps(汇, ensure_ascii=False, indent=1))
    print("JSON:", out)
    return 汇


if __name__ == "__main__":
    main()
