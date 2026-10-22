# -*- coding: utf-8 -*-
"""meeting68/实施/跑_Demo.py —— 最小化音频线 Demo：情绪 × 风格 → 音频 + 控制验证。

验证口径（内生耳，不含外部模型）：
  · 每个样式生成一条 wav（S1 文本）；
  · 耳嵌入应随旋钮方向变化：高兴/激动 F0↑ 能量↑；低落 F0↓ 能量↓；慢 语速↓；快 语速↑；
  · 样式两两不塌缩（耳距离 > 0）；
  · 同参数两趟 sha256 逐位同（确定性）。
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
M68 = HERE.parent
出 = M68 / "产物" / "音频线"
文本 = "今天的风很轻，适合慢慢地说一会儿话。"
情绪s = ["平静", "高兴", "低落", "激动"]
风格s = ["慢", "快"]


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def 主(argv=None):
    线 = 载("音频线", HERE / "音频线_v0.py")
    耳 = 载("内生耳", HERE / "内生耳.py")
    出.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    结果 = []
    for 情 in 情绪s:
        for 风 in 风格s:
            名 = f"{情}_{风}.wav"
            路 = 出 / 名
            r = 线.合成(文本, 情绪=情, 风格=风, 保存=str(路), 采样嵌入=True)
            结果.append(dict(情绪=情, 风格=风, 文件=名, 控制=r["控制"],
                             时长秒=r["时长秒"], 峰值=r["峰值"], sha256=r["sha256"],
                             耳嵌入=r["耳嵌入"]))
            print(f"[{情}/{风}] {r['时长秒']}s sha={r['sha256'][:12]}")
    # 控制方向验证（F0/能量/语速）
    def 取(情, 风, k):
        for r in 结果:
            if r["情绪"] == 情 and r["风格"] == 风:
                return r["耳嵌入"][k]
        return None
    方向 = dict(
        F0_高兴高于低落=取("高兴", "慢", "F0中位") > 取("低落", "慢", "F0中位"),
        能量_高兴高于低落=取("高兴", "慢", "能量均值") > 取("低落", "慢", "能量均值"),
        语速_快高于慢=取("平静", "快", "语速") > 取("平静", "慢", "语速"),
        时长_慢长于快=取("平静", "慢", "时长秒") > 取("平静", "快", "时长秒"),
    )
    # 样式两两距离（不塌缩）
    距离 = []
    for i in range(len(结果)):
        for j in range(i + 1, len(结果)):
            d = 耳.距离(结果[i]["耳嵌入"], 结果[j]["耳嵌入"])["总"]
            距离.append(dict(A=f"{结果[i]['情绪']}/{结果[i]['风格']}",
                             B=f"{结果[j]['情绪']}/{结果[j]['风格']}", 距离=round(d, 5)))
    # 确定性：平静/慢 再生成一次
    r2 = 线.合成(文本, 情绪="平静", 风格="慢", 保存=str(出 / "_确定性复核.wav"), 采样嵌入=False)
    确定性 = (r2["sha256"] == next(r["sha256"] for r in 结果 if r["情绪"] == "平静" and r["风格"] == "慢"))
    报 = dict(工具="跑_Demo.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 文本=文本,
             样式数=len(结果), 结果=结果, 方向验证=方向,
             两两耳距离=距离, 最小耳距离=min((d["距离"] for d in 距离), default=None),
             确定性同sha=确定性, 耗时秒=round(time.time() - t0, 1),
             边界="参数化模仿；耳为自研 DSP；不宣称自然度/可懂度")
    (出 / "Demo报告.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(dict(方向验证=方向, 最小耳距离=报["最小耳距离"], 确定性同sha=确定性,
                          耗时秒=报["耗时秒"]), ensure_ascii=False, indent=1))
    print("已写：", 出 / "Demo报告.json")
    return 0


if __name__ == "__main__":
    sys.exit(主())
