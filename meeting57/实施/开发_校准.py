# -*- coding: utf-8 -*-
"""meeting57/实施/开发_校准.py —— 开发流校准（seed 100–103，**只用于校准，不进判据/留出集**）。

只回答三件事：
  ① A 在易任务上是否可用（非稻草人）；② A 在长程/干扰/矛盾上是否可分辨地退化；
  ③ B/C/S 的量级是否符合设计（B≈C，S 不优于 A）。
不写判据结论；最终阈值在判据顶部冻结。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
结果路径 = os.path.join(HERE, "开发_校准结果.json")


def _载(路径, 名):
    spec = importlib.util.spec_from_file_location(名, 路径)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E = _载(os.path.join(HERE, "环境_长程合成任务.py"), "环境_校准")
M = _载(os.path.join(HERE, "算子_信息保留位置.py"), "载体_校准")


def 跑臂(臂, 任务, 种子, D=M.D默认):
    体 = M.建(臂, 任务, 种子=种子, D=D)
    记 = dict(目标报告=[], 槽位报告={k: [] for k in 任务["关键键列"]}, 查询回答={})
    按步 = E.事件按步(任务)
    查步 = E.查询按步(任务)
    t0 = time.process_time()
    for t in range(int(任务["T"])):
        for e in 按步.get(t, ()):
            体.写(e)
        体.步进(t)
        记["目标报告"].append(体.读(E.目标键))
        for k in 任务["关键键列"]:
            记["槽位报告"][k].append(体.读(k))
        for q in 查步.get(t, ()):
            记["查询回答"][q["编号"]] = 体.读(q["键"])
    秒 = time.process_time() - t0
    果 = E.评分_P0(任务, 记)
    果["CPU秒"] = float(秒)
    return 果


def 主():
    out = {}
    # ① 易任务：A 必须可用（否则载体实现有问题，不是机制结论）
    易 = []
    for s in E.开发集:
        任务 = E.生成(s, "P0易")
        易.append(跑臂("A", 任务, s))
    out["易任务_A"] = dict(
        延迟召回均值=float(np.mean([x["延迟召回率"] for x in 易])),
        目标正确率均值=float(np.mean([x["目标正确率"] for x in 易])),
        关键保真均值=float(np.mean([x["关键槽位保真度"] for x in 易])),
        逐种子=易)

    # ② P0 四臂 + 随机对照，dev 四种子
    臂列 = {}
    for 臂 in ("A", "B", "C", "S", "R"):
        行 = []
        for s in E.开发集:
            任务 = E.生成(s, "P0")
            行.append(跑臂(臂, 任务, s))
        臂列[臂] = dict(
            延迟召回均值=float(np.mean([x["延迟召回率"] for x in 行])),
            延迟8均值=float(np.mean([x["延迟召回率_按延迟"]["8"] for x in 行])),
            延迟48均值=float(np.mean([x["延迟召回率_按延迟"]["48"] for x in 行])),
            延迟96均值=float(np.mean([x["延迟召回率_按延迟"]["96"] for x in 行])),
            目标正确率均值=float(np.mean([x["目标正确率"] for x in 行])),
            目标保持步数均值=float(np.mean([x["目标保持步数"] for x in 行])),
            矛盾污染均值=float(np.mean([x["矛盾污染率"] for x in 行])),
            关键保真均值=float(np.mean([x["关键槽位保真度"] for x in 行])),
            干扰拒答均值=float(np.mean([x["干扰拒答率"] for x in 行])),
            CPU秒均值=float(np.mean([x["CPU秒"] for x in 行])),
            逐种子=行)
    out["P0_dev"] = 臂列

    # ③ 两趟确定性（dev seed 100；CPU 秒不属于确定性口径，剔除）
    def 去时(x):
        y = dict(x)
        y.pop("CPU秒", None)
        return json.dumps(y, sort_keys=True)
    任务 = E.生成(100, "P0")
    a1 = 跑臂("A", 任务, 100)
    a2 = 跑臂("A", E.生成(100, "P0"), 100)
    out["两趟确定性_A"] = dict(逐位同=(去时(a1) == 去时(a2)))

    # ④ 容量扫描（dev 流，仅诊断；说明 A 的退化随同时保留的事实数增加而加重）
    扫 = []
    for N in (32, 64, 128, 192):
        for s in E.开发集:
            任务 = E.生成(s, "P0", 细节数=N, 矛盾细节数=min(16, N))
            a = 跑臂("A", 任务, s)
            扫.append(dict(种子=s, 细节数=N,
                           延迟召回=a["延迟召回率"],
                           延迟8=a["延迟召回率_按延迟"]["8"],
                           延迟48=a["延迟召回率_按延迟"]["48"],
                           延迟96=a["延迟召回率_按延迟"]["96"],
                           目标正确率=a["目标正确率"],
                           关键保真=a["关键槽位保真度"]))
    out["容量扫描_A"] = 扫
    out["P1结构"] = E.生成(0, "P1")["T"]
    return out


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 主()
    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump(果, f, ensure_ascii=False, indent=1, default=str)
    print("[开发_校准] 已写", 结果路径)
    print(json.dumps({k: v for k, v in 果.items() if k != "P0_dev"}, ensure_ascii=False, indent=1, default=str))
    for 臂, v in 果["P0_dev"].items():
        print(臂, {k: round(vv, 3) if isinstance(vv, float) else vv
                  for k, vv in v.items() if k != "逐种子"})
