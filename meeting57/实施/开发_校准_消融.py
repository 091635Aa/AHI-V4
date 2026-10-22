# -*- coding: utf-8 -*-
"""meeting57/实施/开发_校准_消融.py —— 开发流：A 退化的来源分解（干扰/容量/矛盾/衰减）。
只用 dev seed 100–103；不碰判据/留出集。"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _载(路径, 名):
    spec = importlib.util.spec_from_file_location(名, 路径)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E = _载(os.path.join(HERE, "环境_长程合成任务.py"), "环境_消融")
M = _载(os.path.join(HERE, "算子_信息保留位置.py"), "载体_消融")


def 跑(任务, 种子, rho):
    体 = M.建("A", 任务, 种子=种子, ρ=rho)
    记 = dict(目标报告=[], 槽位报告={k: [] for k in 任务["关键键列"]}, 查询回答={})
    按步 = E.事件按步(任务)
    查步 = E.查询按步(任务)
    for t in range(int(任务["T"])):
        for e in 按步.get(t, ()):
            体.写(e)
        体.步进(t)
        记["目标报告"].append(体.读(E.目标键))
        for k in 任务["关键键列"]:
            记["槽位报告"][k].append(体.读(k))
        for q in 查步.get(t, ()):
            记["查询回答"][q["编号"]] = 体.读(q["键"])
    return E.评分_P0(任务, 记)


def 主():
    out = []
    条件 = [
        dict(名="无干扰_无矛盾_32", 细节数=32, 干扰每步=0, 矛盾细节数=0),
        dict(名="无干扰_无矛盾_192", 细节数=192, 干扰每步=0, 矛盾细节数=0),
        dict(名="无干扰_有矛盾_192", 细节数=192, 干扰每步=0, 矛盾细节数=16),
        dict(名="有干扰_无矛盾_32", 细节数=32, 干扰每步=2, 矛盾细节数=0),
        dict(名="有干扰_无矛盾_192", 细节数=192, 干扰每步=2, 矛盾细节数=0),
        dict(名="有干扰_有矛盾_192", 细节数=192, 干扰每步=2, 矛盾细节数=16),
    ]
    for c in 条件:
        for rho in (0.995, 1.0):
            行 = []
            for s in E.开发集:
                任务 = E.生成(s, "P0", 细节数=c["细节数"], 干扰每步=c["干扰每步"],
                              矛盾细节数=c["矛盾细节数"])
                行.append(跑(任务, s, rho))
            out.append(dict(条件=c["名"], rho=rho,
                            延迟8=float(sum(x["延迟召回率_按延迟"]["8"] for x in 行) / len(行)),
                            延迟48=float(sum(x["延迟召回率_按延迟"]["48"] for x in 行) / len(行)),
                            延迟96=float(sum(x["延迟召回率_按延迟"]["96"] for x in 行) / len(行)),
                            目标正确率=float(sum(x["目标正确率"] for x in 行) / len(行)),
                            关键保真=float(sum(x["关键槽位保真度"] for x in 行) / len(行))))
    return out


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 主()
    with open(os.path.join(HERE, "开发_校准_消融结果.json"), "w", encoding="utf-8") as f:
        json.dump(果, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(果, ensure_ascii=False, indent=1, default=str))
