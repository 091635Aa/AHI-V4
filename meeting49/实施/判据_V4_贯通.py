# -*- coding: utf-8 -*-
"""AHI4/meeting49/实施/判据_V4_贯通.py —— ★V4「多核心贯通」判据 + 整合替换评估。

真跑：六内核宿主 × 两区室核 贯通 N 拍 → 落 `判据_V4_贯通结果.json`。
含：①贯通不炸 ②六内核全挂 ③自我态有界有限 ④provenance 自源 ⑤自模型在学 ⑥两趟确定性
    ⑦C1 禁词 ⑧改前备份在场 ⑨★整合替换评估（备选整合必须不优于主）。
阈值写死，不看结果改。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
本件 = os.path.abspath(__file__)
_核py = os.path.join(HERE, "算子_两区室核.py")
_贯通py = os.path.join(HERE, "贯通_多核心.py")
_主程序py = os.path.normpath(os.path.join(HERE, "..", "..", "V4主程序.py"))
结果路径 = os.path.join(HERE, "判据_V4_贯通结果.json")


def _载(路径, 名):
    spec = importlib.util.spec_from_file_location(名, 路径)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


核m = _载(_核py, "算子_两区室核_贯通判据")
贯 = _载(_贯通py, "贯通_多核心_判据")

# ── 冻结阈值 ──
拍数 = 120
自我态界上限 = 16.0
ε上限 = 2.0
允许导入 = ("__future__", "numpy", "hashlib", "json", "os", "sys", "importlib", "math", "types", "runpy")
备份目录 = os.path.normpath(os.path.join(HERE, "..", "..", "备份_贯通前"))


def 扫导入(路径):
    名 = set()
    with open(路径, encoding="utf-8") as f:
        文 = f.read()
    for 行 in 文.splitlines():
        s = 行.strip()
        if s.startswith("import "):
            名.add(s[len("import "):].split()[0].split(",")[0].split(".")[0])
        elif s.startswith("from "):
            名.add(s[len("from "):].split()[0].split(".")[0])
    return sorted(名)


def 跑贯通(seed=310, N=拍数):
    双 = 贯.造双区室(seed=seed)
    return 贯.跑(双, N)


def 主流程():
    条 = []
    臂 = {}

    def 记(名, 通过, 实测, 阈值, 说明=""):
        条.append(bool(通过))
        臂[名] = dict(实测=实测, 阈值=阈值, 通过=bool(通过), 说明=说明)

    # ── 1 贯通不炸 ──
    果 = 跑贯通()
    宿 = 果["内核态"]
    记("贯通_不炸", bool(果["异常"] is None and 果["已跑"] == 拍数),
       dict(已跑=果["已跑"], 异常=果["异常"]), dict(拍数=拍数, 异常须为None=True),
       说明="六内核宿主 × 两区室核 真贯通 N 拍，无异常")

    # ── 2 六内核全挂 ──
    层 = [k for k in 宿.keys() if k != "挂载"]
    全挂 = bool(宿.get("挂载") and all(宿.get(k) is not None for k in ("自模型", "思核", "真学习", "自我回路", "顿悟涌现", "画面")))
    记("贯通_六内核全挂", 全挂, dict(内核层=层, 各层非None={k: (宿.get(k) is not None) for k in 层}),
       dict(六内核全挂=True), 说明="自模型/思核/真学习/自我回路/顿悟涌现/画面 六层均挂载")

    # ── 3 自我态有界有限 ──
    限 = bool(果["自我态有限"] and 果["自我态界"] <= 自我态界上限)
    记("贯通_自我态有界有限", 限, dict(自我态有限=果["自我态有限"], 自我态界=果["自我态界"]),
       dict(界上限=自我态界上限), 说明="递归自更新后的自我态有限且 ≤ 界幅")

    # ── 4 provenance 自源 ──
    回 = 宿.get("自我回路") or {}
    自源过 = bool(int(回.get("自源更新次数", 0)) > 0 and int(回.get("外源次数", 0)) == 0)
    记("贯通_provenance自源", 自源过, 回, dict(自源更新_gt0=True, 外源次数_eq0=True),
       说明="它自己的经历塑造「我」（自源）；外源默认不塑造（见 注入外源规则）")

    # ── 5 自模型在学 ──
    自 = 宿.get("自模型") or {}
    ε = 自.get("ε均值")
    学 = bool(int(自.get("迹条数", 0)) > 0 and ε is not None and float(ε) <= ε上限)
    记("贯通_自模型在学", 学, 自, dict(迹条数_gt0=True, ε均值_le2=True),
       说明="ĝ 自监督在链上学（ε_self 有限、迹在增长）")

    # ── 6 两趟确定性 ──
    果2 = 跑贯通()
    决 = bool(果["已跑"] == 果2["已跑"] and 果["自我态界"] == 果2["自我态界"]
             and (宿.get("自模型") or {}).get("ε均值") == (果2["内核态"].get("自模型") or {}).get("ε均值")
             and 宿.get("自我回路", {}).get("自源更新次数") == 果2["内核态"].get("自我回路", {}).get("自源更新次数"))
    记("贯通_两趟确定性", 决,
       dict(已跑=[果["已跑"], 果2["已跑"]], 自我态界=[果["自我态界"], 果2["自我态界"]],
            ε均值=[(宿.get("自模型") or {}).get("ε均值"), (果2["内核态"].get("自模型") or {}).get("ε均值")]),
       dict(逐位同=True), 说明="同种子两遍贯通，关键读数一致")

    # ── 7 C1 禁词自扫 ──
    禁词 = list(核m.禁词列())
    命中 = {}
    for p in (_贯通py, _主程序py):
        with open(p, encoding="utf-8") as f:
            低 = f.read().lower()
        命中[os.path.basename(p)] = [w for w in 禁词 if w.lower() in 低]
    总 = sum(len(v) for v in 命中.values())
    记("贯通_C1禁词自扫", 总 == 0, dict(命中=命中, 命中总数=总), dict(命中总数=0),
       说明="复用 算子_能力核.禁词列")

    # ── 8 改前备份在场（C14/宪法第八条）──
    名单 = ["V3主程序.py", os.path.join("meeting38", "实施", "内核宿主.py"),
            os.path.join("meeting49", "实施", "算子_两区室核.py")]
    备 = {n: os.path.exists(os.path.join(备份目录, n)) for n in 名单}
    记("贯通_改前备份在场", bool(all(备.values())), dict(备份目录=备份目录, 各件=备),
       dict(全部存在=True), 说明="贯通前先备份待改既有件（C14）")

    # ── 9 ★整合替换评估：备选整合必须不优于主（软限差分）──
    序训 = 核m.生成序列(核m.种子训默认, 核m.训练长, 核m.边界训练)
    序测 = 核m.生成序列(核m.种子测默认, 核m.测试长, 核m.边界训练)
    各 = {}
    for 模式 in ("两区室", "符合门", "差分加符"):
        _, _, 测 = 核m.跑臂(模式, 核m.D默认, 核m.种子模型默认, 序训, 序测, 慢率=0.20, 学模式="credit")
        各[模式] = float(测["Acc_A"])
    主A = 各["两区室"]
    替 = bool(主A >= 各["符合门"] and 主A >= 各["差分加符"])
    记("整合替换评估", 替, 各, dict(主臂最大=True),
       说明="预检：纯双线性符合门 / 差分加符 均不优于 软限差分 ⇒ **不替换**（有更好才换，实测无更好）")

    # ── 10 导入白名单 ──
    导入表 = {os.path.basename(p): 扫导入(p) for p in (_贯通py, _主程序py, 本件)}
    越界 = {k: [m for m in v if m not in 允许导入] for k, v in 导入表.items()}
    越界非空 = {k: v for k, v in 越界.items() if v}
    记("贯通_导入白名单", not 越界非空, dict(导入表=导入表, 越界=越界非空, 白名单=list(允许导入)),
       dict(越界=[]), 说明="只允许标准库 + numpy")

    未过 = [名 for 名, v in 臂.items() if not v["通过"]]
    果all = dict(通过=bool(all(条)), 判定条数=len(条), 未过条数=int(sum(1 for v in 条 if not v)),
                未过臂=未过, 阈值=dict(拍数=拍数, 自我态界上限=自我态界上限, ε上限=ε上限),
                臂=臂,
                实测摘要=dict(已跑=果["已跑"], 异常=果["异常"], 六内核全挂=全挂,
                            自我态界=round(果["自我态界"], 6),
                            自模型迹条数=(宿.get("自模型") or {}).get("迹条数"),
                            自模型ε均值=(宿.get("自模型") or {}).get("ε均值"),
                            自我回路自源更新=(宿.get("自我回路") or {}).get("自源更新次数"),
                            自我回路外源次数=(宿.get("自我回路") or {}).get("外源次数"),
                            顿悟条数=(宿.get("顿悟涌现") or {}).get("顿悟条数"),
                            整合评估=各),
                说明="六内核宿主（真 synthesis 链）× 两区室核 真贯通；阈值先冻结、实测落盘、空着=未测。")
    return 果all


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 主流程()
    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump(果, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(果["实测摘要"], ensure_ascii=False, indent=1, default=str))
    print("[判据_V4_贯通] 判定 %d 条，未过 %d 条 ⇒ %s"
          % (果["判定条数"], 果["未过条数"], "PASS" if 果["通过"] else "FAIL"))
    print("未过臂：", json.dumps(果["未过臂"], ensure_ascii=False))
    raise SystemExit(0 if 果["通过"] else 1)
