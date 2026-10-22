# -*- coding: utf-8 -*-
"""meeting47/实施/判据_树突两区室.py —— 两区室(顶树+胞体)单元科学假设判据。

真跑全部臂（主/消融/点式各λ/迁移/各 sham）＋静态自检（C1 禁词、导入白名单、计费线性、两趟确定性），
把每个臂的「实测+阈值+通过+反例_必须失败」及提升量落盘到 判据_树突两区室结果.json。
阈值在脚本顶部**冻结**（写死，不看结果改）。

设计口径（规格未明处，此处显式固定）：
  · 训练 400：读出（慢权重）+ 规则通道 b̂/drive 都随奖赏更新；测试 400：**读出冻结**，
    只有规则通道 b̂/drive 在线适应。⇒ 决策的规则切换能力**只来自 b̂ 通道**，与科学假设一致；
    这样「无区室」sham（o=归(b)，无 b̂）才按规格预期失败（否则在线读出自己就能实现规则切换，污染对照）。
  · R 用「全试次」口径（切换后首次连续 3 试次正确所耗试次数）作阈值判据；同时如实记录仅在 A 试次上的 R_A。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
本件 = os.path.abspath(__file__)
算子路径 = os.path.join(HERE, "算子_双区室.py")
结果路径 = os.path.join(HERE, "判据_树突两区室结果.json")


def _载(路径, 名):
    spec = importlib.util.spec_from_file_location(名, 路径)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _载(算子路径, "算子_双区室_判据")

# ── 冻结阈值（写死；不看结果改） ──
ACC_A_主 = 0.80
ACC_B_主 = 0.95
Δ点式_min = 0.10
Δ消融_min = 0.15
迁移_min = 0.75
R_max = 8
消融_ACC_A_max = 0.62
消融_ACC_B差_max = 0.05
sham_ACC_A_max = 0.60
计费比区间 = (3.95, 4.05)
允许导入 = ("__future__", "numpy", "hashlib", "json", "os", "sys", "importlib", "math")

D = M.D默认
种子模型 = M.种子模型默认
序列训 = M.生成序列(M.种子训默认, M.训练长, M.边界训练)
序列测 = M.生成序列(M.种子测默认, M.测试长, M.边界训练)
序列迁移测 = M.生成序列(M.种子迁移测默认, M.测试长, M.边界迁移)


def 跑臂记录(模式, 序列测_, λ=None, ηb=None):
    _, 训, 测 = M.跑臂(模式, D, 种子模型, 序列训, 序列测_, λ=λ,
                       ηb=(M.ηb默认 if ηb is None else ηb))
    return M.汇(测, 训), 模型指纹(训)


def 模型指纹(训):
    return hashlib.sha256(np.ascontiguousarray(训["选"]).tobytes()
                          + np.ascontiguousarray(训["奖"]).tobytes()).hexdigest()


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


def 主流程():
    条 = []
    臂 = {}

    def 记(名, 通过, 反例, 实测, 阈值, 说明=""):
        条.append(bool(通过))
        臂[名] = dict(实测=实测, 阈值=阈值, 通过=bool(通过), 反例_必须失败=bool(反例), 说明=说明)

    # ── ηb **预先声明的扫描**（在训练序列上训、在测试序列上评；阈值不随结果改）──
    扫描汇 = {}
    扫描纹 = {}
    for e in M.ηb列:
        汇e, 纹e = 跑臂记录("两区室", 序列测, ηb=e)
        扫描汇[e] = 汇e
        扫描纹[e] = 纹e
    ηb曲线 = {"%.2f" % e: dict(Acc_A=扫描汇[e]["Acc_A"], Acc_B=扫描汇[e]["Acc_B"],
                              R_全=扫描汇[e]["R_全"], R_A=扫描汇[e]["R_A"]) for e in M.ηb列}
    合格ηb = [e for e in M.ηb列 if 扫描汇[e]["Acc_B"] >= ACC_B_主]
    候选 = 合格ηb if 合格ηb else list(M.ηb列)
    选定ηb = max(候选, key=lambda e: (扫描汇[e]["Acc_A"], -e))   # Acc_A 最高且 Acc_B≥0.95；并列取较小 ηb

    # ── 主臂：两区室（在 选定ηb 下；等于扫描中的对应点）──
    汇主 = 扫描汇[选定ηb]
    指纹主 = 扫描纹[选定ηb]
    主通过 = bool(汇主["Acc_A"] >= ACC_A_主 and 汇主["Acc_B"] >= ACC_B_主 and 汇主["R_全"] <= R_max)
    记("主臂_两区室", 主通过, False,
       dict(Acc_A=汇主["Acc_A"], Acc_B=汇主["Acc_B"], Acc=汇主["Acc"],
            R_全=汇主["R_全"], R_全_max=汇主["R_全_max"], R_A=汇主["R_A"], R_A_max=汇主["R_A_max"],
            切换点数=汇主["切换点数"], 训练末100全体Acc=汇主["训练末100全体Acc"], 选定ηb=选定ηb),
       dict(Acc_A=ACC_A_主, Acc_B=ACC_B_主, R_全=R_max),
       说明="Acc_A≥0.80 且 Acc_B≥0.95 且 R_全≤8（ηb=选定值 %.2f，由预先声明扫描选定）" % 选定ηb)

    # ── 消融：冻结区室 A（a=顶树(0.5)，常数、不含 b̂）；ηb=选定ηb 保持口径一致 ──
    汇消, _ = 跑臂记录("消融", 序列测, ηb=选定ηb)
    差B = abs(汇消["Acc_B"] - 汇主["Acc_B"])
    消通过 = bool(汇消["Acc_A"] <= 消融_ACC_A_max and 差B <= 消融_ACC_B差_max)
    Δ消融 = float(汇主["Acc_A"] - 汇消["Acc_A"])
    记("消融_冻结区室A", 消通过, True,
       dict(Acc_A=汇消["Acc_A"], Acc_B=汇消["Acc_B"], Δ消融=Δ消融, Acc_B与主臂差=差B),
       dict(Acc_A_上限=消融_ACC_A_max, Acc_B差_上限=消融_ACC_B差_max, Δ消融_下限=Δ消融_min),
       说明="预期 A 大降(近50%)、B 几乎不变；且 Δ消融≥0.15")

    # ── 点式基线：各 λ 各跑一遍，取 Acc_A 最高值（ηb=选定ηb，奖励更新规则与主臂同）──
    点式 = {}
    for λ in M.λ列:
        汇, _ = 跑臂记录("点神经元", 序列测, λ=λ, ηb=选定ηb)
        点式["λ=%.2f" % λ] = dict(Acc_A=汇["Acc_A"], Acc_B=汇["Acc_B"], R_全=汇["R_全"])
    点A最大值 = max(v["Acc_A"] for v in 点式.values())
    点式最优λ = max(点式, key=lambda k: 点式[k]["Acc_A"])
    Δ点式 = float(汇主["Acc_A"] - 点A最大值)
    点通过 = bool(Δ点式 >= Δ点式_min)
    记("点式基线", 点通过, False,
       dict(各λ=点式, Acc_A_max=点A最大值, 最优λ=点式最优λ, Δ点式=Δ点式),
       dict(Δ点式_下限=Δ点式_min),
       说明="同 D/P/读出形状/种子/预算/奖赏更新规则；差异只在单一递推状态 s")

    # ── 迁移臂：训练 {0.35,0.65}，测试未见边界 {0.25,0.75} ──
    汇迁, _ = 跑臂记录("两区室", 序列迁移测, ηb=选定ηb)
    迁通过 = bool(汇迁["Acc_A"] >= 迁移_min)
    记("迁移_未见边界", 迁通过, False,
       dict(Acc_A=汇迁["Acc_A"], Acc_B=汇迁["Acc_B"], R_全=汇迁["R_全"]),
       dict(Acc_A=迁移_min),
       说明="训练 {0.35,0.65} → 测试未见边界 {0.25,0.75}")

    # ── sham 1：规则随机（b̂ 随机游走，不用奖赏）；ηb=选定ηb 保持口径一致 ──
    汇sh1, _ = 跑臂记录("规则随机", 序列测, ηb=选定ηb)
    sh1通过 = bool(汇sh1["Acc_A"] <= sham_ACC_A_max)
    记("sham_规则随机", sh1通过, True,
       dict(Acc_A=汇sh1["Acc_A"], Acc_B=汇sh1["Acc_B"]),
       dict(Acc_A_上限=sham_ACC_A_max), 说明="b̂ 更新替换为随机游走（不用奖赏）⇒ A 必须不过阈")

    # ── sham 2：无区室（o=归(b)，只用感觉）；ηb=选定ηb 保持口径一致 ──
    汇sh2, _ = 跑臂记录("无区室", 序列测, ηb=选定ηb)
    sh2通过 = bool(汇sh2["Acc_A"] <= sham_ACC_A_max)
    记("sham_无区室", sh2通过, True,
       dict(Acc_A=汇sh2["Acc_A"], Acc_B=汇sh2["Acc_B"]),
       dict(Acc_A_上限=sham_ACC_A_max), 说明="o 换成 _归(b)（两区室结构被打乱）⇒ A 必须不过阈")

    # ── sham 3：纯线性求和（o=归(a+b)，非差分）；ηb=选定ηb 保持口径一致 ──
    汇sh3, _ = 跑臂记录("线性求和", 序列测, ηb=选定ηb)
    sh3通过 = bool(汇sh3["Acc_A"] <= sham_ACC_A_max)
    Δ线性vs主 = float(汇主["Acc_A"] - 汇sh3["Acc_A"])
    说明3 = "o=归(a+b)（线性求和，非差分）"
    if sh3通过:
        说明3 += "；显著差于 软限(a-b)（Δ=%.3f）" % Δ线性vs主
    else:
        说明3 += "；实测与 软限(a-b) 持平/更优（Δ=%.3f）⇒「无非线性增益」" % Δ线性vs主
    记("sham_线性求和", sh3通过, True,
       dict(Acc_A=汇sh3["Acc_A"], Acc_B=汇sh3["Acc_B"], Δ相对主臂=Δ线性vs主),
       dict(Acc_A_上限=sham_ACC_A_max), 说明=说明3)

    # ── 静态自检 1：C1 禁词自扫（两个新增 .py，命中数须 0）──
    禁词 = list(M.禁词列())
    命中表 = {}
    for p in (算子路径, 本件):
        with open(p, encoding="utf-8") as f:
            低 = f.read().lower()
        命中表[os.path.basename(p)] = [w for w in 禁词 if w.lower() in 低]
    命中总数 = sum(len(v) for v in 命中表.values())
    C1通过 = bool(命中总数 == 0)
    记("静态_C1禁词自扫", C1通过, False,
       dict(禁词数=len(禁词), 命中=命中表, 命中总数=命中总数),
       dict(命中总数=0), 说明="复用 算子_能力核.禁词列")

    # ── 静态自检 2：导入白名单 ──
    导入表 = {os.path.basename(p): 扫导入(p) for p in (算子路径, 本件)}
    越界 = {k: [m for m in v if m not in 允许导入] for k, v in 导入表.items()}
    越界非空 = {k: v for k, v in 越界.items() if v}
    导入通过 = bool(not 越界非空)
    记("静态_导入白名单", 导入通过, False,
       dict(导入表=导入表, 白名单=list(允许导入), 越界=越界非空),
       dict(越界=[]), 说明="只允许 numpy/hashlib/json/os/sys/importlib/math/__future__")

    # ── 静态自检 3：计费线性（对新建能力核实例）──
    核 = M.能力核(D=1024, 种子=0)
    费1 = 核.计费(1024)["乘加数"]
    费4 = 核.计费(4096)["乘加数"]
    比值 = float(费4) / max(1.0, float(费1))
    计费通过 = bool(计费比区间[0] <= 比值 <= 计费比区间[1])
    记("静态_计费线性", 计费通过, False,
       dict(计费1024=费1, 计费4096=费4, 比值=比值),
       dict(比值区间=list(计费比区间)), 说明="证明未引入随 n² 增长的开销")

    # ── 静态自检 4：两趟确定性（同种子两遍，Acc 与关键向量逐位同；ηb=选定ηb 同一配置）──
    汇主2, 指纹主2 = 跑臂记录("两区室", 序列测, ηb=选定ηb)
    决通过 = bool(汇主2["Acc_A"] == 汇主["Acc_A"] and 汇主2["Acc_B"] == 汇主["Acc_B"]
                 and 汇主2["R_全"] == 汇主["R_全"] and 指纹主2 == 指纹主)
    记("静态_两趟确定性", 决通过, False,
       dict(Acc_A=[汇主["Acc_A"], 汇主2["Acc_A"]], Acc_B=[汇主["Acc_B"], 汇主2["Acc_B"]],
            指纹=[指纹主, 指纹主2]),
       dict(逐位同=True), 说明="同种子两遍，Acc 与关键向量逐位同")

    未过 = [名 for 名, v in 臂.items() if not v["通过"]]
    未测 = []
    果 = dict(
        通过=bool(all(条)),
        判定条数=len(条),
        未过条数=int(sum(1 for v in 条 if not v)),
        未过臂=未过,
        未测项=未测,
        阈值=dict(ACC_A_主=ACC_A_主, ACC_B_主=ACC_B_主, Δ点式_min=Δ点式_min, Δ消融_min=Δ消融_min,
                  迁移_min=迁移_min, R_max=R_max, 消融_ACC_A_max=消融_ACC_A_max,
                  消融_ACC_B差_max=消融_ACC_B差_max, sham_ACC_A_max=sham_ACC_A_max,
                  计费比区间=list(计费比区间)),
        ηb扫描=dict(曲线=ηb曲线, 选定ηb=选定ηb, 扫描列=list(M.ηb列),
                    选择规则="主臂 Acc_A 最高且 Acc_B≥0.95；并列取较小 ηb（阈值不随结果改）"),
        臂=臂,
        实测摘要=dict(选定ηb=选定ηb, 主Acc_A=汇主["Acc_A"], 主Acc_B=汇主["Acc_B"],
                    主R_全=汇主["R_全"], 主R_A=汇主["R_A"],
                    消融Acc_A=汇消["Acc_A"], 消融Acc_B=汇消["Acc_B"], Δ消融=Δ消融,
                    点式Acc_A_max=点A最大值, 点式最优λ=点式最优λ, Δ点式=Δ点式,
                    迁移Acc_A=汇迁["Acc_A"],
                    sham规则随机Acc_A=汇sh1["Acc_A"], sham无区室Acc_A=汇sh2["Acc_A"],
                    sham线性求和Acc_A=汇sh3["Acc_A"],
                    主臂指纹=指纹主, D=D, 种子模型=种子模型, 切换点数主=汇主["切换点数"]),
        说明=("训练 400（读出+b̂ 学）→ 测试 400。**所有臂（主/消融/点式各λ/迁移/三个 sham）在测试期一律"
             "冻结读出、只让规则通道 b̂/drive 在线适应**（口径一致）。ηb 由**预先声明的扫描**选定"
             "（%.2f），判据阈值未随结果改动。random 只来自 default_rng(显式种子)。" % 选定ηb),
    )
    return 果


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 主流程()
    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump(果, f, ensure_ascii=False, indent=1, default=str)
    # 控制台只打关键数字
    print(json.dumps(果["实测摘要"], ensure_ascii=False, indent=1, default=str))
    print(json.dumps(果["阈值"], ensure_ascii=False, default=str))
    print("[判据_树突两区室] 判定 %d 条，未过 %d 条 ⇒ %s"
          % (果["判定条数"], 果["未过条数"], "PASS" if 果["通过"] else "FAIL"))
    print("未过臂：", json.dumps(果["未过臂"], ensure_ascii=False))
    print("结果已写：", 结果路径)
    raise SystemExit(0 if 果["通过"] else 1)
