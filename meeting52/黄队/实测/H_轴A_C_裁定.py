# -*- coding: utf-8 -*-
"""把主控 H 项裁定作为元数据补进汇总 JSON（只增字段，不改测量值）。"""
import io, json, os
BASE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(BASE, "H_轴A_C_汇总.json")
d = json.load(io.open(p, encoding="utf-8"))
d["主控裁定_20261017"] = {
    "来源": "主控 H 项裁定（已入 裁决/00_裁决记录.md §I/§J 与 最终方案.md U1/§3）",
    "符号约定": "Δ=基−拆（正=掉、负=拆更好）",
    "轴A": "结构性承重轴；写“焊死方向”不写“焊死已证”；可保留承重表述，但须同时写“不可补偿性未达 0.80 阈值，只判方向”（R_DiD 0.052，CI[-0.021,0.134]；TOST 上界 0.121<0.90；残差 0.22–0.25）",
    "轴C": "“平均不可判”撤销；条件性有害轴/学习上限限制器；不写装饰/删除；不作承重，降读出侧可选；R_drop/R_DiD 不适用（0/9 先降；Δ=−0.143，CI[-0.208,−0.077]；ρ=+1.000,p=5.5e-6；低基 +0.248/高基 +0.058；拆C 收敛≈0.951）",
    "320MB": "不需改（H RSS 32.5/32.8MB、commit 18.9MB）；R3「320 仅≤2000 拍」维持",
    "C13": "36/36 两趟逐位同，又一条强证据",
    "状态": "H 项关闭；黄队 standby",
}
json.dump(d, io.open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("OK", len(d), "fields; ruling keys:", list(d["主控裁定_20261017"].keys()))
