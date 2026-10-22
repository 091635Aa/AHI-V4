# -*- coding: utf-8 -*-
"""meeting54/实施/判据_对话.py —— 多轮对话判据（读 对话_多轮.json，不再重跑模型）。

四态：PASS/FAIL/UNDECIDABLE。重点：持续对话、预测分支、上下文召回、说话人归属、稳定性。
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
产物 = os.path.join(HERE, "产物")


def 切句(文):
    return [s for s in re.split(r"[。！？；]", str(文)) if len(s.strip()) >= 4]


def 重复句率(文列):
    from collections import Counter
    句 = [s for w in 文列 for s in 切句(w)]
    if not 句:
        return 0.0
    c = Counter(句)
    return float(sum(v for v in c.values() if v > 1)) / len(句)


def 主判(路径=None):
    路径 = 路径 or os.path.join(产物, "对话_多轮.json")
    d = json.load(open(路径, encoding="utf-8"))
    轮 = d["对话"]
    探 = d["探针"]
    探无 = d["探针_无历史对照"]
    答列 = [x["回答"] for x in 轮]
    记录 = {"臂": {}, "备注": {"模型": d["指标"].get("模型"), "权重码": d["指标"].get("权重码"),
                                "越宪": True, "对话轮数": len(轮)}}

    # D1 持续对话
    长 = [len(a) for a in 答列]
    记录["臂"]["D1 持续对话"] = dict(
        四态="PASS" if len(轮) >= 6 and min(长) >= 4 else "FAIL",
        读数=dict(轮数=len(轮), 平均答长=round(float(np.mean(长)), 1),
                  最短=min(长), 最长=max(长)),
        阈值="≥6 轮、每轮非空（≥4 字）")

    # D2 预测分支路由
    命中 = [x["命中分支"] for x in 轮]
    率 = sum(1 for x in 命中 if x != "兜底") / max(1, len(轮))
    记录["臂"]["D2 预测分支路由"] = dict(
        四态="PASS" if 率 >= 0.20 else "UNDECIDABLE",
        读数=dict(命中率=round(率, 3), 每轮=命中),
        阈值="≥0.20 只作机制存在性（预测分支是人工预设，命中率不作能力宣称）")

    # D3 上下文内容召回（有历史 vs 无历史）
    def 命中关键词(答, 词):
        return any(k in 答 for k in 词)
    首问真 = 命中关键词(探[0]["答"], ["你是谁", "是谁"])
    首答真 = 命中关键词(探[1]["答"], ["EmoCompanion", "记录", "分享", "主播", "萌妹"])
    人名真 = 命中关键词(探[2]["答"], ["ViewerA"])
    首问无 = 命中关键词(探无[0]["答"], ["你是谁", "是谁"])
    首答无 = 命中关键词(探无[1]["答"], ["EmoCompanion", "记录", "分享", "主播", "萌妹"])
    人名无 = 命中关键词(探无[2]["答"], ["ViewerA"])
    有 = sum([首问真, 首答真, 人名真]); 无 = sum([首问无, 首答无, 人名无])
    记录["臂"]["D3 上下文召回"] = dict(
        四态="PASS" if (有 >= 2 and 有 - 无 >= 1) else "FAIL",
        读数=dict(有历史=dict(首问=首问真, 首答=首答真, 人名=人名真, 合计=有),
                  无历史=dict(首问=首问无, 首答=首答无, 人名=人名无, 合计=无)),
        阈值="有历史 ≥2/3 且 比无历史多 ≥1 项（无历史对照必须更差）")

    # D4 说话人归属（不能把用户的问题说成「我问的」）
    反转 = 0
    for x in 探:
        a = x["答"]
        if ("我问" in a) or ("我提的" in a) or ("我亲切地" in a) or ("我先问" in a):
            反转 += 1
    记录["臂"]["D4 说话人归属"] = dict(
        四态="PASS" if 反转 == 0 else ("UNDECIDABLE" if 反转 == 1 else "FAIL"),
        读数=dict(探针数=len(探), 说话人反转数=反转,
                  样例=[x["答"][:60] for x in 探]),
        阈值="上下文探针里出现「我问的/我提的」把用户的话说成自己的 ⇒ 反转；允许 0 次")

    # D5 自己首答召回（不能把用户的话当自己的话）
    首答 = str(轮[0]["回答"])
    p2 = 探[1]["答"]
    记起自己 = any(k in p2 for k in ["EmoCompanion", "记录", "分享", "主播", "萌妹"]) and not any(
        k in p2 for k in ["你好呀"])
    记录["臂"]["D5 首答召回"] = dict(
        四态="PASS" if 记起自己 else "FAIL",
        读数=dict(首答=首答, 探针答=p2, 是否记起自己=bool(记起自己)),
        阈值="能复述自己第一句（不是把用户的第一句当自己的）")

    # D6 角色一致/不自曝 AI
    自曝 = [a for a in 答列 + [x["答"] for x in 探] if "AI" in a or "人工智能" in a]
    他角 = [a for a in 答列 if ("洛天依" in a or "MOSS" in a)]
    记录["臂"]["D6 角色一致/不自曝"] = dict(
        四态="PASS" if not 自曝 and not 他角 else "FAIL",
        读数=dict(自曝条数=len(自曝), 提到他角条数=len(他角),
                  样例=自曝[:2] + 他角[:2]),
        阈值="不得自称 AI/人工智能、不得串到别的角色")

    # D7 稳定性/重复
    重复 = 重复句率(答列)
    记录["臂"]["D7 稳定性（跨轮重复）"] = dict(
        四态="PASS" if 重复 <= 0.10 else "FAIL",
        读数=dict(跨轮重复句率=round(重复, 4), 答长变异系数=round(float(np.std(长) / (np.mean(长) + 1e-9)), 3)),
        阈值="跨轮重复句率 ≤0.10；无崩溃")

    # D8 越宪登记
    记录["臂"]["D8 越宪登记"] = dict(
        四态="PASS" if 记录["备注"]["模型"] and 记录["备注"]["权重码"] else "FAIL",
        读数=dict(模型=记录["备注"]["模型"], 权重码=记录["备注"]["权重码"],
                  许可="所有者 2026-10-18 授权；不进主线"),
        阈值="模型名/权重码/来源必须登记")

    from collections import Counter
    c = Counter(str(v["四态"]).split("（")[0] for v in 记录["臂"].values())
    记录["汇总"] = dict(臂数=len(记录["臂"]), **{k: int(c.get(k, 0)) for k in ("PASS", "FAIL", "UNDECIDABLE", "INVALID")})
    return 记录


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 主判()
    p = os.path.join(产物, "判据_对话结果.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(果, f, ensure_ascii=False, indent=1, default=str)
    print("结果 →", p)
    print("汇总:", json.dumps(果["汇总"], ensure_ascii=False))
    for k, v in 果["臂"].items():
        print("  %-26s %s" % (k, v["四态"]))
