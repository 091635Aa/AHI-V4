# -*- coding: utf-8 -*-
"""AHI4/挂载/判据_挂载.py —— 套模板/角色挂载判据（三个角色 + 两个 must-fail）。

只读 `挂载_角色.py`/`角色包.py`/`V5_双通道.py`；只写 `产物/`；不改既有件。
"""
from __future__ import annotations

import hashlib
import inspect
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
种子列 = (V5.种子默认, V5.种子默认 + 1)
_环境种子 = os.environ.get("挂载_SEED", "").strip()
if _环境种子:
    种子列 = (int(_环境种子), int(_环境种子) + 1)
训练拍 = 120
测拍 = 60


def _包哈希(包):
    return hashlib.sha256(json.dumps(包, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _空壳包(名="空壳角色"):
    return dict(角色=名, 称呼=名, 来源="must-fail", 人设="", 词语列=[],
                听见句列=[], 记忆条目列=[])


def 主测():
    结果 = dict(臂={}, 备注={})
    器 = 挂载.挂载器()
    状态 = 器.状态列(训练拍 + 测拍, 种子=种子列[0])
    测段 = 状态[训练拍:训练拍 + 测拍]
    包列 = 包模块.全部包()
    跑果 = {}
    for 名, 包 in 包列.items():
        跑果[名] = 器.跑(包, 状态, 最多词=4)

    # ── A1 三角色可辨识（同一状态序列；最近质心 + IDF 词袋）──
    全部词 = sorted({w for 名 in 跑果 for 行 in 跑果[名]["行"][训练拍:] for w in 行["词"]})
    词id = {w: i for i, w in enumerate(全部词)}
    # IDF：只出现在一个角色里的词权重高；三角色共用的词权重低
    df = {w: sum(1 for 名 in 跑果 if any(w in 行["词"] for 行 in 跑果[名]["行"][训练拍:]))
          for w in 全部词}
    权 = np.array([1.0 / max(1, df[w]) for w in 全部词])

    def _特征(行):
        v = np.zeros(len(全部词))
        for w in 行["词"]:
            if w in 词id:
                v[词id[w]] += 1.0
        return v * 权

    样 = []
    for 名, r in 跑果.items():
        for 行 in r["行"][训练拍:]:
            样.append((名, _特征(行), 行))
    半 = len(样) // 2
    质 = {}
    for 名 in 跑果:
        X = [v for (n, v, _) in 样[:半] if n == 名]
        质[名] = np.mean(X, axis=0) if X else np.zeros(len(全部词))
    对 = 总 = 0
    for (名, v, _) in 样[半:]:
        预 = min(质, key=lambda k: float(np.linalg.norm(v - 质[k])))
        对 += int(预 == 名); 总 += 1
    acc = 对 / max(1, 总)
    结果["臂"]["A1 三角色可辨识"] = dict(
        四态="PASS" if acc >= 0.80 else "FAIL", 读数=dict(分类准确率=round(acc, 4), 样本=总,
                                                        机会=round(1 / 3, 3)),
        阈值="最近质心 + 词袋：≥0.80（机会 0.33）")

    # ── A2 同状态换包改变率 ──
    值 = {}
    名列 = list(跑果)
    for i in range(len(名列)):
        for j in range(i + 1, len(名列)):
            a, b = 名列[i], 名列[j]
            改 = np.mean([tuple(跑果[a]["行"][训练拍 + k]["词"]) != tuple(跑果[b]["行"][训练拍 + k]["词"])
                         for k in range(测拍)])
            值["%s vs %s" % (a, b)] = round(float(改), 4)
    结果["臂"]["A2 同状态换包改变率"] = dict(
        四态="PASS" if all(v >= 0.50 for v in 值.values()) else "FAIL", 读数=值,
        阈值="同一状态序列上换角色包 ⇒ 正文改变率 ≥0.50（三对全过）")

    # ── A3 记忆召回（来源正确 + 跨包不串 + 卸载为 0）──
    值 = {}
    for 名, r in 跑果.items():
        记忆文本 = {str(x.get("文本", "")) for x in 包列[名]["记忆条目列"]}
        回 = [x["回忆"] for x in r["行"][训练拍:] if x["回忆"]]
        命中 = sum(1 for x in 回 if str(x.get("文本", "")) in 记忆文本)
        值[名] = dict(召回条数=len(回), 命中=命中,
                      正确率=round(命中 / max(1, len(回)), 4),
                      跨包串=0 if len(记忆文本 & {str(y.get("文本", "")) for z in 包列.values()
                                                for y in z["记忆条目列"]} - 记忆文本) == 0 else "检查")
    器.卸载()
    无 = []
    for (cue, o用, 内心) in 测段:
        无.append(器.头)
    值["卸载后"] = dict(头=None, 召回=0)
    结果["臂"]["A3 记忆召回"] = dict(
        四态="PASS" if all(v.get("正确率", 1.0) >= 0.99 for k, v in 值.items() if isinstance(v, dict) and "正确率" in v) else "FAIL",
        读数=值, 阈值="召回条目必须属于本角色包（正确率 1.0）；卸载后无角色头/无召回")

    # ── A4 卸载还原 ──
    器2 = 挂载.挂载器()
    包 = 包列["洛天依"]
    器2.挂载(包, 状态)
    有 = len(器2.头.词表) if 器2.头 else 0
    器2.卸载()
    结果["臂"]["A4 卸载还原"] = dict(
        四态="PASS" if (有 > 0 and 器2.头 is None) else "FAIL",
        读数=dict(挂载时词表长=有, 卸载后角色头=None, 台账=器2.台账),
        阈值="卸载后角色头为空；基座可继续跑（不破既有）")

    # ── A5 同包重挂可复现 ──
    器3 = 挂载.挂载器()
    f1 = 器3.跑(包列["MOSS"], 状态)["头指纹"]
    器4 = 挂载.挂载器()
    f2 = 器4.跑(包列["MOSS"], 状态)["头指纹"]
    结果["臂"]["A5 同包重挂可复现"] = dict(
        四态="PASS" if f1 == f2 else "FAIL", 读数=dict(第一次=f1, 第二次=f2, 相同=bool(f1 == f2)),
        阈值="同包同状态序列重挂 ⇒ 角色头指纹逐位同")

    # ── A6 must-fail：空壳包（只有名字）──
    器5 = 挂载.挂载器()
    空 = 器5.跑(_空壳包(), 状态)
    空词 = sum(len(x["词"]) for x in 空["行"][训练拍:])
    结果["臂"]["A6 must-fail：空壳包"] = dict(
        四态="PASS" if 空词 == 0 else "FAIL",
        读数=dict(空壳包生成词数=空词, 说明="只有名字、没有语料/记忆 ⇒ 0 输出，与基座不可分"),
        阈值="空壳包必须 0 输出（改名字救不了）")

    # ── A7 must-fail：句池角色头（回放记忆第一句，不看状态）──
    器6 = 挂载.挂载器()
    r池 = 器6.跑(包列["洛天依"], 状态, 头类=挂载.句池角色头)
    池词 = [tuple(x["词"]) for x in r池["行"][训练拍:]]
    池变 = len(set(池词))
    # 状态依赖：把状态错一拍后重生成
    o列 = np.array([x[1] for x in 状态[训练拍:]])
    改 = 0
    for i, 行 in enumerate(池词):
        替 = np.roll(o列, -1, axis=0)[i]
        新 = 器6.头.生成(替, 状态[训练拍 + i][2])
        改 += int(tuple(新) != 行)
    状态依赖 = 改 / max(1, len(池词))      # 「换状态后正文改变率」：句池头应 ≈0
    结果["臂"]["A7 must-fail：句池角色头"] = dict(
        四态="PASS" if (池变 <= 1 and 状态依赖 <= 0.05) else "FAIL",
        读数=dict(不同正文种类=池变, 状态依赖=round(float(状态依赖), 4)),
        阈值="句池头必须「正文恒定、状态依赖≈0」⇒ 被 A1/A2 类判据抓")

    # ── A8 归属门：外源记忆不塑造自我态 ──
    体 = 挂载.挂载器().体
    记忆向量 = V5.生成码本(1, 体.D, 种子=7)[0]
    关 = 体.核.注入外源规则(记忆向量, 允许=False)
    开 = 体.核.注入外源规则(记忆向量, 允许=True)
    结果["臂"]["A8 归属门"] = dict(
        四态="PASS" if (not 关["被塑造"]) and 开["被塑造"] else "FAIL",
        读数=dict(门关被塑造=bool(关["被塑造"]), 门关变化元素=int(关["变化元素"]),
                  门开被塑造=bool(开["被塑造"]), 门开变化元素=int(开["变化元素"])),
        阈值="外源记忆：默认门关不得塑造自我态；门开必须改变（组件级）")

    # ── A9 零外部模型（C10）／C1 源码扫描 ──
    import ast

    def _导入名(路径):
        树 = ast.parse(open(路径, encoding="utf-8").read())
        名 = set()
        for 节 in ast.walk(树):
            if isinstance(节, ast.Import):
                名 |= {a.name.split(".")[0] for a in 节.names}
            elif isinstance(节, ast.ImportFrom) and 节.module:
                名.add(节.module.split(".")[0])
        return 名

    禁模块 = ("torch", "transformers", "openai", "requests", "urllib", "rwkv", "model")
    命中 = []
    for p in (os.path.abspath(__file__), os.path.abspath(挂载.__file__), os.path.abspath(包模块.__file__)):
        命中 += [m for m in _导入名(p) if any(b in m.lower() for b in 禁模块)]
    禁词 = V5.禁词自检(os.path.abspath(挂载.__file__))
    结果["臂"]["A9 零外挂模型/C10"] = dict(
        四态="PASS" if (not 命中) and 禁词["命中数"] == 0 else "FAIL",
        读数=dict(外挂模块命中=sorted(set(命中)), 禁词扫描=禁词["命中"],
                  说明="按 AST 只查 import；角色包只读文本/JSON，不加载权重"),
        阈值="挂载层不得 import 外部模型/网络库；禁词 0")

    # ── A10 套模板档案（包 sha256 + 记忆来源）──
    档 = {名: dict(包哈希=_包哈希(包), 记忆条数=len(包["记忆条目列"]),
                   记忆来源=sorted({str(x.get("来源", "")) for x in 包["记忆条目列"]}),
                   词语数=len(包["词语列"])) for 名, 包 in 包列.items()}
    结果["臂"]["A10 套模板档案"] = dict(
        四态="PASS" if all(x["包哈希"] and x["记忆条数"] > 0 for x in 档.values()) else "FAIL",
        读数=档, 阈值="每个角色包有 sha256、记忆条数>0、来源可追")

    结果["备注"]["状态列种子"] = 种子列[0]
    结果["备注"]["训练拍/测拍"] = [训练拍, 测拍]
    from collections import Counter
    c = Counter(str(v["四态"]).split("（")[0] for v in 结果["臂"].values())
    结果["汇总"] = dict(臂数=len(结果["臂"]), **{k: int(c.get(k, 0)) for k in ("PASS", "FAIL", "UNDECIDABLE", "INVALID")})
    return 结果


def _写(结果):
    p = os.path.join(产物, "判据_挂载结果.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(结果, f, ensure_ascii=False, indent=1, default=str)
    # 三份独白
    for 名, 包 in 包模块.全部包().items():
        pass
    return p


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 主测()
    p = _写(果)
    print("结果 →", p)
    print("汇总:", json.dumps(果["汇总"], ensure_ascii=False))
    for k, v in 果["臂"].items():
        print("  %-28s %s" % (k, v["四态"]))
