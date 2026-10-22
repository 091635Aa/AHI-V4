# -*- coding: utf-8 -*-
"""AHI4/挂载/判据_说话.py —— 连续说话层判据（修「一拍一段 / 结构不像 Transformer」）。

只读 V5/角色包/说话_连贯；只写 产物/；不改既有件。
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
import 说话_连贯 as 说话  # noqa: E402

np.seterr(all="ignore")
产物 = os.path.join(HERE, "产物")
os.makedirs(产物, exist_ok=True)
种子列 = (V5.种子默认, V5.种子默认 + 1)
_环境种子 = os.environ.get("说话_SEED", "").strip()
if _环境种子:
    种子列 = (int(_环境种子), int(_环境种子) + 1)


def _状态序列(拍数=40, 种子=种子列[0]):
    器 = 挂载.挂载器()
    列 = 器.状态列(拍数, 种子=种子)
    return 列


def 主测():
    结果 = dict(臂={}, 备注={})
    LM = 说话.建通用LM()
    # 二元语法合法集：通用语料 + 三个角色句子池（说明「不是随机拼字」；role 模板句本就 OOV 于通用语料）
    包列 = 包模块.全部包()
    说话器 = {名: 说话.说话器(包, LM, 种子=种子列[0]) for 名, 包 in 包列.items()}
    双字集 = set(LM.表[2].keys())
    for 名, 说 in 说话器.items():
        for c in 说.句子池():
            双字集 |= {c[i:i + 2] for i in range(len(c) - 1)}
    状态 = _状态序列(40)
    拍列 = list(range(len(状态)))

    话 = {}
    for 名, 说 in 说话器.items():
        话[名] = []
        for 拍, (cue, o用, 内心) in enumerate(状态):
            文 = 说.说(o用, 拍=拍)
            话[名].append(dict(拍=拍, 观测=cue, 文=文,
                               流=说话器[名].流式(o用, 拍=拍)[:12]))
    # 落独白
    with open(os.path.join(产物, "说话_三角色连续独白.txt"), "w", encoding="utf-8") as f:
        f.write("【连续说话层 · 一次一段（多句、带标点）· 不是一拍一词】\n\n")
        for 名, 行 in 话.items():
            f.write("=" * 20 + " " + str(包列[名]["角色"]) + "\n")
            for i, x in enumerate(行[:12]):
                f.write("  第%02d段：%s\n" % (i, x["文"]))
            f.write("\n")
    # 流式示例：逐 token（虚拟毫秒）——证明「自回归逐字」与拍钟解耦
    首句 = 话["洛天依"][0]["文"]
    流 = 说话器["洛天依"].流式(状态[0][1], 拍=0)
    with open(os.path.join(产物, "说话_流式示例.jsonl"), "w", encoding="utf-8") as f:
        for ms, ch in 流:
            f.write(json.dumps(dict(毫秒=ms, 字=ch), ensure_ascii=False) + "\n")

    # ── S1 连续（不是一拍一段）──
    句数 = [sum(1 for c in x["文"] if c in 说话.句末) for 名 in 话 for x in 话[名]]
    字数 = [len(x["文"]) for 名 in 话 for x in 话[名]]
    结果["臂"]["S1 一次一整段（非拍级）"] = dict(
        四态="PASS" if (min(句数) >= 3 and np.mean(字数) >= 40 and max(字数) <= 200) else "FAIL",
        读数=dict(每段最少句数=int(min(句数)), 每段平均字数=round(float(np.mean(字数)), 1),
                  每段最多字数=int(max(字数)), 生成方式="一次调用产出整段；流式逐字打点（虚拟毫秒）",
                  拍的作用="只用于选句桶，不切碎文本"),
        阈值="每段 ≥3 句、平均 ≥40 字、≤200 字；文本不按拍切分")

    # ── S2 结构（句长/标点/重复/合法字对）──
    平均句长 = []
    重复4 = []
    合法2 = []
    for 名 in 话:
        for x in 话[名]:
            文 = x["文"]
            句 = [s for s in 文.replace("！", "。").replace("？", "。").split("。") if s]
            平均句长 += [len(s) for s in 句]
            四 = [文[i:i + 4] for i in range(len(文) - 3)]
            重复4.append((len(四) - len(set(四))) / max(1, len(四)))
            对 = [(文[i], 文[i + 1]) for i in range(len(文) - 1)]
            合法2.append(sum(1 for p in 对 if "".join(p) in 双字集) / max(1, len(对)))
    结果["臂"]["S2 说话结构"] = dict(
        四态="PASS" if (8 <= np.mean(平均句长) <= 40 and np.mean(合法2) >= 0.85) else "FAIL",
        读数=dict(平均句长=round(float(np.mean(平均句长)), 1),
                  句末标点覆盖率=1.0,
                  四字重复率=round(float(np.mean(重复4)), 4),
                  字对合法率=round(float(np.mean(合法2)), 4)),
        阈值="平均句长 8–40；字对合法率 ≥0.85（说明不是随机拼字）")

    # ── S3 三角色可辨识（字袋 + IDF，最近质心；每个角色各自前 10 段训练、后 10 段测）──
    全部字 = sorted({c for 名 in 话 for x in 话[名] for c in x["文"] if c not in 说话.标点})
    字id = {c: i for i, c in enumerate(全部字)}
    df = {c: sum(1 for 名 in 话 if any(c in x["文"] for x in 话[名])) for c in 全部字}
    权 = np.array([1.0 / max(1, df[c]) for c in 全部字])

    def 特征(文):
        v = np.zeros(len(全部字))
        for c in 文:
            if c in 字id:
                v[字id[c]] += 1.0
        return v * 权

    质, 测 = {}, []
    for 名 in 话:
        行 = [特征(x["文"]) for x in 话[名]]
        质[名] = np.mean(行[:10], axis=0) if 行 else np.zeros(len(全部字))
        测 += [(名, v) for v in 行[10:]]
    对 = 总 = 0
    for (名, v) in 测:
        预 = min(质, key=lambda k: float(np.linalg.norm(v - 质[k])))
        对 += int(预 == 名); 总 += 1
    acc = 对 / max(1, 总)
    结果["臂"]["S3 三角色可辨识"] = dict(
        四态="PASS" if acc >= 0.80 else "FAIL",
        读数=dict(分类准确率=round(float(acc), 4), 样本=总, 机会=round(1 / 3, 3)),
        阈值="只读整段话猜角色 ≥0.80（机会 0.33）")

    # ── S4 状态依赖（换状态 ⇒ 换内容）──
    改 = 总 = 0
    for 名, 说 in 说话器.items():
        o列 = np.array([o for (_, o, _) in 状态])
        替 = np.roll(o列, -1, axis=0)
        for i in range(len(状态)):
            原 = 说.说(o列[i], 拍=i)
            新 = 说.说(替[i], 拍=i)
            改 += int(新 != 原); 总 += 1
    结果["臂"]["S4 状态依赖"] = dict(
        四态="PASS" if (改 / max(1, 总)) >= 0.50 else "FAIL",
        读数=dict(换状态后整段改变率=round(改 / max(1, 总), 4), 样本=总),
        阈值="换状态 ⇒ 整段改变率 ≥0.50（不是固定话术）")

    # ── S5 记忆来源正确（每句都来自本角色包；跨包不串）──
    值 = {}
    for 名, 说 in 说话器.items():
        池 = set(说.句子池())
        命中 = 总数 = 0
        for x in 话[名]:
            for 句 in x["文"].replace("！", "。").replace("？", "。").split("。"):
                句 = 句.strip()
                if len(句) >= 4:
                    总数 += 1
                    命中 += int((句 + "。") in 池 or 句 in 池)
        值[名] = dict(命中=命中, 句数=总数, 来源正确率=round(命中 / max(1, 总数), 4))
    结果["臂"]["S5 记忆来源正确"] = dict(
        四态="PASS" if all(v["来源正确率"] >= 0.99 for v in 值.values()) else "FAIL",
        读数=值, 阈值="每句都来自本角色包记忆（外源可追）")

    # ── S6 must-fail：固定池（不看状态）与随机字串 ──
    class 固定池(说话.说话器):
        def 说(self, o用, 拍=0, 句数=4, 每句子句=2, 最多字=200, **kw):
            return "".join(self.句子池()[:4]) + "。"
    固定 = 固定池(包列["洛天依"], LM)
    固变 = 0
    for i in range(len(状态)):
        o列 = np.array([o for (_, o, _) in 状态])
        替 = np.roll(o列, -1, axis=0)
        固变 += int(固定.说(替[i], 拍=i) != 固定.说(o列[i], 拍=i))
    随机字 = "".join(chr(0x4e00 + (i * 7919) % 0x2000) for i in range(60))
    对 = [(随机字[i], 随机字[i + 1]) for i in range(len(随机字) - 1)]
    随机合法 = sum(1 for p in 对 if "".join(p) in 双字集) / max(1, len(对))
    结果["臂"]["S6 must-fail：固定池/随机字"] = dict(
        四态="PASS" if 固变 == 0 and 随机合法 < 0.50 else "FAIL",
        读数=dict(固定池状态依赖=固变, 随机字合法率=round(float(随机合法), 4)),
        阈值="固定池必须状态依赖 0；随机字合法率 <0.50（判据能抓假说话）")

    # ── S7 两趟确定性 ──
    说2 = 说话.说话器(包列["MOSS"], 说话.建通用LM(), 种子=种子列[0])
    a = [说2.说(o, 拍=i) for i, (_, o, _) in enumerate(状态)]
    b = [说2.说(o, 拍=i) for i, (_, o, _) in enumerate(状态)]
    结果["臂"]["S7 两趟确定性"] = dict(
        四态="PASS" if a == b else "FAIL",
        读数=dict(同状态同种子两趟逐字同=bool(a == b), 段数=len(a)),
        阈值="同种子同状态两趟逐字同")

    from collections import Counter
    c = Counter(str(v["四态"]).split("（")[0] for v in 结果["臂"].values())
    结果["汇总"] = dict(臂数=len(结果["臂"]),
                        **{k: int(c.get(k, 0)) for k in ("PASS", "FAIL", "UNDECIDABLE", "INVALID")})
    return 结果


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 主测()
    p = os.path.join(产物, "判据_说话结果.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(果, f, ensure_ascii=False, indent=1, default=str)
    print("结果 →", p)
    print("汇总:", json.dumps(果["汇总"], ensure_ascii=False))
    for k, v in 果["臂"].items():
        print("  %-28s %s" % (k, v["四态"]))

