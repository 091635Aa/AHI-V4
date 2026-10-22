# -*- coding: utf-8 -*-
"""meeting54/实施/判据_语言.py —— 语言底层对照判据：旧「检索式」 vs 新「生成式语言器官」。

只读 挂载/说话_连贯、挂载/角色包、V5；只写 meeting54/实施/产物/。
越宪实验：LLM 臂记录模型名/权重码/种子/温度；结论不进主线宣称。
"""
from __future__ import annotations

import json
import sys
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "挂载"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "meeting53", "实施"))
import V5_双通道 as V5  # noqa: E402
import 角色包 as 包模块  # noqa: E402
import 挂载_角色 as 挂载  # noqa: E402
import 说话_连贯 as 说话  # noqa: E402
import 语言器官_LLM as LLM  # noqa: E402

np.seterr(all="ignore")
产物 = os.path.join(HERE, "产物")
os.makedirs(产物, exist_ok=True)
N = 12          # 每角色 12 段
种子 = 53001


def 切句(文):
    out = []
    for ch in "。！？；":
        文 = 文.replace(ch, "。")
    for s in 文.split("。"):
        s = s.strip()
        if len(s) >= 4:
            out.append(s)
    return out


def 重复句率(文列):
    句 = [s for w in 文列 for s in 切句(w)]
    if not 句:
        return 0.0
    from collections import Counter
    c = Counter(句)
    return float(sum(v for v in c.values() if v > 1)) / len(句)


def distinct_2(文列):
    pairs = []
    for w in 文列:
        pairs += [w[i:i + 2] for i in range(len(w) - 1)]
    if not pairs:
        return 0.0
    return len(set(pairs)) / len(pairs)


def self_bleu(文列, n=4):
    组 = []
    for w in 文列:
        组.append({w[i:i + n] for i in range(len(w) - n + 1)})
    if len(组) < 2:
        return 0.0
    s = 0.0; c = 0
    for i in range(len(组)):
        for j in range(i + 1, len(组)):
            a, b = 组[i], 组[j]
            s += len(a & b) / max(1, len(a | b)); c += 1
    return s / max(1, c)


def ngram_熵(文列, n=4):
    from collections import Counter
    c = Counter()
    for w in 文列:
        c.update(w[i:i + n] for i in range(len(w) - n + 1))
    总 = sum(c.values()) or 1
    return float(-sum(v / 总 * np.log2(v / 总) for v in c.values()))


def 结构(文列):
    句数 = [len(切句(w)) for w in 文列]
    句长 = [len(s) for w in 文列 for s in 切句(w)]
    标点 = np.mean([1.0 if (w and w[-1] in "。！？") else 0.0 for w in 文列])
    return dict(段数=len(文列), 平均句数=round(float(np.mean(句数)), 2),
                平均句长=round(float(np.mean(句长)), 1) if 句长 else 0.0,
                句末标点率=round(float(标点), 3),
                平均字数=round(float(np.mean([len(w) for w in 文列])), 1))


def 角色分类(文表):
    全部字 = sorted({c for 名 in 文表 for w in 文表[名] for c in w if "\u4e00" <= c <= "\u9fff"})
    字id = {c: i for i, c in enumerate(全部字)}
    df = {c: sum(1 for 名 in 文表 if any(c in w for w in 文表[名])) for c in 全部字}
    权 = np.array([1.0 / max(1, df[c]) for c in 全部字])

    def f(w):
        v = np.zeros(len(全部字))
        for c in w:
            if c in 字id:
                v[字id[c]] += 1
        return v * 权

    质, 测 = {}, []
    for 名 in 文表:
        行 = [f(w) for w in 文表[名]]
        质[名] = np.mean(行[:N // 2], axis=0)
        测 += [(名, v) for v in 行[N // 2:]]
    对 = 0
    for 名, v in 测:
        预 = min(质, key=lambda k: float(np.linalg.norm(v - 质[k])))
        对 += int(预 == 名)
    return 对 / max(1, len(测))


def 主测():
    结 = dict(臂={}, 备注={})
    器 = 挂载.挂载器()
    状态列 = 器.状态列(40, 种子=种子)
    包列 = 包模块.全部包()
    LM = 说话.建通用LM()
    旧器 = {名: 说话.说话器(包, LM, 种子=种子) for 名, 包 in 包列.items()}
    新器 = LLM.语言器官()

    def 状态摘要(i):
        cue, o用, 内心 = 状态列[i % len(状态列)]
        return LLM.状态摘要(cue, float(np.mean(np.abs(o用))) * 10.0, 1 + (i % 3))

    旧文, 新文 = {}, {}
    for 名, 包 in 包列.items():
        旧文[名] = [旧器[名].说(状态列[i][1], 拍=i) for i in range(N)]
        新文[名] = [新器.说(包, 状态摘要(i), 种子=种子 + i)["文"] for i in range(N)]

    # ── L1 重复（跨段）──
    值 = {}
    for 名 in 包列:
        值[名] = dict(旧=dict(重复句率=round(重复句率(旧文[名]), 4),
                             distinct2=round(distinct_2(旧文[名]), 4),
                             selfBLEU4=round(self_bleu(旧文[名]), 4),
                             四gram熵=round(ngram_熵(旧文[名]), 3)),
                      新=dict(重复句率=round(重复句率(新文[名]), 4),
                             distinct2=round(distinct_2(新文[名]), 4),
                             selfBLEU4=round(self_bleu(新文[名]), 4),
                             四gram熵=round(ngram_熵(新文[名]), 3)))
    甲 = [v["新"] for v in 值.values()]
    结["臂"]["L1 跨段重复（旧 vs 新）"] = dict(
        四态="PASS" if all(x["重复句率"] <= 0.10 and x["distinct2"] >= 0.30 and x["selfBLEU4"] <= 0.35 for x in 甲) else "FAIL",
        读数=值, 阈值="新：重复句率 ≤0.10、distinct-2 ≥0.30、self-BLEU4 ≤0.35（旧作对照）")

    # ── L2 结构 ──
    值 = {名: dict(旧=结构(旧文[名]), 新=结构(新文[名])) for 名 in 包列}
    结["臂"]["L2 说话结构"] = dict(
        四态="PASS" if all(2 <= v["新"]["平均句数"] <= 6 and 8 <= v["新"]["平均句长"] <= 60
                          and v["新"]["句末标点率"] >= 0.99 for v in 值.values()) else "FAIL",
        读数=值, 阈值="2–6 句/段、平均句长 8–60、句末标点率 ≥0.99")

    # ── L3 角色可辨识 ──
    旧acc, 新acc = 角色分类(旧文), 角色分类(新文)
    结["臂"]["L3 三角色可辨识"] = dict(
        四态="PASS" if 新acc >= 0.80 else "FAIL",
        读数=dict(旧=round(旧acc, 4), 新=round(新acc, 4), 机会=round(1 / 3, 3)),
        阈值="只读说话猜角色 ≥0.80")

    # ── L4 状态依赖 ──
    改 = 总 = 0
    for 名, 包 in 包列.items():
        for i in range(min(6, len(状态列))):
            a = 状态摘要(i)
            j = (i + 1) % len(状态列)
            b = 状态摘要(j)
            x = 新器.说(包, a, 种子=种子 + i)["文"]
            y = 新器.说(包, b, 种子=种子 + i)["文"]
            改 += int(x != y); 总 += 1
    结["臂"]["L4 状态依赖"] = dict(
        四态="PASS" if 改 / max(1, 总) >= 0.50 else "FAIL",
        读数=dict(换状态后改变率=round(改 / max(1, 总), 4), 样本=总),
        阈值="换状态 ⇒ 说的话改变 ≥0.50")

    # ── L5 稳定性：连续 12 段，前 6 vs 后 6 的重复率 ──
    值 = {}
    for 名 in 包列:
        前, 后 = 新文[名][:N // 2], 新文[名][N // 2:]
        值[名] = dict(前段重复率=round(重复句率(前), 4), 后段重复率=round(重复句率(后), 4),
                      上升=round(重复句率(后) - 重复句率(前), 4))
    结["臂"]["L5 稳定性（不越说越重复）"] = dict(
        四态=("PASS" if all(v["上升"] <= 0.10 for v in 值.values()) else ("UNDECIDABLE（n=12 太小，需 400 段长时）" if all(v["上升"] <= 0.20 for v in 值.values()) else "FAIL")),
        读数=值, 阈值="后 6 段重复率比前 6 段上升 ≤0.10")

    # ── L6 假说话必须失败 ──
    固定 = [旧文["洛天依"][0] for _ in range(N)]
    随机 = ["".join(chr(0x4e00 + (i * 7919 + k * 37) % 0x2000) for i in range(60)) for k in range(N)]
    结["臂"]["L6 must-fail：固定池/随机字"] = dict(
        四态="PASS" if (重复句率(固定) >= 0.90 and distinct_2(随机) > 0.9 and 结构(随机)["平均句数"] <= 2) else "FAIL",
        读数=dict(固定池重复率=round(重复句率(固定), 4), 随机字distinct2=round(distinct_2(随机), 4),
                  随机字段句数=结构(随机)["平均句数"]),
        阈值="固定池必须高重复；随机字必须过不了结构（句数/标点）")

    # ── L7 可复现/采样口径 ──
    包 = 包列["MOSS"]
    a = 新器.说(包, 状态摘要(0), 种子=种子)["文"]
    b = 新器.说(包, 状态摘要(0), 种子=种子)["文"]
    结["臂"]["L7 同种子两趟"] = dict(
        四态="PASS" if a == b else "FAIL",
        读数=dict(同种子两趟逐字同=bool(a == b), 采样="do_sample=True, temp=0.75, top_p=0.9, 种子固定"),
        阈值="同种子同 prompt 两趟逐字同（否则记 VOID）")

    # ── 越宪登记 ──
    结["备注"]["越宪登记"] = dict(
        模型=新器.台账["模型"], 权重码=新器.权重码, 设备=新器.台账["设备"],
        调用数=新器.台账["调用"], 平均耗时秒=round(新器.台账["墙钟秒"] / max(1, 新器.台账["调用"]), 3),
        许可="所有者 2026-10-18 授权可破 C1/C9/C10/C13/C14；本件不进 AHI4 主线宣称",
        种子=种子, 温度=0.75, top_p=0.9, max_new_tokens=120)
    from collections import Counter
    c = Counter(str(v["四态"]).split("（")[0] for v in 结["臂"].values())
    结["汇总"] = dict(臂数=len(结["臂"]), **{k: int(c.get(k, 0)) for k in ("PASS", "FAIL", "UNDECIDABLE", "INVALID")})

    # 落独白
    with open(os.path.join(产物, "语言对照_旧vs新.txt"), "w", encoding="utf-8") as f:
        f.write("【语言底层对照：旧=检索式选句；新=本地 Qwen2.5-1.5B 语言器官（越宪）】\n\n")
        for 名 in 包列:
            f.write("=" * 24 + " " + str(包列[名]["角色"]) + "\n")
            for i in range(min(6, N)):
                f.write("  旧%02d：%s\n" % (i, 旧文[名][i]))
            f.write("\n")
            for i in range(min(6, N)):
                f.write("  新%02d：%s\n" % (i, 新文[名][i]))
            f.write("\n")
    return 结


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 主测()
    p = os.path.join(产物, "判据_语言结果.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(果, f, ensure_ascii=False, indent=1, default=str)
    print("结果 →", p)
    print("汇总:", json.dumps(果["汇总"], ensure_ascii=False))
    for k, v in 果["臂"].items():
        print("  %-30s %s" % (k, v["四态"]))



