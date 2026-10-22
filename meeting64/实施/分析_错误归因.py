# -*- coding: utf-8 -*-
"""meeting64/实施/分析_错误归因.py —— 把 v8.1 的 18 样本 ASR 错误拆成
「同音字（LM/转写，不一定声学错）／声母错／韵母错／声调错／删除／插入」六类。

只读 `meeting63/产物/判据/判据_可懂度_ASR_多句.json`（v8.1 官方读数），不改任何既有件（C14）。
输出 `meeting64/产物/诊断/错误归因.json` + 同目录 md。

为什么重要：CER 是字符级，**同音字**（真/测拼音完全相同）会把「声学其实对」的字也算成错；
若不拆开，就会把 LM/上下文错误误当成"渲染器缺线索"，方向判断会错。
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
M64 = os.path.dirname(HERE)
AHI4 = os.path.dirname(M64)
M63 = os.path.join(AHI4, "meeting63")
sys.path.insert(0, os.path.join(AHI4, "meeting61", "实施"))

import 音韵_v5 as Y  # noqa: E402  只读：拼音 → 声/韵/调

源 = os.path.join(M63, "产物", "判据", "判据_可懂度_ASR_多句.json")
出目录 = os.path.join(M64, "产物", "诊断")
os.makedirs(出目录, exist_ok=True)

from pypinyin import lazy_pinyin, Style  # noqa: E402


def 拼(文: str) -> list:
    """逐字拼音（TONE3，带调号数字；非汉字原样）。长度与文一致。"""
    out = []
    for ch in 文:
        if "\u4e00" <= ch <= "\u9fff":
            r = lazy_pinyin(ch, style=Style.TONE3, neutral_tone_with_five=True)
            out.append(r[0] if r else ch)
        else:
            out.append(ch)
    return out


def 解(p: str) -> dict:
    """'san1' → 声=s 韵=an 调=1；非拼音返回 None。"""
    try:
        if not p or not p[0].isascii() or p[0].isdigit():
            return None
        r = Y.解析音节(p)
        return dict(声=r["声"], 韵=r["韵"], 调=int(r["调"]))
    except Exception:  # noqa: BLE001
        return None


def 分类(真: str, 测: str) -> dict:
    """→ dict(类=..., 真拼=..., 测拼=..., 真声/韵/调, 测声/韵/调)。"""
    a, b = 解(真), 解(测)
    if a is None or b is None:
        return dict(类="非音节", 真拼=真, 测拼=测)
    同声 = a["声"] == b["声"]
    同韵 = a["韵"] == b["韵"]
    同调 = a["调"] == b["调"]
    if 同声 and 同韵 and 同调:
        类 = "同音字"
    elif not 同声 and 同韵 and 同调:
        类 = "声母错"
    elif 同声 and not 同韵 and 同调:
        类 = "韵母错"
    elif 同声 and 同韵 and not 同调:
        类 = "声调错"
    elif not 同声 and not 同韵 and 同调:
        类 = "声母+韵母错"
    elif not 同声 and 同韵 and not 同调:
        类 = "声母+声调错"
    elif 同声 and not 同韵 and not 同调:
        类 = "韵母+声调错"
    else:
        类 = "声母+韵母+声调错"
    return dict(类=类, 真拼=真, 测拼=测,
                真=dict(声=a["声"], 韵=a["韵"], 调=a["调"]),
                测=dict(声=b["声"], 韵=b["韵"], 调=b["调"]))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    with open(源, encoding="utf-8") as f:
        报 = json.load(f)

    逐样本 = 报["逐样本"]
    类计 = {}
    声母混淆 = {}
    韵母混淆 = {}
    调混淆 = {}
    按角色 = {}
    按句 = {}
    总操作 = 0
    总S = 总D = 总I = 0
    样本数 = 0
    总音节 = 0
    声母总 = {}
    声母错 = {}
    韵母总 = {}
    韵母错 = {}
    纯调错 = []
    纯段错 = 0

    for 键, v in 逐样本.items():
        句号, 角色 = v["句号"], v["角色"]
        样本数 += 1
        真文, 测文 = v["对齐"]["真文"], v["对齐"]["测文"]
        真拼 = 拼(真文)
        # 测文逐字拼音：用整句上下文读法更准
        测拼 = 拼(测文)
        # 对齐“操作”给出真值字位置（1-based）；测字串需按对齐走一遍
        测位 = 0
        for op in v["对齐"]["操作"]:
            位 = int(op["位"])
            真解 = 解(真拼[位 - 1]) if 位 - 1 < len(真拼) else None
            if 真解 is not None:
                声母总[真解["声"]] = 声母总.get(真解["声"], 0) + 1
                韵母总[真解["韵"]] = 韵母总.get(真解["韵"], 0) + 1
            if op["操作"] == "命中":
                测位 += 1
                continue
            if op["操作"] == "删除":
                c = dict(类="删除", 真拼=真拼[位 - 1] if 位 - 1 < len(真拼) else "?")
                声母混淆.setdefault(c["真拼"], {}).setdefault("(删)", 0)
                声母混淆[c["真拼"]]["(删)"] += 1
            else:  # 替换成X
                if 测位 >= len(测拼):
                    continue
                真p, 测p = 真拼[位 - 1], 测拼[测位]
                c = 分类(真p, 测p)
                if c["类"] in ("声母错", "声母+韵母错", "声母+声调错", "声母+韵母+声调错"):
                    a, b = c["真"], c["测"]
                    声母混淆.setdefault(a["声"], {}).setdefault(b["声"], 0)
                    声母混淆[a["声"]][b["声"]] += 1
                if c["类"] in ("韵母错", "声母+韵母错", "韵母+声调错", "声母+韵母+声调错"):
                    a, b = c["真"], c["测"]
                    韵母混淆.setdefault(a["韵"], {}).setdefault(b["韵"], 0)
                    韵母混淆[a["韵"]][b["韵"]] += 1
                if c["真"]["调"] != c["测"]["调"]:
                    调混淆.setdefault(c["真"]["调"], {}).setdefault(c["测"]["调"], 0)
                    调混淆[c["真"]["调"]][c["测"]["调"]] += 1
            # —— 逐声母/韵母错误率 & 纯声调错明细 ——
            if 真解 is not None:
                声母错[真解["声"]] = 声母错.get(真解["声"], 0) + 1
                韵母错[真解["韵"]] = 韵母错.get(真解["韵"], 0) + 1
            if c["类"] == "声调错":
                纯调错.append(dict(真=真文[位 - 1], 真拼=真p, 测=op.get("测"),
                                  测拼=测p, 角色=角色, 句=句号))
            elif c["类"] in ("声母错", "韵母错", "声母+韵母错"):
                纯段错 += 1
            测位 += 1
            类计[c["类"]] = 类计.get(c["类"], 0) + 1
            按角色.setdefault(角色, {}).setdefault(c["类"], 0)
            按角色[角色][c["类"]] += 1
            按句.setdefault(句号, {}).setdefault(c["类"], 0)
            按句[句号][c["类"]] += 1
            总操作 += 1
            if c["类"] == "删除":
                总D += 1
            else:
                总S += 1
        总I += len(v["对齐"].get("插入", []))
        总音节 += len(真文)

    # 准确率口径：同音字如果算“声学可接受”，则声学错率 = 非同类错 / 总字数
    同音 = 类计.get("同音字", 0)
    非同类错 = 总操作 - 同音
    总字 = 总音节 if 总音节 else 1
    总量 = 总操作 if 总操作 else 1
    宏平均CER = round(float(sum(v["CER"] for v in 逐样本.values()) / max(样本数, 1)), 4)

    def _率(总: dict, 错: dict) -> dict:
        return {k: dict(总=总[k], 错=错.get(k, 0), 错误率=round(错.get(k, 0) / 总[k], 4))
                for k in sorted(总, key=lambda k: (-(错.get(k, 0) / 总[k]), k))}

    声母率 = _率(声母总, 声母错)
    韵母率 = _率(韵母总, 韵母错)
    家族 = {"唇": ("b", "p", "m", "f"), "龈": ("d", "t", "n", "l", "z", "c", "s"),
            "卷舌": ("zh", "ch", "sh", "r"), "硬腭": ("j", "q", "x"),
            "软腭": ("g", "k", "h"), "零声母": ("",)}
    家族率 = {}
    for 名, 成s in 家族.items():
        t = sum(声母总.get(x, 0) for x in 成s)
        e = sum(声母错.get(x, 0) for x in 成s)
        if t:
            家族率[名] = dict(总=t, 错=e, 错误率=round(e / t, 4))
    家族率 = dict(sorted(家族率.items(), key=lambda kv: -kv[1]["错误率"]))

    汇总 = dict(
        工具="分析_错误归因.py", 源=源,
        样本数=样本数, 总字数=总音节,
        错误操作总数=总操作, 替换S=总S, 删除D=总D, 插入I=总I,
        各类计数=dict(sorted(类计.items(), key=lambda kv: -kv[1])),
        各类占比={k: round(v / 总量, 4) for k, v in
                  sorted(类计.items(), key=lambda kv: -kv[1])},
        CER口径=round(总操作 / 总字, 4),
        宏平均CER=宏平均CER,
        纯声调错数=len(纯调错),
        纯声调错明细=纯调错,
        纯段错数=纯段错,
        逐声母错误率=声母率,
        逐韵母错误率=韵母率,
        声母部位家族=家族率,
        扣除同音字后的CER=round(非同类错 / 总字, 4),
        同音字占错误比=round(同音 / 总量, 4),
        声母混淆={k: dict(sorted(v.items(), key=lambda kv: -kv[1])) for k, v in
                  sorted(声母混淆.items(), key=lambda kv: -sum(kv[1].values()))},
        韵母混淆={k: dict(sorted(v.items(), key=lambda kv: -kv[1])) for k, v in
                  sorted(韵母混淆.items(), key=lambda kv: -sum(kv[1].values()))},
        声调混淆={str(k): {str(kk): vv for kk, vv in sorted(v.items(), key=lambda kv: -kv[1])}
                  for k, v in sorted(调混淆.items())},
        按角色={k: dict(sorted(v.items(), key=lambda kv: -kv[1])) for k, v in 按角色.items()},
        按句={k: dict(sorted(v.items(), key=lambda kv: -kv[1])) for k, v in sorted(按句.items())},
        注="同音字＝真/测拼音（声母+韵母+声调）完全相同：字符错但声学可能正确，属 LM/上下文或转写歧义；"
          "扣除同音字后的CER 是「声学驱动错误」的上界估计。",
    )
    out = os.path.join(出目录, "错误归因.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(汇总, f, ensure_ascii=False, indent=1)

    print("=" * 78)
    print("样本 %d ｜ 总字数 %d ｜ 错误操作 %d（S=%d D=%d I=%d）"
          % (样本数, 总音节, 总操作, 总S, 总D, 总I))
    print("字符 CER = %.4f ｜ 扣除同音字后 CER = %.4f ｜ 同音字占错误 %.2f%%"
          % (汇总["CER口径"], 汇总["扣除同音字后的CER"], 汇总["同音字占错误比"] * 100))
    print("官方口径宏平均 CER = %.4f（v8.1 记录 0.5292）" % 宏平均CER)
    print("-" * 78)
    print("错误类别：")
    for k, v in 汇总["各类计数"].items():
        print("  %-18s %3d  (%.1f%%)" % (k, v, 100.0 * v / 总量))
    print("-" * 78)
    print("纯声调错（同声母同韵母，只错调）%d 条；纯段错（同调只错声/韵）%d 条" % (len(纯调错), 纯段错))
    for d in 纯调错:
        print("  %s(%s) → %s(%s)  [%s·%s]" % (d["真"], d["真拼"], d["测"], d["测拼"], d["角色"], d["句"]))
    print("-" * 78)
    print("逐声母错误率（总≥3）：")
    for k, v in 声母率.items():
        if v["总"] >= 3:
            print("  %-4s 总%3d 错%3d  %.1f%%" % (k or "∅", v["总"], v["错"], 100 * v["错误率"]))
    print("-" * 78)
    print("声母部位家族错误率：")
    for k, v in 家族率.items():
        print("  %-4s 总%3d 错%3d  %.1f%%" % (k, v["总"], v["错"], 100 * v["错误率"]))
    print("-" * 78)
    print("声母混淆（真→测，前 12）：")
    for 真, v in list(汇总["声母混淆"].items())[:12]:
        print("  %-4s → %s" % (真 or "∅", "  ".join("%s×%d" % (k or "∅", vv) for k, vv in list(v.items())[:5])))
    print("-" * 78)
    print("韵母混淆（真→测，前 12）：")
    for 真, v in list(汇总["韵母混淆"].items())[:12]:
        print("  %-5s → %s" % (真, "  ".join("%s×%d" % (k, vv) for k, vv in list(v.items())[:5])))
    print("-" * 78)
    print("声调混淆（真调→测调）：")
    for 真, v in 汇总["声调混淆"].items():
        print("  %s → %s" % (真, v))
    print("-" * 78)
    print("按角色：")
    for k, v in 汇总["按角色"].items():
        print("  %-4s %s" % (k, v))
    print("JSON:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
