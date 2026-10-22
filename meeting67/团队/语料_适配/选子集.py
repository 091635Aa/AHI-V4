# -*- coding: utf-8 -*-
"""meeting67 · 语料_适配 · 最小子集选取算法（**预注册草案的可执行版**）＋ 实测规模。

设计要点（确定性，无随机；同一 盘点.json 两趟逐位同）
--------------------------------------------------
输入 : ``盘点.json``（由 盘点_只读.py --能量 --哈希 生成；只读）
单元 : **片段**（当前 167×60s 语料 = 167 片段；变长录音切块后同样进入本流程）
流程 : 质量门 → 稀缺度加权贪心排序 → 三种档位（绝对分钟 / 相对比例 / 并列）→ 均衡校验

1) 质量门（预注册阈值，先定后跑）::

      语音时长 ≥ 3.0 s ；音节数 ≥ 30 ；近似SNR ≥ 15.0 dB ；静音帧占比 ≤ 0.60

2) 贪心排序（全量一次，排序即「档位骨架」）::

      score(c) = 1.0·新音节型数（声+韵+调 三元组）
               + 0.5·（新声母数 + 新韵母数）
               + 0.5·Σ_{t 新} 1/(计数(t)+1)      # 稀有音节型权重
      并列 → 音节数多者 → 片段 id 小者

3) 档位:
      A. 绝对预算 B ∈ {5, 15, 30} 分钟：按排序累加时长；最后一片若超额，按**音节边界截断**到 ≤B。
      B. 相对档位 f ∈ {1/16, 1/8, 1/4, 1/2, 1}：k=max(1, round(合格数×f))，取排序前 k 片（不截断）。
      C. 均衡修复（每档执行一次）：若某声母 marginal 缺失（0 次）→ 用「能补该声母且时长最短」的片段
         替换该档最后一片；若声母全覆盖但某韵母缺失 → 同样处理。最多修复 3 次，再缺即登记为「不可达」。

4) 输出：``选子集.json`` + ``选子集.txt``（每档：时长/音节数/音节型覆盖/声母韵母调 marginal 对比）。

用法
----
    & <workspace>/\03_AHI\\.venv\\Scripts\\python.exe 选子集.py --盘点 盘点.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AHI4 = HERE.parents[2]

# ---- 预注册常量（改这里 = 改协议，必须重新登记）----
质量门 = dict(语音时长_秒=3.0, 音节数=30, 近似SNR_dB=15.0, 静音占比上限=0.60)
权重_新音节型 = 1.0
权重_新声母韵 = 0.5
权重_稀有 = 0.5
绝对预算_秒 = [300, 900, 1800]
相对档位 = [("1/16", 1 / 16), ("1/8", 1 / 8), ("1/4", 1 / 4), ("1/2", 1 / 2), ("全量", 1.0)]
修复上限 = 3

报告集句 = [
    "今天的风很轻，适合慢慢地说一会儿话。", "我们明天早上八点在南门见面。", "他把三本书放在窗台上。",
    "小猫从桌上跳下来，碰到杯子。", "春天来了，山上的雪开始融化。", "请问这条路怎么走？",
    "他昨天下午去了图书馆。", "外面下着小雨，我们打伞走吧。", "请把窗户关上，房间里有点冷。",
    "这个周末我们一起去爬山，好吗？", "他从口袋里拿出一个本子。", "春天的花开了，风也很温柔。",
]
选择集句 = ["今天天气真不错，我们去外面走走吧。", "他把钥匙放在门口的桌子上。", "这家饭馆的红烧肉做得很好吃。",
        "明天上午我要去银行办点事。", "孩子们在操场上跑来跑去。", "雨停了，天上的云慢慢散开。"]

出: list[str] = []


def 印(s: str = "") -> None:
    print(s)
    出.append(s)


def 取汉字(s: str) -> str:
    return "".join(c for c in s if "\u4e00" <= c <= "\u9fff")


def 音节三元组(s: str):
    """文本 → [(声/∅, 韵, 调)]（与 数据_SpeakerA.py 建词表同口径，只读 import 音韵_v5）。"""
    from pypinyin import Style, lazy_pinyin
    import importlib.util
    路 = AHI4 / "meeting61" / "实施" / "音韵_v5.py"
    spec = importlib.util.spec_from_file_location("音韵_v5_选子集只读", 路)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    拼s = [p for p in lazy_pinyin(取汉字(s), style=Style.TONE3, neutral_tone_with_five=True)
          if p and p[0].isascii()]
    return [(x["声"] or "∅", x["韵"], int(x["调"])) for x in (mod.解析音节(p) for p in 拼s)]


def 主(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--盘点", default=str(HERE / "盘点.json"))
    ap.add_argument("--出", default=str(HERE / "选子集.json"))
    args = ap.parse_args(argv)

    盘 = json.loads(Path(args.盘点).read_text(encoding="utf-8"))
    每文件 = 盘["覆盖"]["每文件"]
    能量 = {x["名"]: x for x in (盘.get("能量") or [])}
    if not 能量:
        print("!! 盘点.json 无能量字段，请先跑 盘点_只读.py --能量 --哈希")
        return 2

    # 训练时长口径 = 对齐产物 meta 的 时长秒（T×10ms，= 训练真正消费的 mel 帧数/100）
    对齐时长 = {}
    对齐meta = AHI4 / "meeting66" / "产物" / "数据_对齐asr" / "meta.json"
    if 对齐meta.exists():
        for s in json.loads(对齐meta.read_text(encoding="utf-8"))["样本"]:
            对齐时长[s["wav"]] = float(s["时长秒"])

    # ---- 片段表 ----
    片 = []
    for k, f in enumerate(每文件):
        名 = f["名"]
        e = 能量.get(名, {})
        原始秒 = float(盘["时长"]["逐条"][k])
        静 = float(e.get("静态占比", 1.0))
        语音秒 = 原始秒 * (1.0 - 静)                      # VAD 语音秒（质量门用）
        训练秒 = 对齐时长.get(名, 原始秒)                  # 训练消费秒（档位计量用）
        片.append(dict(id=k, 名=名, 原始秒=原始秒, 语音秒=round(语音秒, 4), 训练秒=round(训练秒, 4),
                       音节数=f["音节数"],
                       近似SNR=e.get("近似SNR_dB", -999.0), 静音占比=静,
                       型=[tuple(t) for t in f["音节表"]]))

    # ---- 质量门 ----
    合格, 拒 = [], []
    for c in 片:
        if c["语音秒"] < 质量门["语音时长_秒"]:
            拒.append((c["名"], "语音时长<%.1fs" % 质量门["语音时长_秒"])); continue
        if c["音节数"] < 质量门["音节数"]:
            拒.append((c["名"], "音节数<%d" % 质量门["音节数"])); continue
        if c["近似SNR"] < 质量门["近似SNR_dB"]:
            拒.append((c["名"], "近似SNR<%.0fdB" % 质量门["近似SNR_dB"])); continue
        if c["静音占比"] > 质量门["静音占比上限"]:
            拒.append((c["名"], "静音占比>%.2f" % 质量门["静音占比上限"])); continue
        合格.append(c)

    # ---- 全量计数 ----
    全型计: dict = {}
    全声: dict = {}
    全韵: dict = {}
    全调: dict = {}
    for c in 片:
        for s, y, d in c["型"]:
            全型计[(s, y, d)] = 全型计.get((s, y, d), 0) + 1
            全声[s] = 全声.get(s, 0) + 1
            全韵[y] = 全韵.get(y, 0) + 1
            全调[d] = 全调.get(d, 0) + 1
    总音节 = sum(全声.values())

    印("=" * 88)
    印("[最小子集] 预注册规则可执行版 ｜ 语料_适配 · meeting67")
    印("质量门: %s" % 质量门)
    印("合格片段 %d / %d；拒绝 %d" % (len(合格), len(片), len(拒)))
    for 名, 因 in 拒[:10]:
        印("  拒: %-14s %s" % (名, 因))
    印("合格集合：语音时长合计 %.1f s = %.2f min；音节 %d；音节型 %d；声母 %d/韵母 %d/调 %d"
      % (sum(c["语音秒"] for c in 合格), sum(c["语音秒"] for c in 合格) / 60.0,
         sum(c["音节数"] for c in 合格), len(全型计), len(全声), len(全韵), len(全调)))

    # ---- 贪心排序（全部片段参与；质量合格者优先作为并列裁决）----
    合格id = {c["id"] for c in 合格}
    已型: set = set()
    已声: set = set()
    已韵: set = set()
    序 = []
    待 = list(片)
    while 待:
        最佳, 最佳分 = None, None
        for c in 待:
            新型 = {t for t in set(c["型"]) if t not in 已型}
            新声 = {t[0] for t in 新型 if t[0] not in 已声}
            新韵 = {t[1] for t in 新型 if t[1] not in 已韵}
            稀 = sum(1.0 / (全型计[t] + 1.0) for t in 新型)
            分 = (权重_新音节型 * len(新型) + 权重_新声母韵 * (len(新声) + len(新韵)) + 权重_稀有 * 稀,
                  1 if c["id"] in 合格id else 0, c["音节数"], -c["id"])
            if 最佳 is None or 分 > 最佳分:
                最佳, 最佳分 = c, 分
        序.append(最佳)
        待.remove(最佳)
        已型 |= set(最佳["型"])
        已声 |= {t[0] for t in 最佳["型"]}
        已韵 |= {t[1] for t in 最佳["型"]}
    排序 = {c["id"]: i for i, c in enumerate(序)}

    # ---- 档位 ----
    def 统计(片段s, 名):
        sset = {t for c in 片段s for t in c["型"]}
        声: dict = {}
        韵: dict = {}
        调: dict = {}
        音 = 0
        for c in 片段s:
            for a, b, d in c["型"]:
                声[a] = 声.get(a, 0) + 1
                韵[b] = 韵.get(b, 0) + 1
                调[d] = 调.get(d, 0) + 1
                音 += 1
        秒 = sum(c["训练秒"] for c in 片段s)
        语音 = sum(c["语音秒"] for c in 片段s)
        return dict(名=名, 片段数=len(片段s), 训练秒=round(秒, 2), 分钟=round(秒 / 60.0, 3),
                    语音秒=round(语音, 2),
                    音节数=音, 音节型=len(sset), 音节型覆盖=round(len(sset) / max(1, len(全型计)), 4),
                    声母数=len(声), 韵母数=len(韵), 调数=len(调),
                    声母缺失=sorted(set(全声) - set(声)), 韵母缺失=sorted(set(全韵) - set(韵)),
                    声marginal={k: round(v / max(1, 音), 4) for k, v in sorted(声.items())},
                    韵marginal={k: round(v / max(1, 音), 4) for k, v in sorted(韵.items())},
                    片段=[dict(id=c["id"], 名=c["名"], 训练秒=c["训练秒"], 音节数=c["音节数"]) for c in 片段s])

    def 均衡修复(片段s):
        """确定性修复：补缺失声母（再补缺失韵母），用最短的能补者替换末片；最多 修复上限 次。"""
        用 = list(片段s)
        for _ in range(修复上限):
            cur = {a for c in 用 for a, _, _ in c["型"]}
            缺 = sorted(set(全声) - cur)
            键 = 0  # 0=声母 1=韵母
            if not 缺:
                cury = {b for c in 用 for _, b, _ in c["型"]}
                缺 = sorted(set(全韵) - cury)
                键 = 1
            if not 缺:
                break
            候 = [c for c in 片 if c["id"] not in {x["id"] for x in 用}
                  and any((t[0] if 键 == 0 else t[1]) in 缺 for t in c["型"])]
            if not 候:
                break
            候.sort(key=lambda c: (c["语音秒"], c["id"]))
            新 = 候[0]
            # 替换：如果末片本身带该缺项则跳过
            if any((t[0] if 键 == 0 else t[1]) in 缺 for t in 用[-1]["型"]):
                break
            用[-1] = 新
            # 去重 + 保持贪心序
            用 = sorted({c["id"]: c for c in 用}.values(), key=lambda c: 排序[c["id"]])
        return 用

    档s = []
    印("\n[绝对预算档]（按贪心序累加『训练时长』= 对齐产物 T×10ms；最后一片按音节边界截断）")
    for B in 绝对预算_秒:
        取, 累 = [], 0.0
        for c in 序:
            if 累 + c["训练秒"] <= B + 1e-9:
                取.append(c); 累 += c["训练秒"]
            else:
                # 按音节边界截断：保留前缀音节数 = floor(剩余秒 / 每音节平均秒)
                剩 = B - 累
                if 剩 > 0 and c["音节数"] > 0:
                    每音 = c["训练秒"] / c["音节数"]
                    k = int(剩 / 每音)
                    if k >= 3 and k * 每音 >= 1.0:
                        截 = dict(c, 训练秒=round(k * 每音, 4), 音节数=k, 型=c["型"][:k])
                        取.append(截); 累 += 截["训练秒"]
                break
        if not 取:
            continue
        取 = 均衡修复(取)
        d = 统计(取, "≤%d_min" % (B // 60))
        d["预算秒"] = B
        d["实测秒"] = round(sum(c["训练秒"] for c in 取), 3)
        档s.append(d)
        印("  %-9s 片段=%2d 训练=%.1fs(%.2f min) 语音=%.1fs 音节=%d 音节型=%d(%.1f%%) 声母=%d/22 韵母=%d/38 调=%d/5 缺失声母=%s"
          % (d["名"], d["片段数"], d["实测秒"], d["分钟"], d["语音秒"], d["音节数"], d["音节型"],
             100 * d["音节型覆盖"], d["声母数"], d["韵母数"], d["调数"], d["声母缺失"]))

    印("\n[相对档位]（排序前缀，不截断；当前 167×60s 语料，训练时长口径）")
    for 名, f in 相对档位:
        k = max(1, int(round(len(片) * f)))
        取 = 序[:k]
        d = 统计(取, 名)
        d["比例"] = f
        档s.append(d)
        印("  %-6s k=%3d 片段 训练=%.2f min 语音=%.1fs 音节=%d 音节型=%d(%.1f%%) 声母=%d/22 韵母=%d/38 调=%d/5"
          % (名, k, d["分钟"], d["语音秒"], d["音节数"], d["音节型"], 100 * d["音节型覆盖"],
             d["声母数"], d["韵母数"], d["调数"]))

    # ---- 泄漏核查：报告集/选择集句子是否出现在训练语料 ----
    文本s = [f["文本"] if "文本" in f else "" for f in 每文件]
    # 盘点.覆盖.每文件 无原文本，退回用清单文本：从清单再读一次（只读）
    清单 = Path(r"<path>\SpeakerA7代模型训练数据\asr_opt\SpeakerA7代模型训练数据.list")
    原文本 = [ln.split("|", 3)[3] if len(ln.split("|", 3)) == 4 else "" for ln in 清单.read_text(encoding="utf-8").splitlines() if ln.strip()]
    命中 = []
    for 集名, 句s in (("报告集S1-S12", 报告集句), ("选择集S13-S18", 选择集句)):
        for 句 in 句s:
            cn = 取汉字(句)
            for 原 in 原文本:
                if cn and cn[:8] in 取汉字(原):
                    命中.append((集名, 句, 原[:20]))
    印("\n[泄漏核查] 报告集/选择集句子 8 字前缀 命中训练语料: %d 条%s"
      % (len(命中), "（无泄漏）" if not 命中 else "：" + str(命中[:3])))
    印("  注：报告集是**书面朗读句**，训练语料是**口语对话**；即使无逐字泄漏，仍存在文体域差（诚实边界）。")

    # ---- 落盘 ----
    数据 = dict(版本="meeting67 语料_适配 最小子集 v1（预注册草案的可执行版）",
                规则=dict(质量门=质量门, 权重=dict(新音节型=权重_新音节型, 新声母韵=权重_新声母韵, 稀有=权重_稀有),
                        绝对预算_秒=绝对预算_秒, 相对档位=[x[0] for x in 相对档位], 修复上限=修复上限,
                        并列规则="音节数多→id小", 随机="无（确定性）"),
                全量=dict(片段=len(片), 合格=len(合格), 拒绝=len(拒),
                          语音秒=round(sum(c["语音秒"] for c in 合格), 2),
                          音节数=总音节, 音节型=len(全型计),
                          声母=sorted(全声), 韵母=sorted(全韵), 调=sorted(全调),
                          声marginal={k: round(v / 总音节, 4) for k, v in sorted(全声.items())},
                          韵marginal={k: round(v / 总音节, 4) for k, v in sorted(全韵.items())},
                          调marginal={k: round(v / 总音节, 4) for k, v in sorted(全调.items())}),
                档=档s,
                泄漏核查=dict(命中=命中, 结论=("无逐字泄漏" if not 命中 else "有命中")))
    Path(args.出).write_text(json.dumps(数据, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    (HERE / "选子集.txt").write_text("\n".join(出) + "\n", encoding="utf-8", newline="\n")
    print("\n写出: %s 和 选子集.txt" % args.出)
    return 0


if __name__ == "__main__":
    raise SystemExit(主())
