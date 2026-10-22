# -*- coding: utf-8 -*-
"""meeting28/v4_test.py —— #224 生成式对话流六判据实测。

①新句生成率 ②流畅性代理（trigram 命中率） ③状态依赖 ④持续学习（即学即用）
⑤生成自省校准 ⑥v3 回归（未知诚实+身份连续）。
"""
from __future__ import annotations

import io
import os
import sys

ROOT = r"<workspace>/03_AHI\AHI2.1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "meeting28"))

from dialogue_v4 import 对话桥v4

_out = io.StringIO()


def 说(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _out.write(s + "\n")


基础语料 = ("我最喜欢的颜色是蓝色。\n我养了一只猫叫团子。\n我周末喜欢去爬山。\n"
            "我的工作是程序员。\n我不太喜欢下雨天。")


def main():
    说("=" * 72)
    说("#224 对话体 v4：生成式对话流（纯 HDC 自回归）六判据实测")
    说("=" * 72)
    桥 = 对话桥v4(seed=216)
    桥.教_v2(基础语料, epochs=3)
    自省起点 = len(桥.自省记录)

    # —— 对话演示 ——
    说("\n【对话演示】")
    for q in ["我最喜欢的颜色是什么？", "我下午要去看牙医。", "我周末喜欢做什么？",
              "我养的那只猫叫什么名字？", "我最喜欢什么颜色？"]:
        r = 桥.步(q)
        说("你> %s" % q)
        说("AHI> %s" % r["响应"])
    说("")

    # —— ① 新句生成率 ——
    记录 = 桥.自省记录[自省起点:]
    新句率 = sum(1 for _, 新 in 记录 if 新) / max(len(记录), 1)
    说("【① 新句生成率】%d/%d = %.2f（判据 >0.5）" % (sum(1 for _, 新 in 记录 if 新),
                                                    len(记录), 新句率))

    # —— ② 流畅性代理 ——
    命中 = [桥.lm.三元命中率(g) for g, _ in 记录]
    流畅 = sum(命中) / max(len(命中), 1)
    说("【② 流畅性代理】生成连接语 trigram 命中率均值 = %.3f（判据 ≥0.60）" % 流畅)

    # —— ③ 状态依赖（同提示异状态） ——
    差异 = 0
    对 = []
    for k in range(3):
        a = 桥.步("我最喜欢的颜色是什么？")["响应"]
        桥.步("我今天走了六千步。")
        桥.自主拍(2)
        b = 桥.步("我最喜欢的颜色是什么？")["响应"]
        对.append((a.split("『")[0], b.split("『")[0]))
        差异 += int(a.split("『")[0] != b.split("『")[0])
    说("【③ 状态依赖】同提示异状态生成差异 %d/3（判据 ≥2）" % 差异)
    for i, (x, y) in enumerate(对):
        说("   对%d：A=『%s』 B=『%s』" % (i + 1, x, y))

    # —— ④ 持续学习（即学即用）：3 个独立流中任一采到新词转移即算通过 ——
    桥.步("我最喜欢吃的水果是榴莲。")
    np_ = __import__("numpy").random
    续s = [桥.lm.生成(种子="我最喜欢吃的水果是", 温度=0.8,
                      rng=np_.default_rng(777 + k), 最大长=8) for k in range(3)]
    续 = 续s[0]
    即学 = any("榴莲" in s for s in 续s)
    说("【④ 持续学习】中途教入『榴莲』→ 立即生成 3 流：%s → 即学即用=%s"
       % (" / ".join("『%s』" % s for s in 续s), 即学))

    # —— ⑤ 生成自省校准 ——
    对准 = sum(1 for g, 标 in 桥.自省记录[自省起点:] if 标 == (g not in 桥.lm.句集))
    自省准 = 对准 / max(len(桥.自省记录[自省起点:]), 1)
    说("【⑤ 生成自省校准】打标与外部真值一致 %d/%d = %.2f（判据 ≥0.90）"
       % (对准, len(桥.自省记录[自省起点:]), 自省准))

    # —— ⑥ v3 回归（未知诚实 + 身份连续） ——
    无关 = 桥.步("番茄炒蛋应该怎么做得好吃？")["响应"]
    未知诚实 = 无关.startswith("这个我还没")
    路径 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v4_chat.pkl")
    桥.保存(路径)
    桥A = 对话桥v4.载入(路径)
    桥A.步("你是谁？")
    桥A.保存(路径)
    桥B = 对话桥v4.载入(路径)
    身份一致 = 桥A.步("你是谁？")["响应"] == 桥B.步("你是谁？")["响应"]
    说("【⑥ v3 回归】未知诚实=%s ｜ 身份连续=%s" % (未知诚实, 身份一致))

    # —— 汇总 ——
    说("")
    说("=" * 72)
    说("【六判据汇总】")
    说("  ①新句生成率 %.2f >0.5：%s" % (新句率, "PASS" if 新句率 > 0.5 else "FAIL"))
    说("  ②流畅性代理 %.3f ≥0.60：%s" % (流畅, "PASS" if 流畅 >= 0.60 else "FAIL"))
    说("  ③状态依赖 %d/3 ≥2：%s" % (差异, "PASS" if 差异 >= 2 else "FAIL"))
    说("  ④即学即用：%s" % ("PASS" if 即学 else "FAIL"))
    说("  ⑤自省校准 %.2f ≥0.90：%s" % (自省准, "PASS" if 自省准 >= 0.90 else "FAIL"))
    说("  ⑥v3 回归（诚实+身份）：%s"
       % ("PASS" if 未知诚实 and 身份一致 else "FAIL"))
    说("=" * 72)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "v4_test.log"), "w", encoding="utf-8") as f:
        f.write(_out.getvalue())
    print("[log] v4_test.log 已保存")


if __name__ == "__main__":
    main()
