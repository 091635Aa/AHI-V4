# -*- coding: utf-8 -*-
"""meeting31/chat_cli.py —— #243 考官 CLI：对话考试 + 自言流查看（/mind）。

用法：python chat_cli.py [--载入 v10_state.pkl]
命令：/mind [n]  查看最近 n 条自言（默认 20）
      /mind 类型=复盘  按类型过滤
      /audit          自言真实性全量审计（G1）
      /quit           退出
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")

from synthesis import 综合体, 隐藏世界


def main():
    载入 = None
    if "--载入" in sys.argv:
        载入 = sys.argv[sys.argv.index("--载入") + 1]
    if 载入 and os.path.exists(载入):
        桥 = 综合体.载入(载入)
        print("[已从 %s 载入（重放式逐位重建）]" % 载入)
    else:
        桥 = 综合体(seed=310)
        for s in ("我最喜欢的颜色是蓝色。", "我养了一只猫叫团子。", "我的家乡是杭州。",
                  "我最好的朋友叫小林。", "我每天早上喝咖啡。"):
            桥.步(s)
        桥.设世界(隐藏世界(seed=77, 动作数=8, 噪声=0.05))
        print("[冷启动综合体 seed=310，已教 5 条事实；世界/教材未接入"
              "——用 /world /book 接入后可断供观察]")
    print("输入与它对话；/mind [n] 看自言流；/audit 审计；/quit 退出。")
    while True:
        try:
            q = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q == "/quit":
            break
        if q == "/audit":
            au = 桥.自言.审计(桥.账本)
            print("自言 %d 条｜引用 %d 处｜违规 %d｜真实率 %.4f ｜ %s" %
                  (len(桥.自言.流), au["引用数"], au["违规数"], au["真实率"],
                   "PASS" if au["通过"] else "FAIL"))
            continue
        if q.startswith("/mind"):
            n, 类型 = 20, None
            for tok in q.split()[1:]:
                if tok.startswith("类型="):
                    类型 = tok[3:]
                elif tok.isdigit():
                    n = int(tok)
            es = 桥.自言.自言流(n, 类型)
            print("—— 自言流（最近 %d 条%s）——" %
                  (len(es), "，类型=%s" % 类型 if 类型 else ""))
            for e in es:
                print("  #%-5d [%s] 拍%-6d %s" % (e["号"], e["类型"], e["拍"],
                                                  e["文本"][:60]))
            continue
        r = 桥.步(q)
        print("AHI> %s" % r["响应"])


if __name__ == "__main__":
    main()
