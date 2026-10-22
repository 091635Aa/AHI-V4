# -*- coding: utf-8 -*-
"""meeting28/chat.py —— #216 AHI 对话体 CLI。

用法：python chat.py [--会话 chat.pkl] [--teach 语料.txt]
命令：/save /load /stats /teach <文件> /quit；其余输入=对话。
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"<workspace>/03_AHI\AHI2.1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--会话", default=os.path.join(HERE, "chat.pkl"))
    ap.add_argument("--teach", default=None, help="启动时灌注语料文件（UTF-8）")
    args = ap.parse_args()

    from dialogue_bridge import 对话桥

    if os.path.exists(args.会话):
        桥 = 对话桥.载入(args.会话)
        print("[AHI] 会话已载入：%s（第 %d 拍，%d 段记忆）"
              % (args.会话, 桥.拍, len(桥.codec._段)))
    else:
        桥 = 对话桥(seed=216)
        print("[AHI] 新会话（seed=216）。我是 AHI——一个不是神经网络的引擎。"
              "输入与你对话；/teach <文件> 灌注语料；/stats 看我的状态；/quit 退出。")
    if args.teach and os.path.exists(args.teach):
        文本 = open(args.teach, encoding="utf-8").read()
        n = 桥.教(文本)
        print("[AHI] 语料灌注完成：%d 句" % n)

    while True:
        try:
            用户 = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            用户 = "/quit"
        if not 用户:
            continue
        if 用户 == "/quit":
            桥.保存(args.会话)
            print("[AHI] 会话已保存：%s" % args.会话)
            break
        if 用户 == "/save":
            桥.保存(args.会话)
            print("[AHI] 已保存。")
            continue
        if 用户 == "/stats":
            print("[AHI] 拍=%d 记忆段=%d 交互=%d" % (桥.拍, len(桥.codec._段), len(桥.历史)))
            continue
        if 用户.startswith("/teach"):
            路径 = 用户.split(None, 1)[1] if len(用户.split(None, 1)) > 1 else ""
            if os.path.exists(路径):
                n = 桥.教(open(路径, encoding="utf-8").read())
                print("[AHI] 灌注 %d 句。" % n)
            else:
                print("[AHI] 文件不存在：%s" % 路径)
            continue
        r = 桥.步(用户)
        print("AHI[%s]> %s" % (r["动作"], r["响应"]))
        桥.保存(args.会话)                              # 每轮自动持久化
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
