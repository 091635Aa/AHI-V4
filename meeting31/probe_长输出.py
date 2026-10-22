# -*- coding: utf-8 -*-
"""meeting31/probe_长输出.py —— #260 单次 ≥10K 中文字符长输出：链式条件生成。

条件=题材行尾+上一句尾（部分链式）；温度随进度爬升；去重=全局唯一句集；
接受宽松（len≥4 且全局唯一）防拒绝死转；每 50 句插入真实遥测注记（不计字数）。
产物：输出/长输出_10K.md。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")

import numpy as np

from synthesis import 综合体

目标字数 = 10000
seed = 311
上限句 = 3000

桥 = 综合体(seed=seed)
桥.设开放域(["<workspace>/"],
           ["<workspace>/\03_AHI\\AHI2.1\\backup_20261004_v10自言体"])
print("预热：先探索 300 拍积累语料…", flush=True)
for _ in range(300):
    桥.断供一拍()
题材s = [x for t, x in 桥.账本 if t == "教材"]
print("细读语料 %d 行，开始长输出（目标 %d 字）…" % (len(题材s), 目标字数), flush=True)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "输出", "长输出_10K.md")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
头部 = ("# 长输出测试文本（AHI v10.13，seed %d）\n\n"
        "> 来源与边界：以下正文为 AHI 基于其细读语料（<workspace>/ 文档，账本 %d 条）"
        "的字符级统计重组（作者式续写）——形似而非语义创作，不作为知识来源。\n\n"
        % (seed, sum(1 for t, _ in 桥.账本 if t == "教材")))
with open(out_path, "w", encoding="utf-8") as f:
    f.write(头部)

生成句 = set()
尾 = ""
总字数 = 0
句数 = 0
拒句 = 0
命中s = []
游标 = 0
段内 = 0
起 = time.time()

while 总字数 < 目标字数 and 句数 < 上限句:
    题材 = 题材s[游标 % len(题材s)]
    游标 += 1
    条件 = (题材[-4:] + 尾)[-6:]
    τ = 0.7 + min(总字数 / 目标字数, 1.0) * 0.9
    接受 = None
    for 尝试 in range(6):
        rng = np.random.default_rng((seed, 桥.拍, 21, 句数, 游标, 尝试))
        s = 桥.lm_m.生成(种子=条件, 温度=τ, rng=rng, 最大长=48)
        if len(s) >= 4 and s not in 生成句:
            接受 = s
            break
        拒句 += 1
    if 接受 is None:
        游标 += 7                                   # 该题材 6 发全重→跳 7 个题材
        continue
    s = 接受
    生成句.add(s)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(s + "\n")
    尾 = s[-2:]
    总字数 += len(s)
    命中s.append(桥.lm_m.三元命中率(s))
    句数 += 1
    段内 += 1
    if 段内 >= 50:                                  # 遥测注记（不计字数）
        段内 = 0
        with open(out_path, "a", encoding="utf-8") as f:
            f.write("\n> 【遥测】第 %d 拍｜消化债 %.2f｜已读 %d 文件｜细读 %d 行｜"
                    "进度 %d/%d 字\n\n" %
                    (桥.拍, 桥.消化债, 桥.导航器.已读文件, 桥.导航器.细读行,
                     总字数, 目标字数))

唯一率 = len(生成句) / max(句数, 1)
流畅 = sum(命中s) / max(len(命中s), 1)
print("=" * 60)
print("交付判定：正文 %d 字（判据 ≥10000）｜句 %d｜唯一率 %.3f（判据 ≥0.95）"
      % (总字数, 句数, 唯一率))
print("流畅性：三元命中率均值 %.3f（对照基线 0.60）｜拒句 %d ｜耗时 %.1f 分钟"
      % (流畅, 拒句, (time.time() - 起) / 60))
print("判定：%s" % ("PASS——已达交付线" if 总字数 >= 10000 and 唯一率 >= 0.95
                    else "FAIL"))
print("产物：%s" % out_path)
