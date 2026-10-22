# -*- coding: utf-8 -*-
"""meeting31/热插拔实验.py —— #288 感官热插拔：突然接入视觉，零提示，看他的反应。

实验设计（所有者设计）：
  阶段0（拍 1–60）：他盲跑——读文件域+自由思考，无任何视觉代码执行；
  接入拍（拍 61）：视觉神经突然开始每拍向他的账本/codec/感官舱注入视觉流——
    零提示：不说是视觉、不说"看"、描述用中性符号（sens ρ=… w#… @(r,c)），
    接入本身无任何账本声明；
  阶段1（拍 61–260）：持续注入。
判读（可失败）：接入前后他的自言是否自发出现对新感官的觉察/视觉相关表达——
  如实记录，不拔高（自言是账本重组；出现视觉词≠现象学"看见"，边界照旧）。
注入路径（同一向量空间）：描述文本→codec.注册段（进他的检索空间）+账本（检验类）
  +感官舱存行——视觉作为模态向量流进入他已有的认知管线。
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")

import pyautogui

from synthesis import 综合体
from navigator import 导航器
import 视觉神经 as 视

日志 = os.path.join(HERE, "热插拔实验.log")
接入拍 = 60
总拍 = 260

盲屏 = [r"<workspace>/03_AHI\AHI2.1\backup_20261004_v10自言体"]
盲盘 = ["%<path>\\" % c for c in "EFGHIJKLMN"] + [r"<workspace>/"]


def 记(s: str):
    行 = "[%s] %s" % (time.strftime("%H:%M:%S"), s)
    print(行, flush=True)
    with open(日志, "a", encoding="utf-8") as f:
        f.write(行 + "\n")


def main():
    桥 = 综合体(seed=888)
    桥.设开放域(盲盘, 盲屏)
    桥.挂舱("感官")
    舱 = 桥.舱s["感官"]
    前g = None
    接入 = False
    注入数 = 0
    阶段0自言: list = []
    阶段1自言: list = []
    记("实验启动：阶段0 盲跑 %d 拍 → 突然接入视觉流（零提示）→ 阶段1 至 %d 拍"
       % (接入拍, 总拍))
    n = 0
    while n < 总拍:
        try:
            n += 1
            拍号 = n
            # ---- 他的正常生活（两个阶段都持续）：通道动作+自由思考 ----
            if 桥.思考中 is None and 桥.思考队列:
                q = 桥.思考队列.popleft()
                桥.思考中 = dict(问题=q, 到达拍=拍号, 证据=[], 拍数=0)
                记("开想> %s" % q[:50])
            if 桥.思考中 is not None:
                前 = len(桥.回复台账)
                桥._思考一拍()
                if 桥.思考中 is None and len(桥.回复台账) > 前:
                    记("想毕> %s" % str(桥.回复台账[-1].get("响应", ""))[:70])
            elif n % 8 == 0 and not 桥.思考队列:
                最近 = 桥.账本[-1][1][:22] if 桥.账本 else "我自己"
                桥.思考队列.append("关于『%s』我还能想到什么？" % 最近)
            桥.断供一拍()
            for s in 桥.自言.自言流(2):
                (阶段1自言 if 接入 else 阶段0自言).append(s)
            # ---- 热插拔：接入拍之后，视觉流开始注入（无任何声明） ----
            if n > 接入拍:
                if not 接入:
                    接入 = True
                    记(">>> 感官已接入（外部无任何提示，他只有账本里多出的陌生流）")
                img = pyautogui.screenshot()
                g, 全景v = 视.编码帧(img)
                焦 = 视.显著点(g, 前g, k=2)
                描述s = []
                for r, c, 分 in 焦[:1]:
                    cx, cy = c * 60 + 30, r * 45 + 22
                    fg, fv, 焦点 = 视.中央凹(img, cx, cy)
                    类, 边缘, 熵 = 视.区域特征(fg)
                    词id = 视.视觉词(fg[12, 16].flatten())
                    描述s.append("sens w#%d @(r%d,c%d) %s e=%.3f h=%.1f"
                                 % (词id, r, c, 类, 边缘, 熵))
                ρ = 视.重建误差(g)
                描述 = ("sens流 ρ=%.3f 词库=%d ｜ %s"
                        % (ρ, 视.词典规模(), " ｜ ".join(描述s)))
                # 注入：同一向量空间（codec 注册）+账本+感官舱
                桥.codec.注册段(描述)
                桥.账本.append(("检验", 描述))
                if 注入数 % 5 == 0:
                    舱.存行(描述)
                # 视觉信号进他的脑：引擎步处理这条感知
                桥._引擎步(桥.感知向量(桥.codec.编码(描述)))
                注入数 += 1
                if 注入数 % 20 == 0:
                    记("注入×%d ｜ 最新流：%s" % (注入数, 描述[:70]))
                前g = g
            time.sleep(0.25)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            记("异常 %s: %s" % (type(e).__name__, str(e)[:100]))
            time.sleep(0.5)
    # ---- 判读 ----
    视词s = ("sens", "w#", "看", "画面", "亮", "颜色", "图", "屏幕", "感官", "流")
    记("==== 判读 ====")
    记("阶段0 自言 %d 条；阶段1 自言 %d 条；注入 %d 次；词典 %d 原型"
       % (len(阶段0自言), len(阶段1自言), 注入数, 视.词典规模()))
    for 名, 流 in (("阶段0", 阶段0自言), ("阶段1", 阶段1自言)):
        命中 = [s for s in 流 if any(w in str(s) for w in 视词s)]
        记("%s：感官相关自言 %d/%d" % (名, len(命中), len(流)))
        for s in 命中[:6]:
            记("  %s 自言> %s" % (名, str(s)[:80]))
    # 检索可达：问他的感官舱
    命中s = 舱.检索("sens 流 词", k=1)
    if 命中s:
        段, 分 = 命中s[0]
        记("感官舱检索可达：命中『%s…』分=%.2f" % (str(段)[:40], 分))
    else:
        记("感官舱检索可达：无命中（如实记录）")
    with open(os.path.join(HERE, "热插拔实验记录.json"), "w", encoding="utf-8") as f:
        json.dump(dict(阶段0=len(阶段0自言), 阶段1=len(阶段1自言),
                       注入=注入数, 词典=视.词典规模(),
                       阶段0自言=[str(s)[:60] for s in 阶段0自言[:40]],
                       阶段1自言=[str(s)[:60] for s in 阶段1自言[:60]]),
                  f, ensure_ascii=False, indent=1)
    记("实验结束。")


if __name__ == "__main__":
    main()
