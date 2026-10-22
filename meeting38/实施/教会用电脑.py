# -*- coding: utf-8 -*-
"""meeting38/实施/教会用电脑.py —— 步5 主程序：安全沙盒里的键鼠自主探索（零真机）

━━━ 这是什么 ━━━
把引擎（眼/耳/手/嘴/语义/脑/我 七路）放进**沙盒世界**（`实施/沙盒.py`，纯 numpy、
确定性）里自己学用电脑：
  · 眼  ：帧源注入 → 看的是沙盒世界（不碰真屏）；
  · 手  ：演练模式 → 绝不真动真鼠标键盘；演练动作经 世界钩子 落地沙盒；
  · 脑  ：探索字母表扩到全键鼠（左/右/中键、滚轮、键入、拖拽）——出什么招由它自己的
          探索律（新异优先+世界差分归因）决定，**没有任何写死的任务流程**；
  · 学习闭环：动作 → 世界回应 → 眼看见 → 脑归因（大变/微变/无感）→ 语义/我 入库。

━━━ 安全（沙盒三重保险）━━━
  ① 手=演练模式（虚拟光标）；② 眼=注入帧源（不截真屏）；③ 世界=纯 numpy 内存件。
  全链对真机的接触面 = 0。真机操作（实机模式）需所有者知情，不属于本程序。

用法：
  python 实施/教会用电脑.py --秒 60 [--静音] [--出图 快照.png] [--种子 3805]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from 沙盒 import 沙盒世界, 沙盒安全源                              # noqa: E402
from 生命体 import 生命体                                          # noqa: E402


def 主(秒: float = 60.0, 静音: bool = True, 出图: str = "", 种子: int = 3805,
       键入词=("你好", "测试", "abc123")) -> dict:
    沙 = 沙盒世界(种子=种子)
    体 = 生命体(用语音=not 静音, 演练=True,
                眼帧源=沙.帧, 手世界=沙.应用,
                脑允许打字=tuple(键入词), 脑允许拖=True)
    体.脑._安全探 = 沙盒安全源()          # 沙盒事实源：窗口恒「沙盒世界」、控件名恒空
    库前 = int(体.我.统计.get("库记录", 0))
    t0 = time.time()
    判 = 体.跑(时长=秒)
    用时 = round(time.time() - t0, 1)
    沙统 = 沙.统计()
    库后 = int(体.我.统计.get("库记录", 0))
    覆盖 = sum(1 for k in ("左键", "右键", "中键", "滚上", "滚下", "键入", "拖拽")
               if 沙统.get(k, 0) > 0)
    果 = dict(
        程序="步5 教会用电脑（安全沙盒，全链零真机）",
        种子=种子, 设定秒=秒, 实际用时s=用时,
        沙盒统计=沙统,
        操作覆盖=覆盖, 覆盖门=3,
        世界事件=dict(菜单开=沙统["菜单开"], 打开=沙统["打开"], 重命名=沙统["重命名"],
                      删除=沙统["删除"]),
        学习=dict(库前=库前, 库后=库后, 增=库后 - 库前),
        生命体判据=判,
    )
    out = os.path.join(_HERE, "步5_教会用电脑结果.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(果, f, ensure_ascii=False, indent=1, default=str)
    if 出图:
        from PIL import Image
        Image.fromarray(沙.帧(带光标=True)).save(出图)
    print("─" * 60)
    print("沙盒统计：%s" % json.dumps(沙统, ensure_ascii=False))
    print("操作覆盖：%d/7（门≥3）｜学习：库 %+d（%d→%d）" % (覆盖, 库后 - 库前, 库前, 库后))
    print("七路存活：%s" % 果["生命体判据"]["各路存活"])
    print("结果落盘：%s" % out)
    return 果


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--秒", type=float, default=60.0)
    ap.add_argument("--静音", action="store_true", default=True)
    ap.add_argument("--出图", type=str, default="")
    ap.add_argument("--种子", type=int, default=3805)
    a = ap.parse_args()
    主(秒=a.秒, 静音=a.静音, 出图=a.出图, 种子=a.种子)
