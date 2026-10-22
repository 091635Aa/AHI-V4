# -*- coding: utf-8 -*-
"""meeting38/实施/实机探索.py —— 步5 实机模式：物理机主屏上自主探索键鼠（所有者知情在场）

━━━ 与 沙盒/教会用电脑 的区别 ━━━
  · 眼：**不注入**帧源——看真实主屏（DDA 桌面复制，输出 0 = 主屏 1 号）；
  · 手：**演练=False**——真动鼠标（左/右/中键 + 滚轮）；
  · 脑：探索律照旧（新异优先+世界差分归因）；**实机默认关 拖拽/键入**——
    拖拽可能移动真实文件、键入会覆盖所有者剪贴板并触发快捷键；需要时用旗标显式开。

━━━ 安全（实机三道闸）━━━
  ① ESC 急停：全局键盘轮询（GetAsyncKeyState），所有者随时按 ESC → 手在两拍内停；
  ② 危词安全闸：真机 UIA 事实源（安全事实.py），支付/删除类控件/窗口整体回避；
  ③ 时限：--秒 到点自动收工；角区回避（右上关闭钮一带）不落点。

━━━ 诚实预期 ━━━
引擎没有语义目标（不认识「WPS」三个字，C9 禁 OCR）——它在桌面上**自主探索**：
单点任务栏/开始菜单/图标确实可能打开 WPS 甚至点中「新建」，但不保证发生；
发生了什么、没发生什么，一律如实落盘 实机_结果.json。

用法：python 实施/实机探索.py --秒 300 [--拖] [--打字] [--发声]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from 生命体 import 生命体                                          # noqa: E402

_VK_ESCAPE = 0x1B


def _ESC按下() -> bool:
    """全局 ESC 是否正被按下（独立于窗口焦点；与 演播台 同口径）。"""
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(_VK_ESCAPE) & 0x8000)
    except Exception:
        return False


def 主(秒: float = 300.0, 拖: bool = False, 打字: bool = False,
       发声: bool = False, 出图: str = "") -> dict:
    体 = 生命体(用语音=发声, 演练=False,                        # 实机：真动鼠标
                脑允许打字=(("你好", "测试") if 打字 else ()),
                脑允许拖=拖)
    库前 = int(体.我.统计.get("库记录", 0))
    停因 = dict(因="时限")
    急停 = threading.Event()

    def _急停监视():
        while not 急停.is_set() and not 体.到点.is_set():
            if _ESC按下():
                停因["因"] = "ESC急停"
                体.到点.set()
                return
            急停.wait(0.05)

    监 = threading.Thread(target=_急停监视, daemon=True)
    监.start()
    t0 = time.time()
    判 = 体.跑(时长=秒)                                        # 到点（时限/ESC）即收工结算
    用时 = round(time.time() - t0, 1)
    库后 = int(体.我.统计.get("库记录", 0))

    终帧 = None
    if 出图:
        try:
            from PIL import Image
            Image.fromarray(体.眼.最近帧).save(出图)           # 眼的最后一帧（主屏实拍）
            终帧 = 出图
        except Exception:
            终帧 = None

    动 = 体.态.取动作()
    语 = 体.语义.判据()
    果 = dict(
        程序="步5 实机探索（物理机主屏；所有者知情在场，录屏由所有者开启）",
        停止原因=停因["因"], 设定秒=秒, 实际用时s=用时,
        动作=dict(动作计数=int(动.get("动作计数", 0)),
                  放弃计数=int(动.get("放弃计数", 0)),
                  纠偏次数=int(动.get("纠偏次数", 0)),
                  最后反馈=str(动.get("反馈", ""))[:80]),
        学习=dict(库前=库前, 库后=库后, 增=库后 - 库前),
        语义=dict(按钮数=int(语.get("按钮数", 0)), 词数=int(语.get("词数", 0)),
                  容器根数=int(语.get("容器根数", 0)), 最大深度=int(语.get("最大深度", 0)),
                  点击裁决数=int(语.get("点击裁决数", 0))),
        生命体判据=判,
        安全=dict(危词闸=dict(getattr(体.脑, "_安全统计", {})),
                  拖拽=bool(拖), 键入=bool(打字)),
    )
    out = os.path.join(_HERE, "实机_结果.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(果, f, ensure_ascii=False, indent=1, default=str)
    print("─" * 60)
    print("停止=%s 用时=%ss｜动作 %d 次（纠偏 %d/放弃 %d）｜学习 %+d"
          % (停因, 用时, 果["动作"]["动作计数"], 果["动作"]["纠偏次数"],
             果["动作"]["放弃计数"], 库后 - 库前))
    print("七路存活：%s" % 判["各路存活"])
    print("结果落盘：%s" % out)
    return 果


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--秒", type=float, default=300.0)
    ap.add_argument("--拖", action="store_true", help="实机允许拖拽（默认关：防移动真实文件）")
    ap.add_argument("--打字", action="store_true", help="实机允许键入（默认关：防覆盖剪贴板）")
    ap.add_argument("--发声", action="store_true", help="允许语音输出（默认静音）")
    ap.add_argument("--出图", type=str, default="")
    a = ap.parse_args()
    主(秒=a.秒, 拖=a.拖, 打字=a.打字, 发声=a.发声, 出图=a.出图)
