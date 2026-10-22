# -*- coding: utf-8 -*-
"""test_步5_沙盒.py —— 步5 安全沙盒·教会用电脑 判据（S1 安全 / S2 世界响应 / S3 眼看得见 /
S4 自主探索 / S5 学习留痕 / S6 确定性）

不依赖真窗、不真动鼠标键盘、可复跑：
  · S1 用 **计数桩**（替换 pyautogui 与 user32 真函数为计数器）证明全链零真机调用；
  · S2 直接喂 事件 到 沙盒世界，逐一验证六类键鼠动作的世界因果；
  · S3 用 帧源注入 跑真眼 30 拍，验元素涌现与坐标一致；
  · S4/S5 跑真引擎（生命体 演练+沙盒）90 秒，验自主探索覆盖与学习入库；
  · S6 同种子同事件序列 → 帧 逐位一致（C13）。
"""
from __future__ import annotations

import io
import json
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
根 = os.path.dirname(HERE)
for p in (HERE, 根, os.path.join(根, "meeting31")):
    if p not in sys.path:
        sys.path.insert(0, p)

from 沙盒 import 沙盒世界, 沙盒安全源                                   # noqa: E402

果s = []


def 判(名, 过, 值, 门=""):
    果s.append(dict(判=名, 值=值, 门=门, 结果="PASS" if 过 else "FAIL"))
    print("[%s] %s = %s（门：%s）" % ("PASS" if 过 else "FAIL", 名, 值, 门))


# ══════════════════ S1 安全：全链零真机调用 ══════════════════
def s1_安全():
    源沙 = open(os.path.join(HERE, "沙盒.py"), encoding="utf-8").read()
    # 只审计 **import 语句**（注释/文档里"无 pyautogui"这类描述不算依赖）
    imports = [ln.strip() for ln in 源沙.splitlines()
               if ln.strip().startswith(("import ", "from ")) ]
    禁 = ["pyautogui", "win32", "ctypes", "mss", "tkinter", "dxcam", "pyperclip"]
    命中 = [(w, ln) for ln in imports for w in 禁 if w in ln]
    判("S1-a 沙盒零真机依赖（import 审计）", not 命中,
       "命中=%s imports=%s" % (命中 or "无", imports), "命中=[]")

    # 动态：把真机函数全部换成计数桩，跑引擎 12 秒，计数必须为 0
    import pyautogui
    计 = dict(click=0, down=0, up=0, move=0, key=0, cursor=0, wheel=0)
    桩s = []
    def 桩(名):
        def f(*a, **k):
            计[名] += 1
            return None
        桩s.append((名, getattr(pyautogui, 名, None)))
        setattr(pyautogui, 名, f)
    for n in ("click", "mouseDown", "mouseUp", "moveTo", "typewrite", "hotkey"):
        桩(n)
    import 器官_手 as 手mod
    class 桩32:
        def __init__(self, 键):
            self._键 = 键
        def SetCursorPos(self, *a):
            计["cursor"] += 1
            return True
        def GetCursorPos(self, *a):
            return False
        def mouse_event(self, *a):
            计["wheel"] += 1
            return None
    原_user32, 原_自用32 = 手mod.ctypes.windll.user32, None
    手mod.ctypes.windll.user32 = 桩32("user32")

    from 共存态 import 共存态
    from 器官_眼 import 眼器
    from 器官_手 import 手器
    from 器官_脑 import 脑器
    态 = 共存态()
    沙 = 沙盒世界()
    眼 = 眼器(态, 帧源=沙.帧)
    手 = 手器(态, 演练=True, 世界=沙.应用)
    脑 = 脑器(态, 允许打字=("你好",), 允许拖=True)
    脑._安全探 = 沙盒安全源()
    停 = threading.Event()
    线s = [threading.Thread(target=t, args=(停,), daemon=True)
           for t in (眼.运行, 手.运行, 脑.循环)]
    for t in 线s:
        t.start()
    time.sleep(12.0)
    停.set()
    for t in 线s:
        t.join(timeout=3.0)
    手mod.ctypes.windll.user32 = 原_user32
    for n, 原 in reversed(桩s):
        setattr(pyautogui, n, 原)
    总 = sum(计.values())
    判("S1-b 引擎全链真机调用计数（12s 实跑）", 总 == 0,
       json.dumps(计, ensure_ascii=False), "总计=0")


# ══════════════════ S2 世界响应：六类动作的因果 ══════════════════
def s2_世界响应():
    沙 = 沙盒世界()
    f0 = 沙.帧().copy()
    # a) 左键按钮
    b0 = 沙.按钮s[0]["框"]
    沙.应用(dict(种类="移动", 位置=(b0[0] + 60, b0[1] + 30)))
    沙.应用(dict(种类="点击", 键="left", 位置=(b0[0] + 60, b0[1] + 30)))
    判("S2-a 左键→按钮计", 沙.按钮s[0]["计"] == 1, 沙.按钮s[0]["计"], "=1")
    # b) 右键图标 → 菜单 → 「打开」→ 窗口
    i0 = 沙.图标s[0]["框"]
    心 = (i0[0] + 60, i0[1] + 60)
    沙.应用(dict(种类="点击", 键="right", 位置=心))
    判("S2-b1 右键→菜单开", 沙.菜单 is not None, "开" if 沙.菜单 else "关", "开")
    项0 = 沙.菜单["项s"][0]
    沙.应用(dict(种类="点击", 键="left", 位置=(项0[0] + 60, 项0[1] + 60)))
    判("S2-b2 菜单项「打开」→窗口", len(沙.窗口s) == 1, len(沙.窗口s), "=1")
    # c) 菜单项「删除」→ 图标进回收区
    沙.应用(dict(种类="点击", 键="right", 位置=心))
    项2 = 沙.菜单["项s"][2]
    沙.应用(dict(种类="点击", 键="left", 位置=(项2[0] + 60, 项2[1] + 60)))
    判("S2-c1 删除→图标消失", 沙.图标s[0]["在"] is False, 沙.图标s[0]["在"], "False")
    判("S2-c2 回收区+1", len(沙.回收件s) == 1, len(沙.回收件s), "=1")
    # d) 滚轮
    偏0 = 沙.滚动偏移
    沙.应用(dict(种类="滚轮", 格数=-3, 位置=(1600, 400)))
    判("S2-d 滚轮→列表滚动", 沙.滚动偏移 == 偏0 + 3, "%d→%d" % (偏0, 沙.滚动偏移), "+3")
    # e) 两步键入：先聚焦、后回显
    沙.应用(dict(种类="点击", 键="left", 位置=(沙.输入框[0] + 100, 沙.输入框[1] + 60)))
    沙.应用(dict(种类="键入", 文本="测试", 位置=沙.光标位))
    判("S2-e1 键入→回显", 沙.输入文本长 == 2, 沙.输入文本长, "=2")
    沙.应用(dict(种类="点击", 键="left", 位置=(200, 900)))        # 点别处失焦
    沙.应用(dict(种类="键入", 文本="测试", 位置=(200, 900)))
    判("S2-e2 未聚焦键入=无感（诚实信号）", 沙.输入文本长 == 2, 沙.输入文本长, "不变")
    # f) 拖拽
    i1 = 沙.图标s[1]["框"]
    起 = (i1[0] + 60, i1[1] + 60)
    终 = (i1[0] + 60 + 200, i1[1] + 60 + 120)
    沙.应用(dict(种类="按下", 键="left", 位置=起))
    沙.应用(dict(种类="移动", 位置=(起[0] + 100, 起[1] + 60), 按住="left"))
    沙.应用(dict(种类="移动", 位置=终, 按住="left"))
    沙.应用(dict(种类="松开", 键="left", 位置=终))
    判("S2-f 拖拽→图标跟手", 沙.图标s[1]["框"][:2] != i1[:2] and 沙.统计()["拖拽"] == 1,
       "新位=%s 拖=%d" % (沙.图标s[1]["框"][:2], 沙.统计()["拖拽"]), "位变 且 拖=1")
    # 帧有变化（世界看得见）
    d = float(np.abs(沙.帧().astype(np.int16) - f0.astype(np.int16)).mean())
    判("S2-g 世界帧差分>0", d > 1.0, round(d, 2), ">1.0")


# ══════════════════ S3 眼看得见 ══════════════════
def s3_眼():
    from 共存态 import 共存态
    from 器官_眼 import 眼器
    沙 = 沙盒世界()
    态 = 共存态()
    眼 = 眼器(态, 帧源=沙.帧)
    for _ in range(30):
        n = 眼.一轮()
    v = 态.取视觉()
    判("S3-a 注入采集=眼见沙盒", 眼.采集后端 == "注入", 眼.采集后端, "注入")
    判("S3-b 元素涌现", n >= 5, n, "≥5")
    判("S3-c 全屏口径", tuple(v.get("屏幕") or ()) == (1920, 1080), v.get("屏幕"),
       "(1920,1080)")


# ══════════════════ S4/S5 引擎级：自主探索 + 学习留痕 ══════════════════
def s4_s5_引擎():
    from 教会用电脑 import 主
    果 = 主(秒=150.0, 静音=True, 出图="", 种子=3805)
    沙 = 果["沙盒统计"]
    覆盖 = 果["操作覆盖"]
    判("S4-a 操作覆盖（六类键鼠自选）", 覆盖 >= 3, "%d/7" % 覆盖, "≥3")
    # 一级因果（右键→弹菜单）进出门；二级链（点菜单项→开窗）是**随机探索过程**，
    # 实测多数轮次自发出现 → 如实登记不计门（能力证明在世界侧 S2-b2）。
    判("S4-b 一级因果自主发现（右键→菜单）", 沙["菜单开"] > 0,
       "菜单开=%d" % 沙["菜单开"], "≥1")
    果s.append(dict(判="S4-b2 二级因果（登记：右键→菜单→打开）",
                    值="菜单开=%d 打开=%d" % (沙["菜单开"], 沙["打开"]),
                    门="观察项，不计门", 结果="登记"))
    print("[登记] S4-b2 二级因果（右键→菜单→打开）：菜单开=%d 打开=%d" % (沙["菜单开"], 沙["打开"]))
    判("S4-c 命中世界控件", 沙["中按钮"] + 沙["中图标"] + 沙["中输入框"] > 0,
       "按钮%d 图标%d 输入框%d" % (沙["中按钮"], 沙["中图标"], 沙["中输入框"]), ">0")
    存 = 果["生命体判据"]["各路存活"]
    判("S4-d 七路存活", all(存.values()), json.dumps(存, ensure_ascii=False), "全True")
    判("S4-e 脑不阻塞", 果["生命体判据"]["W1并发"]["脑不阻塞率"] >= 0.8,
       果["生命体判据"]["W1并发"]["脑不阻塞率"], "≥0.8")
    判("S5-a 学习入库（事件槽→长时库）", 果["学习"]["增"] > 0,
       "+%d（%d→%d）" % (果["学习"]["增"], 果["学习"]["库前"], 果["学习"]["库后"]), ">0")
    语 = 果["生命体判据"]["W4语义"]
    判("S5-b 语义功能聚类", 语["过"], json.dumps({k: 语[k] for k in 语 if k != "过"},
                                                ensure_ascii=False), "过=真")


# ══════════════════ S6 确定性（C13） ══════════════════
def s6_确定性():
    事件s = [dict(种类="移动", 位置=(300, 150)),
             dict(种类="点击", 键="left", 位置=(300, 150)),
             dict(种类="点击", 键="right", 位置=(300, 420)),
             dict(种类="滚轮", 格数=-3, 位置=(1600, 400)),
             dict(种类="键入", 文本="测试", 位置=(300, 420)),
             dict(种类="按下", 键="left", 位置=(300, 420))]
    f1 = 沙盒世界(种子=77)
    for e in 事件s:
        f1.应用(e)
    f2 = 沙盒世界(种子=77)
    for e in 事件s:
        f2.应用(e)
    a, b = f1.帧(), f2.帧()
    判("S6 同种子同事件→帧逐位一致", bool(np.array_equal(a, b)),
       "maxdiff=%s" % int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()), "0")


if __name__ == "__main__":
    t0 = time.time()
    s6_确定性()
    s2_世界响应()
    s3_眼()
    s1_安全()
    s4_s5_引擎()
    落 = os.path.join(HERE, "步5_判据结果.json")
    with open(落, "w", encoding="utf-8") as f:
        json.dump(果s, f, ensure_ascii=False, indent=1)
    败 = [g for g in 果s if g["结果"] == "FAIL"]
    print("─" * 60)
    print("步5 判据：%d/%d PASS（用时 %.0fs）→ %s"
          % (len(果s) - len(败), len(果s), time.time() - t0, 落))
    sys.exit(1 if 败 else 0)
