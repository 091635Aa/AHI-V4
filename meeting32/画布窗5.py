# -*- coding: utf-8 -*-
"""meeting32/画布窗5.py —— 画布 v5：色块跟随重绘=拖放语义反馈（环境=真值裁决者）。

v4 的验证只能读轨迹；v5 环境自己给语义反馈：
  - 真实鼠标事件驱动（pyautogui 拖拽=OS 事件，tkinter 全收到）；
  - 按住色块拖动 → 色块实时跟随光标（视觉跟随=拖动中语义）；
  - 释放进白框 → 色块吸附进框内格子重绘 + 计数标签「已放入 n/3」；
  - 释放框外 → 色块弹回原位（拖放失败的语义反馈）。
AHI 无需通道：一切用眼睛看（截图+HSV 数框内色块）。
"""
import tkinter as tk

根 = tk.Tk()
根.title("AHI画布v5")
根.geometry("1000x700+460+100")
根.attributes("-topmost", True)
c = tk.Canvas(根, width=1000, height=700, bg="#1a1a2e", highlightthickness=0)
c.pack()

框区 = (640, 120, 940, 380)                    # 白框（内部格子吸附）
c.create_rectangle(*框区, outline="#ffffff", width=6, dash=(8, 4))
c.create_text(790, 95, text="目标框", fill="#ffffff", font=("Arial", 16))
计数 = c.create_text(790, 410, text="已放入 0/3", fill="#ffe66d",
                     font=("Arial", 18))

色块s = []                                     # (名, id, 原位x, 原位y, 色hex)
for i, (名, 色hex, x, y) in enumerate((("红", "#e63946", 200, 480),
                                       ("绿", "#2a9d8f", 420, 560),
                                       ("蓝", "#4361ee", 620, 470))):
    pid = c.create_oval(x - 60, y - 60, x + 60, y + 60, fill=色hex,
                        outline=色hex, width=4)
    c.create_text(x, y + 85, text=名, fill="#ffffff", font=("Arial", 14))
    色块s.append([名, pid, x, y, 色hex, False])   # False=未入框

已入 = []
拖中 = None                                    # (名, id)


def 在框内(x, y) -> bool:
    return 框区[0] < x < 框区[2] and 框区[1] < y < 框区[3]


def 吸附位(i: int):
    """入框格子位（2 行 2 列）。"""
    格 = [(720, 190), (870, 190), (720, 320), (870, 320)]
    return 格[i % 4]


def 按下(ev):
    global 拖中
    for 名, pid, x, y, 色hex, 入框 in 色块s:
        if 入框:
            continue
        if (x - 65) < ev.x < (x + 65) and (y - 65) < ev.y < (y + 65):
            拖中 = (名, pid)
            c.tag_raise(pid)
            break


def 拖动(ev):
    if 拖中 is None:
        return
    名, pid = 拖中
    c.coords(pid, ev.x - 60, ev.y - 60, ev.x + 60, ev.y + 60)


def 释放(ev):
    global 拖中
    if 拖中 is None:
        return
    名, pid = 拖中
    拖中 = None
    for 条 in 色块s:
        if 条[0] != 名:
            continue
        if 在框内(ev.x, ev.y) and not 条[5]:
            gx, gy = 吸附位(len(已入))
            c.coords(pid, gx - 60, gy - 60, gx + 60, gy + 60)
            条[5] = True
            已入.append(名)
            c.itemconfig(计数, text="已放入 %d/3" % len(已入))
        elif not 在框内(ev.x, ev.y) and not 条[5]:
            c.coords(pid, 条[2] - 60, 条[3] - 60, 条[2] + 60, 条[3] + 60)


c.bind("<Button-1>", 按下)
c.bind("<B1-Motion>", 拖动)
c.bind("<ButtonRelease-1>", 释放)
根.mainloop()
