# -*- coding: utf-8 -*-
"""meeting31/拖拽画布窗.py —— 拖拽画布进程（tkinter 必须主线程 mainloop）。"""
import tkinter as tk

根 = tk.Tk()
根.title("AHI拖拽画布")
根.geometry("1000x700+460+100")
根.attributes("-topmost", True)
c = tk.Canvas(根, width=1000, height=700, bg="#1a1a2e", highlightthickness=0)
c.pack()
c.create_rectangle(640, 120, 940, 380, outline="#ffffff", width=6, dash=(8, 4))
c.create_text(790, 95, text="目标框", fill="#ffffff", font=("Arial", 16))
for 色, x, y in (("#e63946", 200, 480), ("#2a9d8f", 420, 560),
                 ("#4361ee", 620, 470)):
    c.create_oval(x - 60, y - 60, x + 60, y + 60, fill=色, outline=色, width=4)
根.mainloop()
