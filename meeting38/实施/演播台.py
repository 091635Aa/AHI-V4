# -*- coding: utf-8 -*-
"""meeting38/实施/演播台.py —— 全屏只读实况演播台（视觉重启 步4）

所有者（2026-10-06 重启视觉项目）：「① 全亮视觉（真看清整屏）」。本演播台是这一步的
**看得见**的产物：把眼在**真·全屏**上的运转画成一个**全屏只读**窗口，让"看"从判据变成
看得见的轨迹（旧演示 `演示_看想点.py` 是区域裁剪版，已隔离；本件按全屏重做）。

渲染（全部读共享工作区的**最新值**；本件**只读**，不写任何槽）：
  · 底图 = 眼最近帧（**全屏**，非旧演示的区域裁剪）
  · 网格 16×9 ＋ 逐格意外**惊讶热力**（红 = 预测失败）
  · **注视链折线**（越新越亮）= 眼跳轨迹 ＋ 当前注视黄十字
  · **在场者框**（误差场涌现 / 注视确认出的结构，句柄标注）
  · **思考流**：脑原生思考 —— **只渲染「层=外」**（步1 分层：深度回响「层=内」不外显）；
    底部"嘟囔"= 最新一句朝外正文
  · HUD：拍频｜采集后端｜网格｜注视(格,停留)｜意外度｜眼跳｜在场者｜帧年龄

只读纪律（与已隔离的 可视桥.py 同律，所有者"非第八路"）：
  · 单线程画窗；只读 态.取视场 / 取视觉 / 取动作 / 取意图 / 取思考 + 眼.最近帧；
  · **不 join、不 call 任何器官**、**不写任何槽**、**不进 生命体 装配**；
  · 引擎（眼／脑／语义／嘴-文本）由本件自起，好让"看"与"想"有东西可看——但本件自身
    绝不写槽（判据 V-只读：静态源码审计 ＋ 运行期写调用拦截双重证明）。

★ 颜色（勿再翻）：DDA／眼帧是 **RGB**。本件用 tkinter+PIL 显示（PIL 期望 **RGB**），故
  **全程保持 RGB**、叠加色一律按 RGB 定义——这与 可视桥/旧演示（cv2.imshow 期望 BGR，
  故先 RGB→BGR）是**同一事实的两条正确路径**。只有可选 `--录 mp4`（cv2.VideoWriter 期望
  BGR）在落盘前经 `转()` 做一次 RGB→BGR；显示路径**绝不**做这一步（那才会红蓝互换）。

ESC 急停：tkinter 绑定 + 系统级 `GetAsyncKeyState` 轮询（不依赖窗口焦点）。

跑法：python 实施/演播台.py --秒 60                 （全屏只读，ESC/q 退出）
      python 实施/演播台.py --秒 45 --录 实施/演播台_演示.mp4
      python 实施/演播台.py --秒 20 --无头 --出图 实施/演播台_快照.png  （不开窗，供判据/CI）

铁律：C1 禁 Transformer｜C9 禁 OCR｜C10 禁外挂模型当器官｜C13 确定性可复现｜C14 不破既有。
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import threading
import time

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))       # .../meeting38/实施
根 = os.path.dirname(HERE)                              # .../meeting38
sys.path.insert(0, HERE)
sys.path.insert(0, 根)

# ── 连续采集预热必须在 共存态(torch) 之前（次序依赖，见 连续采集.py 文件头）──
try:
    from 连续采集 import 预热 as _预热
    _预热()
except Exception:
    pass

from 共存态 import 共存态                                # noqa: E402
from 器官_眼 import 眼器                                 # noqa: E402
from 器官_脑 import 脑器                                 # noqa: E402
from 器官_嘴 import 嘴器                                 # noqa: E402
from 语义 import 语义图                                  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:                                       # pragma: no cover
    Image = None

# 思考类 → 颜色（**RGB**；与全项目"语义色只按 RGB 定义一次"一致）
色表 = dict(看=(208, 208, 214), 想=(108, 210, 255), 招=(255, 209, 102),
            预=(206, 148, 255), 果=(110, 226, 138), 路=(255, 170, 120))

底栏高 = 92                 # 底部状态/嘟囔栏
面板最小宽 = 320            # 右侧「思考流」栏最小宽


def 转(rgb) -> tuple:
    """RGB → BGR。**仅**用于可选 `--录 mp4`（cv2.VideoWriter 期望 BGR）。
    显示路径（tkinter+PIL 期望 RGB）**不调用本函数**（见文件头 ★ 颜色）。"""
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def 思考行(态, n: int = 128) -> list:
    """取思考流**外层**（判据 V-思考外层 的可测接缝）：只要外=True → 内层回响不外显。
    返回 [{类,文,层}]（时间序，旧→新）。"""
    xs = 态.取思考(int(n), 只要外=True)
    return [dict(类=str(x.get("类", "?")), 文=str(x.get("文", "")), 层=str(x.get("层", "外")))
            for x in xs]


def _字体(号: int):
    if Image is None:
        return None
    for p in (r"<path>\Windows\Fonts\msyh.ttc", r"<path>\Windows\Fonts\simhei.ttf"):
        try:
            return ImageFont.truetype(p, 号)
        except Exception:
            continue
    return None


# ══════════════════════ 渲染器（无窗、可单测） ══════════════════════

class 演播渲染:
    """把共享工作区的最新值合成成一帧 RGB 图（**纯只读**，不碰任何槽的写口）。
    几何：世界视图 = 左上 (W-面板宽)×(H-底栏高)，**整屏帧等比缩放居中**（letterbox）；
          右侧 = 思考流；底部 = 状态＋嘟囔。全屏 W×H 由屏上量得。"""

    def __init__(self, 态, 眼=None, 尺寸=None):
        self.态 = 态
        self.眼 = 眼
        self.W, self.H = (int(尺寸[0]), int(尺寸[1])) if 尺寸 else (1920, 1080)
        self.面板宽 = max(面板最小宽, self.W // 5)
        self.ww, self.wh = self.W - self.面板宽, self.H - 底栏高
        self.s = 1.0                       # 帧→画布 缩放
        self.offx = self.offy = 0          # 居中偏移
        self._屏 = (self.W, self.H)
        self._字 = {13: _字体(13), 15: _字体(15), 18: _字体(18)}
        self.始t = time.time()
        self._上注 = None
        self.注视采样: list = []
        self._上次动作计数 = 0
        self._闪t = -9.0

    # ── 坐标换算（屏幕系 → 画布系；全屏帧：屏幕系 = 帧像素系） ──
    def _屏到画布(self, x, y):
        return (int(self.offx + float(x) * self.s), int(self.offy + float(y) * self.s))

    def _格到画布(self, bx, by):
        Wf, Hf = self._屏
        gx = self.眼.前端.格x if self.眼 is not None else 16
        gy = self.眼.前端.格y if self.眼 is not None else 9
        return self._屏到画布((bx + 0.5) * Wf / gx, (by + 0.5) * Hf / gy)

    def _网格(self):
        if self.眼 is not None:
            return int(self.眼.前端.格x), int(self.眼.前端.格y)
        return 16, 9

    # ── 底图（全屏帧等比居中） ──
    def _贴底图(self, 画布, 帧):
        if 帧 is None:
            画布[0:self.wh, 0:self.ww] = 34
            return
        Hf, Wf = 帧.shape[:2]
        self._屏 = (Wf, Hf)
        self.s = min(self.ww / max(Wf, 1), self.wh / max(Hf, 1))
        dw, dh = max(1, int(Wf * self.s)), max(1, int(Hf * self.s))
        self.offx, self.offy = (self.ww - dw) // 2, (self.wh - dh) // 2
        底 = cv2.resize(帧, (dw, dh), interpolation=cv2.INTER_AREA)     # 帧是 RGB；仅缩放，不改通道序
        画布[self.offy:self.offy + dh, self.offx:self.offx + dw] = 底

    # ── 覆盖层：网格 / 热力 / 注视链 / 注视 / 在场者 / 轨迹 / 意图 / 光标 ──
    def _覆盖(self, 画布, 场, 视, 动, 意):
        gxN, gyN = self._网格()
        Wf, Hf = self._屏
        x0, y0, w, h = self.offx, self.offy, int(Wf * self.s), int(Hf * self.s)
        # 网格（只画世界视图内）
        for bx in range(1, gxN):
            xx = x0 + bx * w // gxN
            cv2.line(画布, (xx, y0), (xx, y0 + h), (56, 56, 56), 1)
        for by in range(1, gyN):
            yy = y0 + by * h // gyN
            cv2.line(画布, (x0, yy), (x0 + w, yy), (56, 56, 56), 1)
        # 惊讶热力（红 = 预测失败）
        逐格 = 场.get("逐格意外")
        if 逐格 is not None:
            for i, v in enumerate(np.asarray(逐格).ravel()):
                v = float(v)
                if v <= 0.05:
                    continue
                bx, by = i % gxN, i // gxN
                rx0, ry0 = x0 + bx * w // gxN, y0 + by * h // gyN
                rx1, ry1 = x0 + (bx + 1) * w // gxN, y0 + (by + 1) * h // gyN
                roi = 画布[ry0:ry1, rx0:rx1]
                if roi.size == 0:
                    continue
                a = min(0.6, 0.10 + v * 0.7)
                色 = np.full_like(roi, (255, 64, 64))               # RGB 红
                cv2.addWeighted(roi, 1.0 - a, 色, a, 0, roi)
        # 注视链（眼跳轨迹；越新越亮）
        链 = [g for g in (场.get("注视链") or []) if g][-48:]
        点s = [self._格到画布(g[0], g[1]) for g in 链]
        for k in range(1, len(点s)):
            t = k / max(len(点s), 1)
            cv2.line(画布, 点s[k - 1], 点s[k],
                     (int(120 * (1 - t) + 90 * t), int(150 * (1 - t) + 235 * t),
                      int(255 * (1 - t) + 255 * t)),
                     2 if k > len(点s) - 3 else 1)
        # 在场者框（误差场涌现 / 注视确认的结构）
        屏 = 视.get("屏幕") or self._屏
        sx, sy = w / max(屏[0], 1), h / max(屏[1], 1)
        for e in (视.get("元素s") or []):
            ex, ey, ew, eh = e.get("框") or (0, 0, 0, 0)
            p1 = (x0 + int(ex * sx), y0 + int(ey * sy))
            p2 = (x0 + int((ex + ew) * sx), y0 + int((ey + eh) * sy))
            cv2.rectangle(画布, p1, p2, (110, 235, 130), 1)          # RGB 绿
            cv2.putText(画布, str(e.get("句柄", "")), (p1[0] + 2, p1[1] + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 235, 130), 1)
        # 手轨迹（若有手在动；本件默认无手 → 空）
        轨迹 = [p for p in (动.get("轨迹") or []) if len(p) == 2][-64:]
        if len(轨迹) >= 2:
            点t = [(x0 + int(float(p[0]) * sx), y0 + int(float(p[1]) * sy)) for p in 轨迹]
            for k in range(1, len(点t)):
                cv2.line(画布, 点t[k - 1], 点t[k], (140, 200, 255), 1)
        # 意图落点（脑"想点哪里"）
        当 = (意 or {}).get("当前") or {}
        点 = 当.get("目标点") or 当.get("心")
        if 点 and 屏[0] > 0:
            px, py = self._屏到画布(点[0], 点[1])
            if 0 <= px <= x0 + w and 0 <= py <= y0 + h:
                cv2.drawMarker(画布, (px, py), (255, 214, 0),
                               cv2.MARKER_TILTED_CROSS, 12, 2)
        # 真实光标（若有手；区域系绝对坐标）
        光标 = 动.get("光标")
        if 光标:
            ccx, ccy = self._屏到画布(光标[0], 光标[1])
            if 0 <= ccx <= x0 + w and 0 <= ccy <= y0 + h:
                cv2.circle(画布, (ccx, ccy), 4, (255, 255, 255), -1)
                cv2.circle(画布, (ccx, ccy), 9, (255, 255, 255), 1)
        # 当前注视（黄十字；最上层）
        注 = 场.get("注视")
        if 注:
            cx, cy = self._格到画布(int(注[0]), int(注[1]))
            cv2.circle(画布, (cx, cy), 13, (255, 255, 0), 1)
            cv2.line(画布, (cx - 16, cy), (cx + 16, cy), (255, 255, 0), 2)
            cv2.line(画布, (cx, cy - 16), (cx, cy + 16), (255, 255, 0), 2)
        # 点击/动作闪光
        if int(动.get("动作计数", 0) or 0) != self._上次动作计数:
            self._上次动作计数 = int(动.get("动作计数", 0) or 0)
            self._闪t = time.time()

    def _取帧率(self):
        注 = self.态.取视场().get("注视")
        if 注 and (int(注[0]), int(注[1])) != self._上注:
            self._上注 = (int(注[0]), int(注[1]))
            self.注视采样.append(self._上注)

    # ── 文本层（PIL 中文，一次往返画完） ──
    def _文(self, 画布, 项s):
        if self._字[15] is None:
            for (xy, 文, 色, 号) in 项s:
                cv2.putText(画布, str(文).encode("ascii", "replace").decode(),
                            (xy[0], xy[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 色, 1)
            return
        im = Image.fromarray(画布)                                  # 画布是 RGB → PIL RGB（正确）
        d = ImageDraw.Draw(im)
        for (xy, 文, 色, 号) in 项s:
            d.text(xy, 文, font=(self._字.get(号) or self._字[15]), fill=tuple(色))
        画布[:] = np.asarray(im)

    def _面板(self, 画布, 场, 视):
        gxN, gyN = self._网格()
        x0 = self.ww
        cv2.line(画布, (x0, 0), (x0, self.H), (60, 60, 66), 1)
        cv2.line(画布, (0, self.H - 底栏高), (self.W, self.H - 底栏高), (60, 60, 66), 1)
        注3 = 场.get("注视") or (0, 0, 0)
        dt = max(time.time() - self.始t, 1e-6)
        拍 = int(getattr(self.眼, "轮", 0)) if self.眼 is not None else 0
        hz = 拍 / dt
        后端 = getattr(self.眼, "采集后端", "?") if self.眼 is not None else "?"
        帧年龄 = "?"
        try:
            帧年龄 = self.眼.连续.状态().get("帧年龄ms")
        except Exception:
            pass
        项s = []
        项s.append(((x0 + 8, 8), "思考流（脑原生 · 只显外层）", (200, 200, 206), 15))
        # 思考流：只渲染外层（步1 分层）
        思 = 思考行(self.态, 128)
        lineh = 22
        容量 = max(1, (self.H - 底栏高 - 40) // lineh - 4)
        尾 = 思[-容量:]
        基 = self.H - 底栏高 - 40 - len(尾) * lineh
        for i, x in enumerate(尾):
            类, 文 = x["类"], x["文"]
            if len(文) > 26:
                文 = 文[:25] + "…"
            色 = 色表.get(类, (200, 200, 200))
            if i == len(尾) - 1:
                色 = (255, 255, 255)
            项s.append(((x0 + 8, 基 + i * lineh), "[%s] %s" % (类, 文), 色, 15))
        # HUD（底栏）
        项s.append(((10, self.H - 底栏高 + 8),
                    "全屏只读演播｜拍频 %.1f Hz｜采集=%s｜网格 %d×%d｜在场者 %d｜帧年龄 %sms"
                    % (hz, 后端, gxN, gyN, len(视.get("元素s") or []), 帧年龄),
                    (232, 232, 238), 15))
        项s.append(((10, self.H - 底栏高 + 32),
                    "注视=(%d,%d) 停留%d拍｜意外 %.3f｜眼跳 %d"
                    % (int(注3[0]), int(注3[1]), int(注3[2]) if len(注3) > 2 else 0,
                       float(场.get("意外度") or 0.0), int(场.get("眼跳数") or 0)),
                    (192, 192, 198), 15))
        if 思:
            末 = 思[-1]
            文 = "嘟囔：%s" % 末["文"]
            if len(文) > 60:
                文 = 文[:59] + "…"
            项s.append(((10, self.H - 底栏高 + 56), 文, 色表.get(末["类"], (230, 230, 230)), 18))
        项s.append(((self.ww - 260, self.H - 底栏高 + 56),
                    "红=惊讶 黄=注视 绿=在场者   ESC/q 退出", (150, 150, 158), 13))
        self._文(画布, 项s)

    def 画(self) -> np.ndarray:
        """合成一帧 —— **纯只读**：仅 取视场/取视觉/取动作/取意图/取思考 + 眼.最近帧。"""
        画布 = np.full((self.H, self.W, 3), 16, np.uint8)       # RGB 画布
        场 = self.态.取视场()
        视 = self.态.取视觉()
        动 = self.态.取动作()
        意 = self.态.取意图()
        帧 = getattr(self.眼, "最近帧", None) if self.眼 is not None else None
        self._贴底图(画布, 帧)
        self._覆盖(画布, 场, 视, 动, 意)
        self._取帧率()
        self._面板(画布, 场, 视)
        return 画布


# ══════════════════════ 窗口（tkinter 全屏；可建窗=False 单测） ══════════════════════

def _ESC按下() -> bool:
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000)
    except Exception:
        return False


class 演播窗:
    """全屏只读窗口。ESC：tkinter 绑定 ＋ 系统级轮询（不依赖焦点）。
    建窗=False 时不创建 Tk（供判据单测 ESC/退出口径）。"""

    def __init__(self, 渲染器: 演播渲染, 停: threading.Event = None, 每帧s: float = 1 / 15.0,
                 录: str = None, 建窗: bool = True):
        self.渲染 = 渲染器
        self.停 = 停 or threading.Event()
        self.退出因 = None
        self.每帧s = float(每帧s)
        self.root = None
        self.画布 = None
        self._photo = None
        self._写 = None
        if 录:
            self._写 = cv2.VideoWriter(录, cv2.VideoWriter_fourcc(*"mp4v"),
                                       10.0, (渲染器.W, 渲染器.H))
            if not self._写.isOpened():
                self._写 = None
                print("[演播台] 视频写入器打不开（%s），本次不录" % 录)
        if 建窗:
            self._开窗()

    def _开窗(self):
        import tkinter as tk
        r = tk.Tk()
        r.title("AHI 全屏实况演播台（只读）")
        r.configure(bg="#0e0e12")
        r.geometry("%dx%d+0+0" % (self.渲染.W, self.渲染.H))
        try:
            r.attributes("-fullscreen", True)
            r.attributes("-topmost", True)
        except Exception:
            pass
        c = tk.Canvas(r, width=self.渲染.W, height=self.渲染.H,
                      highlightthickness=0, bg="#0e0e12")
        c.pack(fill="both", expand=True)
        r.bind("<Escape>", self.按ESC)
        r.bind("<Key-q>", self.按键q)
        r.bind("<Key-Q>", self.按键q)
        r.protocol("WM_DELETE_WINDOW", lambda: self.置停("关窗"))
        self.root, self.画布 = r, c

    # ── 退出（单测可直接调） ──
    def 按ESC(self, event=None):
        self.置停("ESC")

    def 按键q(self, event=None):
        self.置停("q")

    def 置停(self, 因: str):
        self.退出因 = 因
        self.停.set()

    def _显示(self, 帧):
        from PIL import ImageTk
        self._photo = ImageTk.PhotoImage(Image.fromarray(帧))     # RGB → ImageTk（正确通道序）
        self.画布.delete("all")
        self.画布.create_image(0, 0, anchor="nw", image=self._photo)

    def 循环(self, 秒: float = 0.0):
        t0 = time.time()
        while not self.停.is_set():
            帧 = self.渲染.画()
            if self._写 is not None:
                self._写.write(cv2.cvtColor(帧, cv2.COLOR_RGB2BGR))   # ★ 仅录像落盘前转 BGR
            self._显示(帧)
            try:
                self.root.update()
            except Exception:
                self.置停("关窗")
                break
            if _ESC按下():
                self.置停("ESC(系统级)")
                break
            if 秒 and (time.time() - t0) >= 秒:
                self.置停("到时")
            time.sleep(self.每帧s)
        return self.退出因

    def 关(self):
        if self._写 is not None:
            try:
                self._写.release()
            except Exception:
                pass
            self._写 = None
        if self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass
            self.root = None


# ══════════════════════ 装配 + 主循环 ══════════════════════

def _屏尺寸():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)      # 物理像素对齐（Tk/DDA/度量一致）
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    u = ctypes.windll.user32
    return (int(u.GetSystemMetrics(0)) or 1920, int(u.GetSystemMetrics(1)) or 1080)


def 跑(秒: float = 60.0, 录: str = None, 无头: bool = False, 出图: str = None,
       屏: tuple = None) -> dict:
    屏 = 屏 or _屏尺寸()
    态 = 共存态(种子=3800)
    语 = 语义图(态, None)
    眼 = 眼器(态, None)                                     # 区域=None ⇒ **全屏**（全亮视觉）
    脑 = 脑器(态, 语, None)
    嘴 = 嘴器(态, None, 用语音=False)                        # 声音先不用
    停 = threading.Event()
    线s = []

    def 起(名, fn):
        t = threading.Thread(target=fn, args=(停,), name=名, daemon=True)
        t.start()
        线s.append(t)

    起("眼", 眼.运行)
    起("脑", 脑.循环)
    起("嘴", 嘴.运行)
    起("语义", 语.运行)
    time.sleep(0.6)
    渲染 = 演播渲染(态, 眼, 尺寸=屏)
    备注 = "到时"
    t0 = time.time()
    快照 = None
    try:
        if 无头:
            time.sleep(max(0.0, 秒))                        # 纯跑，不画（供引擎/只读审计）
            快照 = 渲染.画()
            备注 = "无头"
        else:
            窗 = 演播窗(渲染, 停, 录=(录 or None))
            备注 = 窗.循环(秒=秒) or "到时"
            窗.关()
    finally:
        dt = max(time.time() - t0, 1e-6)
        停.set()
        for t in 线s:
            t.join(timeout=2.0)
        if 出图:
            try:
                帧 = 快照 if 快照 is not None else 渲染.画()
                # 中文路径：cv2.imwrite 在 Windows 上对非 ASCII 路径会静默失败 → 走 PIL/np
                if Image is not None:
                    Image.fromarray(帧).save(出图)                  # 帧是 RGB；PIL 正确处理 Unicode 路径
                else:
                    ok, buf = cv2.imencode(".png", cv2.cvtColor(帧, cv2.COLOR_RGB2BGR))
                    if not ok:
                        raise RuntimeError("imencode 失败")
                    buf.tofile(出图)
                print("[演播台] 出图 → %s" % 出图)
            except Exception as ex:
                print("[演播台] 出图失败：%r" % (ex,))
        果 = dict(时长s=round(dt, 1), 备注=备注, 无头=bool(无头),
                  屏=list(屏), 全屏=bool(屏[0] > 0),
                  眼=dict(拍=int(眼.轮), 采集=眼.采集后端,
                          网格="%dx%d" % (眼.前端.格x, 眼.前端.格y),
                          眼跳=眼.前端.眼跳数),
                  脑=dict(思维拍=脑.思维拍, 已碰元素=len(脑._已试)),
                  思考外=len(思考行(态, 512)))
        import json
        with open(os.path.join(HERE, "演播台结果.json"), "w", encoding="utf-8") as f:
            json.dump(果, f, ensure_ascii=False, indent=1, default=str)
        print("[演播台] 收工（%s）：屏=%s 眼拍=%d 采集=%s 眼跳=%d 思考外=%d"
              % (备注, 屏, 眼.轮, 眼.采集后端, 眼.前端.眼跳数, 果["思考外"]))
    return 果


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--秒", type=float, default=60.0, help="0=一直跑（ESC/q 退出）")
    ap.add_argument("--录", type=str, default="")
    ap.add_argument("--无头", action="store_true", help="不开窗（判据/CI；仍跑引擎并出图）")
    ap.add_argument("--出图", type=str, default="", help="把一帧渲染写成 PNG（无头模式常用）")
    a = ap.parse_args()
    跑(秒=a.秒, 录=(a.录 or None), 无头=a.无头, 出图=(a.出图 or None))
    return 0


if __name__ == "__main__":
    sys.exit(main())