# -*- coding: utf-8 -*-
"""meeting38/安全事实.py —— 实机安全闸的「系统自报事实源」（辅助事实源，仅用于安全闸）。

背景（步0台账 §10.3；会议_视觉 00_会议纪要 §6.6-2，专家02 独有发现）：
铁律 C9 禁 OCR 后，`脑._注意` 的「钱包反射」危词表原本依赖 `元素s[*].字`（OCR 产物）——
字恒空后安全闸失效，这是**安全性回归**。裁定配套：危词改由**系统自报事实源**驱动：
  ① 前景窗口标题（win32gui；快、无 COM）
  ② 落点处 UI Automation 控件名（comtypes 直调 UIAutomationCore；无需 uiautomation/pywinauto）

定位红线（不得越界）：本模块只服务安全闸——UIA/窗口标题是**辅助事实源**，不是「看」的本体
（会议否决登记：系统自报不得作为看的主路）。看的主路是 视场前端（P1/P2/P3）。

可用性纪律（不得拖慢七路）：
  · 任何失败 → 返回空串（该次查询上安全闸退化，与旧版 OCR 缺失时同界）；
  · UIA 查询失败/超时累计 ≥3 次 → 本进程**永久停用 UIA**（只留窗口标题这条快通道）；
  · 点查询缓存（4px 量化、TTL 1.0s、上限 512 条）；窗口标题缓存 TTL 0.25s；
  · 量级（接线预检实测）：窗口标题 ~0.1ms；UIA 点查 ~3–40ms（缓存命中 ~0）。
"""
from __future__ import annotations

import time


class 事实源:
    """系统自报事实源：窗口标题（快通道）+ UIA 控件名（点查询，带缓存与熔断）。"""

    def __init__(self):
        self._赢 = ""
        self._赢t = 0.0
        self._uia = None
        self._uia停 = False
        self._uia坏 = 0
        self._缓存: dict = {}
        self._缓存上限 = 512
        self.查询数 = 0
        self.失败数 = 0

    # ── ① 前景窗口标题（win32gui，0.25s 缓存） ──
    def 窗口标题(self) -> str:
        now = time.time()
        if now - self._赢t < 0.25:
            return self._赢
        t = ""
        try:
            import win32gui
            t = win32gui.GetWindowText(win32gui.GetForegroundWindow()) or ""
        except Exception:
            t = ""
        self._赢, self._赢t = t, now
        return t

    # ── ② 落点控件名（comtypes UIA，1.0s 缓存、4px 量化、熔断） ──
    def _起uia(self) -> bool:
        if self._uia停:
            return False
        if self._uia is not None:
            return True
        try:
            import comtypes.client
            mod = comtypes.client.GetModule("UIAutomationCore.dll")
            self._uia = comtypes.client.CreateObject(mod.CUIAutomation)
            return True
        except Exception:
            self._uia停 = True
            return False

    def 点控件名(self, 点) -> str:
        try:
            x, y = int(点[0]), int(点[1])
        except Exception:
            return ""
        key = (x // 4, y // 4)
        now = time.time()
        中 = self._缓存.get(key)
        if 中 is not None and now - 中[0] < 1.0:
            return 中[1]
        if not self._起uia():
            return ""
        nm = ""
        self.查询数 += 1
        t0 = time.perf_counter()
        try:
            from ctypes import wintypes
            el = self._uia.ElementFromPoint(wintypes.POINT(x, y))
            nm = str(el.CurrentName or "")
        except Exception:
            self.失败数 += 1
            self._uia坏 += 1
            if self._uia坏 >= 3:
                self._uia停 = True                       # 熔断：本进程放弃 UIA（快通道仍在）
            nm = ""
        else:
            if time.perf_counter() - t0 > 0.5:           # 单次超慢也计一次坏
                self._uia坏 += 1
                if self._uia坏 >= 3:
                    self._uia停 = True
        if len(self._缓存) > self._缓存上限:
            self._缓存.clear()
        self._缓存[key] = (now, nm)
        return nm

    # ── 供台账/测试 ──
    def 状态(self) -> dict:
        return dict(uia可用=bool(self._uia is not None and not self._uia停),
                    uia停用=bool(self._uia停), 查询数=int(self.查询数),
                    失败数=int(self.失败数), 缓存条数=len(self._缓存))


默认 = None


def 默认源() -> 事实源:
    """进程级单例（懒建，避免 import 即碰 COM/GUI）。"""
    global 默认
    if 默认 is None:
        默认 = 事实源()
    return 默认