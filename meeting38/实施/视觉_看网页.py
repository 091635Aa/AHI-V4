# -*- coding: utf-8 -*-
"""meeting38/实施/视觉_看网页.py —— V3 模块三：视觉看网页（输入层重构，纯新增）。

问题（所有者 2026-10-06）：看桌面图标还行，看网页就有问题——网页是高密度结构化信息，
2.1 的视场前端是为桌面图标设计的。

V3 双通道：
  通道A 系统自报结构（UIA）：浏览器有完整可访问性树 → 直接拿 控件名 / 角色 / 位置。
        这是「系统告诉它有什么」，**不是 OCR**，不违反 C9。
  通道B 视觉预测误差：在 UIA 给的控件结构上跑（轻量）视场前端 → 只看
        「UIA 没覆盖的区域」与「UIA 说有但视觉对不上的区域」。
  两者交叉验证。

学习网页路径：先看整体布局（UIA 结构）→ 分「正文区域」「导航区域」→
  正文区提取文本向量存入记忆 M；导航区标记为可点击、不急于点。

红线：C9（**不 OCR**：正文文本来自 UIA 的 Name 属性＝系统自报，不是像素识别）
     ｜C1/C10（无 Transformer / 无外挂模型）｜C13（UIA 遍历按树序确定；视觉块统计无随机）。
"""
from __future__ import annotations

import os
import time

import numpy as np

# ── 控件角色分类（本地化类型串，中英兼容；纯字符串匹配）──
# ★修复（审计 P1-4）：原 `_正文词` 含 "edit"/"编辑"，而**地址栏**的 UIA 角色正是「编辑」
#   → 正文预览以「地址和搜索栏」开头。现把 "edit"/"编辑" 从正文词**移除**（它们是输入框，
#   不是正文），并对浏览器外壳（地址栏/工具栏/标签页/窗口标题栏）单独显式屏蔽。
_正文词 = ("text", "文档", "document", "static", "paragraph", "文本",
          "窗格", "pane", "标题", "heading", "静态", "段落", "list", "列表")
_导航词 = ("hyperlink", "超链接", "tab", "选项卡", "menu", "菜单", "button", "按钮",
          "link", "链接", "标记")
# 浏览器外壳（地址栏/搜索栏/工具栏/标签/标题栏/收藏栏）——绝不入正文
_外壳词 = ("地址和搜索栏", "地址栏", "搜索栏", "address and search bar", "address bar",
          "omnibox", "工具栏", "toolbar", "标签页", "tab strip", "选项卡条",
          "标题栏", "title bar", "收藏夹", "favorites", "书签栏", "bookmarks bar",
          "chrome legacy window", "窗口控件", "系统菜单", "最小化", "最大化", "关闭")


def _在外壳(名: str, 角色: str) -> bool:
    """是否浏览器外壳控件（地址栏/工具栏/标签页…）——这类不得计入正文。"""
    a = str(名 or "").lower()
    b = str(角色 or "").lower()
    return any(w in a or w in b for w in _外壳词)


_壳 = ("",)
_零宽 = ("\u200b", "\u200c", "\u200d", "\ufeff", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e")


def _净(s) -> str:
    """剥零宽/方向控制字符（浏览器标题常带 `\\u200b`，会污染标题匹配与 GBK 控制台输出）。"""
    t = str(s or "")
    for ch in _零宽:
        t = t.replace(ch, "")
    return t


def _框归一(rect):
    """UIA RECT → (l,t,w,h)；失败返回 None。"""
    try:
        if hasattr(rect, "left"):
            l, t, r, b = rect.left, rect.top, rect.right, rect.bottom
        else:
            l, t, r, b = rect[0], rect[1], rect[2], rect[3]
        return (int(l), int(t), max(0, int(r - l)), max(0, int(b - t)))
    except Exception:
        return None


class 网页眼:
    """双通道网页视觉：UIA 结构（主）＋ 视觉块交叉验证（辅）。"""

    def __init__(self, 种子: int = 3800, 编码器=None, 视场=None, 记忆=None,
                 格=(16, 9)):
        self.种子 = int(种子)
        self.编码 = 编码器
        self.视场 = 视场
        self.记忆 = 记忆
        self.格x, self.格y = int(格[0]), int(格[1])
        self._uia = None
        self._uia坏 = 0
        self._uia停 = False
        self.统计 = dict(看=0, 控件=0, uia失败=0)

    # ══════════ 通道A：UIA 结构 ══════════
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

    def 前景窗口(self) -> int:
        try:
            import win32gui
            return int(win32gui.GetForegroundWindow())
        except Exception:
            return 0

    def 窗口标题(self) -> str:
        try:
            import win32gui
            return _净(win32gui.GetWindowText(self.前景窗口()) or "")
        except Exception:
            return ""

    def 取控件(self, 上限: int = 600, 深: int = 26) -> list:
        """遍历前景窗口 UIA 树（RawView 前序，确定性）→ [{名,角色,框,可点,层}]。"""
        if not self._起uia():
            return []
        hwnd = self.前景窗口()
        if not hwnd:
            return []
        出 = []
        try:
            root = self._uia.ElementFromHandle(hwnd)
            walker = self._uia.RawViewWalker
        except Exception:
            self._uia坏 += 1
            return []
        # 迭代式前序（栈），带深度限制与节点上限——确定性、有界
        栈 = [(root, 0)]
        while 栈 and len(出) < 上限:
            el, d = 栈.pop()
            if el is None:
                continue
            try:
                名 = _净(el.CurrentName or "")
            except Exception:
                名 = ""
            try:
                角色 = str(el.CurrentLocalizedControlType or "")
            except Exception:
                角色 = ""
            try:
                框 = _框归一(el.CurrentBoundingRectangle)
            except Exception:
                框 = None
            try:
                可点 = bool(el.CurrentIsEnabled) and not bool(el.CurrentIsOffscreen)
            except Exception:
                可点 = True
            if 框 and 框[2] > 0 and 框[3] > 0:
                出.append(dict(名=名, 角色=角色, 框=框, 可点=可点, 层=d))
            if d < 深:
                try:
                    ch = walker.GetFirstChildElement(el)
                except Exception:
                    ch = None
                兄s = []
                while ch is not None and len(栈) + len(出) < 上限 + 深 * 4:
                    兄s.append(ch)
                    try:
                        ch = walker.GetNextSiblingElement(ch)
                    except Exception:
                        break
                # 逆序入栈 ⇒ 出栈顺序=原序（前序确定性）
                for c in reversed(兄s):
                    栈.append((c, d + 1))
        self.统计["控件"] = len(出)
        return 出

    # ══════════ 截图 + 通道B：视觉块 ══════════
    def 截图(self):
        try:
            import mss
            with mss.mss() as sct:
                mon = sct.monitors[1]
                img = np.asarray(sct.grab(mon))[:, :, :3]      # BGRA→BGR
            return img
        except Exception:
            return None

    def 视觉块(self, 帧) -> list:
        """轻量视觉通道：16×9 网格，逐格做光适应（减均值/除标准差）取结构能量；
        超阈格=「眼看见的结构块」。纯 numpy、确定性（与视场前端粗块同律）。
        若挂了真视场前端，则改走其「在场者」。返回 [{框,能量}]。"""
        if 帧 is None:
            return []
        H, W = int(帧.shape[0]), int(帧.shape[1])
        bw, bh = W // self.格x, H // self.格y
        if bw <= 0 or bh <= 0:
            return []
        if self.视场 is not None:
            try:
                r = self.视场.一拍(帧, 屏幕=(W, H))
                在 = self.视场.在场者(r.get("误差场"))
                out = []
                for e in 在:
                    框 = e.get("框") or e.get("点")
                    if 框:
                        out.append(dict(框=tuple(int(x) for x in 框[:4]) if len(框) >= 4
                                        else (int(框[0]), int(框[1]), bw, bh), 能量=1.0))
                if out:
                    return out
            except Exception:
                pass
        能 = np.zeros((self.格y, self.格x), np.float32)
        for gy in range(self.格y):
            for gx in range(self.格x):
                blk = 帧[gy * bh:(gy + 1) * bh, gx * bw:(gx + 1) * bw, :].astype(np.float32)
                if blk.size == 0:
                    continue
                局 = blk - blk.mean()
                能[gy, gx] = float(np.sqrt((局 ** 2).mean()))
        阈 = float(能.mean() + 1.0 * 能.std())
        出 = []
        for gy in range(self.格y):
            for gx in range(self.格x):
                if 能[gy, gx] >= 阈 and 能[gy, gx] > 1e-6:
                    出.append(dict(框=(gx * bw, gy * bh, bw, bh), 能量=round(float(能[gy, gx]), 2)))
        return 出

    # ══════════ 交叉验证 ══════════
    @staticmethod
    def _心(框):
        return (框[0] + 框[2] / 2.0, 框[1] + 框[3] / 2.0)

    def 交叉验证(self, 控件s, 视觉块s) -> dict:
        """UIA 未覆盖区＝视觉看见但无 UIA 控件盖住；不一致区＝UIA 说有但视觉无结构。"""
        未覆盖, 不一致 = [], []
        for b in 视觉块s:
            cx, cy = self._心(b["框"])
            盖 = any(x["框"][0] <= cx <= x["框"][0] + x["框"][2]
                     and x["框"][1] <= cy <= x["框"][1] + x["框"][3] for x in 控件s)
            if not 盖:
                未覆盖.append(b)
        for c in 控件s:
            if c["框"][2] * c["框"][3] < 20 * 20:
                continue
            cx, cy = self._心(c["框"])
            有 = any(b["框"][0] <= cx <= b["框"][0] + b["框"][2]
                     and b["框"][1] <= cy <= b["框"][1] + b["框"][3] for b in 视觉块s)
            if not 有:
                不一致.append(c)
        return dict(未覆盖区=未覆盖, 不一致区=不一致,
                    未覆盖数=len(未覆盖), 不一致数=len(不一致))

    # ══════════ 布局：正文区 / 导航区 ══════════
    @staticmethod
    def _类(角色: str, 词s) -> bool:
        r = str(角色 or "").lower()
        return any(w in r for w in 词s)

    def 布局(self, 控件s) -> dict:
        """分正文区/导航区。

        ★修复（审计 P1-4）：**先剔浏览器外壳**（地址栏/工具栏/标签页/标题栏）——
        它们此前被当正文，导致「正文预览以『地址和搜索栏』开头」。
        外壳判定 = 名称/角色命中 `_外壳词`，**或** 位于窗口顶部外壳带内（且非文档/文本节点）。
        """
        正, 导 = [], []
        壳 = []
        # 窗口顶带：浏览器标题栏+标签栏（实测 ~48px；地址栏在下方工具带）。取保守值 90px，
        # 且**绝不**剔除导航控件（页内链接常在顶部）——首轮实测过宽（150px）会把导航误剔。
        y项s = [c["框"][1] for c in 控件s if c.get("框")]
        顶 = min(y项s) if y项s else 0
        壳带 = 顶 + 90
        for c in 控件s:
            名, 角, 框 = c.get("名", ""), c.get("角色", ""), c.get("框")
            if _在外壳(名, 角):
                壳.append(c)
                continue
            是文档 = self._类(角, ("文档", "document", "文本", "text"))
            是导航 = self._类(角, _导航词)
            if 框 and 框[1] < 壳带 and not 是文档 and not 是导航:
                壳.append(c)                       # 顶带内的非文档/非导航控件 = 外壳
                continue
            if not 名 and not self._类(角, _正文词):
                continue
            if 是导航:
                导.append(c)
            elif self._类(角, _正文词):
                正.append(c)
        def _围(xs):
            if not xs:
                return None
            l = min(x["框"][0] for x in xs); t = min(x["框"][1] for x in xs)
            r = max(x["框"][0] + x["框"][2] for x in xs); b = max(x["框"][1] + x["框"][3] for x in xs)
            return (l, t, r - l, b - t)
        正文 = "".join((c["名"] + "\n") for c in 正 if c["名"])
        return dict(正文区=_围(正), 导航区=_围(导),
                    正文控件=正, 导航控件=导, 外壳控件=壳,
                    正文文本=正文.strip(), 正文段数=len([c for c in 正 if c["名"]]),
                    外壳剔除数=len(壳))

    # ══════════ 存正文 → 记忆 M ══════════
    def 存正文(self, 正文: str, 来源: str = "网页") -> dict:
        if not 正文:
            return dict(存=False, 原因="正文为空")
        向量 = None
        if self.编码 is not None:
            try:
                向量 = self.编码.文本码(正文[:512])
            except Exception:
                向量 = None
        rid = None
        if self.记忆 is not None:
            try:
                # ★把语义向量一并入库：正文才可被「查经历」真语义检索到（见 容量桥 文件头）
                rid = self.记忆.记经历(拍=0, 类型="网页正文", 文本=正文[:400], 来源=来源,
                                       向量=向量)
            except Exception:
                rid = None
        return dict(存=bool(向量 is not None or rid is not None), 向量=向量, 记录=rid,
                    字数=len(正文))

    # ══════════ 主入口：看 ══════════
    def 看(self, 存记忆: bool = True) -> dict:
        """看一屏网页：UIA 结构 → 视觉块 → 交叉验证 → 布局 → 正文入 M。"""
        t0 = time.time()
        self.统计["看"] += 1
        控件s = self.取控件()
        帧 = self.截图()
        视觉块s = self.视觉块(帧) if 帧 is not None else []
        交 = self.交叉验证(控件s, 视觉块s)
        布 = self.布局(控件s)
        存 = self.存正文(布["正文文本"], 来源="网页") if (存记忆 and 布["正文文本"]) else None
        return dict(
            t=round(time.time() - t0, 3),
            窗口=dict(句柄=self.前景窗口(), 标题=self.窗口标题()),
            控件数=len(控件s), UIA可用=bool(self._uia is not None and not self._uia停),
            控件样本=[dict(名=c["名"][:30], 角色=c["角色"], 框=c["框"], 可点=c["可点"])
                    for c in 控件s[:24]],
            视觉通道=dict(块数=len(视觉块s), 有帧=bool(帧 is not None)),
            交叉验证=dict(未覆盖数=交["未覆盖数"], 不一致数=交["不一致数"],
                        未覆盖样本=[b["框"] for b in 交["未覆盖区"][:6]],
                        不一致样本=[c["名"][:20] for c in 交["不一致区"][:6]]),
            布局=dict(正文区=布["正文区"], 导航区=布["导航区"],
                    正文段数=布["正文段数"], 导航数=len(布["导航控件"]),
                    外壳剔除数=布.get("外壳剔除数", 0),
                    正文预览=布["正文文本"][:2000]),
            存入记忆=(dict(存=存["存"], 字数=存["字数"], 记录=存["记录"])
                    if isinstance(存, dict) else None),
        )


def 打开网页(url: str, 用edge: bool = False, 参数=()) -> bool:
    """打开网页/本地文件。用edge=True 走 msedge/chrome，并默认带
    `--force-renderer-accessibility`（强制 Chromium 建可访问性树 → UIA 能读到网页正文）。"""
    try:
        if 用edge or str(url).lower().endswith((".html", ".htm")):
            旗 = ["--force-renderer-accessibility", "--disable-features=msEdgeSidebarV2"]
            旗 += [str(x) for x in 参数]
            for exe in ("msedge", "chrome"):
                try:
                    import subprocess
                    subprocess.Popen(["cmd", "/c", "start", "", exe] + 旗 + [url], shell=False)
                    return True
                except Exception:
                    continue
        import webbrowser
        return bool(webbrowser.open(url))
    except Exception:
        return False