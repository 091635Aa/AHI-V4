# -*- coding: utf-8 -*-
"""meeting38/实施/沙盒.py —— 步5 安全沙盒：确定性模拟世界（纯 numpy 渲染，零 GUI、零真机）

━━━ 定位（所有者 2026-10-06「教会用电脑」）━━━
给引擎一个**能对动作作答的世界**：演练手的每个动作（左/右/中键、滚轮、键入、拖拽）
在本世界里产生**眼看得见**的确定效果——眼看见 → 脑归因 → 语义/我 入库（学习闭环）。
不写死任何任务流程：脑出什么招由它自己的探索律决定，世界只负责**如实回应**。

━━━ 安全 ━━━
· 本件**纯 numpy**：无 tkinter、无 pyautogui、无 win32、无截屏——世界只存在于内存里；
· 全链（眼帧源注入 + 演练手 + 本世界）**分毫不动真机**（真鼠标/键盘/真屏）；
· 确定性（C13）：布局由 种子 决定；同样的动作序列 → 同样的世界状态 → 同样的帧。

━━━ 世界规则（对引擎而言是"物理"，不是脚本）━━━
· 按钮 ×3    ：左键点击 → 白色进度条增长（有功能的控件=大变）；
· 文件图标 ×4：左键=选中高亮切换；中键=高亮切换；右键=弹出菜单（打开/重命名/删除）；
               菜单项左键：打开→开一扇"窗"；重命名→图标加标记；删除→图标移入回收区；
· 输入框     ：先点击聚焦、再键入才回显（两步因果——键入在别处=无感，诚实的学习信号）；
· 滚动列表   ：滚轮 ±格 → 列表滚动（上=−1 / 下=+1，有界）；
· 拖拽       ：左键按下图标→拖动中图标跟手→松开（位移>6px 记拖拽，否则按左键点击处理）；
· 点击菜单外 → 菜单关闭（和真实菜单一样的"物理"）。

━━━ 唯一真相源 ━━━
  读：眼 经 帧源=帧() 看世界；判据/主程序 经 统计() 读台账。
  写：手 经 世界=应用(事件) 改世界。线程安全：一把 _锁 护状态与渲染。
"""
from __future__ import annotations

import threading

import numpy as np

屏宽, 屏高 = 1920, 1080

背景色 = (38, 42, 50)
面板色 = (52, 58, 68)
回收色 = (30, 32, 38)

按钮色 = [(196, 84, 84), (92, 168, 96), (86, 122, 198)]
图标色 = [(208, 176, 84), (156, 108, 200), (96, 188, 188), (204, 132, 88)]
菜单色 = [(120, 190, 120), (210, 200, 110), (210, 110, 110)]   # 打开/重命名/删除
窗口色 = (70, 96, 128)
输入框色 = (232, 232, 226)
光标色 = (250, 250, 250)
高亮色 = (250, 220, 60)

_菜单项名 = ("打开", "重命名", "删除")


class 沙盒世界:
    """确定性沙盒世界：应用(事件) 收演练动作；帧() 出 1920×1080 RGB 帧；统计() 出台账。"""

    def __init__(self, 种子: int = 3805):
        self.种子 = int(种子)
        self._锁 = threading.Lock()
        self._背景 = np.full((屏高, 屏宽, 3), 背景色, dtype=np.uint8)
        self._重置(种子)                                   # 先定布局（回收区/面板坐标）
        self._画面板带()                                   # 再画静态件（只画一次）

    # ══════════════════ 布局与重置 ══════════════════
    # 【重要】眼的原生元素分辨率 = 16×9 视场格（120px/格）：元素由误差场 BFS 涌现，
    # 框/心落在格边界上。控件必须 ≥1 格且格对齐，涌现团的心才≈控件的心（瞄得中）。
    # 这是"世界按眼的分辨率造"，不是判据放水——脑仍要自己选目标、自己发现因果。
    _格 = 120
    _容差 = 24                                            # 命中容差（吸收涌现团心的残余偏差）

    def _重置(self, 种子: int):
        rng = np.random.default_rng(int(种子))
        G = self._格
        # 按钮 ×3：240×120（2×1 格），顶部一排（避开脑的角区回避：右上 x>1700 且 y<80）
        self.按钮s = [dict(框=(240 + i * 360, 120, 2 * G, G),
                           色=按钮色[i], 计=0) for i in range(3)]
        # 文件图标 ×4：120×120（1×1 格），2×2 网格（无抖动——格对齐优先，C13 更严）
        self.图标s = []
        for i in range(4):
            gx, gy = 240 + (i % 2) * 360, 360 + (i // 2) * 240
            self.图标s.append(dict(框=(gx, gy, G, G),
                                  色=图标色[i], 在=True, 选中=False,
                                  改名标记=False, 窗色=窗口色, 纹样=i))
        # 滚动列表：右侧面板（12 条彩色横条，滚轮滚动）
        self.列表框 = (1480, 150, 280, 640)
        self.列表条数 = 12
        self.滚动偏移 = 0
        # 输入框：360×120（3×1 格），中部（点击聚焦 → 键入回显）
        self.输入框 = (1080, 360, 3 * G, G)
        self.输入聚焦 = False
        self.输入文本长 = 0
        # 菜单（右键弹出的上下文菜单；一次只开一个）
        self.菜单 = None                                    # None | dict(图标=idx, 项s=[框...])
        # 窗口（「打开」产物；左键其右上红块关闭）
        self.窗口s = []                                     # [dict(框=(x,y,w,h), 色=(r,g,b))]
        # 回收区：底部横条（被删除的图标排进来）
        self.回收框 = (80, 970, 1300, 80)
        self.回收件s = []                                   # [(x, 色)]
        # 拖拽进行中
        self._拖中 = None                                   # 图标 idx | None
        self._按下点 = None
        # 虚拟光标（世界里的十字；随演练手报告移动）
        self.光标位 = (960, 540)
        # 台账（判据读这里）
        self.统计_ = dict(左键=0, 右键=0, 中键=0, 滚上=0, 滚下=0, 键入=0,
                          拖拽=0, 菜单开=0, 打开=0, 重命名=0, 删除=0, 移动=0,
                          中按钮=0, 中图标=0, 中输入框=0, 中空处=0,
                          抓住=0, 键入回显=0, 双击=0)

    def 重置(self):
        with self._锁:
            self._画面板带()
            self._重置(self.种子)

    # ══════════════════ 世界对动作的回应 ══════════════════
    def 应用(self, 事件: dict):
        """演练手 → 世界：种类=点击/按下/松开/滚轮/键入/移动。世界的事故不外抛（手不背锅）。"""
        try:
            self._应用(事件)
        except Exception:
            pass

    def _应用(self, 事件: dict):
        种 = str(事件.get("种类", ""))
        p = 事件.get("位置")
        p = (int(p[0]), int(p[1])) if p is not None else tuple(self.光标位)
        with self._锁:
            if 种 == "移动":
                self.光标位 = p
                self.统计_["移动"] += 1
                if self._拖中 is not None and str(事件.get("按住") or "") == "left":
                    self._图标随手(self._拖中, p)
                return
            if 种 == "按下":
                self._按下点 = p
                if str(事件.get("键")) == "left":
                    i = self._图标在(p)
                    if i is not None:
                        self.统计_["抓住"] += 1
                    self._拖中 = i if i is not None else None
                return
            if 种 == "松开":
                d = 0.0
                if self._按下点 is not None:
                    d = float(np.hypot(p[0] - self._按下点[0], p[1] - self._按下点[1]))
                拖中, self._拖中, self._按下点 = self._拖中, None, None
                if 拖中 is not None and d > 6.0:            # 真拖拽：落到松开点
                    self._图标随手(拖中, p)
                    self.统计_["拖拽"] += 1
                elif 拖中 is not None:                      # 原地松开 ≈ 左键点击
                    self._点左(p)
                return
            if 种 == "点击":
                键 = str(事件.get("键", "left"))
                if 键 == "left":
                    self._点左(p)
                elif 键 == "right":
                    self._右键(p)
                elif 键 == "双":                            # 步7：双击图标=运行（桌面物理）
                    self.统计_["双击"] += 1
                    i = self._图标在(p)
                    if i is not None:
                        self.菜单 = None
                        self._菜单动作(i, 0)                 # 同菜单「打开」
                    return
                elif 键 == "middle":
                    self.统计_["中键"] += 1
                    i = self._图标在(p)
                    if i is not None:
                        self.图标s[i]["选中"] = not self.图标s[i]["选中"]
                return
            if 种 == "滚轮":
                格 = int(事件.get("格数", 0))
                if 格 > 0:
                    self.统计_["滚上"] += 1
                    self.滚动偏移 = max(0, self.滚动偏移 - abs(格))
                elif 格 < 0:
                    self.统计_["滚下"] += 1
                    self.滚动偏移 = min(self.列表条数 - 4, self.滚动偏移 + abs(格))
                return
            if 种 == "键入":
                self.统计_["键入"] += 1
                if self.输入聚焦:                           # 两步因果：先聚焦、后回显
                    self.输入文本长 = min(24, self.输入文本长 + max(1, len(str(事件.get("文本", "")))))
                    self.统计_["键入回显"] += 1
                return

    def _点左(self, p):
        self.统计_["左键"] += 1
        T = self._容差
        # 命中诊断（判据/调试用：看落点分布是否对得上网格世界）
        if any(self._在框(b["框"], p, T) for b in self.按钮s):
            self.统计_["中按钮"] += 1
        elif self._图标在(p) is not None:
            self.统计_["中图标"] += 1
        elif self._在框(self.输入框, p, T):
            self.统计_["中输入框"] += 1
        else:
            self.统计_["中空处"] += 1
        # ① 菜单优先：点在菜单项上 → 执行；点在菜单外 → 关菜单（继续走 ②）
        if self.菜单 is not None:
            j = self._菜单项在(p)
            if j is not None:
                self._菜单动作(self.菜单["图标"], j)
                self.菜单 = None
                return
            if not self._在框(self.菜单.get("框", (0, 0, 0, 0)), p):
                self.菜单 = None                            # 点在菜单外 → 关闭，落点照常生效
        # ② 按钮
        for b in self.按钮s:
            if self._在框(b["框"], p, T):
                b["计"] = min(10, b["计"] + 1)
                return
        # ③ 输入框：点击聚焦；点别处失焦
        if self._在框(self.输入框, p, T):
            self.输入聚焦 = True
            return
        self.输入聚焦 = False
        # ④ 窗口关闭块（每窗右上 60×60 红）
        for k, w in enumerate(self.窗口s):
            x, y, ww, wh = w["框"]
            if self._在框((x + ww - 72, y + 6, 60, 60), p, T):
                self.窗口s.pop(k)
                return
        # ⑤ 图标：切换选中高亮
        i = self._图标在(p)
        if i is not None:
            self.图标s[i]["选中"] = not self.图标s[i]["选中"]

    def _右键(self, p):
        self.统计_["右键"] += 1
        i = self._图标在(p)
        if i is None:
            self.统计_["中空处"] += 1                       # 右键落空也计入命中诊断
            self.菜单 = None                                # 空处右键：只关菜单
            return
        if self.菜单 is not None and self.菜单["图标"] == i:
            self.菜单 = None                                # 同图标再右键：关
            return
        x, y, w, h = self.图标s[i]["框"]
        G = self._格
        # 菜单项 2×1 格，贴图标右侧竖排（像真菜单：叠在世界之上，点外面就关）
        项 = [(x + G, y + k * G, 2 * G, G) for k in range(3)]
        self.菜单 = dict(图标=i, 项s=项,
                         框=(x + G - 4, y - 4, 2 * G + 8, 3 * G + 8))
        self.统计_["菜单开"] += 1

    def _菜单动作(self, i, j):
        if j == 0:                                          # 打开 → 开一扇窗（错位排布）
            n = len(self.窗口s)
            self.窗口s.append(dict(框=(1080 + (n % 2) * 480, 600 + (n // 2) * 380, 480, 360),
                                   色=self.图标s[i]["色"]))
            self.统计_["打开"] += 1
        elif j == 1:                                        # 重命名 → 图标加标记（键入聚焦后生效的落点）
            self.图标s[i]["改名标记"] = True
            self.统计_["重命名"] += 1
        elif j == 2:                                        # 删除 → 移入回收区
            self.图标s[i]["在"] = False
            x = self.回收框[0] + 16 + len(self.回收件s) * 90
            self.回收件s.append((x, self.图标s[i]["色"]))
            self.统计_["删除"] += 1

    def _图标随手(self, i, p):
        x, y, w, h = self.图标s[i]["框"]
        nx = int(min(max(p[0] - w / 2.0, 0), 屏宽 - w))
        ny = int(min(max(p[1] - h / 2.0, 0), 屏高 - h))
        self.图标s[i]["框"] = (nx, ny, w, h)

    # ── 命中检测（含容差：吸收涌现团心的残余偏差） ──
    @staticmethod
    def _在框(框, p, 容差=0):
        x, y, w, h = 框
        return (x - 容差 <= p[0] < x + w + 容差) and (y - 容差 <= p[1] < y + h + 容差)

    def _图标在(self, p):
        for i, ic in enumerate(self.图标s):
            if ic["在"] and self._在框(ic["框"], p, self._容差):
                return i
        return None

    def _菜单项在(self, p):
        for j, 框 in enumerate(self.菜单["项s"]):
            if self._在框(框, p):
                return j
        return None

    # ══════════════════ 渲染（确定性；眼 经 帧源 看这里） ══════════════════
    def _画面板带(self):
        self._背景[self.回收框[1]:self.回收框[1] + self.回收框[3],
                   self.回收框[0]:self.回收框[0] + self.回收框[2]] = 回收色
        lx, ly, lw, lh = self.列表框
        self._背景[ly - 14:ly + lh + 14, lx - 14:lx + lw + 14] = 面板色

    def 帧(self, 带光标: bool = False) -> np.ndarray:
        """出当前世界帧（RGB uint8 [1080,1920,3]）。眼线程调用；与 应用 共用一把锁。
        默认**不含虚拟光标**——与真机一致（DDA/mss 桌面复制都不合成鼠标指针），
        否则眼把"会动的白十字"当元素追自己的手（实测自追环：96 次点击全落空）。
        带光标=True 仅供主人出图叠加看，不进眼。"""
        with self._锁:
            img = self._背景.copy()
            for b in self.按钮s:
                x, y, w, h = b["框"]
                self._矩形(img, x, y, w, h, b["色"])
                self._矩形(img, x + 8, y + h - 18, int((w - 16) * b["计"] / 10.0), 10,
                           (255, 255, 255))                 # 点击进度（视觉反馈=有功能）
            for ic in self.图标s:
                if not ic["在"]:
                    continue
                x, y, w, h = ic["框"]
                self._矩形(img, x, y, w, h, ic["色"])
                self._纹样(img, x, y, w, h, ic.get("纹样", 0))
                if ic["选中"]:
                    self._描边(img, x - 4, y - 4, w + 8, h + 8, 高亮色, 4)
                if ic["改名标记"]:
                    self._矩形(img, x + w - 18, y - 10, 16, 10, 高亮色)
            if self.菜单 is not None:
                x, y, w, h = self.菜单["框"]
                self._矩形(img, x, y, w, h, (28, 30, 36))
                for j, 框 in enumerate(self.菜单["项s"]):
                    mx, my, mw, mh = 框
                    self._矩形(img, mx, my, mw, mh, 菜单色[j])
            for w in self.窗口s:
                x, y, ww, wh = w["框"]
                self._矩形(img, x, y, ww, wh, w["色"])
                self._矩形(img, x, y, ww, 40, (46, 56, 70))
                self._矩形(img, x + ww - 72, y + 6, 60, 60, (200, 60, 60))
            ix, iy, iw, ih = self.输入框
            self._矩形(img, ix, iy, iw, ih, 输入框色)
            self._描边(img, ix - 3, iy - 3, iw + 6, ih + 6,
                       高亮色 if self.输入聚焦 else (120, 120, 120), 3)
            if self.输入文本长 > 0:                       # 键入回显（条长∝已收字数）
                self._矩形(img, ix + 8, iy + 12, int((iw - 16) * self.输入文本长 / 24.0),
                           ih - 24, (60, 60, 66))
            lx, ly, lw, lh = self.列表框                  # 滚动列表（裁剪在面板内）
            条高, 条距 = 48, 56
            for k in range(self.列表条数):
                by = ly + (k - self.滚动偏移) * 条距
                if by + 条高 < ly or by > ly + lh:
                    continue
                c = 图标色[k % 4]
                y0, y1 = max(by, ly), min(by + 条高, ly + lh)
                if y1 > y0:
                    img[y0:y1, lx:lx + lw] = c
            for x, c in self.回收件s:                     # 回收区里的遗体
                self._矩形(img, x, self.回收框[1] + 12, 56, 56, c)
            if 带光标:                                    # 主人叠加（不进眼）
                cx, cy = self.光标位
                self._矩形(img, cx - 1, cy - 12, 3, 24, 光标色)
                self._矩形(img, cx - 12, cy - 1, 24, 3, 光标色)
            return img

    @staticmethod
    def _矩形(img, x, y, w, h, 色):
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(img.shape[1], int(x + w)), min(img.shape[0], int(y + h))
        if x1 > x0 and y1 > y0:
            img[y0:y1, x0:x1] = 色

    @staticmethod
    def _纹样(img, x, y, w, h, 种: int):
        """步8：每个图标一个独特白色纹样（W 折线/尖三角/双横杠/菱形）——
        图标身份=形状（所有者："向量团不是绝对红，是形状"）；颜色不再承担身份。"""
        白 = (255, 255, 255)
        x, y, w, h = int(x), int(y), int(w), int(h)
        种 = 种 % 4
        if 种 == 0:                                          # W 折线（三段下折）
            for k in range(4):
                x0 = x + w // 6 + k * (w * 2 // 9)
                y0 = y + h // 4 + (h // 2 if k % 2 == 0 else h // 5)
                沙盒世界._矩形(img, x0, y0, max(3, w // 14), h // 3, 白)
        elif 种 == 1:                                        # 尖三角（堆叠三层）
            for r in range(4):
                rw = w * (4 - r) // 5
                沙盒世界._矩形(img, x + (w - rw) // 2, y + h // 4 + r * (h // 8), rw,
                               max(3, h // 14), 白)
        elif 种 == 2:                                        # 双横杠
            沙盒世界._矩形(img, x + w // 6, y + h // 3, w * 2 // 3, max(3, h // 9), 白)
            沙盒世界._矩形(img, x + w // 6, y + h * 2 // 3, w * 2 // 3, max(3, h // 9), 白)
        else:                                                # 菱形（点阵对角）
            c = 3
            for r in range(-c, c + 1):
                rw = (c - abs(r)) * (w // 8) * 2 + max(3, w // 14)
                沙盒世界._矩形(img, x + w // 2 - rw // 2, y + h // 2 + r * (h // 10),
                               rw, max(3, h // 14), 白)

    @staticmethod
    def _描边(img, x, y, w, h, 色, 厚=3):
        x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
        if 厚 <= 0 or x1 <= x0 or y1 <= y0:
            return
        沙盒世界._矩形(img, x0, y0, x1 - x0, 厚, 色)
        沙盒世界._矩形(img, x0, y1 - 厚, x1 - x0, 厚, 色)
        沙盒世界._矩形(img, x0, y0, 厚, y1 - y0, 色)
        沙盒世界._矩形(img, x1 - 厚, y0, 厚, y1 - y0, 色)

    # ══════════════════ 台账（判据/主程序读） ══════════════════
    def 统计(self) -> dict:
        with self._锁:
            return dict(self.统计_)


# ══════════════════ 沙盒安全事实源（脑安全闸的沙盒侧注入） ══════════════════
class 沙盒安全源:
    """脑安全闸在沙盒里的事实源：窗口标题恒「沙盒世界」（无危词）、控件名恒空——
    安全闸照常运转但永不误拦（沙盒世界本身无害）。替代 真机 安全事实源（判据确定性）。"""

    def 窗口标题(self) -> str:
        return "沙盒世界"

    def 点控件名(self, 点) -> str:
        return ""


def 主():                                                     # 自测：画一帧存盘
    import os
    from PIL import Image
    w = 沙盒世界()
    w.应用(dict(种类="移动", 位置=(900, 500)))
    w.应用(dict(种类="点击", 键="left", 位置=(155, 120)))     # 点按钮1
    w.应用(dict(种类="点击", 键="right", 位置=(150, 350)))    # 开图标1菜单
    w.应用(dict(种类="键入", 文本="测试", 位置=(770, 226)))   # 未聚焦 → 不回显
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "沙盒_快照.png")
    Image.fromarray(w.帧()).save(out)
    print("快照：%s" % out)
    print("统计：%s" % w.统计())


if __name__ == "__main__":
    主()
