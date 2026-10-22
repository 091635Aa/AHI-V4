# -*- coding: utf-8 -*-
"""meeting38/器官_手.py —— AHI 第五路器官：手（360° 自由控制器）

本器官只读写共享工作区，不调用任何其他器官。

━━━ 定位（所有者 2026-10-06）━━━
「手，之所以称为手，是能 360 度无死角的旋转操控鼠标，而不是强制的一个 A 点、B 点的过渡器。」

因此本器官**不再是"给定起终点 → 算一条轨迹"的过渡器**——旧版 `_计划/_段/点s`（步骤表 +
预采样点列，起→过冲→终共线）**已废除**，连"贝塞尔化"也一并否掉（那仍是从 A 到 B 的插值）。
它现在是一台**逐拍速度/方向控制器**，像一只真手握着鼠标：
  · 每拍读一次「意图槽.当前」（落点/句柄/键/滚轮/键入/拖到）与「视觉槽」里该元素的**最新位置**；
  · 解出"这一拍往哪个方向、多快"（有界速度 + 有界加速度 + 到达减速），把光标推进一小步；
  · **能 360° 转向、能中途任意改向、能原地微调、能画圈**；无"起点/终点"概念、**零瞬移**。

━━━ 非 Transformer（与全项目红线一致）━━━
控制律是**确定性数学**：比例减速（远处全速、近处按距离减速）＋ 一阶惯性平滑（限加速度）＋
到达门停靠。全程无神经网络、无 attention / QKV / softmax、无梯度。

━━━ 唯一真相源 ━━━
  读：态.取意图() / 态.取动作() / 态.找元素(句柄) / 态.取视觉() / 态.秒()
  写：态.写动作(...) / 态.写意图(...) / 态.排队说(文本)
手的反馈只写回动作槽，供眼/脑读取——这就是"手执行→屏幕变→眼看见→脑改意图→
手在移动途中改方向"这条依赖链的落点。

原子能力均以"意图→连续控制"形式存在（不是一次性函数）：
  移到(点) · 左键() · 右键() · 中键() · 滚轮(格数) · 拖拽(起点,终点) · 键入(文本)
它们只把意图投喂进工作区，由 运行() 的连续循环展开成逐拍速度并驱动光标。
"""
from __future__ import annotations

import ctypes
import math
import threading
import time

from collections import deque
from ctypes import wintypes

import numpy as np
import pyautogui

# 出屏保护：光标可越界不抛异常，动作之间不额外停顿（时序由本器官自控）
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# 鼠标事件常量（滚轮用 ctypes 直发；pyautogui.scroll 实测无效）
_MOUSEEVENTF_WHEEL = 0x0800

# ── 控制律常量（逐拍速度/方向控制；确定性数学，无学习/无梯度）──
Vmax = 2200.0          # 速度上限 px/s（60Hz 下每拍 ≤ ~35px：有界步长 ⇒ 零瞬移）
减速半径 = 160.0       # 进入此半径内按距离线性减速（平滑靠拢、抑制过冲）
平滑系数 = 0.30        # 一阶惯性（限加速度：转向要几拍，不瞬移、不突变）
到达门 = 2.0           # 距目标 ≤ 此值 → 到达（执行待动作/收势）
到达停顿 = 0.25        # 步7（移动性偏置）：到了**先停一下**再点——像人"看清了再按"（确定性常量）
死区 = 0.4             # 更近自然停住（允许原地微调）
抖动σ = 0.35           # 人手微颤（每拍 ≤ ~1px；有界、可复现=固定种子）
迹长 = 64              # 动作槽「轨迹」= 最近光标位置环（实际路径，非预采样计划）


class 手器:
    """360° 自由控制器：每拍读意图/视觉最新值，解出方向与速度推进光标一小步。
    状态机在动作槽里流转（闲 → 移动中 → 闲）；移动途中随时可改向、可纠偏。"""

    键映射 = {"左": "left", "右": "right", "中": "middle"}

    def __init__(self, 态, 台账=None, 演练: bool = False, 偏移=(0, 0), 种子: int = 3801,
                 世界=None):
        """演练=True：控制律/状态机/反馈全链路照常运转，但**绝不真动鼠标键盘**——
        光标用虚拟坐标模拟（速度照算、进度照推进、动作槽照写）。
        自动化验证一律用演练模式；只有所有者知情时才用实机模式。
        偏移=(x0,y0)：坐标平移（默认 (0,0) 时与既有行为**逐位一致**）。
        种子：人手微颤的确定性种子（C13 可复现）。
        世界：演练动作落地钩子（步5 安全沙盒）。为可调用对象时，演练分支的每次
        键/滚轮/键入/拖按/拖移动 都以事件 dict 回调 世界(事件)——动作在**沙盒世界**
        里产生眼看得见的效果（学习闭环），真机依旧分毫不动。默认 None → 行为不变。"""
        self.态 = 态
        self.台账 = 台账
        self.演练 = bool(演练)
        self.世界 = 世界
        self._偏移 = (int(偏移[0]), int(偏移[1]))

        # Windows 光标/鼠标：光标用 SetCursorPos（连续推进），点击用 pyautogui
        # 【注意】绝不设置 windll.user32.mouse_event.argtypes——windll.user32 是缓存的
        # 单例句柄，改 argtypes 会污染 pyautogui 内部对同一函数的调用（实测点击抛
        # ArgumentError: argument 2 wrong type）。滚轮改用独立 WinDLL 实例。
        self._用户32 = ctypes.windll.user32
        self._自用32 = ctypes.WinDLL("user32")           # 独立加载，滚轮专用，不碰 pyautogui

        self._采样 = 0.016                      # 连续循环节拍（约 60Hz）
        self._抖动 = np.random.default_rng(int(种子))
        self._虚拟光标 = (0, 0)                 # 演练模式的光标（实机模式仅作镜像）

        # ── 控制器状态（仅本线程写）──
        self._速度 = np.zeros(2, dtype=np.float64)   # 当前速度向量 (px/s)
        self._目标 = None                            # 当前落点（屏幕系）
        self._句柄 = None                            # 跟踪中的元素（做中途纠偏）
        self._兜底 = None                            # 元素消失时的兜底落点
        self._待动作 = None                          # 到达后执行：{键|滚动|键入|拖}
        self._拖到 = None                            # 拖拽目的地
        self._拖动中 = False                         # 已按下、正拖
        self._放弃 = False
        self._最后结果 = None                        # 收尾写回动作槽的"反馈=结果"
        self._纠偏计 = 0
        self._意图版本 = -1                           # 已消费的意图版本（防重复执行）
        self._迹 = deque(maxlen=迹长)                 # 实际路径（供演播台画轨迹）
        self._按住 = None                             # 当前按下的鼠标键（拖拽状态；收工保险据此松开）
        # ★备B 安全闭环（meeting40 §5）：可选钩子，**默认关**（None ⇒ 行为与既有逐位一致，C14）
        self.令牌闸 = None                            # 一次性许可令牌闸（不可逆动作 fail-closed）
        self.安全壳 = None                            # 真机/演练差分标签 + 最小 R 适配上报口

    # ══════════════════ 基础：光标 / 微颤 / 坐标平移 ══════════════════

    def _光标(self) -> tuple:
        if self.演练:
            return tuple(self._虚拟光标)
        pt = wintypes.POINT()
        self._用户32.GetCursorPos(ctypes.byref(pt))
        return (int(pt.x), int(pt.y))

    def _移光标(self, 点):
        p = (int(round(float(点[0]))), int(round(float(点[1]))))
        self._虚拟光标 = p
        self._迹.append(p)
        self._报世界(dict(种类="移动", 位置=p, 按住=self._按住))
        if not self.演练:
            self._用户32.SetCursorPos(p[0], p[1])

    def _报世界(self, 事件: dict):
        """演练动作落地钩子（步5）：世界 非 None 时把本拍动作回调给沙盒世界。
        绝不抛异常、绝不影响控制律与状态机（世界的事故不波及手）。"""
        w = self.世界
        if w is None:
            return
        try:
            w(事件)
        except Exception:
            pass

    def _微颤(self) -> np.ndarray:
        """人手微颤（有界、可复现）：给每拍一个 ≤~1px 的随机扰动，轨迹更自然。"""
        return np.array([float(self._抖动.normal(0.0, 抖动σ)),
                         float(self._抖动.normal(0.0, 抖动σ))])

    def _映(self, 点):
        """坐标平移（世界窗口偏移）。默认 (0,0) 时**原样返回**，既有路径逐位不变。"""
        p = tuple(点)
        if self._偏移 == (0, 0):
            return p
        return (p[0] + self._偏移[0], p[1] + self._偏移[1])

    # ══════════════════ 控制律：期望速度 ══════════════════

    @staticmethod
    def _期望速度(方向到目标: np.ndarray, 距离: float) -> np.ndarray:
        """比例减速：远处全速、近处按距离线性减速 → 平滑靠拢、不过冲。"""
        if 距离 <= 死区:
            return np.zeros(2, dtype=np.float64)
        方向 = 方向到目标 / max(距离, 1e-9)
        速率 = Vmax * min(1.0, 距离 / 减速半径)
        return 方向 * 速率

    # ══════════════════ 意图读取（每拍；任意时刻可改向的入口） ══════════════════

    @staticmethod
    def _取待动作(cur: dict):
        if cur.get("拖到") is not None:
            return dict(键=(cur.get("键") or "左"), 拖=True)
        if cur.get("键") is not None:
            return dict(键=cur["键"])
        if cur.get("滚动") is not None:
            return dict(滚动=int(cur["滚动"]))
        if cur.get("键入") is not None:
            return dict(键入=str(cur["键入"]))
        return None

    def _读意图(self) -> bool:
        """读意图槽：版本变了（或首次）且能解出落点 → 更新目标/句柄/待动作。返回是否更新。
        每拍调用——这是"任意时刻任意方向改向"的入口（手不锁死一条 A→B）。"""
        if self._拖动中:
            return False                                  # 拖拽进行中不被新意图劫持
        try:
            意 = self.态.取意图()
        except Exception:
            return False
        cur = 意.get("当前")
        ver = int(意.get("版本", 0))
        if not isinstance(cur, dict) or ver == self._意图版本:
            return False
        目标点 = cur.get("目标点")
        句柄 = cur.get("句柄")
        if 目标点 is None and 句柄 is None:
            return False
        首 = None
        if 目标点 is not None:
            首 = self._映(tuple(目标点))
        elif 句柄 is not None:
            元 = self.态.找元素(句柄)
            if 元 is not None and 元.get("心") is not None:
                首 = self._映(tuple(元["心"]))
            elif cur.get("兜底") is not None:
                首 = self._映(tuple(cur["兜底"]))
        self._意图版本 = ver
        if 首 is None:
            return False
        self._目标 = np.asarray(首, dtype=np.float64)
        self._句柄 = 句柄
        self._兜底 = self._映(tuple(cur["兜底"])) if cur.get("兜底") is not None else None
        self._待动作 = self._取待动作(cur)
        self._拖到 = self._映(tuple(cur["拖到"])) if cur.get("拖到") is not None else None
        self._放弃 = False
        return True

    # ══════════════════ 中途纠偏：本器官存在的理由 ══════════════════

    def _计纠偏(self, 句柄, 新心):
        self._纠偏计 += 1
        self.态.写动作(纠偏次数=self._纠偏计)
        self._记台账("中途改向#%d 句柄=%s → %s"
                    % (self._纠偏计, 句柄,
                       tuple(round(float(v), 1) for v in 新心)))
        if self._纠偏计 == 1:
            self.态.排队说("目标挪了，我中途拐过去。")

    def _纠偏(self):
        """每拍读该元素**最新**位置：还在且心变了 → 飞行中改向；消失 → 兜底或放弃。"""
        if self._句柄 is None:
            return
        元 = self.态.找元素(self._句柄)
        if 元 is None:
            if self._兜底 is not None:
                self._目标 = np.asarray(self._兜底, dtype=np.float64)
                self._句柄 = None
                self._计纠偏(self._句柄, self._兜底)
            else:
                self.态.写动作(反馈="目标消失")
                self._中止()
            return
        心 = 元.get("心")
        if 心 is None:
            框 = 元.get("框")
            if 框 is None:
                return
            心 = (框[0] + 框[2] / 2.0, 框[1] + 框[3] / 2.0)
        心 = self._映(心)
        if np.hypot(float(心[0]) - float(self._目标[0]),
                    float(心[1]) - float(self._目标[1])) < 2.5:
            return                                       # 尚未明显移动，不打扰
        self._目标 = np.asarray(心, dtype=np.float64)
        self._计纠偏(self._句柄, 心)

    # ══════════════════ 推进一拍 ══════════════════

    def _推进(self):
        """推进一拍：顺意图（可改向）→ 纠偏 → 控制律算速度 → 走一小步 → 到达判定。"""
        self._读意图()                                   # 中途任意改向
        self._纠偏()
        if self._放弃:
            return
        if self._目标 is None:
            self._收尾()
            return
        现在 = np.asarray(self._光标(), dtype=np.float64)
        d = self._目标 - 现在
        距离 = float(np.hypot(float(d[0]), float(d[1])))
        期望 = self._期望速度(d, 距离)
        self._速度 = self._速度 + (期望 - self._速度) * 平滑系数      # 一阶惯性（限加速度）
        速率 = float(np.hypot(float(self._速度[0]), float(self._速度[1])))
        if 速率 > Vmax:                                  # 限幅（步长有界 ⇒ 零瞬移）
            self._速度 *= Vmax / 速率
        步 = self._速度 * self._采样 + self._微颤()
        新 = 现在 + 步
        self._移光标(新)
        新距 = float(np.hypot(float(self._目标[0]) - float(新[0]),
                             float(self._目标[1]) - float(新[1])))
        if 新距 <= 到达门:
            self._到达()
        else:
            τ = max(0.0, min(1.0, 1.0 - 新距 / max(距离, 1e-6)))
            self.态.写动作(进度=round(float(τ), 3),
                           光标=(int(round(float(新[0]))), int(round(float(新[1])))),
                           轨迹=[list(p) for p in self._迹])

    def _到达(self):
        """到达落点：拖拽则按下/松开；否则执行待动作（点击/滚轮/键入）；最后收势。"""
        self._速度 = np.zeros(2, dtype=np.float64)
        if self._拖到 is not None and not self._拖动中:
            self._执行键("按下", (self._待动作 or {}).get("键", "左"))
            self._拖动中 = True
            self._目标 = np.asarray(self._拖到, dtype=np.float64)
            self._句柄 = None
            self._兜底 = None
            self._最后结果 = "抓起来了"
            self.态.写动作(进度=0.0, 反馈="抓起来了，我拖过去。")
            return
        if self._拖到 is not None and self._拖动中:
            self._执行键("松开", (self._待动作 or {}).get("键", "左"))
            self._拖动中 = False
            self._最后结果 = "拖拽完成"
            self._收尾()
            return
        待 = self._待动作
        if 待:
            time.sleep(到达停顿)                         # 步7：到了先停一下（看清了再按；拖拽抓放不停）
            if "键" in 待:
                self._执行键("点击", 待["键"])
                self._最后结果 = "点击%s" % 待["键"]
            elif "滚动" in 待:
                self._滚轮(待["滚动"])
                self._最后结果 = "滚轮%d格" % 待["滚动"]
            elif "键入" in 待:
                self._键入(待["键入"])
                self._最后结果 = "键入%d字" % len(待["键入"])
        self._收尾()

    # ══════════════════ ★备B 安全闭环钩子（默认关；meeting40 §5） ══════════════════

    def _令牌放行(self, 动作指纹: str) -> bool:
        """不可逆动作的**一次性许可令牌**闸。**闸未装（None）⇒ 恒放行**（既有行为逐位不变）。

        闸已装 ⇒ 真机分支必须持匹配、未用、未过期、带确认事件的令牌；否则**拒绝执行**。
        异常一律 **fail-closed**（拒），与 `安全事实.py` 的失败纪律一致——不让动作漏出去。
        """
        闸 = getattr(self, "令牌闸", None)
        if 闸 is None:
            return True
        try:
            return bool(闸.放行(str(动作指纹)))
        except Exception:
            return False

    def _报安全壳(self, 事件: dict):
        """把动作落地上报安全壳（真机/演练差分标签）。**壳未装 ⇒ 立即返回**，零副作用。"""
        壳 = getattr(self, "安全壳", None)
        if 壳 is None:
            return
        try:
            d = dict(事件)
            d.setdefault("真机", not self.演练)
            壳.标执行域(d)
        except Exception:
            pass

    # ══════════════════ 执行：键 / 滚轮 / 键入 ══════════════════

    def _执行键(self, 种类: str, 键: str):
        b = self.键映射.get(键, "left")
        双 = (键 == "双")                                 # 步7：双击=运行它（移动性偏置链的末端动作）
        if not self.演练 and not self._令牌放行("键:%s:%s" % (种类, 键)):    # ★备B：不可逆 fail-closed
            self.态.写动作(反馈="拒执行（无有效令牌）：%s%s键" % (种类, 键), 令牌拒=1)
            self._报安全壳(dict(动作="键:%s:%s" % (种类, 键), 真机=True, 拒绝=True))
            return
        if self.演练:                                    # 演练：真机不被触动，状态机照常；
            if 种类 == "按下":                           # 世界 钩子在则动作落地沙盒（步5）
                self._按住 = "left" if 双 else b
            elif 种类 in ("松开", "点击"):
                self._按住 = None
            self._报世界(dict(种类=str(种类), 键=("双" if 双 else b), 位置=self._光标()))
        elif 种类 == "按下":
            pyautogui.mouseDown(button=b)
            self._按住 = b
        elif 种类 == "松开":
            pyautogui.mouseUp(button=b)
            self._按住 = None
        elif 种类 == "点击":
            if 双:                                       # 双击：两击间隔 120ms（双击窗口内）
                pyautogui.click(button="left", clicks=2, interval=0.12)
            else:
                pyautogui.click(button=b)
        self.态.写动作(反馈="%s%s键" % ("演练" if self.演练 else "", 种类))
        self._报安全壳(dict(动作="键:%s:%s" % (种类, 键)))

    def _滚轮(self, 格数: int):
        """滚轮必须用 ctypes 直发 mouse_event（WHEEL=0x0800，dwData=120*格数）。
        用独立 WinDLL 实例调用，避免污染 pyautogui（见 __init__ 注释）。"""
        if not self.演练 and not self._令牌放行("滚轮:%d" % int(格数)):      # ★备B：不可逆 fail-closed
            self.态.写动作(反馈="拒执行（无有效令牌）：滚轮%d格" % int(格数), 令牌拒=1)
            self._报安全壳(dict(动作="滚轮:%d" % int(格数), 真机=True, 拒绝=True))
            return
        if self.演练:
            self._报世界(dict(种类="滚轮", 格数=int(格数), 位置=self._光标()))
            self._报安全壳(dict(动作="滚轮:%d" % int(格数)))
            return
        值 = int(120 * int(格数)) & 0xFFFFFFFF
        self._自用32.mouse_event(_MOUSEEVENTF_WHEEL, 0, 0, 值, None)
        self._报安全壳(dict(动作="滚轮:%d" % int(格数)))

    def _键入(self, 文本: str):
        """pyperclip 放入剪贴板 + Ctrl+V 粘贴（中文/长串安全）。"""
        if not self.演练 and not self._令牌放行("键入:%s" % str(文本)[:32]):  # ★备B：不可逆 fail-closed
            self.态.写动作(反馈="拒执行（无有效令牌）：键入", 令牌拒=1)
            self._报安全壳(dict(动作="键入", 真机=True, 拒绝=True))
            return
        if self.演练:
            self._报世界(dict(种类="键入", 文本=str(文本), 位置=self._光标()))
            self._报安全壳(dict(动作="键入"))
            return
        import pyperclip
        pyperclip.copy(str(文本))
        time.sleep(0.02)
        pyautogui.hotkey("ctrl", "v")
        self._报安全壳(dict(动作="键入"))

    # ══════════════════ 收尾 / 中止 ══════════════════

    def _松手(self):
        if self._按住:
            try:
                if not self.演练:
                    pyautogui.mouseUp(button=self._按住)
            except Exception:
                pass
            self._按住 = None

    def _复位(self):
        self._目标 = None
        self._句柄 = None
        self._兜底 = None
        self._待动作 = None
        self._拖到 = None
        self._拖动中 = False
        self._放弃 = False
        self._速度 = np.zeros(2, dtype=np.float64)

    def _收尾(self):
        """正常走完：复位到"闲"，动作计数+1，写回结果反馈。"""
        self._松手()
        try:
            计 = int(self.态.取动作().get("动作计数", 0)) + 1
        except Exception:
            计 = 1
        反 = self._最后结果 or "完成"
        self.态.写动作(状态="闲", 目标=None, 句柄=None,
                       进度=1.0, 动作计数=计, 反馈=反, 轨迹=[list(p) for p in self._迹])
        self._复位()
        self._最后结果 = None

    def _中止(self):
        """放弃（如目标消失且无兜底）：复位到"闲"，不计动作计数。写「放弃计数」单调计数——
        脑据此**立刻**释放连招位（防死锁）；旧版脑读「反馈」列表转字符串恒假，此计数是修复接口。"""
        self._松手()
        try:
            弃 = int(self.态.取动作().get("放弃计数", 0)) + 1
        except Exception:
            弃 = 1
        self.态.写动作(状态="闲", 目标=None, 句柄=None, 反馈="放弃", 放弃计数=弃)
        self._复位()
        self._最后结果 = None

    # ══════════════════ 主循环 ══════════════════

    def 运行(self, to_stop: threading.Event):
        """约 60Hz 连续循环：每拍读动作槽→（起手/推进）→写回动作槽。
        全程 try/except：异常写台账并复位到"闲"，循环不退出（除非 to_stop）。"""
        self._记台账("手器启动：360° 自由控制器（60Hz 逐拍速度/方向，无 A→B 过渡器）")
        while not to_stop.is_set():
            拍起 = time.perf_counter()
            try:
                self._一拍()
            except Exception as e:
                self._记台账("手器异常已复位：%r" % (e,))
                self._松手()
                try:
                    self.态.写动作(状态="闲", 目标=None, 句柄=None, 反馈="异常复位:%s" % e)
                except Exception:
                    pass
                self._复位()
            用时 = time.perf_counter() - 拍起
            if 用时 < self._采样:
                time.sleep(self._采样 - 用时)
        self._松手()                              # 收工保险：中断在拖拽中也不许卡死鼠标键
        self._记台账("手器停止")

    def _一拍(self):
        """一拍：读动作槽判状态；闲→顺意图起手；移动中→推进（并可改向）。"""
        try:
            状态 = self.态.取动作().get("状态", "闲")
        except Exception:
            状态 = "闲"
        if 状态 != "闲":
            self._推进()
            return
        if self._读意图():
            self.态.写动作(状态="移动中", 目标=tuple(self._目标),
                           句柄=self._句柄, 进度=0.0, 反馈="起手")
            self._最后结果 = None
            self._速度 = np.zeros(2, dtype=np.float64)
            self._推进()

    # ══════════════════ 台账 ══════════════════

    def _记台账(self, 文: str):
        t = self.台账
        if t is None:
            return
        try:
            if callable(t):
                t(文)
            elif hasattr(t, "记") and callable(getattr(t, "记")):
                t.记(文)
            elif hasattr(t, "write"):
                t.write(文 + "\n")
            elif isinstance(t, str):
                with open(t, "a", encoding="utf-8") as f:
                    f.write(文 + "\n")
        except Exception:
            pass

    # ══════════════════ 对外：投喂意图 + 原子能力 ══════════════════
    # 原子能力只把意图写进工作区（由 运行 展开成逐拍速度并驱动执行），
    # 绝不在调用点直接动光标/点击——脑与自检都用同一条通道驱动手。

    def 投喂意图(self, 意图: dict):
        """把意图写进工作区意图槽，供本器官的连续循环取用。
        意图字段：{句柄, 目标点, 键:"左"|"右"|"中"|None,
                  滚动:int|None, 键入:str|None, 拖到:(x,y)|None,
                  兜底:(x,y)|None}"""
        意图 = dict(意图 or {})
        self.态.写意图(当前=意图,
                       强度=float(意图.get("强度", 1.0)),
                       来源=str(意图.get("来源", "投喂")))
        return 意图

    def 移到(self, 点):
        return self.投喂意图(dict(目标点=tuple(点), 来源="手器.移到"))

    def 左键(self):
        return self.投喂意图(dict(目标点=self._光标(), 键="左", 来源="手器.左键"))

    def 右键(self):
        return self.投喂意图(dict(目标点=self._光标(), 键="右", 来源="手器.右键"))

    def 中键(self):
        return self.投喂意图(dict(目标点=self._光标(), 键="中", 来源="手器.中键"))

    def 双击(self, 点):
        """步7：双击=运行它（看→靠近→停→点 的末端动作；意图键="双"）。"""
        return self.投喂意图(dict(目标点=tuple(点), 键="双", 来源="手器.双击"))

    def 滚轮(self, 格数: int):
        return self.投喂意图(dict(目标点=self._光标(), 滚动=int(格数),
                                  来源="手器.滚轮"))

    def 拖拽(self, 起点, 终点):
        return self.投喂意图(dict(目标点=tuple(起点), 拖到=tuple(终点), 键="左",
                                  来源="手器.拖拽"))

    def 键入(self, 文本: str):
        return self.投喂意图(dict(目标点=self._光标(), 键入=str(文本),
                                  来源="手器.键入"))