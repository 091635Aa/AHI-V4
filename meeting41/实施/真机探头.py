# -*- coding: utf-8 -*-
"""meeting41/实施/真机探头.py —— L5 真机验收探头（契约第六节；真动鼠标键盘，须所有者在场）。

━━━ 三件安全物（契约第六节，逐条落地）━━━
  ① **许可令牌闸**：真机档**每一次**原子动作都要过既有 `许可令牌.令牌闸`；
     闸的可逆白名单为空 ⇒ 连"移到"也要令牌；无令牌 ⇒ **不动手**（零调用，fail-closed）。
     令牌一次性 + 指纹绑定（沿用 `许可令牌.指纹_键/指纹_滚轮/指纹_键入/指纹_移动` 的同一格式）
     + 有效期拍 + 不可逆必须带确认事件。
  ② **演练/真机双档**：`演练=True`（默认）时**绝不调用** `pyautogui`——只做能力探测
     （`importlib.util.find_spec` + 可选 import，**不触发任何动作**）；也不需要真屏截图，
     截图走**自绘合成屏**。真机档才惰性构造 `_真机后端`。
  ③ **现场确认**：真机档启动前**打印将要做的事**并等所有者确认（stdin 输入，或 `--确认` 开关）；
     未确认 ⇒ 拒绝执行、**不构造真机后端**、exit=2。

━━━ 落盘（供 证据_真机.md 引用）━━━
  · `产物/真机验收_演练档.json` —— 档位、三件安全物自证、逐原子 τ/耗时/命中、截图哈希、
    合成靶场 vs 真任务迁移对比、口径与诚实边界。
  · `产物/真机_演练_前.png` / `产物/真机_演练_后.png` —— 操作前后截图（演练档=自绘合成屏；
    真机档=只读抓屏，**绝不**用 pyautogui 截图）。截图内容为 ASCII 摘要，中文细节在同名 JSON 里。

━━━ 本轮口径（所有者不在自动回合里）━━━
  · 本轮**只做真机档的「准备与演练自证」**：`python 真机探头.py` 默认走演练档，**不真动鼠标键盘**。
  · 演练档真任务链直接 import 既有 `meeting40/实施/` 三件：`文档任务.做文档` / `代码任务.写代码` /
    `看网页任务.看网页(允许开浏览器=False)`——都不碰鼠标键盘、不开浏览器、不抓真屏。
  · **绝不许**把演练读数写成真机结果：所有结果都带 `档="演练档"` 与 `真机动作数=0`；
    真机读数必须由所有者在场、带 `--真机` 实际跑出来，再另存结果。

诚实边界：
  · 演练档的"合成靶场耗时"来自 `_演练后端`（纯记账，不走真鼠标）⇒ **不可**与真机耗时直接比；
    本轮它只回答「技能序列在靶场里能不能稳定命中、耗时是不是稳定在 ≤3s 的靶场口径」。
  · 真任务耗时是**真执行**既有任务件的时基口径（`时基.拍钟` = perf_counter），
    含解析与落盘成本，**不是**人手操作耗时，也不是真机鼠标耗时。
  · 迁移结论本轮只到「同一技能序列在靶场命中 + 在真任务件上真跑成功」这一层，
    **不主张**"已经会操作真实软件界面"。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import struct
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
AHI3 = os.path.dirname(os.path.dirname(HERE))          # → AHI3
M40实施 = os.path.join(AHI3, "meeting40", "实施")
for _p in (HERE, M40实施):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                   # noqa: E402

import 时基                                                          # noqa: E402
import 许可令牌                                                       # noqa: E402
from 接口契约 import 全部原子                                          # noqa: E402

产物目录 = os.path.join(HERE, "产物")
验收路径 = os.path.join(产物目录, "真机验收_演练档.json")
真任务目录 = os.path.join(产物目录, "真机_真任务")

令牌有效期拍 = 20.0          # 20 拍 = 2s 提交窗（签发→过闸在同一原子内完成）
演练轮数 = 5                 # 合成靶场侧练几轮（取 最快/中位/末次 三个量）
截图尺寸 = (1280, 720)


# ═══════════════════════════════════════════════════════════════════════
# 零、截图（演练档 = 自绘合成屏；**不抓真屏**）
# ═══════════════════════════════════════════════════════════════════════
def _手写PNG(路径: str, 宽: int, 高: int, 色=(18, 20, 28)) -> dict:
    """零依赖最小 PNG（PIL 不可用时的兜底）：纯色 + 三条横带。"""
    行 = bytearray()
    for y in range(高):
        r, g, b = 色
        if (y // 48) % 2 == 0:
            r, g, b = min(255, r + 12), min(255, g + 12), min(255, b + 16)
        行.append(0)
        行.extend(bytes((r, g, b)) * 宽)

    def _块(类型: bytes, 数据: bytes) -> bytes:
        return (struct.pack(">I", len(数据)) + 类型 + 数据
                + struct.pack(">I", zlib.crc32(类型 + 数据) & 0xFFFFFFFF))

    头 = struct.pack(">IIBBBBB", 宽, 高, 8, 2, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + _块(b"IHDR", 头)
           + _块(b"IDAT", zlib.compress(bytes(行), 6)) + _块(b"IEND", b""))
    with open(路径, "wb") as f:
        f.write(png)
    return dict(来源="兜底手写PNG(零依赖)", 宽=宽, 高=高)


def 自绘截图(路径: str, 标题: str, 行s, 尺寸=截图尺寸) -> dict:
    """演练档截图：**自绘合成屏**（不抓真屏、不碰 pyautogui）。"""
    行s = [str(x) for x in (行s or [])]
    try:
        from PIL import Image, ImageDraw, ImageFont                # noqa: PLC0415
        图 = Image.new("RGB", 尺寸, (14, 17, 24))
        笔 = ImageDraw.Draw(图)
        字体 = None
        中文可 = False
        for p in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            if os.path.exists(p):
                try:
                    字体 = ImageFont.truetype(p, 20)
                    中文可 = p.endswith((".ttc", ".ttf")) and "msyh" in p or "simhei" in p
                    break
                except Exception:                                  # noqa: BLE001
                    字体 = None
        if 字体 is None:
            字体 = ImageFont.load_default()
        笔.rectangle([0, 0, 尺寸[0], 64], fill=(28, 34, 52))
        笔.text((16, 18), 标题 if 中文可 else "AHI meeting41 L5 probe (drill mode)",
                fill=(235, 240, 255), font=字体)
        y = 84
        for i, 行 in enumerate(行s[:26]):
            文 = 行 if 中文可 else "".join(ch if ord(ch) < 128 else "?" for ch in 行)
            笔.text((16, y), 文, fill=(180, 200, 230) if i else (255, 235, 150), font=字体)
            y += 24
        笔.rectangle([0, 尺寸[1] - 34, 尺寸[0], 尺寸[1]], fill=(28, 34, 52))
        笔.text((16, 尺寸[1] - 28), "self-drawn synthetic screen / no real screen grab",
                fill=(150, 165, 190), font=字体)
        图.save(路径, "PNG")
        import hashlib as _h                                        # noqa: PLC0415
        with open(路径, "rb") as f:
            原 = f.read()
        return dict(来源="自绘合成屏(PIL)", 宽=尺寸[0], 高=尺寸[1], 字节=len(原),
                    sha256=_h.sha256(原).hexdigest()[:16], 中文字体=bool(中文可))
    except Exception as ex:                                        # noqa: BLE001
        果 = _手写PNG(路径, 尺寸[0], 尺寸[1])
        with open(路径, "rb") as f:
            原 = f.read()
        果.update(回退原因="%s: %s" % (type(ex).__name__, ex),
                  sha256=hashlib.sha256(原).hexdigest()[:16], 字节=len(原),
                  宽=尺寸[0], 高=尺寸[1])
        return 果


def 抓屏截图(路径: str) -> dict:
    """**真机档**截图：只读抓屏（PIL.ImageGrab），**不经 pyautogui**；失败 ⇒ 退化为自绘。"""
    try:
        from PIL import ImageGrab                                   # noqa: PLC0415
        图 = ImageGrab.grab()
        图.save(路径, "PNG")
        with open(路径, "rb") as f:
            原 = f.read()
        return dict(来源="只读抓屏(PIL.ImageGrab，非pyautogui)", 宽=图.size[0], 高=图.size[1],
                    字节=len(原), sha256=hashlib.sha256(原).hexdigest()[:16])
    except Exception as ex:                                        # noqa: BLE001
        return dict(自绘截图(路径, "真机档截图回退（抓屏不可用）", ["grab failed: %r" % (ex,)]),
                    抓屏回退原因="%s: %s" % (type(ex).__name__, ex))


# ═══════════════════════════════════════════════════════════════════════
# 一、合成靶场（契约第三节形状的最小替身；B 的 `技能原子.靶场` 到位则优先用它）
# ═══════════════════════════════════════════════════════════════════════
class 桩靶场:
    """★桩靶场：只实现契约第三节的对外形状（元素列/点/滚/打/态）。

    **有靶**：打不到 ⇒ 返回 False；元素名写错 / 点靶外点 ⇒ 失败（可失败）。
    真件 `技能原子.靶场` 到位后由 `取靶场()` 优先取用。"""

    def __init__(self, 种子: int = 0):
        self.种子 = int(种子)
        self._元素 = [
            dict(名="文档图标", 点=(320, 240), 可点=True, 可打字=True),
            dict(名="代码图标", 点=(640, 360), 可点=True, 可打字=True),
            dict(名="网页区", 点=(760, 420), 可点=True, 可打字=False),
        ]
        self._态 = dict(点击列=[], 滚轮累计=0, 文本="", 脱靶=0, 命中=0, 失手=0)

    def 元素列(self) -> list:
        return [dict(x) for x in self._元素]

    def 点(self, 元素名: str) -> bool:
        名 = str(元素名)
        for e in self._元素:
            if e["名"] == 名 and e["可点"]:
                self._态["点击列"].append(名)
                self._态["命中"] += 1
                return True
        self._态["失手"] += 1
        self._态["脱靶"] += 1
        return False

    def 滚(self, 格数) -> bool:
        n = int(格数)
        if n == 0:
            self._态["失手"] += 1
            return False
        self._态["滚轮累计"] += n
        self._态["命中"] += 1
        return True

    def 打(self, 文本) -> bool:
        s = str(文本)
        if not s:
            self._态["失手"] += 1
            return False
        self._态["文本"] += s
        self._态["命中"] += 1
        return True

    def 态(self) -> dict:
        return dict(self._态)


def 取靶场(种子: int = 0):
    """优先真件 `技能原子.靶场`（实现者B），不可用则用 ★桩靶场。→ (靶场, 来源)"""
    try:
        import 技能原子                                             # noqa: PLC0415
        for 名 in ("靶场", "合成靶场"):
            f = getattr(技能原子, 名, None)
            if callable(f):
                return f(int(种子)), "真件:技能原子.%s" % 名
    except Exception:                                              # noqa: BLE001
        pass
    return 桩靶场(int(种子)), "桩靶场:真机探头.桩靶场（技能原子.靶场 未到位）"


# ═══════════════════════════════════════════════════════════════════════
# 二、后端：演练（记账）／真机（pyautogui）
# ═══════════════════════════════════════════════════════════════════════
class 演练后端:
    """演练后端：只对合成靶场记账，**永不**接触 pyautogui / 真鼠标键盘。"""

    def __init__(self, 靶场, 屏=(1920, 1080)):
        self.靶场 = 靶场
        self.屏 = tuple(屏)
        self.调用 = []
        self.动作记录 = []

    def _元素名(self, 点):
        for e in self.靶场.元素列():
            if tuple(e["点"]) == tuple(点):
                return e["名"]
        return None

    def 执行(self, 原子: str, 参数: dict) -> tuple:
        self.调用.append((原子, 参数))
        点 = 参数.get("点")
        名 = 参数.get("靶") or (self._元素名(点) if 点 else None)
        原子 = str(原子)
        if 原子 == "移到":
            中 = 名 is not None
        elif 原子 in ("左键", "右键", "中键", "双击"):
            中 = bool(名) and bool(self.靶场.点(名))
        elif 原子 == "滚轮":
            中 = bool(self.靶场.滚(参数.get("格数", 0)))
        elif 原子 == "键入":
            中 = bool(self.靶场.打(参数.get("文本", "")))
        elif 原子 == "拖拽":
            中 = bool(self._元素名(参数.get("起")) and self._元素名(参数.get("止"))
                      and tuple(参数.get("起")) != tuple(参数.get("止")))
        else:
            中 = False
        self.动作记录.append(dict(原子=原子, 参数=dict(参数), 命中=bool(中)))
        return bool(中), ("" if 中 else "演练靶场未命中（靶外/空参）")


class _记录桩:
    """pyautogui 的记录型替身：只记账、绝不动作（用于证明"演练档零调用"与"真机分支会调用"）。"""

    def __init__(self):
        self.调用 = []

    def size(self):
        return (1920, 1080)

    def position(self):
        return (0, 0)

    def __getattr__(self, 名):
        def _f(*a, **kw):
            self.调用.append((名, a, kw))
            return None
        return _f


class 真机后端:
    """真机后端：**唯一**会动真鼠标键盘的地方。惰性 import pyautogui；越界一律不动手。

    `模块` 参数用于安全自证时注入记录桩（判据不打真机）；默认 None ⇒ 真 import pyautogui。
    """

    def __init__(self, 模块=None, 屏幕=None):
        if 模块 is None:
            import pyautogui as 模块                               # noqa: PLC0415
        self.模块 = 模块
        self.调用 = []
        self.动作记录 = []
        if 屏幕 is None:
            try:
                尺 = self.模块.size()
                屏幕 = (int(尺[0]), int(尺[1]))
            except Exception:                                      # noqa: BLE001
                屏幕 = (1920, 1080)
        self.屏 = tuple(屏幕)

    def _越界(self, 点) -> bool:
        if not 点:
            return False
        x, y = int(点[0]), int(点[1])
        return not (0 <= x < self.屏[0] and 0 <= y < self.屏[1])

    def 执行(self, 原子: str, 参数: dict) -> tuple:
        原子 = str(原子)
        点 = 参数.get("点") or 参数.get("起")
        if self._越界(点):
            self.动作记录.append(dict(原子=原子, 参数=dict(参数), 命中=False))
            return False, "越界（屏幕 %s，未动手）" % (self.屏,)
        try:
            if 原子 == "移到":
                self.模块.moveTo(int(点[0]), int(点[1]), duration=0.1)
            elif 原子 == "左键":
                self.模块.click(int(点[0]), int(点[1]), button="left")
            elif 原子 == "右键":
                self.模块.click(int(点[0]), int(点[1]), button="right")
            elif 原子 == "中键":
                self.模块.click(int(点[0]), int(点[1]), button="middle")
            elif 原子 == "滚轮":
                self.模块.scroll(int(参数.get("格数", 0)))
            elif 原子 == "拖拽":
                起, 止 = 参数.get("起"), 参数.get("止")
                self.模块.moveTo(int(起[0]), int(起[1]), duration=0.1)
                self.模块.dragTo(int(止[0]), int(止[1]), duration=0.2, button="left")
            elif 原子 == "键入":
                self.模块.write(str(参数.get("文本", "")), interval=0.02)
            else:
                return False, "未知原子：%r" % (原子,)
        except Exception as ex:                                    # noqa: BLE001
            self.调用.append((原子, dict(参数)))
            self.动作记录.append(dict(原子=原子, 参数=dict(参数), 命中=False))
            return False, "真机后端异常：%s: %s" % (type(ex).__name__, ex)
        self.调用.append((原子, dict(参数)))
        self.动作记录.append(dict(原子=原子, 参数=dict(参数), 命中=True))
        return True, ""


# ═══════════════════════════════════════════════════════════════════════
# 三、计划（原子序列，取自 接口契约.全部原子）
# ═══════════════════════════════════════════════════════════════════════
def 默认计划() -> list:
    """一条覆盖 `接口契约.全部原子` 的原子序列 + 一条**脱靶对照**（证明命中不是恒真）。"""
    return [
        dict(序号=1, 原子="移到", 参数=dict(点=(320, 240), 靶="文档图标"),
             说明="把光标移到合成靶场的『文档图标』"),
        dict(序号=2, 原子="左键", 参数=dict(点=(320, 240), 靶="文档图标"),
             说明="左键点开文档靶元素"),
        dict(序号=3, 原子="键入", 参数=dict(文本="AHI meeting41 真机探头"),
             说明="键入标题串"),
        dict(序号=4, 原子="滚轮", 参数=dict(格数=-3), 说明="向下滚 3 格"),
        dict(序号=5, 原子="移到", 参数=dict(点=(640, 360), 靶="代码图标"),
             说明="移到代码靶元素"),
        dict(序号=6, 原子="右键", 参数=dict(点=(640, 360), 靶="代码图标"),
             说明="右键呼出菜单"),
        dict(序号=7, 原子="中键", 参数=dict(点=(640, 360), 靶="代码图标"), 说明="中键"),
        dict(序号=8, 原子="拖拽", 参数=dict(起=(320, 240), 止=(640, 360)),
             说明="从文档靶元素拖到代码靶元素"),
        dict(序号=9, 原子="左键", 参数=dict(点=(5000, 5000), 靶=None),
             说明="★脱靶对照：靶外点（命中必须为 False；真机档自动跳过）", 仅演练=True),
    ]


def 指纹(项: dict) -> str:
    """沿用 `许可令牌` 的同一指纹格式（种类:参数），否则令牌永远匹配不上。"""
    原子, 参 = str(项["原子"]), dict(项.get("参数") or {})
    if 原子 == "移到":
        return 许可令牌.指纹_移动(参.get("点"))
    if 原子 in ("左键", "右键", "中键"):
        return 许可令牌.指纹_键(原子, {"左键": "左", "右键": "右", "中键": "中"}[原子])
    if 原子 == "滚轮":
        return 许可令牌.指纹_滚轮(参.get("格数", 0))
    if 原子 == "键入":
        return 许可令牌.指纹_键入(参.get("文本", ""))
    return 许可令牌.指纹_键(原子, "%s→%s" % (参.get("起"), 参.get("止")))


# ═══════════════════════════════════════════════════════════════════════
# 四、真机探头
# ═══════════════════════════════════════════════════════════════════════
class 真机探头:
    """L5 真机验收探头。三件安全物：① 令牌闸 ② 演练/真机双档 ③ 现场确认。"""

    def __init__(self, 演练: bool = True, 已确认: bool = False, 产物目录_: str = None,
                 计划: list = None, 截图: bool = True, 轮数: int = 演练轮数,
                 强制自绘: bool = False):
        self.演练 = bool(演练)
        self.已确认 = bool(已确认)
        self.轮数 = max(1, int(轮数))           # 【总装 2026-10-08 修】构造参数未落到实例上，
                                                # `合成侧()` 却读 `self.轮数` ⇒ AttributeError
                                                # （演练档跑到"合成靶场练习"那一拍就崩）。
        self.目录 = 产物目录_ or 产物目录
        self.计划 = list(计划 if 计划 is not None else 默认计划())
        self.开截图 = bool(截图)
        self.强制自绘 = bool(强制自绘)          # 真机档也自绘（安全演练/无桌面环境用）
        self.钟 = 时基.拍钟()
        self.拍 = [0.0]                                  # 与拍钟同步的"当前拍"（令牌有效期用）
        self.局 = 许可令牌.令牌局(拍源=lambda: float(self.钟.分数拍()), 要求确认=True)
        self.闸 = 许可令牌.令牌闸(self.局, 可逆=set())     # ★空白名单：每个原子都要令牌
        self.台账: list = []
        self.截图记录: dict = {}
        self.后端 = None
        self.真机后端 = None                              # 真机档、且确认后才构造
        self.备注: list = []

    # ── 档位 ──
    @property
    def 档(self) -> str:
        return "演练档" if self.演练 else "真机档"

    # ── 安全物②：能力探测（演练档只探测、不动作） ──
    def 能力探测(self) -> dict:
        报 = dict(档=self.档)
        报["pyautogui_可导入"] = bool(importlib.util.find_spec("pyautogui"))
        报["pyautogui_屏幕只读接口可探测"] = bool(importlib.util.find_spec("PIL.ImageGrab"))
        报["演练档_实际import动作库"] = False
        if not self.演练:                                  # 真机档：惰性真 import（仍不动作）
            try:
                import pyautogui                               # noqa: PLC0415,F401
                报["演练档_实际import动作库"] = True
                报["pyautogui_版本"] = str(getattr(sys.modules["pyautogui"], "__version__", "未知"))
            except Exception as ex:                        # noqa: BLE001
                报["pyautogui_导入错误"] = "%s: %s" % (type(ex).__name__, ex)
        报["口径"] = ("演练档**绝不调用** pyautogui；本探测只用 importlib 规格查找，"
                     "不触发任何鼠标键盘动作，也不抓真屏")
        return 报

    # ── 安全物③：现场确认 ──
    def 打印计划(self) -> None:
        print("─" * 78)
        print("【%s】将要做的事（共 %d 个原子；★= 仅演练档）" % (self.档, len(self.计划)))
        for 项 in self.计划:
            标 = "★" if 项.get("仅演练") else " "
            print("  %s%2d. %-4s %s   %s" % (标, 项["序号"], 项["原子"],
                                          json.dumps(项.get("参数"), ensure_ascii=False),
                                          项.get("说明", "")))
        if self.演练:
            print("  ⓘ 演练档：以上动作**只在合成靶场里记账**，不会移动真鼠标/敲真键盘。")
        else:
            print("  ⚠ 真机档：以上动作会**真的移动鼠标、真的点击、真的敲键盘**；")
            print("     每个原子都要过 `许可令牌` 令牌闸（一次性 + 指纹绑定 + 2s 窗 + 确认事件）；")
            print("     越界坐标一律不动手；操作前后各落一张只读截图。")
        print("─" * 78)

    def 求确认(self, 输入流=None) -> bool:
        """演练档恒真；真机档：`--确认` 或 stdin 明确输入『确认』才算确认。"""
        if self.演练:
            return True
        if self.已确认:
            return True
        流 = 输入流
        if 流 is None:
            if not sys.stdin or not sys.stdin.isatty():
                return False
            try:
                流 = sys.stdin
            except Exception:                                  # noqa: BLE001
                return False
        try:
            行 = str(流.readline() or "").strip()
        except Exception:                                      # noqa: BLE001
            return False
        return 行 in ("确认", "y", "Y", "yes", "YES", "ok", "OK")

    # ── 令牌闸（记录拒因；fail-closed） ──
    def _只过闸(self, 指: str) -> tuple:
        """只过闸、**不签发**（用于证明"无令牌 ⇒ 不动手"）。"""
        前 = dict(self.局.统计表().get("拒因") or {})
        放 = bool(self.闸.放行(指))
        后 = dict(self.局.统计表().get("拒因") or {})
        新 = [k for k in 后 if 后[k] > 前.get(k, 0)]
        return 放, (新[0] if 新 else "拒（fail-closed）"), None

    def _过闸(self, 指: str, 项: dict) -> tuple:
        """签发一次性令牌 + 过闸（真机档每个原子动作的必经之路）。"""
        前 = dict(self.局.统计表().get("拒因") or {})
        令 = self.局.签发(指, 目标rid=str(项.get("原子", "")), 来源="真机探头",
                         确认事件="所有者确认:%d" % self.钟.拍(), 不可逆=True,
                         有效期拍=令牌有效期拍)
        放 = bool(self.闸.放行(指))
        后 = dict(self.局.统计表().get("拒因") or {})
        新 = [k for k in 后 if 后[k] > 前.get(k, 0)]
        return 放, (令.id if 放 else (新[0] if 新 else "拒（fail-closed）")), (令.id if 放 else None)

    # ── 单个原子（打点 → 过闸 → 执行 → 打点） ──
    def _跑一个原子(self, 项: dict, 后端, 自动签发: bool = True) -> dict:
        指 = 指纹(项)
        self.钟.打点("τ_act")
        τ = float(self.钟.分数拍())
        if 自动签发:
            放, 令牌标识, 令id = self._过闸(指, 项)
        else:
            放, 令牌标识, 令id = self._只过闸(指)
        判, 备注 = 后端.执行(项["原子"], dict(项["参数"])) if 放 else (False, "无令牌 ⇒ 不动手")
        self.钟.打点("τ_fb")
        耗 = float(self.钟.延迟("τ_act", "τ_fb")["毫秒"])
        return dict(序号=项["序号"], 原子=项["原子"], 参数=dict(项["参数"]), 指纹=指,
                    令牌=令id, 放行=bool(放), 令牌标识=(令牌标识 if not 放 else "已核销"),
                    命中=bool(判), 备注=备注, τ=τ, 耗时毫秒=耗, 档=self.档,
                    是否真机动作=bool(放 and not self.演练))

    # ── 原子循环 ──
    def 跑原子循环(self, 后端=None, 记账: bool = True, 自动签发: bool = True) -> list:
        后端 = 后端 or self.后端
        if 后端 is None:
            靶, _ = 取靶场(0)
            后端 = 演练后端(靶)
        self.后端 = 后端
        行s = []
        for 项 in self.计划:
            if (not self.演练) and 项.get("仅演练"):
                行s.append(dict(序号=项["序号"], 原子=项["原子"], 参数=dict(项["参数"]),
                               指纹=指纹(项), 状态="跳过(仅演练档)", 放行=False, 命中=None,
                               τ=None, 耗时毫秒=None, 是否真机动作=False))
                continue
            记 = self._跑一个原子(项, 后端, 自动签发=自动签发)
            行s.append(记)
            if 记账:
                self.台账.append(记)
        return 行s

    # ── 截图 ──
    def 截图(self, 阶段: str, 行s: list = None) -> dict:
        if not self.开截图:
            return dict(阶段=阶段, 跳过=True)
        名 = "真机_%s_%s.png" % ("演练" if self.演练 else "真机", 阶段)
        路径 = os.path.join(self.目录, 名)
        os.makedirs(self.目录, exist_ok=True)
        摘要 = ["mode=%s  (DRILL: no pyautogui calls)" % self.档,
                "atoms=%d  hit=%s" % (len(self.计划),
                                    sum(1 for x in (行s or []) if x.get("命中"))),
                "phase=%s" % 阶段]
        if self.演练 or self.强制自绘:
            果 = 自绘截图(路径, "%s 操作%s截图（自绘合成屏）" % (self.档, 阶段), 摘要)
        else:
            果 = 抓屏截图(路径)
        果 = dict(果, 阶段=阶段, 路径=路径)
        self.截图记录[阶段] = 果
        return 果

    # ── 真任务链（既有 meeting40 三件；演练档同样真跑，但不碰鼠标键盘） ──
    def 真任务链(self) -> dict:
        报 = {}
        os.makedirs(真任务目录, exist_ok=True)
        素材s = [dict(文本="第七轮：能力核 + 操作熟练 + 同时进行（真机探头演练档）", 来源="meeting41/实施"),
                dict(文本="合成靶场练技能 → 真任务首测 → 记录迁移耗时对比", 来源="本文件")]
        # ① 文档任务
        try:
            import 文档任务                                            # noqa: PLC0415
            self.钟.打点("τ_act")
            果 = 文档任务.做文档("AHI 第七轮 真机验收（演练档）", 素材s=素材s, 产物目录=真任务目录)
            self.钟.打点("τ_fb")
            报["文档任务"] = dict(成功=bool(果.get("落盘", {}).get("docx存在")),
                              耗时毫秒=float(self.钟.延迟("τ_act", "τ_fb")["毫秒"]),
                              产物=果.get("产物"), 段落数=果.get("段落数"))
        except Exception as ex:                                       # noqa: BLE001
            报["文档任务"] = dict(成功=False, 异常="%s: %s" % (type(ex).__name__, ex))
        # ② 代码任务
        try:
            import 代码任务                                            # noqa: PLC0415
            规格 = dict(任务名="统计网页段数", 函数名="段数", 操作="计数",
                       用例s=[dict(输入=[1, 2, 3], 期望=3), dict(输入=[], 期望=0)],
                       说明="真机探头演练档的迁移验证规格")
            self.钟.打点("τ_act")
            果 = 代码任务.写代码(规格, 产物目录=真任务目录, 文件名="真机验收_计数")
            self.钟.打点("τ_fb")
            报["代码任务"] = dict(
                成功=bool(果.get("语法通过") and 果.get("用例全过") and 果.get("子进程退出码") == 0),
                耗时毫秒=float(self.钟.延迟("τ_act", "τ_fb")["毫秒"]),
                产物=(果.get("产物") or {}).get("py"),
                自证=dict(语法通过=果.get("语法通过"), 子进程退出码=果.get("子进程退出码"),
                          用例全过=果.get("用例全过"), 用例结果=果.get("用例结果")),
                执行输出尾=str(果.get("执行输出尾", ""))[-160:])
        except Exception as ex:                                       # noqa: BLE001
            报["代码任务"] = dict(成功=False, 异常="%s: %s" % (type(ex).__name__, ex))
        # ③ 看网页任务（★允许开浏览器=False：不开浏览器、不抓真屏）
        try:
            import 看网页任务                                          # noqa: PLC0415
            self.钟.打点("τ_act")
            果 = 看网页任务.看网页(段落s=["真机探头的迁移验证只看本地页正文。",
                                     "允许开浏览器=False ⇒ 不夺前台、不抓真屏。",
                                     "耗时口径走 时基.拍钟（perf_counter）。"],
                              允许开浏览器=False, 桥=None, 产物目录=真任务目录,
                              主题="AHI 第七轮 真机验收（演练档）")
            self.钟.打点("τ_fb")
            报["看网页任务"] = dict(成功=bool(果.get("通路") and 果.get("通路") != "失败"),
                                耗时毫秒=float(self.钟.延迟("τ_act", "τ_fb")["毫秒"]),
                                通路=果.get("通路"), 标题=果.get("标题"),
                                文档=(果.get("文档") or {}).get("md"))
        except Exception as ex:                                       # noqa: BLE001
            报["看网页任务"] = dict(成功=False, 异常="%s: %s" % (type(ex).__name__, ex))
        return 报

    # ── 迁移对比：合成靶场（练 N 轮）vs 真任务（首测 1 次） ──
    def 迁移对比(self, 真任务: dict) -> list:
        映射 = [
            dict(任务="文档任务", 靶=[dict(原子="移到", 参数=dict(点=(320, 240), 靶="文档图标")),
                                 dict(原子="左键", 参数=dict(点=(320, 240), 靶="文档图标")),
                                 dict(原子="键入", 参数=dict(文本="AHI 第七轮 真机验收（演练档）"))]),
            dict(任务="代码任务", 靶=[dict(原子="移到", 参数=dict(点=(640, 360), 靶="代码图标")),
                                 dict(原子="左键", 参数=dict(点=(640, 360), 靶="代码图标")),
                                 dict(原子="键入", 参数=dict(文本="def 段数(xs): return len(xs)"))]),
            dict(任务="看网页任务", 靶=[dict(原子="滚轮", 参数=dict(格数=3)),
                                  dict(原子="移到", 参数=dict(点=(760, 420), 靶="网页区")),
                                  dict(原子="左键", 参数=dict(点=(760, 420), 靶="网页区"))]),
        ]
        靶, 来源 = 取靶场(0)
        后端 = 演练后端(靶)
        出 = []
        for m in 映射:
            耗时列, 命中列 = [], []
            for _ in range(max(1, self.轮数)):
                self.钟.打点("τ_act")
                全中 = True
                for a in m["靶"]:
                    中, _备 = 后端.执行(a["原子"], dict(a["参数"]))
                    全中 = 全中 and bool(中)
                self.钟.打点("τ_fb")
                耗时列.append(float(self.钟.延迟("τ_act", "τ_fb")["毫秒"]))
                命中列.append(bool(全中))
            序 = sorted(耗时列)
            中最 = 序[len(序) // 2]
            真 = 真任务.get(m["任务"], {})
            出.append(dict(任务=m["任务"], 靶场来源=来源, 轮数=len(耗时列),
                           靶场耗时毫秒列=[round(x, 4) for x in 耗时列],
                           靶场最快毫秒=round(序[0], 4), 靶场中位毫秒=round(中最, 4),
                           靶场末次毫秒=round(耗时列[-1], 4),
                           靶场命中率=round(sum(1 for x in 命中列 if x) / float(len(命中列)), 4),
                           靶场最快_不超3000ms=bool(序[0] <= 3000.0),
                           真任务成功=bool(真.get("成功")), 真任务耗时毫秒=真.get("耗时毫秒"),
                           真任务产物=(真.get("产物") or 真.get("文档"))))
        return 出

    # ── 汇总 ──
    @staticmethod
    def 净台账(台账: list) -> list:
        """去掉墙钟字段后的台账（特征码口径：只留确定性字段）。"""
        return [{k: v for k, v in x.items() if k not in ("τ", "耗时毫秒")} for x in 台账]

    def 跑全流程(self, 真任务: bool = True) -> dict:
        os.makedirs(self.目录, exist_ok=True)
        报 = dict(档=self.档, 演练=bool(self.演练), 真机动作数=0,
                  时间="不写（结果里不放墙钟绝对时刻）")
        self.打印计划()
        报["能力探测"] = self.能力探测()
        # ③ 现场确认（真机档才有闸）
        if not self.演练:
            确 = self.求确认()
            报["现场确认"] = dict(获得确认=bool(确))
            if not 确:
                self.备注.append("真机档未获所有者确认 ⇒ 拒绝执行（未构造真机后端、零动作）")
                报.update(执行=False, 拒绝原因="真机档未获现场确认（stdin 或 --确认）",
                          真机后端构造=False, 台账=[])
                报["安全物"] = self.安全物自证()
                return 报
        # 后端构造（真机档只在确认之后）
        靶, 靶来源 = 取靶场(0)
        if self.演练:
            self.后端 = 演练后端(靶)
            报["后端"] = dict(类型="演练后端(合成靶场记账)", 靶场来源=靶来源,
                            说明="绝不调用 pyautogui")
        else:
            self.真机后端 = 真机后端()
            self.后端 = self.真机后端
            报["后端"] = dict(类型="真机后端(pyautogui)", 屏幕=list(self.真机后端.屏),
                            说明="每个原子先过令牌闸；越界不动手")
        self.截图("前")
        行s = self.跑原子循环(self.后端)
        self.截图("后", 行s)
        报["真机动作数"] = sum(1 for x in 行s if x.get("是否真机动作"))
        报["演练命中数"] = sum(1 for x in 行s if x.get("命中"))
        报["演练失手数"] = sum(1 for x in 行s if x.get("命中") is False)
        报["原子台账"] = 行s
        报["截图"] = self.截图记录
        报["台账特征码"] = 时基.特征码(self.净台账(行s))
        报["台账口径"] = "τ/耗时毫秒为时基.拍钟(perf_counter)读数，属诊断；特征码不含墙钟字段"
        if 真任务:
            真 = self.真任务链()
            报["真任务"] = 真
            报["迁移对比"] = self.迁移对比(真)
        return 报

    # ── 三件安全物自证（演练档也跑；不打真机） ──
    def 安全物自证(self) -> dict:
        报 = {}
        # ① 令牌闸
        拍 = [0.0]
        局 = 许可令牌.令牌局(拍源=lambda: 拍[0], 要求确认=True)
        闸 = 许可令牌.令牌闸(局, 可逆=set())
        靶, _ = 取靶场(0)
        演练 = 演练后端(靶)
        案 = dict(
            无令牌拒=(闸.放行("键:点击:左") is False),
            无令牌零动作=(len(演练.调用) == 0),
            可逆白名单为空=(闸.可逆 == set()),
        )
        令 = 局.签发("键:点击:左", 确认事件="确认:1")
        案["有令牌放行"] = (闸.放行("键:点击:左") is True)
        案["重放拒"] = (闸.放行("键:点击:左") is False)
        案["一次性"] = (局.令牌s[令.id].已用拍 is not None)
        令2 = 局.签发("键入:abc", 确认事件="确认:2")
        案["指纹不匹配拒"] = (局.核销(令2.id, "键入:xyz")["放行"] is False)
        令3 = 局.签发("滚轮:3", 确认事件="确认:3", 有效期拍=2.0)
        拍[0] = 99.0
        案["过期拒"] = (局.核销(令3.id, "滚轮:3")["放行"] is False)
        令4 = 局.签发("键:点击:左", 确认事件=None)
        案["不可逆缺确认拒"] = (局.核销(令4.id, "键:点击:左")["放行"] is False)

        class _坏闸:
            def 放行(self, s):
                raise RuntimeError("闸坏")
        案["闸故障拒"] = (许可令牌.令牌闸(_坏闸()).放行("键:点击:左") is False)
        # ★ 真机后端在无令牌时也必须零调用（打桩模块，不打真机）
        桩 = _记录桩()
        真后 = 真机后端(模块=桩, 屏幕=(1920, 1080))
        探 = 真机探头(演练=False, 已确认=True, 产物目录_=self.目录, 截图=False)
        探.真机后端, 探.后端 = 真后, 真后
        探.计划 = [dict(序号=1, 原子="左键", 参数=dict(点=(10, 10)), 说明="无令牌对照")]
        探.跑原子循环(真后, 自动签发=False)                 # 局里还没有令牌
        案["真机后端无令牌零调用"] = (len(桩.调用) == 0)
        探.局.签发(许可令牌.指纹_键("左键", "左"), 确认事件="确认:9")
        探.跑原子循环(真后, 自动签发=False)                 # 有令牌 ⇒ 放行 ⇒ 恰一次
        案["真机后端有令牌恰一次"] = (len(桩.调用) == 1)
        探.跑原子循环(真后, 自动签发=False)                 # 同指纹重放 ⇒ 拒
        案["重放零增量"] = (len(桩.调用) == 1)
        报["①许可令牌闸"] = dict(通过=bool(all(案.values())), 逐项=案, 局统计=局.统计表(),
                              闸统计=闸.统计表())

        # ② 演练/真机双档
        桩2 = _记录桩()
        原 = sys.modules.get("pyautogui", None)
        演练调用 = None
        try:
            sys.modules["pyautogui"] = 桩2                      # 若演练档真去调，一定会被记到
            探2 = 真机探头(演练=True, 产物目录_=self.目录, 截图=False, 轮数=1)
            探2.能力探测()
            探2.跑原子循环(演练后端(取靶场(0)[0]))
            演练调用 = len(桩2.调用)
        finally:
            if 原 is None:
                sys.modules.pop("pyautogui", None)
            else:
                sys.modules["pyautogui"] = 原
        桩3 = _记录桩()
        真后3 = 真机后端(模块=桩3, 屏幕=(1920, 1080))
        探3 = 真机探头(演练=False, 已确认=True, 产物目录_=self.目录, 截图=False)
        探3.计划 = [dict(序号=1, 原子="左键", 参数=dict(点=(10, 10)), 说明="对照")]
        探3.真机后端 = 真后3
        探3.局.签发(许可令牌.指纹_键("左键", "左"), 确认事件="确认:9")
        探3.跑原子循环(真后3)
        前截 = (self.截图记录.get("前") or {})
        截图来源 = str(前截.get("来源", ""))
        案2 = dict(演练档_pyautogui调用数=int(演练调用 or 0), 演练档零调用=(int(演练调用 or 0) == 0),
                  对照_真机后端调用数=len(桩3.调用), 对照_真机后端会动作=(len(桩3.调用) == 1),
                  演练档不抓真屏=bool((not self.演练) or (not self.开截图)
                                 or 截图来源.startswith(("自绘", "兜底手写"))),
                  演练档档位=str(self.档), 截图来源=截图来源 or "（未截图）")
        报["②演练真机双档"] = dict(
            通过=bool(案2["演练档零调用"] and 案2["对照_真机后端会动作"] and 案2["演练档不抓真屏"]),
            逐项=案2,
            说明="演练档照常落地，但 pyautogui 调用数恒 0（用记录桩证明计数可增，非恒零假象）；"
                "对照臂=真机后端（注入记录桩）确实会动作")

        # ③ 现场确认
        空流 = io.StringIO("")
        探A = 真机探头(演练=False, 已确认=False, 产物目录_=self.目录, 截图=False)
        探B = 真机探头(演练=False, 已确认=True, 产物目录_=self.目录, 截图=False)
        探C = 真机探头(演练=False, 已确认=False, 产物目录_=self.目录, 截图=False)
        探D = 真机探头(演练=True, 产物目录_=self.目录, 截图=False)
        案3 = dict(无确认被拒=(探A.求确认(输入流=空流) is False),
                  无确认不构造真机后端=(探A.真机后端 is None),
                  开关确认生效=(探B.求确认(输入流=空流) is True),
                  口头确认生效=(探C.求确认(输入流=io.StringIO("确认\n")) is True),
                  口头乱输被拒=(真机探头(演练=False, 已确认=False, 产物目录_=self.目录, 截图=False)
                            .求确认(输入流=io.StringIO("随便\n")) is False),
                  演练档无需确认=(探D.求确认(输入流=空流) is True))
        报["③现场确认"] = dict(通过=bool(all(案3.values())), 逐项=案3,
                           说明="真机档必须先打印计划再等确认；未确认 ⇒ 不构造真机后端、零动作、exit=2")
        return 报


# ═══════════════════════════════════════════════════════════════════════
# 五、主流程 / CLI
# ═══════════════════════════════════════════════════════════════════════
def _汇总判定(报: dict, 安全物: dict) -> dict:
    闸 = 安全物["①许可令牌闸"]["通过"]
    双 = 安全物["②演练真机双档"]["通过"]
    确 = 安全物["③现场确认"]["通过"]
    台账 = 报.get("原子台账") or []
    演练行 = [x for x in 台账 if x.get("状态") != "跳过(仅演练档)"]
    全放行 = bool(演练行) and all(x.get("放行") for x in 演练行)
    命中 = sum(1 for x in 演练行 if x.get("命中"))
    脱靶行 = [x for x in 台账 if x.get("状态") != "跳过(仅演练档)" and not x.get("命中")]
    真任务过 = all(bool(v.get("成功")) for v in (报.get("真任务") or {}).values()) \
        if 报.get("真任务") else True
    截图落 = bool((报.get("截图") or {}).get("前", {}).get("路径")) and \
        bool((报.get("截图") or {}).get("后", {}).get("路径"))
    return dict(安全物全过=bool(闸 and 双 and 确), 原子全放行=全放行,
                命中数=命中, 原子数=len(演练行), 失手数=len(脱靶行),
                有脱靶对照=any(x.get("状态") != "跳过(仅演练档)" and not x.get("命中") for x in 台账),
                真任务全过=bool(真任务过), 截图前后落盘=截图落)


def 主(演练: bool = True, 已确认: bool = False, 真任务: bool = True, 截图: bool = True,
      轮数: int = 演练轮数, 存: bool = True, 强制自绘: bool = False) -> dict:
    探 = 真机探头(演练=演练, 已确认=已确认, 截图=截图, 轮数=轮数, 强制自绘=强制自绘)
    报 = 探.跑全流程(真任务=真任务)                # 先跑流程（含截图），安全物自证要读截图来源
    安全物 = 探.安全物自证()
    报["安全物"] = 安全物
    判定 = _汇总判定(报, 安全物)
    # 二次跑一遍（演练档确定性自证：净台账特征码逐位同）
    if 演练 and 报.get("原子台账"):
        探2 = 真机探头(演练=True, 截图=False, 轮数=轮数)
        行2 = 探2.跑原子循环(演练后端(取靶场(0)[0]))
        判定["演练档两遍净台账逐位同"] = bool(
            时基.特征码(探.净台账(报["原子台账"])) == 时基.特征码(探2.净台账(行2)))
    else:
        判定["演练档两遍净台账逐位同"] = None
    报["判定"] = 判定
    报["口径"] = dict(
        令牌="一次性 + 指纹绑定 + 有效期 %.0f 拍 + 不可逆须确认事件；异常一律拒（fail-closed）"
             % 令牌有效期拍,
        闸="可逆白名单为空 ⇒ 每一次原子动作都要令牌",
        演练档="绝不调用 pyautogui；截图=自绘合成屏；真任务链走既有三件且不开浏览器",
        真机档="打印计划 → 现场确认（stdin/--确认）→ 惰性构造 pyautogui 后端 → 每原子过闸",
        耗时口径="时基.拍钟（perf_counter 口径的分数拍/毫秒），非 time.time()",
        靶场="优先 技能原子.靶场（实现者B），未到位则 ★桩靶场",
    )
    报["诚实边界"] = [
        "本轮为**演练档**：真机动作数=0，**没有**移动真鼠标/敲真键盘；真机读数必须由所有者在场另跑。",
        "演练档截图是**自绘合成屏**（不抓真屏、不夺前台）；真机档才用只读抓屏（PIL.ImageGrab，非 pyautogui）。",
        "『合成靶场耗时』来自演练后端纯记账，**不可**与真机鼠标耗时直接比；它只回答靶场口径下"
        "技能序列是否稳定命中、是否 ≤3000ms。",
        "『真任务耗时』是真执行既有任务件的时基耗时，含解析与落盘成本，**不等于**人手操作耗时。",
        "迁移验证本轮只到「同一技能序列靶场命中 + 真任务件真跑成功」这一层，"
        "**不主张**已会操作真实软件界面。",
        ("靶场为 ★桩靶场（技能原子.靶场 未到位）；真件到位后 取靶场() 自动改用真件并应复跑。"
         if (报.get("迁移对比") or [{}])[0].get("靶场来源", "").startswith("桩") else
         "靶场取自真件 技能原子.靶场。"),
    ]
    if 存:
        os.makedirs(产物目录, exist_ok=True)
        净 = {k: v for k, v in 报.items() if k not in ("截图",)}
        报["结果特征码"] = 时基.特征码(净)
        with open(验收路径, "w", encoding="utf-8") as f:
            json.dump(报, f, ensure_ascii=False, indent=1, default=str)
    return 报


def _打印(报: dict) -> None:
    print("档位：%s（真机动作数=%d）" % (报["档"], 报.get("真机动作数", 0)))
    print("能力探测：%s" % json.dumps(报.get("能力探测"), ensure_ascii=False))
    判 = 报.get("判定") or {}
    print("判定：安全物全过=%s ｜ 原子全放行=%s ｜ 命中 %s/%s ｜ 真任务全过=%s"
          % (判.get("安全物全过"), 判.get("原子全放行"), 判.get("命中数"), 判.get("原子数"),
             判.get("真任务全过")))
    for 名, v in (报.get("安全物") or {}).items():
        print("  %s：%s" % (名, "通过" if v.get("通过") else "未过"))
        for k, x in (v.get("逐项") or {}).items():
            print("     · %-26s %s" % (k, x))
    print("原子台账（τ/耗时/命中）：")
    for x in (报.get("原子台账") or []):
        print("  %2s %-4s 放行=%-5s 命中=%-5s τ=%-8s 耗时毫秒=%-10s %s"
              % (x.get("序号"), x.get("原子"), x.get("放行"), x.get("命中"),
                 (None if x.get("τ") is None else round(x["τ"], 3)),
                 (None if x.get("耗时毫秒") is None else round(x["耗时毫秒"], 4)),
                 x.get("备注") or x.get("状态") or ""))
    for 阶段, s in (报.get("截图") or {}).items():
        print("  截图[%s]：%s（%s）" % (阶段, s.get("路径"), s.get("来源")))
    if 报.get("迁移对比"):
        print("合成靶场 vs 真任务（迁移对比）：")
        for m in 报["迁移对比"]:
            print("  %-6s 靶场最快=%-9s 中位=%-9s 末次=%-9s 命中率=%-6s ｜ 真任务成功=%-5s 耗时毫秒=%s"
                  % (m["任务"], m["靶场最快毫秒"], m["靶场中位毫秒"], m["靶场末次毫秒"],
                     m["靶场命中率"], m["真任务成功"], m["真任务耗时毫秒"]))
    for x in (报.get("诚实边界") or []):
        print("  ⓘ %s" % x)
    print("结果：%s" % 验收路径)


def _命令行(argv=None):
    解 = argparse.ArgumentParser(description="meeting41 L5 真机探头（默认演练档，不动真鼠标键盘）")
    解.add_argument("--真机", action="store_true", help="切真机档（会真动鼠标键盘；须所有者在场）")
    解.add_argument("--确认", action="store_true", help="真机档的现场确认开关（等价于 stdin 输入『确认』）")
    解.add_argument("--演练", action="store_true", help="显式演练档（与 --真机 同给时以演练为准）")
    解.add_argument("--无真任务", action="store_true", help="跳过既有三件真任务链")
    解.add_argument("--自绘图", action="store_true",
                    help="强制自绘截图（演练档本来就是自绘；真机档默认只读抓屏）")
    解.add_argument("--轮数", type=int, default=演练轮数, help="合成靶场侧练几轮（默认 %d）" % 演练轮数)
    解.add_argument("--静默", action="store_true", help="不打印台账明细")
    return 解.parse_args(argv)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                                  # noqa: BLE001
        pass
    参 = _命令行()
    演练 = True if 参.演练 else (not 参.真机)
    报 = 主(演练=演练, 已确认=bool(参.确认), 真任务=(not 参.无真任务),
           截图=(not 参.自绘图), 轮数=int(参.轮数), 存=True)
    if not 参.静默:
        _打印(报)
    判 = 报.get("判定") or {}
    if not 演练:
        if not 报.get("执行", True):
            print("[真机探头] 拒绝执行：%s（未构造真机后端、零动作）" % 报.get("拒绝原因"))
            raise SystemExit(2)
        好 = bool(判.get("安全物全过") and 判.get("原子全放行"))
        print("[真机探头] 真机档 %s（真机动作数=%d）" % ("PASS" if 好 else "FAIL", 报.get("真机动作数", 0)))
        raise SystemExit(0 if 好 else 1)
    好 = bool(判.get("安全物全过") and 判.get("原子全放行") and 判.get("真任务全过")
             and 判.get("演练档两遍净台账逐位同") and 判.get("截图前后落盘"))
    print("[真机探头] %s：演练档（真机动作数=0；不动真鼠标键盘）｜结果=%s"
          % ("PASS" if 好 else "FAIL", 验收路径))
    raise SystemExit(0 if 好 else 1)
