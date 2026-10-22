# -*- coding: utf-8 -*-
"""分支_原生多模态/浏览器体.py —— #280 三位一体：思 × 言 × 行 + 回响空间。

三条流同一节拍并行（机器人同构，实时 ~2Hz）：
  思（内部）：思考态每拍推进一次迭代检索（不阻塞动作）——延答机制 #270 同源
  言（说出）：每条自言实时写日志（自言自语可 tail）；感知先入账再引用 #242
  行（操作）：滚动/移动/点击/读页/输入/观察轮替——读页=系统文本通道（Ctrl+A+C，
              非 OCR：文字来自系统剪贴板，非图像识别）；输入=他自己的疑问去必应搜索
  回响空间：保留最近 48 帧视野，每 7 拍回放一帧做视觉再认（来过/演化/大不一样），
            『大不一样』产生疑问 → 思考队列 → 内部思考 → 想毕投递
无 OCR：视觉只有向量；页面文字走系统文本通道（窗口标题 + 剪贴板全文）。
用法：python 浏览器体.py [起始URL]     # 无限实时环
"""
import ctypes
import json
import os
import re
import subprocess
import sys
import time
from collections import deque

from ctypes import wintypes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")

import numpy as np
import pyautogui
import pyperclip
import pyttsx3

from synthesis import 综合体
from 视网膜 import 网格, 变化度, 相似度, 区域差分, 投影

pyautogui.FAILSAFE = False
日志 = os.path.join(HERE, "浏览器体日志.log")

# ---------- #285 声带（语音输出：说话——离线 SAPI） ----------
try:
    _声带 = pyttsx3.init()
    _声带.setProperty("rate", 190)
    _声带.setProperty("volume", 0.55)
except Exception:
    _声带 = None


def 说出口(文本: str):
    if _声带 is None:
        return
    try:
        _声带.say(文本[:60])
        _声带.runAndWait()
    except Exception:
        pass


def 记(s: str):
    行 = "[%s] %s" % (time.strftime("%H:%M:%S"), s)
    print(行, flush=True)
    with open(日志, "a", encoding="utf-8") as f:
        f.write(行 + "\n")


def 看():
    """一帧双产物：(256 维投影向量, 2304 维空间网格)。"""
    g = 网格(pyautogui.screenshot())
    return g.flatten() @ 投影, g


# ---------- Win32 窗口管理（视觉=主屏，浏览器必须在主屏最前台） ----------
当前窗口 = [None, ""]


def 找浏览器窗口(关键词s=("edge", "chrome", "必应", "bing", "firefox")):
    user32 = ctypes.windll.user32
    命中 = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def 回调(hwnd, lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        t = buf.value.lower()
        if any(k in t for k in 关键词s):
            命中.append((hwnd, buf.value))
        return True

    user32.EnumWindows(回调, 0)
    return 命中


def 前置浏览器() -> str:
    命中 = 找浏览器窗口()
    if not 命中:
        return ""
    user32 = ctypes.windll.user32
    hwnd, 标题 = 命中[0]
    user32.ShowWindow(hwnd, 9)            # SW_RESTORE
    # 视网膜只覆盖主屏——窗口若开在副屏要移回来（虚拟屏可能是多显示器）
    user32.MoveWindow(hwnd, 0, 0, 1600, 900, True)
    # 后台进程会被 Windows 前景锁拒绝 SetForegroundWindow——SwitchToThisWindow 实测有效
    user32.SwitchToThisWindow(hwnd, True)
    当前窗口[0], 当前窗口[1] = hwnd, 标题
    return 标题


def 窗口标题() -> str:
    user32 = ctypes.windll.user32
    h = 当前窗口[0]
    if h is None:
        return ""
    n = user32.GetWindowTextLengthW(h)
    if n <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(h, buf, n + 1)
    当前窗口[1] = buf.value
    return buf.value


# ---------- 手的动作（读页=系统文本通道；输入=他自己的疑问驱动） ----------
def 读页文本() -> str:
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.08)
    pyautogui.hotkey("ctrl", "c")
    time.sleep(0.25)
    文 = pyperclip.paste() or ""
    pyautogui.press("esc")                # 取消全选高亮
    return 文


def 搜索疑问(q: str):
    # 地址栏直达（Ctrl+L 粘贴回车）——粘贴的是提炼后的关键词（整句疑问会被
    # Windows 当文件名，结果页变成『Windows 找不到…』——卡死循环的根因）
    pyperclip.copy(q[:24])
    pyautogui.hotkey("ctrl", "l")
    time.sleep(0.15)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.1)
    pyautogui.press("enter")


def 导航(url: str):
    pyperclip.copy(url)
    pyautogui.hotkey("ctrl", "l")
    time.sleep(0.15)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.1)
    pyautogui.press("enter")


def 疑问转搜索词(q: str) -> str:
    """疑问句→可搜索的关键词（整句必败，先提炼）。"""
    if "怎么让页面动起来" in q:
        return "网页 交互 操作"
    if "还有什么我能试的" in q:
        return "浏览器 使用技巧"
    if "很不一样" in q:
        return "网页 变化 原因"
    t = re.sub(r"拍\d+的画面", "画面", q)
    for w in ("这中间发生了什么", "我在原地停了很久", "——",
              "？", "。", "！", "，", "和现在"):
        t = t.replace(w, " ")
    t = t.strip()
    return t[:20] if len(t) >= 2 else "网页 交互"


# #285 课程表（所有者指派，EmoCompanion视角学东西：编程/开源社区/云/剪辑）
课程表 = deque([
    ("GitHub 趋势", "https://github.com/trending"),
    ("Gitee 开源社区", "https://gitee.com/explore/all"),
    ("阿里云开发者社区", "https://developer.aliyun.com/community"),
    ("视频剪辑入门", "https://cn.bing.com/search?q=%E8%A7%86%E9%A2%91%E5%89%AA%E8%BE%91%E5%85%A5%E9%97%A8%E6%95%99%E7%A8%8B"),
])


# ---------- #286 按键课：教学先验 + 试错经验（不同位置该用什么键） ----------
# 教学先验（像人被教鼠标一样先懂语义）：
#   左键=激活、双击=选词、右键=上下文菜单、中键=链接新开/标签关、拖选=选段落、
#   滚轮=滚动；但"哪个位置用哪个键有反应"必须他自己按出来——效果表说话。
按键s = ("左键", "双击", "右键", "中键", "拖选")
键效果 = {}                    # (格row, 格col, 键) -> [效果分(diff 值)…]
待评定 = None                  # (格row, 格col, 键)——上拍按的键等本拍 diff 回填


def 九宫格(x: int, y: int):
    """屏幕粗化为 3×3 格——'位置'的最低分辨率表示。"""
    return (min(int(y) // 360, 2), min(int(x) // 640, 2))


def 执行键(键: str, 拍: int):
    x = 400 + (拍 * 211) % 1100
    y = 200 + (拍 * 149) % 600
    if 键 == "左键":
        pyautogui.click(x, y)
    elif 键 == "双击":
        pyautogui.click(x, y, clicks=2, interval=0.08)
    elif 键 == "右键":
        pyautogui.click(x, y, button="right")
    elif 键 == "中键":
        pyautogui.middleClick(x, y)
    elif 键 == "拖选":
        pyautogui.moveTo(x, y)
        pyautogui.mouseDown()
        pyautogui.moveRel(150, 45, duration=0.3)
        pyautogui.mouseUp()
    return x, y


def 选键(格位: tuple, 拍: int) -> str:
    """经验驱动选键：70% 利用（该格平均效果最好的键），30% 探索（确定性游标）。"""
    探索键 = 按键s[拍 % len(按键s)]
    if 拍 % 3 == 2:
        return 探索键
    best, best分 = None, -1.0
    for 键 in 按键s:
        效果s = 键效果.get(格位 + (键,))
        均 = sum(效果s) / len(效果s) if 效果s else -1.0
        if 均 > best分:
            best, best分 = 键, 均
    if best is None or best分 <= 0.0:
        return 探索键
    return best


# ---------- #284 资源自律：感知内存与标签数，学会自己开关 ----------
class 内存状态(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class 进程内存计数(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def 内存压力():
    """返回 (系统内存占用%, 可用GB)。"""
    st = 内存状态()
    st.dwLength = ctypes.sizeof(内存状态)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return int(st.dwMemoryLoad), st.ullAvailPhys / 2**30


def 自身内存MB() -> float:
    pmc = 进程内存计数()
    pmc.cb = ctypes.sizeof(进程内存计数)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    h = k32.GetCurrentProcess()
    psapi = ctypes.windll.psapi
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                           ctypes.POINTER(进程内存计数),
                                           wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
        return 0.0
    return pmc.WorkingSetSize / 2**20


def 标签数(标题: str) -> int:
    m = re.search(r"和另外 (\d+) 个页面", 标题)
    return (int(m.group(1)) + 1) if m else 1


def 重启浏览器(起始: str):
    subprocess.run(["taskkill", "/IM", "msedge.exe", "/F"],
                   capture_output=True)
    time.sleep(2)
    subprocess.Popen(["cmd", "/c", "start", "", 起始],
                     shell=False, creationflags=subprocess.CREATE_NO_WINDOW)
    time.sleep(6)
    前置浏览器()


# ---------- 主循环：三位一体 ----------
def 主(起始: str):
    global 待评定
    桥 = 综合体(seed=999)
    # #281 所有者授权：开放域=项目根目录+E–N 全部盘（备份目录仍屏蔽——保护他自己的
    # 血统档案，防自读死循环）+ 联网（浏览/搜索/下载）；浏览器登录态沿用系统 Edge
    屏蔽 = [r"<workspace>/03_AHI\AHI2.1\backup_20261004_v10自言体"]
    盘符 = ["%<path>\\" % c for c in "EFGHIJKLMN"] + [r"<workspace>/"]
    桥.设开放域(盘符, 屏蔽)
    桥.账本.append(("接收", "联网授权：允许网络访问（浏览/搜索/下载）；浏览器"
                    "已保存登录态沿用系统 Edge（所有者授权）"))
    桥.挂舱("视觉")
    舱 = 桥.舱s["视觉"]
    # #285 EmoCompanion视角：身份附身（人设里"对新事物永远好奇"——学习者人格）
    身份p = os.path.join(os.path.dirname(HERE), "身份_EmoCompanion.json")
    if os.path.exists(身份p):
        from identity import 身份模板
        桥.加载身份(身份模板.从文件(身份p))
        记("EmoCompanion身份已加载（学习者人格）")
    前帧 = None
    前网格 = None
    静止连拍 = 0
    验证挂起 = 0                          # 人机验证时停手拍数（等待所有者）
    重启冷却 = 0                          # 重启浏览器后的冷却拍数（防重启循环）
    搜索失败连 = 0                        # 连续搜索失败（错误页）计数
    回响池 = deque(maxlen=48)             # (拍号, 帧向量)——回响空间
    回响游标 = 0
    最近疑问 = None
    上次疑问 = None
    拍 = 0

    def 说(类型: str, 文本: str, 引用) -> int:
        """自言自语：内部流+实时说出（日志）。"""
        n = 桥._自言(类型, 文本, ("内生", "浏览器体"), 引用)
        记("自言[%s] %s" % (类型, 文本))
        return n

    记("分支体启动（三位一体）：起始页 %s" % 起始)
    subprocess.Popen(["cmd", "/c", "start", "", 起始],
                     shell=False, creationflags=subprocess.CREATE_NO_WINDOW)
    time.sleep(6)
    记("前置浏览器窗口：%s" % (前置浏览器() or "未找到"))
    while True:
        try:
            t0 = time.time()
            拍 += 1
            # ---- 思（内部流）：思考态推进（与动作同拍并行，不阻塞） ----
            if 桥.思考中 is None and 桥.思考队列:
                q = 桥.思考队列.popleft()
                桥.思考中 = dict(问题=q, 到达拍=拍, 证据=[], 拍数=0)
                记("开想 ｜ %s" % q[:60])
            if 桥.思考中 is not None:
                前数 = len(桥.回复台账)
                桥._思考一拍()
                if 桥.思考中 is None and len(桥.回复台账) > 前数:
                    rec = 桥.回复台账[-1]
                    记("想毕 ｜ 答：%s" % str(rec.get("响应", ""))[:80])
                    说出口(str(rec.get("响应", "")))

            # ---- 看（感知流） ----
            帧, 网格v = 看()
            diff = 变化度(前帧, 帧) if 前帧 is not None else 1.0
            区行, 区列, 区量 = (区域差分(前网格, 网格v) if 前网格 is not None
                              else (0, 0, diff))
            前帧, 前网格 = 帧, 网格v
            标题 = 窗口标题()
            # 感知数据先入账（检验类），自言引用之（#242 溯源纪律）
            账号 = len(桥.账本)
            桥.账本.append(("检验", "视觉 拍=%d diff=%.4f 变区=(%d,%d)q=%.3f 标题[%s]"
                            % (拍, diff, 区行, 区列, 区量, 标题[:24])))
            # ---- 按键效果回填（#286：上拍按的键，这拍看反应） ----
            if 待评定 is not None:
                效果s = 键效果.setdefault(待评定, [])
                效果s.append(round(diff, 4))
                if len(效果s) > 20:
                    效果s.pop(0)
                if 待评定[2] == "右键":
                    pyautogui.press("esc")          # 菜单已观测完毕（出没出都收起）
                待评定 = None

            # ---- 言（自言自语流）+ 门控视觉段学习 ----
            if diff > 0.02:
                说("假设", "【视觉】画面变了 %.0f%%，最大变化在(%d,%d)区——页面响应了。"
                   % (diff * 100, 区行, 区列), [账号])
                静止连拍 = 0
                指纹 = ",".join("%.2f" % x for x in 帧[::32][:40])
                if 舱.命中(指纹) < 0.5:
                    舱.存行("帧#%d|[%s]|%s" % (拍, 标题[:20], 指纹))
            else:
                静止连拍 += 1
                if 静止连拍 % 10 == 0:
                    说("复盘", "【视觉】画面已静止 %d 拍——考虑换动作。" % 静止连拍,
                       [账号])
                if 静止连拍 == 30:
                    静止连拍 = 0
                    t2 = 前置浏览器()
                    记("连续静止——自检重前置窗口：%s" % (t2 or "未找到"))

            # ---- 资源自律（#284）：感知内存与标签数，学会自己开关 ----
            占用, 可用GB = 内存压力()
            标签n = 标签数(标题)
            资源动作 = None
            if 重启冷却 > 0:
                重启冷却 -= 1
            elif 占用 > 88 and 可用GB < 2.5:      # 二级：系统重度紧张
                if 标签n > 1:
                    for _ in range(min(标签n - 1, 12)):
                        pyautogui.hotkey("ctrl", "w")
                        time.sleep(0.12)
                    资源动作 = ("内存紧张（占用%d%%、可用%.1fGB）——"
                                "我关掉了多余标签" % (占用, 可用GB))
                else:
                    重启浏览器(起始)
                    重启冷却 = 240                 # 2 分钟内不再重启（防循环）
                    资源动作 = ("内存重度紧张且只剩一个标签——"
                                "我重启了浏览器")
                静止连拍 = 0
            elif 占用 > 80 and 标签n > 3:         # 一级：轻度紧张
                pyautogui.hotkey("ctrl", "w")
                资源动作 = ("内存偏高（占用%d%%）——我关了 1 个标签（剩 %d）"
                            % (占用, 标签n - 1))
            自身MB = 自身内存MB()
            if 自身MB > 600:                      # 自己胖了：落地账本再自重启
                落地 = os.path.join(HERE, "浏览器体账本落地.jsonl")
                with open(落地, "a", encoding="utf-8") as f:
                    for t, x in 桥.账本:
                        f.write(json.dumps({"类型": t, "内容": str(x)[:200]},
                                           ensure_ascii=False) + "\n")
                记("自身内存 %.0fMB——账本落地，自重启" % 自身MB)
                os.execv(sys.executable,
                         [sys.executable, os.path.abspath(__file__), 起始])
            if 资源动作:
                帐号r = len(桥.账本)
                桥.账本.append(("自我", 资源动作))
                说("复盘", "【资源】%s" % 资源动作, [帐号r])
                记("拍%4d %s ｜ 资源自律" % (拍, 资源动作))
                time.sleep(max(0.0, 0.5 - (time.time() - t0)))
                continue

            # ---- 回响空间（内部回放与再认） ----
            if 拍 % 7 == 3 and 回响池:
                回响游标 = 回响游标 % len(回响池)
                旧拍, 旧帧 = 回响池[回响游标]
                回响游标 += 1
                sim = 相似度(旧帧, 帧)
                if sim > 0.995:
                    说("复盘", "【回响】拍%d的画面与现在几乎一样——我停在原地。" % 旧拍,
                       [账号])
                    # 停滞本身成为他自己的疑问（感知驱动，非外部指令；同一疑问不复读）
                    if (最近疑问 is None and 桥.思考中 is None
                            and not 桥.思考队列):
                        q = ("我在原地停了很久——怎么让页面动起来？"
                             if 上次疑问 is None else
                             "画面又停住了——还有什么我能试的？")
                        if q != 上次疑问:
                            说("疑问", "【回响】%s" % q, [账号])
                            最近疑问 = q
                            上次疑问 = q
                            桥.思考队列.append(q)
                elif sim < 0.90:
                    q = "拍%d的画面和现在很不一样——这中间发生了什么？" % 旧拍
                    if q != 上次疑问:
                        说("疑问", "【回响】%s" % q, [账号])
                        最近疑问 = q
                        上次疑问 = q
                        桥.思考队列.append(q)
                else:
                    说("假设", "【回响】拍%d与现在似而不同（相似%.2f）——页面在演化。"
                       % (旧拍, sim), [账号])
            回响池.append((拍, 帧))

            # ---- 行（操作流）：按键课 v6——经验驱动，不同位置用学到的键 ----
            k = 拍 % 10
            if 验证挂起 > 0:                      # 人机验证期：停手等待所有者
                验证挂起 -= 1
                动作 = "停手(等待所有者处理人机验证)"
            elif k in (3, 7, 8):                  # 智能键：经验选键（左/双/右/中/拖选）
                x = 400 + (拍 * 211) % 1100
                y = 200 + (拍 * 149) % 600
                格位 = 九宫格(x, y)
                键 = 选键(格位, 拍)
                执行键(键, 拍)
                待评定 = 格位 + (键,)
                动作 = "按键[%s]@(%d,%d)" % (键, x, y)
                if 拍 % 30 == 0 and 键效果:       # 按键课自言：汇报学到的心得
                    心得s = []
                    for (gr, gc, 键名), 效果s in list(键效果.items())[:9]:
                        if 效果s and sum(效果s) / len(效果s) > 0.02:
                            心得s.append("(%d,%d)格用%s灵" % (gr, gc, 键名))
                    if 心得s:
                        说("复盘", "【按键课】我学到了：%s。" %
                           "；".join(心得s[:3]), [账号])
            elif k == 1:
                pyautogui.scroll(-4)
                动作 = "滚轮下"
            elif k == 2:
                x = 960 + (拍 * 37) % 400 - 200
                y = 540 + (拍 * 53) % 300 - 150
                pyautogui.moveTo(x, y, duration=0.15)
                动作 = "移动(%d,%d)" % (x, y)
            elif k == 4:
                文 = 读页文本()                      # 读页：看见页面内容
                字数 = len(文)
                新学 = 0
                if 字数 > 40:
                    低文 = 文.lower()
                    # 搜索失败页识别（切断『疑问→错误页→新疑问』的繁殖链）
                    if ("windows 找不到" in 低文 or "请检查拼写" in 低文
                            or "无法找到该页" in 低文):
                        搜索失败连 += 1
                        说("复盘", "【搜索】这次搜索落到了错误页——我提炼的词"
                           "不行，换个说法，先回课程。", [账号])
                        搜索失败连触发 = 搜索失败连 >= 2
                    else:
                        搜索失败连 = 0
                        搜索失败连触发 = False
                    # 人机验证感知（系统文本通道）：检测到即停手求助，不尝试绕过
                    特征词s = [w for w in ("人机验证", "安全验证", "验证码",
                                           "captcha", "滑动验证", "拖动滑块",
                                           "真人验证", "行为验证", "安全组件",
                                           "complete verification") if w in 低文]
                    if 特征词s and 验证挂起 <= 0:
                        验证挂起 = 60
                        账号3 = len(桥.账本)
                        桥.账本.append(("自我", "人机验证特征：%s" %
                                        "、".join(特征词s[:3])))
                        说("复盘", "【验证】页面出现人机验证（%s）——这是区分"
                           "人与机器的关卡，我不绕过；停手，等待所有者。"
                           % "、".join(特征词s[:3]), [账号3])
                        记("【需所有者】人机验证：%s" % "、".join(特征词s[:3]))
                        说出口("主人，遇到人机验证了，我来不了，需要你。")
                    if not 搜索失败连触发:
                        for 句 in 文.replace("\r", "").split("\n"):
                            句 = 句.strip()
                            if 6 < len(句) <= 120 and 新学 < 5 and 舱.命中(句) < 0.5:
                                舱.存行(句)
                                新学 += 1
                        账号2 = len(桥.账本)
                        桥.账本.append(("教材", "读页 拍=%d %d 字 新学 %d 句" %
                                        (拍, 字数, 新学)))
                        if 新学:
                            说("自评", "【读页】读到 %d 字，新学 %d 句——内容进舱了。"
                               % (字数, 新学), [账号2])
                else:
                    说("复盘", "【读页】这次只读到 %d 字——页面可能没有正文。" % 字数,
                       [len(桥.账本) - 1])
                动作 = "读页(%d字)" % 字数
            elif k == 5:
                if 最近疑问:                          # 他自己的疑问驱动搜索（先提炼成关键词）
                    词 = 疑问转搜索词(最近疑问)
                    搜索疑问(词)
                    动作 = "输入搜索[%s]" % 词
                    最近疑问 = None
                else:
                    pyautogui.scroll(3)
                    动作 = "滚动上"
            elif k == 9:                              # 键盘输入（英数，探索）
                pyautogui.write("python java learn 123", interval=0.04)
                pyautogui.press("esc")
                动作 = "键盘输入"
            else:
                动作 = "观察"

            # #285 课程轮换：每 80 拍换下一课（GitHub/Gitee/阿里云/剪辑），搜索连续失败也切课
            if 拍 % 80 == 0 or 搜索失败连 >= 2:
                课名, 课址 = 课程表[0]
                课程表.rotate(-1)
                导航(课址)
                搜索失败连 = 0
                动作 = "换课[%s]" % 课名
                说("假设", "【课程】我去学《%s》了—— Programming、开源、云、剪辑，"
                   "一样一样来。" % 课名, [账号])

            记("拍%4d %s ｜ 变化度 %.3f ｜ 延迟 %.2fs"
               % (拍, 动作, diff, time.time() - t0))
            time.sleep(max(0.0, 0.5 - (time.time() - t0)))   # 实时节律 ~2Hz
        except KeyboardInterrupt:
            raise
        except Exception as e:
            记("异常 %s: %s" % (type(e).__name__, str(e)[:120]))
            time.sleep(1)


if __name__ == "__main__":
    主(sys.argv[1] if len(sys.argv) > 1 else "https://cn.bing.com")
