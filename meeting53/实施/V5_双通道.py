# -*- coding: utf-8 -*-
"""meeting53/实施/V5_双通道.py —— ★双通道自我意识体原型（P1 主正文 / P2 旁路由 / P3 内心 / P4 多脑会议）。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
所有者要的（2026-10-17）：
 ① 「输出要和大模型（对标其最终效果，不解剖其内部）一样的正文内容，可以被可观测」；
 ② 「旁路由是还可以边输出正文边操作电脑、边操作接口应用……内部还会做深度思考」；
 ③ 「不同组人脑之间的不同协作，需要 agent 开会商议」。

本件只做**最小可判闭环**，不作「已完成」宣称：
 · P1 主正文：每拍由**自身状态**生成 2–3 个词组成的一句话（词音外采·词义内生；
   词的用法＝它在自己状态里被听见的统计，不是预写句池）；可只读输出流复读。
 · P2 旁路由：同一拍里同时产出**键鼠原语动作**（左键/右键/中键/滚轮/键入），
   动作由**同一内部态**选出；沙盒接口的**结果回流**进下一拍输入。
 · P3 内心：内部潜态迭代（`迭代` 算子）**不进正文流**；其终态以 γ 混入读出，
   故替换/扰动/冻结必须改变正文或动作（可失败）。
 · P4 多脑：3 个自态体各看一部分观测，**会议消息**作为彼此的输入 token；
   切断消息 ⇒ 联合任务掉到机会水平；归属门（`注入外源规则`）挡外源。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
铁律对齐：
 C1  零全局归一指数加权/零跨维注意力：只用既有 8 算子（含 `迭代` 逐元素收缩）＋硬 argmax 读出。
 C13 确定性：显式种子；`_归` 2^-20 定点；不 import time；两趟 sha256 逐位同。
 C14 不破既有：**只读 import** `AHI4/meeting49/实施/算子_两区室核.py`；本件纯新增。
 C10 无外挂模型：全部 numpy + 本仓库算子。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
诚实边界（写进代码，不靠事后解释）：
 · D=64 玩具规模；词表 24 词；任务为 5 键沙盒与 3 脑组合任务。
 · 「正文像主流大模型」只主张**可观测、连续、状态依赖、经历依赖**四条，不主张语言能力。
 · 「人脑复刻」**不主张**；只把关键步骤映射到可测子集。
 · 「无限深度」**不主张**：内心步数上限 4，且是否带来收益由 `判据_V5_双通道.py` 的
   自适应 vs 固定步数对照臂判——若没收益就如实记「构造性/无收益」。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import deque

import numpy as np

# ───────────────────────── 路径自愈 + 只读 import 既有核（C14）─────────────────────────
_本目录 = os.path.dirname(os.path.abspath(__file__))
_AHI4 = os.path.dirname(os.path.dirname(_本目录))
_核目录 = os.path.join(_AHI4, "meeting49", "实施")
if _核目录 not in sys.path:
    sys.path.insert(0, _核目录)

import 算子_两区室核 as _核模块  # noqa: E402  ★只读 import；不修改它

两区室核 = _核模块.两区室核
软限 = _核模块.软限
_归 = _核模块._归
界幅 = _核模块.界幅
生成码本 = _核模块.生成码本
禁词自检 = _核模块._m.禁词自检

# ───────────────────────── 冻结常量（判据不据结果改）─────────────────────────
D默认 = 64
种子默认 = 53001
词表 = ("我", "你", "它", "看", "听", "想", "动", "手", "开", "关",
        "好", "坏", "累", "紧", "松", "左", "右", "上", "下", "前",
        "后", "成", "败", "这")
词id = {w: i for i, w in enumerate(词表)}
M词 = len(词表)
动作表 = ("左键", "右键", "中键", "滚轮", "键入")   # 键鼠两原语的上层动作口（沙盒映射到 5 个按钮）
动作id = {a: i for i, a in enumerate(动作表)}
A动作 = len(动作表)
侧词 = ("左", "右", "上", "下", "前")
终止词 = {"左", "右", "上", "下", "前", "好", "坏", "成", "败", "累", "紧", "松", "手"}

# 符号 token 占位（都落在码本 0..63 内；不越界）
观测基 = 0        # 0..4  观测（5 个 cue）
符号_成 = 5
符号_败 = 6
动作基 = 7        # 7..11 上一动作
词基 = 12         # 12..35 词 token

# 读出超参（冻结；判据不据结果改）
γ内心 = 0.55      # 内心终态混入读出的权重
w结果 = 0.2       # 「刚成/刚败」绑定进感觉的权重
w动作 = 0.2       # 「刚做过什么」绑定进感觉的权重
w内词 = 0.9       # 内心词义绑定在正文读出里的权重
β文法 = 0.85      # 听见序列二元统计的权重
δ句首 = 0.60      # 听见序列「句首词」先验权重
η词 = 0.02        # 词义绑定（状态→词）的运行均值步长
η动 = 0.05        # 动作读出的线性 TD(0) 步长（单脑）
η动多脑 = 0.10    # 多脑共享读出的 TD 步长（多脑任务需要更快收敛；预先声明）
探索初 = 0.35
探索末 = 0.20


def 老师句(观测id: int, 上一奖励, 拍: int):
    """教师（外部）说的「听见序列」——注意：这是**输入**，不是系统输出表。

    偶数拍描述看到的（"我看左"），奇数拍描述刚发生的结果（"这好"/"这坏"）。
    """
    if int(拍) % 2 == 0:
        return ["我", "看", 侧词[int(观测id) % len(侧词)]] if int(观测id) < len(侧词) else ["我", "看", 侧词[0]]
    if 上一奖励 is None:
        return ["我", "看", 侧词[int(观测id) % len(侧词)]]
    return ["这", "好"] if int(上一奖励) == 1 else ["这", "坏"]


def 句池_固定(拍: int):
    """★must-fail 对照：meeting41 `输出流.py` 式预写句池（固定轮转）。仅用于证伪臂。"""
    _池 = ("这一步先稳住，落点对上再往下走。",
           "把目标框住，手上这点活不耽误。",
           "先看当前这一屏，抓最靠上的那一项。",
           "顺序别乱：先定位，再动手，后核对。",
           "动作和话对齐，回头一看就能对上号。")
    return _池[int(拍) % len(_池)]


class 双通道体:
    """一个自态体：既有两区室核 + 内生词义绑定 + 动作读出 + 内心迭代 + 会议消息入 token。"""

    def __init__(self, D: int = D默认, 种子: int = 种子默认, 名字: str = "我"):
        self.D = int(D)
        self.种子 = int(种子)
        self.名字 = str(名字)
        self.核 = 两区室核(D=self.D, 种子=self.种子)
        self.码本 = self.核.码本                      # 64 × D（±1）
        self.上下文 = None                              # 核的递推上下文
        self.W词 = _归(np.zeros((M词, self.D)))        # 词义绑定（状态运行均值）
        self.W词_内 = _归(np.zeros((M词, self.D)))     # 词义绑定（内心运行均值）
        self.W动 = _归(np.zeros((A动作, self.D)))      # 动作读出
        self.b动 = np.zeros(A动作, dtype=np.float64)
        self.C2 = np.zeros((M词, M词), dtype=np.float64)   # 听见序列二元统计
        self.C首 = np.zeros(M词, dtype=np.float64)         # 句首统计
        self.上一词 = None
        self.上动作 = None
        self.上一奖励 = None
        self.自我标量 = 0.5
        self.o = np.zeros(self.D, dtype=np.float64)
        self.o用 = np.zeros(self.D, dtype=np.float64)
        self.内心 = np.zeros(self.D, dtype=np.float64)
        self.内心_上一拍 = np.zeros(self.D, dtype=np.float64)
        self.内心步数 = 1
        self.惊喜 = 0.0
        self.奖EWMA = 0.5
        self.历史o = deque(maxlen=8)
        self.历史词 = deque(maxlen=4)
        self.rng = np.random.default_rng(self.种子 + 7001)
        self.rng2 = np.random.default_rng(self.种子 + 9001)
        self.台账 = dict(拍数=0, 词数=0, 动作数=0, 内心总步数=0,
                         词绑定更新=0, 动作更新=0, 会议消息数=0)

    # ─────────────── token 构造：观测 / 结果 / 上动作 / 会议消息 / 自身上词 ───────────────
    def _token列(self, 观测id: int, 消息id列, 上词id列):
        t = [self.码本[观测基 + int(观测id) % 5]]
        t.append(self.码本[符号_成 if int(self.上一奖励 or 0) == 1 else 符号_败])
        if self.上动作 is not None:
            t.append(self.码本[动作基 + int(self.上动作)])
        for m in (消息id列 or []):
            t.append(self.码本[观测基 + int(m) % 5])
        for w in (上词id列 or [])[-2:]:
            t.append(self.码本[词基 + int(w)])
        return t

    # ─────────────── 一拍：读状态 → 内心 → 正文 → 动作 ───────────────
    def 一拍(self, 观测id: int, 消息id列=None, 学习: bool = True,
             老师: bool = True, 拍: int = 0, 句池臂: bool = False,
             固定内心: int = 0, 探索率=None, 内心覆盖=None, 状态覆盖=None):
        self.台账["拍数"] += 1
        self._拍 = int(拍)
        self.台账["会议消息数"] += len(消息id列 or [])
        r = self.核.前向_双区室(self._token列(观测id, 消息id列, list(self.历史词)[-2:]),
                          上下文=self.上下文, 迭代步数=1)
        self.上下文 = r["终态"]
        B列 = r["世界态"]
        # ★感觉绑定：用既有算子里最基础的 `绑`（键=符号，值=自我态）把「看到/听到/刚做了什么」
        #   逐元素绑进状态（无跨维混合、无全局归一）。这是 P1 状态依赖的物理入口。
        A = r["自我态"]
        self.观测绑定 = self.核.绑(self.码本[观测基 + int(观测id) % 5], A)
        self.消息绑定 = _归(sum((self.核.绑(self.码本[观测基 + int(m) % 5], A)
                              for m in (消息id列 or [])), np.zeros(self.D)))
        感觉 = _归(
            self.观测绑定 + self.消息绑定
            + w结果 * self.核.绑(self.码本[符号_成 if int(self.上一奖励 or 0) == 1 else 符号_败], A)
            + (w动作 * self.核.绑(self.码本[动作基 + int(self.上动作)], A) if self.上动作 is not None
               else np.zeros(self.D))
        )
        self.o = _归(感觉 + 0.30 * _归(B列[-1]))
        self.自我标量 = float(r["自我标量"])

        # ── P3 内心：按「惊讶 + 新异」调度步数（1..4），迭代不进正文 ──
        新异 = 0.0 if not self.历史o else float(np.mean(np.abs(self.o - self.历史o[-1])) / 界幅)
        惊讶 = abs(float(self.上一奖励 or 0) - self.奖EWMA)
        self.惊喜 = float(惊讶)
        self.内心步数 = 1 + int(round(3.0 * (0.5 * 惊讶 + 0.5 * 新异)))
        self.内心步数 = max(1, min(4, self.内心步数))
        if 固定内心 > 0:
            self.内心步数 = int(固定内心)
        x = self.内心_上一拍.copy()               # 想 = 接着上一拍的思路往下想（跨拍连续），用重合门当驱动
        for _ in range(self.内心步数):
            x = self.核.迭代(x, r["门"][-1], 1)
        _范x = float(np.linalg.norm(x)) + 1e-9
        _范o = float(np.linalg.norm(self.o)) + 1e-9
        x = _归(x * (_范o * 0.8 / _范x))             # 思考强度与当前状态同量级（方向来自迭代）
        if 内心覆盖 == "冻结":
            x = np.zeros(self.D)          # 冻结 = 关掉思考（不是把首拍值冻住）
        elif 内心覆盖 == "替换":
            x = _归(self.rng2.standard_normal(self.D) * 4.0)
        elif 内心覆盖 == "扰动":
            x = _归(x + self.rng2.standard_normal(self.D) * 4.0)
        self.内心_上一拍 = x.copy()
        self.内心 = _归(x)
        self.o用 = _归(软限(self.o + γ内心 * self.内心))
        self.o动 = _归(self.观测绑定 + self.消息绑定)     # 动作只用「感知绑定」——思考不直接改命（诚实：见判据 T3 双臂）
        if 状态覆盖 is not None:
            self.o用 = _归(状态覆盖)
        self.历史o.append(self.o.copy())
        self.台账["内心总步数"] += self.内心步数

        # ── P1 正文：状态 → 词（内生绑定 + 听见序列统计）──
        if 句池臂:
            词列 = [句池_固定(拍)]
        else:
            词列 = self._生成词(self.o用)
        if 句池臂:
            self.历史词.append(0)
        else:
            try:
                self.历史词.append(int(词列[0]) if 词列 else 0)
            except (TypeError, ValueError):
                self.历史词.append(0)
        self.台账["词数"] += len(词列)

        # ── P2 旁路由：同一内部态 → 键鼠原语动作 ──
        if 探索率 is None:
            探 = 探索初 + (探索末 - 探索初) * min(1.0, self.台账["拍数"] / 300.0)
        else:
            探 = float(探索率)
        动作 = self._选动作(self.o动, 探)
        self.台账["动作数"] += 1

        # ── 学习：词义（听见序列）与动作（结果）──
        if 学习 and 老师:
            for w in 老师句(观测id, self.上一奖励, 拍):
                wi = 词id[w]
                self.W词[wi] = _归((1.0 - η词) * self.W词[wi] + η词 * self.o用 / 界幅)
                self.W词_内[wi] = _归((1.0 - η词) * self.W词_内[wi] + η词 * self.内心 / 界幅)
                if self.上一词 is not None:
                    self.C2[self.上一词, wi] += 1.0
                else:
                    self.C首[wi] += 1.0
                self.上一词 = wi
                self.台账["词绑定更新"] += 1
        else:
            self.上一词 = None

        self.上动作 = int(动作)
        _安全词列 = [(词表[int(w)] if isinstance(w, (int, np.integer)) else str(w)) for w in 词列]
        return dict(词列=_安全词列,
                    动作=动作表[int(动作)], 动作id=int(动作),
                    内心步数=int(self.内心步数), 惊喜=float(self.惊喜),
                    自我标量=float(self.自我标量))

    # ─────────────── 生成词（硬 argmax；无跨维混合；无全局归一指数加权）───────────────
    def _生成词(self, o用, 内心=None):
        u = _归(o用 / (float(np.max(np.abs(o用))) + 1e-9))
        范 = np.linalg.norm(self.W词, axis=1) + 1e-9
        sc = (self.W词 @ u) / 范                    # 余弦：去掉词行范数差异
        _内 = self.内心 if 内心 is None else _归(内心)
        内u = _归(_内 / (float(np.max(np.abs(_内))) + 1e-9))
        范内 = np.linalg.norm(self.W词_内, axis=1) + 1e-9
        sc = sc + w内词 * ((self.W词_内 @ 内u) / 范内)  # 「想到的词」：内心也有自己的词义绑定
        sc = sc - float(np.mean(sc))                 # 去公共方向
        尺 = float(np.max(np.abs(sc))) + 1e-9
        sc = sc / 尺
        词序 = []
        上 = None
        for 步 in range(4):
            if 上 is None:
                分 = sc + δ句首 * (self.C首 / (self.C首.max() + 1e-9))
            else:
                bg = self.C2[上]
                if float(bg.sum()) <= 0.0:
                    break
                分 = β文法 * sc + (bg / (bg.max() + 1e-9))
            w = int(np.argmax(分))
            if 上 is not None and float(分[w]) <= 0.05:
                break
            词序.append(w)
            上 = w
            if 词表[w] in 终止词:
                break
        if not 词序:
            词序 = [0]
        return 词序

    def _选动作(self, o用, 探索率: float):
        u = _归(o用 / (float(np.max(np.abs(o用))) + 1e-9))
        分 = self.W动 @ u + self.b动
        if 探索率 > 0.0 and float(self.rng.random()) < float(探索率):
            return int(self.rng.integers(0, A动作))
        return int(np.argmax(分))

    def 更新(self, 奖励: int, 动作id: int, 学习动作: bool = True):
        """动作学习：成功把该动作的读出拉向本次状态，失败轻推离；自我态走核的内生 credit。"""
        奖励 = int(奖励)
        u = _归(self.o用 / (float(np.max(np.abs(self.o用))) + 1e-9))
        if 学习动作:
            # 线性 TD(0)：Q(a)=W动[a]·u；误差 δ=r−Q ⇒ 从奖赏里长出「看到 X ⇒ 按 Y」的程序性记忆
            q = float(self.W动[动作id] @ u)
            δ = float(奖励) - q
            self.W动[动作id] = _归(self.W动[动作id] + η动 * δ * u)
            self.台账["动作更新"] += 1
        self.奖EWMA = 0.95 * self.奖EWMA + 0.05 * 奖励
        self.上一奖励 = 奖励
        # 核的内生 credit（先补 最近门/_最近B，再调既有学自我，不改既有件）
        try:
            self.核.最近门 = _归(self.核.符(self.核.自我态, self.o))
            self.核._最近B = self.o
            self.核.学自我(1 if 奖励 == 1 else 0, 奖励)
            self.自我标量 = float(self.核.自我标量)
        except Exception:
            pass

    def 特征指纹(self) -> str:
        h = hashlib.sha256()
        h.update(np.asarray(self.W词).tobytes())
        h.update(np.asarray(self.W词_内).tobytes())
        h.update(np.asarray(self.W动).tobytes())
        h.update(np.asarray(self.C2).tobytes())
        h.update(np.asarray(self.自我标量).tobytes())
        return h.hexdigest()[:32]


class 接口沙盒:
    """旁路由的目标世界：一个 5 按钮的「接口应用」。观测＝当前 cue，动作＝键鼠原语，结果回流。"""

    def __init__(self, 种子: int = 种子默认):
        self.rng = np.random.default_rng(int(种子) + 31)
        self.cue = 0
        self.目标 = 0
        self.上一成功 = None
        self.历史 = deque(maxlen=8)

    def 新试次(self):
        self.cue = int(self.rng.integers(0, 5))
        self.目标 = self.cue
        self.历史.append(self.cue)

    def 走(self, 动作id: int):
        成功 = int(int(动作id) == int(self.目标))
        self.上一成功 = 成功
        return 成功


class 多脑沙盒:
    """3 脑组合任务：每脑只见一部分观测，目标动作 = (o0+o1+o2) mod 5；必须靠会议消息组合。"""

    def __init__(self, 种子: int = 种子默认):
        self.rng = np.random.default_rng(int(种子) + 53)
        self.观测 = [0, 0, 0]
        self.目标 = 0
        self.上一成功 = None

    def 新试次(self):
        # 每脑 1 bit（0/1）；目标动作 = 三脑观测之和（0..3）⇒ 单脑看不到全因（机会 1/5）
        self.观测 = [int(self.rng.integers(0, 2)) for _ in range(3)]
        self.目标 = int(sum(self.观测)) % 5
        return list(self.观测)

    def 走(self, 动作id: int):
        成功 = int(int(动作id) == int(self.目标))
        self.上一成功 = 成功
        return 成功


def 跑单脑(拍数: int = 600, 测拍: int = 200, D: int = D默认, 种子: int = 种子默认,
           句池臂: bool = False, 固定内心: int = 0, 断观测: bool = False,
           学习动作: bool = True, 老师: bool = True, 内心覆盖=None, 测探索率: float = 0.0):
    """单脑：主正文 + 旁路由 + 内心。返回轨迹与指标。"""
    体 = 双通道体(D=D, 种子=种子, 名字="我")
    环 = 接口沙盒(种子=种子 + 1)
    训 = []
    for t in range(int(拍数)):
        环.新试次()
        观测 = 环.cue if not 断观测 else 0
        r = 体.一拍(观测id=观测, 消息id列=None, 学习=True, 老师=老师, 拍=t,
                    句池臂=句池臂, 固定内心=固定内心, 内心覆盖=内心覆盖)
        成功 = 环.走(r["动作id"])
        体.更新(成功, r["动作id"], 学习动作=学习动作)
        训.append(dict(拍=t, 词=r["词列"], 动作=r["动作"], 成功=成功,
                       内心步数=r["内心步数"], 惊喜=r["惊喜"], 自我标量=r["自我标量"]))
    测 = []
    for t in range(int(测拍)):
        环.新试次()
        观测 = 环.cue if not 断观测 else 0
        r = 体.一拍(观测id=观测, 消息id列=None, 学习=False, 老师=False, 拍=t,
                    句池臂=句池臂, 固定内心=固定内心, 内心覆盖=内心覆盖,
                    探索率=float(测探索率))
        成功 = 环.走(r["动作id"])
        体.更新(成功, r["动作id"], 学习动作=False)
        测.append(dict(拍=t, 词=r["词列"], 动作=r["动作"], 成功=成功,
                       内心步数=r["内心步数"], 惊喜=r["惊喜"], 自我标量=r["自我标量"],
                       观测=观测, o用=体.o用.copy(), 状态=体.o.copy()))
    return dict(体=体, 环=环, 训=训, 测=测,
                训练成功率=float(np.mean([x["成功"] for x in 训])) if 训 else 0.0,
                测试成功率=float(np.mean([x["成功"] for x in 测])) if 测 else 0.0,
                测试后段成功率=float(np.mean([x["成功"] for x in 测[-100:]])) if 测 else 0.0,
                指纹=体.特征指纹())


def 跑多脑(拍数: int = 600, 测拍: int = 200, D: int = D默认, 种子: int = 种子默认,
           通信: bool = True, 全观测单脑: bool = False, 部分观测单脑: bool = False):
    """3 脑会议（或对照臂）：联合动作、联合奖励；消息＝伙伴观测 token（感知共享）。

    对照臂：通信=False（各脑只听自己）／全观测单脑=True（一脑看全三路观测）／
            部分观测单脑=True（一脑只看一路 + 3 倍维数，用来看「多算力能否替代通信」）。
    ⚠ 诚实边界：这是**感知事件共享**，不是语言报数；本件不主张「会议语言」已成立。
    """
    种子 = int(种子)
    体数 = 1 if (全观测单脑 or 部分观测单脑) else 3
    体列 = [双通道体(D=D, 种子=种子 + 100 * i, 名字="脑%d" % i) for i in range(体数)]
    环 = 多脑沙盒(种子=种子 + 2)
    动读出 = _归(np.zeros((A动作, D)))
    rng = np.random.default_rng(种子 + 777)

    def _拍一次(全观测列, 学习, 拍):
        o列, 记录, u列 = [], [], []
        for i, 体 in enumerate(体列):
            if 全观测单脑:
                观测i, 消息 = 全观测列[0], list(全观测列[1:])   # 自己的观测只出现一次
            elif 部分观测单脑:
                观测i, 消息 = 全观测列[0], []
            else:
                观测i = 全观测列[i]
                消息 = [全观测列[j] for j in range(3) if j != i] if 通信 else []
            r = 体.一拍(观测id=观测i, 消息id列=消息, 学习=学习, 老师=True, 拍=拍)
            o列.append(体.o用.copy())
            _u = _归(体.o动 / (float(np.max(np.abs(体.o动))) + 1e-9))
            u列.append(_u)
            记录.append(dict(脑=i, 词="".join(r["词列"]), 动作=r["动作"],
                            内心步数=r["内心步数"], 消息=list(消息), 观测=int(观测i)))
        # ★三脑各出一票：联合分数 = 各脑用自己的状态算出的动作分数的平均（脑1/2 真的参与决策）
        分 = np.mean([动读出 @ u for u in u列], axis=0)
        if 学习:
            探 = 0.35 + (0.20 - 0.35) * min(1.0, (拍 + 1) / 300.0)
            用探索 = bool(float(rng.random()) < 探)
        else:
            用探索 = False
        动作 = int(rng.integers(0, A动作)) if 用探索 else int(np.argmax(分))
        for 体 in 体列:
            体.上动作 = 动作
        return 动作, u列, o列, 记录

    def _学(动作, 成功, u列, o列):
        # 三脑各出一次误差，共享读出按三脑平均更新（脑1/2 的消息进入各自状态，故真的参与）
        for _u in u列:
            q = float(动读出[动作] @ _u)
            δ = float(成功) - q
            动读出[动作] = _归(动读出[动作] + η动多脑 * δ * _u / max(1, len(u列)))
        for 体 in 体列:
            体.更新(成功, 动作, 学习动作=False)
        for 体 in 体列:
            体.W动 = 动读出.copy()          # 共享读出（便于判据读同一份）

    训, 测 = [], []
    for t in range(int(拍数)):
        全观测列 = 环.新试次()
        动作, u, o列, 记录 = _拍一次(全观测列, True, t)
        成功 = 环.走(动作)
        _学(动作, 成功, u, o列)
        训.append(dict(拍=t, 动作=动作表[动作], 动作id=动作, 成功=成功,
                       记录=记录, 自我标量=[float(b.自我标量) for b in 体列]))
    for t in range(int(测拍)):
        全观测列 = 环.新试次()
        动作, u, o列, 记录 = _拍一次(全观测列, False, t)
        成功 = 环.走(动作)
        测.append(dict(拍=t, 动作=动作表[动作], 动作id=动作, 成功=成功, 记录=记录))
    return dict(体列=体列, 环=环, 训=训, 测=测, 动读出=动读出,
                训练成功率=float(np.mean([x["成功"] for x in 训])) if 训 else 0.0,
                测试成功率=float(np.mean([x["成功"] for x in 测])) if 测 else 0.0,
                测试后段成功率=float(np.mean([x["成功"] for x in 测[-100:]])) if 测 else 0.0,
                通信=bool(通信), 体数=int(体数),
                全观测=bool(全观测单脑), 部分观测=bool(部分观测单脑))


# ═══════════════════════════════════ 演示直跑 ═══════════════════════════════════
def 演示(拍数: int = 12, 训练拍: int = 400) -> dict:
    跑 = 跑单脑(拍数=训练拍, 测拍=拍数, 种子=种子默认)
    行 = []
    for x in 跑["测"]:
        行.append("".join(x["词"]) + " ｜" + x["动作"])
    return dict(训练成功率=跑["训练成功率"], 测试成功率=跑["测试成功率"], 正文=" ".join(行))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    np.seterr(all="ignore")
    果 = 演示()
    print("训练成功率=%.3f 测试成功率=%.3f" % (果["训练成功率"], 果["测试成功率"]))
    print("正文（测 12 拍）：")
    print(果["正文"])
    print("C1 自检（本文件）：", json.dumps(禁词自检(__file__), ensure_ascii=False))
