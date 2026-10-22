# -*- coding: utf-8 -*-
"""meeting38/共存态.py —— AHI 共享意识工作区（唯一真相源 · 五路并发读写）

━━━ 红线一（架构）━━━
本工作区是 AHI 的唯一"意识状态"。眼·耳·脑·手·嘴五路器官**各自独立循环**
（各占一条线程/进程），只经本工作区读写耦合：
  · 任何器官不得直接调用、不得等待另一器官（不许 join / 不许 call）。
  · 跨模态依赖（手执行→屏幕变→眼看见→脑改意图→手**在移动途中**改轨迹）
    由"每拍读对方写的**最新值**"自然涌现，而不是靠顺序调用。
  · 由此，脑子在想、嘴在说、手在动、眼睛在盯屏 —— 同一时刻并行发生。

━━━ 红线二（非 Transformer）━━━
本文件不引入任何 Transformer / attention / QKV / softmax 结构，无梯度、无反向传播。
语义与特征一律用 **HDC（高维计算）双极超向量**：随机投影（固定不训练）+ 字袋编码
+ 按位绑定/捆绑 + 余弦相似度。这是全自主的、与 Transformer 无关的表示与推理。

━━━ 槽位（每槽一把独立锁：单路写不阻塞他路读）━━━
  视觉槽 元素表/帧号/变化率/注意力焦点           ← 眼写，脑/手/嘴读
  听觉槽 声事件流/最近起音                       ← 耳写，脑读
  意图槽 目标栈/当前意图向量/强度/来源            ← 脑写，手读
  动作槽 手状态/轨迹/进度/光标/句柄/反馈           ← 手写，眼/脑读（手的反馈回流）
  表达槽 待说队列/最近一句/说话计数               ← 脑写，嘴读
  语义槽 按钮 affordance 图/容器树/叠加优先级      ← 语义层写，脑/手读
"""
from __future__ import annotations

import threading
import time

from collections import deque

import numpy as np

D默认 = 1024                # HDC 超向量维度（结构常量，非先验自由度）


# ══════════════════════ 后端：GPU 优先，CPU 降级（不阻塞主线程靠架构，不靠算力） ══════════════════════
try:
    import torch as _torch
    _有torch = True
except Exception:                                    # pragma: no cover
    _torch = None
    _有torch = False


def 设备名() -> str:
    if _有torch and _torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ══════════════════════ HDC 编码器（非 Transformer） ══════════════════════

class 编码器:
    """HDC 双极编码器：字 → 字码，文本 → 字袋码，任意特征 → 随机投影码。
    码本随机生成后**固定不训练**（无梯度、无 attention）；相似度=归一化点积。
    码本常驻 GPU（若可用），批量匹配在 GPU 上做——CPU 不做向量活。"""

    def __init__(self, 维度: int = D默认, 种子: int = 3800, 用gpu: bool = True):
        self.D = int(维度)
        self.用gpu = bool(用gpu) and 设备名() == "cuda"
        self._dev = "cuda" if self.用gpu else "cpu"
        self._rng = np.random.default_rng(int(种子))
        self._字索引: dict = {}
        self._字码本 = None                          # (N,D) 双极，住 GPU/CPU
        self._特征投影: dict = {}                    # 原始维度 → (d,D) 投影阵
        self._零 = np.zeros(self.D, dtype=np.float32)

    # ── 码本管理（随手追加，常驻） ──
    def _新码(self, n: int):
        v = np.sign(self._rng.standard_normal((n, self.D))).astype(np.float32)
        v[v == 0] = 1.0
        return v

    def _追加码本(self, 新码: np.ndarray):
        if self.用gpu:
            t = _torch.as_tensor(新码, device=self._dev)
            self._字码本 = t if self._字码本 is None else _torch.cat([self._字码本, t], 0)
        else:
            self._字码本 = 新码 if self._字码本 is None else \
                np.concatenate([self._字码本, 新码], 0)

    def _码本行(self, 行: np.ndarray) -> np.ndarray:
        if self.用gpu:
            return self._字码本[行].detach().cpu().numpy()
        return self._字码本[行]

    def 字码(self, 字: str) -> np.ndarray:
        i = self._字索引.get(字)
        if i is None:
            i = len(self._字索引)
            self._字索引[字] = i
            self._追加码本(self._新码(1))
        return self._码本行(i)

    def 文本码(self, 文本: str) -> np.ndarray:
        """字袋编码：逐字码的按位多数表决（bundle）。空串→零向量。"""
        码s = [self.字码(c) for c in str(文本)]
        if not 码s:
            return self._零.copy()
        M = np.stack(码s, 0)
        b = np.sign(M.sum(0)).astype(np.float32)
        b[b == 0] = 1.0
        return b

    def 特征码(self, 向量) -> np.ndarray:
        """任意实数特征 → 固定随机投影 → 双极码。投影阵按原始维度缓存，永不训练。"""
        v = np.asarray(向量, dtype=np.float32).ravel()
        key = int(v.shape[0])
        P = self._特征投影.get(key)
        if P is None:
            P = (self._rng.standard_normal((key, self.D))
                 / np.sqrt(max(key, 1))).astype(np.float32)
            self._特征投影[key] = P
        z = P.T @ v
        b = np.sign(z).astype(np.float32)
        b[b == 0] = 1.0
        return b

    # ── HDC 算子（非 Transformer 的"推理"） ──
    def 绑(self, a: np.ndarray, b: np.ndarray, 移位: int = 0) -> np.ndarray:
        """绑定：按位乘 ⊕ 循环移位（位置绑定）。用于"键=角色⊗填充"。"""
        a = np.asarray(a, dtype=np.float32).ravel()
        if 移位:
            a = np.roll(a, int(移位) % self.D)
        c = a * np.asarray(b, dtype=np.float32).ravel()
        c = np.sign(c).astype(np.float32)
        c[c == 0] = 1.0
        return c

    def 束(self, 码s) -> np.ndarray:
        """捆绑：多数表决（多路信息叠加到一个超向量）。"""
        码s = [np.asarray(c, dtype=np.float32).ravel() for c in 码s]
        if not 码s:
            return self._零.copy()
        b = np.sign(np.stack(码s, 0).sum(0)).astype(np.float32)
        b[b == 0] = 1.0
        return b

    def 相似(self, a, b) -> float:
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()
        na = float(np.linalg.norm(a)) or 1.0
        nb = float(np.linalg.norm(b)) or 1.0
        return float(a @ b / (na * nb))

    def 批相似(self, q, 码本) -> np.ndarray:
        """批量余弦（码本在 GPU 时在 GPU 算）。返回 (N,) float64。"""
        q = np.asarray(q, dtype=np.float32).ravel()
        if self.用gpu:
            tq = _torch.as_tensor(q, device=self._dev)
            t = _torch.as_tensor(码本, device=self._dev) if not isinstance(码本, _torch.Tensor) else 码本
            num = t @ tq
            den = t.norm(dim=1) * (tq.norm() + 1e-9)
            return (num / den.clamp_min(1e-9)).detach().cpu().numpy()
        M = np.asarray(码本, dtype=np.float32)
        num = M @ q
        den = (np.linalg.norm(M, axis=1) * (np.linalg.norm(q) + 1e-9)) + 1e-9
        return num / den


# ══════════════════════ 共享意识工作区 ══════════════════════

class 共存态:
    """五路器官共享的唯一状态。所有 get/set 都持独立细粒度锁，只做深拷贝级快照，
    锁内不做计算——保证任何一路写不会把另一路卡住。"""

    def __init__(self, 种子: int = 3800, 用gpu: bool = True):
        self.编码 = 编码器(种子=种子, 用gpu=用gpu)
        self.设备 = 设备名() if 用gpu else "cpu"
        self.t0 = time.time()

        self._拍锁 = threading.Lock()
        self._拍 = 0

        # ── 视觉槽（眼写；脑/手/嘴读） ──
        self._视觉锁 = threading.Lock()
        self._视觉 = dict(元素s=[], 帧号=0, 更新t=0.0, 变化率=0.0, 焦点=None, 屏幕=(0, 0))

        # ── 听觉槽（耳写；脑读） ──
        self._听觉锁 = threading.Lock()
        self._听觉 = dict(事件s=deque(maxlen=120), 最近起音t=0.0, 最近主频=0.0,
                          最近能量=0.0, 更新t=0.0)

        # ── 意图槽（脑写；手读） ──
        self._意图锁 = threading.Lock()
        self._意图 = dict(目标栈=[], 当前=None, 强度=0.0, 来源="", 版本=0)

        # ── ◆目标槽（多产者：教学/脑自生；脑读）——步7 追加（移动性偏置的目的端）──
        # 「我现在在找什么」：主色+参考心+名称（步8 教学命名）。
        # 空=无目标（既有探索行为）；非空=注意/出招被目标吸引（找到→多巴胺）。
        self._目标锁 = threading.Lock()
        self._目标 = dict(主色=None, 心=None, 句柄=None, 名称="", 来源="", 版本=0)

        # ── 动作槽（手写；眼/脑读 —— 手的反馈回流通道） ──
        self._动作锁 = threading.Lock()
        self._动作 = dict(状态="闲", 轨迹=None, 进度=0.0, 光标=(0, 0),
                          句柄=None, 目标=None, 反馈=[], 动作计数=0, 纠偏次数=0)

        # ── 表达槽（脑写；嘴读） ──
        self._表达锁 = threading.Lock()
        self._表达 = dict(队列=deque(maxlen=32), 最近="", 说数=0)

        # ── 语义槽（语义层写；脑/手读） ──
        self._语义锁 = threading.Lock()
        self._语义 = dict(按钮s={}, 词表=[], 容器树=[], 叠加={}, 点击裁决=0)

        # ── 度量 ──
        self._度量锁 = threading.Lock()
        self._度量 = dict(读写次数={}, 最长写ms=0.0)

        # ── ◆自我槽（自我线程单写；脑/嘴读）——步3 追加（meeting38 五专家 05 契约 §2.2）──
        self._自我锁 = threading.Lock()
        self._自我 = dict(提案=None, 遥测={}, 向量=None, 自言摘要="", 类型覆盖={},
                          审计={}, 自体拍=0, 更新t=0.0)

        # ── ◆事件槽（多生产者单消费者——手/眼/耳回流，自我线程消费） ──
        self._事件锁 = threading.Lock()
        self._事件 = deque(maxlen=256)

        # ── ◆计划槽（脑单写；嘴/自我线程读） ──
        self._计划锁 = threading.Lock()
        self._计划 = dict(当前=None, 待办=[], 已办=[], 说做对账={}, 版本=0)

        # ── ◆视场槽（眼单写；语义/脑/我读）——视觉会议步1 追加（纪要 §3.1，纯追加）──
        # 「看」的本体：连续保拓扑视觉场 + 注视状态 + 预测误差（非框列表）
        self._视场锁 = threading.Lock()
        self._视场 = dict(视场=None, 注视=None, 误差场=None, 意外度=0.0, 逐格意外=None,
                          眼跳数=0, 注视链=deque(maxlen=64), 更新t=0.0)

        # ── ◆思考槽（脑单写；演播台/嘴只读）——所有者 2026-10-05「要看得见思考过程」──
        # 原生思考链（看/想/招/预/果/路）：由脑器官每拍在**自己的循环里**产出的实义思考文本，
        # 经共享工作区流转。不是外部工具调用、不是外挂模型、不含任何 Transformer 结构——
        # 与眼/手/嘴同一通信范式（只读写槽位，互不调用）。
        self._思考锁 = threading.Lock()
        self._思考 = deque(maxlen=256)

        # ── ◆对话槽（文本态：界面写「用户说」/ 脑写「它说·它想」；文本流只读）──
        # 纯文本对话态（所有者 2026-10-06「第一步/第二步」）。与眼/手/嘴同一通信范式：
        # 只读写槽位、互不调用（原生支持式，非工具调用）。
        # 状态 ∈ 空闲|等待|回合；「醒」由生命自身内在状态决定（本槽只存结果，不设阈值）。
        #
        # 多会话（所有者 2026-10-06「同一生命·多窗口」）：
        #   一个"我"可同时与多个窗口说话——**如 QQ 多开**，但背后是同一个生命：
        #     · 会话表 = {会话ID: 该窗口的 用户箱/对话流/脑内/回合/它说/它想/状态}
        #       「对话流」按窗口隔离（A 窗口的话不会串到 B 窗口）；
        #     · **记忆不在本文件**——记忆=综合体 codec._段 + 容量桥，由全生命共享，
        #       故 A 窗口得知的事，B 窗口天然能想起（跨窗口共享，这正是"一个人"的证据）。
        #     · 「醒」是**生命级**（一个生命一条命），不随窗口分叉，故单列 self._醒。
        #     · 兼容：所有方法 会话 默认 "主" ⇒ 旧调用（不传会话）逐位不变（C14）。
        self._对话锁 = threading.Lock()
        self._会话表: dict = {"主": self._新会话槽()}
        self._醒 = 1                                   # 生命级：不随窗口分叉

    # ══════════ 会话表辅助（多窗口；对话槽用） ══════════
    @staticmethod
    def _新会话槽() -> dict:
        """一个窗口的全部对话状态。注意：**记忆不在此**（记忆=综合体/容量桥，全生命共享）。"""
        return dict(用户箱=deque(maxlen=64), 对话流=deque(maxlen=1024),
                    脑内=deque(maxlen=512),            # 内部思考通道（不入对话流，只供审计）
                    上次用户t=0.0, 回合=0, 它说="", 它想="", 状态="空闲")

    def _槽(self, 会话: str) -> dict:
        """取（或建）某窗口的状态槽。**调用方须持 _对话锁。**"""
        会话 = str(会话)
        d = self._会话表.get(会话)
        if d is None:
            d = self._新会话槽()
            self._会话表[会话] = d
        return d

    # ══════════ 拍钟 ══════════
    def 拍(self) -> int:
        with self._拍锁:
            self._拍 += 1
            return self._拍

    def 当前拍(self) -> int:
        with self._拍锁:
            return self._拍

    def 秒(self) -> float:
        return time.time() - self.t0

    def _计(self, 名: str, dt: float):
        with self._度量锁:
            self._度量["读写次数"][名] = self._度量["读写次数"].get(名, 0) + 1
            if dt > self._度量["最长写ms"]:
                self._度量["最长写ms"] = round(dt * 1000, 2)

    # ══════════ 视觉槽 ══════════
    def 写视觉(self, 元素s, 帧号=None, 变化率=None, 屏幕=None):
        t0 = time.perf_counter()
        with self._视觉锁:
            self._视觉["元素s"] = 元素s
            if 帧号 is not None:
                self._视觉["帧号"] = int(帧号)
            if 变化率 is not None:
                self._视觉["变化率"] = float(变化率)
            if 屏幕 is not None:
                self._视觉["屏幕"] = tuple(屏幕)
            self._视觉["更新t"] = time.time()
        self._计("写视觉", time.perf_counter() - t0)

    def 取视觉(self) -> dict:
        with self._视觉锁:
            v = dict(self._视觉)
            v["元素s"] = list(v["元素s"])
        return v

    def 置焦点(self, 句柄):
        with self._视觉锁:
            self._视觉["焦点"] = 句柄

    def 取焦点(self):
        with self._视觉锁:
            return self._视觉["焦点"]

    def 找元素(self, 句柄) -> dict | None:
        """按句柄取元素最新位置——手每拍调用它做**中途纠偏**。"""
        if 句柄 is None:
            return None
        with self._视觉锁:
            for e in self._视觉["元素s"]:
                if e.get("句柄") == 句柄:
                    return dict(e)
        return None

    # ══════════ 听觉槽 ══════════
    def 写听觉(self, 事件=None, 主频=None, 能量=None, 起音=False):
        t0 = time.perf_counter()
        with self._听觉锁:
            if 事件 is not None:
                self._听觉["事件s"].append(事件)
            if 主频 is not None:
                self._听觉["最近主频"] = float(主频)
            if 能量 is not None:
                self._听觉["最近能量"] = float(能量)
            if 起音:
                self._听觉["最近起音t"] = time.time()
            self._听觉["更新t"] = time.time()
        self._计("写听觉", time.perf_counter() - t0)

    def 取听觉(self) -> dict:
        with self._听觉锁:
            a = dict(self._听觉)
            a["事件s"] = list(a["事件s"])
        return a

    # ══════════ 意图槽 ══════════
    def 写意图(self, 当前=None, 强度=None, 来源=None, 推目标=None, 弹目标=False):
        t0 = time.perf_counter()
        with self._意图锁:
            if 弹目标 and self._意图["目标栈"]:
                self._意图["目标栈"].pop(0)
            if 推目标 is not None:
                self._意图["目标栈"].append(推目标)
            if 当前 is not None:
                self._意图["当前"] = 当前
            if 强度 is not None:
                self._意图["强度"] = float(强度)
            if 来源 is not None:
                self._意图["来源"] = str(来源)
            self._意图["版本"] += 1
        self._计("写意图", time.perf_counter() - t0)

    # ── ◆目标槽（步7 追加；步8 加「名称」——教学指认给形状/位置起名，图标簿据此直达） ──
    def 写目标(self, 主色=None, 心=None, 句柄=None, 来源="", 清=False, 名称=None):
        """写目标（主色=(r,g,b) 参考特征 + 心=(x,y) 参考位置 + 名称=教学命名）；
        清=True 清空（无目标态）。"""
        t0 = time.perf_counter()
        with self._目标锁:
            if 清:
                self._目标 = dict(主色=None, 心=None, 句柄=None, 名称="", 来源="", 版本=0)
            else:
                if 主色 is not None:
                    self._目标["主色"] = tuple(int(v) for v in 主色)
                if 心 is not None:
                    self._目标["心"] = (int(心[0]), int(心[1]))
                if 句柄 is not None:
                    self._目标["句柄"] = 句柄
                if 名称 is not None:
                    self._目标["名称"] = str(名称)
                if 来源:
                    self._目标["来源"] = str(来源)
                self._目标["版本"] += 1
        self._计("写目标", time.perf_counter() - t0)

    def 取目标(self) -> dict:
        t0 = time.perf_counter()
        with self._目标锁:
            v = dict(self._目标)
        self._计("取目标", time.perf_counter() - t0)
        return v

    def 取意图(self) -> dict:
        with self._意图锁:
            d = dict(self._意图)
            d["目标栈"] = list(d["目标栈"])
        return d

    # ══════════ 动作槽（手的反馈回流） ══════════
    def 写动作(self, **kw):
        t0 = time.perf_counter()
        with self._动作锁:
            for k, v in kw.items():
                if k == "反馈":
                    self._动作["反馈"].append(v)
                    if len(self._动作["反馈"]) > 200:
                        del self._动作["反馈"][:100]
                else:
                    self._动作[k] = v
        self._计("写动作", time.perf_counter() - t0)

    def 取动作(self) -> dict:
        with self._动作锁:
            d = dict(self._动作)
            d["反馈"] = list(d["反馈"])
        return d

    def 手忙(self) -> bool:
        with self._动作锁:
            return self._动作["状态"] != "闲"

    # ══════════ 表达槽 ══════════
    def 排队说(self, 文本: str, 会话: str = "主"):
        """待说队列入队。带「会话」= 从哪个窗口说出（嘴据此分流打印 [窗口]）。"""
        with self._表达锁:
            self._表达["队列"].append((time.time(), str(文本), str(会话)))

    def 取下一句(self):
        """取走一句待说：返回 (t, 文, 会话)；空返回 None。"""
        with self._表达锁:
            if self._表达["队列"]:
                t, 文, 会话 = self._表达["队列"].popleft()
                self._表达["最近"] = 文
                self._表达["说数"] += 1
                return t, 文, 会话
        return None

    def 取表达(self) -> dict:
        with self._表达锁:
            return dict(队列=len(self._表达["队列"]), 最近=self._表达["最近"],
                        说数=self._表达["说数"])

    # ══════════ 语义槽 ══════════
    def 写语义(self, **kw):
        t0 = time.perf_counter()
        with self._语义锁:
            self._语义.update(kw)
        self._计("写语义", time.perf_counter() - t0)

    def 取语义(self) -> dict:
        with self._语义锁:
            return dict(self._语义)

    # ══════════ 度量 ══════════
    def 取度量(self) -> dict:
        with self._度量锁:
            return {k: (dict(v) if isinstance(v, dict) else v)
                    for k, v in self._度量.items()}

    # ══════════ ◆自我槽（步3 追加：自我线程单写） ══════════
    def 写自我(self, **kw):
        t0 = time.perf_counter()
        with self._自我锁:
            self._自我.update(kw)
            if "自体拍" in kw:
                self._自我["自体拍"] = int(kw["自体拍"])
            self._自我["更新t"] = time.time()
        self._计("写自我", time.perf_counter() - t0)

    def 取自我(self) -> dict:
        with self._自我锁:
            v = dict(self._自我)
            v["类型覆盖"] = dict(v.get("类型覆盖") or {})
            v["审计"] = dict(v.get("审计") or {})
            return v

    # ══════════ ◆事件槽（多生产者单消费者；取走即清） ══════════
    def 投事件(self, 事件: dict):
        t0 = time.perf_counter()
        with self._事件锁:
            self._事件.append(dict(事件, 投递t=time.time()))   # 满时 deque 自动丢最旧
        self._计("投事件", time.perf_counter() - t0)

    def 取事件(self, 上限: int = 64) -> list:
        with self._事件锁:
            取走 = []
            while self._事件 and len(取走) < int(上限):
                取走.append(self._事件.popleft())
            return 取走

    # ══════════ ◆计划槽（脑单写；嘴/自我线程读） ══════════
    def 写计划(self, **kw):
        t0 = time.perf_counter()
        with self._计划锁:
            self._计划.update(kw)
            if kw.get("当前") is not None or "待办" in kw:
                self._计划["版本"] += 1
        self._计("写计划", time.perf_counter() - t0)

    def 取计划(self) -> dict:
        with self._计划锁:
            v = dict(self._计划)
            v["待办"] = list(v["待办"]); v["已办"] = list(v["已办"])
            v["说做对账"] = dict(v["说做对账"])
            return v

    def 推待办(self, 步骤) -> int:
        t0 = time.perf_counter()
        with self._计划锁:
            self._计划["待办"].append(步骤)
            self._计划["版本"] += 1
        self._计("推待办", time.perf_counter() - t0)
        return len(self._计划["待办"])

    def 取待办(self):
        """弹出下一个待办步骤（取走即清）；空返回 None。"""
        with self._计划锁:
            if self._计划["待办"]:
                self._计划["版本"] += 1
                return self._计划["待办"].pop(0)
            return None

    # ══════════ ◆视场槽（视觉会议步1 追加：眼单写） ══════════
    def 写视场(self, **kw):
        """写「看」的本体：视场/注视/误差场/意外度/逐格意外/眼跳数。
        锁内只赋值不做计算（沿用本文件纪律）——矩阵/编解码一律在锁外算好。"""
        t0 = time.perf_counter()
        with self._视场锁:
            self._视场.update(kw)
            if "注视" in kw:
                self._视场["注视链"].append(tuple(kw["注视"]) if kw["注视"] else None)
            self._视场["更新t"] = time.time()
        self._计("写视场", time.perf_counter() - t0)

    def 取视场(self) -> dict:
        """深拷贝快照（ndarray 复制、注视链复制），锁内不做计算。"""
        with self._视场锁:
            v = dict(self._视场)
        for k in ("视场", "误差场", "逐格意外"):
            if isinstance(v.get(k), np.ndarray):
                v[k] = v[k].copy()
        v["注视链"] = list(v["注视链"])
        return v

    # ══════════ ◆思考槽（脑单写；演播台/嘴只读） ══════════
    def 写思考(self, 类: str, 文: str, 层: str = "外"):
        """追加一条原生思考：类 ∈ 看|想|招|预|果|路（脑自定，槽位不加语义约束）。
        层 ∈ 外|内（所有者 2026-10-06「深度思考不外显，只看『我应该点哪里』这种」）：
          外 = 朝外正文（我要干什么/该点哪里/点了会怎样/下一步）——演播台、文本流渲染；
          内 = 深度回响（感知汇总/联想/权衡/自评）——只留档，不外显。
        锁内只 append（不做计算/格式化），与全文件纪律一致。默认「外」⇒ 旧调用逐位不变（C14）。"""
        t0 = time.perf_counter()
        with self._思考锁:
            self._思考.append(dict(t=time.time(), 类=str(类), 文=str(文), 层=str(层)))
        self._计("写思考", time.perf_counter() - t0)

    def 取思考(self, 上限: int = 40, 只要外: bool = True) -> list:
        """取最近 上限 条思考（时间序，旧→新）。
        只要外=True（默认）：只给「外」层正文（演播台/文本流用，内层回响不外显）；
        只要外=False：含内层（审计/统计用）。"""
        with self._思考锁:
            if 上限 <= 0:
                return []
            xs = [dict(x) for x in list(self._思考)]
        if 只要外:
            xs = [x for x in xs if x.get("层", "外") == "外"]
        return xs[-int(上限):]

    # ══════════ ◆对话槽（纯文本对话态；界面/脑写，文本流只读） ══════════
    # 多窗口：每个窗口（会话ID）有独立的 用户箱/对话流/脑内；记忆由综合体共享（见类 docstring）。
    # 所有方法 会话 默认 "主" —— 旧调用不传会话即落"主"窗口，行为逐位不变（C14）。
    def 写用户说(self, 文: str, 谁: str = "我", 会话: str = "主"):
        """用户的一句话入箱（界面写）。刷新上次用户时刻、入**该窗口**文本流、置「等待」。
        「谁」= 发言人标识（多账号共用同一生命时用于检索隔离；默认「我」）。
        「会话」= 哪个窗口（默认「主」）。"""
        t0 = time.perf_counter()
        with self._对话锁:
            d = self._槽(会话)
            t = time.time()
            d["用户箱"].append((t, str(文), str(谁)))
            d["上次用户t"] = t
            d["对话流"].append((t, str(谁), str(文)))
            d["状态"] = "等待"
        self._计("写用户说", time.perf_counter() - t0)

    def 取用户说(self, 会话: str = "主"):
        """取走该窗口一条用户消息（取走即清）；无则 None。返回 (t, 文, 谁)；置「回合」。"""
        with self._对话锁:
            d = self._槽(会话)
            if d["用户箱"]:
                t, 文, 谁 = d["用户箱"].popleft()
                d["状态"] = "回合"
                return t, 文, 谁
            return None

    def 有用户说(self, 会话: str = "主") -> bool:
        with self._对话锁:
            return bool(self._槽(会话)["用户箱"])

    def 写它说(self, 文: str, 会话: str = "主"):
        """它回的那一句（脑写）。入**该窗口**文本流、占用一个回合、置「空闲」。"""
        t0 = time.perf_counter()
        with self._对话锁:
            d = self._槽(会话)
            d["它说"] = str(文)
            d["对话流"].append((time.time(), "它", str(文)))
            d["回合"] += 1
            d["状态"] = "空闲"
        self._计("写它说", time.perf_counter() - t0)

    def 写它想(self, 文: str, 会话: str = "主"):
        """它的「内心想」那一句（脑写）：回忆/自言/权衡。

        ★ 多窗口·正常对话流（所有者 2026-10-06「别来什么思考，正常人不会这么干」）：
          内心念头 **不入对话流**（对话流只留 我/它 的对白），改走**脑内**内部通道，
          供审计/统计/演播台读取（取脑内/取脑内条数）。这样"对话流正常一点"。"""
        t0 = time.perf_counter()
        with self._对话锁:
            d = self._槽(会话)
            d["它想"] = str(文)
            d["脑内"].append((time.time(), str(文)))
        self._计("写它想", time.perf_counter() - t0)

    def 取脑内(self, 上限: int = 40, 会话: str = "主") -> list:
        """取该窗口最近 上限 条**内部念头**（时间序，旧→新），每条 (t, 文)。只供审计。"""
        with self._对话锁:
            if 上限 <= 0:
                return []
            return [tuple(x) for x in list(self._槽(会话)["脑内"])[-int(上限):]]

    def 取对话(self, 上限: int = 40, 会话: str = "主") -> list:
        """取该窗口最近 上限 条文本流（时间序，旧→新），每条 (t, 谁, 文)。"""
        with self._对话锁:
            if 上限 <= 0:
                return []
            return [tuple(x) for x in list(self._槽(会话)["对话流"])[-int(上限):]]

    def 对话态(self, 会话: str = "主") -> dict:
        with self._对话锁:
            d = self._槽(会话)
            return dict(排队=len(d["用户箱"]), 回合=d["回合"],
                        它说=d["它说"], 它想=d["它想"],
                        状态=d["状态"], 上次用户t=d["上次用户t"],
                        醒=int(self._醒), 会话=str(会话),
                        脑内=len(d["脑内"]))

    def 会话们(self) -> list:
        """当前已知的所有窗口 ID（含"主"）。"""
        with self._对话锁:
            return list(self._会话表.keys())

    def 待处理会话(self) -> list:
        """有未读用户消息的窗口 ID 列表（按入箱顺序；供多窗口分发）。"""
        with self._对话锁:
            return [k for k, d in self._会话表.items() if d["用户箱"]]

    def 置醒(self, 醒):
        """生命自身的醒/睡结果落槽（决策在生命内，本槽只存状态）。**生命级**，不随窗口。"""
        with self._对话锁:
            self._醒 = int(bool(醒))
