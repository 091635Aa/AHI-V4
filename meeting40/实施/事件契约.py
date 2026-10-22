# -*- coding: utf-8 -*-
"""meeting40/实施/事件契约.py —— P0 [正交] 事件 schema 与 provenance 固定枚举（终稿 §7.4/§7.6）。

本文件是**唯一口径的代码化**（文字口径见终稿 §7.4b/§7.6 与审计 §8 冲突①–④）：

  InputEvent  = { t_pai:int64, seq:int64, 模态, 内容句柄, τ, prov:Provenance,
                  conf:float16, 效价, 可执行目标点 }
      **禁**整帧 / 逐格误差 / 内部得分 / 字面量（`元素s.字` 恒空）。
  OutputEvent = { t_pai:int64, seq:int64, 通道, R版本, 断言级, 内容sha256,
                  优先级, 提交拍, 可逆性, 令牌id, 结果, prov_exec }
  两者共享 `t_pai`（拍钟，**禁 wall-clock**）与 `prov`。

  时基口径（与 `时基.py`、终稿 §8.2a 一致）：`τ`＝**分数拍**（判决量），
  `t_pai = floor(τ)`＝整数拍。150ms = 1.5 拍这类非整数栅格一律由 τ 承载。

  provenance 固定枚举（§7.6）：
      prov_acq   ∈ {自源, 用户, 网页, 未知}      获取源
      prov_org   ∈ {自源, 外源}                  内容源
      prov_route ∈ {实感, 回忆, 想象, 自说}      来路（「自说」为本次新增值）
      prov_exec  ∈ {真机, 演练}                  执行域（仅输出/动作通道）

  唯一写闸（§7.6，三条，代码化在 `可写记忆` / `可塑W_self` / `自听落点`）：
      ① 写 M 仅当 prov_route=实感 ∧ **非**「本系统输出回灌」（自说/输出不算记忆来源）
      ② 更新 W_self/ĝ 仅当 prov_org=自源（外源/想象灌入 ⇒ ‖ΔW_self‖=0）
      ③ 自听**只写学习迹**，内容记忆 Δ=0

纪律：纯新增文件；不 import 任何器官；无全局随机态；所有比较/序列化确定性（C13）。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

# ─────────────────────────── 固定枚举 ───────────────────────────
获取源枚举 = ("自源", "用户", "网页", "未知")
内容源枚举 = ("自源", "外源")
来路枚举 = ("实感", "回忆", "想象", "自说")
执行域枚举 = ("真机", "演练")
断言级枚举 = ("事实", "回忆", "推测")
输出通道枚举 = ("文本", "动作", "语音")
优先级枚举 = ("P0", "P1", "P2", "P3", "P4")            # P0 最高（抢占）
模态枚举 = ("视觉", "听觉", "视频", "文本", "音频", "向量", "特征")
_模态附加: set = set()                                  # 可注册扩展（不得绕过 C1/C9/C10）

# 禁字段（§7.4b）：InputEvent 里出现任一 ⇒ 校验失败（防「整帧/得分/字面量」偷偷进屋）
禁字段 = frozenset({"帧", "整帧", "原始帧", "误差场", "逐格误差", "得分", "score",
                    "字", "字面量", "文本", "像素"})
写闸依据 = dict(M="prov_route=实感 ∧ 非输出回灌", W_self="prov_org=自源",
                自听="只写学习迹（内容记忆 Δ=0）")


def 注册模态(名: str) -> None:
    """登记一个附加输入模态名（工程上允许，红线不允许绕过）。"""
    s = str(名)
    if not s:
        raise ValueError("模态名不得为空")
    _模态附加.add(s)


def _枚举校验(名: str, 值, 表) -> str:
    v = str(值)
    if v not in 表:
        raise ValueError("%s=%r 不在固定枚举 %s 内（终稿 §7.6 唯一口径）" % (名, 值, list(表)))
    return v


# ─────────────────────────── 确定性序列化 ───────────────────────────
def 规范字节(对象) -> bytes:
    """规范 JSON 字节：键排序、紧凑分隔符、非 ASCII 不转义（C13：同值 ⇒ 同字节）。"""
    return json.dumps(对象, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str).encode("utf-8")


def sha256(对象) -> str:
    if isinstance(对象, (bytes, bytearray)):
        return hashlib.sha256(bytes(对象)).hexdigest()
    return hashlib.sha256(规范字节(对象)).hexdigest()


def 内容sha256(内容) -> str:
    """内容指纹（OutputEvent.内容sha256 / SensorFrame.sha256 用）。"""
    return sha256(内容)


# ─────────────────────────── provenance ───────────────────────────
@dataclass(frozen=True)
class Provenance:
    """来源双/多标记（§7.6）。构造即校验；不可变 ⇒ 事件可安全共享。"""
    prov_acq: str = "未知"
    prov_org: str = "外源"
    prov_route: str = "实感"
    prov_exec: str = None                     # 仅输出/动作通道填

    def __post_init__(self):
        object.__setattr__(self, "prov_acq", _枚举校验("prov_acq", self.prov_acq, 获取源枚举))
        object.__setattr__(self, "prov_org", _枚举校验("prov_org", self.prov_org, 内容源枚举))
        object.__setattr__(self, "prov_route", _枚举校验("prov_route", self.prov_route, 来路枚举))
        if self.prov_exec is not None:
            object.__setattr__(self, "prov_exec",
                               _枚举校验("prov_exec", self.prov_exec, 执行域枚举))

    def 字典(self) -> dict:
        d = dict(prov_acq=self.prov_acq, prov_org=self.prov_org, prov_route=self.prov_route)
        if self.prov_exec is not None:
            d["prov_exec"] = self.prov_exec
        return d

    @staticmethod
    def 从字典(d) -> "Provenance":
        d = dict(d or {})
        return Provenance(prov_acq=d.get("prov_acq", "未知"),
                          prov_org=d.get("prov_org", "外源"),
                          prov_route=d.get("prov_route", "实感"),
                          prov_exec=d.get("prov_exec"))


# ─────────────────────────── 唯一写闸（§7.6） ───────────────────────────
def 可写记忆(prov: Provenance, 输出回灌: bool = False) -> bool:
    """写 M 的**唯一**许可：来路=实感，且内容不是本系统输出的回灌。"""
    return bool(prov.prov_route == "实感" and not 输出回灌)


def 可塑W_self(prov: Provenance) -> bool:
    """更新 W_self/ĝ 的**唯一**许可：内容源=自源（外源/想象灌入 ⇒ ‖ΔW_self‖=0）。"""
    return bool(prov.prov_org == "自源")


def 自听落点(prov: Provenance) -> str:
    """自听的**唯一**落点：学习迹（内容记忆 Δ=0）。返回落点名，供调用方记账。"""
    return "学习迹"


def 写闸(prov: Provenance, 输出回灌: bool = False) -> dict:
    """一次算全三闸（审计字段）：谁可写、谁不可写、依据（§7.6）。"""
    return dict(可写M=可写记忆(prov, 输出回灌), 可塑W_self=可塑W_self(prov),
                自听落点=自听落点(prov), 依据=dict(写闸依据), prov=prov.字典())


# ─────────────────────────── SensorFrame ───────────────────────────
@dataclass
class SensorFrame:
    """一帧原生输入（帧源 → 前端）。形状/dtype/数据 必须自洽；t_mono 单调（拍钟口径）。"""
    模态: str
    t_mono: float
    seq: int
    形状: tuple
    dtype: str
    数据: object
    标定: dict = field(default_factory=dict)
    质量: float = 1.0
    来源: str = "未知"

    def __post_init__(self):
        self.模态 = _模态校验(self.模态)
        self.t_mono = float(self.t_mono)
        self.seq = int(self.seq)
        self.形状 = tuple(int(x) for x in self.形状)
        self.dtype = str(self.dtype)
        self.质量 = float(self.质量)
        import math
        if not math.isfinite(self.t_mono):
            raise ValueError("t_mono 必须有限（拍钟口径，禁 NaN/Inf）")
        if not (0.0 <= self.质量 <= 1.0):
            raise ValueError("质量须在 [0,1]：%r" % self.质量)

    def 字节(self) -> bytes:
        数 = self.数据
        if hasattr(数, "tobytes"):
            return 数.tobytes()
        if isinstance(数, (bytes, bytearray)):
            return bytes(数)
        return 规范字节(数)

    def 特征码(self) -> str:
        """录制/重放判据用：同帧 ⇒ 同特征码（与 PYTHONHASHSEED 无关）。"""
        h = hashlib.sha256()
        h.update(("%s|%.9f|%d|%s|%s|%s" % (self.模态, self.t_mono, self.seq,
                                           list(self.形状), self.dtype, self.来源)).encode("utf-8"))
        h.update(self.字节())
        h.update(规范字节(self.标定))
        return h.hexdigest()

    def 字典(self) -> dict:
        return dict(模态=self.模态, t_mono=self.t_mono, seq=self.seq, 形状=list(self.形状),
                    dtype=self.dtype, 字节数=len(self.字节()), 标定=dict(self.标定),
                    质量=self.质量, 来源=self.来源, sha256=self.特征码())


def _模态校验(值) -> str:
    v = str(值)
    if v not in 模态枚举 and v not in _模态附加:
        raise ValueError("模态=%r 不在 %s（或注册模态() 后）" % (值, list(模态枚举)))
    return v


# ─────────────────────────── InputEvent / OutputEvent ───────────────────────────
@dataclass
class InputEvent:
    """输入事件原子（§7.4b）。`τ`＝分数拍；`t_pai`＝floor(τ) 整数拍。"""
    τ: float
    seq: int
    模态: str
    内容句柄: str
    prov: Provenance
    conf: float
    效价: float = 0.0
    可执行目标点: tuple = None
    附加: dict = field(default_factory=dict)      # 任何附加字段都要过 `禁字段` 关

    def __post_init__(self):
        import numpy as np
        self.τ = float(self.τ)
        self.seq = int(self.seq)
        self.模态 = _模态校验(self.模态)
        self.内容句柄 = str(self.内容句柄)
        if not isinstance(self.prov, Provenance):
            self.prov = Provenance.从字典(self.prov)
        self.conf = float(np.float16(float(self.conf)))          # float16 口径（§7.4b）
        if not (0.0 <= self.conf <= 1.0):
            raise ValueError("conf 须在 [0,1]（float16 取整后）：%r" % self.conf)
        self.效价 = float(np.float16(float(self.效价)))
        if not (-1.0 <= self.效价 <= 1.0):
            raise ValueError("效价 须在 [-1,1]：%r" % self.效价)
        if self.可执行目标点 is not None:
            self.可执行目标点 = tuple(float(x) for x in self.可执行目标点)
        越界 = sorted(set(self.附加) & 禁字段)
        if 越界:
            raise ValueError("InputEvent 禁字段现身：%s（§7.4b：禁整帧/逐格误差/得分/字面量）"
                             % 越界)

    @property
    def t_pai(self) -> int:
        return int(self.τ // 1.0) if self.τ >= 0 else -int((-self.τ) // 1.0) - 1

    def 字典(self) -> dict:
        d = dict(t_pai=int(self.t_pai), seq=self.seq, 模态=self.模态, 内容句柄=self.内容句柄,
                 τ=self.τ, prov=self.prov.字典(), conf=self.conf, 效价=self.效价,
                 可执行目标点=(None if self.可执行目标点 is None else list(self.可执行目标点)))
        if self.附加:
            d.update(self.附加)
        return d

    def 特征码(self) -> str:
        return sha256(self.字典())


@dataclass
class OutputEvent:
    """输出事件（§7.4b）。仲裁器 → 审计/WAL。动作/不可逆必须带令牌 id（fail-closed）。"""
    τ: float
    seq: int
    通道: str
    R版本: int
    断言级: str
    内容sha256: str
    优先级: str = "P4"
    可逆性: bool = True
    令牌id: str = None
    结果: str = None
    prov_exec: str = "演练"
    提交拍: int = None
    附加: dict = field(default_factory=dict)

    def __post_init__(self):
        self.τ = float(self.τ)
        self.seq = int(self.seq)
        self.通道 = _枚举校验("通道", self.通道, 输出通道枚举)
        self.断言级 = _枚举校验("断言级", self.断言级, 断言级枚举)
        self.优先级 = _枚举校验("优先级", self.优先级, 优先级枚举)
        self.prov_exec = _枚举校验("prov_exec", self.prov_exec, 执行域枚举)
        self.R版本 = int(self.R版本)
        self.可逆性 = bool(self.可逆性)
        self.提交拍 = int(self.τ // 1.0) if self.提交拍 is None else int(self.提交拍)
        内容 = str(self.内容sha256)
        if len(内容) != 64 or any(c not in "0123456789abcdef" for c in 内容.lower()):
            raise ValueError("内容sha256 须为 64 位十六进制：%r" % self.内容sha256)
        self.内容sha256 = 内容.lower()
        if self.通道 == "动作" and not self.令牌id:
            raise ValueError("动作通道缺令牌id ⇒ fail-closed（§7.6/G-R3-14）")
        if not self.可逆性 and not self.令牌id:
            raise ValueError("不可逆输出缺令牌id ⇒ fail-closed（§7.6/G-R3-14）")

    @property
    def t_pai(self) -> int:
        return int(self.τ // 1.0) if self.τ >= 0 else -int((-self.τ) // 1.0) - 1

    def 字典(self) -> dict:
        d = dict(t_pai=int(self.t_pai), seq=self.seq, 通道=self.通道, R版本=self.R版本,
                 断言级=self.断言级, 内容sha256=self.内容sha256, 优先级=self.优先级,
                 提交拍=self.提交拍, 可逆性=self.可逆性, 令牌id=self.令牌id,
                 结果=self.结果, prov_exec=self.prov_exec)
        if self.附加:
            d.update(self.附加)
        return d

    def 特征码(self) -> str:
        return sha256(self.字典())


def 仲裁比较(甲: "OutputEvent", 乙: "OutputEvent") -> int:
    """全序比较器（§7.1）：先比优先级（P0 最高），再比 τ（先到先出），再比 seq。
    返回 −1/0/1。确定性：无随机、无墙钟。"""
    序 = {p: i for i, p in enumerate(优先级枚举)}
    a, b = 序[甲.优先级], 序[乙.优先级]
    if a != b:
        return -1 if a < b else 1
    if 甲.τ != 乙.τ:
        return -1 if 甲.τ < 乙.τ else 1
    if 甲.seq != 乙.seq:
        return -1 if 甲.seq < 乙.seq else 1
    return 0


# ─────────────────────────── 便捷构造（接拍钟埋点） ───────────────────────────
def 造输入事件(钟, 模态: str, 内容句柄: str, prov, conf: float,
              效价: float = 0.0, 可执行目标点=None, 埋点: str = "τ_in") -> InputEvent:
    """用 `时基.拍钟` 打一个埋点，并据其 τ 造 InputEvent（t0=输入事件合成拍）。"""
    d = 钟.打点(埋点)
    return InputEvent(τ=d["分数拍"], seq=len(钟), 模态=模态, 内容句柄=内容句柄,
                      prov=prov, conf=conf, 效价=效价, 可执行目标点=可执行目标点)


# ─────────────────────────── 自测 ───────────────────────────
def 自测() -> dict:
    好 = Provenance(prov_acq="网页", prov_org="外源", prov_route="实感")
    e = InputEvent(τ=1.5, seq=7, 模态="视觉", 内容句柄="E12", prov=好, conf=0.9)
    拒 = {}
    for 名, 造 in (("禁字段", lambda: InputEvent(τ=0.0, seq=0, 模态="视觉", 内容句柄="E1",
                                               prov=好, conf=1.0, 附加={"得分": 3})),
                   ("枚举越界", lambda: Provenance(prov_route="幻觉")),
                   ("动作缺令牌", lambda: OutputEvent(τ=0.0, seq=0, 通道="动作", R版本=1,
                                                 断言级="事实", 内容sha256="0" * 64)),
                   ("不可逆缺令牌", lambda: OutputEvent(τ=0.0, seq=0, 通道="文本", R版本=1,
                                                  断言级="回忆", 内容sha256="a" * 64,
                                                  可逆性=False))):
        try:
            造()
            拒[名] = False
        except ValueError:
            拒[名] = True
    闸 = 写闸(Provenance(prov_acq="用户", prov_org="外源", prov_route="自说"))
    同 = (InputEvent(τ=1.5, seq=7, 模态="视觉", 内容句柄="E12", prov=好, conf=0.9).特征码()
          == e.特征码())
    return dict(通过=bool(all(拒.values()) and 闸["可写M"] is False
                        and 闸["可塑W_self"] is False and 同),
                t_pai对=(e.t_pai == 1), 特征码同=同, 拒绝=拒, 写闸=闸)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 自测()
    print(json.dumps(果, ensure_ascii=False, indent=1, default=str))
    print("[事件契约] 自测：%s" % ("PASS" if 果["通过"] else "FAIL"))
    raise SystemExit(0 if 果["通过"] else 1)
