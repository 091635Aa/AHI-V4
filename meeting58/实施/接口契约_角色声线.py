# -*- coding: utf-8 -*-
"""meeting58/实施/接口契约_角色声线.py —— 角色声线层**唯一接口真相源**（冻结件）。

所有者 2026-10-19 会议前置任务：「先把接口理清楚」。
本契约为**纯新增件**（C14）：不改 AHI3/AHI4 任何既有文件；只被 meeting58 的实现 import。

━━━ 分层（命名避开内核的 A/B 区室）━━━
  内核（只读）：双通道体（V5）→ o用/内心/遥测 → 状态帧
  行层（角色行为逻辑）：只产文本，不碰波形
  声层（音色输出）：只产音频，不产文本
  两层只通过 `文本产物` / `渲染请求` 通信：行层不得 import 声层，声层不得 import 行层。
  外部控制层（demo/扮演）：只能实现同一个 `音色渲染协议` 或 `角色逻辑协议` 插进来；
  若使用外部模型（RVC/TTS），必须在 `声线档案.来源` 标 `外部·越宪` 并登记模型名/sha256，
  **不得进内核**（C10）。

━━━ 铁律 ━━━
  C1 禁 Transformer ｜ C9 禁 OCR ｜ C10 禁外挂模型当器官 ｜ C13 确定性可复现 ｜ C14 纯新增
  「复刻」口径 = **参数化模仿**，不是真声克隆；任何输出不得宣称是角色本人。

━━━ 归属（写越界=冲突）━━━
  `接口契约_角色声线.py`  ← 总装（冻结，只读）
  `声线库.py`              ← 声层（声线档案数据 + 定制估计）
  `协议_音色渲染.py`       ← 声层（渲染实现）
  `协议_状态源.py`         ← 内核侧适配（状态帧 + 确定性投影）
  `协议_角色逻辑.py`       ← 行层（角色模板 + 文本）
  `注册表_挂载.py`         ← 总装（热插拔生命周期）
  `装配_角色声线.py`       ← 总装（拍循环装配）
  `判据_接口与解耦.py`     ← 判据（跑前冻结阈值）
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, TypedDict, runtime_checkable

契约版本 = "role-voice-1.0"
拍长秒 = 0.1
默认种子 = 300

# ───────────────────────────── 错误码（码 + 字符串双写） ─────────────────────────────
E1001 = "E1001"; E1002 = "E1002"; E1003 = "E1003"; E1004 = "E1004"
E1005 = "E1005"; E1006 = "E1006"; E1007 = "E1007"; E1008 = "E1008"
E1009 = "E1009"; E1010 = "E1010"; E1011 = "E1011"; E1012 = "E1012"
错误说明 = {
    E1001: "未知插件/未注册",
    E1002: "协议版本不兼容",
    E1003: "依赖缺失或不符",
    E1004: "插件未装载",
    E1005: "重复注册/重复装载",
    E1006: "声线档案非法（越界/缺键/超参数域）",
    E1007: "状态源失配（D核 ≠ 帧.D核）",
    E1008: "渲染超预算",
    E1009: "插件抛异常（已隔离）",
    E1010: "回滚目标不存在",
    E1011: "C1/C9/C10 审计命中",
    E1012: "栅栏冲突（切换被拒）",
}


class 契约错误(Exception):
    def __init__(self, 码: str, 细节: str = ""):
        self.码 = str(码)
        self.细节 = str(细节)
        super().__init__("%s %s %s" % (self.码, 错误说明.get(self.码, "?"), self.细节))


def 抛(码: str, 细节: str = "") -> None:
    raise 契约错误(码, 细节)


# ───────────────────────────── 审计锚 ─────────────────────────────
# 新实现文件里不得出现这些子串（大小写不敏感）——C1/C10 静态审计
禁词 = ("attention", "attn", "qkv", "softmax", "multihead", "multi_head",
        "transformer", "位置编码", "positional_encoding", "scaled_dot",
        "torch", "transformers", "rwkv", "self_attn", "cross_attn",
        "rvc", "sovits", "vits", "tacotron", "waveglow", "hifigan", "whisper")
# 运行期禁止 import 的顶层模块（外部模型当器官 = C10）
禁import = ("torch", "transformers", "rwkv", "tensorflow", "onnxruntime",
            "pyttsx3", "sounddevice", "cv2", "pytesseract")


def 审计文件(路径列: Sequence[str]) -> dict:
    """静态审计：禁词 + 禁 import。返回 {通过, 命中}。"""
    命中 = []
    for p in 路径列:
        try:
            with open(p, encoding="utf-8", errors="ignore") as f:
                s = f.read()
        except OSError:
            命中.append([p, "读失败", ""])
            continue
        低 = s.lower()
        for w in 禁词:
            if w in 低:
                命中.append([str(p), "禁词", w])
        for line in s.splitlines():
            t = line.strip()
            if t.startswith("import ") or t.startswith("from "):
                首 = t.split()[1].split(".")[0] if len(t.split()) > 1 else ""
                if 首 in 禁import:
                    命中.append([str(p), "禁import", 首])
    return dict(通过=not 命中, 命中=命中)


# ───────────────────────────── 数据结构 ─────────────────────────────
@dataclass(frozen=True)
class 状态帧:
    """内核状态的**不可变**只读快照。插件写不进去。"""
    拍: int
    观测: int
    D核: int
    o用: bytes
    内心: bytes
    自我标量: float = 0.5
    惊喜: float = 0.0
    遥测: tuple = ()                      # ((名, 值), ...) 有序、可哈希
    指纹: str = ""

    @staticmethod
    def 建(拍: int, 观测: int, o用, 内心, 自我标量: float = 0.5,
           惊喜: float = 0.0, 遥测: Mapping[str, float] | None = None) -> "状态帧":
        import numpy as np
        a = np.asarray(o用, dtype=np.float64).ravel()
        b = np.asarray(内心, dtype=np.float64).ravel()
        if a.shape != b.shape:
            抛(E1007, "o用/内心 维度不同")
        t = tuple(sorted((str(k), float(v)) for k, v in (遥测 or {}).items()))
        return 状态帧(拍=int(拍), 观测=int(观测), D核=int(a.size),
                      o用=a.astype("<f8").tobytes(), 内心=b.astype("<f8").tobytes(),
                      自我标量=float(max(0.0, min(1.0, 自我标量))), 惊喜=float(惊喜),
                      遥测=t, 指纹="")._带指纹()

    def _带指纹(self) -> "状态帧":
        h = hashlib.sha256()
        h.update(b"state|%d|%d|%d|" % (self.拍, self.观测, self.D核))
        h.update(self.o用); h.update(self.内心)
        h.update(repr((round(self.自我标量, 9), round(self.惊喜, 9), self.遥测)).encode())
        return 状态帧(self.拍, self.观测, self.D核, self.o用, self.内心,
                      自我标量=self.自我标量, 惊喜=self.惊喜, 遥测=self.遥测,
                      指纹=h.hexdigest()[:16])

    def 数组(self, 名: str = "o用"):
        import numpy as np
        b = self.o用 if 名 == "o用" else self.内心
        return np.frombuffer(b, dtype="<f8").reshape(self.D核)

    def 转dict(self) -> dict:
        return dict(拍=self.拍, 观测=self.观测, D核=self.D核, 自我标量=self.自我标量,
                    惊喜=self.惊喜, 遥测=dict(self.遥测), 指纹=self.指纹)


声线参数域 = {
    "基频": (70.0, 400.0), "语速": (1.5, 7.0), "幅度": (0.1, 1.0),
    "气声度": (0.0, 0.8), "紧喉度": (0.0, 1.0), "粗糙度": (0.0, 1.0),
    "送气量": (0.0, 1.0), "声区": (0.0, 1.0),
    "谱倾斜": (-0.85, 0.85), "增益": (0.0, 2.0),
    "气声谱重心": (0.0, 1.0), "抖动": (0.0, 1.0),
    "共振峰缩放": (0.70, 1.30),
}
来源枚举 = ("内生·参数化", "外采·参考录音", "外采·公开资料", "外部·越宪", "模型·登记")


@dataclass(frozen=True)
class 声线档案:
    """声层唯一输入：不可变声线档案（参数化模仿，不是声克隆）。"""
    名: str
    版本: str = "1.0"
    来源: str = "内生·参数化"
    基频: float = 190.0
    语速: float = 3.2
    幅度: float = 0.50
    气声度: float = 0.20
    紧喉度: float = 0.30
    粗糙度: float = 0.05
    送气量: float = 0.20
    声区: float = 0.05
    共振峰缩放: tuple = (1.0, 1.0, 1.0)
    谱倾斜: float = 0.0
    气声谱重心: float = 0.5
    抖动: float = 0.0
    增益: float = 1.0
    备注: str = ""

    def 校验(self) -> None:
        for k, (lo, hi) in 声线参数域.items():
            if k == "共振峰缩放":
                v = tuple(float(x) for x in self.共振峰缩放)
                if len(v) != 3 or any(not (lo <= x <= hi) for x in v):
                    抛(E1006, "%s=%r" % (k, self.共振峰缩放))
            else:
                v = float(getattr(self, k))
                if not (lo <= v <= hi):
                    抛(E1006, "%s=%r 不在 [%s,%s]" % (k, v, lo, hi))
        if self.来源 not in 来源枚举:
            抛(E1006, "来源=%r" % (self.来源,))
        if not str(self.名):
            抛(E1006, "名为空")

    def 指纹(self) -> str:
        d = self.转dict()
        return hashlib.sha256(json.dumps(d, ensure_ascii=False, sort_keys=True,
                                         default=str).encode()).hexdigest()[:16]

    def 转dict(self) -> dict:
        d = dict(self.__dict__)
        d["共振峰缩放"] = [float(x) for x in self.共振峰缩放]
        return d

    @staticmethod
    def 从dict(d: Mapping[str, Any]) -> "声线档案":
        from dataclasses import fields as _f
        允许 = {f.name for f in _f(声线档案)}
        x = {k: v for k, v in dict(d).items() if k in 允许}
        x["共振峰缩放"] = tuple(float(v) for v in x.get("共振峰缩放", (1.0, 1.0, 1.0)))
        f = 声线档案(**x)
        f.校验()
        return f

    def 派生(self, **改) -> "声线档案":
        d = self.转dict(); d.update(改); d["共振峰缩放"] = tuple(d["共振峰缩放"])
        return 声线档案.从dict(d)


@dataclass(frozen=True)
class 角色模板:
    """行层唯一输入：角色模板 = 角色包名 + 声线名（只写名字，不持有声层对象）。"""
    名: str
    版本: str = "1.0"
    来源: str = "内生·参数化"
    角色包名: str = ""
    声线名: str = ""
    情绪默认: str = "平静"
    句数: int = 3
    腔调: tuple = ()          # (("降调半音",-0.5),("句末拖音",0.2),...)
    台词: tuple = ()          # 演示台词（空=用角色包池按状态选句；非空=按序取，仅 demo）
    备注: str = ""

    def 指纹(self) -> str:
        return hashlib.sha256(json.dumps(
            dict(self.__dict__, 腔调=[list(x) for x in self.腔调],
                 台词=list(self.台词)),
            ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]

    @staticmethod
    def 从dict(d: Mapping[str, Any]) -> "角色模板":
        from dataclasses import fields as _f
        允许 = {f.name for f in _f(角色模板)}
        x = {k: v for k, v in dict(d).items() if k in 允许}
        x["腔调"] = tuple(tuple(v) for v in x.get("腔调", ()))
        x["台词"] = tuple(str(v) for v in x.get("台词", ()))
        return 角色模板(**x)


# ───────────────────────────── 产物（跨层唯一数据） ─────────────────────────────
class 文本产物(TypedDict, total=False):
    文本: str; 拍: int; 角色: str; 模板: str; 模板指纹: str
    句数: int; 记忆命中: str; 来源: str; 降级: bool; 错: str


class 渲染产物(TypedDict, total=False):
    文本: str; 采样率: int; 样本数: int; 时长秒: float
    波形: Any; wav: str; 声线: str; 声线指纹: str; 状态指纹: str; 种子: int
    参数: dict; 律: dict; 自检: dict; 实时率: float
    降级: bool; 错: str; 溯源: dict


# ───────────────────────────── 三协议 ─────────────────────────────
@runtime_checkable
class 状态源协议(Protocol):
    协议版本: str
    D核: int
    def 快照(self, 拍: int | None = None) -> 状态帧: ...
    def 轨迹指纹(self) -> str: ...
    def 关(self) -> None: ...


@runtime_checkable
class 角色逻辑协议(Protocol):
    协议版本: str
    名: str
    模板名: str
    def 装(self, 模板: 角色模板, 源: 状态源协议, 种子: int = 0) -> None: ...
    def 卸(self) -> None: ...
    def 步(self, 帧: 状态帧) -> 文本产物: ...


@runtime_checkable
class 音色渲染协议(Protocol):
    协议版本: str
    def 渲染(self, 文本: str, *, 声线: 声线档案, 情绪: str = "平静",
             状态: 状态帧 | None = None, 语速倍: float = 1.0,
             音量倍: float = 1.0, 种子: int = 默认种子,
             腔调: tuple = ()) -> 渲染产物: ...
    def 预热(self, 声线: 声线档案) -> None: ...


# ───────────────────────────── 契约自检 ─────────────────────────────
def 自检() -> dict:
    项 = []
    def 查(名, 条件, 细节=""):
        项.append(dict(项=名, 通过=bool(条件), 细节=str(细节)))
    查("契约版本", 契约版本 == "role-voice-1.0", 契约版本)
    查("拍长秒", abs(拍长秒 - 0.1) < 1e-12, 拍长秒)
    查("错误码齐全", len(错误说明) == 12, len(错误说明))
    a = 声线档案(名="自检", 基频=200.0)
    a.校验()
    查("声线档案可哈希", isinstance(hash(a), int))
    查("声线档案指纹", len(a.指纹()) == 16, a.指纹())
    查("声线档案非法被拒", _拒(lambda: 声线档案(名="坏", 基频=999.0).校验(), E1006))
    t = 角色模板(名="自检", 角色包名="洛天依", 声线名="洛天依")
    查("角色模板指纹", len(t.指纹()) == 16, t.指纹())
    帧 = 状态帧.建(拍=0, 观测=1, o用=[0.1, 0.2], 内心=[0.3, 0.4], 遥测={"能量": 0.5})
    查("状态帧指纹", len(帧.指纹) == 16, 帧.指纹)
    查("状态帧只读", _拒(lambda: setattr(帧, "拍", 9), Exception))
    查("状态帧可哈希", isinstance(hash(帧), int))
    查("协议结构", all([isinstance(状态源协议, type), isinstance(角色逻辑协议, type),
                        isinstance(音色渲染协议, type)]))
    return dict(通过=all(x["通过"] for x in 项), 项=项)


def _拒(fn, 期望) -> bool:
    try:
        fn()
    except 契约错误 as e:
        if isinstance(期望, str):
            return e.码 == 期望
        return isinstance(期望, type) and isinstance(e, 期望)
    except Exception as e:  # noqa: BLE001
        return isinstance(期望, type) and isinstance(e, 期望)
    return False


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    结 = 自检()
    for x in 结["项"]:
        print(("PASS " if x["通过"] else "FAIL ") + x["项"], x["细节"])
    print("总判:", 结["通过"])
