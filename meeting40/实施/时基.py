# -*- coding: utf-8 -*-
"""meeting40/实施/时基.py —— P0 [正交] 拍钟：t_mono / t_pai / τ*（唯一口径 = 终稿 §8.2）。

【口径，不得擅自改动；改动须回到所有者裁决 §12-6】
    τ  = t_pai = floor((t_mono − t0) / 0.1)      ⇒ 1 拍 = 100ms（10Hz）
    τ* =          (t_mono − t0) / 0.1            ⇒ **分数拍**（判决量；报表单位「拍」）
    · 150ms 与 40ms **不在整数拍栅格上** ⇒ 判决一律用 τ*，不得用整数拍硬判定。
    · 换算表（§8.2a）：40ms=0.4拍｜100ms=1拍｜150ms=1.5拍｜200ms=2拍｜300ms=3拍｜500ms=5拍。
    · 33ms/67ms 是**视觉帧/端到端单拍性能**口径，与拍钟解耦，不得混用（§8.2a ⚠）。

【禁 wall-clock 作判据（C-R2-10 / C-R3-9）】
    t_mono 只来自：① `time.perf_counter()`（单调）② 注入时钟 ③ 重放序列（`回放钟`）。
    `time.time()` 永不参与 τ / 延迟判决——本模块**不 import time.time 的调用路径**，
    仅在诊断字段里由调用方自行附加。判据 `判据_时基()` 的第 ③ 条会**把 time.time 打坏**来证这点。

【8 个埋点（§8.2b）——P0 建表；t1 侧文件尚不存在者如实标「未接」，不伪造数值】
    τ_in 输入事件合成拍(通用 t0) ｜ τ_R R契约交付拍 ｜ τ_q 文本首字入表达槽拍
    τ_arb 仲裁提交拍 ｜ τ_f0 语音首帧入环拍 ｜ τ_act 动作提交拍(_执行键)
    τ_stop 停声拍 ｜ τ_fb 反馈入槽拍 ｜（另 τ_intr 打断入槽拍，供打断判据 t0）

纪律：纯新增文件；不 import 任何器官；不写共享槽；无全局随机态（C13）。
"""
from __future__ import annotations

import hashlib
import json
import time

拍长秒 = 0.1                      # 1 拍 = 100ms（10Hz）——唯一换算依据：终稿 §8.2a
一拍的秒 = 拍长秒                  # 兼容别名

# 埋点登记表（顺序即 §8.2b 的表格顺序；P0 只建表，不伪造未接埋点的数值）
埋点名 = ("τ_in", "τ_R", "τ_q", "τ_arb", "τ_f0", "τ_act", "τ_stop", "τ_fb", "τ_intr")

# 已具备落点的埋点（P0 可实测）与必须等 P3 组件才存在的埋点（如实标「未接」）
落点已存在 = {"τ_in", "τ_R", "τ_q", "τ_act", "τ_fb", "τ_intr"}
落点未存在 = {"τ_arb", "τ_f0", "τ_stop"}


# ─────────────────────────── 换算（纯函数，确定性） ───────────────────────────
# ★用**整数纳秒**做所有换算：拍长 = 100ms = 100_000_000 ns。
#   原因：0.15/0.1 在二进制浮点下 = 1.4999999999999998，会让「150ms = 1.5 拍」这类
#   栅格判决与换算表对不上。定点整数换算在 C13 下同样逐位可复现，且十进制栅格精确。
拍纳秒 = 100_000_000               # 1 拍 = 100ms（ns）


def 纳秒(秒: float) -> int:
    """秒 → 整数纳秒（round-half-even，确定性；输入有限即可）。"""
    return int(round(float(秒) * 1_000_000_000))


def 秒转拍(秒: float) -> float:
    """秒 → 分数拍 τ*（定点换算：0.15s → 1.5；向下取整交给 `分数拍转整数拍`）。"""
    return 纳秒(秒) / float(拍纳秒)


def 拍转秒(拍: float) -> float:
    """分数拍 → 秒（定点：1.5 拍 → 0.15s）。"""
    return (int(round(float(拍) * 拍纳秒))) / 1_000_000_000.0


def 分数拍转整数拍(分数拍: float) -> int:
    """floor(τ*)：分数拍 → 整数拍（与 τ=t_pai 的定义逐字一致）。"""
    import math
    return int(math.floor(float(分数拍)))


def 换算表() -> list:
    """§8.2a 换算表：[(毫秒, 分数拍, 是否落在整数拍栅格)]。"""
    return [(ms, 秒转拍(ms / 1000.0), abs(秒转拍(ms / 1000.0) % 1.0) < 1e-9)
            for ms in (40, 100, 150, 200, 300, 500)]


def 特征码(对象) -> str:
    """确定性特征码：规范 JSON（键排序、ensure_ascii=False）→ sha256。
    用于「同一固定序列两次运行 → 逐位同」的判据（C13）。"""
    return hashlib.sha256(
        json.dumps(对象, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                   default=str).encode("utf-8")).hexdigest()


# ─────────────────────────── 注入式时钟（重放/C13） ───────────────────────────
class 回放钟:
    """固定时间序列时钟：逐次调用按序返回；序列耗尽后停在末值（幂等，不抛）。

    用途：C13 确定性重放——同一序列喂两次，τ/埋点/特征码必须逐位相同。
    这只给**单调**值（调用方应传 perf_counter 口径的序列），不是墙钟。
    """

    def __init__(self, 序列):
        self.序列 = [float(x) for x in 序列]
        self.次数 = 0

    def __call__(self) -> float:
        if not self.序列:
            return 0.0
        v = self.序列[min(self.次数, len(self.序列) - 1)]
        self.次数 += 1
        return v

    def 归零(self):
        self.次数 = 0
        return self


# ─────────────────────────── 拍钟 ───────────────────────────
class 拍钟:
    """单调拍钟。t0 = 首次取时（或显式传入）；一切判决量都是它导出的 τ/τ*。

    用法：
        钟 = 拍钟()                    # t0 = 首次 t_mono
        钟.打点("τ_in"); ... ; 钟.打点("τ_q")
        钟.延迟("τ_in", "τ_q")          # → dict(毫秒=, 拍=τ*, 整数拍=, 起=, 止=)
        len(钟)  # 打分点数
    """

    def __init__(self, t0: float = None, 钟=None, 起点拍: int = 0):
        self._钟 = 钟 if 钟 is not None else time.perf_counter
        self.t0 = None if t0 is None else float(t0)
        self.起点拍 = int(起点拍)
        self.点: dict = {}                 # 名 → dict(名,t_mono,分数拍,拍)
        self.序: list = []                 # 登记顺序（审计可读）

    # ── 取时 ──
    def t_mono(self) -> float:
        t = float(self._钟())
        if self.t0 is None:
            self.t0 = t
        return t

    def 分数拍(self, t: float = None) -> float:
        t = self.t_mono() if t is None else float(t)
        if self.t0 is None:
            self.t0 = t
        return 秒转拍(t - self.t0)

    def 拍(self, t: float = None) -> int:
        """整数拍 = floor(τ*)；再叠加 `起点拍`（分段重放时把绝对拍接上）。"""
        return 分数拍转整数拍(self.分数拍(t)) + self.起点拍

    # ── 埋点 ──
    def 打点(self, 名: str, t: float = None) -> dict:
        名 = str(名)
        if 名 not in 埋点名:
            raise KeyError("未登记的埋点名：%r（合法：%s）" % (名, "、".join(埋点名)))
        t = self.t_mono() if t is None else float(t)
        if self.t0 is None:
            self.t0 = t
        τ_ = 秒转拍(t - self.t0)
        d = dict(名=名, t_mono=t, 分数拍=τ_, 拍=分数拍转整数拍(τ_) + self.起点拍)
        self.点[名] = d
        self.序.append(名)
        return dict(d)

    def 取埋点(self, 名: str) -> dict:
        d = self.点.get(str(名))
        return None if d is None else dict(d)

    def 全部埋点(self) -> dict:
        return {k: dict(v) for k, v in self.点.items()}

    def 未接埋点(self) -> list:
        """还没打过的埋点（P0 的诚实字段：`落点未存在` 的天然进这里）。"""
        return [n for n in 埋点名 if n not in self.点]

    # ── 判决量 ──
    def 延迟(self, 起, 止) -> dict:
        """两埋点之间 → dict(毫秒=, 拍=τ*, 整数拍=, 起=, 止=)。
        `起`/`止` 可为埋点名，或含 `t_mono` 的 dict（便于用外部记账的时刻）。"""
        def _t(x):
            if isinstance(x, str):
                d = self.点.get(x)
                if d is None:
                    raise KeyError("埋点未打：%r" % x)
                return d["t_mono"], x
            d = dict(x)
            return float(d["t_mono"]), str(d.get("名", "?"))
        t0, n0 = _t(起)
        t1, n1 = _t(止)
        差ns = 纳秒(t1) - 纳秒(t0)
        τ_ = 差ns / float(拍纳秒)
        return dict(毫秒=round(差ns / 1_000_000.0, 6), 拍=τ_, 整数拍=分数拍转整数拍(τ_),
                    起=n0, 止=n1)

    def 特征码(self) -> str:
        """埋点表 + t0/起点拍 的确定性特征码（重放判据用）。"""
        return 特征码(dict(t0=self.t0, 起点拍=self.起点拍, 点=self.全部埋点()))

    def 状态(self) -> dict:
        return dict(t0=self.t0, 起点拍=self.起点拍, 已打点=list(self.序),
                    未接=tuple(self.未接埋点()), 拍长秒=拍长秒, 口径="τ=floor((t_mono−t0)/0.1)")

    def __len__(self) -> int:
        return len(self.点)


# ─────────────────────────── 自测（轻量；完整判据见 判据_P0.py） ───────────────────────────
def 自测() -> dict:
    def _跑一遍():
        钟 = 拍钟(t0=100.0, 钟=回放钟([100.0, 100.15, 100.30, 100.50]))
        for n in ("τ_in", "τ_q", "τ_act", "τ_fb"):
            钟.打点(n)
        return 钟

    钟, 钟2 = _跑一遍(), _跑一遍()
    栅格 = 换算表()
    分数拍对 = (钟.取埋点("τ_q")["分数拍"] == 1.5)          # 150ms = 1.5 拍（非整数栅格）
    整数拍对 = (钟.取埋点("τ_q")["拍"] == 1)               # floor(1.5) = 1
    重放确定 = bool(钟.特征码() == 钟2.特征码()
                  and 钟.全部埋点() == 钟2.全部埋点())
    延 = 钟.延迟("τ_in", "τ_q")
    延迟对 = bool(延["毫秒"] == 150.0 and 延["拍"] == 1.5 and 延["整数拍"] == 1)
    回环对 = bool(拍转秒(秒转拍(0.15)) == 0.15 and 秒转拍(拍转秒(1.5)) == 1.5)
    return dict(通过=bool(分数拍对 and 整数拍对 and 重放确定 and 延迟对 and 回环对),
                分数拍对=分数拍对, 整数拍对=整数拍对, 重放确定=重放确定,
                延迟对=延迟对, 秒拍回环=回环对, 延迟=延,
                换算表=栅格, 状态=钟.状态())


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 自测()
    print(json.dumps(果, ensure_ascii=False, indent=1, default=str))
    print("[时基] 自测：%s" % ("PASS" if 果["通过"] else "FAIL"))
    raise SystemExit(0 if 果["通过"] else 1)
