# -*- coding: utf-8 -*-
"""meeting40/实施/同源事件.py —— 备B 输入侧：`SensorFrame` 时间戳适配器 + 同源事件绑定（±100ms）。

口径（终稿 §5 备B「输入」条 + §7.1 图 + §7.3 数据流）：
    「用 `SensorFrame` 时间戳适配器包装既有取视觉/取听觉；只加**同源事件绑定**与 L0/L1 写入；
      **不做原生音/视频前端**。」
    原子形状（§7.1）：`⟨模态, 内容, τ, prov, conf⟩`，同源窗 **±100ms = ±1 拍**（§8.2：1 拍 = 100ms）。

两条能力（**都不改任何既有件**，纯新增、只读既有槽）：

  ① `取视觉帧()` / `取听觉帧()`：把 `态.取视觉()` / `态.取听觉()` 的**当下内容**包成 `SensorFrame`
     （模态/t_mono/seq/形状/dtype/数据/标定/质量/来源/sha256）。
     ⚠ 诚实边界：数据是**既有槽内容的规范摘要**（结构化元素/听觉遥测），**不是原生像素/波形**——
     备B 明确不做原生前端；原生前端属主A/C。时间戳**必须由 `时基.拍钟` 提供**（禁 wall-clock）。

  ② `同源绑定`：把 ±1 拍窗内的多模态事件绑成一个**同源原子**。规则（确定性、可失败）：
     · 基准 τ = 组内**最早**事件；窗 = `[基准, 基准 + 窗拍]`；
     · `conf` 取组内**最小**（保守）；`效价` 取组内**首个**（不做跨模态加权——备B 不引入新超参）；
     · `prov` 合并：`prov_org=外源` 只要**任一**成分外源；`prov_route=实感` 要求**全部**实感；
     · `可写M = all(成分可写记忆)` ⇒ **含一个「自说/输出回灌」成分，整原子就不许写 M**（§7.6 唯一写闸）。

纪律：纯新增；只读既有槽（不改槽内容）；无全局随机态；同序列 ⇒ 逐位同（C13）。
"""
from __future__ import annotations

from collections import deque

import 事件契约 as 契
import 时基

拍长秒 = 0.1
同源窗拍 = 1.0                      # ±100ms = ±1 拍（§8.2 拍长 100ms）
_模态序 = {m: i for i, m in enumerate(契.模态枚举)}

契.注册模态("同源")                  # 绑定后的原子用「同源」模态名（附加枚举，不越红线）


# ─────────────────────────── ① SensorFrame 时间戳适配器 ───────────────────────────
def _摘要(对象) -> bytes:
    """槽内容 → 规范字节（确定性；同内容 ⇒ 同字节）。"""
    return 契.规范字节(对象)


def 取视觉帧(态, 拍钟=None, 来源: str = "器官_眼/模块三(UIA+光适应)") -> "契.SensorFrame":
    """既有视觉槽 → `SensorFrame`（**只读**）。`拍钟` 不给 ⇒ t_mono=0.0（调用方须给拍钟才有时基）。"""
    v = dict(态.取视觉() or {})
    元素s = list(v.get("元素s") or [])
    屏 = tuple(int(x) for x in (v.get("屏幕") or (0, 0)))
    t = float(拍钟.t_mono()) if 拍钟 is not None else 0.0
    序 = int(v.get("帧号") or 0)
    数据 = _摘要([[e.get("句柄"), list(e.get("框") or ()), e.get("字"), list(e.get("主色") or ())]
                  for e in 元素s])
    return 契.SensorFrame(模态="视觉", t_mono=t, seq=序, 形状=(len(元素s),) + (屏 or (0, 0)),
                          dtype="结构化元素", 数据=数据,
                          标定=dict(焦点=v.get("焦点"), 变化率=v.get("变化率")),
                          质量=1.0, 来源=来源)


def 取听觉帧(态, 拍钟=None, 来源: str = "器官_耳(WASAPI 回环)") -> "契.SensorFrame":
    """既有听觉槽 → `SensorFrame`（**只读**）。"""
    a = dict(态.取听觉() or {})
    t = float(拍钟.t_mono()) if 拍钟 is not None else 0.0
    事件s = list(a.get("事件s") or [])
    数据 = _摘要([a.get("最近主频"), a.get("最近能量"), a.get("最近起音t"), len(事件s)])
    return 契.SensorFrame(模态="听觉", t_mono=t, seq=len(事件s), 形状=(len(事件s),),
                          dtype="听觉遥测", 数据=数据,
                          标定=dict(最近主频=a.get("最近主频"), 最近能量=a.get("最近能量")),
                          质量=1.0, 来源=来源)


# ─────────────────────────── ② 同源事件绑定 ───────────────────────────
class 同源绑定:
    """把落在同一 ±窗拍窗口内的多模态事件绑成一个同源原子（§7.1）。**确定性**。"""

    def __init__(self, 窗拍: float = 同源窗拍, 上限: int = 64):
        self.窗拍 = float(窗拍)
        self.上限 = int(上限)
        self.原子s: list = []
        self._组: list = []
        self._基准: float = None
        self.统计 = dict(投=0, 成原子=0, 冲=0, 成分数=0)

    # ── 内部 ──
    def _并prov(self, 组: list):
        字典s = [e["prov"].字典() for e in 组]
        org = "外源" if any(d["prov_org"] == "外源" for d in 字典s) else "自源"
        route = "实感" if all(d["prov_route"] == "实感" for d in 字典s) else \
            next(d["prov_route"] for d in 字典s if d["prov_route"] != "实感")
        acq = 字典s[0]["prov_acq"] if all(d["prov_acq"] == 字典s[0]["prov_acq"] for d in 字典s) \
            else "未知"
        exec域 = 字典s[0].get("prov_exec") if all(d.get("prov_exec") == 字典s[0].get("prov_exec")
                                              for d in 字典s) else None
        return 契.Provenance(prov_acq=acq, prov_org=org, prov_route=route, prov_exec=exec域), 字典s

    def _成原子(self, 组: list) -> dict:
        组 = sorted(组, key=lambda e: (float(e["τ"]), _模态序.get(e["模态"], 99), str(e["句柄"])))
        p, 字典s = self._并prov(组)
        可 = [契.可写记忆(契.Provenance.从字典(d)) for d in 字典s]
        原子 = dict(类型="同源原子",
                    模态=tuple(e["模态"] for e in 组),
                    内容=tuple((e["模态"], str(e["句柄"])) for e in 组),
                    τ=float(组[0]["τ"]),
                    t_pai=时基.分数拍转整数拍(float(组[0]["τ"])),
                    conf=min(float(e["conf"]) for e in 组),
                    效价=float(组[0]["效价"]),
                    prov=p.字典(), provs=字典s, 成分数=len(组),
                    可写M=bool(all(可)))
        self.原子s.append(原子)
        self.统计["成原子"] += 1
        self.统计["成分数"] += len(组)
        return 原子

    # ── 投事件 ──
    def 投(self, 模态: str, 内容句柄: str, τ: float, prov=None, conf: float = 1.0,
          效价: float = 0.0) -> dict:
        """投一个模态事件。**返回刚闭合的原子**（还在窗口内则返回 None，等后续事件或 `冲()`）。"""
        p = prov if isinstance(prov, 契.Provenance) else 契.Provenance.从字典(prov or {})
        e = dict(模态=str(模态), 句柄=str(内容句柄), τ=float(τ), prov=p,
                 conf=float(conf), 效价=float(效价))
        self.统计["投"] += 1
        if not self._组:
            self._组, self._基准 = [e], e["τ"]
            return None
        if e["τ"] - self._基准 <= self.窗拍:           # 同源窗内 ⇒ 并入
            self._组.append(e)
            if len(self._组) >= self.上限:
                return self._封组()
            return None
        return self._封组(e)                            # 窗外 ⇒ 先封旧组，新事件起新组

    def _封组(self, 新事件: dict = None) -> dict:
        原子 = self._成原子(self._组)
        self._组 = []
        self._基准 = None
        if 新事件 is not None:
            self._组, self._基准 = [新事件], 新事件["τ"]
        return 原子

    def 冲(self) -> list:
        """把挂起组封掉（序列结束时调）。返回本次新成的原子列表。"""
        出 = []
        if self._组:
            出.append(self._封组())
        self.统计["冲"] += 1
        return 出

    def 取原子s(self) -> list:
        return [dict(x) for x in self.原子s]

    def 统计表(self) -> dict:
        d = dict(self.统计)
        d["窗拍"] = self.窗拍
        d["挂起"] = len(self._组)
        d["口径"] = "基准τ=组内最早；窗=[基准, 基准+%.2f拍]；可写M=all(成分可写)" % self.窗拍
        return d


def 造输入事件(原子: dict, seq: int = 0) -> "契.InputEvent":
    """同源原子 → `InputEvent`（模态=「同源」；内容句柄=各模态句柄的规范摘要）。"""
    return 契.InputEvent(τ=float(原子["τ"]), seq=int(seq), 模态="同源",
                         内容句柄=契.sha256(list(原子["内容"]))[:16],
                         prov=契.Provenance.从字典(原子["prov"]), conf=float(原子["conf"]),
                         效价=float(原子["效价"]))


# ─────────────────────────── 自测 ───────────────────────────
def 自测() -> dict:
    实 = 契.Provenance(prov_acq="网页", prov_org="外源", prov_route="实感")
    自说 = 契.Provenance(prov_acq="自源", prov_org="自源", prov_route="自说")
    案 = {}
    # ① 同窗（差 0.5 拍 / 1.0 拍）⇒ 绑成一个原子
    b = 同源绑定(窗拍=1.0)
    b.投("视觉", "E1", 3.0, prov=实, conf=0.9)
    b.投("听觉", "A1", 3.5, prov=实, conf=0.7)
    a1 = b.投("文本", "T1", 4.0, prov=实, conf=0.8)        # 4.0-3.0 = 1.0 ≤ 窗 ⇒ 仍并入
    a1 = a1 or b.冲()[0]
    案["同窗绑定"] = (a1["成分数"] == 3 and a1["模态"] == ("视觉", "听觉", "文本")
                  and a1["conf"] == 0.7 and a1["可写M"] is True)
    # ② 窗外（差 1.01 拍）⇒ 分成两个原子
    b2 = 同源绑定(窗拍=1.0)
    b2.投("视觉", "E1", 3.0, prov=实)
    a2 = b2.投("听觉", "A1", 4.01, prov=实)               # 4.01-3.0 = 1.01 > 窗 ⇒ 封旧组
    案["窗外不绑"] = (a2 is not None and a2["成分数"] == 1 and a2["模态"] == ("视觉",))
    案["窗外_新组挂起"] = (b2.统计表()["挂起"] == 1)
    # ③ 相隔 3 拍 ⇒ 两个原子
    b3 = 同源绑定(窗拍=1.0)
    b3.投("视觉", "E1", 1.0, prov=实)
    b3.投("听觉", "A1", 4.0, prov=实)
    b3.冲()
    案["远隔两原子"] = (len(b3.原子s) == 2)
    # ④ 含一个「自说」成分 ⇒ 整原子不可写 M（§7.6）
    b4 = 同源绑定(窗拍=1.0)
    b4.投("视觉", "E1", 2.0, prov=实)
    原 = b4.投("文本", "T9", 2.4, prov=自说) or b4.冲()[0]
    案["自说成分⇒拒写"] = (原["可写M"] is False and 原["prov"]["prov_route"] == "自说")
    # ⑤ 确定性：同序列两次 ⇒ 逐位同
    def _跑():
        z = 同源绑定(窗拍=1.0)
        z.投("视觉", "E1", 2.0, prov=实, conf=0.9)
        z.投("听觉", "A1", 2.3, prov=实, conf=0.8)
        z.冲()
        return z.原子s
    案["重放逐位同"] = (契.sha256(_跑()) == 契.sha256(_跑()))
    # ⑥ 原子 → InputEvent（模态=同源；禁字段校验仍然生效）
    b5 = 同源绑定()
    b5.投("视觉", "E7", 1.2, prov=实, conf=0.6)
    ev = 造输入事件(b5.冲()[0], seq=5)
    案["造事件"] = (ev.模态 == "同源" and ev.t_pai == 1 and abs(ev.conf - 0.6) < 1e-3)
    return dict(通过=bool(all(案.values())), 逐项=案, 统计=b.统计表())


if __name__ == "__main__":
    import json
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 自测()
    print(json.dumps(果, ensure_ascii=False, indent=1, default=str))
    print("[同源事件] 自测：%s" % ("PASS" if 果["通过"] else "FAIL"))
    raise SystemExit(0 if 果["通过"] else 1)
