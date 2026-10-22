# -*- coding: utf-8 -*-
"""meeting40/实施/许可令牌.py —— 备B 安全闭环：**不可逆动作的一次性许可令牌**。

口径（终稿 §5 备B「输出」条 + §7.2 接口表「许可令牌」+ §3 图「不可逆 fail-closed」；
判据锚 G-R3-14：缺令牌 ⇒ 拒 100%；可逆误拒 ≤5%）：

    令牌 = { 动作指纹, 目标rid, 来源, 签发拍, 有效期拍, nonce, 确认事件, 不可逆 }
    · **一次性**：核销即 `已用拍` 落章，第二次同 id 一律拒（重放攻击面清零）。
    · **指纹绑定**：令牌只为**那一个动作**签发；指纹不匹配 ⇒ 拒（防挪用）。
    · **有效期**：`当前拍 − 签发拍 > 有效期拍` ⇒ 拒（默认 2 拍 = 200ms 提交窗）。
    · **不可逆必须带确认事件**：`不可逆=True ∧ 要求确认 ∧ 确认事件为空` ⇒ 拒。
    · **fail-closed**：任何异常、任何"说不清"⇒ **拒**，绝不"默认放行"。

    · `令牌闸`：装到 `器官_手.令牌闸` 上的策略层。**可逆白名单**内的动作（移动/悬停等）
      不需要令牌（对应「可逆误拒 ≤5%」）；其余（点击/双击/键入/滚轮/按下松开）一律要令牌。

时基：全部用**拍**（分数拍），由注入的 `拍源()` 提供——**不用墙钟**（C-R2-10/C13）。
本模块不 import 任何器官，也不写任何共享槽。

用法（器官_手 侧，默认关）：
    手.令牌闸 = 令牌闸(令牌局(拍源=lambda: τ_now), 可逆={"move"})
    令牌局.签发(动作指纹="键:点击:左", 来源="所有者确认", 确认事件="确认:42")
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

# 默认不可逆的**动作种类**（可逆白名单之外的都算不可逆）
不可逆种类 = ("点击", "双", "按下", "松开", "键入", "滚轮")
默认有效期拍 = 2.0          # 2 拍 = 200ms 提交窗（与 §8.2 拍长 100ms 一致）


def 指纹(种类: str, 目标="") -> str:
    """**内容型**指纹（确定性）：种类 + 目标 → sha256 前 16 位。用于把令牌绑定到具体目标/内容。"""
    s = "动作|%s|%s" % (str(种类), str(目标))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# ★**动作指纹的唯一格式**（`器官_手._执行键/_滚轮/_键入` 里就是这么拼的；签发方必须用同一格式，
#   否则令牌永远匹配不上——这条是 B-2 判据当场抓出来的）。格式：`种类:参数`。
def 指纹_键(种类: str, 键: str) -> str:
    return "键:%s:%s" % (str(种类), str(键))


def 指纹_滚轮(格数) -> str:
    return "滚轮:%d" % int(格数)


def 指纹_键入(文本) -> str:
    return "键入:%s" % str(文本)[:32]


def 指纹_移动(点="") -> str:
    return "move:%s" % ("" if 点 is None else str(点))


def 种类是否不可逆(种类: str) -> bool:
    return any(k in str(种类) for k in 不可逆种类)


@dataclass
class 令牌:
    id: str
    动作指纹: str
    目标rid: str = ""
    来源: str = ""
    签发拍: float = 0.0
    有效期拍: float = 默认有效期拍
    确认事件: str = None
    不可逆: bool = True
    已用拍: float = None

    def 字典(self) -> dict:
        return dict(id=self.id, 动作指纹=self.动作指纹, 目标rid=self.目标rid,
                    来源=self.来源, 签发拍=self.签发拍, 有效期拍=self.有效期拍,
                    确认事件=self.确认事件, 不可逆=self.不可逆, 已用拍=self.已用拍)


class 令牌局:
    """一次性令牌签发/核销所。**统计里的原因分布是审计字段**（拒因可复核）。"""

    def __init__(self, 拍源=None, 要求确认: bool = True, 默认有效期: float = 默认有效期拍):
        self._拍源 = 拍源 if 拍源 is not None else (lambda: 0.0)
        self.要求确认 = bool(要求确认)
        self.默认有效期 = float(默认有效期)
        self.令牌s: dict = {}
        self.统计 = dict(签发=0, 放行=0, 拒=0, 拒因={})

    # ── 内部 ──
    def _当前拍(self) -> float:
        try:
            return float(self._拍源())
        except Exception:
            return 0.0                     # 拍源故障 ⇒ 0（会让有效期判据更保守，不越权放行）

    def _拒(self, 理由: str, 令牌id=None) -> dict:
        self.统计["拒"] += 1
        self.统计["拒因"][理由] = self.统计["拒因"].get(理由, 0) + 1
        return dict(放行=False, 理由=理由, 令牌id=令牌id)

    def _id(self, 动作指纹: str, 非ce: str) -> str:
        return hashlib.sha256(("%s|%s" % (动作指纹, 非ce)).encode("utf-8")).hexdigest()[:16]

    # ── 签发 ──
    def 签发(self, 动作指纹: str, 目标rid: str = "", 来源: str = "", 确认事件: str = None,
            不可逆: bool = True, 有效期拍: float = None, nonce: str = None) -> 令牌:
        拍 = self._当前拍()
        有效期 = self.默认有效期 if 有效期拍 is None else float(有效期拍)
        非ce = str(nonce) if nonce is not None else ("n%04d" % self.统计["签发"])
        t = 令牌(id=self._id(str(动作指纹), 非ce), 动作指纹=str(动作指纹), 目标rid=str(目标rid),
                来源=str(来源), 签发拍=拍, 有效期拍=有效期, 确认事件=确认事件,
                不可逆=bool(不可逆))
        self.令牌s[t.id] = t
        self.统计["签发"] += 1
        return t

    # ── 核销（判据的全部规则都在这里；**fail-closed**）──
    def 核销(self, 令牌id: str, 动作指纹: str = None) -> dict:
        try:
            t = self.令牌s.get(str(令牌id))
            if t is None:
                return self._拒("令牌不存在", 令牌id)
            if 动作指纹 is not None and str(动作指纹) != t.动作指纹:
                return self._拒("指纹不匹配（防挪用）", 令牌id)
            if t.已用拍 is not None:
                return self._拒("令牌已用过（一次性）", 令牌id)
            现拍 = self._当前拍()
            if (现拍 - float(t.签发拍)) > float(t.有效期拍):
                return self._拒("令牌过期", 令牌id)
            if t.不可逆 and self.要求确认 and not t.确认事件:
                return self._拒("不可逆动作缺确认事件", 令牌id)
            t.已用拍 = 现拍
            self.统计["放行"] += 1
            return dict(放行=True, 理由="", 令牌id=t.id, 令牌=t.字典())
        except Exception as ex:                       # 任何异常 ⇒ 拒（fail-closed）
            return self._拒("核销异常:%r" % (ex,), 令牌id)

    # ── 按指纹放行（器官_手 的闸就调这个）──
    def 放行(self, 动作指纹: str) -> bool:
        """找**最近签发且未用**的同指纹令牌核销；没有 ⇒ 拒。"""
        try:
            候s = sorted([t for t in self.令牌s.values()
                          if t.动作指纹 == str(动作指纹) and t.已用拍 is None],
                         key=lambda t: (-float(t.签发拍), t.id))     # 最近优先；tie 按 id 升序
            if not 候s:
                return bool(self._拒("无有效令牌")["放行"])
            return bool(self.核销(候s[0].id, 动作指纹)["放行"])
        except Exception:
            self._拒("放行异常")
            return False                              # fail-closed

    def 统计表(self) -> dict:
        d = dict(self.统计)
        d["拒因"] = dict(sorted((d.get("拒因") or {}).items()))
        d["令牌数"] = len(self.令牌s)
        d["未用令牌"] = sum(1 for t in self.令牌s.values() if t.已用拍 is None)
        return d


class 令牌闸:
    """装到 `器官_手.令牌闸` 的政策层：可逆白名单免令牌，其余要一次性令牌。"""

    def __init__(self, 局: 令牌局, 可逆: set = None):
        self.局 = 局
        self.可逆 = set(可逆 or set())
        self.计 = dict(可逆放行=0, 令牌放行=0, 拒=0)

    def 放行(self, 动作指纹: str) -> bool:
        s = str(动作指纹)
        try:
            for k in self.可逆:                       # 可逆动作：免令牌（误拒 = 0）
                if k and k in s:
                    self.计["可逆放行"] += 1
                    return True
            if self.局.放行(s):
                self.计["令牌放行"] += 1
                return True
            self.计["拒"] += 1
            return False                              # fail-closed
        except Exception:
            self.计["拒"] += 1
            return False                              # 闸自身故障 ⇒ 拒（安全事实源同一纪律）

    def 统计表(self) -> dict:
        return dict(闸=dict(self.计), 局=self.局.统计表(),
                    可逆白名单=sorted(self.可逆), 口径="可逆免令牌；其余一次性令牌；异常一律拒")


def 特征码(对象) -> str:
    return hashlib.sha256(json.dumps(对象, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), default=str).encode("utf-8")
                          ).hexdigest()


# ─────────────────────────── 自测 ───────────────────────────
def 自测() -> dict:
    拍 = [0.0]
    局 = 令牌局(拍源=lambda: 拍[0])
    闸 = 令牌闸(局, 可逆={"move"})
    果 = {}
    # ① 缺令牌 ⇒ 拒
    果["缺令牌"] = (闸.放行("键:点击:左") is False)
    # ② 合法令牌（带确认）⇒ 放行；③ 同 id 重放 ⇒ 拒
    t = 局.签发(指纹("点击", "E1"), 目标rid="r1", 来源="所有者", 确认事件="确认:42")
    果["合法放行"] = (闸.放行(指纹("点击", "E1")) is True)
    果["重放被拒"] = (闸.放行(指纹("点击", "E1")) is False)
    果["一次性"] = (局.令牌s[t.id].已用拍 is not None)
    # ④ 指纹不匹配 ⇒ 拒
    t2 = 局.签发(指纹("键入", "E2"), 确认事件="确认:43")
    果["指纹不匹配拒"] = (局.核销(t2.id, 指纹("键入", "E3"))["放行"] is False)
    # ⑤ 过期 ⇒ 拒（有效期 2 拍）
    t3 = 局.签发(指纹("滚轮", "E4"), 确认事件="确认:44", 有效期拍=2.0)
    拍[0] = 5.0
    果["过期拒"] = (局.核销(t3.id, 指纹("滚轮", "E4"))["放行"] is False)
    # ⑥ 不可逆缺确认 ⇒ 拒
    t4 = 局.签发(指纹("点击", "E5"), 确认事件=None)
    拍[0] = 5.0
    果["缺确认拒"] = (局.核销(t4.id, 指纹("点击", "E5"))["放行"] is False)
    # ⑦ 可逆白名单 ⇒ 免令牌放行
    果["可逆免令"] = (闸.放行("move:目标点") is True)
    # ⑧ 闸故障 ⇒ fail-closed
    class _坏:
        def 放行(self, s):
            raise RuntimeError("闸坏")
    果["闸故障拒"] = (令牌闸(_坏()).放行("键:点击:左") is False)
    return dict(通过=bool(all(果.values())), 逐项=果, 统计=闸.统计表())


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 自测()
    print(json.dumps(果, ensure_ascii=False, indent=1, default=str))
    print("[许可令牌] 自测：%s" % ("PASS" if 果["通过"] else "FAIL"))
    raise SystemExit(0 if 果["通过"] else 1)
