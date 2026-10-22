# -*- coding: utf-8 -*-
"""meeting40/实施/安全壳.py —— 备B 安全闭环「输出侧」：最小 R 适配 + 真机/演练差分标签 + 删得分。

口径（终稿 §5 备B「输出」条）：
    「保留 SAPI 整句 `嘴` + `器官_手` 现有控制律 + 现有 `生成正文`；仅加**最小 R 适配**
      （置信降级话术）+ **一次性许可令牌** + **真机/演练差分标签** + **删得分**。」
    · `输出层.py` 在 §13.4 **冻结哈希**里 ⇒ **本模块绝不改它**；适配发生在**调用点**（壳包一层）。
    · 令牌在 `许可令牌.py`；本模块只做它的「安装器」（把闸装到 `器官_手` 上）。

三条能力，**全部默认关**（`开关=False` ⇒ 原样返回，旧文本路径逐位不变 —— C14）：

  ① 最小 R 适配（G-R3-24／§7.1 断言级）：
       R 为空 / 冲突>0 / 置信 < `置信下限` ⇒ **降级话术**（固定、确定性、不编内容）；
       医学口径断言级上限 ≤ 回忆（§10）——超限即降级。
  ② 真机/演练差分标签（§7.6 `prov_exec`）：
       每次动作落地记一条 `prov_exec∈{真机, 演练}`；**演练下「真机调用数」恒 0**（可失败判据）。
  ③ 删得分（§7.4b 禁内部得分）：
       出站事件里的 `得分/分数/score/相似/相似度` 键**一律剥除**；话术里的分数片段**一律删掉**。

纪律：纯新增文件；不 import 任何器官（`装()` 只接受外部传进来的手/局）；无全局随机态（C13）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import 事件契约 as 契
import 回忆契约 as 忆

断言级序 = {"事实": 2, "回忆": 1, "推测": 0}

# 出站禁键（§7.4b：输出/输入事件都不得带内部得分）
得分键 = ("得分", "分数", "score", "相似", "相似度", "相似分", "内部得分")
# 话术里的分数片段：如「（相似度 0.87）」「得分:0.9」「similarity=0.87」
_分数片段 = re.compile(r"[（(]?\s*(?:相似度|相似分|相似|得分|分数|score|similarity)"
                       r"\s*[:：=＝]?\s*[01]?\.\d+\s*[)）]?", re.IGNORECASE)

# 降级话术（固定、确定性；只表达「我不确定」，不编事实）
_降级 = {
    "R空": "这事我记不太清了，先不敢乱说。",
    "R冲突": "我这儿有两种说法对不上，先不下定论。",
    "R低置信": "我拿不太准——",
    "超上限": "这我只能说到「我记得」这一层，不敢说死。",
}


def 删得分(事件: dict = None, 文本: str = None) -> dict:
    """剥除内部得分：事件里的得分键删掉；文本里的分数片段删掉。确定性、不改入参。"""
    ev = dict(事件 or {})
    抹 = [k for k in list(ev) if str(k) in 得分键]
    for k in 抹:
        ev.pop(k, None)
    t = "" if 文本 is None else str(文本)
    t2 = _分数片段.sub("", t)
    t2 = re.sub(r"[（(]\s*[)）]", "", t2).strip()
    return dict(事件=ev, 文本=t2, 抹掉键=抹, 文本有改动=(t2 != t))


@dataclass
class 安全壳:
    """输出侧安全壳。**默认关**（`开关=False` ⇒ 所有方法原样返回）。"""
    开关: bool = False
    置信下限: float = 0.5
    台账: list = field(default_factory=list)
    计数: dict = field(default_factory=lambda: dict(真机调用=0, 演练调用=0, 降级=0,
                                                    过壳=0, 拒绝=0, 抹得分=0))
    令牌闸记录: object = None

    # ── ① 最小 R 适配 ──
    def 最小R适配(self, 话: str, R=None) -> dict:
        """返回 dict(文本, 降级, 理由, 断言级)。开关关 ⇒ 原文返回（逐位不变）。"""
        if not self.开关:
            return dict(文本=str(话 or ""), 降级=False, 理由="壳关", 断言级=None)
        self.计数["过壳"] += 1
        级 = None
        try:
            if R is None:
                return self._降("R空", 话, 级)
            级 = R.断言级()
            if not R.可发布():
                理 = "R冲突" if int(getattr(R, "冲突", 0)) > 0 else "R空"
                return self._降(理, 话, 级)
            if float(max((c.置信 for c in R.候选), default=0.0)) < float(self.置信下限):
                return self._降("R低置信", 话, 级)
            if R.超上限(忆.医学断言级上限):
                return self._降("超上限", 话, 级)
            return dict(文本=str(话 or ""), 降级=False, 理由="", 断言级=级)
        except Exception:
            return self._降("R空", 话, 级)              # 异常 ⇒ 降级（fail-safe，不出断言）

    def _降(self, 因: str, 话: str, 级) -> dict:
        self.计数["降级"] += 1
        头 = _降级.get(因, _降级["R空"])
        if 因 == "R低置信":
            文 = 头 + str(话 or "")
        else:
            文 = 头
        return dict(文本=文, 降级=True, 理由=因, 断言级=级)

    # ── ② 真机/演练差分标签 ──
    def 标执行域(self, 事件: dict) -> dict:
        """器官_手 的落地回调：记 `prov_exec` 与计数。**开关关 ⇒ 只记账不写槽**。"""
        真机 = bool((事件 or {}).get("真机"))
        条 = dict(动作=str((事件 or {}).get("动作", "")), prov_exec=("真机" if 真机 else "演练"),
                  拍=(事件 or {}).get("拍"))
        if self.开关:
            self.计数["真机调用" if 真机 else "演练调用"] += 1
            self.台账.append(条)
            if len(self.台账) > 512:
                del self.台账[:256]
        return 条

    def 真机调用数(self) -> int:
        return int(self.计数["真机调用"])

    # ── ③ 出站事件（删得分 + 可选 OutputEvent）──
    def 出站(self, 文本: str, R=None, 通道: str = "文本", 优先级: str = "P4",
            可逆性: bool = True, 令牌id: str = None, 结果: str = None, 拍: float = 0.0,
            seq: int = 0, 事件额外: dict = None) -> dict:
        """一条出站记录：先过最小 R 适配，再删得分；`造事件=True` 时附 `OutputEvent`。"""
        适 = self.最小R适配(文本, R)
        净 = 删得分(事件=dict(事件额外 or {}), 文本=适["文本"])
        if 净["抹掉键"]:
            self.计数["抹得分"] += 1
        出 = dict(文本=净["文本"], 降级=适["降级"], 理由=适["理由"], 断言级=适["断言级"],
                   prov_exec="演练", 事件=净["事件"])
        try:
            出["OutputEvent"] = 契.OutputEvent(
                τ=float(拍), seq=int(seq), 通道=通道, R版本=int(getattr(R, "版本", 0) or 0),
                断言级=(适["断言级"] or "推测"), 内容sha256=契.sha256(净["文本"]),
                优先级=优先级, 可逆性=可逆性, 令牌id=令牌id, 结果=结果,
                prov_exec="演练").字典()
        except Exception as ex:
            出["OutputEvent"] = None
            出["事件异常"] = repr(ex)
        return 出

    # ── 安装器：把手/局接上（默认关时只挂引用，不改变任何行为）──
    def 装(self, 手=None, 令牌闸=None) -> dict:
        果 = dict(手=False, 闸=False)
        if 手 is not None:
            try:
                手.安全壳 = self
                果["手"] = True
            except Exception:
                pass
        if 令牌闸 is not None and 手 is not None:
            try:
                手.令牌闸 = 令牌闸
                self.令牌闸记录 = 令牌闸
                果["闸"] = True
            except Exception:
                pass
        return 果

    def 统计(self) -> dict:
        return dict(开关=bool(self.开关), 计数=dict(self.计数), 台账条数=len(self.台账),
                    置信下限=float(self.置信下限),
                    台账尾=self.台账[-3:], 口径="壳关 ⇒ 原样返回；壳开 ⇒ 降级/差分/删得分")


def 特征码(对象) -> str:
    return 契.sha256(对象)


# ─────────────────────────── 自测 ───────────────────────────
def 自测() -> dict:
    壳关 = 安全壳(开关=False)
    原 = 壳关.最小R适配("我在。", None)
    壳 = 安全壳(开关=True, 置信下限=0.5)
    空R = 壳.最小R适配("我在。", 忆.空R())
    好候 = 忆.R候选(rid="r1", 文本="苏州", 断言级="回忆", 置信=0.9)
    好 = 壳.最小R适配("我在。", 忆.R契约(候选=[好候], 版本=1))
    低 = 壳.最小R适配("我在。", 忆.R契约(候选=[忆.R候选(rid="r2", 断言级="回忆", 置信=0.2)],
                                     版本=1))
    冲 = 壳.最小R适配("我在。", 忆.R契约(候选=[好候], 版本=1, 冲突=1))
    上 = 壳.最小R适配("我在。", 忆.R契约(候选=[忆.R候选(rid="r3", 断言级="事实",
                                                    置信=0.9, 引用=("L1:1",))], 版本=1))
    净 = 删得分(dict(得分=0.87, 文本="a"), "确认了（相似度 0.87）这件事")
    标 = 壳.标执行域(dict(动作="键入", 真机=False))
    return dict(通过=bool(原["文本"] == "我在。" and 原["降级"] is False
                        and 空R["降级"] and 冲["降级"] and 低["降级"] and 上["降级"]
                        and 好["降级"] is False and 好["断言级"] == "回忆"
                        and 净["抹掉键"] == ["得分"] and "0.87" not in 净["文本"]
                        and 标["prov_exec"] == "演练" and 壳.真机调用数() == 0),
                壳关原样=原, 空R=空R["理由"], 冲突=冲["理由"], 低置信=低["理由"],
                超医学上限=上["理由"], 好R=好, 删得分=净, 差分=标, 统计=壳.统计())


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 自测()
    print(json.dumps(果, ensure_ascii=False, indent=1, default=str))
    print("[安全壳] 自测：%s" % ("PASS" if 果["通过"] else "FAIL"))
    raise SystemExit(0 if 果["通过"] else 1)
