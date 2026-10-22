# -*- coding: utf-8 -*-
"""meeting40/实施/测量台.py —— P0 [正交] 可失败测量台（双集 + sham + 确定性重放）。

会议纪律（章程 §1）在代码里的落地：

  ① **双集**：判据集 seed 0–5 ／ 留出集 seed 6–11。留出集**只判不调**——阈值不得看留出集来定。
  ② **sham/零假设对照臂**：主臂过 ∧ 对照臂不过，才记 PASS（防「恒真判据」与「任何机制都能过」）。
  ③ **确定性重放**：同 seed 同输入跑两遍 ⇒ 结果结构特征码**逐位同**（C13）；不同 ⇒ 该判据
     自带不确定源，必须暴露而不是掩盖。
  ④ 本台只**跑与比对**，不改任何既有件；随机性只能来自显式 seed 的 `default_rng`。

用法：
    台 = 测量台("P0")
    台.注册("某判据", 主臂=fn, sham=零假设臂, 门槛=fn主零, 说明="…")
    报 = 台.跑双集(); 台.存("路径.json")
臂签名：`臂(种子) -> dict`，其中 `通过` 键是布尔（其余字段随判据自定，须可 JSON 化）。
"""
from __future__ import annotations

import hashlib
import json
import os

判据集种子 = (0, 1, 2, 3, 4, 5)
留出集种子 = (6, 7, 8, 9, 10, 11)
双集 = {"判据集": 判据集种子, "留出集": 留出集种子}


def _可序列(对象):
    """把结果转成规范 JSON 可序列形式（numpy 标量/数组、tuple、set 一律规范化）。"""
    if hasattr(对象, "tolist"):
        return _可序列(对象.tolist())
    if isinstance(对象, dict):
        return {str(k): _可序列(v) for k, v in sorted(对象.items(), key=lambda kv: str(kv[0]))}
    if isinstance(对象, (list, tuple)):
        return [_可序列(x) for x in 对象]
    if isinstance(对象, (set, frozenset)):
        return sorted(_可序列(x) for x in 对象)
    if isinstance(对象, (str, bool, int, float)) or 对象 is None:
        return 对象
    return str(对象)


def 特征码(对象) -> str:
    """结果结构 → sha256（键排序 + 紧凑分隔符；同值 ⇒ 同码，跨进程/跨 PYTHONHASHSEED 稳定）。"""
    return hashlib.sha256(json.dumps(_可序列(对象), ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), default=str).encode("utf-8")
                          ).hexdigest()


def _默认门槛(主, 零, 种子) -> bool:
    """默认：主臂通过 ∧ 对照臂（若有）**不通过**。"""
    if 零 is None:
        return bool(主.get("通过"))
    return bool(主.get("通过")) and not bool(零.get("通过"))


class 测量台:
    def __init__(self, 名: str = "未命名"):
        self.名 = str(名)
        self.条目: dict = {}

    def 注册(self, 名: str, 主臂, sham=None, 门槛=None, 说明: str = "") -> None:
        名 = str(名)
        if 名 in self.条目:
            raise KeyError("判据重名：%r" % 名)
        self.条目[名] = dict(主臂=主臂, sham=sham, 门槛=(门槛 or _默认门槛), 说明=str(说明))

    # ── 跑一集 ──
    def 跑一集(self, 集, 测确定性: bool = True) -> dict:
        全报 = {}
        for 名 in sorted(self.条目):
            条 = self.条目[名]
            逐 = []
            for 种子 in 集:
                主 = 条["主臂"](种子)
                零 = 条["sham"](种子) if 条["sham"] is not None else None
                确定 = True
                if 测确定性:
                    主2 = 条["主臂"](种子)
                    确定 = (特征码(主) == 特征码(主2))
                    主 = dict(主, 复跑特征码=特征码(主2))
                try:
                    ok = bool(条["门槛"](主, 零, 种子))
                except Exception as ex:
                    ok, 主 = False, dict(主, 门槛异常=repr(ex))
                逐.append(dict(种子=int(种子), 通过=ok, 确定性=bool(确定),
                               主=主, 零=(None if 零 is None else 零)))
            全报[名] = dict(说明=条["说明"], 逐种子=逐,
                            全过=all(x["通过"] and x["确定性"] for x in 逐),
                            门槛全过=all(x["通过"] for x in 逐),
                            确定性全过=all(x["确定性"] for x in 逐),
                            有sham=(条["sham"] is not None))
        return 全报

    def 跑双集(self, 测确定性: bool = True) -> dict:
        报 = {集名: self.跑一集(集, 测确定性=测确定性) for 集名, 集 in 双集.items()}
        总 = all(v["全过"] and v["确定性全过"]
                for 集 in 报.values() for v in 集.values())
        return dict(台=self.名, 种子口径=dict(判据集=list(判据集种子), 留出集=list(留出集种子)),
                    集=报, 总过=bool(总),
                    判据数=len(self.条目),
                    判据名=sorted(self.条目))

    # ── 单点：确定性重放（判据自己也可单独调） ──
    def 重放(self, 名: str, 种子: int = 0, 遍数: int = 2) -> dict:
        臂 = self.条目[str(名)]["主臂"]
        码s = [特征码(臂(种子)) for _ in range(max(2, int(遍数)))]
        return dict(名=str(名), 种子=int(种子), 遍数=len(码s),
                    逐位同=all(c == 码s[0] for c in 码s), 特征码=码s)

    # ── 存 ──
    def 存(self, 路径: str, 报: dict = None) -> dict:
        报 = 报 or self.跑双集()
        报 = dict(报, 本台特征码=特征码(报))
        with open(路径, "w", encoding="utf-8") as f:
            json.dump(报, f, ensure_ascii=False, indent=1, default=str)
        return 报


# ─────────────────────────── 测量台自证（台本身可失败） ───────────────────────────
def _臂过(种子):
    return dict(通过=True, 种子=int(种子), 值=1.0)


def _臂不过(种子):
    return dict(通过=False, 种子=int(种子), 值=0.0)


def _臂主过零不过(种子):
    """像样的主臂：过，且量与种子无关（确定性）。"""
    return dict(通过=True, 种子=int(种子), 效应=0.5 + 0.0 * int(种子))


def _臂不确定(种子):
    """带不确定源的主臂：同 seed 两遍会不同 ⇒ 确定性门必须抓到并判失败。"""
    import random
    return dict(通过=True, 种子=int(种子), 噪声=random.random())


def 本台自证() -> dict:
    """证三件事：① 能给出 FAIL（可失败）② 能抓到不确定源 ③ sham 对照真的起作用。"""
    台 = 测量台("自证")
    台.注册("恒过臂", _臂过)
    台.注册("恒不过臂", _臂不过)
    台.注册("主过_零不过", _臂主过零不过, sham=_臂不过, 说明="标准可失败形态")
    台.注册("主过_零也过", _臂主过零不过, sham=_臂过, 说明="sham 与主臂同过 ⇒ 应判 FAIL")
    台.注册("不确定臂", _臂不确定, 说明="确定性门必须抓到")
    报 = 台.跑双集()
    判 = 报["集"]["判据集"]
    判据集不过 = [n for n, v in 判.items() if not v["全过"]]
    确定不过 = [n for n, v in 判.items() if not v["确定性全过"]]
    好 = (set(判据集不过) == {"主过_零也过", "恒不过臂", "不确定臂"}
          and 确定不过 == ["不确定臂"]
          and 判["主过_零不过"]["全过"]
          and 报["集"]["留出集"]["主过_零不过"]["全过"])
    return dict(通过=bool(好), 判据集不过=判据集不过, 确定性不过=确定不过,
                标准形态过=bool(判["主过_零不过"]["全过"]),
                留出集同判=bool(报["集"]["留出集"]["主过_零不过"]["全过"]),
                总过=报["总过"], 判据数=报["判据数"])


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 本台自证()
    print(json.dumps(果, ensure_ascii=False, indent=1, default=str))
    print("[测量台] 自证：%s" % ("PASS" if 果["通过"] else "FAIL"))
    raise SystemExit(0 if 果["通过"] else 1)
