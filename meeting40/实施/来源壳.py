# -*- coding: utf-8 -*-
"""meeting40/实施/来源壳.py —— 备B 安全闭环「中枢侧」：L0 感缓 + L1 写入 + 去重下限 + 最小来源双标记壳。

口径（终稿 §5 备B「中枢」条 + §7.5 L0–L3 分层 + §7.6 唯一写闸）：
    「`容量桥.记经历`（语义真通路已修复）+ 语义余弦 + **去重下限（exact + cos≥0.98）**
      + **最小来源双标记壳** + 快照；不做 T 重构、不做 M/IM 双系统；**L0/L1 内存**。」

  · **L0 感缓**（§7.5）：秒级/拍内瞬时缓冲，**不落库**（`感缓` 类；只活一拍）。
  · **L1 快事件**：经本壳写入 `容量桥`（= FSW 长时库记录）。**L2 慢语义不在备B 范围**。
  · **去重下限**：`exact`（同文本指纹）**或** `cos ≥ 0.98`（语义向量余弦）⇒ **不重复写**（返回既有/None）。
  · **最小来源双标记壳**：写 M 必须过 §7.6 的**唯一写闸** `可写记忆(prov, 输出回灌)`——
    `prov_route=实感 ∧ 非「本系统输出回灌」`；否则**拒写**并记账（`自说/回忆/想象` 全拒）。
  · **L3** 是索引层、**不是记忆层**，本壳不碰。

**默认关**（`开关=False`）：`记经历` 原样代理到桥（含 `向量`），行为与今天**逐位一致**（C14）；
`开关=True` 才走去重 + 写闸 + 双标记。随机性：无（只用确定的 sha256 与余弦）。

用法：
    壳 = 来源壳(桥, 编码器=态.编码, 开关=True)
    rid = 壳.记经历(拍=1, 类型="事件", 文本="网页正文…", 向量=v, prov=Prov(实感, 网页))
"""
from __future__ import annotations

import hashlib
from collections import deque

import 事件契约 as 契

去重阈值 = 0.98                 # §5：cos ≥ 0.98 视为同一条
L0上限 = 256                   # 一拍内的感觉缓冲上限（§7.5：拍内瞬时，不落库）


def _指纹(文本) -> str:
    return hashlib.sha256(str(文本 or "").encode("utf-8")).hexdigest()


class 感缓:
    """**L0** 感觉缓冲：拍内瞬时，**不落库**（§7.5）。`放()` 入，`取()` 读，`清拍()` 在拍切换时丢。"""

    def __init__(self, 上限: int = L0上限):
        self._d = deque(maxlen=int(上限))
        self.拍 = None

    def 放(self, 事件: dict, 拍=None) -> dict:
        e = dict(事件 or {})
        if 拍 is not None:
            e.setdefault("t_pai", int(拍))
            self.拍 = int(拍)
        self._d.append(e)
        return e

    def 取(self, 拍=None) -> list:
        if 拍 is None:
            return [dict(x) for x in self._d]
        return [dict(x) for x in self._d if x.get("t_pai") == int(拍)]

    def 清拍(self) -> int:
        n = len(self._d)
        self._d.clear()
        return n

    def __len__(self) -> int:
        return len(self._d)


class 来源壳:
    """L1 写入壳：去重下限 + 唯一写闸 + 来源双标记。**默认关**。"""

    def __init__(self, 桥, 编码器=None, 开关: bool = False, 去重: float = 去重阈值):
        self.桥 = 桥
        self.编码器 = 编码器
        self.开关 = bool(开关)
        self.去重 = float(去重)
        self.感缓 = 感缓()
        self._文本指纹: dict = {}          # 指纹 → rid（exact 去重）
        self._向量: list = []              # [(rid, np.ndarray)]（余弦去重）
        self._标记: dict = {}              # rid → prov 字典（来源双标记）
        self.统计 = dict(写=0, 透传=0, 拒写=0, 去重_exact=0, 去重_cos=0, 拒因={})

    # ── 内部 ──
    def _拒(self, 因: str, rid=None):
        self.统计["拒写"] += 1
        self.统计["拒因"][因] = self.统计["拒因"].get(因, 0) + 1
        return None

    def _余弦最大(self, v) -> tuple:
        try:
            import numpy as np
            q = np.asarray(v, dtype=np.float32).ravel()
            nq = float(np.linalg.norm(q))
            if nq <= 0 or not self._向量:
                return None, 0.0
            最, 最分 = None, -1.0
            for rid, w in self._向量:                      # 顺序即插入序 ⇒ 确定性
                nw = float(np.linalg.norm(w))
                if nw <= 0:
                    continue
                s = float(w @ q) / (nw * nq)
                if s > 最分:
                    最, 最分 = rid, s
            return 最, 最分
        except Exception:
            return None, 0.0

    # ── L1 写入（唯一入口）──
    def 记经历(self, 拍: int, 类型: str, 文本: str, 来源: str = "自我线程", 向量=None,
              prov=None, 输出回灌: bool = False) -> int:
        """写入一条 L1。**壳关 ⇒ 原样透传桥**（`向量` 一起传，行为逐位不变）。"""
        if not self.开关:
            self.统计["透传"] += 1
            return self.桥.记经历(拍=拍, 类型=类型, 文本=文本, 来源=来源, 向量=向量)
        p = prov if isinstance(prov, 契.Provenance) else 契.Provenance.从字典(prov or {})
        # ① 唯一写闸（§7.6）：来路=实感 且 非本系统输出回灌
        if not 契.可写记忆(p, 输出回灌):
            return self._拒("写闸拒：route=%s 回灌=%s" % (p.prov_route, bool(输出回灌)))
        # ② 去重下限：exact 或 cos≥阈值
        fp = _指纹(文本)
        if fp in self._文本指纹:
            self.统计["去重_exact"] += 1
            return self._文本指纹[fp]
        if 向量 is not None:
            rid0, 分 = self._余弦最大(向量)
            if rid0 is not None and 分 >= self.去重:
                self.统计["去重_cos"] += 1
                return rid0
        # ③ 真写入（桥是唯一落库点）
        rid = self.桥.记经历(拍=拍, 类型=类型, 文本=文本, 来源=来源, 向量=向量)
        self._文本指纹[fp] = rid
        self._标记[rid] = p.字典()
        try:                                              # 双标记**同时落到桥的元数据**（可审计）
            if hasattr(self.桥, "_文本") and rid in self.桥._文本:
                self.桥._文本[rid]["prov"] = p.字典()
        except Exception:
            pass
        if 向量 is not None:
            try:
                import numpy as np
                self._向量.append((rid, np.asarray(向量, dtype=np.float32).ravel()))
            except Exception:
                pass
        self.统计["写"] += 1
        return rid

    # ── 读（只读，不改库）──
    def 标记(self, rid) -> dict:
        return dict(self._标记.get(rid, {}))

    def 库大小(self) -> int:
        try:
            return int(self.桥.库大小())
        except Exception:
            return -1

    def 统计表(self) -> dict:
        d = dict(self.统计)
        d["拒因"] = dict(sorted((d.get("拒因") or {}).items()))
        d["库大小"] = self.库大小()
        d["L0条数"] = len(self.感缓)
        d["口径"] = "L0 不落库；L1 去重(exact 或 cos≥%.2f) + 写闸(实感∧非回灌) + 双标记" % self.去重
        return d


# ─────────────────────────── 自测 ───────────────────────────
def 自测() -> dict:
    import numpy as np
    from 容量桥 import 容量桥
    桥 = 容量桥(种子=3800)
    可 = 契.Provenance(prov_acq="网页", prov_org="外源", prov_route="实感")
    自说 = 契.Provenance(prov_acq="自源", prov_org="自源", prov_route="自说")
    回 = 契.Provenance(prov_acq="自源", prov_org="自源", prov_route="实感")
    v1 = np.ones(1024, dtype=np.float32)
    v2 = np.ones(1024, dtype=np.float32)
    v2[0] = 0.0                                          # 与 v1 余弦 ≈ 0.999 ≥ 0.98 ⇒ 去重
    v3 = np.zeros(1024, dtype=np.float32)
    v3[:512] = 1.0                                       # 与 v1 余弦 ≈ 0.707 ⇒ 新写
    壳 = 来源壳(桥, 开关=True)
    案 = {}
    n0 = 壳.库大小()
    r1 = 壳.记经历(1, "事件", "网页正文A", 向量=v1, prov=可)
    案["实感写入"] = (r1 is not None and 壳.库大小() == n0 + 1)
    案["exact去重"] = (壳.记经历(2, "事件", "网页正文A", 向量=v1, prov=可) == r1
                    and 壳.库大小() == n0 + 1)
    r2 = 壳.记经历(3, "事件", "网页正文B", 向量=v2, prov=可)
    案["cos去重"] = (r2 == r1 and 壳.库大小() == n0 + 1)
    r3 = 壳.记经历(4, "事件", "网页正文C", 向量=v3, prov=可)
    案["异向量新写"] = (r3 is not None and r3 != r1 and 壳.库大小() == n0 + 2)
    案["自说拒写"] = (壳.记经历(5, "事件", "我自己说的", 向量=None, prov=自说) is None
                   and 壳.库大小() == n0 + 2)
    案["输出回灌拒写"] = (壳.记经历(6, "事件", "输出回灌", 向量=None, prov=回, 输出回灌=True) is None
                     and 壳.库大小() == n0 + 2)
    案["双标记"] = (壳.标记(r1).get("prov_route") == "实感"
                 and 桥._文本.get(r1, {}).get("prov", {}).get("prov_acq") == "网页")
    # L0：拍内可取、不落库
    壳.感缓.放(dict(模态="视觉"), 拍=1)
    壳.感缓.放(dict(模态="听觉"), 拍=1)
    案["L0不落库"] = (壳.库大小() == n0 + 2 and len(壳.感缓) == 2 and len(壳.感缓.取(1)) == 2)
    壳.感缓.清拍()
    案["L0清拍"] = (len(壳.感缓) == 0 and 壳.库大小() == n0 + 2)
    # 默认关：透传（行为与直接调桥一致）
    桥2 = 容量桥(种子=3800)
    壳关 = 来源壳(桥2, 开关=False)
    关1 = 壳关.记经历(1, "事件", "透传A", 向量=np.ones(1024, dtype=np.float32))
    关2 = 桥2.记经历(2, "事件", "透传B", 向量=np.ones(1024, dtype=np.float32))
    案["默认关透传"] = (关1 == 0 and 关2 == 1 and 壳关.统计["透传"] == 1)
    return dict(通过=bool(all(案.values())), 逐项=案, 统计=壳.统计表(), 关壳统计=壳关.统计)


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "meeting38", "实施"))
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import json
    果 = 自测()
    print(json.dumps(果, ensure_ascii=False, indent=1, default=str))
    print("[来源壳] 自测：%s" % ("PASS" if 果["通过"] else "FAIL"))
    raise SystemExit(0 if 果["通过"] else 1)
