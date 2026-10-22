# -*- coding: utf-8 -*-
"""meeting31/inner_speech.py —— #243 v10 自言空间：持久可审计的内部言语流。

任务书 #242 核心组件。规格：
- 六类条目：疑问（内生问题）/假设（我认为可能是…）/预测（我预期如果…）/
  复盘（结果与预测差在哪）/计划（接下来做什么、为什么）/自评（状态/行为分布/进度）；
- 每条强制携带：拍数时间戳、触发源（内生描述 / 外部账本号）、≥1 个账本引用号
  （号, 摘录）——引用不存在的记录=虚假自言=FAIL 项；
- 生成机制：由引擎 agency + 账本查证驱动（禁关键词路由，#229 纪律）——
  本模块只负责流的存储/审计/查询，条目生成在 synthesis.py（状态量→候选类型→
  引擎 a星序号裁决），不存在任何词表；
- 审计：独立核对每个引用号存在且摘录 ⊆ 被引账本条目文本（账本只追加不改动，
  摘录前缀快照恒定成立）；
- 查询接口：自言流(n)——考官可随时检查。
"""
from __future__ import annotations

自言类型 = ("疑问", "假设", "预测", "复盘", "计划", "自评")


class 自言空间:
    """内部言语流（#243）：存储 + 独立审计 + 查询。"""

    def __init__(self, seed: int = 310):
        self.seed = int(seed)
        self.流: list = []          # dict(号, 拍, 类型, 文本, 触发源, 引用)
        self._号 = 0

    # ---------- 追加（生成方在 synthesis.py；此处强制规格） ----------
    def 追加(self, 拍: int, 类型: str, 文本: str, 触发源, 引用: list) -> int:
        if 类型 not in 自言类型:
            raise ValueError("未知自言类型：%r（六类=%s）" % (类型, 自言类型))
        if not 引用:
            raise ValueError("自言条目必须携带≥1 个账本引用（#242 规格）")
        self._号 += 1
        self.流.append(dict(号=self._号, 拍=int(拍), 类型=类型, 文本=str(文本),
                            触发源=触发源, 引用=[(int(a), str(b)) for a, b in 引用]))
        return self._号

    # ---------- 查询接口（考官入口） ----------
    def 自言流(self, n: int = 20, 类型: str = None) -> list:
        es = [e for e in self.流 if 类型 is None or e["类型"] == 类型]
        return es[-max(int(n), 0):]

    def 类型覆盖(self) -> dict:
        计: dict = {}
        for e in self.流:
            计[e["类型"]] = 计.get(e["类型"], 0) + 1
        return 计

    # ---------- 独立审计（G1）：引用号存在 ∧ 摘录 ⊆ 被引条目文本 ----------
    def 审计(self, 账本: list, 抽检: int = None) -> dict:
        """账本=[(类型,文本)] 只追加序列。抽检=None 全量；抽检=k 均匀采样 k 条。"""
        es = self.流
        if 抽检 is not None and 0 < 抽检 < len(es):
            步 = len(es) / 抽检
            es = [es[int(i * 步)] for i in range(抽检)]
        违规 = []
        for e in es:
            if not e["引用"]:
                违规.append((e["号"], -1, "无引用"))
                continue
            for (号, 摘录) in e["引用"]:
                if not (0 <= 号 < len(账本)):
                    违规.append((e["号"], 号, "引用不存在（账本仅 %d 条）" % len(账本)))
                    continue
                if 摘录 and 摘录 not in 账本[号][1]:
                    违规.append((e["号"], 号, "摘录不符：%r" % 摘录[:12]))
        n检 = sum(len(e["引用"]) for e in es)
        率 = (1.0 if n检 == 0 else 1.0 - len(违规) / n检)
        return dict(条目=len(es), 引用数=n检, 违规数=len(违规), 违规=违规[:20],
                    真实率=round(率, 4), 通过=(len(违规) == 0))
