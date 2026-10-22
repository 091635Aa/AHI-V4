# -*- coding: utf-8 -*-
"""meeting31/旁路.py —— #272 亚思考可视化旁路：把向量遥测翻译成自然语言。

两条诚实旁路：
①符号轨迹：推理环/自言流本就是自然语言（符号中介架构的天然优势）；
②向量→词：最近邻解码——把引擎候选自由能/能量/新异/消化债/饱足 逐拍念出，
  并把当前感知向量对账本做最近邻（『此刻脑中最近联想』）。
不编造含义：只报告数值与其对应的动作/事实，不做过度诠释。
"""
from __future__ import annotations

import json
import os

from dialogue_bridge import 动作名


class 旁路可视化器:
    def __init__(self, 桥, 路径: str, 轮转上限: int = 64 * 1024 * 1024):
        self.桥 = 桥
        self.路径 = 路径
        self.轮转上限 = int(轮转上限)          # 字节；超过则轮转（防 732MB 无界增长）
        self.轮转号 = 0
        self.句数 = 0

    def _轮转(self) -> None:
        """旁路.jsonl → 旁路.001.jsonl（保留历史，另起新文件）。"""
        try:
            根, 扩 = os.path.splitext(self.路径)
            while True:
                self.轮转号 += 1
                新 = "%s.%03d%s" % (根, self.轮转号, 扩)
                if not os.path.exists(新):
                    break
            os.replace(self.路径, 新)
        except OSError:
            pass                                     # 轮转失败不拖垮主环

    def 逐拍(self, rec: dict, 动作: str, 感知文本: str) -> str:
        桥 = self.桥
        ê = 桥.最近候选ê if hasattr(桥, "最近候选ê") else None
        if ê is not None:
            候选 = "、".join("%s %.2f" % (动作名[i % 4], float(v))
                             for i, v in enumerate(list(ê)[:4]))
        else:
            候选 = "（无）"
        联想 = 桥._检索(感知文本, k=1) if 感知文本 else []
        联想词 = "『%s…』(%.2f)" % (联想[0][0][:10], 联想[0][1]) if 联想 else "（无）"
        句 = ("⟨旁路⟩ 候选自由能 %s → 选『%s』；能量 %.2f、新异 %.2f、"
              "消化债 %.2f、饱足 %.2f；感知『%s…』最近联想 %s。") % (
            候选, 动作, 桥.e if 桥.e is not None else -1.0, 桥.ñ or 0.0,
            桥.消化债, 桥.饱足, 感知文本[:8], 联想词)
        self.句数 += 1
        try:
            if (os.path.exists(self.路径)
                    and os.path.getsize(self.路径) >= self.轮转上限):
                self._轮转()
        except OSError:
            pass
        with open(self.路径, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(拍=桥.拍, 动作=动作, 候选ê=[float(x) for x in ê]
                                    if ê is not None else None,
                                    能量=桥.e, 消化债=桥.消化债, 饱足=桥.饱足,
                                    旁路=句), ensure_ascii=False) + "\n")
        return 句
