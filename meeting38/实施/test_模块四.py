# -*- coding: utf-8 -*-
"""test_模块四.py —— V3 模块四判据：三阶段消化（摄入 / 关联 / 复盘）。

判据：
  M4-1 摄入：文本 → 向量 → 缓冲舱（缓冲数增长）。
  M4-2 关联：新知识与记忆底图 M 匹配 → 建立「新↔旧」绑定（绑定非空，且相似度**高于下限**；
            无关文本不产生绑定 —— 审计 P1-3）。
  M4-3 复盘：提取核心概念 → 压缩快照（有束）→ 写入**独立学习迹**（学习迹号非空）。
  M4-4 确定性（C13）：同种子同输入两遍 ⇒ 关键词与摘要逐字一致。
  M4-5 学习≠记忆（所有者硬命题 / 裁定 H5）：复盘后 **学习迹增长**、而**记忆底图大小不变**
            （审计 P0-2：原先以「记忆库变大」为通过标准，与硬命题冲突，已改）。
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
M38 = os.path.dirname(HERE)
for p in (HERE, M38):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── 出口编码自愈（C13/CI）：默认 GBK 控制台下 print ✅/❌ 会抛 UnicodeEncodeError 并
#    让进程 exit=1（判据全过却被判失败，见 V3审计报告 P0-3）。此处就地重配 stdout/stderr，
#    不改任何既有输出内容，只保证非 utf-8 控制台也能安全写出。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from 学习_三阶段 import 三阶段消化                              # noqa: E402

新料 = [
    "Python 的字典 dict 是一种键值映射，可以用 json.dumps 序列化。",
    "JSON 解析：json.loads 把字符串反序列化成 Python 字典。",
    "列表 list 是有序可变序列，元组 tuple 不可变。",
    "json 模块提供 dump、dumps、load、loads 四个常用函数。",
]
旧知 = "Python 的 json 模块把字典序列化成 JSON 字符串，json.loads 反过来解析。"


def _建():
    from 共存态 import 共存态
    from 容量桥 import 容量桥
    编码 = 共存态(种子=3800, 用gpu=False).编码
    记忆 = 容量桥(种子=3800, 编码器=编码)          # ★编码器：入库自动补语义向量
    记忆.记经历(拍=0, 类型="旧知", 文本=旧知, 来源="预置")
    return 三阶段消化(编码器=编码, 记忆桥=记忆), 记忆


# ── 语义刻度判据用（对齐上界=人工同义改写）──
同义甲 = "中国外交部回应了相关问题。"
同义乙 = "外交部就相关问题作出回应。"
无关丙 = "今天的晚饭是番茄炒蛋，味道还行。"


def _语义刻度():
    """★实测语义刻度（第二轮）：真站的「学习」能不能分辨同义与无关。

    这是对「不理解语义」这一指控的**直接口径**：
      · 同义改写对（人工构造，作为上界）→ 必须绑上；
      · 无关对 → 必须不绑（可失败判据）。
    """
    from 共存态 import 共存态
    from 容量桥 import 容量桥
    编码 = 共存态(种子=3800, 用gpu=False).编码
    记忆 = 容量桥(种子=3800, 编码器=编码)
    消 = 三阶段消化(编码器=编码, 记忆桥=记忆)
    记忆.记经历(拍=0, 类型="预置", 文本=同义甲, 来源="预置")
    消.摄入(同义乙, 来源="网页")
    绑同义 = 消.关联()
    消.摄入(无关丙, 来源="网页")
    绑无关 = 消.关联()
    return dict(同义绑定数=len(绑同义), 无关绑定数=len(绑无关),
                同义相似=(绑同义[0]["相似"] if 绑同义 else None),
                无关相似=(绑无关[0]["相似"] if 绑无关 else None))


def main():
    消, 记 = _建()
    库0 = 记.库大小()
    迹0 = 消.学迹.条数()
    for t in 新料:
        消.摄入(t, 来源="网页")
    M4_1 = (len(消.缓冲) == len(新料))
    绑 = 消.关联()
    M4_2 = (len(绑) >= 1)
    # 反例（可失败）：与旧知语义无关的文本**不得**产生绑定
    消.摄入("今天的晚饭是番茄炒蛋，味道还行。", 来源="无关")
    绑无关 = 消.关联()
    M4_2b = (len(绑无关) == 0)
    复 = 消.复盘()
    M4_3 = (bool(复.get("关键词")) and bool(复.get("有束")) and 复.get("学习迹") is not None)
    库1 = 记.库大小()
    迹1 = 消.学迹.条数()
    # M4-5：学习迹增长 且 记忆底图**不变**（学习≠记忆，物理分离）
    M4_5 = (迹1 > 迹0) and (库1 == 库0)

    # 确定性：同输入两遍
    消2, _ = _建()
    for t in 新料:
        消2.摄入(t, 来源="网页")
    消2.关联()
    复2 = 消2.复盘()
    M4_4 = (复.get("关键词") == 复2.get("关键词") and 复.get("摘要") == 复2.get("摘要"))

    # ★M4-6 语义刻度：同义改写必绑、无关必不绑（「理解语义」的可失败判据）
    刻 = _语义刻度()
    M4_6 = bool(刻["同义绑定数"] >= 1 and 刻["无关绑定数"] == 0
                and (刻["同义相似"] or 0) >= 0.3)

    报 = dict(
        M4_1_摄入=dict(过=bool(M4_1), 缓冲=len(消.缓冲), 有向量=all(x["向量"] is not None for x in 消.缓冲)),
        M4_2_关联=dict(过=bool(M4_2), 绑定数=len(绑), 相似下限=0.2, 样本=绑[:4]),
        M4_2b_无关不绑=dict(过=bool(M4_2b), 无关绑定数=len(绑无关)),
        M4_3_复盘=dict(过=bool(M4_3), 关键词=复.get("关键词"), 摘要=复.get("摘要"),
                     有束=复.get("有束"), 学习迹=复.get("学习迹"), 簇数=复.get("簇数")),
        M4_4_确定性=dict(过=bool(M4_4), 关键词2=复2.get("关键词")),
        M4_5_学习非记忆=dict(过=bool(M4_5), 学习迹0=迹0, 学习迹1=迹1, 库0=库0, 库1=库1,
                          记忆底图未增=bool(库1 == 库0)),
        M4_6_语义刻度=dict(过=bool(M4_6), **刻),
    )
    全过 = all(报[k]["过"] for k in 报 if k.startswith("M4_"))
    报["全过"] = bool(全过)
    with open(os.path.join(HERE, "test_模块四_结果.json"), "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)
    print(json.dumps(报, ensure_ascii=False, indent=1))
    return 全过


if __name__ == "__main__":
    ok = main()
    print("\n[模块四] %s" % ("全过 ✅" if ok else "未全过 ❌"))