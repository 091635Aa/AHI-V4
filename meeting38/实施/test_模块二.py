# -*- coding: utf-8 -*-
"""test_模块二.py —— V3 模块二判据：操作干净化（三层过滤 + 节奏感）。

判据：
  M2-1 界面类型识别：含「WPS」的标题 ⇒ 类型=文档编辑器，且带操作经验。
  M2-2 功能标签（第二层）：点击「新建」致原处变化 ⇒ 可用；点击「AI模板」冒出一堆新东西 ⇒ 干扰项。
  M2-3 干扰项回避（第三层）：候选全为干扰项 ⇒ 不出招。
  M2-4 可用优先：可用与干扰项同场 ⇒ 点「可用」的那件。
  M2-5 未知先看：未知元素前两次只「看」不「点」，看够后允许试探一次。
  M2-6 节奏感：远全速 / 近减速 / 到达前停 0.2s。
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

from 功能档案 import 功能档案, 元素键                          # noqa: E402


def _e(句柄, 框, 心, 字="", 主色=(120, 120, 120)):
    return dict(句柄=句柄, 框=框, 心=心, 字=字, 主色=主色, 形状="按钮")


def main():
    档 = 功能档案(种子=3800)

    # 第一层
    界 = 档.识界面("WPS Office - 文档1")
    M2_1 = (界["类型"] == "文档编辑器" and bool(界["经验"]))

    # 构造元素
    A = _e("新建", (10, 10, 80, 30), (50, 25), 字="新建")
    B = _e("AI模板", (110, 10, 120, 30), (170, 25), 字="AI模板")
    C = _e("未知件", (10, 200, 80, 30), (50, 215), 字="??")

    # 第二层：点击「新建」→ 原处字变化（可用）
    rA = 档.记点击(元素键(A), [A], [dict(A, 字="新建▸")])
    # 点击「AI模板」→ 冒出一堆新东西（干扰项）
    后B = [B] + [_e("弹%d" % i, (300 + i * 10, 100, 60, 20), (330 + i * 10, 110)) for i in range(4)]
    rB = 档.记点击(元素键(B), [B], 后B)
    M2_2 = (rA["标签"] == "可用" and rB["标签"] == "干扰项")

    # 第三层
    v全干扰 = 档.挑(B, [B])
    v可用优先 = 档.挑(None, [B, A])
    M2_3 = (v全干扰["动作"] is None)
    M2_4 = (v可用优先["动作"] == "点" and 元素键(v可用优先["目标"]) == 元素键(A))

    # 未知先看
    序 = [档.挑(C, [C])["动作"] for _ in range(3)]
    M2_5 = (序[0] == "看" and 序[1] == "看" and 序[2] == "点")

    # 节奏
    r800, r200, r10 = 档.节奏(800), 档.节奏(200), 档.节奏(10)
    M2_6 = (r800["速度系数"] == 1.0 and r800["到达停顿s"] == 0.0
            and 0.0 < r200["速度系数"] < 1.0
            and r10["到达停顿s"] == 0.2)

    报 = dict(
        M2_1_界面识别=dict(过=bool(M2_1), 类型=界["类型"], 经验=界["经验"]),
        M2_2_功能标签=dict(过=bool(M2_2), 新建=rA, AI模板=rB),
        M2_3_干扰项回避=dict(过=bool(M2_3), 裁决=v全干扰),
        M2_4_可用优先=dict(过=bool(M2_4), 裁决=dict(v可用优先, 目标=元素键(v可用优先.get("目标")))),
        M2_5_未知先看=dict(过=bool(M2_5), 动作序=序),
        M2_6_节奏感=dict(过=bool(M2_6), 远=r800, 近=r200, 到=r10),
        快照=档.快照(),
    )
    全过 = all(报[k]["过"] for k in 报 if k.startswith("M2_"))
    报["全过"] = bool(全过)
    with open(os.path.join(HERE, "test_模块二_结果.json"), "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)
    print(json.dumps(报, ensure_ascii=False, indent=1))
    return 全过


if __name__ == "__main__":
    ok = main()
    print("\n[模块二] %s" % ("全过 ✅" if ok else "未全过 ❌"))