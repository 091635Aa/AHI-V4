# -*- coding: utf-8 -*-
"""test_模块一.py —— V3 模块一判据：思考不公布 + 输出有个性 + 探针不污染输出。

判据：
  M1-1 个性可见（同句不同天）：同意图/同记忆，两个不同 S ⇒ 输出不同。
  M1-2 确定性（C13）：同一输入两次 ⇒ 逐字一致。
  M1-3 记忆调制：有 M 检索命中 ⇒ 输出引用该记忆。
  M1-4 无元话语：输出不含「我想到/我的内部状态是/向量/激活…」等。
  M1-5 探针不污染：探针写入后，对话流/脑内/表达队列**零新增**（输出流原样）。
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

from 输出层 import 输出层, 净化                                   # noqa: E402
from 探针_思考核 import 思考核探针                                 # noqa: E402

元话语 = ("我想到", "我的内部状态", "内部状态", "向量", "激活", "遥测", "特征", "模型")


def main():
    层 = 输出层(种子=310)
    意说 = {"类别": "说"}
    S甲 = {"拍": 10, "点": {"V": 0.6, "A": 0.7}, "ñ": 0.2, "消化债": 0.0,
           "饱足": 1.0, "区": "喜悦"}
    S乙 = {"拍": 99, "点": {"V": -0.5, "A": 0.2}, "ñ": 0.8, "消化债": 0.4,
           "饱足": 0.3, "区": "悲伤"}

    a = 层.生成正文(意说, S甲, None)
    b = 层.生成正文(意说, S乙, None)
    a2 = 层.生成正文(意说, S甲, None)
    c = 层.生成正文({"类别": "问"}, S甲, [("猫是我的宠物", 0.52)])
    d = 层.生成正文({"类别": "操作", "键": "双", "目标点": (640, 360)}, S甲, None)

    M1_1 = (a != b)
    M1_2 = (a == a2)
    M1_3 = ("猫是我的宠物" in c)
    M1_4 = all(not any(k in x for k in 元话语) for x in (a, b, c, d))
    M1_5 = None
    探针行 = 0

    # ── 探针不污染：真起一个 共存态，写思考，跑探针，看输出槽是否原样 ──
    try:
        from 共存态 import 共存态
        态 = 共存态(种子=3800, 用gpu=False)
        态.写思考("想", "内部深度回响一条（不该外显）", 层="内")
        态.写思考("招", "我该点右上角那件", 层="外")
        对话前 = list(态.取对话(会话="主"))
        脑内前 = list(态.取脑内(会话="主"))
        队前 = 态.取下一句()
        探 = 思考核探针(路径=os.path.join(HERE, "_探针_思考核.ndjson"), 维=4, 种子=3800)
        探.扫(态, 上限=20)
        探针行 = 探.计数
        对话后 = list(态.取对话(会话="主"))
        脑内后 = list(态.取脑内(会话="主"))
        队后 = 态.取下一句()
        M1_5 = (对话前 == 对话后) and (脑内前 == 脑内后) and (队前 is None) and (队后 is None)
    except Exception as ex:
        M1_5 = False
        探错 = repr(ex)

    报 = dict(
        M1_1_个性可见=dict(过=bool(M1_1), 甲=a, 乙=b),
        M1_2_确定性=dict(过=bool(M1_2), 样本=a),
        M1_3_记忆调制=dict(过=bool(M1_3), 样本=c),
        M1_4_无元话语=dict(过=bool(M1_4), 样本=[a, b, c, d]),
        M1_5_探针不污染=dict(过=bool(M1_5), 探针行=探针行),
    )
    全过 = all(v["过"] for v in 报.values())
    报["全过"] = bool(全过)
    out = os.path.join(HERE, "test_模块一_结果.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)
    print(json.dumps(报, ensure_ascii=False, indent=1))
    return 全过


if __name__ == "__main__":
    ok = main()
    print("\n[模块一] %s" % ("全过 ✅" if ok else "未全过 ❌"))