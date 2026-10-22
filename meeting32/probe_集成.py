# -*- coding: utf-8 -*-
"""meeting32/probe_集成.py —— 声带挂载 综合体 集成探针 + meeting31 回归 smoke。

判据（先定）：
  I1 挂载零侵入：挂载后 桥.步 正常返回响应，综合体逻辑不受扰
  I2 自动发声：每次 步() 后 发声台账新增一条且通过自检
  I3 异常隔离：发声层抛错不影响主体（模拟 wav 目录只读等）
  I4 回归：meeting31 smoke.py 全项 PASS（声带未挂载时行为不变）
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "meeting31"))
sys.path.insert(0, os.path.join(HERE, "..", "meeting28"))

结果 = []


def 判(名, 条件, 明细=""):
    结果.append((名, bool(条件), 明细))
    print("[%s] %s %s" % ("PASS" if 条件 else "FAIL", 名, 明细), flush=True)


def main():
    from synthesis import 综合体
    from 声带 import 声带器官

    台账 = os.path.join(HERE, "集成_发声台账.jsonl")
    if os.path.exists(台账):
        os.remove(台账)
    带 = 声带器官(seed=300, D=1024, 目录=os.path.join(HERE, "声音"),
                 台账路径=台账)

    桥 = 综合体(seed=777)
    带.接综合体(桥)
    判("I1 挂载零侵入", hasattr(桥, "声带") and callable(桥.步),
       "桥.声带 已挂载，步 可调用")

    前 = len(带.台账流(100000)) if os.path.exists(台账) else 0
    r1 = 桥.步("你好")
    条 = 带.台账流(100000)
    新 = [t for t in 条 if t.get("通过")]
    判("I2 自动发声", len(条) > 前 and len(新) > 0 and bool(str(r1.get("响应", "")).strip()),
       "步1次→台账%d条(通过%d) 响应=%s" % (len(条), len(新), str(r1.get("响应", ""))[:30]))
    r2 = 桥.步("你叫什么名字？")
    条2 = 带.台账流(100000)
    判("I2b 连续步进", len(条2) >= len(条) + 1,
       "步2次→台账%d条" % len(条2))

    # I3 异常隔离：把目录换成文件（写 wav 必炸），发声层应吞掉异常
    带.目录 = os.path.join(HERE, "集成_不可写.wav")
    with open(带.目录, "w") as f:
        f.write("x")
    try:
        r3 = 桥.步("今天天气怎么样？")
        ok3 = bool(str(r3.get("响应", "")).strip())
    except Exception as e:
        ok3 = False
        print("  异常泄漏:", e)
    判("I3 异常隔离", ok3, "wav 落盘失败被吞，主体返回正常")
    os.remove(带.目录)

    # I4 回归：meeting31 smoke.py（用实质标记判读，不依赖 "PASS" 字样）
    r = subprocess.run([sys.executable, os.path.join(HERE, "..", "meeting31",
                                                     "smoke.py")],
                       capture_output=True, text=True, cwd=os.path.join(HERE, "..",
                                                                        "meeting31"),
                       timeout=600)
    out = (r.stdout or "") + (r.stderr or "")
    审计过 = "通过=True" in out and "违规 0" in out
    考试 = [l for l in out.splitlines() if "top3符号正确率" in l]
    考试准 = 0.0
    if 考试:
        try:
            考试准 = float(考试[-1].split("'top3符号正确率': ")[1].split(",")[0]
                           .strip("'} "))
        except Exception:
            考试准 = 0.0
    判("I4 meeting31 回归", 审计过 and 考试准 >= 0.7,
       "自言审计通过=%s 知识考试top3=%.3f（判据≥0.7）" % (审计过, 考试准))

    fail = [n for n, ok, _ in 结果 if not ok]
    print("\n===== 集成探针：%d/%d PASS =====" % (len(结果) - len(fail), len(结果)))
    if fail:
        print("FAIL:", fail)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
