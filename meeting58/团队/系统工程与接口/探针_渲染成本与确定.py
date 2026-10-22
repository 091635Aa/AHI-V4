# -*- coding: utf-8 -*-
"""meeting58/E3 探针：声带渲染成本 / 构造成本 / 确定性 / 「换音色」的增量成本。

只读 import 既有件（meeting32/声带.py），不改任何既有文件；产物全部落在本目录。
CPU-only，单次 <10s，不碰声卡、不启动 RVC、不写 meeting55/meeting32 目录。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AHI4 = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # meeting58/团队/<me> → AHI4
sys.path.insert(0, os.path.join(AHI4, "meeting32"))
import 声带 as SD  # noqa: E402

产物 = os.path.join(HERE, "产物")
os.makedirs(产物, exist_ok=True)

D = 1024
向量 = np.sign(np.random.default_rng(55).standard_normal(D)).astype(np.float32)
遥测 = dict(能量=0.5, 新异=0.5, 消化债=0.0, 饱足=0.5)
短句 = "你好。"
长句 = "你好，我是天依。今天想听我唱歌吗？"


def 造(seed=300):
    """构造一个声带器官（临时目录在本目录内）。"""
    return SD.声带器官(seed=seed, D=D,
                        目录=os.path.join(产物, "发声"),
                        台账路径=os.path.join(产物, "发声台账.jsonl"))


def 计时(fn, 次=3):
    列 = []
    for _ in range(次):
        t0 = time.perf_counter()
        r = fn()
        dt = (time.perf_counter() - t0) * 1000.0
        列.append(round(dt, 1))
    return 列, r


def 主():
    out = dict(口径="CPU-only, 本目录产物, 只读既有件")

    # ── 1. 构造成本（音色底/数轴/共振峰表都在 __init__ 里）──
    列 = []
    for _ in range(3):
        t0 = time.perf_counter()
        造()
        列.append(round((time.perf_counter() - t0) * 1000.0, 1))
    out["构造毫秒"] = 列

    # ── 2. 整句渲染成本 ──
    器 = 造()
    列短, r短 = 计时(lambda: 器.发声(向量, 短句, 遥测), 3)
    列长, r长 = 计时(lambda: 器.发声(向量, 长句, 遥测), 3)
    out["渲染"] = dict(
        短句=dict(文本=短句, 毫秒=列短, 时长秒=r短["时长秒"], 实时率=r短["实时率"],
                  拼音数=r短["拼音数"], 通过=r短["通过"]),
        长句=dict(文本=长句, 毫秒=列长, 时长秒=r长["时长秒"], 实时率=r长["实时率"],
                  拼音数=r长["拼音数"], 通过=r长["通过"]))

    # ── 3. 确定性：两个独立实例，同输入 ⇒ 波形逐位同 ──
    a = 造().发声(向量, 长句, 遥测)["波形"]
    b = 造().发声(向量, 长句, 遥测)["波形"]
    # 同一实例连跑两次（合成波形内部 rng 每句重设 seed=自 seed）
    c = 器.发声(向量, 长句, 遥测)["波形"]
    d = 器.发声(向量, 长句, 遥测)["波形"]
    out["确定性"] = dict(
        跨实例逐位同=bool(a.shape == b.shape and np.array_equal(a, b)),
        同实例连跑逐位同=bool(c.shape == d.shape and np.array_equal(c, d)),
        长度=int(len(a)))

    # ── 4. 「换音色」增量成本：不改构造、只换律（子类覆盖 韵律，与 说_音色 同法）──
    class 换律(SD.声带器官):
        def __init__(self):
            super().__init__(seed=300, D=D, 目录=os.path.join(产物, "发声"),
                              台账路径=os.path.join(产物, "发声台账.jsonl"))
            self.当前律 = None

        def 韵律(self, 遥测):
            return dict(self.当前律)

    器2 = 换律()
    律甲 = dict(基频=215.0, 语速=3.6, 幅度=0.55, 气声度=0.15,
                紧喉度=0.28, 粗糙度=0.05, 送气量=0.20, 声区=0.05, _ñ=0.5)
    律乙 = dict(基频=112.0, 语速=2.9, 幅度=0.55, 气声度=0.20,
                紧喉度=0.40, 粗糙度=0.08, 送气量=0.18, 声区=0.02, _ñ=0.5)
    器2.当前律 = dict(律甲)
    甲 = 器2.发声(向量, 长句, 遥测)["波形"]
    器2.当前律 = dict(律乙)                     # ★热切换 = 一次 dict 赋值
    乙 = 器2.发声(向量, 长句, 遥测)["波形"]
    out["换音色"] = dict(切换成本="O(1) dict 赋值，无构造/无重载",
                         波形不同=bool(not np.array_equal(甲, 乙)),
                         F0参数=(律甲["基频"], 律乙["基频"]))

    # ── 5. 隐藏全局态检查：说_音色 的 _器 单例 + 当前律 是可变态 ──
    out["全局态风险"] = dict(
        说_音色_取器="模块级 _器 单例（说_音色.py:93-100），跨调用共享 发声号/当前律",
        当前律="实例可变状态（说_音色.py:106），并发两路渲染会互相覆盖律",
        合成rng="每句 default_rng(self.seed) 重设（声带.py:367）⇒ 波形确定性不依赖调用序")
    return out


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 主()
    p = os.path.join(产物, "渲染成本.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(果, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in 果.items() if k != "全局态风险"},
                     ensure_ascii=False, indent=1))
    print("落盘:", p)
