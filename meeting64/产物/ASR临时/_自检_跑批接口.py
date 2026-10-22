# -*- coding: utf-8 -*-
"""meeting64/产物/ASR临时/_自检_跑批接口.py —— **不载 ASR** 的接口/plumbing 自检：

1. `跑批` 一次载「假引擎」（转写恒返 ""，不载 Paraformer）串行跑 2 个条件：
   c1 = v8.1 基线（角色改=None）；c2 = 同基线 + `角色改`（基频 ×1.03、共振峰缩放 F2 ×0.98）
   ⇒ 验证：跑批 JSON 落盘、两条件波形 sha256 不同（`角色改` 真的生效）、
     临时文件名含 标签+条件号+run_id 且**全局唯一**。
2. 合成对拍：v8.1 波形与 meeting63 官方基线 18 个 wav **逐位同**（文件名变了，音频不变）。

注意：CER 读数在假引擎下无条件为 1.0，**不是**测量值，只验管道。
跑：
  & "<workspace>/\03_AHI\\.venv\\Scripts\\python.exe" "<workspace>/\03_AHI\\AHI4\\meeting64\\产物\\ASR临时\\_自检_跑批接口.py"
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
M64 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(M64, "实施"))

import numpy as np  # noqa: E402

import 量具_v9 as 量  # noqa: E402
import 韵律_v8 as H8  # noqa: E402

假引擎 = dict(名="假引擎(不载ASR)", 转写=lambda p: "")


def 改(名, 音):
    d = dict(音)
    d["基频"] = float(d.get("基频", 200.0)) * 1.03
    f = list(d.get("共振峰缩放", (1.0, 1.0, 1.0)))
    f[1] = float(f[1]) * 0.98
    d["共振峰缩放"] = tuple(f)
    return d


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    np.seterr(all="ignore")
    输出 = os.path.join(HERE, "_自检_跑批.json")
    条件 = [
        dict(名="c1_基线", 韵律工厂=H8.人类韵律_v8, 角色s=量.多句角色, 角色改=None, 做sham=False),
        dict(名="c2_角色改", 韵律工厂=H8.人类韵律_v8, 角色s=量.多句角色, 角色改=改, 做sham=False),
    ]
    果 = 量.跑批(条件, 引=假引擎, 输出=输出)
    a = 果["逐条件"]["c1_基线"]["逐样本"]
    b = 果["逐条件"]["c2_角色改"]["逐样本"]
    全变 = all(a[k]["sha256"] != b[k]["sha256"] for k in a)
    音色变 = all(a[k]["音色指纹"] != b[k]["音色指纹"] for k in a)
    文件名 = [a[k]["wav"] for k in a] + [b[k]["wav"] for k in a]
    唯一 = len(set(文件名)) == len(文件名)
    含标签 = all(("c1_基线" in w) or ("c2_角色改" in w) for w in 文件名)
    含条件号 = all(("_j01_" in w) or ("_j02_" in w) for w in 文件名)
    参考 = os.path.join(量.M63, "产物", "ASR临时", "多句")
    参考表 = sorted(f for f in os.listdir(参考) if f.lower().endswith(".wav"))
    序表 = [(s, n) for s, _ in 量.句集 for n in 量.多句角色]
    对拍全同 = all(
        量._sha256(os.path.join(参考, 参考表[i])) == a["%s·%s" % (句号, 名)]["sha256"]
        for i, (句号, 名) in enumerate(序表))
    print("=" * 88)
    print("跑批 JSON 落盘 =", os.path.exists(输出), "｜ 条件数 =", len(果["逐条件"]))
    print("c1↔c2 波形 sha256 全变 =", 全变, "｜ 音色指纹全变 =", 音色变)
    print("文件名唯一 =", 唯一, "｜ 含标签 =", 含标签, "｜ 含条件号 =", 含条件号)
    print("文件名样例 =", 文件名[0], "｜", 文件名[-1])
    print("c1 合成对拍 meeting63 基线 18/18 逐位同 =", 对拍全同)
    print("（CER 值在假引擎下恒为 1.0，非测量值）多句CER均值 =",
          {k: v["多句CER均值"] for k, v in 果["逐条件"].items()})
    通过 = bool(全变 and 音色变 and 唯一 and 含标签 and 含条件号 and 对拍全同 and os.path.exists(输出))
    print("接口自检 =", 通过)
    raise SystemExit(0 if 通过 else 1)
