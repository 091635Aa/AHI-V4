# -*- coding: utf-8 -*-
"""绿队 R3 补臂 · F5-M6 800 拍流式（主控 2026-10-17 要求；不跑大实验）

只读导入既有件；只写本目录；真跑；两趟复核；并发期墙钟仅记录。
设计：K=8 键、D=256、800 事件（50% 自源 / 50% 外源）；E 以 η=1/16 衰减叠加
（有效窗口≈16 ＝ R3 实测容量阈值 K*=16@D=256）；预测 ê=解绑(键,E) 后与观测 o 求 cos。
三臂：有状态 / 可重算（ê=键，仅当前输入）/ 历史置换（把查询键置换到别键的历史）。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
绿队 = os.path.dirname(HERE)
meeting52 = os.path.dirname(绿队)
AHI4 = os.path.dirname(meeting52)
_核py = os.path.join(AHI4, "meeting41", "实施", "算子_能力核.py")


def 载能力核():
    spec = importlib.util.spec_from_file_location("能力核_只读_r3b", _核py)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_m = 载能力核()
能力核 = _m.能力核
_归 = _m._归


def 双极(rng, n):
    return np.where(rng.integers(0, 2, size=int(n)) == 1, 1.0, -1.0)


def AUC(正, 负):
    正 = np.asarray(正, dtype=np.float64); 负 = np.asarray(负, dtype=np.float64)
    if 正.size == 0 or 负.size == 0:
        return float("nan")
    return float(np.mean(正[:, None] > 负[None, :]) + 0.5 * np.mean(正[:, None] == 负[None, :]))


def 跑流(种子=53001, D=256, K键=8, 事件数=800, 自源比=0.5, η=1.0 / 16.0):
    核 = 能力核(D=D, 种子=int(种子))
    rng = np.random.default_rng(int(种子))
    值 = np.array([双极(rng, D) for _ in range(K键)])
    键码 = np.array([双极(rng, D) for _ in range(K键)])
    # 事件序列（固定，三臂共用）
    事件 = []
    for t in range(int(事件数)):
        k = int(rng.integers(0, K键))
        自 = bool(rng.random() < 自源比)
        o = 值[k].copy() if 自 else 双极(rng, D)     # 外源：与键值无关的随机观测
        事件.append((k, 自, o))
    置换 = np.random.default_rng(int(种子) + 1).permutation(K键)
    果 = dict(臂={})
    for 臂 in ("有状态", "可重算", "历史置换"):
        E = np.zeros(D, dtype=np.float64)
        正晚, 负晚, 正全, 负全 = [], [], [], []
        for t, (k, 自, o) in enumerate(事件):
            if 臂 == "有状态":
                ê = 核.解绑(键码[k], E)
            elif 臂 == "可重算":
                ê = 键码[k].copy()
            else:
                ê = 核.解绑(键码[int(置换[k])], E)
            a = float(np.dot(ê, o)) / D
            if t >= 200:
                (正全 if 自 else 负全).append(a)
                if t >= 600:
                    (正晚 if 自 else 负晚).append(a)
            E = _归((1.0 - float(η)) * E + float(η) * 核.绑(键码[k], o))
        果["臂"][臂] = dict(AUC_全=float(AUC(正全, 负全)), AUC_晚=float(AUC(正晚, 负晚)),
                           正均值晚=float(np.mean(正晚)) if 正晚 else float("nan"),
                           负均值晚=float(np.mean(负晚)) if 负晚 else float("nan"),
                           n正晚=len(正晚), n负晚=len(负晚))
    果.update(种子=int(种子), D=D, K键=K键, 事件数=int(事件数), 自源比=float(自源比), η=float(η),
              有效窗口=int(round(1.0 / float(η))))
    return 果


def main():
    t0 = time.time()
    果 = 跑流()
    果["两趟"] = 跑流()
    同 = all(果["臂"][k] == 果["两趟"]["臂"][k] for k in 果["臂"])
    果["两趟逐位同"] = bool(同)
    果["meta"] = dict(件="R3 补臂 · F5-M6 800 拍流式", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
                      并发态="会议并发期；墙钟仅记录、不作效率宣称",
                      墙钟_s=round(time.time() - t0, 3), 只读来源=_核py)
    果["四态判定"] = ("PASS" if (果["臂"]["有状态"]["AUC_晚"] >= 0.75 and 果["臂"]["可重算"]["AUC_晚"] <= 0.60)
                   else "FAIL")
    with open(os.path.join(HERE, "汇总_F5M6流式.json"), "w", encoding="utf-8") as f:
        json.dump(果, f, ensure_ascii=False, indent=1)
    print(json.dumps(果, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        np.seterr(all="ignore")
    except Exception:
        pass
    main()
