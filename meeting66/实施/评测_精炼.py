# -*- coding: utf-8 -*-
"""meeting66 / 评测_精炼.py —— 精炼器（两阶段）双集 CER 评测。

流程：base → 粗 mel；精炼器迭代 N 轮 → mel → Griffin-Lim → wav → ASR（两趟）→ CER。
与 `声码器与评测.py` 同句子集（S1–S12）、同 ASR 仪器、同 GL 参数。
用法：python 评测_精炼.py --精炼 <ckpt> [--base <ckpt>] [--迭代 N] [--阻尼 a] [--每音节帧数 24]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
M66 = HERE.parent
AHI4 = M66.parent
for _p in (str(AHI4 / "meeting64" / "实施"), str(AHI4 / "meeting61" / "实施"),
           str(AHI4 / "meeting58" / "实施")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import 模型_学说话 as M  # noqa: E402
import 模型_精炼 as R  # noqa: E402
import 声码器与评测 as V  # noqa: E402


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--精炼", required=True)
    ap.add_argument("--base", default=str(M66 / "产物" / "模型_v13_K4_320" / "ckpt_e0150.pt"))
    ap.add_argument("--迭代", type=int, default=3)
    ap.add_argument("--阻尼", type=float, default=0.7)
    ap.add_argument("--每音节帧数", type=int, default=24)
    ap.add_argument("--出", default=str(M66 / "产物" / "音频_精炼"))
    a = ap.parse_args()
    base = M.加载(a.base)
    精 = R.加载精炼(a.精炼)
    出目录 = Path(a.出)
    出目录.mkdir(parents=True, exist_ok=True)
    run = "精炼_%s" % Path(a.精炼).parent.name
    逐句 = {"历史": [], "留出": []}
    记录 = []
    print("[合成] base+精炼 迭代=%d 阻尼=%.2f" % (a.迭代, a.阻尼), flush=True)
    for 集合名, 句s in (("历史", V.句集历史), ("留出", V.句集留出)):
        for 句号, 句 in 句s:
            mel = R.合成精炼(精, base, 句, 每音节帧数=a.每音节帧数,
                             迭代=a.迭代, 阻尼=a.阻尼)
            wav = V.mel转波(mel)
            p = 出目录 / ("%s_%s_%s.wav" % (run, 集合名, 句号))
            V.写wav(p, wav)
            记录.append(dict(集合=集合名, 句号=句号, 文=句, 路径=str(p), mel形状=list(mel.shape)))
    print("[ASR] 载引擎…", flush=True)
    量 = V.量具模块()
    引 = 量.载引擎()
    for 条 in 记录:
        t1 = 引["转写"](条["路径"])
        t2 = 引["转写"](条["路径"])
        cer = 量.CER(条["文"], t1)
        条.update(转写=t1, CER=cer, 两趟一致=(t1 == t2))
        逐句[条["集合"]].append(条["CER"])
        print("  %-2s %-4s CER=%.4f 两趟一致=%s 转写=%s"
              % (条["集合"], 条["句号"], cer, t1 == t2, t1[:40]), flush=True)
    汇总 = {名: float(np.mean(v)) if v else None for 名, v in 逐句.items()}
    print("=" * 70)
    print("历史 CER=%.4f（v8.1 SpeakerA 0.5853）｜ 留出 CER=%.4f（v8.1 SpeakerA 0.6893）"
          % (汇总["历史"], 汇总["留出"]))
    out = M66 / "产物" / "判据" / "判据_精炼.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(精炼=a.精炼, base=a.base, 迭代=a.迭代, 阻尼=a.阻尼,
                                   每音节帧数=a.每音节帧数, 汇总=汇总, 逐句=记录),
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("→", out)


if __name__ == "__main__":
    raise SystemExit(主())
