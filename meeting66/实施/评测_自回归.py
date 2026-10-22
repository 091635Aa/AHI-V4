# -*- coding: utf-8 -*-
"""meeting66 / 评测_自回归.py —— 自回归声学模型的双集 CER 评测。

用法：python 评测_自回归.py --ckpt <ckpt> [--每音节帧数 24]
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

import 模型_自回归 as A  # noqa: E402
import 声码器与评测 as V  # noqa: E402


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--每音节帧数", type=int, default=24)
    ap.add_argument("--出", default=str(M66 / "产物" / "音频_自回归"))
    a = ap.parse_args()
    模型 = A.加载自回归(a.ckpt)
    出目录 = Path(a.出)
    出目录.mkdir(parents=True, exist_ok=True)
    记录 = []
    print("[合成] 自回归逐帧生成 每音节=%d 帧" % a.每音节帧数, flush=True)
    for 集合名, 句s in (("历史", V.句集历史), ("留出", V.句集留出)):
        for 句号, 句 in 句s:
            mel = A.合成mel_自回归(模型, 句, 每音节帧数=a.每音节帧数)
            wav = V.mel转波(mel)
            p = 出目录 / ("自回归_%s_%s.wav" % (集合名, 句号))
            V.写wav(p, wav)
            记录.append(dict(集合=集合名, 句号=句号, 文=句, 路径=str(p),
                            mel形状=list(mel.shape)))
    print("[ASR] 载引擎…", flush=True)
    量 = V.量具模块()
    引 = 量.载引擎()
    for 条 in 记录:
        t1 = 引["转写"](条["路径"])
        t2 = 引["转写"](条["路径"])
        cer = 量.CER(条["文"], t1)
        条.update(转写=t1, CER=cer, 两趟一致=(t1 == t2))
        print("  %-2s %-4s CER=%.4f 两趟一致=%s 转写=%s"
              % (条["集合"], 条["句号"], cer, t1 == t2, t1[:40]), flush=True)
    汇总 = {名: float(np.mean([条["CER"] for 条 in 记录 if 条["集合"] == 名]))
            for 名 in ("历史", "留出")}
    print("=" * 70)
    print("历史 CER=%.4f（v8.1 SpeakerA 0.5853）｜ 留出 CER=%.4f（v8.1 SpeakerA 0.6893）"
          % (汇总["历史"], 汇总["留出"]))
    out = M66 / "产物" / "判据" / "判据_自回归.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(ckpt=a.ckpt, 每音节帧数=a.每音节帧数,
                                   汇总=汇总, 逐句=记录), ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("→", out)


if __name__ == "__main__":
    raise SystemExit(主())
