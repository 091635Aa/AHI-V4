# -*- coding: utf-8 -*-
"""meeting66 / 评测_时长.py —— 学习型时长 + 自条件迭代 的双集 CER 评测。

用法：python 评测_时长.py --ckpt <声学 ckpt> --时长 <时长 ckpt> [--迭代 4] [--阻尼 0.5]
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
import 时长模型 as H  # noqa: E402
import 声码器与评测 as V  # noqa: E402


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--时长", required=True)
    ap.add_argument("--迭代", type=int, default=4)
    ap.add_argument("--阻尼", type=float, default=0.5)
    ap.add_argument("--出", default=str(M66 / "产物" / "音频_时长"))
    a = ap.parse_args()
    模型 = M.加载(a.ckpt)
    时长 = H.加载时长(a.时长)
    出目录 = Path(a.出)
    出目录.mkdir(parents=True, exist_ok=True)
    run = "时长_%s" % Path(a.ckpt).stem
    记录 = []
    print("[合成] 学习型时长 + 迭代=%d 阻尼=%.2f" % (a.迭代, a.阻尼), flush=True)
    for 集合名, 句s in (("历史", V.句集历史), ("留出", V.句集留出)):
        for 句号, 句 in 句s:
            mel, 帧数 = H.合成mel_时长(模型, 时长, 句, 迭代=a.迭代, 阻尼=a.阻尼)
            wav = V.mel转波(mel)
            p = 出目录 / ("%s_%s_%s.wav" % (run, 集合名, 句号))
            V.写wav(p, wav)
            记录.append(dict(集合=集合名, 句号=句号, 文=句, 路径=str(p),
                            帧数=[int(x) for x in 帧数], mel形状=list(mel.shape)))
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
    out = M66 / "产物" / "判据" / "判据_时长.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(ckpt=a.ckpt, 时长=a.时长, 迭代=a.迭代, 阻尼=a.阻尼,
                                   汇总=汇总, 逐句=记录), ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("→", out)


if __name__ == "__main__":
    raise SystemExit(主())
