# -*- coding: utf-8 -*-
"""meeting66 / 选模型.py —— 用「推理 CER」而不是 teacher-forced val 选模型。

背景（06_实验_v5v6 §7）：v9 把 val 从 0.0407 降到 0.0216，推理反而变差 ⇒ val 不是有效选择指标。
本件：对训练过程中按周期保存的 ckpt_e*.pt（+ ckpt.pt / ckpt_末.pt）逐个合成**探针句**（与训练/评测集都不重合），
过 ASR 算 CER，按 CER 排序选 best。

探针句（预注册，不参与 S1–S12 评测，也不在训练语料里）：
  P1 明天下午我们去公园散步吧。
  P2 他把窗户打开，让新鲜空气进来。
  P3 这家店的面条味道很不错。

用法：python 选模型.py --目录 <模型目录> [--迭代 3] [--阻尼 1.0]
只写 <模型目录>/选模型.json；ASR 单进程；确定性（同 ckpt 两趟转写一致）。
"""
from __future__ import annotations

import argparse
import json
import os
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
import 声码器与评测 as V  # noqa: E402
from 量具_v9 import CER, 载引擎  # noqa: E402

探针P = (
    ("P1", "明天下午我们去公园散步吧。"),
    ("P2", "他把窗户打开，让新鲜空气进来。"),
    ("P3", "这家店的面条味道很不错。"),
)
验证集S13 = (
    ("S13", "今天天气真不错，我们去外面走走吧。"),
    ("S14", "他把钥匙放在门口的桌子上。"),
    ("S15", "这家饭馆的红烧肉做得很好吃。"),
    ("S16", "明天上午我要去银行办点事。"),
    ("S17", "孩子们在操场上跑来跑去。"),
    ("S18", "雨停了，天上的云慢慢散开。"),
)


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--目录", required=True)
    ap.add_argument("--迭代", type=int, default=3)
    ap.add_argument("--阻尼", type=float, default=1.0)
    ap.add_argument("--每音节帧数", type=int, default=24)
    ap.add_argument("--集", choices=("P", "S13"), default="P", help="P=3 句探针（快筛）；S13=6 句独立验证集")
    args = ap.parse_args()
    os.environ["M66_迭代推理"] = str(args.迭代)
    os.environ["M66_阻尼"] = str(args.阻尼)
    目录 = Path(args.目录)
    探针 = 验证集S13 if args.集 == "S13" else 探针P
    cks = sorted([p for p in 目录.glob("ckpt_e*.pt")] +
                 [p for p in (目录 / "ckpt.pt", 目录 / "ckpt_末.pt") if p.exists()])
    print("候选 ckpt: %d 个" % len(cks))
    if not cks:
        print("没有候选 ckpt")
        return 1
    引 = 载引擎()
    出目录 = 目录 / "_选模型音频"
    出目录.mkdir(exist_ok=True)
    结果 = []
    for i, ck in enumerate(cks):
        模型 = M.加载(str(ck))
        逐句, 转写s = [], []
        for 号, 文 in 探针:
            mel = M.合成mel(模型, 文, 每音节帧数=args.每音节帧数)
            wav = V.mel转波(mel)
            p = 出目录 / ("%s_%s.wav" % (ck.stem, 号))
            V.写wav(p, wav)
            t1 = 引["转写"](str(p))
            t2 = 引["转写"](str(p))
            cer = CER(文, t1)
            逐句.append(dict(号=号, 文=文, 转写=t1, CER=cer, 两趟一致=(t1 == t2)))
            转写s.append(t1)
        cer均值 = float(np.mean([x["CER"] for x in 逐句]))
        结果.append(dict(ckpt=ck.name, CER均值=round(cer均值, 4), 逐句=逐句,
                       两趟一致=all(x["两趟一致"] for x in 逐句)))
        del 模型
        print("  [%d/%d] %-18s CER=%.4f  转写=%s" % (i + 1, len(cks), ck.name, cer均值,
                                                " | ".join(转写s)))
    结果.sort(key=lambda x: x["CER均值"])
    best = 结果[0]
    out = dict(目录=str(目录), 迭代=args.迭代, 阻尼=args.阻尼,
               每音节帧数=args.每音节帧数, 探针=[dict(号=a, 文=b) for a, b in 探针],
               候选=结果, 最佳=best["ckpt"], 最佳CER=best["CER均值"])
    (目录 / "选模型.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("最佳: %s  CER=%.4f  → %s" % (best["ckpt"], best["CER均值"], 目录 / "选模型.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(主())
