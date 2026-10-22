# -*- coding: utf-8 -*-
"""meeting66 / 评测_集成.py —— 多模型/多设置 mel 平均集成，双集 CER 评测。

动机：v13 的推理在不同迭代/阻尼下各有强弱（4轮/0.5 历史最好、7轮/0.5 留出最好）；
把多个预测的 log-mel 直接平均（不引入任何新模型、无 attention、C1 允许）可能降低单模型方差。
用法示例：
  python 评测_集成.py --模型 "c1.pt:4:0.5" "c2.pt:7:0.5" "c3.pt:3:0.7" [--集 S13]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

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

验证集S13 = (
    ("S13", "今天天气真不错，我们去外面走走吧。"),
    ("S14", "他把钥匙放在门口的桌子上。"),
    ("S15", "这家饭馆的红烧肉做得很好吃。"),
    ("S16", "明天上午我要去银行办点事。"),
    ("S17", "孩子们在操场上跑来跑去。"),
    ("S18", "雨停了，天上的云慢慢散开。"),
)


def 解析(s):
    p = s.split("|")
    return p[0], int(p[1]) if len(p) > 1 else 4, float(p[2]) if len(p) > 2 else 0.5


def 单mel(模型, 文本, 迭代, 阻尼, 每):
    """复刻 模型_学说话.合成mel 的归一化 mel（返回归一化前的 log-mel）。"""
    tokens, 有效, 跳过 = M.文本到token(模型.词表, 文本)
    帧数 = np.full(len(tokens), 每, dtype=np.int64)
    声, 韵, 调, 特征 = M.展开帧(tokens, 帧数)
    设备 = next(模型.parameters()).device
    张 = lambda a: torch.from_numpy(a).unsqueeze(0).to(设备)  # noqa: E731
    K = int(getattr(模型, "上一帧数", 0))
    with torch.no_grad():
        prev = None
        平滑 = None
        for _ in range(max(int(迭代), 1)):
            出 = 模型(张(声), 张(韵), 张(调), 张(特征), prev)
            if K > 0:
                平滑 = 出 if 平滑 is None else (阻尼 * 出 + (1.0 - 阻尼) * 平滑)
                prev = M._上帧张量(平滑, K)
    mel = 出[0].detach().cpu().numpy().astype(np.float32)
    m = np.asarray(模型.mel均值, np.float32).reshape(-1)
    sd = np.asarray(模型.mel标准差, np.float32).reshape(-1)
    return mel * (sd[0] if sd.size == 1 else sd.reshape(1, -1)) + \
        (m[0] if m.size == 1 else m.reshape(1, -1))


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--模型", nargs="+", required=True, help="ckpt:迭代:阻尼 …")
    ap.add_argument("--集", choices=("S13", "双集"), default="S13")
    ap.add_argument("--每音节帧数", type=int, default=24)
    ap.add_argument("--出", default=str(M66 / "产物" / "音频_集成"))
    a = ap.parse_args()
    模型s = []
    for spec in a.模型:
        ck, it, da = 解析(spec)
        模型s.append((M.加载(ck), it, da, Path(ck).stem))
    出目录 = Path(a.出)
    出目录.mkdir(parents=True, exist_ok=True)
    记录 = []
    for 集合名, 句s in (("S13", 验证集S13), ("历史", V.句集历史), ("留出", V.句集留出)):
        if a.集 == "S13" and 集合名 != "S13":
            continue
        for 句号, 句 in 句s:
            mels = [单mel(模, 句, it, da, a.每音节帧数) for 模, it, da, _ in 模型s]
            n = min(x.shape[0] for x in mels)
            mel = np.mean([x[:n] for x in mels], axis=0).astype(np.float32)
            wav = V.mel转波(mel)
            p = 出目录 / ("集成_%s_%s.wav" % (集合名, 句号))
            V.写wav(p, wav)
            记录.append(dict(集合=集合名, 句号=句号, 文=句, 路径=str(p)))
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
    for 名 in ("S13", "历史", "留出"):
        vs = [x["CER"] for x in 记录 if x["集合"] == 名]
        if vs:
            print("%s 均值 CER=%.4f" % (名, float(np.mean(vs))))
    out = M66 / "产物" / "判据" / "判据_集成.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(模型=a.模型, 集=a.集, 每音节帧数=a.每音节帧数,
                                   逐句=记录), ensure_ascii=False, indent=1), encoding="utf-8")
    print("→", out)


if __name__ == "__main__":
    raise SystemExit(主())
