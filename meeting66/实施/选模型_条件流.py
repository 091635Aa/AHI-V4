# -*- coding: utf-8 -*-
"""meeting66 / 选模型_条件流.py —— 条件流（v18）的「按推理 CER 选模型 / 选推理设置」。

预注册协议（`07_预注册_独立验证集.md`）：
  · P1–P3 探针只作**快速预筛**（不报）；
  · S13–S18 只用于**选择**（ckpt / 推理设置），不进入对外报告；对外只报 S1–S12 一次；
  · 两候选 CER 差 < 0.02 视为持平，取结构更简单/更早者。

阶段：
  预筛   python 选模型_条件流.py --目录 <模型目录> --阶段 预筛 --集 P
  网格   python 选模型_条件流.py --目录 <模型目录> --阶段 网格 --集 S13 \
             --ckpts ckpt_e0200.pt,ckpt_e0350.pt
  （网格 = 预注册的 8 组推理设置 × 指定 ckpt；写 选模型_条件流.json）

ASR 单进程；同 ckpt 两趟转写不一致的候选作废（C13 延伸）。
"""
from __future__ import annotations

import argparse
import json
import re
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

import 模型_条件流 as F  # noqa: E402
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

# —— 预注册推理设置网格（先于任何 S13 结果写定；零起点为主，噪声起点为对照） ——
# 复核 F-01：训练 x0~N(0,I)，推理零起点不在训练分布；噪声起点（固定种子）同样满足 C13，
# 故网格必须同时含零起点与噪声起点、末轮与平滑两类终值，由 S13 选择而不是先验指定。
网格 = (
    dict(名="N1", 采样="零起点", 步数=8, 迭代=1, 阻尼=1.0, 终值="末轮"),
    dict(名="N3a", 采样="零起点", 步数=8, 迭代=3, 阻尼=0.5, 终值="末轮"),
    dict(名="N4a", 采样="零起点", 步数=8, 迭代=4, 阻尼=0.5, 终值="末轮"),
    dict(名="N4b", 采样="零起点", 步数=8, 迭代=4, 阻尼=1.0, 终值="末轮"),
    dict(名="N7a", 采样="零起点", 步数=8, 迭代=7, 阻尼=0.5, 终值="末轮"),
    dict(名="N4c", 采样="零起点", 步数=8, 迭代=4, 阻尼=0.5, 终值="平滑"),
    dict(名="N4s16", 采样="零起点", 步数=16, 迭代=4, 阻尼=0.5, 终值="末轮"),
    dict(名="Z3a", 采样="噪声", 步数=8, 迭代=3, 阻尼=0.5, 终值="末轮"),
    dict(名="Z4a", 采样="噪声", 步数=8, 迭代=4, 阻尼=0.5, 终值="末轮"),
    dict(名="Z7a", 采样="噪声", 步数=8, 迭代=7, 阻尼=0.5, 终值="末轮"),
    dict(名="Z4c", 采样="噪声", 步数=8, 迭代=4, 阻尼=0.5, 终值="平滑"),
)
默认设置 = dict(名="默认", 采样="零起点", 步数=10, 迭代=4, 阻尼=0.5, 终值="末轮")


def _轮号(名: str) -> int:
    m = re.search(r"e(\d+)\.pt$", str(名))
    if m:
        return int(m.group(1))
    if str(名).startswith("ckpt_末"):
        return 99998
    return 99999


def _网格序(设: dict) -> int:
    for i, x in enumerate(网格):
        if x["名"] == (设 or {}).get("名"):
            return i
    return 999


def _合成(模型, 文, 设, 出目录, 前缀, 每音节帧数=24, 写wav=True):
    mel = F.合成mel_条件流(模型, 文, 每音节帧数=每音节帧数, 迭代=设["迭代"],
                           阻尼=设["阻尼"], 步数=设["步数"], 采样=设["采样"],
                           终值=设.get("终值", "末轮"))
    wav = V.mel转波(mel)
    p = 出目录 / ("%s_%s.wav" % (前缀, 文[:4]))
    if 写wav:
        V.写wav(p, wav)
    return p, mel


def _评(引, 句号, 文, p):
    t1 = 引["转写"](str(p))
    t2 = 引["转写"](str(p))
    return dict(号=句号, 文=文, 转写=t1, CER=CER(文, t1), 两趟一致=bool(t1 == t2))


def _候选ckpt(目录: Path):
    cks = sorted(目录.glob("ckpt_e*.pt"))
    for 名 in ("ckpt_末.pt", "ckpt.pt"):
        if (目录 / 名).exists():
            cks.append(目录 / 名)
    return cks


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--目录", required=True)
    ap.add_argument("--阶段", choices=("预筛", "网格"), default="预筛")
    ap.add_argument("--集", choices=("P", "S13"), default=None)
    ap.add_argument("--ckpts", default=None, help="逗号分隔；缺省=全部周期 ckpt")
    ap.add_argument("--每音节帧数", type=int, default=24)
    ap.add_argument("--出", default=None)
    ap.add_argument("--设备", default="cpu", choices=["cpu", "cuda"],
                    help="选择阶段默认 CPU（与 v13 正式读数同设备口径，也避开 GPU 训练争用）")
    a = ap.parse_args()
    目录 = Path(a.目录)
    出目录 = Path(a.出) if a.出 else 目录 / "_选模型音频_条件流"
    出目录.mkdir(parents=True, exist_ok=True)
    集 = a.集 or ("P" if a.阶段 == "预筛" else "S13")
    句s = 探针P if 集 == "P" else 验证集S13
    if a.ckpts:
        cks = [目录 / x.strip() for x in a.ckpts.split(",") if x.strip()]
    else:
        cks = _候选ckpt(目录)
    if not cks:
        print("没有候选 ckpt：%s" % 目录)
        return 1
    print("[选模型] 阶段=%s 集=%s 候选 ckpt=%d 句数=%d 每音节=%d 帧"
          % (a.阶段, 集, len(cks), len(句s), a.每音节帧数), flush=True)
    引 = 载引擎()
    结果 = []
    for i, ck in enumerate(cks):
        if not ck.exists():
            print("  ! 跳过不存在：%s" % ck)
            continue
        if a.阶段 == "预筛":
            设置 = [默认设置]
        else:
            设置 = list(网格)
        for 设 in 设置:
            模型 = F.加载条件流(ck, 设备=a.设备)
            逐句 = []
            for 号, 文 in 句s:
                p, _mel = _合成(模型, 文, 设, 出目录, "%s_%s" % (ck.stem, 设["名"]),
                               每音节帧数=a.每音节帧数)
                逐句.append(_评(引, 号, 文, p))
            del 模型
            cer均值 = float(np.mean([x["CER"] for x in 逐句]))
            一致 = bool(all(x["两趟一致"] for x in 逐句))
            结果.append(dict(ckpt=ck.name, 设置=dict(设), CER均值=round(cer均值, 4),
                           两趟一致=一致, 逐句=逐句))
            print("  [%d/%d] %-16s %-6s CER=%.4f 一致=%s  转写=%s"
                  % (i + 1, len(cks), ck.name, 设["名"], cer均值, 一致,
                     " | ".join(x["转写"][:12] for x in 逐句)), flush=True)
    有效 = [r for r in 结果 if r["两趟一致"]]
    有效.sort(key=lambda x: (x["CER均值"], _轮号(x["ckpt"]), _网格序(x.get("设置"))))
    # 07/10 预注册持平规则：差 <0.02 视为持平，取**更早/更简单**者（先按轮号、再按网格序）。
    最佳 = None
    if 有效:
        最小 = 有效[0]["CER均值"]
        持平 = [r for r in 有效 if r["CER均值"] <= 最小 + 0.02]
        持平.sort(key=lambda x: (_轮号(x["ckpt"]), _网格序(x.get("设置"))))
        最佳 = 持平[0]
    out = dict(目录=str(目录), 阶段=a.阶段, 集=集, 每音节帧数=a.每音节帧数, 设备=a.设备,
               网格=[dict(x) for x in 网格] if a.阶段 == "网格" else [dict(默认设置)],
               候选=结果,
               最佳=最佳,
               最佳CER=(最佳["CER均值"] if 最佳 else None),
               持平阈=0.02,
               选择规则="按整组均值 CER 最小；差<0.02 视为持平，取更早（轮号小）/更简单（网格序前）者",
               说明="两候选差<0.02 视为持平（取更早/更简单）；只用于选择，不对外报告")
    out名 = "选模型_条件流.json" if a.阶段 == "网格" else "选模型_条件流_预筛.json"
    (目录 / out名).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    if 最佳:
        print("最佳: %s %s CER=%.4f（持平阈 %.2f）→ %s"
              % (最佳["ckpt"], 最佳["设置"]["名"], 最佳["CER均值"], 0.02,
                 目录 / out名), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(主())
