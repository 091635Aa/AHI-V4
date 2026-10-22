# -*- coding: utf-8 -*-
"""meeting66 / 模型_对抗.py —— mel 域对抗训练（非 Transformer、无 attention）。

动机（06_实验 §14）：L1(log-mel) 是均值回归损失，推理输出过度平滑。本件加一个**小型 Conv 判别器**
（只作训练信号，不进生成路径；C1 允许）：
  生成器 = 与 v13 同架构的 说话模型（音素+前 K 帧 → mel）；
  判别器 = Conv1d 堆（80→128→256→1，patch 打分）；
  生成器损失 = L1 + λ·BCE(D(假)→真)；判别器损失 = BCE(D(真)→真)+BCE(D(假)→假)。
训练时保持计划采样 p=0.5（与 v6/v13 推理同款自条件），推理仍用 `模型_学说话.合成mel`（迭代+阻尼）。
ckpt 格式与 `模型_学说话` 完全兼容 ⇒ 可直接用 `声码器与评测.py --ckpt` 复评。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
M66 = HERE.parent
AHI4 = M66.parent
for _p in (str(AHI4 / "meeting64" / "实施"), str(AHI4 / "meeting61" / "实施"),
           str(AHI4 / "meeting58" / "实施")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import 模型_学说话 as M  # noqa: E402

数据目录 = M.数据目录
模型目录 = M.模型目录


class 判别器(nn.Module):
    """mel 判别器：[B,T,80] → 每帧 logit，取均值。纯 Conv，无注意力。"""

    def __init__(self, 带=M.MEL带, 通道=128):
        super().__init__()
        self.网络 = nn.Sequential(
            nn.Conv1d(带, 通道, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(通道, 通道 * 2, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(通道 * 2, 通道 * 2, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(通道 * 2, 1, 3, padding=1),
        )

    def forward(self, mel):
        # mel: [B,T,80]
        x = mel.transpose(1, 2)
        return self.网络(x).mean(dim=-1)  # [B,1]


def 训练对抗(数据路=数据目录, 出=模型目录 / "模型_对抗", 分钟=20.0, 通道=320,
            膨胀=(1, 2, 4, 8, 16), 上一帧数=4, 自条件p=0.5, 对抗权重=0.5,
            学习率=2e-4, 周期保存=50, 种子=300, 批样本=8, 最大帧=3000):
    出 = Path(出)
    出.mkdir(parents=True, exist_ok=True)
    M._固定种子(种子)
    数据集_ = M.数据集(数据路, 前N=None, 打印=False)
    设备 = M._设备()
    词表 = 数据集_.词表
    均值, 标准差 = 数据集_.均值, 数据集_.标准差
    均值_t = torch.from_numpy(np.asarray(均值, np.float32)).to(设备)
    标准差_t = torch.from_numpy(np.asarray(标准差, np.float32)).to(设备)
    配置 = dict(M.默认训练配置)
    配置["时长分钟"] = float(分钟)
    配置["自条件p"] = float(自条件p)
    G = M.说话模型(词表, 通道=通道, 膨胀=tuple(膨胀), 上一帧数=上一帧数).to(设备)
    D = 判别器().to(设备)
    参G = M.参数数(G)
    参D = M.参数数(D)
    optG = torch.optim.AdamW(G.parameters(), lr=学习率, weight_decay=1e-4)
    optD = torch.optim.AdamW(D.parameters(), lr=学习率, weight_decay=1e-4)
    rng = np.random.default_rng(种子)
    训练ids = 数据集_.训练ids
    开始 = time.time()
    总预算 = float(分钟 * 60.0)
    曲线 = dict(轮=[], 生成损=[], 判别损=[], 对抗损=[])
    print("[T-G] 对抗训练开始 G=%s D=%s 预算=%.0fs" % (f"{参G:,}", f"{参D:,}", 总预算), flush=True)

    def 造窗():
        窗口s = []
        for sid in rng.permutation(训练ids):
            s = 数据集_.样本[sid]
            N = len(s["帧数"])
            if int(s["mel"].shape[0]) <= 最大帧:
                窗口s.append(M.裁窗(s, 0, 最大帧))
            else:
                for _ in range(4):
                    起 = int(rng.integers(0, N))
                    w = M.裁窗(s, 起, 最大帧)
                    if w is not None and len(w["帧数"]) >= 6:
                        窗口s.append(w)
                        break
        return 窗口s

    bce = nn.BCEWithLogitsLoss()
    轮 = 0
    while (time.time() - 开始) < 总预算:
        轮 += 1
        G.train()
        D.train()
        损G = 损D = 损A = 0.0
        帧 = 0
        for 批 in M.起批(造窗(), 批样本, 最大帧):
            b = M.拼批(批, 均值, 标准差)
            b = {k: v.to(设备, non_blocking=True) for k, v in b.items()}
            上帧 = M._上帧张量(b["mel"], G.上一帧数)
            if 自条件p > 0 and float(np.random.random()) < 自条件p:
                with torch.no_grad():
                    粗 = G(b["声"], b["韵"], b["调"], b["特征"], None)
                上帧 = M._上帧张量(粗, G.上一帧数)
            假 = G(b["声"], b["韵"], b["调"], b["特征"], 上帧)
            # —— 判别器 ——
            真分 = D(b["mel"])
            假分 = D(假.detach())
            损D_b = 0.5 * (bce(真分, torch.ones_like(真分)) + bce(假分, torch.zeros_like(假分)))
            optD.zero_grad(set_to_none=True)
            损D_b.backward()
            optD.step()
            # —— 生成器 ——
            损L1 = M.批损失(假, b["mel"], b["掩码"], 均值_t, 标准差_t, 配置)
            假分2 = D(假)
            损A_b = bce(假分2, torch.ones_like(假分2))
            损G_b = 损L1 + 对抗权重 * 损A_b
            optG.zero_grad(set_to_none=True)
            损G_b.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            optG.step()
            n = int(b["掩码"].sum().item())
            损G += float(损G_b.item()) * n
            损D += float(损D_b.item()) * n
            损A += float(损A_b.item()) * n
            帧 += n
        本G, 本D, 本A = 损G / max(帧, 1), 损D / max(帧, 1), 损A / max(帧, 1)
        曲线["轮"].append(轮)
        曲线["生成损"].append(round(本G, 6))
        曲线["判别损"].append(round(本D, 6))
        曲线["对抗损"].append(round(本A, 6))
        if 轮 % 5 == 0 or 轮 == 1:
            print("  轮%4d G %.5f D %.5f A %.5f 已用 %.0fs"
                  % (轮, 本G, 本D, 本A, time.time() - 开始), flush=True)
        if 周期保存 and 轮 % 周期保存 == 0:
            _保存(出 / ("ckpt_e%04d.pt" % 轮), G, 词表, 均值, 标准差, G.配置, 参G,
                  dict(轮=轮, 生成损=本G, 判别损=本D, 对抗损=本A))
    _保存(出 / "ckpt.pt", G, 词表, 均值, 标准差, G.配置, 参G,
          dict(轮=轮, 生成损=曲线["生成损"][-1] if 曲线["生成损"] else None,
               训练秒=round(time.time() - 开始, 1)))
    (出 / "训练曲线.json").write_text(
        json.dumps(dict(配置=配置, 曲线=曲线, 参数G=参G, 参数D=参D), ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("[T-G] 结束：%d 轮 → %s" % (轮, 出 / "ckpt.pt"), flush=True)
    return dict(参数G=参G, 轮=轮)


def _保存(path: Path, G, 词表, 均值, 标准差, 配置, 参, 训练信息):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    state = {k: v.detach().cpu().clone() for k, v in G.state_dict().items()}
    torch.save(dict(state_dict=state, 词表=词表, 配置=配置, 参数数=int(参),
                    mel均值=np.asarray(均值, np.float32),
                    mel标准差=np.asarray(标准差, np.float32),
                    训练信息=训练信息), tmp)
    import os
    os.replace(tmp, path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--数据", default=str(数据目录))
    ap.add_argument("--出", default=str(模型目录 / "模型_对抗"))
    ap.add_argument("--分钟", type=float, default=20.0)
    ap.add_argument("--通道", type=int, default=320)
    ap.add_argument("--膨胀", default="1,2,4,8,16")
    ap.add_argument("--上一帧数", type=int, default=4)
    ap.add_argument("--自条件p", type=float, default=0.5)
    ap.add_argument("--对抗权重", type=float, default=0.5)
    ap.add_argument("--学习率", type=float, default=2e-4)
    ap.add_argument("--周期保存", type=int, default=50)
    a = ap.parse_args()
    训练对抗(数据路=Path(a.数据), 出=Path(a.出), 分钟=a.分钟, 通道=a.通道,
            膨胀=tuple(int(x) for x in a.膨胀.split(",")), 上一帧数=a.上一帧数,
            自条件p=a.自条件p, 对抗权重=a.对抗权重, 学习率=a.学习率,
            周期保存=a.周期保存)
