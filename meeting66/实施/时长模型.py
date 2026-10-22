# -*- coding: utf-8 -*-
"""meeting66 / 时长模型.py —— 学习型音节时长预测（非 Transformer、无 attention）。

动机：推理时每音节固定 24 帧，而训练数据的音节帧数 p5–p95 = 12–65 帧。
固定时长会把长韵母压短、把短音节拉长，直接破坏节奏与共振峰过渡（ASR 也会对错时间）。
本件从对齐数据学「音节音素 → 帧数」：Conv1d 序列模型（核 3、dilation 1/2/4、通道 128），
输入音节三嵌入 + 相邻上下文（Conv 提供），输出 log1p(帧数)，L1 损失。
推理：文本 → 音节序列 → 每音节预测帧数（≥4）→ 交给 模型_学说话.展开帧 + 自条件迭代。
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


class 时长模型(nn.Module):
    def __init__(self, 词表, 嵌入维=64, 通道=128):
        super().__init__()
        self.词表 = 词表
        self.配置 = dict(嵌入维=嵌入维, 通道=通道)
        self.声母嵌入 = nn.Embedding(len(词表["声母"]), 嵌入维)
        self.韵母嵌入 = nn.Embedding(len(词表["韵母"]), 嵌入维)
        self.调嵌入 = nn.Embedding(len(词表["调"]), 嵌入维)
        self.输入投影 = nn.Linear(3 * 嵌入维 + 2, 通道)
        self.块s = nn.ModuleList([
            nn.Conv1d(通道, 通道, 3, padding=1, dilation=1),
            nn.Conv1d(通道, 通道, 3, padding=2, dilation=2),
            nn.Conv1d(通道, 通道, 3, padding=4, dilation=4),
        ])
        self.范数s = nn.ModuleList([nn.LayerNorm(通道) for _ in range(3)])
        self.输出 = nn.Linear(通道, 1)

    def forward(self, 声, 韵, 调):
        # [B,N] → [B,N] log1p(帧数)
        N = 声.shape[1]
        位置 = torch.linspace(0, 1, N, device=声.device).unsqueeze(0).expand(声.shape[0], -1)
        位置2 = torch.linspace(1, 0, N, device=声.device).unsqueeze(0).expand(声.shape[0], -1)
        e = torch.cat([self.声母嵌入(声), self.韵母嵌入(韵), self.调嵌入(调),
                       位置.unsqueeze(-1), 位置2.unsqueeze(-1)], dim=-1)
        x = self.输入投影(e)
        h = x.transpose(1, 2)
        for 卷, 范 in zip(self.块s, self.范数s):
            h = F.gelu(范((h + 卷(h)).transpose(1, 2)).transpose(1, 2))
        return self.输出(h.transpose(1, 2)).squeeze(-1)  # [B,N]


def 训练时长(数据路=数据目录, 出=模型目录 / "时长模型", 分钟=4.0, 通道=128,
            学习率=2e-3, 种子=300):
    出 = Path(出)
    出.mkdir(parents=True, exist_ok=True)
    M._固定种子(种子)
    数据集_ = M.数据集(数据路, 前N=None, 打印=False)
    设备 = M._设备()
    样本s = [数据集_.样本[i] for i in 数据集_.训练ids]
    词表 = 数据集_.词表
    模型 = 时长模型(词表, 通道=通道).to(设备)
    参 = int(sum(p.numel() for p in 模型.parameters()))
    opt = torch.optim.AdamW(模型.parameters(), lr=学习率, weight_decay=1e-4)
    print("[T-H] 时长模型 参数=%s 样本=%d" % (f"{参:,}", len(样本s)), flush=True)
    开始 = time.time()
    损失s = []
    轮 = 0
    while (time.time() - 开始) < 分钟 * 60.0:
        轮 += 1
        模型.train()
        总, 数 = 0.0, 0
        for s in 样本s:
            tok = s["tokens"]  # [N,3]
            帧 = np.asarray(s["帧数"], np.float32) if "帧数" in s else None
            if 帧 is None:
                continue
            声 = torch.from_numpy(np.asarray(tok[:, 0], np.int64)).unsqueeze(0).to(设备)
            韵 = torch.from_numpy(np.asarray(tok[:, 1], np.int64)).unsqueeze(0).to(设备)
            调 = torch.from_numpy(np.asarray(tok[:, 2], np.int64)).unsqueeze(0).to(设备)
            目标 = torch.from_numpy(np.log1p(帧)).unsqueeze(0).to(设备)
            预 = 模型(声, 韵, 调)
            损 = F.l1_loss(预, 目标)
            opt.zero_grad(set_to_none=True)
            损.backward()
            opt.step()
            总 += float(损.item()) * len(帧)
            数 += len(帧)
        本 = 总 / max(数, 1)
        损失s.append(round(本, 5))
        if 轮 % 20 == 0 or 轮 == 1:
            print("  轮%4d 损 %.5f 已用 %.0fs" % (轮, 本, time.time() - 开始), flush=True)
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in 模型.state_dict().items()},
                    词表=词表, 配置=模型.配置, 参数量=参,
                    训练信息=dict(轮=轮, 损失终=损失s[-1] if 损失s else None,
                                  耗时秒=round(time.time() - 开始, 1))), 出 / "ckpt.pt")
    (出 / "训练曲线.json").write_text(json.dumps(dict(损失=损失s, 参数量=参), indent=1),
                                     encoding="utf-8")
    print("[T-H] 结束 → %s" % (出 / "ckpt.pt"), flush=True)
    return dict(参数量=参, 轮=轮, 损失终=损失s[-1] if 损失s else None)


def 加载时长(ckpt路径, 设备=None):
    设备 = 设备 or M._设备()
    ck = torch.load(str(ckpt路径), map_location="cpu", weights_only=False)
    模型 = 时长模型(ck["词表"], **ck["配置"]).to(设备)
    模型.load_state_dict(ck["state_dict"])
    模型.eval()
    return 模型


def 预测帧数(时长, 词表, 文本: str) -> np.ndarray:
    tokens, 有效, 跳过 = M.文本到token(词表, 文本)
    if len(tokens) == 0:
        raise ValueError("文本无可合成音节：%r" % (文本,))
    设备 = next(时长.parameters()).device
    with torch.no_grad():
        声 = torch.from_numpy(np.asarray(tokens[:, 0], np.int64)).unsqueeze(0).to(设备)
        韵 = torch.from_numpy(np.asarray(tokens[:, 1], np.int64)).unsqueeze(0).to(设备)
        调 = torch.from_numpy(np.asarray(tokens[:, 2], np.int64)).unsqueeze(0).to(设备)
        帧 = torch.expm1(时长(声, 韵, 调))[0].cpu().numpy()
    帧 = np.clip(np.rint(帧), 4, 80).astype(np.int64)
    return 帧


def 合成mel_时长(模型, 时长, 文本: str, 迭代=4, 阻尼=0.5):
    """与 M.合成mel 同流程，但每音节帧数来自时长模型。"""
    tokens, 有效, 跳过 = M.文本到token(模型.词表, 文本)
    if len(tokens) == 0:
        raise ValueError("文本无可合成音节：%r" % (文本,))
    帧数 = 预测帧数(时长, 模型.词表, 文本)
    声, 韵, 调, 特征 = M.展开帧(tokens, 帧数)
    设备 = next(模型.parameters()).device
    张 = lambda a: torch.from_numpy(a).unsqueeze(0).to(设备)  # noqa: E731
    with torch.no_grad():
        prev = None
        平滑 = None
        K = int(getattr(模型, "上一帧数", 0))
        for _ in range(max(int(迭代), 1)):
            出 = 模型(张(声), 张(韵), 张(调), 张(特征), prev)
            if K > 0:
                平滑 = 出 if 平滑 is None else (阻尼 * 出 + (1.0 - 阻尼) * 平滑)
                prev = M._上帧张量(平滑, K)
    mel = 出[0].detach().cpu().numpy().astype(np.float32)
    m = np.asarray(模型.mel均值, np.float32).reshape(-1)
    sd = np.asarray(模型.mel标准差, np.float32).reshape(-1)
    mel = mel * (sd[0] if sd.size == 1 else sd.reshape(1, -1)) + \
        (m[0] if m.size == 1 else m.reshape(1, -1))
    return mel.astype(np.float32), 帧数


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("命令", choices=["训练"])
    ap.add_argument("--数据", default=str(数据目录))
    ap.add_argument("--出", default=str(模型目录 / "时长模型"))
    ap.add_argument("--分钟", type=float, default=4.0)
    ap.add_argument("--通道", type=int, default=128)
    ap.add_argument("--学习率", type=float, default=2e-3)
    a = ap.parse_args()
    训练时长(数据路=Path(a.数据), 出=Path(a.出), 分钟=a.分钟, 通道=a.通道, 学习率=a.学习率)
