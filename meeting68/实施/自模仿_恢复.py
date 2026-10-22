# -*- coding: utf-8 -*-
"""meeting68/实施/自模仿_恢复.py —— 自模仿·恢复（v2；所有者修正后主线）。

思路（不强制参数、非 Transformer、非传统渲染管线）：
  告诉它「这是你自己的声音」→ 把这段自己的声音弄退化（增益/低通/噪声/时间掩蔽）
  → 小网络（Conv1D 编码-解码 + 语气潜变量 z）恢复 → 用自己的耳（可微 mel 统计）评判
  → 课程式一步步引导（L1→L4，级间门）。z 由网络自监督学出，不喂情绪标签。

命令：
  & $py 自模仿_恢复.py 训练            # 课程训练 + 直训 L4 对照 + 评测
  & $py 自模仿_恢复.py 自检            # 只做数据/退化/前向自检（不训练）
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
M68 = HERE.parent
M66 = Path(r"<workspace>/03_AHI\AHI4\meeting66")
数据根 = Path(r"<path>\SpeakerA7代模型训练数据")
出根 = M68 / "产物" / "自模仿恢复"
SR = 22050
N_FFT = 1024
HOP = 220
WIN = 551
MEL = 80
T = 500                     # 5 s 窗口
种子 = 300
train数, val数 = 120, 20
课程 = [1, 2, 3, 4]
每级步数 = 250
批次 = 2
lr = 2e-4


# ───────────────────────────── 数据 ─────────────────────────────
def 载mel(路: Path) -> np.ndarray:
    import librosa
    x, sr = sf.read(str(路), dtype="float32", always_2d=True)
    m = np.ascontiguousarray(x.mean(axis=1))
    if sr != SR:
        g = math.gcd(int(sr), SR)
        import scipy.signal as sps
        m = np.ascontiguousarray(sps.resample_poly(m, SR // g, int(sr) // g).astype(np.float32))
    S = librosa.feature.melspectrogram(y=m, sr=SR, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
                                       window="hann", center=True, pad_mode="constant",
                                       power=2.0, n_mels=MEL, fmin=0, fmax=SR / 2.0,
                                       norm="slaney", htk=False)
    return np.log(S + 1e-6).T.astype(np.float32)      # [Tframes, 80]


def 载数据():
    fs = sorted(数据根.glob("*.wav"))
    if len(fs) < train数 + val数 + 10:
        raise FileNotFoundError("数据不足")
    rng = np.random.default_rng(种子)
    取 = rng.permutation(len(fs))
    训 = [fs[i] for i in 取[:train数]]
    验 = [fs[i] for i in 取[train数:train数 + val数]]
    出 = {}
    for 名, 组 in (("train", 训), ("val", 验)):
        ms = []
        for p in 组:
            try:
                ms.append(载mel(p))
            except Exception as e:  # noqa: BLE001
                print("[跳过]", p.name, type(e).__name__)
        出[名] = ms
    print(f"[数据] train {len(出['train'])} 条 / val {len(出['val'])} 条")
    return 出


# ───────────────────────────── 退化（mel 域，确定性） ─────────────────────────────
def 退化(mel: torch.Tensor, 级: int, rs: np.random.RandomState) -> torch.Tensor:
    x = mel.clone()
    b = torch.linspace(0.0, 1.0, x.shape[2], device=x.device)
    if 级 >= 1:
        x = x + float(rs.uniform(-0.7, 0.7))                    # ±0.7 ln ≈ ±6 dB
    if 级 >= 2:
        强 = float(rs.uniform(0.5, 1.0))
        x = x - 2.9 * 强 * (b ** 2)[None, None, :]              # 高频最多 −25 dB
    if 级 >= 3:
        σ = float(rs.uniform(0.05, 0.2))
        noise = torch.from_numpy(rs.randn(*x.shape).astype(np.float32)).to(x.device)
        x = x + noise * σ
    if 级 >= 4:
        n = int(rs.randint(2, 4))
        for _ in range(n):
            L = int(rs.randint(8, 16))
            上 = max(1, x.shape[0] - L)
            s = int(rs.randint(0, 上))
            x[s:s + L, :] = float(x.min())
    return x


# ───────────────────────────── 可微「耳统计」（自监督损失） ─────────────────────────────
def 耳统计(mel: torch.Tensor) -> torch.Tensor:
    """mel [B,T,80]（log 域）→ 6 维语气/音色统计（可微）。"""
    e = mel.mean(dim=2)                       # [B,T] 帧能量代理
    b = torch.linspace(0.0, 1.0, mel.shape[2], device=mel.device)
    w = torch.softmax(mel.mean(dim=1) * 2.0, dim=1)      # [B,80]
    cen = (w * b[None, :]).sum(dim=1)
    cen帧 = (torch.softmax(mel * 2.0, dim=2) * b[None, None, :]).sum(dim=2)
    低 = mel[:, :, :20].mean(dim=(1, 2))
    高 = mel[:, :, 60:].mean(dim=(1, 2))
    return torch.stack([e.mean(1), e.std(1), cen, cen帧.std(1), 低, 高], dim=1)


# ───────────────────────────── 网络（Conv1D，无 attention/Transformer） ─────────────────────────────
class 残差块(nn.Module):
    def __init__(self, ch, d):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 5, padding=2 * d, dilation=d)
        self.c2 = nn.Conv1d(ch, ch, 5, padding=2, dilation=1)
        self.n1 = nn.GroupNorm(8, ch); self.n2 = nn.GroupNorm(8, ch)

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.gelu(x + h)


class 恢复器(nn.Module):
    def __init__(self, ch=112, z=8):
        super().__init__()
        self.inp = nn.Conv1d(MEL, ch, 5, padding=2)
        self.enc = nn.Sequential(残差块(ch, 1), 残差块(ch, 2), 残差块(ch, 4))
        self.z头 = nn.Linear(ch, z)
        self.语气头 = nn.Linear(z, 6)
        self.融合 = nn.Conv1d(ch + z, ch, 1)
        self.dec = nn.Sequential(残差块(ch, 1), 残差块(ch, 2), 残差块(ch, 4))
        self.out = nn.Conv1d(ch, MEL, 1)

    def forward(self, x, z外部=None):         # x: [B,T,80]
        h = x.transpose(1, 2)
        h = F.gelu(self.inp(h))
        h = self.enc(h)
        z = self.z头(h.mean(dim=2)) if z外部 is None else z外部
        zg = z[:, :, None].expand(-1, -1, h.shape[2])
        h = F.gelu(self.融合(torch.cat([h, zg], dim=1)))
        h = self.dec(h)
        r = self.out(h).transpose(1, 2)
        return x + r, z, self.语气头(z)


# ───────────────────────────── 训练/评测 ─────────────────────────────
def 取窗(m, rs: np.random.RandomState, n=T):
    if m.shape[0] <= n:
        return torch.from_numpy(m).float()
    s = int(rs.randint(0, m.shape[0] - n))
    return torch.from_numpy(m[s:s + n]).float()


def 评val(模型, 验, 级, dev):
    模型.eval()
    ds, rs_ = [], []
    with torch.no_grad():
        for m in 验:
            if m.shape[0] < T:
                continue
            x = torch.from_numpy(m[:T]).float().to(dev)[None]
            rs = np.random.RandomState(种子 + 99 + 级 * 7)   # 每级固定退化，可复现
            deg = 退化(x, 级, rs)
            恢, z, _ = 模型(deg)
            真 = 耳统计(x)
            ds.append(float((耳统计(deg) - 真).pow(2).mean()))
            rs_.append(float((耳统计(恢) - 真).pow(2).mean()))
    return (float(np.mean(ds)), float(np.mean(rs_))) if ds else (0.0, 0.0)


def 训练一级(模型, 训, 验, 级, 步数, dev, opt=None):
    if opt is None:
        opt = torch.optim.Adam(模型.parameters(), lr=lr)
    曲线 = []
    for 步 in range(1, 步数 + 1):
        模型.train()
        rs = np.random.RandomState(种子 * 100 + 级 * 1000 + 步)
        批 = [取窗(训[rs.randint(0, len(训))], rs).to(dev) for _ in range(批次)]
        X = torch.stack(批)
        级s = int(rs.randint(1, 级 + 1))          # 课程回放：本级内随机抽 ≤级 的退化，防遗忘
        d = 退化(X, 级s, rs)
        恢, z, 语气 = 模型(d)
        loss = F.mse_loss(恢, X) + 0.2 * F.mse_loss(耳统计(恢), 耳统计(X)) + 0.1 * F.mse_loss(语气, 耳统计(X))
        opt.zero_grad(); loss.backward(); opt.step()
        if 步 % 50 == 0 or 步 == 步数:
            dv, rv = 评val(模型, 验, 级, dev)
            impr = (dv - rv) / dv if dv > 1e-12 else 0.0
            曲线.append(dict(级=级, 步=步, 训练损失=round(float(loss.item()), 6),
                              val_退化距离=round(dv, 6), val_恢复距离=round(rv, 6),
                              val_改善=round(float(impr), 4)))
            print(f"[L{级} 步{步}] loss {loss.item():.5f} val 退化 {dv:.5f} → 恢复 {rv:.5f} (改善 {impr*100:.1f}%)")
    return 曲线


def 主(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="自模仿·恢复 v2")
    子 = ap.add_subparsers(dest="命令", required=True)
    子.add_parser("训练", help="课程训练 + 直训 L4 对照 + 评测")
    子.add_parser("自检", help="数据/退化/前向自检")
    参 = ap.parse_args(argv)
    torch.manual_seed(种子); np.random.seed(种子); random.seed(种子)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    数据 = 载数据()
    if 参.命令 == "自检":
        模型 = 恢复器().to(dev)
        rs = np.random.RandomState(种子)
        x = 取窗(数据["train"][0], rs).to(dev)[None]
        d = 退化(x, 4, rs)
        r, z, t = 模型(d)
        print("自检 OK:", dict(输入=list(x.shape), 退化有限=bool(torch.isfinite(d).all()),
                              恢复有限=bool(torch.isfinite(r).all()), z=z.shape,
                              参数量=sum(p.numel() for p in 模型.parameters())))
        return 0
    出根.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[设备] {dev}")
    模型 = 恢复器().to(dev)
    print(f"[参数] {sum(p.numel() for p in 模型.parameters())}")
    曲线 = []
    for 级 in 课程:
        曲线 += 训练一级(模型, 数据["train"], 数据["val"], 级, 每级步数, dev)
    # 巩固：低强度两级再练 100 步（防高阶课程把 L1/L2 冲掉）
    曲线 += 训练一级(模型, 数据["train"], 数据["val"], 2, 100, dev)
    # 直训 L4 对照（同总步数）
    直模 = 恢复器().to(dev)
    直曲线 = 训练一级(直模, 数据["train"], 数据["val"], 4, 每级步数 * len(课程), dev)
    # 评测：全 4 级 val 恢复改善 + z 语气探针
    评测 = {}
    for 级 in 课程:
        dv, rv = 评val(模型, 数据["val"], 级, dev)
        评测[f"L{级}"] = dict(退化距离=round(dv, 6), 恢复距离=round(rv, 6),
                              改善=round((dv - rv) / dv if dv > 1e-12 else 0.0, 4))
    dv直, rv直 = 评val(直模, 数据["val"], 4, dev)
    模型.eval()
    zs, ts = [], []
    with torch.no_grad():
        for m in 数据["val"]:
            if m.shape[0] < T:
                continue
            x = torch.from_numpy(m[:T]).float().to(dev)[None]
            gen = torch.Generator(device="cpu").manual_seed(种子)
            _, z, _ = 模型(x)
            zs.append(z[0].cpu().numpy()); ts.append(耳统计(x)[0].cpu().numpy())
    zs = np.stack(zs); ts = np.stack(ts)
    # 线性探针 R²（z → 语气统计）
    A = np.concatenate([zs, np.ones((len(zs), 1))], axis=1)
    r2 = []
    for k in range(ts.shape[1]):
        coef, *_ = np.linalg.lstsq(A, ts[:, k], rcond=None)
        pred = A @ coef
        ss = ((ts[:, k] - ts[:, k].mean()) ** 2).sum()
        r2.append(float(1 - ((ts[:, k] - pred) ** 2).sum() / ss) if ss > 1e-12 else 0.0)
    torch.save(dict(state_dict=模型.state_dict(), config=dict(ch=112, z=8)), 出根 / "恢复器.pt")
    torch.save(dict(state_dict=直模.state_dict(), config=dict(ch=112, z=8)), 出根 / "恢复器_直训L4.pt")
    报 = dict(工具="自模仿_恢复.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 设备=dev,
              数据=dict(train=len(数据["train"]), val=len(数据["val"]), T=T, sr=SR),
              课程=dict(级=课程, 每级步数=每级步数, 批次=批次, lr=lr),
              训练曲线=曲线, 直训L4曲线=直曲线,
              评测=评测, 直训L4=dict(退化距离=round(dv直, 6), 恢复距离=round(rv直, 6),
                                    改善=round((dv直 - rv直) / dv直 if dv直 > 1e-12 else 0.0, 4)),
              语气探针R2=[round(v, 4) for v in r2],
              参数量=int(sum(p.numel() for p in 模型.parameters())),
              耗时秒=round(time.time() - t0, 1),
              边界="自监督恢复的是『统计上像自己』；z 为自监督语气潜变量，无情绪标签；mel→GL 未做人耳")
    (出根 / "训练报告.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez(出根 / "z与语气.npz", z=zs, 语气=ts)
    print(json.dumps({k: 报[k] for k in ("评测", "直训L4", "语气探针R2", "参数量", "耗时秒")},
                     ensure_ascii=False, indent=1))
    print("已写：", 出根 / "训练报告.json")
    return 0


if __name__ == "__main__":
    sys.exit(主())
