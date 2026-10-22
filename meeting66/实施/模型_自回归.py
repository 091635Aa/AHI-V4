# -*- coding: utf-8 -*-
"""meeting66 / 模型_自回归.py —— 自回归帧级解码器（非 Transformer、无 attention）。

所有者裁决（2026-10-20）：「换生成范式 或者修改核心 有必要时核心可以动」。
本件是**核心改动**：不再用「全帧并行回归」，改为**逐帧自回归生成**：
  编码器（音素 + 帧特征 → 上下文 [T,C]，Conv1D 残差堆）
  解码器（GRU，逐帧：输入 = [上下文_t, 上一帧 mel] → 隐藏 → mel_t，含上一帧直连残差）
训练：teacher forcing（nn.GRU 整段并行，快）；推理：逐帧生成（无 teacher prev 输入，
      训练/推理同构，从根上避开「全帧回归过度平滑 + 计划采样又漂移」的两难）。
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


class 自回归声学(nn.Module):
    def __init__(self, 词表, 嵌入维=128, 通道=256, 卷积核=5, 膨胀=(1, 2, 4, 8),
                 隐藏=384, 输出维=M.MEL带, 特征维=M.帧特征数, 上一帧数=4):
        super().__init__()
        self.词表 = 词表
        self.上一帧数 = int(上一帧数)
        self.配置 = dict(嵌入维=嵌入维, 通道=通道, 卷积核=卷积核, 膨胀=list(膨胀),
                        隐藏=隐藏, 输出维=输出维, 特征维=特征维,
                        上一帧数=self.上一帧数)
        self.声母嵌入 = nn.Embedding(len(词表["声母"]), 嵌入维)
        self.韵母嵌入 = nn.Embedding(len(词表["韵母"]), 嵌入维)
        self.调嵌入 = nn.Embedding(len(词表["调"]), 嵌入维)
        self.输入投影 = nn.Linear(3 * 嵌入维 + 特征维, 通道)
        self.输入范数 = nn.LayerNorm(通道)
        self.块s = nn.ModuleList([M.残差块(通道, 卷积核, d) for d in 膨胀])
        self.末范数 = nn.LayerNorm(通道)
        self.声学投影 = nn.Linear(self.上一帧数 * 输出维, 通道)
        self.融合 = nn.Linear(2 * 通道, 通道)
        self.gru = nn.GRU(通道, 隐藏, batch_first=True)
        self.输出 = nn.Linear(隐藏, 输出维)
        self.旁路 = nn.Linear(self.上一帧数 * 输出维, 输出维)

    def 编码(self, 声, 韵, 调, 特征):
        e = torch.cat([self.声母嵌入(声), self.韵母嵌入(韵),
                       self.调嵌入(调), 特征], dim=-1)
        x = self.输入范数(self.输入投影(e))
        h = x.transpose(1, 2)
        for 块 in self.块s:
            h = 块(h)
        h = self.末范数(h.transpose(1, 2))   # [B,T,C]
        return F.gelu(h)  # [B,T,C]

    def forward(self, 声, 韵, 调, 特征, mel_prev):
        """teacher forcing：[B,T,K*80] 前 K 帧（t<K 为 0）→ [B,T,80]。"""
        C = self.编码(声, 韵, 调, 特征)
        a = self.声学投影(mel_prev)
        x = F.gelu(self.融合(torch.cat([C, a], dim=-1)))
        h, _ = self.gru(x)
        return self.输出(h) + self.旁路(mel_prev)

    def 生成_可导(self, 声, 韵, 调, 特征):
        """自由运行生成（可导，供 free-running 微调；推理请用 生成）。"""
        C = self.编码(声, 韵, 调, 特征)
        B, T, _ = C.shape
        h = None
        维 = self.配置["输出维"]
        K = self.上一帧数
        buf = torch.zeros(B, 1, K * 维, device=C.device, dtype=C.dtype)
        outs = []
        for t in range(T):
            a = self.声学投影(buf)
            x = F.gelu(self.融合(torch.cat([C[:, t:t + 1], a], dim=-1)))
            o, h = self.gru(x, h)
            mel_t = self.输出(o) + self.旁路(buf)
            # 更新 K 帧缓冲：新帧放最前，旧帧右移
            buf = torch.cat([mel_t, buf[:, :, :(K - 1) * 维]], dim=2)
            outs.append(mel_t)
        return torch.cat(outs, dim=1)

    @torch.no_grad()
    def 生成(self, 声, 韵, 调, 特征):
        return self.生成_可导(声, 韵, 调, 特征)


def _保存(path: Path, 模型, 词表, 均值, 标准差, 参, 训练信息):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in 模型.state_dict().items()},
                    词表=词表, 配置=模型.配置, 参数数=int(参),
                    mel均值=np.asarray(均值, np.float32),
                    mel标准差=np.asarray(标准差, np.float32),
                    训练信息=训练信息), tmp)
    import os
    os.replace(tmp, path)


def 训练自回归(数据路=数据目录, 出=模型目录 / "模型_自回归", 分钟=25.0, 通道=256,
              膨胀=(1, 2, 4, 8), 隐藏=384, 学习率=2e-4, 周期保存=50, 批样本=8,
              最大帧=3000, 种子=300, 自条件p=0.5, 阻尼=0.7, 初始化=None,
              自由运行p=0.0, 自由运行权=0.3, 上一帧数=4):
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
    模型 = 自回归声学(词表, 通道=通道, 膨胀=tuple(膨胀), 隐藏=隐藏, 上一帧数=上一帧数).to(设备)
    if 初始化:
        _ck = torch.load(str(初始化), map_location="cpu", weights_only=False)
        _缺 = 模型.load_state_dict(_ck["state_dict"], strict=False)
        print("[初始化] 从 %s 载入；missing=%d unexpected=%d"
              % (初始化, len(_缺.missing_keys), len(_缺.unexpected_keys)), flush=True)
    参 = int(sum(p.numel() for p in 模型.parameters()))
    opt = torch.optim.AdamW(模型.parameters(), lr=学习率, weight_decay=1e-4)
    rng = np.random.default_rng(种子)
    训练ids = 数据集_.训练ids
    开始 = time.time()
    总预算 = float(分钟 * 60.0)
    曲线 = dict(轮=[], 训练损失=[])
    print("[T-I] 自回归声学 参数=%s 预算=%.0fs" % (f"{参:,}", 总预算), flush=True)

    def 造窗():
        窗口s = []
        for sid in rng.permutation(训练ids):
            s = 数据集_.样本[sid]
            N = len(s["帧数"])
            if int(s["mel"].shape[0]) <= 最大帧:
                窗口s.append(M.裁窗(s, 0, 最大帧))
            else:
                for _ in range(3):
                    起 = int(rng.integers(0, N))
                    w = M.裁窗(s, 起, 最大帧)
                    if w is not None and len(w["帧数"]) >= 6:
                        窗口s.append(w)
                        break
        return 窗口s

    轮 = 0
    while (time.time() - 开始) < 总预算:
        轮 += 1
        模型.train()
        总损, 总帧 = 0.0, 0
        for 批 in M.起批(造窗(), 批样本, 最大帧):
            b = M.拼批(批, 均值, 标准差)
            b = {k: v.to(设备, non_blocking=True) for k, v in b.items()}
            真 = b["mel"]
            上一 = M._上帧张量(真, 模型.上一帧数)
            预测 = 模型(b["声"], b["韵"], b["调"], b["特征"], 上一)
            损 = M.批损失(预测, 真, b["掩码"], 均值_t, 标准差_t, 配置)
            # 自条件第二遍（AR 版计划采样）：用自己第一遍的输出（detach）+ 阻尼混合当 prev，
            # 让训练见到「自己会漂」的输入，抑制自回归曝光偏差。
            if 自条件p > 0 and float(np.random.random()) < 自条件p:
                自prev = M._上帧张量(预测.detach(), 模型.上一帧数)
                混合 = 阻尼 * 自prev + (1.0 - 阻尼) * 上一
                预测2 = 模型(b["声"], b["韵"], b["调"], b["特征"], 混合)
                损 = 损 + M.批损失(预测2, 真, b["掩码"], 均值_t, 标准差_t, 配置)
            # 自由运行微调（free-running）：直接滚推理轨迹、整段算损失（只取前 400 帧控制成本）
            if 自由运行p > 0 and float(np.random.random()) < 自由运行p:
                短 = min(400, int(真.shape[1]))
                if 短 >= 8:
                    滚 = 模型.生成_可导(b["声"][:, :短], b["韵"][:, :短],
                                        b["调"][:, :短], b["特征"][:, :短])
                    损 = 损 + 自由运行权 * M.批损失(滚, 真[:, :短], b["掩码"][:, :短],
                                                   均值_t, 标准差_t, 配置)
            opt.zero_grad(set_to_none=True)
            损.backward()
            torch.nn.utils.clip_grad_norm_(模型.parameters(), 1.0)
            opt.step()
            n = int(b["掩码"].sum().item())
            总损 += float(损.item()) * n
            总帧 += n
        本 = 总损 / max(总帧, 1)
        曲线["轮"].append(轮)
        曲线["训练损失"].append(round(本, 6))
        if 轮 % 5 == 0 or 轮 == 1:
            print("  轮%4d 损 %.5f 已用 %.0fs" % (轮, 本, time.time() - 开始), flush=True)
        if 周期保存 and 轮 % 周期保存 == 0:
            _保存(出 / ("ckpt_e%04d.pt" % 轮), 模型, 词表, 均值, 标准差, 参,
                  dict(轮=轮, 训练损失=本))
    _保存(出 / "ckpt.pt", 模型, 词表, 均值, 标准差, 参,
          dict(轮=轮, 训练损失=曲线["训练损失"][-1] if 曲线["训练损失"] else None,
               训练秒=round(time.time() - 开始, 1)))
    (出 / "训练曲线.json").write_text(
        json.dumps(dict(配置=配置, 曲线=曲线, 参数数=参), ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("[T-I] 结束：%d 轮 → %s" % (轮, 出 / "ckpt.pt"), flush=True)
    return dict(参数数=参, 轮=轮)


def 加载自回归(ckpt路径, 设备=None):
    设备 = 设备 or M._设备()
    ck = torch.load(str(ckpt路径), map_location="cpu", weights_only=False)
    模型 = 自回归声学(ck["词表"], **ck["配置"]).to(设备)
    模型.load_state_dict(ck["state_dict"])
    模型.eval()
    模型.mel均值 = np.asarray(ck["mel均值"], np.float32)
    模型.mel标准差 = np.asarray(ck["mel标准差"], np.float32)
    return 模型


def 合成mel_自回归(模型, 文本: str, 每音节帧数=24):
    每 = max(int(每音节帧数), 4)
    tokens, 有效, 跳过 = M.文本到token(模型.词表, 文本)
    if len(tokens) == 0:
        raise ValueError("文本无可合成音节：%r" % (文本,))
    帧数 = np.full(len(tokens), 每, dtype=np.int64)
    声, 韵, 调, 特征 = M.展开帧(tokens, 帧数)
    设备 = next(模型.parameters()).device
    张 = lambda a: torch.from_numpy(a).unsqueeze(0).to(设备)  # noqa: E731
    出 = 模型.生成(张(声), 张(韵), 张(调), 张(特征))
    mel = 出[0].detach().cpu().numpy().astype(np.float32)
    m = np.asarray(模型.mel均值, np.float32).reshape(-1)
    sd = np.asarray(模型.mel标准差, np.float32).reshape(-1)
    return (mel * (sd[0] if sd.size == 1 else sd.reshape(1, -1)) +
            (m[0] if m.size == 1 else m.reshape(1, -1))).astype(np.float32)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("命令", choices=["训练"])
    ap.add_argument("--数据", default=str(数据目录))
    ap.add_argument("--出", default=str(模型目录 / "模型_自回归"))
    ap.add_argument("--分钟", type=float, default=25.0)
    ap.add_argument("--通道", type=int, default=256)
    ap.add_argument("--膨胀", default="1,2,4,8")
    ap.add_argument("--隐藏", type=int, default=384)
    ap.add_argument("--学习率", type=float, default=2e-4)
    ap.add_argument("--周期保存", type=int, default=50)
    ap.add_argument("--最大帧", type=int, default=3000)
    ap.add_argument("--自条件p", type=float, default=0.5, help="AR 版计划采样概率（第二遍用自产 prev）")
    ap.add_argument("--阻尼", type=float, default=0.7, help="自产 prev 与真值 prev 的混合系数")
    ap.add_argument("--初始化", default=None, help="从已有 AR ckpt 继续训练")
    ap.add_argument("--自由运行p", type=float, default=0.0, help="自由运行微调概率（滚推理轨迹算损失）")
    ap.add_argument("--自由运行权", type=float, default=0.3, help="自由运行损失权重")
    ap.add_argument("--上一帧数", type=int, default=4, help="AR 输入前 K 帧 mel")
    a = ap.parse_args()
    训练自回归(数据路=Path(a.数据), 出=Path(a.出), 分钟=a.分钟, 通道=a.通道,
              膨胀=tuple(int(x) for x in a.膨胀.split(",")), 隐藏=a.隐藏,
              学习率=a.学习率, 周期保存=a.周期保存, 最大帧=a.最大帧,
              自条件p=a.自条件p, 阻尼=a.阻尼, 初始化=a.初始化,
              自由运行p=a.自由运行p, 自由运行权=a.自由运行权, 上一帧数=a.上一帧数)
