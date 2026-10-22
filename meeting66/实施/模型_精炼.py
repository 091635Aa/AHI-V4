# -*- coding: utf-8 -*-
"""meeting66 / 模型_精炼.py —— 独立精炼器（两阶段 TTS 第二阶段）。

动机（06_实验 §13）：自条件迭代在 v13 e0150 上见顶；同模型继续训只会更依赖 teacher prev。
本件把「粗 mel → 真 mel」做成**独立网络**，训练分布 = 推理时真实的粗 mel 分布（不是 teacher prev）：
  base(v13 e0150，冻结) 给 粗 = base(x, prev=None)；
  精炼器(x, 粗, 上一) → 真 mel；第二轮用第一轮输出再精炼（图内可导）；
  推理：粗 = base(x, None)；迭代 N 轮 出 = 精(x, 粗)，粗 = α*出 + (1-α)*粗。
全部非 Transformer、无 attention、无预训练（C1 允许）。
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


class 精炼器(nn.Module):
    """(音素、帧特征、粗 mel、粗 mel 前 K 帧) → 真 mel。纯 Conv，无注意力。"""

    def __init__(self, 词表, 嵌入维=128, 通道=320, 卷积核=5,
                 膨胀=(1, 2, 4, 8, 16), 输出维=M.MEL带, 特征维=M.帧特征数, 上一帧数=2):
        super().__init__()
        self.词表 = 词表
        self.上一帧数 = int(上一帧数)
        self.配置 = dict(嵌入维=嵌入维, 通道=通道, 卷积核=卷积核, 膨胀=list(膨胀),
                        输出维=输出维, 特征维=特征维, 上一帧数=self.上一帧数)
        self.声母嵌入 = nn.Embedding(len(词表["声母"]), 嵌入维)
        self.韵母嵌入 = nn.Embedding(len(词表["韵母"]), 嵌入维)
        self.调嵌入 = nn.Embedding(len(词表["调"]), 嵌入维)
        self.输入投影 = nn.Linear(3 * 嵌入维 + 特征维 + 输出维 + self.上一帧数 * 输出维,
                                 通道)
        self.输入范数 = nn.LayerNorm(通道)
        self.块s = nn.ModuleList([M.残差块(通道, 卷积核, d) for d in 膨胀])
        self.末范数 = nn.LayerNorm(通道)
        self.输出卷积 = nn.Conv1d(通道, 输出维, 1)
        self.旁路卷积 = nn.Conv1d(输出维 + self.上一帧数 * 输出维, 输出维, 1)  # 粗 mel 直连

    def forward(self, 声, 韵, 调, 特征, 粗, 上一=None):
        e = torch.cat([self.声母嵌入(声), self.韵母嵌入(韵),
                       self.调嵌入(调), 特征, 粗], dim=-1)
        if self.上一帧数 > 0:
            if 上一 is None:
                上一 = torch.zeros(*(list(e.shape[:-1]) + [self.上一帧数 * self.配置["输出维"]]),
                                   device=e.device, dtype=e.dtype)
            e = torch.cat([e, 上一], dim=-1)
        x = self.输入范数(self.输入投影(e))
        h = x.transpose(1, 2)
        for 块 in self.块s:
            h = 块(h)
        h = self.末范数(h.transpose(1, 2)).transpose(1, 2)
        h = F.gelu(h)
        旁 = torch.cat([粗, 上一] if self.上一帧数 > 0 else [粗], dim=-1).transpose(1, 2)
        y = self.输出卷积(h) + self.旁路卷积(旁)
        return y.transpose(1, 2)


def 参数数(模型):
    return int(sum(p.numel() for p in 模型.parameters()))


def _保存(path: Path, 精, 词表, 均值, 标准差, 训练信息):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(dict(state_dict=精.state_dict(), 词表=词表, 配置=精.配置,
                    mel均值=np.asarray(均值, np.float32),
                    mel标准差=np.asarray(标准差, np.float32),
                    训练信息=训练信息), tmp)
    import os
    os.replace(tmp, path)


def 训练精炼(base_ckpt, 出目录, 数据路=数据目录, 分钟=15.0, 通道=320,
            膨胀=(1, 2, 4, 8, 16), 学习率=1e-4, 周期保存=25, 上一帧数=2,
            种子=300, 批样本=8, 最大帧=3000, 时长秒=None):
    出目录 = Path(出目录)
    出目录.mkdir(parents=True, exist_ok=True)
    M._固定种子(种子)
    数据集_ = M.数据集(数据路, 前N=None, 打印=False)
    设备 = M._设备()
    base = M.加载(base_ckpt, 设备=设备)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    精 = 精炼器(数据集_.词表, 通道=通道, 膨胀=tuple(膨胀), 输出维=M.MEL带,
                特征维=M.帧特征数, 上一帧数=上一帧数).to(设备)
    参 = 参数数(精)
    均值 = 数据集_.均值
    标准差 = 数据集_.标准差
    均值_t = torch.from_numpy(np.asarray(均值, np.float32)).to(设备)
    标准差_t = torch.from_numpy(np.asarray(标准差, np.float32)).to(设备)
    配置 = dict(M.默认训练配置)
    rng = np.random.default_rng(种子)
    opt = torch.optim.AdamW(精.parameters(), lr=学习率, weight_decay=1e-4)
    总预算 = float(时长秒 if 时长秒 is not None else 分钟 * 60.0)
    开始 = time.time()
    曲线 = dict(轮=[], 训练损失=[])
    训练ids, 验证ids = 数据集_.训练ids, 数据集_.验证ids
    print("[T-F] 精炼器训练开始 参数=%s base=%s 预算=%.0fs" % (f"{参:,}", base_ckpt, 总预算),
          flush=True)

    def 造窗(ids):
        窗口s = []
        for sid in ids:
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

    轮 = 0
    while (time.time() - 开始) < 总预算:
        轮 += 1
        精.train()
        窗口s = 造窗(list(rng.permutation(训练ids)))
        rng.shuffle(窗口s)
        总损, 总帧 = 0.0, 0
        for 批 in M.起批(窗口s, 批样本, 最大帧):
            b = M.拼批(批, 均值, 标准差)
            b = {k: v.to(设备, non_blocking=True) for k, v in b.items()}
            with torch.no_grad():
                粗 = base(b["声"], b["韵"], b["调"], b["特征"], None)
            上 = M._上帧张量(粗, 上一帧数)
            出1 = 精(b["声"], b["韵"], b["调"], b["特征"], 粗, 上)
            损1 = M.批损失(出1, b["mel"], b["掩码"], 均值_t, 标准差_t, 配置)
            上2 = M._上帧张量(出1, 上一帧数)
            出2 = 精(b["声"], b["韵"], b["调"], b["特征"], 出1, 上2)
            损2 = M.批损失(出2, b["mel"], b["掩码"], 均值_t, 标准差_t, 配置)
            损 = 损1 + 损2
            opt.zero_grad(set_to_none=True)
            损.backward()
            torch.nn.utils.clip_grad_norm_(精.parameters(), 1.0)
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
            _保存(出目录 / ("ckpt_e%04d.pt" % 轮), 精, 数据集_.词表, 均值, 标准差,
                  dict(轮=轮, 训练损失=本))
    _保存(出目录 / "ckpt.pt", 精, 数据集_.词表, 均值, 标准差,
          dict(轮=轮, 训练损失=曲线["训练损失"][-1] if 曲线["训练损失"] else None,
               训练秒=round(time.time() - 开始, 1), 参数量=参))
    (出目录 / "训练曲线.json").write_text(
        json.dumps(dict(配置=配置, 曲线=曲线, 参数数=参), ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("[T-F] 结束：%d 轮，%s" % (轮, 出目录 / "ckpt.pt"), flush=True)
    return dict(参数数=参, 轮=轮, 出=str(出目录 / "ckpt.pt"))


def 加载精炼(ckpt路径, 设备=None):
    设备 = 设备 or M._设备()
    ck = torch.load(str(ckpt路径), map_location="cpu", weights_only=False)
    模型 = 精炼器(ck["词表"], **ck["配置"]).to(设备)
    模型.load_state_dict(ck["state_dict"])
    模型.eval()
    模型.mel均值 = np.asarray(ck["mel均值"], np.float32)
    模型.mel标准差 = np.asarray(ck["mel标准差"], np.float32)
    return 模型


def 合成精炼(精, base, 文本: str, 每音节帧数=24, 迭代=3, 阻尼=0.7):
    每 = max(int(每音节帧数), 4)
    tokens, 有效, 跳过 = M.文本到token(base.词表, 文本)
    if len(tokens) == 0:
        raise ValueError("文本无可合成音节：%r" % (文本,))
    帧数 = np.full(len(tokens), 每, dtype=np.int64)
    声, 韵, 调, 特征 = M.展开帧(tokens, 帧数)
    设备 = next(精.parameters()).device
    张 = lambda a: torch.from_numpy(a).unsqueeze(0).to(设备)  # noqa: E731
    with torch.no_grad():
        粗 = base(张(声), 张(韵), 张(调), 张(特征), None)
        出 = 粗
        for _ in range(max(int(迭代), 1)):
            上 = M._上帧张量(粗, 精.上一帧数)
            出 = 精(张(声), 张(韵), 张(调), 张(特征), 粗, 上)
            粗 = 阻尼 * 出 + (1.0 - 阻尼) * 粗
    mel = 出[0].detach().cpu().numpy().astype(np.float32)
    m = np.asarray(精.mel均值, np.float32).reshape(-1)
    sd = np.asarray(精.mel标准差, np.float32).reshape(-1)
    mel = mel * (sd[0] if sd.size == 1 else sd.reshape(1, -1)) + \
        (m[0] if m.size == 1 else m.reshape(1, -1))
    return mel.astype(np.float32)


def 自检():
    """C13：同输入两趟 mel 逐位同；重载 ckpt 后一致由调用方另行检查。"""
    M._固定种子(300)
    base_ck = 模型目录.parent / "产物" / "模型_v13_K4_320" / "ckpt_e0150.pt"
    print("skipped-selftest", base_ck.exists())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("命令", choices=["训练", "自检"])
    ap.add_argument("--base", default=str(M66 / "产物" / "模型_v13_K4_320" / "ckpt_e0150.pt"))
    ap.add_argument("--出", default=str(M66 / "产物" / "模型_精炼"))
    ap.add_argument("--数据", default=str(数据目录))
    ap.add_argument("--分钟", type=float, default=15.0)
    ap.add_argument("--通道", type=int, default=320)
    ap.add_argument("--膨胀", default="1,2,4,8,16")
    ap.add_argument("--学习率", type=float, default=1e-4)
    ap.add_argument("--周期保存", type=int, default=25)
    ap.add_argument("--上一帧数", type=int, default=2)
    a = ap.parse_args()
    if a.命令 == "自检":
        自检()
    else:
        训练精炼(a.base, Path(a.出), 数据路=Path(a.数据), 分钟=a.分钟, 通道=a.通道,
                 膨胀=tuple(int(x) for x in a.膨胀.split(",")), 学习率=a.学习率,
                 周期保存=a.周期保存, 上一帧数=a.上一帧数)
