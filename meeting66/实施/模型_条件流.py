# -*- coding: utf-8 -*-
"""meeting66 / 模型_条件流.py —— 条件 Flow Matching（CFM）声学核心（非 Transformer、无 attention）。

对应 `06_实验_v5v6.md` §19.4 / `08_当前最佳与复现.md` §5 路线 (b)：
**换核心**为条件 VAE/flow 的**最小参考预实验**（最小参考 = 只保留让 CFM 成立的必要组件，
不引入 U-Net / 注意力 / Transformer / 预训练；与 v13 同数据、同评测协议，只换「核心」）。

──────────────────────────────── 最小参考实现（本件全部内容） ────────────────────────────────
  ① 编码器（与 v13 同构，只读复用 `模型_学说话` 的嵌入/帧特征/残差块）：
       音素(声母/韵母/调)嵌入 ⊕ 10 维帧级特征 ⊕ 前 K 帧 mel（K=4）
       → Linear → LayerNorm → Conv1D 残差堆（核 5、dilation 1/2/4/8/16、通道 320）→ 上下文 [B,T,C]
  ② 速度场（flow 的「核心」）：条件 Flow Matching（Lipman et al. 2022, Flow Matching for
       Generative Modeling；CFM 版：x_s=(1−s)x0+s·x1，目标 v=x1−x0）：
       v(x_s, s | 上下文) = 输出卷积( 头残差堆( 融合([上下文, x_s, 正弦时间嵌入(s)]) ) ) + 线性(x_s)
  ③ 训练：x1 = 真 mel（逐带标准化域），x0 ~ N(0,I)，s ~ U(0,1)，损 = 掩码 MSE(v̂, x1−x0)；
       优化器/预热/裁剪/周期保存与既有 v13 训练同配方（便于对照）。
  ④ 推理：Euler 积分 s:0→1（步数可调；**零起点** x0=0 为确定性主模式，噪声起点为对照），
       再套用与 v13 同款「迭代自条件 + 阻尼」：第 r 轮用第 r−1 轮的（阻尼）估计当 prev。
  ⑤ 红线：无自注意力主干/无序列注意力层/无第三方预训练权重/无第三方推理栈（C1）；
       生成路径只读本仓自训练 ckpt（C10）；同 ckpt 两趟推理 mel 逐位同（C13）；
       本件只新增 `meeting66/实施/`，不改既有件（C14）。

──────────────────────────────── 与 v13 的可比性 ────────────────────────────────
  数据 = `产物/数据_对齐asr`（167 条 / 2.7h，与 v13 同一份）；每音节 24 帧；
  评测 = S1–S12（历史 S1–S6 / 留出 S7–S12）× 同 Griffin-Lim × 同 ASR 两趟转写；
  选择 = S13–S18 独立验证集（`07_预注册_独立验证集.md`，只选不报）。

用法：
  python 模型_条件流.py 自检
  python 模型_条件流.py 训练 --数据 <数据目录> --出 <模型目录> --分钟 35
  python 模型_条件流.py 合成 "你好，世界" --ckpt <ckpt> --npz out.npy
"""
from __future__ import annotations

import argparse
import json
import math
import os
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

import 模型_学说话 as M  # noqa: E402  只读复用：数据契约 / 展开帧 / 残差块 / mel 统计

数据目录 = M66 / "产物" / "数据_对齐asr"     # 与 v13 同一条 2.7h 数据
模型目录 = M66 / "产物" / "模型_条件流"
种子 = 300
MEL带 = M.MEL带
帧特征数 = M.帧特征数

# GPU 确定性（C13：同 ckpt 两趟推理逐位同）
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:  # noqa: BLE001
    pass


# ════════════════════════════════════════════════════════════════════════
# 1. 时间嵌入（正弦；纯函数、无参数，确定性）
# ════════════════════════════════════════════════════════════════════════
def 时间嵌入(s: torch.Tensor, 维: int = 32, 尺度: float = 1000.0) -> torch.Tensor:
    """s: [B]（flow 时间，0..1）→ [B, 维]（sin/cos 各半）。"""
    if 维 % 2 == 1:
        维 += 1
    半 = 维 // 2
    频率 = torch.exp(-math.log(10000.0) * torch.arange(半, device=s.device, dtype=torch.float32)
                     / max(半 - 1, 1))
    角 = s.reshape(-1, 1).float() * 尺度 * 频率.reshape(1, -1)
    return torch.cat([torch.sin(角), torch.cos(角)], dim=-1)


# ════════════════════════════════════════════════════════════════════════
# 2. 条件流声学核心
# ════════════════════════════════════════════════════════════════════════
class 条件流声学(nn.Module):
    """音素 + 帧特征 + 前 K 帧 mel → 上下文；条件 flow 速度场 → mel 速度 [B,T,80]。

    没有注意力层；全部是 Linear / Conv1D / LayerNorm / GELU（C1 允许的从零训练网络）。
    """

    def __init__(self, 词表, 嵌入维=128, 通道=320, 卷积核=5,
                 膨胀=(1, 2, 4, 8, 16), 输出维=MEL带, 特征维=帧特征数,
                 上一帧数=4, 时间维=32, 头层=2):
        super().__init__()
        self.词表 = 词表
        self.上一帧数 = int(上一帧数)
        if int(时间维) % 2 == 1:      # 正弦时间嵌入是 sin/cos 各半，奇数会与融合层维度不符
            时间维 = int(时间维) + 1
        self.配置 = dict(嵌入维=嵌入维, 通道=通道, 卷积核=卷积核,
                        膨胀=list(膨胀), 输出维=输出维, 特征维=特征维,
                        上一帧数=self.上一帧数, 时间维=int(时间维),
                        头层=min(max(int(头层), 1), 3))
        self.声母嵌入 = nn.Embedding(len(词表["声母"]), 嵌入维)
        self.韵母嵌入 = nn.Embedding(len(词表["韵母"]), 嵌入维)
        self.调嵌入 = nn.Embedding(len(词表["调"]), 嵌入维)
        self.输入投影 = nn.Linear(3 * 嵌入维 + 特征维 + self.上一帧数 * 输出维, 通道)
        self.输入范数 = nn.LayerNorm(通道)
        self.块s = nn.ModuleList([M.残差块(通道, 卷积核, d) for d in 膨胀])
        self.末范数 = nn.LayerNorm(通道)
        # —— 速度场头（flow 的核心；轻量：融合 + 2 个残差块 + 1x1 输出 + x_s 直连） ——
        self.融合 = nn.Linear(通道 + 输出维 + 时间维, 通道)
        self.头块s = nn.ModuleList([M.残差块(通道, 3, d) for d in [1, 2, 4][:self.配置["头层"]]])
        self.头末范数 = nn.LayerNorm(通道)
        self.输出卷积 = nn.Conv1d(通道, 输出维, 1)
        self.线性 = nn.Linear(输出维, 输出维)   # x_s 直连（类似 v13 的输入旁路）

    def 编码(self, 声, 韵, 调, 特征, 上一=None):
        """→ 上下文 [B,T,C]（同一轮迭代内与 flow 时间 s 无关，可在采样中缓存）。"""
        e = torch.cat([self.声母嵌入(声), self.韵母嵌入(韵),
                       self.调嵌入(调), 特征], dim=-1)
        if self.上一帧数 > 0:
            if 上一 is None:
                上一 = torch.zeros(*(list(e.shape[:-1]) + [self.上一帧数 * self.配置["输出维"]]),
                                   device=e.device, dtype=e.dtype)
            e = torch.cat([e, 上一], dim=-1)
        x = self.输入范数(self.输入投影(e))
        h = x.transpose(1, 2)
        for 块 in self.块s:
            h = 块(h)
        return F.gelu(self.末范数(h.transpose(1, 2)))  # [B,T,C]

    def 速度(self, 上下文, x_s, s):
        """条件速度场：上下文 [B,T,C]，x_s [B,T,80]，s [B] → [B,T,80]。"""
        t = 时间嵌入(s, self.配置["时间维"]).unsqueeze(1).expand(-1, 上下文.shape[1], -1)
        h = self.融合(torch.cat([上下文, x_s, t], dim=-1)).transpose(1, 2)  # [B,C,T]
        for 块 in self.头块s:
            h = 块(h)
        h = F.gelu(self.头末范数(h.transpose(1, 2)))                      # [B,T,C]
        return (self.输出卷积(h.transpose(1, 2)).transpose(1, 2)
                + self.线性(x_s))  # [B,T,80]


def 参数数(模型):
    return int(sum(p.numel() for p in 模型.parameters()))


# ════════════════════════════════════════════════════════════════════════
# 3. CFM 目标 / 掩码损失
# ════════════════════════════════════════════════════════════════════════
def 掩码均方(预测, 目标, 掩码):
    w = 掩码.unsqueeze(-1)
    分母 = (w.sum() * 预测.shape[-1]).clamp_min(1e-6)
    return ((预测 - 目标) ** 2 * w).sum() / 分母


def 掩码L1(预测, 目标, 掩码):
    w = 掩码.unsqueeze(-1)
    分母 = (w.sum() * 预测.shape[-1]).clamp_min(1e-6)
    return ((预测 - 目标).abs() * w).sum() / 分母


# ════════════════════════════════════════════════════════════════════════
# 4. 保存 / 训练
# ════════════════════════════════════════════════════════════════════════
def _保存(path: Path, 模型, 词表, 均值, 标准差, 参, 训练信息):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in 模型.state_dict().items()},
                    词表=词表, 配置=模型.配置, 参数数=int(参),
                    mel均值=np.asarray(均值, np.float32),
                    mel标准差=np.asarray(标准差, np.float32),
                    训练信息=训练信息), tmp)
    os.replace(tmp, path)


def 训练条件流(数据路=数据目录, 出=模型目录, 分钟=35.0, 通道=320,
              膨胀=(1, 2, 4, 8, 16), 上一帧数=4, 学习率=2e-4, 周期保存=50,
              批样本=8, 最大帧=3000, 种子_=种子, 头层=2, 时间维=32, 损失="mse",
              验证窗口数=24, 自条件p=0.5, 自条件步数=4, 自条件阻尼=0.7):
    出 = Path(出)
    出.mkdir(parents=True, exist_ok=True)
    M._固定种子(种子_)
    数据集_ = M.数据集(数据路, 前N=None, 打印=False)
    设备 = M._设备()
    词表 = 数据集_.词表
    均值, 标准差 = 数据集_.均值, 数据集_.标准差
    配置 = dict(嵌入维=128, 通道=int(通道), 卷积核=5,
                膨胀=[int(x) for x in 膨胀], 输出维=MEL带, 特征维=帧特征数,
                上一帧数=int(上一帧数), 时间维=int(时间维), 头层=int(头层),
                范式="条件流match", 损失=损失, 批样本=int(批样本), 最大帧=int(最大帧),
                学习率=float(学习率), 周期保存=int(周期保存), 时长分钟=float(分钟),
                自条件p=float(自条件p), 自条件步数=int(自条件步数),
                自条件阻尼=float(自条件阻尼))
    if 损失 not in ("mse", "l1", "混合"):
        raise ValueError("损失须为 mse/l1/混合，实际 %r" % (损失,))
    模型 = 条件流声学(词表, 嵌入维=128, 通道=int(通道), 卷积核=5,
                     膨胀=tuple(int(x) for x in 膨胀),
                     输出维=MEL带, 特征维=帧特征数, 上一帧数=int(上一帧数),
                     时间维=int(时间维), 头层=int(头层)).to(设备)
    参 = 参数数(模型)
    优化器 = torch.optim.AdamW(模型.parameters(), lr=float(学习率), weight_decay=1e-4)
    rng = np.random.default_rng(种子_)

    def 造窗(ids):
        窗口s = []
        for sid in rng.permutation(ids):
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
        return [w for w in 窗口s if w is not None]

    # 固定验证窗口（仅记录 CFM 损失曲线，不用来选模型——选模型一律按推理 CER）
    验窗 = []
    for sid in 数据集_.验证ids:
        s = 数据集_.样本[sid]
        验窗.append(M.裁窗(s, 0, 最大帧))
    验窗 = [w for w in 验窗 if w is not None][:验证窗口数]

    开始 = time.time()
    总预算 = float(分钟 * 60.0)
    曲线 = dict(轮=[], 训练损失=[], 验证损失=[])
    日志 = open(出 / "训练日志.txt", "w", encoding="utf-8")

    def 打(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        日志.write(s + "\n")
        日志.flush()

    打("=" * 78)
    打("[条件流] 最小参考预实验 · 条件 Flow Matching（非 Transformer / 无 attention）")
    打("  开始 %s  torch=%s  cuda=%s  设备=%s"
       % (time.strftime("%Y-%m-%d %H:%M:%S"), torch.__version__, torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
    打("  数据=%s（训练 %d / 验证 %d 样本）" % (数据路, len(数据集_.训练ids), len(数据集_.验证ids)))
    打("  参数=%s（%.3fM）  预算=%.0fs  损失=%s  前K帧=%d"
       % (f"{参:,}", 参 / 1e6, 总预算, 损失, 上一帧数))
    打("  配置=%s" % json.dumps(配置, ensure_ascii=False))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    步 = 0
    轮 = 0
    最佳 = dict(val=float("inf"), 轮=-1)
    验证损 = None
    训练损 = None
    while (time.time() - 开始) < 总预算:
        轮 += 1
        模型.train()
        损和, 帧和 = 0.0, 0
        for 批 in M.起批(造窗(数据集_.训练ids), 批样本, 最大帧):
            b = M.拼批(批, 均值, 标准差)
            b = {k: v.to(设备, non_blocking=True) for k, v in b.items()}
            真 = b["mel"]
            上帧 = M._上帧张量(真, 上一帧数)
            for g in 优化器.param_groups:      # 线性预热（同 v13 配方：300 步）
                if 步 < 300:
                    g["lr"] = float(学习率) * (步 + 1) / 300.0
            # 自条件（减轻 teacher-prev / 自产-prev 的曝光偏差；与 v13 的「自条件p」同义）：
            # 以 p 的概率先用「prev=None + 少步数 flow」滚出一个粗估计，
            # 再按 阻尼:1−阻尼 与真值混合当 prev（默认 0.7 粗 / 0.3 真），主损失只走第二遍。
            if 自条件p > 0 and int(上一帧数) > 0 and float(torch.rand(1).item()) < 自条件p:
                with torch.no_grad():
                    C0 = 模型.编码(b["声"], b["韵"], b["调"], b["特征"], None)
                    T = int(C0.shape[1])
                    粗 = torch.zeros((1, T, 模型.配置["输出维"]), device=设备)
                    for _k in range(max(int(自条件步数), 1)):
                        s_k = torch.full((1,), _k / float(max(int(自条件步数), 1)), device=设备)
                        粗 = 粗 + 模型.速度(C0, 粗, s_k) / float(max(int(自条件步数), 1))
                if 真.shape[1] == 粗.shape[1]:
                    上帧 = M._上帧张量(float(自条件阻尼) * 粗 + (1.0 - float(自条件阻尼)) * 真,
                                     模型.上一帧数)
            x0 = torch.randn_like(真)
            s = float(torch.rand(1).item())
            x_s = (1.0 - s) * x0 + s * 真
            目标 = 真 - x0
            上下文 = 模型.编码(b["声"], b["韵"], b["调"], b["特征"], 上帧)
            预测 = 模型.速度(上下文, x_s, torch.full((真.shape[0],), s, device=设备))
            if 损失 == "l1":
                损 = 掩码L1(预测, 目标, b["掩码"])
            elif 损失 == "混合":
                损 = 0.5 * 掩码均方(预测, 目标, b["掩码"]) + 0.5 * 掩码L1(预测, 目标, b["掩码"])
            else:
                损 = 掩码均方(预测, 目标, b["掩码"])
            优化器.zero_grad(set_to_none=True)
            损.backward()
            torch.nn.utils.clip_grad_norm_(模型.parameters(), 1.0)
            优化器.step()
            步 += 1
            n = int(b["掩码"].sum().item())
            损和 += float(损.item()) * n
            帧和 += n
        训练损 = 损和 / max(帧和, 1)
        # 验证：固定窗口 + 固定 x0/s（只作曲线，不作选择指标）
        模型.eval()
        v和, v帧 = 0.0, 0
        with torch.no_grad():
            g_ = torch.Generator(device="cpu").manual_seed(种子_)
            for 批 in M.起批(验窗, 批样本, 最大帧):
                b = M.拼批(批, 均值, 标准差)
                b = {k: v.to(设备) for k, v in b.items()}
                真 = b["mel"]
                上帧 = M._上帧张量(真, 上一帧数)
                x0 = torch.randn(真.shape, generator=g_).to(设备)
                s = 0.5
                x_s = (1.0 - s) * x0 + s * 真
                上下文 = 模型.编码(b["声"], b["韵"], b["调"], b["特征"], 上帧)
                预测 = 模型.速度(上下文, x_s, torch.full((真.shape[0],), s, device=设备))
                v = 掩码均方(预测, 真 - x0, b["掩码"])
                n = int(b["掩码"].sum().item())
                v和 += float(v.item()) * n
                v帧 += n
        验证损 = v和 / max(v帧, 1)
        曲线["轮"].append(轮)
        曲线["训练损失"].append(round(训练损, 6))
        曲线["验证损失"].append(round(验证损, 6))
        if 轮 <= 3 or 轮 % 10 == 0:
            打("  轮%4d  训练 %.5f  验证@0.5 %.5f  已用 %.0fs"
               % (轮, 训练损, 验证损, time.time() - 开始))
        if 周期保存 and 轮 % int(周期保存) == 0:
            _保存(出 / ("ckpt_e%04d.pt" % 轮), 模型, 词表, 均值, 标准差, 参,
                  dict(轮=轮, 训练损失=训练损, 验证损失=验证损,
                       参数数=参, 范式="条件流match", 训练配置=配置, 完成=False))
    模型.eval()
    _保存(出 / "ckpt.pt", 模型, 词表, 均值, 标准差, 参,
          dict(轮=轮, 训练损失=训练损, 验证损失=验证损, 参数数=参,
               范式="条件流match", 训练配置=配置,
               训练秒=round(time.time() - 开始, 1), 完成=True))
    曲线["配置"] = 配置
    曲线["参数数"] = 参
    曲线["轮数"] = 轮
    曲线["训练秒"] = round(time.time() - 开始, 1)
    (出 / "训练曲线.json").write_text(json.dumps(曲线, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    打("[条件流] 结束：%d 轮 → %s" % (轮, 出 / "ckpt.pt"))
    if torch.cuda.is_available():
        打("  显存峰值 %.0f MB" % (torch.cuda.max_memory_allocated() / 1e6))
    日志.close()
    return dict(参数数=参, 轮=轮, 训练秒=round(time.time() - 开始, 1))


# ════════════════════════════════════════════════════════════════════════
# 5. 加载 / 推理
# ════════════════════════════════════════════════════════════════════════
def 加载条件流(ckpt路径, 设备=None):
    设备 = 设备 or M._设备()
    ck = torch.load(str(ckpt路径), map_location="cpu", weights_only=False)
    模型 = 条件流声学(ck["词表"], **ck["配置"])
    模型.load_state_dict(ck["state_dict"])
    模型.词表 = ck["词表"]
    模型.mel均值 = np.asarray(ck["mel均值"], np.float32)
    模型.mel标准差 = np.asarray(ck["mel标准差"], np.float32)
    模型.训练信息 = ck.get("训练信息", {})
    模型.参数数 = ck.get("参数数")
    模型.eval()
    模型.to(设备)
    return 模型


def 合成mel_条件流(模型, 文本: str, 每音节帧数: int = 24, 迭代: int = 4,
                  阻尼: float = 0.5, 步数: int = 10, 采样: str = "零起点",
                  终值: str = "末轮", 种子_: int = 种子):
    """文本 → 条件 flow 采样 → 未标准化 log-mel [T,80]。

    采样: 零起点（x0=0，确定性；主模式）| 噪声（x0~N(0,I)，固定种子）
    终值: 末轮（最后一轮输出，同 v13）| 平滑（阻尼后的估计，更稳）
    迭代: 自条件轮数（与 v13 的 3–7 轮同义）；阻尼: 平滑系数（1.0=不阻尼）
    """
    每 = max(int(每音节帧数), 4)
    tokens, 有效, 跳过 = M.文本到token(模型.词表, 文本)
    if len(tokens) == 0:
        raise ValueError("文本无可合成音节：%r 跳过=%s" % (文本, 跳过[:5]))
    帧数 = np.full(len(tokens), 每, dtype=np.int64)
    声, 韵, 调, 特征 = M.展开帧(tokens, 帧数)
    设备 = next(模型.parameters()).device

    def 张(a):
        return torch.from_numpy(a).unsqueeze(0).to(设备)

    声t, 韵t, 调t, 特t = 张(声), 张(韵), 张(调), 张(特征)
    K = int(getattr(模型, "上一帧数", 0))
    迭代 = max(int(迭代), 1)
    步数 = max(int(步数), 1)
    噪 = (采样 == "噪声")
    生成器 = torch.Generator(device=设备)
    生成器.manual_seed(int(种子_))
    上帧, 平滑, 出 = None, None, None
    with torch.no_grad():
        for _ in range(迭代):
            上下文 = 模型.编码(声t, 韵t, 调t, 特t, 上帧)
            T = int(上下文.shape[1])
            if 噪:
                x = torch.randn((1, T, 模型.配置["输出维"]), generator=生成器, device=设备)
            else:
                x = torch.zeros((1, T, 模型.配置["输出维"]), device=设备)
            for k in range(步数):
                s = torch.full((1,), k / float(步数), device=设备)
                x = x + 模型.速度(上下文, x, s) / float(步数)
            出 = x
            平滑 = 出 if 平滑 is None else (float(阻尼) * 出 + (1.0 - float(阻尼)) * 平滑)
            if K > 0:
                上帧 = M._上帧张量(平滑, K)
    sel = 平滑 if 终值 == "平滑" else 出
    mel = sel[0].detach().cpu().numpy().astype(np.float32)
    m = np.asarray(模型.mel均值, np.float32).reshape(-1)
    sd = np.asarray(模型.mel标准差, np.float32).reshape(-1)
    return (mel * (sd[0] if sd.size == 1 else sd.reshape(1, -1)) +
            (m[0] if m.size == 1 else m.reshape(1, -1))).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════
# 6. 自检 / 红线静态审计
# ════════════════════════════════════════════════════════════════════════
def 架构审计(文件: Path = None) -> dict:
    文 = Path(文件 or __file__).read_text(encoding="utf-8")
    禁 = ["Multi" + "head" + "Att" + "ention", "scaled_dot_" + "product_" + "att" + "ention",
          "Trans" + "former" + "Encoder", "Trans" + "former" + "Decoder",
          "from_" + "pretrained", "torch." + "hub", "import " + "trans" + "formers",
          "nn." + "LSTM"]
    命中 = [p for p in 禁 if p in 文]
    模型 = 条件流声学(M.默认词表())
    模块 = [type(m).__name__ for m in 模型.modules()]
    可疑 = [n for n in 模块 if ("Att" + "ention") in n or ("Multi" + "head") in n
            or ("Trans" + "former") in n]
    return dict(命中=命中, 命中数=len(命中), 可疑模块=可疑, 参数数=参数数(模型))


def 自检(ckpt=None):
    报 = dict(检查=[], 异常=[])
    def 记(名, ok, 值=None):
        报["检查"].append(dict(项=名, ok=bool(ok), 值=值))
        if not ok:
            报["异常"].append(名)

    审 = 架构审计()
    记("红线静态审计 0 命中", 审["命中数"] == 0, 审["命中"])
    记("无注意力/预训练子模块", not 审["可疑模块"], 审["可疑模块"])
    记("参数量 > 1M", 审["参数数"] > 1_000_000, 审["参数数"])

    # 形状自检（随机小模型，不依赖训练）
    M._固定种子(种子)
    小 = 条件流声学(M.默认词表(), 通道=32, 膨胀=(1, 2, 4, 8), 头层=1)
    小.mel均值 = np.zeros(MEL带, np.float32)
    小.mel标准差 = np.ones(MEL带, np.float32)
    小.eval()
    声 = torch.zeros(1, 24, dtype=torch.long)
    韵 = torch.zeros(1, 24, dtype=torch.long)
    调 = torch.zeros(1, 24, dtype=torch.long)
    特 = torch.zeros(1, 24, 帧特征数)
    C = 小.编码(声, 韵, 调, 特)
    记("编码形状 [1,24,C]", tuple(C.shape) == (1, 24, 32), list(C.shape))
    v = 小.速度(C, torch.zeros(1, 24, MEL带), torch.tensor([0.5]))
    记("速度形状 [1,24,80]", tuple(v.shape) == (1, 24, MEL带), list(v.shape))
    记("速度非 NaN", bool(torch.isfinite(v).all()), float(v.detach().abs().mean()))

    # CFM 恒等性：x_s=(1-s)x0+s·x1 在 s=0/1 处还原端点
    x0, x1 = torch.randn(4, 7), torch.randn(4, 7)
    for s in (0.0, 1.0):
        xs = (1 - s) * x0 + s * x1
        记("插值端点 s=%.0f" % s, bool(torch.allclose(xs, x0 if s == 0 else x1)), s)

    # 推理两趟逐位同（C13）
    m1 = 合成mel_条件流(小, "你好，世界", 每音节帧数=24, 迭代=2, 阻尼=0.5, 步数=2)
    m2 = 合成mel_条件流(小, "你好，世界", 每音节帧数=24, 迭代=2, 阻尼=0.5, 步数=2)
    记("两趟推理 mel 逐位同（零起点）", bool(np.array_equal(m1, m2)), list(m1.shape))
    n1 = 合成mel_条件流(小, "你好，世界", 每音节帧数=24, 迭代=2, 阻尼=0.5, 步数=2, 采样="噪声")
    n2 = 合成mel_条件流(小, "你好，世界", 每音节帧数=24, 迭代=2, 阻尼=0.5, 步数=2, 采样="噪声")
    记("两趟推理 mel 逐位同（噪声/固定种子）", bool(np.array_equal(n1, n2)), list(n1.shape))
    记("mel 形状 [4×24,80]", tuple(m1.shape) == (4 * 24, MEL带), list(m1.shape))

    if ckpt:
        try:
            模型 = 加载条件流(ckpt)
            a = 合成mel_条件流(模型, "今天的风很轻，适合慢慢地说一会儿话。", 迭代=2, 步数=2)
            b = 合成mel_条件流(模型, "今天的风很轻，适合慢慢地说一会儿话。", 迭代=2, 步数=2)
            记("ckpt 两趟推理逐位同", bool(np.array_equal(a, b)), list(a.shape))
        except Exception as e:  # noqa: BLE001
            记("ckpt 两趟推理逐位同", False, "%s: %s" % (type(e).__name__, e))
    报["PASS"] = not 报["异常"]
    print(json.dumps(报, ensure_ascii=False, indent=1))
    return 报


# ════════════════════════════════════════════════════════════════════════
# 7. CLI
# ════════════════════════════════════════════════════════════════════════
def _主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("命令", nargs="?", default="自检", choices=["自检", "训练", "合成"])
    ap.add_argument("--数据", default=str(数据目录))
    ap.add_argument("--出", default=str(模型目录))
    ap.add_argument("--分钟", type=float, default=35.0)
    ap.add_argument("--通道", type=int, default=320)
    ap.add_argument("--膨胀", default="1,2,4,8,16")
    ap.add_argument("--上一帧数", type=int, default=4)
    ap.add_argument("--学习率", type=float, default=2e-4)
    ap.add_argument("--周期保存", type=int, default=50)
    ap.add_argument("--批样本", type=int, default=8)
    ap.add_argument("--最大帧", type=int, default=3000)
    ap.add_argument("--头层", type=int, default=2)
    ap.add_argument("--损失", default="mse", choices=["mse", "l1", "混合"])
    ap.add_argument("--自条件p", type=float, default=0.5)
    ap.add_argument("--自条件步数", type=int, default=4)
    ap.add_argument("--自条件阻尼", type=float, default=0.7)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--文本", default=None)
    ap.add_argument("--npz", default=None)
    ap.add_argument("--迭代", type=int, default=4)
    ap.add_argument("--阻尼", type=float, default=0.5)
    ap.add_argument("--步数", type=int, default=10)
    ap.add_argument("--采样", default="零起点", choices=["零起点", "噪声"])
    a = ap.parse_args()
    if a.命令 == "自检":
        return 0 if 自检(a.ckpt)["PASS"] else 1
    if a.命令 == "训练":
        print(json.dumps(训练条件流(数据路=Path(a.数据), 出=Path(a.出), 分钟=a.分钟,
                                  通道=a.通道, 膨胀=tuple(int(x) for x in a.膨胀.split(",")),
                                  上一帧数=a.上一帧数, 学习率=a.学习率,
                                  周期保存=a.周期保存, 批样本=a.批样本, 最大帧=a.最大帧,
                                  头层=a.头层, 损失=a.损失, 自条件p=a.自条件p,
                                  自条件步数=a.自条件步数, 自条件阻尼=a.自条件阻尼),
                         ensure_ascii=False, indent=1))
        return 0
    if a.命令 == "合成":
        if not a.ckpt or not a.文本:
            print("合成需要 --ckpt 与 --文本")
            return 2
        模型 = 加载条件流(a.ckpt)
        mel = 合成mel_条件流(模型, a.文本, 迭代=a.迭代, 阻尼=a.阻尼, 步数=a.步数, 采样=a.采样)
        if a.npz:
            np.save(a.npz, mel)
        print("mel", mel.shape, "均值 %.4f 标准差 %.4f" % (float(mel.mean()), float(mel.std())))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(_主())
