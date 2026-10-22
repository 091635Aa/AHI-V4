# -*- coding: utf-8 -*-
"""meeting66/实施/模型_学说话.py —— T-E「学说话」声学模型：音节音素 → log-mel。

红线（`meeting66/00_裁决`，所有者 2026-10-20 口头授权）：
  · 本模型是**从零训练**的卷积网络：embedding（声母/韵母/调）+ 帧级特征 +
    Conv1D 残差堆（kernel 5、通道 256、dilation 1/2/4/8、LayerNorm + GELU）+ 1x1 conv → 80 维；
  · **无自注意力主干、无序列注意力层、无第三方预训练权重、无第三方推理栈**（C1）；
  · 生成路径只读本仓自训练 ckpt（C10）；torch 只用于本模型；
  · C13：同一 ckpt 两趟推理 mel 逐位同（`np.array_equal`）；
  · C14：只新增 `meeting66/`；既有件只读（复用 `音韵_v5.解析音节` 做 token 化）。

数据契约（T-D `数据_SpeakerA.py` 产出，见 task-8）：
  meeting66/产物/数据/meta.json      —— 词表 / mel 统计 / 样本清单 / train-val 划分
  meeting66/产物/数据/样本/NNNN.npz  —— mel [T,80]、音节 token [N,3]、每音节帧数 [N]

用法：
  python 模型_学说话.py 自检                          # 假数据端到端自检（不依赖真实数据）
  python 模型_学说话.py 数据                          # 只读诊断：数据契约 / 词表 / 统计
  python 模型_学说话.py 训练 --前N 40 --分钟 40        # 训练（前 40 样本，GPU 时间 ≤40min）
  python 模型_学说话.py 合成 "你好，世界" --npz out.npy # 读 ckpt 合成 mel
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M66 = HERE.parent
AHI4 = M66.parent
数据目录 = M66 / "产物" / "数据"
模型目录 = M66 / "产物" / "模型"
M61实施 = AHI4 / "meeting61" / "实施"
if str(M61实施) not in sys.path:
    sys.path.insert(0, str(M61实施))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import 音韵_v5 as 音韵  # noqa: E402  只读复用：文本 → 声母/韵母/调

种子 = 300
MEL带 = 80
帧特征数 = 10
帧率 = 100  # 10ms 跳
边界掩码帧 = 40  # 拼接样本边界处不参与 loss（卷积感受野 30 帧）

# GPU 确定性（C13：推理两趟逐位同）
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:  # noqa: BLE001
    pass


# ════════════════════════════════════════════════════════════════════════
# 0. 小工具
# ════════════════════════════════════════════════════════════════════════
def _挑(d, 候选):
    """dict 里按候选键名顺序挑第一个存在的键（数据契约可能有多套命名）。"""
    if not isinstance(d, dict):
        return None
    for k in 候选:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _as_list(x, 数值=False):
    """列表 / dict(token→id) → 有序列表。"""
    if x is None:
        return None
    if isinstance(x, dict):
        对 = sorted(x.items(), key=lambda kv: (float(kv[1]), str(kv[0])))
        return [k for k, _ in 对]
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _标量或数组(x, 缺省):
    if x is None:
        return np.asarray(缺省, dtype=np.float32)
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return np.asarray(缺省, dtype=np.float32)
    return a


def _设备():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _固定种子(s=种子):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _json值(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


# ════════════════════════════════════════════════════════════════════════
# 1. 词表（声母 / 韵母 / 调）
# ════════════════════════════════════════════════════════════════════════
def 默认词表():
    """从 `音韵_v5` 推导默认词表（自检/兜底用；正式训练以 meta.json 为准）。"""
    声 = [""] + [s for s in 音韵.声母序]
    韵 = list(音韵.韵母表.keys())
    调 = [1, 2, 3, 4, 5]
    return {"声母": 声, "韵母": 韵, "调": 调}


def 读词表(meta: dict):
    """meta.json → 词表；兼容 list 与 {token:id} 两种表示。"""
    源 = _挑(meta, ["词表", "vocab", "音素词表", "vocabulary", "token词表"]) or meta
    声 = _挑(源, ["声母", "声母表", "声母词表", "initials", "shengmu"])
    韵 = _挑(源, ["韵母", "韵母表", "韵母词表", "finals", "yunmu"])
    调 = _挑(源, ["调", "声调", "调表", "声调表", "tones", "tone"])
    声, 韵, 调 = _as_list(声), _as_list(韵), _as_list(调)
    if not 声 or not 韵 or not 调:
        回退 = 默认词表()
        if not 声:
            声 = 回退["声母"]
        if not 韵:
            韵 = 回退["韵母"]
        if not 调:
            调 = 回退["调"]
    # 词表**顺序即索引**：原样保留（数据 token id 以此为准），只补 ∅ 的写法
    声 = [str(s) for s in 声]
    if "" not in 声 and "∅" not in 声:
        声 = [""] + 声
    return {"声母": 声, "韵母": [str(s) for s in 韵],
            "调": [int(v) for v in 调]}


def 建索引(词表):
    声_i = {}
    for i, s in enumerate(词表["声母"]):
        声_i[str(s)] = i
    # ∅ / 空串互为别名（索引以词表里真实出现者为准）
    声_i.setdefault("", 声_i.get("∅", 0))
    声_i.setdefault("∅", 声_i.get("", 0))
    韵_i = {str(v): i for i, v in enumerate(词表["韵母"])}
    调_i = {int(v): i for i, v in enumerate(词表["调"])}
    return 声_i, 韵_i, 调_i


# ════════════════════════════════════════════════════════════════════════
# 2. 音节 token → 帧级输入（每音节按帧数展开：声母段 25% / 韵母 75%）
# ════════════════════════════════════════════════════════════════════════
def 声母帧数分配(帧数: np.ndarray) -> np.ndarray:
    """每音节的声母帧数（默认 25%，至少 2 帧，不超过该音节帧数）。"""
    帧数 = np.asarray(帧数, dtype=np.int64)
    if 帧数.size == 0:
        return 帧数
    b = np.maximum(2, np.rint(帧数.astype(np.float64) * 0.25).astype(np.int64))
    return np.minimum(b, np.maximum(帧数, 1))


def 展开帧(tokens: np.ndarray, 帧数: np.ndarray):
    """音节级 → 帧级。

    tokens: [N,3] int64 (声母_id, 韵母_id, 调_id)
    帧数:   [N]   int64
    → (声[T], 韵[T], 调[T], 特征[T,10])  特征=帧内相对位置/段内位置/时长/句位
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    帧数 = np.asarray(帧数, dtype=np.int64)
    N = int(len(帧数))
    if N == 0:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64),
                np.zeros(0, np.int64), np.zeros((0, 帧特征数), np.float32))
    T = int(帧数.sum())
    音 = np.repeat(np.arange(N, dtype=np.int64), 帧数)
    音内 = np.zeros(T, np.float32)
    段内 = np.zeros(T, np.float32)
    是声 = np.zeros(T, np.float32)
    声占比 = np.zeros(T, np.float32)
    界 = 声母帧数分配(帧数)
    起 = 0
    for j in range(N):
        n = int(帧数[j])
        if n <= 0:
            continue
        b = int(min(max(int(界[j]), 1), n))
        p = np.arange(n, dtype=np.float32)
        音内[起:起 + n] = p / float(n)
        是声[起:起 + n] = (p < b).astype(np.float32)
        段内[起:起 + n] = np.where(
            p < b,
            (p + 0.5) / float(max(b, 1)),
            (p - b + 0.5) / float(max(n - b, 1)))
        声占比[起:起 + n] = float(b) / float(n)
        起 += n
    总帧 = float(max(T - 1, 1))
    音节相对 = 音.astype(np.float32) / float(max(N - 1, 1))
    音节帧数 = 帧数.astype(np.float32)
    特征 = np.stack([
        音内,                                                     # 0 音内位置
        段内,                                                     # 1 段内位置（声/韵各 0..1）
        是声,                                                     # 2 是否声母段
        声占比,                                                   # 3 声母段占比
        np.clip(np.log1p(音节帧数)[音] / np.log1p(50.0), 0.0, 2.0),  # 4 音节时长（归一）
        np.arange(T, dtype=np.float32) / 总帧,                    # 5 全局位置
        音节相对,                                                 # 6 音节相对位置
        (音 == 0).astype(np.float32),                             # 7 句首
        (音 == N - 1).astype(np.float32),                         # 8 句末
        np.minimum(音内, 1.0 - 音内) * 2.0,                       # 9 离音节边界
    ], axis=1).astype(np.float32)
    return tokens[音, 0], tokens[音, 1], tokens[音, 2], 特征


# ════════════════════════════════════════════════════════════════════════
# 3. 模型：embedding + Conv1D 残差堆 + 1x1（全部非注意力）
# ════════════════════════════════════════════════════════════════════════
class 残差块(nn.Module):
    """Conv1D(dilation) → LayerNorm → GELU → 残差。"""

    def __init__(self, 通道: int, 卷积核: int = 5, 膨胀: int = 1):
        super().__init__()
        self.卷积 = nn.Conv1d(通道, 通道, 卷积核,
                            padding=(卷积核 // 2) * 膨胀, dilation=膨胀)
        self.范数 = nn.LayerNorm(通道)
        self.激活 = nn.GELU()

    def forward(self, x):  # x: [B,C,T]
        y = self.卷积(x)
        y = self.激活(self.范数(y.transpose(1, 2)).transpose(1, 2))
        return x + y


class 说话模型(nn.Module):
    """音节音素 → 帧级 log-mel [B,T,80]。无注意力层、无预训练。

    `上一帧数 > 0` 时（v5 迭代自条件）：输入额外拼接前 K 帧（训练用真值 teacher forcing，
    推理用自身上一轮输出迭代 refine）——不引入 attention/Transformer，仍纯 Conv。
    """

    def __init__(self, 词表, 嵌入维=128, 通道=256, 卷积核=5,
                 膨胀=(1, 2, 4, 8), 输出维=MEL带, 特征维=帧特征数, 上一帧数=0):
        super().__init__()
        self.词表 = 词表
        self.上一帧数 = int(上一帧数)
        self.配置 = dict(嵌入维=嵌入维, 通道=通道, 卷积核=卷积核,
                       膨胀=list(膨胀), 输出维=输出维, 特征维=特征维,
                       上一帧数=self.上一帧数)
        self.声母嵌入 = nn.Embedding(len(词表["声母"]), 嵌入维)
        self.韵母嵌入 = nn.Embedding(len(词表["韵母"]), 嵌入维)
        self.调嵌入 = nn.Embedding(len(词表["调"]), 嵌入维)
        self.输入投影 = nn.Linear(3 * 嵌入维 + 特征维 + self.上一帧数 * 输出维, 通道)
        self.输入范数 = nn.LayerNorm(通道)
        self.块s = nn.ModuleList([残差块(通道, 卷积核, d) for d in 膨胀])
        self.末范数 = nn.LayerNorm(通道)
        self.输出卷积 = nn.Conv1d(通道, 输出维, 1)
        self.旁路卷积 = nn.Conv1d(通道, 输出维, 1)  # 输入直连（1x1），加速收敛

    def forward(self, 声, 韵, 调, 特征, 上一=None):
        # 声/韵/调: [B,T] int64；特征: [B,T,F]；上一: [B,T,K*80] 或 None
        e = torch.cat([self.声母嵌入(声), self.韵母嵌入(韵),
                       self.调嵌入(调), 特征], dim=-1)
        if self.上一帧数 > 0:
            if 上一 is None:
                上一 = torch.zeros(*(list(e.shape[:-1]) + [self.上一帧数 * self.配置["输出维"]]),
                                   device=e.device, dtype=e.dtype)
            e = torch.cat([e, 上一], dim=-1)
        x = self.输入范数(self.输入投影(e))          # [B,T,C]
        h = x.transpose(1, 2)                        # [B,C,T]
        for 块 in self.块s:
            h = 块(h)
        h = self.末范数(h.transpose(1, 2)).transpose(1, 2)
        h = F.gelu(h)
        y = self.输出卷积(h) + self.旁路卷积(x.transpose(1, 2))
        return y.transpose(1, 2)                     # [B,T,80]


def 参数数(模型):
    return int(sum(p.numel() for p in 模型.parameters()))


# ════════════════════════════════════════════════════════════════════════
# 4. 数据读取（兼容 T-D 数据契约的多套键名）
# ════════════════════════════════════════════════════════════════════════
_MEL键 = ["mel", "mels", "logmel", "log_mel", "mel谱", "logmel谱", "log-mel", "melsp"]
_TOKEN键 = ["tokens", "音节token", "音节tokens", "token", "token序列", "音素token", "syllable_tokens"]
_帧数键 = ["帧数", "音节帧数", "每音节帧数", "frames", "syl_frames", "n_frames", "帧数s"]
清单键 = ["样本清单", "样本", "samples", "清单", "文件清单", "sample_list", "文件"]
训练键 = ["训练", "训练集", "train", "train_ids", "训练样本", "训练清单"]
验证键 = ["验证", "验证集", "val", "valid", "val_ids", "验证样本", "验证清单"]
划分键 = ["划分", "split", "数据划分", "train_val", "trainval"]


def 读mel统计(meta: dict):
    """→ (均值, 标准差)：可为标量或 [80] 数组。"""
    源 = _挑(meta, ["mel统计", "mel_stats", "频谱统计", "统计", "mel统计_全局"]) or meta
    均值 = _挑(源, ["mel均值", "mel_mean", "均值", "mean", "mel均值_全局", "logmel均值"])
    if isinstance(均值, dict):
        均值 = _挑(均值, ["全局", "global", "值", "value"])
    方差 = _挑(源, ["mel方差", "mel_var", "方差", "var"])
    标准差 = _挑(源, ["mel标准差", "mel_std", "标准差", "std", "mel_std_global", "logmel标准差"])
    if isinstance(标准差, dict):
        标准差 = _挑(标准差, ["全局", "global", "值", "value"])
    if isinstance(方差, dict):
        方差 = _挑(方差, ["全局", "global", "值", "value"])
    if 标准差 is None and 方差 is not None:
        标准差 = np.sqrt(np.maximum(np.asarray(方差, dtype=np.float32), 0.0) + 1e-10)
    m = _标量或数组(均值, [0.0])
    s = _标量或数组(标准差, [1.0])
    s = np.maximum(s, 1e-4)
    return m.astype(np.float32), s.astype(np.float32)


def _样本路径归一(项, 序, 数据目录: Path):
    """样本清单条目 → (id, 绝对路径；可能不存在)。"""
    if isinstance(项, dict):
        p = _挑(项, ["路径", "path", "文件", "npz", "文件路径", "rel", "相对路径"])
        if p is not None:
            p = Path(str(p))
            return p.stem, (p if p.is_absolute() else 数据目录 / p)
        i = _挑(项, ["id", "编号", "名称", "name", "stem", "样本号"])
        if i is not None:
            return str(i), 数据目录 / "样本" / f"{str(i)}.npz"
        return f"{序:04d}", 数据目录 / "样本" / f"{序:04d}.npz"
    if isinstance(项, str):
        p = Path(项)
        return p.stem, (p if p.is_absolute() else 数据目录 / p)
    if isinstance(项, (int, np.integer)):
        return f"{int(项):04d}", 数据目录 / "样本" / f"{int(项):04d}.npz"
    raise TypeError(f"无法识别的样本清单条目：{type(项)} {项!r}")


def _路径兜底(path: Path, 数据目录: Path):
    if path.exists():
        return path
    候选 = [数据目录 / "样本" / (path.stem + ".npz"),
            数据目录 / (path.stem + ".npz"),
            数据目录 / path]
    候选 += sorted((数据目录 / "样本").glob(f"{path.stem}*.npz"))
    for c in 候选:
        if c.exists():
            return c
    return path


def 读npz样本(path: Path, 词表):
    包 = np.load(path, allow_pickle=False)
    键s = set(包.files)
    mel键 = next((k for k in _MEL键 if k in 键s), None)
    if mel键 is None:
        raise KeyError(f"{path.name}: 找不到 mel 键，现有键={sorted(键s)}")
    mel = np.asarray(包[mel键], dtype=np.float32)
    if mel.ndim != 2:
        raise ValueError(f"{path.name}: mel 维度应为 2，实际 {mel.shape}")
    if mel.shape[0] == MEL带 and mel.shape[1] != MEL带:
        mel = mel.T  # 兼容 [80,T]
    音节token = next((k for k in _TOKEN键 if k in 键s), None)
    if 音节token is not None:
        tokens = np.asarray(包[音节token], dtype=np.int64)
        if tokens.ndim == 1 and tokens.size % 3 == 0:
            tokens = tokens.reshape(-1, 3)
    else:
        s键 = next((k for k in ["声母id", "声母", "声"] if k in 键s), None)
        y键 = next((k for k in ["韵母id", "韵母", "韵"] if k in 键s), None)
        t键 = next((k for k in ["调id", "调", "声调"] if k in 键s), None)
        if s键 and y键 and t键:
            tokens = np.stack([np.asarray(包[s键], dtype=np.int64),
                               np.asarray(包[y键], dtype=np.int64),
                               np.asarray(包[t键], dtype=np.int64)], axis=1)
        else:
            raise KeyError(f"{path.name}: 找不到音节 token，现有键={sorted(键s)}")
    帧数键名 = next((k for k in _帧数键 if k in 键s), None)
    if 帧数键名 is not None:
        帧数 = np.asarray(包[帧数键名], dtype=np.int64).reshape(-1)
    else:
        帧数 = None
    if tokens.ndim != 2 or tokens.shape[1] != 3:
        raise ValueError(f"{path.name}: tokens 形状应为 [N,3]，实际 {tokens.shape}")
    N = int(tokens.shape[0])
    T = int(mel.shape[0])
    if 帧数 is None or len(帧数) != N:
        # 均分兜底（末音节吃余数）
        if N > 0:
            基 = T // N
            帧数 = np.full(N, 基, dtype=np.int64)
            帧数[-1] += T - int(帧数.sum())
        else:
            帧数 = np.zeros(0, np.int64)
    帧数 = np.maximum(帧数, 1).astype(np.int64)
    if int(帧数.sum()) != T:
        # 与 mel 长度对齐：多余/不足全记到末音节
        帧数[-1] = max(int(帧数[-1]) + (T - int(帧数.sum())), 1)
        if int(帧数.sum()) != T:
            raise ValueError(f"{path.name}: 帧数总和 {int(帧数.sum())} != mel 帧数 {T}")
    # token 越界检查
    声_i, 韵_i, 调_i = 建索引(词表)
    if N and (tokens[:, 0].max() >= len(词表["声母"]) or tokens[:, 1].max() >= len(词表["韵母"])
              or tokens[:, 2].max() >= len(词表["调"]) or tokens[:, 2].min() < 0):
        raise ValueError(f"{path.name}: token 越界，词表 声={len(词表['声母'])} "
                         f"韵={len(词表['韵母'])} 调={len(词表['调'])}")
    return dict(id=path.stem, mel=mel, tokens=tokens, 帧数=帧数, 路径=str(path))


def _id归一(项, 全部ids, 索引表=None):
    """train/val 条目（int 索引 / 路径 / id）→ id 字符串。"""
    if isinstance(项, dict):
        项 = _挑(项, ["id", "编号", "路径", "path", "文件"]) or 项
    if isinstance(项, (int, np.integer)):
        i = int(项)
        if 索引表 is not None and 0 <= i < len(索引表):
            return 索引表[i]
        return f"{i:04d}"
    s = Path(str(项)).stem
    if s in 全部ids:
        return s
    try:
        i = int(s)
    except ValueError:
        return s
    return 索引表[i] if (索引表 is not None and 0 <= i < len(索引表)) else s


class 数据集:
    """读 meta.json + 样本/*.npz；兼容多套键名；全部 mel 常驻内存（40 样本约 80MB）。"""

    def __init__(self, 目录: Path = 数据目录, 前N=None, 打印=True):
        self.目录 = Path(目录)
        meta_path = self.目录 / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"数据未就绪：{meta_path} 不存在（等 T-D 产出）")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.词表 = 读词表(self.meta)
        self.均值, self.标准差 = 读mel统计(self.meta)
        self.索引 = 建索引(self.词表)
        清单 = _挑(self.meta, 清单键)
        if 清单 is None:
            清单 = sorted((self.目录 / "样本").glob("*.npz"))
            if not 清单:
                raise FileNotFoundError(f"数据未就绪：{self.目录 / '样本'} 下没有 npz")
        if not isinstance(清单, (list, tuple)):
            清单 = [清单]
        self.全部ids = []
        self.路径 = {}
        for i, 项 in enumerate(清单):
            sid, p = _样本路径归一(项, i, self.目录)
            if sid in self.路径:  # 同名去重
                sid = f"{sid}_{i}"
            self.全部ids.append(sid)
            self.路径[sid] = _路径兜底(Path(p), self.目录)
        # —— train/val 划分 ——
        划分 = _挑(self.meta, 划分键) or {}
        tr = _挑(划分, 训练键) if isinstance(划分, dict) else None
        va = _挑(划分, 验证键) if isinstance(划分, dict) else None
        if tr is None:
            tr = _挑(self.meta, 训练键)
        if va is None:
            va = _挑(self.meta, 验证键)
        if tr is not None:
            训练ids = [x for x in (_id归一(t, set(self.全部ids), self.全部ids) for t in tr)
                       if x in set(self.全部ids)]
        else:
            训练ids = []
        if va is not None:
            验证ids = [x for x in (_id归一(t, set(self.全部ids), self.全部ids) for t in va)
                       if x in set(self.全部ids)]
        else:
            验证ids = []
        if not 训练ids or not 验证ids:
            # 确定性兜底：seed 300 洗牌 95/5
            rng = np.random.default_rng(种子)
            序 = list(self.全部ids)
            rng.shuffle(序)
            n_val = max(1, int(round(len(序) * 0.05)))
            验证ids = 序[:n_val]
            训练ids = 序[n_val:]
            self.划分来源 = "兜底(seed300,95/5)"
        else:
            self.划分来源 = "meta.json"
        # —— 前N 限制（默认全量；任务口径：先只用前 40 个样本）——
        if 前N is not None and int(前N) > 0:
            取 = set(self.全部ids[:int(前N)])
            训练ids = [x for x in 训练ids if x in 取]
            验证ids = [x for x in 验证ids if x in 取]
            if not 训练ids:
                训练ids = list(self.全部ids[:int(前N)])
            if not 验证ids:
                rng = np.random.default_rng(种子)
                序 = list(训练ids)
                rng.shuffle(序)
                n_val = max(1, int(round(len(序) * 0.05)))
                验证ids = 序[:n_val]
                训练ids = 序[n_val:] or 序
                self.划分来源 += f" +前N{前N}内重划"
        self.训练ids = 训练ids
        self.验证ids = 验证ids
        # —— 真正读样本 ——
        self.样本 = {}
        缺 = []
        for sid in self.全部ids:
            p = self.路径[sid]
            if not Path(p).exists():
                缺.append(str(p))
                continue
            try:
                self.样本[sid] = 读npz样本(Path(p), self.词表)
            except Exception as e:  # noqa: BLE001
                缺.append(f"{p} ({e})")
        self.缺文件 = 缺
        self.训练ids = [x for x in self.训练ids if x in self.样本]
        self.验证ids = [x for x in self.验证ids if x in self.样本]
        if not self.训练ids:
            raise ValueError(f"训练集为空；缺文件={缺[:5]}")
        if not self.验证ids:
            self.验证ids = self.训练ids[-max(1, len(self.训练ids) // 20):]
        if 打印:
            self.打印诊断()

    def 打印诊断(self):
        tr, va = self.训练ids, self.验证ids
        tr帧 = sum(int(self.样本[i]["mel"].shape[0]) for i in tr)
        va帧 = sum(int(self.样本[i]["mel"].shape[0]) for i in va)
        tr音 = sum(len(self.样本[i]["帧数"]) for i in tr)
        va音 = sum(len(self.样本[i]["帧数"]) for i in va)
        print(f"[数据] {self.目录}")
        print(f"  词表: 声母 {len(self.词表['声母'])} / 韵母 {len(self.词表['韵母'])} / 调 {len(self.词表['调'])}")
        print(f"  mel 统计: 均值 {np.round(self.均值[:3], 4)}.. 标准差 {np.round(self.标准差[:3], 4)}.."
              f"（{'数组80' if self.均值.size > 1 else '标量'}）")
        print(f"  样本 {len(self.样本)}（清单 {len(self.全部ids)}，缺 {len(self.缺文件)}）；"
              f"划分来源={self.划分来源}")
        print(f"  训练 {len(tr)} 样本 / {tr帧} 帧 / {tr音} 音节；"
              f"验证 {len(va)} 样本 / {va帧} 帧 / {va音} 音节")
        print(f"  样本时长中位数 {np.median([self.样本[i]['mel'].shape[0] / 帧率 for i in tr]):.1f}s")
        for sid in tr[:3]:
            s = self.样本[sid]
            print(f"  [{sid}] mel{s['mel'].shape} 音节{len(s['帧数'])} "
                  f"首token={np.asarray(s['tokens'])[0].tolist()} 帧数首={int(np.asarray(s['帧数'])[0])}")
        if self.缺文件:
            print(f"  ! 缺文件示例: {self.缺文件[:2]}")


# ════════════════════════════════════════════════════════════════════════
# 5. 裁剪 / 组批 / 损失
# ════════════════════════════════════════════════════════════════════════
def 裁窗(样本, 起音节: int, 最大帧: int):
    """按音节边界裁一个 ≤最大帧 的窗口（保持音节 token 与帧数自洽）。"""
    帧数 = 样本["帧数"]
    N = len(帧数)
    if N == 0:
        return None
    起 = int(np.clip(起音节, 0, N - 1))
    止, 总 = 起, 0
    while 止 < N and 总 + int(帧数[止]) <= 最大帧:
        总 += int(帧数[止])
        止 += 1
    if 止 == 起:
        止 = 起 + 1
        总 = min(int(帧数[起]), int(样本["mel"].shape[0]))
    if 总 < 4:
        return None
    起帧 = int(帧数[:起].sum())
    mel = 样本["mel"][起帧:起帧 + 总]
    if mel.shape[0] != 总:
        return None
    return dict(mel=mel, tokens=样本["tokens"][起:止],
                帧数=帧数[起:止].astype(np.int64), 源=s样本身份(样本))


def s样本身份(样本):
    return 样本.get("id", "?")


def 起批(窗口s, 批样本=8, 最大帧=3000):
    批, 总 = [], 0
    for w in 窗口s:
        T = int(w["mel"].shape[0])
        if 批 and (len(批) >= 批样本 or 总 + T > 最大帧):
            yield 批
            批, 总 = [], 0
        批.append(w)
        总 += T
    if 批:
        yield 批


def 拼批(批, 均值, 标准差, 掩码帧=边界掩码帧):
    """多窗口按时间拼接成一个长序列 → 张量（无 padding；边界帧不参与 loss）。"""
    mels, 声s, 韵s, 调s, 特征s, 掩码s = [], [], [], [], [], []
    for w in 批:
        s, y, t, f = 展开帧(w["tokens"], w["帧数"])
        T = int(w["mel"].shape[0])
        m = np.ones(T, np.float32)
        b = int(min(掩码帧, T // 2))
        if b > 0:
            m[:b] = 0.0
            m[T - b:] = 0.0
        mels.append(w["mel"].astype(np.float32))
        声s.append(s)
        韵s.append(y)
        调s.append(t)
        特征s.append(f)
        掩码s.append(m)
    mel = np.concatenate(mels, axis=0)
    mel = (mel - 均值.reshape(-1)[0] if 均值.size == 1 else mel - 均值.reshape(1, -1))
    mel = mel / (标准差.reshape(-1)[0] if 标准差.size == 1 else 标准差.reshape(1, -1))
    return dict(
        mel=torch.from_numpy(mel.astype(np.float32)).unsqueeze(0),
        声=torch.from_numpy(np.concatenate(声s)).unsqueeze(0),
        韵=torch.from_numpy(np.concatenate(韵s)).unsqueeze(0),
        调=torch.from_numpy(np.concatenate(调s)).unsqueeze(0),
        特征=torch.from_numpy(np.concatenate(特征s, axis=0)).unsqueeze(0),
        掩码=torch.from_numpy(np.concatenate(掩码s)).unsqueeze(0),
    )


def _上帧张量(mel, K: int):
    """[B,T,80] 归一化 mel → [B,T,K*80]（前 K 帧，t<k 处补 0）。K<=0 返回 None。"""
    if not K or K <= 0:
        return None
    B, T, D = mel.shape
    部分 = []
    for k in range(1, K + 1):
        if k >= T:
            移 = torch.zeros_like(mel)
        else:
            移 = torch.cat([torch.zeros_like(mel[:, :k]), mel[:, :T - k]], dim=1)
        部分.append(移)
    return torch.cat(部分, dim=-1)


def 批损失(预测, 目标, 掩码, 均值, 标准差, 配置):
    """L1(log-mel) + 时间差分平滑(可选) + 线性域 L2 代理(可选)。"""
    w = 掩码.unsqueeze(-1)  # [1,T,1]
    通道 = 预测.shape[-1]
    分母 = (w.sum() * 通道).clamp_min(1e-6)
    损 = 配置["损失L1"] * (((预测 - 目标).abs() * w).sum() / 分母)
    if 配置.get("损失时间差分", 0.0) > 0 and 预测.shape[1] > 1:
        dp = 预测[:, 1:] - 预测[:, :-1]
        dt = 目标[:, 1:] - 目标[:, :-1]
        dw = (掩码[:, 1:] * 掩码[:, :-1]).unsqueeze(-1)
        分母2 = (dw.sum() * 通道).clamp_min(1e-6)
        损 = 损 + 配置["损失时间差分"] * (((dp - dt).abs() * dw).sum() / 分母2)
    # 多尺度时间差分（v17）：强制 2/4/8 帧尺度的谱变化，专攻「过度平滑/瞬态丢失」
    多尺 = 配置.get("多尺度步长") or []
    if 多尺 and 预测.shape[1] > max(int(x) for x in 多尺):
        权 = float(配置.get("多尺度权", 0.05))
        for 步 in 多尺:
            步 = int(步)
            if 步 <= 0 or 预测.shape[1] <= 步:
                continue
            dp = 预测[:, 步:] - 预测[:, :-步]
            dt = 目标[:, 步:] - 目标[:, :-步]
            dw = (掩码[:, 步:] * 掩码[:, :-步]).unsqueeze(-1)
            分母m = (dw.sum() * 通道).clamp_min(1e-6)
            损 = 损 + 权 * (((dp - dt).abs() * dw).sum() / 分母m)
    if 配置.get("损失线性", 0.0) > 0:
        m = 均值.reshape(1, 1, -1) if 均值.numel() > 1 else 均值.reshape(1, 1, 1)
        sd = 标准差.reshape(1, 1, -1) if 标准差.numel() > 1 else 标准差.reshape(1, 1, 1)
        下 = m - 3.0 * sd
        上 = m + 3.0 * sd
        lp = torch.exp(torch.clamp(预测 * sd + m, 下, 上))
        lt = torch.exp(torch.clamp(目标 * sd + m, 下, 上))
        分母3 = (lt * lt * w).sum().clamp_min(1e-6)
        损 = 损 + 配置["损失线性"] * (((lp - lt) ** 2 * w).sum() / 分母3)
    return 损


# ════════════════════════════════════════════════════════════════════════
# 6. 训练
# ════════════════════════════════════════════════════════════════════════
默认训练配置 = dict(
    种子=种子,
    批样本=8,
    最大帧=3000,
    学习率=2e-4,
    权重衰减=1e-4,      # v2a 配方（v2b 的 5e-4 实测 val 更差，已回退）
    预热步=300,         # v2a 配方（v2b 的 100 已回退）
    最小学习率=2e-5,
    轮上限=1200,
    早停轮=120,
    最小改进=2e-5,
    损失L1=1.0,
    损失线性=0.02,
    损失时间差分=0.05,  # v2a 配方（v2b 的 0.30 实测 val 更差，已回退）
    梯度裁剪=1.0,
    时长分钟=40.0,
    边界掩码帧=边界掩码帧,
)


def _保存ckpt(path: Path, state, 词表, mel均值, mel标准差, 配置, 参数, 训练信息):
    内容 = dict(state_dict=state, 词表=词表, 配置=配置, 参数数=int(参数),
              mel均值=np.asarray(mel均值, np.float32),
              mel标准差=np.asarray(mel标准差, np.float32),
              训练信息=训练信息)
    tmp = path.with_suffix(".tmp")
    torch.save(内容, tmp)
    os.replace(tmp, path)


def 训练(前N=40, 分钟=40.0, 输出目录: Path = 模型目录, 数据路: Path = 数据目录,
       详细轮=True, 轮上限=None, 早停轮=None, 通道=256, 膨胀=(1, 2, 4, 8), 学习率=None,
       上一帧数=0, 自条件p=0.0, 迭代训练=1, 初始化=None, 自条件ramp=0, 自条件p最大=0.9,
       周期保存=0, 推理训练=0, 推理阻尼=0.7, 推理前轮权=0.3, 多尺度=None, 多尺度权=0.05):
    输出目录 = Path(输出目录)
    输出目录.mkdir(parents=True, exist_ok=True)
    _固定种子(种子)
    配置 = dict(默认训练配置)
    配置["时长分钟"] = float(分钟)
    配置["自条件p"] = float(自条件p)
    配置["自条件ramp"] = int(自条件ramp)
    配置["自条件p最大"] = float(自条件p最大)
    配置["迭代训练"] = int(迭代训练)
    配置["周期保存"] = int(周期保存)
    配置["多尺度步长"] = [int(x) for x in (多尺度 or [])]
    配置["多尺度权"] = float(多尺度权)
    配置["推理训练"] = int(推理训练)
    配置["推理阻尼"] = float(推理阻尼)
    配置["推理前轮权"] = float(推理前轮权)
    if 学习率 is not None:
        配置["学习率"] = float(学习率)
    if 轮上限 is not None:
        配置["轮上限"] = int(轮上限)
    if 早停轮 is not None:
        配置["早停轮"] = int(早停轮)

    日志 = open(输出目录 / "训练日志.txt", "w", encoding="utf-8")

    def 打(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        日志.write(s + "\n")
        日志.flush()

    开始 = time.time()
    打("=" * 78)
    打(f"[T-E] 学说话模型训练  开始 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    打(f"  torch={torch.__version__}  cuda={torch.cuda.is_available()}"
       f"  设备={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    打(f"  数据={数据路}  产出={输出目录}")

    数据集_ = 数据集(数据路, 前N=前N, 打印=False)
    打(f"[数据] 训练 {len(数据集_.训练ids)} 样本 / 验证 {len(数据集_.验证ids)} 样本；"
       f"划分来源={数据集_.划分来源}；缺文件 {len(数据集_.缺文件)}")
    词表 = 数据集_.词表
    均值 = 数据集_.均值
    标准差 = 数据集_.标准差
    设备 = _设备()
    模型 = 说话模型(词表, 通道=int(通道), 膨胀=tuple(int(x) for x in 膨胀),
                   上一帧数=int(上一帧数)).to(设备)
    if 初始化:
        _ck = torch.load(str(初始化), map_location="cpu", weights_only=False)
        _缺 = 模型.load_state_dict(_ck["state_dict"], strict=False)
        打(f"[初始化] 从 {初始化} 载入；missing={len(_缺.missing_keys)} unexpected={len(_缺.unexpected_keys)}")
    参 = 参数数(模型)
    打(f"[模型] 参数量 {参:,}（{参 / 1e6:.3f}M）；配置={模型.配置}")
    打(f"[模型] 无自注意力主干 / 无预训练权重（static audit 由本脚本末尾执行）")
    打(f"[数据] mel 均值 {np.round(均值[:2], 4)} 标准差 {np.round(标准差[:2], 4)}；"
       f"全局帧数 训练 {sum(int(数据集_.样本[i]['mel'].shape[0]) for i in 数据集_.训练ids)}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    优化器 = torch.optim.AdamW(模型.parameters(), lr=配置["学习率"],
                              weight_decay=配置["权重衰减"])
    调度 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        优化器, mode="min", factor=0.5, patience=max(10, int(配置["早停轮"] // 4)),
        min_lr=配置["最小学习率"])

    训练ids, 验证ids = 数据集_.训练ids, 数据集_.验证ids
    均值_t = torch.from_numpy(均值).to(设备).reshape(1, 1, -1) if 均值.size > 1 else \
        torch.tensor(float(均值[0]), device=设备)
    标准差_t = torch.from_numpy(标准差).to(设备).reshape(1, 1, -1) if 标准差.size > 1 else \
        torch.tensor(float(标准差[0]), device=设备)

    曲线 = dict(轮=[], 训练损失=[], 验证损失=[], 学习率=[])
    best = dict(val=float("inf"), 轮=-1, state=None)
    rng = np.random.default_rng(种子)
    步 = 0
    提前停 = False
    训练损失起 = None

    def 造训练窗口():
        """每轮：每样本随机窗口（音节边界起点），拼到 ≤最大帧。"""
        序 = list(训练ids)
        rng.shuffle(序)
        窗口s = []
        for sid in 序:
            s = 数据集_.样本[sid]
            N = len(s["帧数"])
            w = None
            if int(s["mel"].shape[0]) <= 配置["最大帧"]:
                w = 裁窗(s, 0, 配置["最大帧"])
            else:
                for _ in range(4):
                    起 = int(rng.integers(0, N))
                    w = 裁窗(s, 起, 配置["最大帧"])
                    if w is not None and len(w["帧数"]) >= 6:
                        break
                if w is None or len(w["帧数"]) < 6:
                    w = 裁窗(s, 0, 配置["最大帧"])
            if w is not None:
                窗口s.append(w)
        return 窗口s

    def 造验证窗口():
        窗口s = []
        for sid in 验证ids:
            s = 数据集_.样本[sid]
            N = len(s["帧数"])
            起s = [0]
            if int(s["mel"].shape[0]) > 配置["最大帧"] and N > 4:
                起s += sorted({N // 2, N - 3})
            for c in 起s:
                w = 裁窗(s, c, 配置["最大帧"])
                if w is not None:
                    窗口s.append(w)
        return 窗口s

    验证窗口 = 造验证窗口()

    def 评估():
        模型.eval()
        总损失, 总帧 = 0.0, 0
        with torch.no_grad():
            for 批 in 起批(验证窗口, 配置["批样本"], 配置["最大帧"]):
                b = 拼批(批, 均值, 标准差)
                b = {k: v.to(设备) for k, v in b.items()}
                出 = 模型(b["声"], b["韵"], b["调"], b["特征"], _上帧张量(b["mel"], 模型.上一帧数))
                损 = 批损失(出, b["mel"], b["掩码"], 均值_t, 标准差_t, 配置)
                n = int(b["掩码"].sum().item())
                总损失 += float(损.item()) * n
                总帧 += n
        return 总损失 / max(总帧, 1)

    打(f"[训练] 开始：轮上限={配置['轮上限']} 早停轮={配置['早停轮']} "
       f"lr={配置['学习率']} batch≤{配置['批样本']}样本/{配置['最大帧']}帧 "
       f"时长预算={配置['时长分钟']}min")
    for 轮 in range(1, int(配置["轮上限"]) + 1):
        if (time.time() - 开始) / 60.0 > 配置["时长分钟"]:
            打(f"[训练] 到达时长预算 {配置['时长分钟']}min，停止于第 {轮} 轮")
            break
        模型.train()
        窗口s = 造训练窗口()
        总损, 总帧, 批数 = 0.0, 0, 0
        自条件p本 = 配置.get("自条件p", 0.0)
        if 配置.get("自条件ramp", 0) and 配置.get("自条件p最大", 0.0) > 0:
            自条件p本 = 配置["自条件p最大"] * min(1.0, 轮 / float(配置["自条件ramp"]))
        for 批 in 起批(窗口s, 配置["批样本"], 配置["最大帧"]):
            b = 拼批(批, 均值, 标准差)
            b = {k: v.to(设备, non_blocking=True) for k, v in b.items()}
            for g in 优化器.param_groups:  # 线性预热
                if 步 < 配置["预热步"]:
                    g["lr"] = 配置["学习率"] * (步 + 1) / 配置["预热步"]
            上帧 = _上帧张量(b["mel"], 模型.上一帧数)
            迭代训练 = int(配置.get("迭代训练", 1))
            推理训练 = int(配置.get("推理训练", 0))
            if 模型.上一帧数 > 0 and 推理训练 > 1:
                # 训练=推理同款：第 1 轮 prev=0，阻尼迭代 N 轮，前轮小权重锚定、末轮主损失
                上 = None
                平滑 = None
                损 = None
                权 = 0.0
                阻尼w = float(配置.get("推理阻尼", 0.7))
                for _it in range(推理训练):
                    出 = 模型(b["声"], b["韵"], b["调"], b["特征"], 上)
                    本轮损 = 批损失(出, b["mel"], b["掩码"], 均值_t, 标准差_t, 配置)
                    w = 1.0 if _it == 推理训练 - 1 else float(配置.get("推理前轮权", 0.3))
                    损 = (本轮损 * w) if 损 is None else (损 + 本轮损 * w)
                    权 += w
                    if _it < 推理训练 - 1:
                        平滑 = 出 if 平滑 is None else (阻尼w * 出 + (1.0 - 阻尼w) * 平滑)
                        上 = _上帧张量(平滑, 模型.上一帧数)
                损 = 损 / max(权, 1e-6)
            elif 模型.上一帧数 > 0 and 迭代训练 > 1:
                # 端到端迭代精炼：第 1 轮 prev=0，第 2..N 轮用上一轮自己的输出（图内可导）；
                # 两轮都算损失（锚住第 1 轮，防止只优化末轮把模型带偏）
                上 = None
                损 = None
                for _it in range(迭代训练):
                    出 = 模型(b["声"], b["韵"], b["调"], b["特征"], 上)
                    本轮损 = 批损失(出, b["mel"], b["掩码"], 均值_t, 标准差_t, 配置)
                    损 = 本轮损 if 损 is None else (损 + 本轮损)
                    if _it < 迭代训练 - 1:
                        上 = _上帧张量(出, 模型.上一帧数)
                损 = 损 / float(迭代训练)
            else:
                if 模型.上一帧数 > 0 and 自条件p本 > 0 and float(np.random.random()) < 自条件p本:
                    # 计划采样：用模型自己（无 prev）的粗输出当 prev，减轻曝光偏差
                    with torch.no_grad():
                        粗 = 模型(b["声"], b["韵"], b["调"], b["特征"], None)
                    上帧 = _上帧张量(粗, 模型.上一帧数)
                出 = 模型(b["声"], b["韵"], b["调"], b["特征"], 上帧)
                损 = 批损失(出, b["mel"], b["掩码"], 均值_t, 标准差_t, 配置)
            优化器.zero_grad(set_to_none=True)
            损.backward()
            if 配置["梯度裁剪"] > 0:
                torch.nn.utils.clip_grad_norm_(模型.parameters(), 配置["梯度裁剪"])
            优化器.step()
            步 += 1
            n = int(b["掩码"].sum().item())
            总损 += float(损.item()) * n
            总帧 += n
            批数 += 1
        训练损 = 总损 / max(总帧, 1)
        if 训练损失起 is None:
            训练损失起 = 训练损
        验证损 = 评估()
        调度.step(验证损)
        曲线["轮"].append(轮)
        曲线["训练损失"].append(round(训练损, 6))
        曲线["验证损失"].append(round(验证损, 6))
        曲线["学习率"].append(float(优化器.param_groups[0]["lr"]))
        if 验证损 < best["val"] - 配置["最小改进"] or best["state"] is None:
            best = dict(val=float(验证损), 轮=轮,
                        state={k: v.detach().cpu().clone() for k, v in 模型.state_dict().items()})
        if 详细轮 and (轮 <= 5 or 轮 % 10 == 0 or 轮 == 1):
            打(f"  轮{轮:4d}  训练 {训练损:.5f}  验证 {验证损:.5f}  "
               f"lr={优化器.param_groups[0]['lr']:.2e}  最佳 {best['val']:.5f}@{best['轮']}  "
               f"已用 {(time.time() - 开始):.0f}s")
        if 轮 % 10 == 0:
            训练信息 = dict(轮=轮, 训练损失=训练损, 验证损失=验证损,
                         最佳轮=best["轮"], 最佳验证损失=best["val"],
                         耗时秒=round(time.time() - 开始, 1), 完成=False)
            _保存ckpt(输出目录 / "ckpt.pt", best["state"], 词表, 均值, 标准差,
                     模型.配置, 参, 训练信息)
            (输出目录 / "训练曲线.json").write_text(
                json.dumps(dict(配置=配置, 曲线=曲线, 训练信息=训练信息,
                                训练样本数=len(训练ids), 验证样本数=len(验证ids),
                                参数数=参), ensure_ascii=False, indent=1),
                encoding="utf-8")
        _周期 = int(配置.get("周期保存", 0) or 0)
        if _周期 > 0 and 轮 % _周期 == 0:
            _态 = {k: v.detach().cpu().clone() for k, v in 模型.state_dict().items()}
            _保存ckpt(输出目录 / ("ckpt_e%04d.pt" % 轮), _态, 词表, 均值, 标准差,
                      模型.配置, 参, dict(说明="周期保存", 轮=轮,
                                          训练损失=训练损, 验证损失=验证损))
        if 轮 - best["轮"] >= 配置["早停轮"]:
            打(f"[训练] 早停：验证损失 {配置['早停轮']} 轮无改进（最佳 {best['val']:.5f}@{best['轮']}）")
            提前停 = True
            break

    # 保存末轮状态：便于按「推理 CER」而不是 val 选模型（val 会因曝光偏差误导）
    try:
        _末态 = {k: v.detach().cpu().clone() for k, v in 模型.state_dict().items()}
        _保存ckpt(输出目录 / "ckpt_末.pt", _末态, 词表, 均值, 标准差, 模型.配置, 参,
                  dict(说明="末轮状态（训练收尾）", 轮=len(曲线["轮"]),
                       训练损失=float(曲线["训练损失"][-1]) if 曲线["训练损失"] else None))
    except Exception as _e:  # noqa: BLE001
        print("[警告] 末轮 ckpt 保存失败:", _e)

    耗时 = time.time() - 开始
    峰值MB = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0
    验证终 = 曲线["验证损失"][-1] if 曲线["验证损失"] else float("nan")
    训练终 = 曲线["训练损失"][-1] if 曲线["训练损失"] else float("nan")
    打(f"[训练] 结束：耗时 {耗时:.0f}s（{耗时 / 60:.1f}min）；"
       f"训练损失 {训练损失起:.5f}→{训练终:.5f}；验证损失 {曲线['验证损失'][0]:.5f}→{验证终:.5f}；"
       f"最佳 {best['val']:.5f}@{best['轮']} 轮；显存峰值 {峰值MB:.0f}MB")

    训练信息 = dict(轮=len(曲线["轮"]), 训练损失起=训练损失起, 训练损失终=训练终,
                 验证损失起=曲线["验证损失"][0] if 曲线["验证损失"] else None,
                 验证损失终=验证终, 最佳轮=best["轮"], 最佳验证损失=best["val"],
                 耗时秒=round(耗时, 1), 提前停=提前停, 完成=True,
                 显存峰值MB=round(峰值MB, 1), 种子=种子,
                 torch版本=torch.__version__,
                 设备=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                 脚本sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 时间戳=time.strftime("%Y-%m-%d %H:%M:%S"))
    _保存ckpt(输出目录 / "ckpt.pt", best["state"], 词表, 均值, 标准差,
             模型.配置, 参, 训练信息)

    # —— 两趟推理一致性（C13）——
    模型.load_state_dict(best["state"])
    模型.eval()
    模型.mel均值 = 均值
    模型.mel标准差 = 标准差
    测试文本 = "今天天气不错，我们出去走走吧。"
    趟1 = 合成mel(模型, 测试文本)
    趟2 = 合成mel(模型, 测试文本)
    一致同 = bool(np.array_equal(趟1, 趟2))
    重载 = 加载(输出目录 / "ckpt.pt")
    趟3 = 合成mel(重载, 测试文本)
    一致跨载 = bool(np.array_equal(趟1, 趟3))
    打(f"[C13] 两趟推理逐位同={一致同}；重载 ckpt 后与内存模型逐位同={一致跨载}；"
       f"合成 mel 形状={趟1.shape} dtype={趟1.dtype} 有限={bool(np.isfinite(趟1).all())}")
    训练信息["两趟一致"] = 一致同
    训练信息["重载一致"] = 一致跨载
    训练信息["合成mel形状"] = list(趟1.shape)

    # 用最好的 state 再存一次（带上 C13 结果）
    _保存ckpt(输出目录 / "ckpt.pt", best["state"], 词表, 均值, 标准差,
             模型.配置, 参, 训练信息)
    (输出目录 / "训练曲线.json").write_text(
        json.dumps(dict(配置=配置, 曲线=曲线, 训练信息=训练信息,
                        训练样本数=len(训练ids), 验证样本数=len(验证ids),
                        参数数=参), ensure_ascii=False, indent=1), encoding="utf-8")
    审计 = 架构审计(Path(__file__))
    写训练摘要(输出目录 / "训练摘要.md", 参, 配置, 训练信息, 曲线, 数据集_, 审计,
            输出目录 / "ckpt.pt")
    打(f"[审计] 禁用调用命中 {审计['命中数']}；产物：ckpt.pt / 训练曲线.json / 训练摘要.md")
    日志.close()
    return dict(参数数=参, 训练信息=训练信息, 曲线=曲线, 审计=审计,
                训练样本数=len(训练ids), 验证样本数=len(验证ids))


# ════════════════════════════════════════════════════════════════════════
# 7. 推理 API
# ════════════════════════════════════════════════════════════════════════
def 加载(ckpt路径: Path | str, 设备=None):
    """ckpt → 模型（eval 模式，附带 词表 / mel 统计 / 配置）。生成路径只读本仓自训练权重。"""
    ck = torch.load(str(ckpt路径), map_location="cpu", weights_only=False)
    模型 = 说话模型(ck["词表"].copy() if isinstance(ck["词表"], dict) else ck["词表"],
                   **ck["配置"])
    模型.load_state_dict(ck["state_dict"])
    模型.词表 = ck["词表"]
    模型.mel均值 = np.asarray(ck["mel均值"], np.float32)
    模型.mel标准差 = np.asarray(ck["mel标准差"], np.float32)
    模型.训练信息 = ck.get("训练信息", {})
    模型.eval()
    设备 = 设备 if 设备 is not None else _设备()
    模型.to(设备)
    return 模型


def 文本到token(词表, 文本: str):
    """文本 → (tokens[N,3], 有效音节列表, 跳过的音节)。"""
    声_i, 韵_i, 调_i = 建索引(词表)
    音节s = 音韵.音节列表(文本)
    ids, 有效, 跳过 = [], [], []
    for s in 音节s:
        声 = str(s.get("声") or "")
        韵 = str(s.get("韵") or "")
        调 = int(s.get("调") or 5)
        i声 = 声_i.get(声)
        if i声 is None:
            i声 = 声_i.get("", 0)
        i韵 = 韵_i.get(韵)
        if i韵 is None:
            跳过.append(dict(拼音=s.get("拼音"), 韵=韵))
            continue
        i调 = 调_i.get(调)
        if i调 is None:
            i调 = 调_i.get(5, len(词表["调"]) - 1)
        ids.append((i声, i韵, i调))
        有效.append(s)
    return np.asarray(ids, dtype=np.int64).reshape(-1, 3), 有效, 跳过


def 合成mel(模型, 文本: str, 每音节帧数=None, 音节帧数=None):
    """文本 → log-mel [T,80]（float32，未归一化回 mel 统计尺度）。

    每音节默认 20 帧（200ms）；声母段 25% / 韵母 75%（至少 2 帧声母）。
    """
    每 = int(音节帧数 if 音节帧数 is not None
            else (每音节帧数 if 每音节帧数 is not None else 20))
    每 = max(每, 4)
    tokens, 有效, 跳过 = 文本到token(模型.词表, 文本)
    if len(tokens) == 0:
        raise ValueError(f"文本无可合成音节：{文本!r} 跳过={跳过[:5]}")
    帧数 = np.full(len(tokens), 每, dtype=np.int64)
    声, 韵, 调, 特征 = 展开帧(tokens, 帧数)
    设备 = next(模型.parameters()).device
    张 = lambda a: torch.from_numpy(a).unsqueeze(0).to(设备)  # noqa: E731
    with torch.no_grad():
        prev = None
        默认迭代 = int(os.environ.get("M66_迭代推理", "3"))
        阻尼a = float(os.environ.get("M66_阻尼", "1.0"))   # 1.0=不阻尼；<1 时与上轮估计指数平均，抑制漂移
        收敛阈 = float(os.environ.get("M66_收敛阈", "0.0"))  # >0：迭代收敛即停（自适应轮数）
        迭代 = 默认迭代 if int(getattr(模型, "上一帧数", 0)) > 0 else 1
        平滑 = None
        上轮 = None
        for _ in range(迭代):
            出 = 模型(张(声), 张(韵), 张(调), 张(特征), prev)
            if 收敛阈 > 0 and 上轮 is not None:
                if float((出 - 上轮).abs().mean().item()) < 收敛阈:
                    break
            上轮 = 出
            if int(getattr(模型, "上一帧数", 0)) > 0:
                平滑 = 出 if 平滑 is None else (阻尼a * 出 + (1.0 - 阻尼a) * 平滑)
                prev = _上帧张量(平滑, 模型.上一帧数)
    mel = 出[0].detach().cpu().numpy().astype(np.float32)
    m = np.asarray(模型.mel均值, np.float32).reshape(-1)
    sd = np.asarray(模型.mel标准差, np.float32).reshape(-1)
    mel = mel * (sd[0] if sd.size == 1 else sd.reshape(1, -1)) + \
        (m[0] if m.size == 1 else m.reshape(1, -1))
    return mel.astype(np.float32)


# ════════════════════════════════════════════════════════════════════════
# 8. 红线静态审计（只查真实调用点，不查注释文字）
# ════════════════════════════════════════════════════════════════════════
def 架构审计(文件: Path):
    文 = Path(文件).read_text(encoding="utf-8")
    # 拼接构造模式串：不把禁用英文词直接写进本文件
    禁 = [
        "Multi" + "head" + "Att" + "ention",
        "scaled_dot_" + "product_" + "att" + "ention",
        "Trans" + "former" + "Encoder",
        "Trans" + "former" + "Decoder",
        "from_" + "pretrained",
        "torch." + "hub",
        "import " + "trans" + "formers",
        "nn." + "LSTM",
    ]
    命中 = []
    for p in 禁:
        if p in 文:
            命中.append(p)
    # 模型类自查：不得出现注意力/预训练相关子模块
    模型 = 说话模型(默认词表())
    模块名 = [type(m).__name__ for m in 模型.modules()]
    可疑 = [n for n in 模块名
            if ("Att" + "ention") in n or ("Multi" + "head") in n
            or ("Trans" + "former") in n]
    return dict(命中=命中, 命中数=len(命中), 可疑模块=可疑, 模块类型=sorted(set(模块名)))


# ════════════════════════════════════════════════════════════════════════
# 9. 训练摘要
# ════════════════════════════════════════════════════════════════════════
def 写训练摘要(path: Path, 参, 配置, 训练信息, 曲线, 数据集_, 审计, ckpt路径):
    ck = torch.load(str(ckpt路径), map_location="cpu", weights_only=False)
    tr帧 = sum(int(数据集_.样本[i]["mel"].shape[0]) for i in 数据集_.训练ids)
    va帧 = sum(int(数据集_.样本[i]["mel"].shape[0]) for i in 数据集_.验证ids)
    tr音 = sum(len(数据集_.样本[i]["帧数"]) for i in 数据集_.训练ids)
    va音 = sum(len(数据集_.样本[i]["帧数"]) for i in 数据集_.验证ids)
    mel均值 = np.asarray(ck["mel均值"], np.float32)
    mel标准差 = np.asarray(ck["mel标准差"], np.float32)
    行 = [
        "# meeting66 · T-E 训练摘要 —— 学说话声学模型（音节音素 → log-mel）",
        "",
        f"> 生成于 {训练信息.get('时间戳')}｜种子 {种子}｜脚本 sha256 `{训练信息.get('脚本sha256')}`",
        f"> 产物：`{ckpt路径}`、`训练曲线.json`（本目录）；数据：`{数据集_.目录}`",
        "",
        "## 1. 结论（实测）",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 参数量 | **{参:,}**（{参 / 1e6:.3f}M） |",
        f"| 训练样本数 / 验证样本数 | {len(数据集_.训练ids)} / {len(数据集_.验证ids)} |",
        f"| 训练帧数 / 验证帧数 | {tr帧:,} / {va帧:,}（10ms/帧） |",
        f"| 训练音节数 / 验证音节数 | {tr音:,} / {va音:,} |",
        f"| 训练耗时 | **{训练信息['耗时秒'] / 60:.1f} min**（轮数 {训练信息['轮']}，早停={训练信息['提前停']}） |",
        f"| 训练 loss（起 → 终） | {训练信息['训练损失起']:.5f} → {训练信息['训练损失终']:.5f} |",
        f"| val loss（起 → 终） | {训练信息['验证损失起']:.5f} → {训练信息['验证损失终']:.5f} |",
        f"| **val loss 最佳** | **{训练信息['最佳验证损失']:.5f}**（第 {训练信息['最佳轮']} 轮） |",
        f"| 两趟推理逐位同（C13） | **{训练信息.get('两趟一致')}**；重载 ckpt 后逐位同 = {训练信息.get('重载一致')} |",
        f"| 显存峰值 | {训练信息['显存峰值MB']:.0f} MB |",
        f"| 设备 | {训练信息['设备']}，torch {训练信息['torch版本']} |",
        f"| 红线审计（真实调用点） | 命中 **{审计['命中数']}** 条；可疑子模块 {审计['可疑模块']} |",
        "",
        "## 2. 架构（非注意力框架，从零训练）",
        "",
        "```",
        "音节 token [声母_id, 韵母_id, 调_id] × 每音节帧数",
        "  → 帧级展开（声母段 25% / 韵母段 75%，至少 2 帧声母）",
        "  → embedding(声母/韵母/调, dim 128) ⊕ 10 维帧级特征（音内位置/段内位置/是否声母/声母占比/",
        "      音节时长/全局位置/音节相对位置/句首/句末/离边界）",
        "  → Linear → LayerNorm(256)",
        "  → 4×[Conv1D(k=5, C=256, dilation=1/2/4/8) → LayerNorm → GELU → 残差]",
        "  → LayerNorm → GELU → Conv1D 1×1 → 80 维 log-mel（+ 输入 1×1 旁路）",
        "```",
        "",
        "无自注意力主干、无序列注意力层、无第三方预训练权重、无第三方推理栈（C1）；",
        "生成路径只读本仓自训练权重（C10）。",
        "",
        "## 3. 训练配置",
        "",
        "| 项 | 值 |",
        "|---|---|",
    ]
    for k in ["批样本", "最大帧", "学习率", "权重衰减", "预热步", "早停轮", "最小改进",
              "损失L1", "损失线性", "损失时间差分", "梯度裁剪", "时长分钟", "轮上限"]:
        行.append(f"| {k} | {配置[k]} |")
    行 += [
        "",
        "损失 = L1(log-mel 标准化域) + 0.05×L1(时间差分) + 0.02×线性域 L2 代理；"
        "AdamW，lr 2e-4（300 步线性预热 + ReduceLROnPlateau）。"
        "拼接样本边界 40 帧不参与 loss（卷积感受野 30 帧）。",
        "",
        "## 4. 数据",
        "",
        f"- 目录：`{数据集_.目录}`；词表：声母 {len(数据集_.词表['声母'])} / 韵母 {len(数据集_.词表['韵母'])} / 调 {len(数据集_.词表['调'])}",
        f"- 划分来源：{数据集_.划分来源}；缺文件 {len(数据集_.缺文件)} 个",
        f"- mel 统计：均值（前 3）{np.round(mel均值[:3], 4).tolist()}，标准差（前 3）{np.round(mel标准差[:3], 4).tolist()}"
        f"（{'逐带 80' if mel均值.size > 1 else '标量'}）",
        "- 对齐：VAD 语音段内按音节数均分帧（T-D 口径），本模型只学「token→谱包络」，不学对齐。",
        "",
        "## 5. loss 曲线（每 10 轮抽样）",
        "",
        "| 轮 | 训练 | 验证 | lr |",
        "|---|---|---|---|",
    ]
    for i in range(0, len(曲线["轮"]), max(1, len(曲线["轮"]) // 20)):
        行.append(f"| {曲线['轮'][i]} | {曲线['训练损失'][i]:.5f} | "
                  f"{曲线['验证损失'][i]:.5f} | {曲线['学习率'][i]:.2e} |")
    行 += [
        "",
        "完整曲线见 `训练曲线.json`（键：轮/训练损失/验证损失/学习率）。",
        "",
        "## 6. API 用法",
        "",
        "```python",
        "import sys; sys.path.insert(0, r'meeting66/实施')",
        "import 模型_学说话 as M",
        "模型 = M.加载(r'meeting66/产物/模型/ckpt.pt')",
        "mel = M.合成mel(模型, '你好，世界', 每音节帧数=20)   # → np.float32 [T,80]",
        "```",
        "",
        "## 7. 诚实边界",
        "",
        "- 首版**无帧级对齐**（音节均分），时长/韵律粗糙；mel 由后续声码器（Griffin-Lim 等）出声，"
        "**本件不宣称可懂/可听**，可懂度需 T-F 双集 ASR CER 与所有者盲听判定。",
        "- 训练仅用前 40 个样本（约 40 分钟音频）跑通；全量 167 样本属下一轮（若时间允许）。",
        "- GPU 训练端浮点不保证跨机逐位同（C13 口径＝固定种子 + 归档 ckpt + 同 ckpt 推理两趟逐位同）。",
        "- 未做音色/情感；单说话人（SpeakerA）。",
        "",
        f"*meeting66 · T-E 模型_学说话.py ｜ 只新增 meeting66/（C14）｜ {训练信息.get('时间戳')}*",
    ]
    Path(path).write_text("\n".join(行) + "\n", encoding="utf-8")


# ════════════════════════════════════════════════════════════════════════
# 10. 自检（假数据；不依赖 T-D 真实产物）
# ════════════════════════════════════════════════════════════════════════
def _造假mel(T: int, rng, 带=80):
    """平滑随机 log-mel（确定性），只为跑通链路。"""
    t = np.linspace(0, 1, T, dtype=np.float32)[:, None]
    b = np.linspace(0, 1, 带, dtype=np.float32)[None, :]
    基线 = -6.0 + 2.0 * np.sin(2 * np.pi * (b * 3 + t * 0.7))
    mel = 基线 + 0.8 * np.sin(2 * np.pi * (b * 7 - t * 1.3)) + rng.normal(0, 0.15, (T, 带))
    return mel.astype(np.float32)


def 自检(保留临时=False):
    print("=" * 78)
    print("[自检] 假数据端到端：模型前向 / 训练 2 轮 / 存读 ckpt / 合成 mel / 两趟一致 / 审计")
    临时 = Path(tempfile.mkdtemp(prefix="m66模型自检_"))
    数据 = 临时 / "数据"
    (数据 / "样本").mkdir(parents=True)
    输出 = 临时 / "模型"
    词表 = 默认词表()
    rng = np.random.default_rng(种子)
    清单 = []
    for i in range(8):
        N = int(rng.integers(6, 14))
        tokens = np.stack([rng.integers(0, len(词表["声母"]), N),
                           rng.integers(0, len(词表["韵母"]), N),
                           rng.integers(0, len(词表["调"]), N)], axis=1).astype(np.int64)
        帧数 = rng.integers(16, 34, N).astype(np.int64)
        T = int(帧数.sum())
        mel = _造假mel(T, rng)
        np.savez(数据 / "样本" / f"{i:04d}.npz", mel=mel, tokens=tokens, 帧数=帧数)
        清单.append(f"样本/{i:04d}.npz")
    meta = dict(
        词表=词表, sr=22050, hop_ms=10, mel_bins=MEL带,
        mel均值=float(np.mean([np.load(数据 / "样本" / f"{i:04d}.npz")["mel"].mean() for i in range(8)])),
        mel标准差=float(np.mean([np.load(数据 / "样本" / f"{i:04d}.npz")["mel"].std() for i in range(8)])),
        样本清单=清单,
        划分=dict(train=[f"样本/{i:04d}.npz" for i in range(7)],
                 val=[f"样本/{i:04d}.npz" for i in range(7, 8)]),
    )
    (数据 / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    报告 = dict(通过=False, 项={})

    # 1. 数据读取
    ds = 数据集(数据, 打印=True)
    报告["项"]["数据读取"] = dict(训练=len(ds.训练ids), 验证=len(ds.验证ids),
                             均值=float(np.asarray(ds.均值).ravel()[0]))

    # 2. 模型前向 / 参数量
    模型 = 说话模型(词表)
    参 = 参数数(模型)
    声, 韵, 调, 特 = 展开帧(np.asarray([[1, 2, 3], [4, 5, 2]]), np.asarray([20, 24]))
    with torch.no_grad():
        出 = 模型(torch.from_numpy(声).unsqueeze(0), torch.from_numpy(韵).unsqueeze(0),
                 torch.from_numpy(调).unsqueeze(0), torch.from_numpy(特).unsqueeze(0))
    形状对 = 出.shape == (1, int(20 + 24), MEL带)
    报告["项"]["前向"] = dict(参数=参, 输出形状=list(出.shape), 形状对=bool(形状对),
                          数值有限=bool(torch.isfinite(出).all()))
    assert 形状对
    assert 1.0e6 <= 参 <= 3.0e6, f"参数量 {参} 不在 1–3M"

    # 3. 训练（小预算）
    r = 训练(前N=None, 分钟=3.0, 输出目录=输出, 数据路=数据, 详细轮=True,
            轮上限=40, 早停轮=40)
    ck = 输出 / "ckpt.pt"
    q = 输出 / "训练曲线.json"
    s = 输出 / "训练摘要.md"
    ck_ok = ck.exists() and q.exists() and s.exists()
    曲线 = r["曲线"]
    训练降 = (曲线["训练损失"][0] - 曲线["训练损失"][-1]) if len(曲线["训练损失"]) >= 2 else 0.0
    报告["项"]["训练"] = dict(轮=len(曲线["轮"]), 训练损失起=曲线["训练损失"][0],
                          训练损失终=曲线["训练损失"][-1],
                          验证损失起=曲线["验证损失"][0], 验证损失终=曲线["验证损失"][-1],
                          训练损失下降=float(训练降), 产物齐全=ck_ok,
                          两趟一致=r["训练信息"].get("两趟一致"),
                          重载一致=r["训练信息"].get("重载一致"))

    # 4. 加载 + 合成
    载入 = 加载(ck)
    文 = "你好世界"
    mel = 合成mel(载入, 文)
    音节数 = len(音韵.音节列表(文))
    形状ok = mel.shape == (音节数 * 20, MEL带) and mel.dtype == np.float32
    mel2 = 合成mel(载入, 文)
    报告["项"]["合成"] = dict(形状=list(mel.shape), 期望=[音节数 * 20, MEL带],
                           dtype=str(mel.dtype), 形状对=bool(形状ok),
                          有限=bool(np.isfinite(mel).all()),
                          两趟逐位同=bool(np.array_equal(mel, mel2)))
    假数据版 = 合成mel(载入, 文)
    报告["项"]["合成"]["两趟再验"] = bool(np.array_equal(mel, 假数据版))

    # 5. 审计
    审计 = 架构审计(Path(__file__))
    报告["项"]["审计"] = 审计
    报告["项"]["GPU"] = dict(可用=torch.cuda.is_available(),
                          名称=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    通过 = bool(ck_ok and 报告["项"]["合成"]["两趟逐位同"]
                and 报告["项"]["合成"]["形状对"] and 审计["命中数"] == 0
                and r["训练信息"].get("两趟一致") is True
                and 参 >= 1e6)
    报告["通过"] = 通过
    (输出 / "自检报告.json").write_text(json.dumps(报告, ensure_ascii=False, indent=1,
                                              default=_json值), encoding="utf-8")
    print("-" * 78)
    print(f"[自检] 通过={通过}")
    print(json.dumps(报告["项"], ensure_ascii=False, indent=1, default=_json值))
    print(f"[自检] 报告：{输出 / '自检报告.json'}")
    if not 保留临时:
        print(f"[自检] 临时目录（保留供查）：{临时}")
    return 通过, 报告


# ════════════════════════════════════════════════════════════════════════
# 11. CLI
# ════════════════════════════════════════════════════════════════════════
def _主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    解析 = argparse.ArgumentParser(description="meeting66 T-E 学说话声学模型")
    子 = 解析.add_subparsers(dest="命令", required=True)

    p = 子.add_parser("自检", help="假数据端到端自检")
    p.add_argument("--保留", action="store_true")

    p = 子.add_parser("数据", help="只读诊断数据契约")
    p.add_argument("--数据", default=str(数据目录))
    p.add_argument("--前N", type=int, default=None)

    p = 子.add_parser("训练", help="训练（默认前 40 样本 / 40 分钟预算）")
    p.add_argument("--前N", type=int, default=40)
    p.add_argument("--分钟", type=float, default=40.0)
    p.add_argument("--数据", default=str(数据目录))
    p.add_argument("--出", default=str(模型目录))
    p.add_argument("--轮上限", type=int, default=None)
    p.add_argument("--早停轮", type=int, default=None)
    p.add_argument("--全量", action="store_true", help="不限前 N（全量 167 样本）")
    p.add_argument("--通道", type=int, default=256, help="Conv1D 通道数（N3 加大容量用）")
    p.add_argument("--膨胀", default="1,2,4,8", help="逗号分隔的 dilation 序列，如 1,2,4,8,16")
    p.add_argument("--自条件ramp", type=int, default=0, help="计划采样 ramp 轮数（p 从 0 线性升到 p最大）")
    p.add_argument("--自条件p最大", type=float, default=0.9, help="计划采样 p 上限")
    p.add_argument("--周期保存", type=int, default=0, help="每 N 轮另存 ckpt_e{N}.pt（供推理 CER 选模型）")
    p.add_argument("--推理训练", type=int, default=0, help="训练=推理同款：N 轮阻尼自条件迭代（0=关）")
    p.add_argument("--推理阻尼", type=float, default=0.7, help="推理训练时的阻尼系数")
    p.add_argument("--推理前轮权", type=float, default=0.3, help="非末轮的损失权重（锚定）")
    p.add_argument("--多尺度", default="", help="多尺度时间差分步长，如 2,4,8（空=关）")
    p.add_argument("--多尺度权", type=float, default=0.05, help="多尺度差分损失权重")
    p.add_argument("--学习率", type=float, default=None)
    p.add_argument("--上一帧数", type=int, default=0, help="v5 迭代自条件：输入前 K 帧（0=旧模型）")
    p.add_argument("--自条件p", type=float, default=0.0, help="计划采样概率：用模型自产 prev 代替真值 prop")
    p.add_argument("--迭代训练", type=int, default=1, help="端到端迭代精炼轮数（1=不迭代；2=两轮图内可导）")
    p.add_argument("--初始化", default=None, help="从已有 ckpt 初始化权重（微调用）")

    p = 子.add_parser("合成", help="ckpt → mel")
    p.add_argument("文本")
    p.add_argument("--ckpt", default=str(模型目录 / "ckpt.pt"))
    p.add_argument("--每音节帧数", type=int, default=20)
    p.add_argument("--npz", default=None, help="把 mel 存成 npz（键 mel）")

    参 = 解析.parse_args()
    if 参.命令 == "自检":
        通, _ = 自检(保留临时=参.保留)
        sys.exit(0 if 通 else 1)
    if 参.命令 == "数据":
        ds = 数据集(Path(参.数据), 前N=参.前N)
        print(json.dumps(dict(训练=len(ds.训练ids), 验证=len(ds.验证ids),
                              缺文件=ds.缺文件[:5], 划分来源=ds.划分来源),
                         ensure_ascii=False, indent=1, default=_json值))
        return
    if 参.命令 == "训练":
        前N = None if 参.全量 else 参.前N
        膨胀 = tuple(int(x) for x in str(参.膨胀).split(",") if str(x).strip())
        r = 训练(前N=前N, 分钟=参.分钟, 输出目录=Path(参.出),
                数据路=Path(参.数据), 轮上限=参.轮上限, 早停轮=参.早停轮,
                通道=参.通道, 膨胀=膨胀, 学习率=参.学习率,
                上一帧数=参.上一帧数, 自条件p=参.自条件p, 迭代训练=参.迭代训练,
                初始化=参.初始化, 自条件ramp=参.自条件ramp, 自条件p最大=参.自条件p最大,
                周期保存=参.周期保存, 推理训练=参.推理训练, 推理阻尼=参.推理阻尼,
                推理前轮权=参.推理前轮权,
                多尺度=[int(x) for x in str(参.多尺度).split(",") if str(x).strip()],
                多尺度权=参.多尺度权)
        print(json.dumps(dict(参数数=r["参数数"], 训练信息=r["训练信息"]),
                         ensure_ascii=False, indent=1, default=_json值))
        return
    if 参.命令 == "合成":
        模型 = 加载(参.ckpt)
        mel = 合成mel(模型, 参.文本, 每音节帧数=参.每音节帧数)
        print(f"mel {mel.shape} {mel.dtype} 值域 [{mel.min():.3f}, {mel.max():.3f}]")
        if 参.npz:
            np.savez(参.npz, mel=mel)
            print(f"已写 {参.npz}")


if __name__ == "__main__":
    try:
        _主()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)
