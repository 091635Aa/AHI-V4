# -*- coding: utf-8 -*-
"""meeting67 · P2-3：F0/浊音头（冻结 v13 主干，只训 2 层 MLP 双头；≤64k 参数）。

口径（预注册 `11` §2.2）：
  - 主干 = v13 `模型_v13_K4_320/ckpt_e0150.pt`（或指定 ckpt）**冻结**；只读重组编码器前向
    （直接复用只读模块 `模型_学说话.说话模型` 的子模块：输入投影/输入范数/残差块/末范数；
    **不改任何既有文件**）。
  - 头 = MLP 320→160→64→2（logF0 头 | voicing 头），参数量 ≤64k；输入 = 编码器末层隐藏 h [T,320]。
  - 训练上下文 = **v13 推理同款**上下文（3/4/7 轮迭代 + 阻尼，与 A2 声码器同一预测 mel），
    即头在「模型会看到的隐藏」上学，而不是教师强制真 mel 上下文（避免训练/推理分布不一致）。
  - 损失：logF0 Huber（仅浊音帧）+ voicing BCE（pos_weight=清/浊比，类别加权）；λ 各 1.0。
  - 数据：`meeting66/产物/数据_对齐asr`（167 条）的内部划分 train=159 / val=8（**不碰 S1–S12/S13–S18**）。
  - 目标 F0/浊音：`meeting67/产物/P2/f0/<id>.npz`（离线 pyworld 提取，仅训练信号）。
  - 后处理（确定性）：voicing<0.5 ⇒ F0=0；F0 夹到 [60,500] Hz；10 ms 帧对齐。

产物：`meeting67/产物/P2/头/头_<标签>.pt`、`头质量_<标签>.json`；日志 `meeting67/团队/P2_实现/`。

用法：
  & <venv python> meeting67/实施/头_F0浊音.py --ckpt e0150 --迭代 4 --阻尼 0.5 --标签 e0150_4轮05
  & <venv python> meeting67/实施/头_F0浊音.py --ckpt e0100 --迭代 7 --阻尼 0.5 --标签 e0100_7轮05
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
M67 = HERE.parent
AHI4 = M67.parent
M66 = AHI4 / "meeting66"
M66实施 = M66 / "实施"
对齐目录 = M66 / "产物" / "数据_对齐asr"
样本目录 = 对齐目录 / "样本"
META = 对齐目录 / "meta.json"
F0目录 = M67 / "产物" / "P2" / "f0"
头目录 = M67 / "产物" / "P2" / "头"
日志目录 = M67 / "团队" / "P2_实现"
模型目录 = M66 / "产物" / "模型_v13_K4_320"

SR = 22050
帧移ms = 10.0
F0下限 = 60.0
F0上限 = 500.0
头隐藏 = (160, 64)          # ≤64k 参数：320*160+160*64+64*2 ≈ 61.8k
Huberδ = 1.0
训练步数 = 3000             # GPU 上 2 层 MLP，约 1–2 min（远低于预注册 10–20 min 上限）
批窗 = 8
窗长 = 256
学习率 = 1e-3
早停耐心 = 6                # 每 250 步验证一次
随机种子 = 67021

_模块 = {}


def _按路径载(名: str, 路径: Path):
    if 名 in _模块:
        return _模块[名]
    spec = importlib.util.spec_from_file_location(名, str(路径))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _模块[名] = mod
    return mod


def 模型模块():
    return _按路径载("模型_学说话_只读", M66实施 / "模型_学说话.py")


def _sha256文件(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for 块 in iter(lambda: f.read(1 << 20), b""):
            h.update(块)
    return h.hexdigest()


def _json可写(o):
    if isinstance(o, dict):
        return {str(k): _json可写(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json可写(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_, torch.Tensor)):
        return o.item() if isinstance(o, torch.Tensor) else bool(o)
    return o


def ckpt路径(名: str) -> Path:
    p = Path(名)
    if p.exists():
        return p
    名 = str(名)
    候 = 模型目录 / (名 if 名.startswith("ckpt_") else ("ckpt_%s.pt" % 名))
    if not 候.suffix:
        候 = 候.with_suffix(".pt")
    if not 候.exists():
        raise FileNotFoundError("ckpt 不存在：%s（候=%s）" % (名, 候))
    return 候


# ════════════════════════════════════════════════════════════════════════
# 1. 头（2 层 MLP 双头）
# ════════════════════════════════════════════════════════════════════════
class F0浊音头(nn.Module):
    """编码器隐藏 h [B,T,320] → logF0 [B,T] + voicing logit [B,T]。"""

    def __init__(self, 输入维: int = 320, 隐藏: tuple = 头隐藏):
        super().__init__()
        self.输入维 = int(输入维)
        self.配置 = dict(输入维=self.输入维, 隐藏=list(隐藏), 结构="MLP 320→160→64→2（双头）")
        self.注册均值 = nn.Parameter(torch.zeros(输入维), requires_grad=False)
        self.注册标准差 = nn.Parameter(torch.ones(输入维), requires_grad=False)
        self.第一 = nn.Linear(输入维, int(隐藏[0]))
        self.第二 = nn.Linear(int(隐藏[0]), int(隐藏[1]))
        self.logf0头 = nn.Linear(int(隐藏[1]), 1)
        self.voicing头 = nn.Linear(int(隐藏[1]), 1)

    def forward(self, h):
        z = (h - self.注册均值) / self.注册标准差.clamp_min(1e-4)
        z = F.gelu(self.第一(z))
        z = F.gelu(self.第二(z))
        return self.logf0头(z).squeeze(-1), self.voicing头(z).squeeze(-1)

    def 参数数(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))


# ════════════════════════════════════════════════════════════════════════
# 2. 只读重组编码器前向（复用 ckpt 权重，不改既有件）
# ════════════════════════════════════════════════════════════════════════
def 载主干(ckpt路: Path, 设备: str = "cuda"):
    模 = 模型模块().加载(ckpt路, 设备=设备)
    for p in 模.parameters():
        p.requires_grad_(False)
    模.eval()
    return 模


def _编码(模, 声, 韵, 调, 特征, 上一):
    """说话模型.forward 的前半段（到末范数+GELU），返回 (h 编码器隐藏, x 输入投影)。"""
    e = torch.cat([模.声母嵌入(声), 模.韵母嵌入(韵), 模.调嵌入(调), 特征], dim=-1)
    if 模.上一帧数 > 0:
        if 上一 is None:
            上一 = torch.zeros(*(list(e.shape[:-1]) + [模.上一帧数 * 模.配置["输出维"]]),
                              device=e.device, dtype=e.dtype)
        e = torch.cat([e, 上一], dim=-1)
    x = 模.输入范数(模.输入投影(e))                     # [B,T,320]
    h = x.transpose(1, 2)
    for 块 in 模.块s:
        h = 块(h)
    h = 模.末范数(h.transpose(1, 2)).transpose(1, 2)
    h = F.gelu(h)
    return h.transpose(1, 2), x                          # [B,T,320] 编码器隐藏, 输入投影


def _出(模, h, x):
    return (模.输出卷积(h.transpose(1, 2)) + 模.旁路卷积(x.transpose(1, 2))).transpose(1, 2)


def _上帧(模, x):
    """复用只读模块的 `_上帧张量(mel, K)`（K=模.上一帧数）；不复制实现。"""
    return 模型模块()._上帧张量(x, 模.上一帧数)


def 样本张量(模, z: dict, 设备):
    """对齐 npz → 帧级 (声,韵,调,特征) 张量 [1,T,*]。"""
    模m = 模型模块()
    tokens = np.stack([np.asarray(z["声母"], np.int64),
                       np.asarray(z["韵母"], np.int64),
                       np.asarray(z["调"], np.int64)], axis=1)
    帧数 = np.asarray(z["音节帧数"], np.int64)
    声, 韵, 调, 特征 = 模m.展开帧(tokens, 帧数)
    张 = lambda a: torch.from_numpy(np.asarray(a)).unsqueeze(0).to(设备)  # noqa: E731
    return (张(声).long(), 张(韵).long(), 张(调).long(), 张(特征).float())


def 预计算隐藏(模, ids, 迭代: int, 阻尼: float, 设备: str = "cuda", 打印=True):
    """对每条样本：按 v13 推理同款迭代（预测上下文）跑编码器，取最后一趟隐藏 h [T,320]。

    → dict(id → dict(h=float16 np[T,320], logf0=float32 np[T], voicing=int8 np[T], 帧数))
    """
    出 = {}
    t0 = time.time()
    with torch.no_grad():
        for 序, id_ in enumerate(ids):
            z = np.load(str(样本目录 / ("%04d.npz" % int(id_))))
            T = int(z["mel"].shape[0])
            f0z = np.load(str(F0目录 / ("%04d.npz" % int(id_))))
            logf0 = np.asarray(f0z["logf0"], np.float32).reshape(-1)
            voi = np.asarray(f0z["voicing"], np.int8).reshape(-1)
            if logf0.size != T:
                raise ValueError("%04d F0 帧数 %d ≠ mel 帧数 %d" % (id_, logf0.size, T))
            声, 韵, 调, 特征 = 样本张量(模, z, 设备)
            prev = None
            平滑 = None
            h = x = None
            for _ in range(max(int(迭代), 1)):
                h, x = _编码(模, 声, 韵, 调, 特征, prev)
                出i = _出(模, h, x)
                平滑 = 出i if 平滑 is None else (float(阻尼) * 出i + (1.0 - float(阻尼)) * 平滑)
                prev = _上帧(模, 平滑)
            出[int(id_)] = dict(h=h[0].detach().to(torch.float16).cpu().numpy(),
                               logf0=logf0, voicing=voi, 帧数=T,
                               mel_sha256=hashlib.sha256(
                                   np.ascontiguousarray(np.asarray(z["mel"], np.float32)).tobytes()
                               ).hexdigest())
            if 打印 and (序 + 1) % 40 == 0:
                print("[隐藏] %d/%d  %.1fs" % (序 + 1, len(ids), time.time() - t0), flush=True)
    if 打印:
        print("[隐藏] 完成 %d 条，%.1fs（迭代=%d 阻尼=%.2f）"
              % (len(出), time.time() - t0, 迭代, 阻尼), flush=True)
    return 出


# ════════════════════════════════════════════════════════════════════════
# 3. 训练 / 评估
# ════════════════════════════════════════════════════════════════════════
def _损失(头, h, logf0, voi):
    p_logf0, p_voi = 头(h)
    浊 = voi > 0.5
    损_f0 = F.huber_loss(p_logf0[浊], logf0[浊], delta=Huberδ) if bool(浊.any()) else p_logf0.sum() * 0.0
    损_voi = F.binary_cross_entropy_with_logits(p_voi, voi)
    return 损_f0 + 损_voi, dict(f0=float(损_f0.detach()), voi=float(损_voi.detach()))


def _采样窗(池: list, rng: np.random.Generator, 批: int = 批窗, 长: int = 窗长):
    hs, ls, vs = [], [], []
    for _ in range(int(批)):
        id_ = int(rng.integers(len(池)))
        项 = 池[id_]
        T = 项["帧数"]
        长i = min(int(长), T)
        起 = int(rng.integers(0, max(T - 长i, 1)))
        hs.append(torch.from_numpy(项["h"][起:起 + 长i].astype(np.float32)))
        ls.append(torch.from_numpy(项["logf0"][起:起 + 长i].astype(np.float32)))
        vs.append(torch.from_numpy(项["voicing"][起:起 + 长i].astype(np.float32)))
    return hs, ls, vs


def _填充批(hs, ls, vs, 设备):
    长 = max(h.shape[0] for h in hs)
    B = len(hs)
    H = torch.zeros((B, 长, hs[0].shape[-1]), dtype=torch.float32)
    L = torch.zeros((B, 长), dtype=torch.float32)
    V = torch.zeros((B, 长), dtype=torch.float32)
    M = torch.zeros((B, 长), dtype=torch.float32)
    for i, (h, l, v) in enumerate(zip(hs, ls, vs)):
        n = h.shape[0]
        H[i, :n] = h
        L[i, :n] = l
        V[i, :n] = v
        M[i, :n] = 1.0
    return H.to(设备), L.to(设备), V.to(设备), M.to(设备)


@torch.no_grad()
def 头输出(头, 池, 设备: str = "cuda"):
    """池 → 逐样本 (pred_logf0, pred_voi_prob, logf0, voi) numpy 列表。"""
    头.eval()
    出 = []
    for 项 in 池:
        h = torch.from_numpy(项["h"].astype(np.float32)).unsqueeze(0).to(设备)
        p_l, p_v = 头(h)
        出.append(dict(pred_logf0=p_l[0].float().cpu().numpy(),
                        pred_voi=torch.sigmoid(p_v[0]).float().cpu().numpy(),
                        logf0=项["logf0"], voicing=项["voicing"], id=项["id"]))
    return 出


def 头质量(头, 池, 设备: str = "cuda", 打印=True) -> dict:
    """A3 头质量：F0 Pearson（浊音帧）/ 浊音准确率 / 清音 F1（预注册门槛 0.80/0.90/0.80）。"""
    出s = 头输出(头, 池, 设备)
    pl, tl, pv, tv = [], [], [], []
    for r in 出s:
        pl.append(r["pred_logf0"])
        tl.append(r["logf0"])
        pv.append(r["pred_voi"])
        tv.append((r["voicing"] > 0.5).astype(np.float32))
    pl = np.concatenate(pl); tl = np.concatenate(tl)
    pv = np.concatenate(pv); tv = np.concatenate(tv)
    pm = (pv >= 0.5).astype(np.float32)
    t_浊 = tv > 0.5
    双浊 = t_浊 & (pm > 0.5)
    def _pearson(a, b):
        a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
        if a.size < 2 or a.std() < 1e-9 or b.std() < 1e-9:
            return None
        return round(float(np.corrcoef(a, b)[0, 1]), 4)
    相关_双浊 = _pearson(pl[双浊], tl[双浊]) if int(双浊.sum()) >= 2 else None
    相关_真浊 = _pearson(pl[t_浊], tl[t_浊]) if int(t_浊.sum()) >= 2 else None   # 辅助口径（预测可能含 0）
    准 = float((pm == tv).mean()) if tv.size else None
    # 清音类（voicing=0）为阳性：F1
    def _f1(真正, 测正):
        tp = float(np.sum(真正 & 测正)); fp = float(np.sum(~真正 & 测正)); fn = float(np.sum(真正 & ~测正))
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        return (round(2 * p * r / (p + r), 4) if p + r > 0 else 0.0), round(p, 4), round(r, 4)
    清f1, 清p, 清r = _f1(tv < 0.5, pm < 0.5)
    浊f1, 浊p, 浊r = _f1(tv > 0.5, pm > 0.5)
    报 = dict(
        口径="内部留出 8 条（未见句）；预测上下文隐藏（v13 推理同款迭代）；逐帧汇总",
        帧数=int(tv.size), 真浊音帧=int(t_浊.sum()), 真清音帧=int((~t_浊).sum()),
        预测浊音帧=int((pm > 0.5).sum()),
        F0相关_浊音帧_双浊=相关_双浊,
        F0相关_浊音帧_真浊=相关_真浊,
        浊音准确率=round(准, 4) if 准 is not None else None,
        清音F1=清f1, 清音精确率=清p, 清音召回率=清r,
        浊音F1=浊f1, 浊音精确率=浊p, 浊音召回率=浊r,
        门槛=dict(F0相关=0.80, 浊音准确率=0.90, 清音F1=0.80),
    )
    报["达标"] = dict(
        F0相关=bool(相关_双浊 is not None and 相关_双浊 >= 0.80),
        浊音准确率=bool(准 is not None and 准 >= 0.90),
        清音F1=bool(清f1 >= 0.80),
    )
    报["全达标"] = bool(all(报["达标"].values()))
    逐条 = []
    for r in 出s:
        t_ = r["voicing"] > 0.5
        p_ = r["pred_voi"] >= 0.5
        逐条.append(dict(id=int(r["id"]), 帧数=int(t_.size),
                        浊音帧=int(t_.sum()), 预测浊音帧=int(p_.sum()),
                        F0相关_双浊=_pearson(r["pred_logf0"][t_ & p_], r["logf0"][t_ & p_]),
                        准确率=round(float((p_ == t_).mean()), 4)))
    报["逐条"] = 逐条
    if 打印:
        print("[头质量] F0相关(双浊)=%s 准确率=%s 清音F1=%s ⇒ 全达标=%s"
              % (报["F0相关_浊音帧_双浊"], 报["浊音准确率"], 报["清音F1"], 报["全达标"]), flush=True)
    return 报


def 训练头(模, 训练池: list, 验证池: list, 设备: str = "cuda",
          步数: int = 训练步数, 打印=True) -> tuple:
    torch.manual_seed(随机种子)
    rng = np.random.default_rng(随机种子)
    头 = F0浊音头(输入维=int(训练池[0]["h"].shape[-1])).to(设备)
    参 = 头.参数数()
    if 参 > 64000:
        raise RuntimeError("头参数 %d > 64000（违反预注册 ≤64k）" % 参)
    # 输入标准化统计（训练集隐藏）
    和 = np.zeros(头.输入维, np.float64); 平方 = np.zeros(头.输入维, np.float64); n = 0
    for 项 in 训练池:
        h = 项["h"].astype(np.float32)
        和 += h.sum(0, dtype=np.float64); 平方 += (h.astype(np.float64) ** 2).sum(0); n += h.shape[0]
    均 = 和 / max(n, 1); 方差 = np.maximum(平方 / max(n, 1) - 均 ** 2, 1e-8)
    with torch.no_grad():
        头.注册均值.copy_(torch.from_numpy(均.astype(np.float32)).to(设备))
        头.注册标准差.copy_(torch.from_numpy(np.sqrt(方差).astype(np.float32)).to(设备))
    优 = torch.optim.Adam([p for p in 头.parameters() if p.requires_grad], lr=学习率)
    训练帧数 = int(sum(x["帧数"] for x in 训练池))
    正浊 = float(sum(int(x["voicing"].sum()) for x in 训练池))
    正清 = float(训练帧数 - 正浊)
    pos_weight = float(np.clip(正清 / max(正浊, 1.0), 0.5, 3.0))
    if 打印:
        print("[训练] 头参数=%d 训练帧=%d 浊音比=%.3f pos_weight=%.3f 步数=%d"
              % (参, 训练帧数, 正浊 / max(训练帧数, 1), pos_weight, 步数), flush=True)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=设备))
    最好 = dict(验证损失=1e9, 步=-1, 指标=None)
    历史 = []
    t0 = time.time()
    for 步 in range(1, int(步数) + 1):
        头.train()
        hs, ls, vs = _采样窗(训练池, rng)
        H, L, V, M = _填充批(hs, ls, vs, 设备)
        p_l, p_v = 头(H)
        浊 = (V > 0.5) & (M > 0.5)
        损_f0 = F.huber_loss(p_l[浊], L[浊], delta=Huberδ) if bool(浊.any()) else p_l.sum() * 0.0
        # 掩码 BCE：只算有效帧
        损_voi = (F.binary_cross_entropy_with_logits(p_v, V, reduction="none") * M).sum() / M.sum().clamp_min(1.0)
        损 = 损_f0 + 损_voi
        优.zero_grad(set_to_none=True)
        损.backward()
        优.step()
        if 步 % 250 == 0 or 步 == 步数:
            with torch.no_grad():
                Hv, Lv, Vv, Mv = _填充批(*zip(*[(torch.from_numpy(x["h"].astype(np.float32)),
                                               torch.from_numpy(x["logf0"].astype(np.float32)),
                                               torch.from_numpy(x["voicing"].astype(np.float32)))
                                              for x in 验证池]), 设备)
                pv_l, pv_v = 头(Hv)
                浊v = (Vv > 0.5) & (Mv > 0.5)
                验证损 = float((F.huber_loss(pv_l[浊v], Lv[浊v], delta=Huberδ)
                               if bool(浊v.any()) else pv_l.sum() * 0.0).item())
                验证损 += float(((F.binary_cross_entropy_with_logits(pv_v, Vv, reduction="none") * Mv).sum()
                               / Mv.sum().clamp_min(1.0)).item())
            历史.append(dict(步=步, 训练损=round(float(损.item()), 5),
                             f0损=round(float(损_f0.item()), 5), voi损=round(float(损_voi.item()), 5),
                            验证损=round(验证损, 5), 耗时秒=round(time.time() - t0, 1)))
            if 验证损 < 最好["验证损失"]:
                最好 = dict(验证损失=round(验证损, 5), 步=步)
                if 打印:
                    print("[训练] 步%4d 训练损=%.4f 验证损=%.4f（最好）"
                          % (步, float(损.item()), 验证损), flush=True)
            elif 步 - 最好["步"] >= 早停耐心 * 250:
                if 打印:
                    print("[训练] 早停 @步%d（最好步=%d）" % (步, 最好["步"]), flush=True)
                break
    耗 = time.time() - t0
    return 头, dict(参数数=参, 步数=int(历史[-1]["步"]) if 历史 else 0, 最好=最好,
                   历史=历史, 耗时秒=round(耗, 1), pos_weight=round(pos_weight, 4),
                   训练帧数=训练帧数, lr=学习率, 批窗=批窗, 窗长=窗长, Huberδ=Huberδ)


# ════════════════════════════════════════════════════════════════════════
# 4. 推理接口（供 评测_P2.py 用；不 import 提取工具，F0 全来自本头）
# ════════════════════════════════════════════════════════════════════════
def 载头(头路径: Path | str, 设备: str = "cuda"):
    ck = torch.load(str(头路径), map_location="cpu", weights_only=False)
    配置 = dict(ck["头配置"])
    配置.pop("结构", None)                       # 说明性字段，不进构造
    头 = F0浊音头(**配置)
    头.load_state_dict(ck["state_dict"])
    头.to(设备).eval()
    return 头, ck


def 预测F0(模, 头, 文本: str, 每音节帧数: int = 24, 迭代: int = 3,
           阻尼: float = 1.0, 设备: str = "cuda") -> dict:
    """文本 → v13 预测 mel（否定量化回 mel 尺度）+ 头预测 F0/浊音（与 A2 声码器同 mel）。"""
    模m = 模型模块()
    头.eval()
    with torch.no_grad():
        # ① 与 合成mel 完全同款：迭代 + 阻尼 + 上一帧自条件
        tokens, 有效, 跳过 = 模m.文本到token(模.词表, str(文本))
        if len(tokens) == 0:
            raise ValueError("文本无可合成音节：%r" % (文本,))
        帧数 = np.full(len(tokens), int(每音节帧数), dtype=np.int64)
        声a, 韵a, 调a, 特征a = 模m.展开帧(tokens, 帧数)
        张 = lambda a: torch.from_numpy(np.asarray(a)).unsqueeze(0).to(设备)  # noqa: E731
        声, 韵, 调, 特征 = 张(声a).long(), 张(韵a).long(), 张(调a).long(), 张(特征a).float()
        prev = None; 平滑 = None; h = x = None
        for _ in range(max(int(迭代), 1)):
            h, x = _编码(模, 声, 韵, 调, 特征, prev)
            出i = _出(模, h, x)                     # 与 合成mel 一致：mel 输出取最后一趟原始输出
            平滑 = 出i if 平滑 is None else (float(阻尼) * 出i + (1.0 - float(阻尼)) * 平滑)
            prev = _上帧(模, 平滑)
        p_logf0, p_voi = 头(h)
        p_logf0 = p_logf0[0].float().cpu().numpy()
        p_voi = torch.sigmoid(p_voi[0]).float().cpu().numpy()
        mel = 出i[0].float().cpu().numpy()          # ★ 与 合成mel 的 `出[0]` 逐字同口径
    m = np.asarray(模.mel均值, np.float32).reshape(-1)
    sd = np.asarray(模.mel标准差, np.float32).reshape(-1)
    mel = mel * (sd[0] if sd.size == 1 else sd.reshape(1, -1)) + (m[0] if m.size == 1 else m.reshape(1, -1))
    voi = (p_voi >= 0.5).astype(np.int8)
    f0 = np.where(voi > 0, np.clip(np.exp(p_logf0), F0下限, F0上限), 0.0).astype(np.float32)
    return dict(mel=np.asarray(mel, np.float32), f0=f0, voicing=voi,
                logf0=np.where(voi > 0, p_logf0, 0.0).astype(np.float32),
                浊音比=round(float(voi.mean()) if voi.size else 0.0, 4),
                有效音节数=int(len(有效)), 跳过=跳过,
                推理设置=dict(迭代=int(迭代), 阻尼=float(阻尼), 每音节帧数=int(每音节帧数)))


# ════════════════════════════════════════════════════════════════════════
# 5. 主流程
# ════════════════════════════════════════════════════════════════════════
def 主(argv=None) -> int:
    ap = argparse.ArgumentParser(description="meeting67 P2-3 F0/浊音头（冻结 v13）")
    ap.add_argument("--ckpt", default="ckpt_e0150.pt")
    ap.add_argument("--迭代", type=int, default=4)
    ap.add_argument("--阻尼", type=float, default=0.5)
    ap.add_argument("--标签", default="e0150_4轮05")
    ap.add_argument("--步数", type=int, default=训练步数)
    ap.add_argument("--设备", default="cuda")
    args = ap.parse_args(argv)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    设备 = args.设备 if (args.设备 != "cuda" or torch.cuda.is_available()) else "cpu"
    头目录.mkdir(parents=True, exist_ok=True)
    日志目录.mkdir(parents=True, exist_ok=True)
    if not F0目录.exists():
        raise FileNotFoundError("缺 F0 产物：%s（先跑 提取_F0浊音.py）" % F0目录)
    ck = ckpt路径(args.ckpt)
    meta = json.loads(META.read_text(encoding="utf-8"))
    划分 = meta["划分"]
    训练ids = [int(i) for i in 划分["train"]]
    验证ids = [int(i) for i in (划分.get("val") or 划分.get("验证"))]
    缺 = [i for i in 训练ids + 验证ids if not (F0目录 / ("%04d.npz" % i)).exists()]
    if 缺:
        raise RuntimeError("F0 产物缺 %d 条（如 %s），先完成提取" % (len(缺), 缺[:5]))
    if set(训练ids) & {75}:
        pass
    print("[头] ckpt=%s 迭代=%d 阻尼=%.2f 训练=%d 验证=%d 设备=%s"
          % (ck, args.迭代, args.阻尼, len(训练ids), len(验证ids), 设备), flush=True)

    t0 = time.time()
    模 = 载主干(ck, 设备)
    # 隐藏预计算（训练/验证都走预测上下文，与 A2/A3 一致）
    训练隐 = 预计算隐藏(模, 训练ids, args.迭代, args.阻尼, 设备)
    验证隐 = 预计算隐藏(模, 验证ids, args.迭代, args.阻尼, 设备)
    训练池 = [dict(id=i, **训练隐[i]) for i in 训练ids]
    验证池 = [dict(id=i, **验证隐[i]) for i in 验证ids]
    头, 训练报 = 训练头(模, 训练池, 验证池, 设备=设备, 步数=args.步数)
    质量 = 头质量(头, 验证池, 设备)
    质量["训练"] = 训练报
    质量["验证ids"] = 验证ids

    头路 = 头目录 / ("头_%s.pt" % args.标签)
    torch.save(dict(state_dict=头.state_dict(), 头配置=头.配置,
                    模型ckpt=dict(路径=str(ck), sha256=_sha256文件(ck)),
                    推理设置=dict(迭代=int(args.迭代), 阻尼=float(args.阻尼)),
                    f0源=dict(目录=str(F0目录), 索引=str(F0目录 / "索引.json"),
                             sha256=(_sha256文件(F0目录 / "索引.json")
                                     if (F0目录 / "索引.json").exists() else None)),
                    训练信息=dict(标签=args.标签, 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
                                **{k: v for k, v in 训练报.items() if k != "历史"})),
               str(头路))
    质量.update(dict(工具="头_F0浊音.py", 版本="1.0", 标签=args.标签,
                     ckpt=dict(路径=str(ck), sha256=_sha256文件(ck)),
                     ckpt训练信息=dict(getattr(模, "训练信息", {}) or {}),
                     头=dict(路径=str(头路), sha256=_sha256文件(头路), 参数数=头.参数数()),
                     脚本sha256=_sha256文件(Path(__file__).resolve()),
                     推理设置=dict(迭代=int(args.迭代), 阻尼=float(args.阻尼)),
                     时间=time.strftime("%Y-%m-%d %H:%M:%S"),
                     耗时秒=round(time.time() - t0, 1)))
    (头目录 / ("头质量_%s.json" % args.标签)).write_text(
        json.dumps(_json可写(质量), ensure_ascii=False, indent=2), encoding="utf-8")
    print("[头] 产物：%s" % 头路, flush=True)
    print("[头] 质量：%s" % (头目录 / ("头质量_%s.json" % args.标签)), flush=True)
    print("[头] F0相关=%s 浊音准确率=%s 清音F1=%s 全达标=%s 耗时=%.1fs"
          % (质量["F0相关_浊音帧_双浊"], 质量["浊音准确率"], 质量["清音F1"],
             质量["全达标"], 质量["耗时秒"]), flush=True)
    return 0 if 质量["全达标"] else 2


if __name__ == "__main__":
    raise SystemExit(主())
