# -*- coding: utf-8 -*-
"""meeting71/实施/训练_P6.py —— P6 控制生成器（非 Transformer，从零训练）。

用法：
  & $py 训练_P6.py 训练 [轮数]
  & $py 训练_P6.py 推理
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M71 = HERE.parent
M70 = M71.parent / "meeting70"
M64 = M71.parent / "meeting64" / "实施"
产 = M71 / "产物" / "P6"
SR = 22050
声母表 = ["∅", "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s"]
韵母表 = ["a", "o", "e", "i", "u", "v", "er", "ii", "iii", "ai", "ei", "ao", "ou", "ia", "ie", "iao", "iou", "ua", "uo",
        "uai", "uei", "ve", "an", "en", "in", "uen", "vn", "ang", "eng", "ing", "ong", "iong", "ian", "uan", "uang",
        "van", "", "iang", "io"]
句表 = {
    "P6主句": "今天天气很好我听见了自己的声音我想再说一遍",
    "P6新句1": "今天的风很轻我慢慢说给你听",
    "P6新句2": "窗外的雨停了我该休息一会",
    "P6新句3": "你先别走我还想再说两句",
}
参数 = dict(OQ=0.62, OQ1=0.42, 噪声尺度=0.40, 去加重=0.97)


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


器 = 载("器官v2", M70 / "实施" / "器官v2.py")
量 = 载("量具_v9", M64 / "量具_v9.py")

import torch
import torch.nn as nn

设备 = "cuda" if torch.cuda.is_available() else "cpu"


class 控制生成器(nn.Module):
    def __init__(self, 通道=192, 层=4):
        super().__init__()
        self.e_sm = nn.Embedding(22, 32)
        self.e_ym = nn.Embedding(39, 32)
        self.e_tone = nn.Embedding(5, 32)
        输入 = 96 + 3
        self.ina = nn.Conv1d(输入, 通道, 5, padding=2)
        self.blocks = nn.ModuleList()
        for k in range(层):
            d = 2 ** k
            self.blocks.append(nn.Sequential(
                nn.Conv1d(通道, 通道, 5, padding=2 * d, dilation=d), nn.GELU(),
                nn.Conv1d(通道, 通道, 1), nn.GELU(),
            ))
        self.head = nn.Conv1d(通道, 28, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tok, rel, pos, logT):
        # tok (B,T,3) rel/pos/logT (B,T)
        e = torch.cat([self.e_sm(tok[..., 0]), self.e_ym(tok[..., 1]), self.e_tone(tok[..., 2])], -1)
        x = torch.cat([e, rel[..., None], pos[..., None], logT[..., None]], -1)
        x = x.transpose(1, 2)
        h = self.ina(x)
        for b in self.blocks:
            h = h + b(h)
        y = self.head(h).transpose(1, 2)          # (B,T,28)
        A = y[..., :25]
        f0 = y[..., 25]
        logE = y[..., 26]
        NR = torch.sigmoid(y[..., 27])
        return A, f0, logE, NR


def 构造帧输入(toks, durs):
    T = int(np.sum(durs))
    tokf = np.repeat(toks, durs, axis=0)[:T]
    rel = np.concatenate([np.linspace(0, 1, d, endpoint=True) for d in durs if d > 0])[:T]
    pos = np.linspace(0, 1, T, endpoint=True)
    logT = np.full(T, np.log(max(T, 1)))
    return tokf, rel, pos, logT


def 载数据():
    with open(产 / "数据.pkl", "rb") as f:
        return pickle.load(f)


def 训练(轮数=300):
    产.mkdir(parents=True, exist_ok=True)
    d = 载数据()
    件 = d["件"]
    sids = sorted(x["sid"] for x in 件)
    val_sids = set(sids[-3:])
    train = [x for x in 件 if x["sid"] not in val_sids]
    val = [x for x in 件 if x["sid"] in val_sids]
    print("train", len(train), "val", len(val), flush=True)
    模型 = 控制生成器().to(设备)
    opt = torch.optim.AdamW(模型.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=轮数)
    # 预构造
    def 预(件表):
        out = []
        for x in 件表:
            tok, rel, pos, logT = 构造帧输入(x["toks"], x["durs"])
            T = min(len(tok), len(x["r"]))
            out.append(dict(tok=tok[:T], rel=rel[:T], pos=pos[:T], logT=logT[:T],
                            A=np.pad(x["r"][:T] * 0, 0), r=x["r"][:T], f0=x["f0"][:T],
                            logE=x["logE"][:T], NR=x["NR"][:T]))
        return out
    tr = 预(train); va = 预(val)
    最佳 = 1e9
    for ep in range(1, 轮数 + 1):
        模型.train()
        批 = []
        loss_sum = 0.0
        for _ in range(24):
            x = tr[np.random.randint(0, len(tr))]
            T = len(x["f0"]); W = 800
            if T <= W:
                s = 0
            else:
                s = np.random.randint(0, T - W)
            e = min(T, s + W)
            idx = np.arange(s, e)
            # 目标 A：由 r 反推不可直接；改用 r 作为回归目标（预测 25 维 r）
            r = x["r"][idx]
            v = (x["f0"][idx] > 0).astype(np.float32)
            f0s = np.where(v > 0, 12 * np.log2(np.maximum(x["f0"][idx], 1e-3) / 300.0), 0.0)
            批.append((x["tok"][idx], x["rel"][idx], x["pos"][idx], x["logT"][idx],
                      r, f0s, v, x["logE"][idx], x["NR"][idx]))
        # 拼批（等长 W）
        W = min(len(b[0]) for b in 批)
        批 = [tuple(np.asarray(a)[:W] if isinstance(a, np.ndarray) else a for a in b) for b in 批]
        def T_(arr):
            return torch.as_tensor(arr, dtype=torch.float32, device=设备)
        tok = torch.as_tensor(np.stack([b[0] for b in 批]).astype(np.int64), device=设备)
        rel = T_(np.stack([b[1] for b in 批])); pos = T_(np.stack([b[2] for b in 批]))
        logT = T_(np.stack([b[3] for b in 批]))
        r = T_(np.stack([b[4] for b in 批]))
        f0 = T_(np.stack([b[5] for b in 批])); v = T_(np.stack([b[6] for b in 批]))
        lE = T_(np.stack([b[7] for b in 批])); NR = T_(np.stack([b[8] for b in 批]))
        A, pf0, plE, pNR = 模型(tok, rel, pos, logT)
        l_r = ((A - r) ** 2).mean()
        l_f = (((pf0 - f0) ** 2) * v).sum() / (v.sum() + 1e-6)
        l_e = ((plE - lE) ** 2).mean()
        l_n = ((pNR - NR) ** 2).mean()
        loss = l_r + 1.0 * l_f + 0.5 * l_e + 0.5 * l_n
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(模型.parameters(), 3.0); opt.step()
        loss_sum += float(loss.item())
        sched.step()
        if ep % 10 == 0 and _ == 23:
            # val
            模型.eval()
            with torch.no_grad():
                vl = 0.0
                for x in va:
                    T = len(x["f0"]); Wv = min(800, T)
                    if T == 0:
                        continue
                    s = max(0, (T - Wv) // 2)
                    idx = np.arange(s, s + Wv)
                    tokv = torch.as_tensor(x["tok"][idx][None].astype(np.int64), device=设备)
                    relv = torch.as_tensor(x["rel"][idx][None], dtype=torch.float32, device=设备)
                    posv = torch.as_tensor(x["pos"][idx][None], dtype=torch.float32, device=设备)
                    logTv = torch.as_tensor(x["logT"][idx][None], dtype=torch.float32, device=设备)
                    A2, f2, e2, n2 = 模型(tokv, relv, posv, logTv)
                    rv = torch.as_tensor(x["r"][idx][None], dtype=torch.float32, device=设备)
                    vv = torch.as_tensor((x["f0"][idx] > 0).astype(np.float32)[None], dtype=torch.float32, device=设备)
                    f0v = torch.as_tensor(np.where(x["f0"][idx] > 0, 12 * np.log2(np.maximum(x["f0"][idx], 1e-3) / 300.0), 0.0)[None], dtype=torch.float32, device=设备)
                    lEv = torch.as_tensor(x["logE"][idx][None], dtype=torch.float32, device=设备)
                    nv = torch.as_tensor(x["NR"][idx][None], dtype=torch.float32, device=设备)
                    vl += float((((A2 - rv) ** 2).mean() + (((f2 - f0v) ** 2) * vv).sum() / (vv.sum() + 1e-6)
                                 + 0.5 * ((e2 - lEv) ** 2).mean() + 0.5 * ((n2 - nv) ** 2).mean()).item())
                vl /= max(1, len(va))
            print("| ep", ep, "loss %.5f" % (loss_sum / 24), "val %.5f" % vl, flush=True)
            if vl < 最佳:
                最佳 = vl
                torch.save({"模型": 模型.state_dict(), "val": vl}, 产 / "模型.pt")
    模型.eval()
    torch.save({"模型": 模型.state_dict(), "val": 最佳}, 产 / "模型.pt")
    print("训练完成 最佳val", 最佳, flush=True)
    return 0


def 键_of(文本, 字表):
    from pypinyin import lazy_pinyin, Style
    字s = [ch for ch in 文本 if "\u4e00" <= ch <= "\u9fff"]
    out = []
    for ch in 字s:
        if ch in 字表:
            out.append(字表[ch]); continue
        ys = lazy_pinyin(ch, style=Style.TONE3, neutral_tone_with_five=True)
        if not ys:
            continue
        s = ys[0]; tone = 5
        if s and s[-1].isdigit():
            tone = int(s[-1]); s = s[:-1]
        sm = "∅"; rest = s
        for ini in ("zh", "ch", "sh"):
            if s.startswith(ini):
                sm, rest = ini, s[len(ini):]; break
        if sm == "∅":
            for ini in "bpmfdtnlgkhjqxrzcs":
                if s.startswith(ini):
                    sm, rest = ini, s[1:]; break
        if sm in 声母表 and rest in 韵母表:
            out.append((声母表.index(sm), 韵母表.index(rest), max(0, min(4, tone - 1))))
    return out


def 自相关_由(预测A, 阶=24, n_fft=1024):
    out = np.empty((len(预测A), 阶 + 1), np.float64)
    for i, a in enumerate(预测A):
        aa = np.concatenate([[1.0], a[1:]])
        H = np.fft.rfft(aa, n=n_fft)
        P = 1.0 / (np.abs(H) ** 2 + 1e-9)
        r = np.fft.irfft(P, n=n_fft)[: 阶 + 1]
        out[i] = r / (r[0] + 1e-12)
    return out


def levinson(r, p):
    a = np.zeros(p + 1); a[0] = 1.0
    e = float(r[0]) + 1e-12
    for i in range(1, p + 1):
        acc = float(r[i]) + (float(np.dot(a[1:i], r[i - 1:0:-1])) if i > 1 else 0.0)
        k = -acc / e
        if i > 1:
            a[1:i] = a[1:i] + k * a[i - 1:0:-1]
        a[i] = k
        e *= (1 - k * k)
        if e <= 1e-12:
            break
    return a


def 写(p, a):
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())


def 推理(每音节=24):
    d = 载数据()
    字表 = d["字表"]
    ck = torch.load(产 / "模型.pt", map_location=设备)
    模型 = 控制生成器().to(设备)
    模型.load_state_dict(ck["模型"]); 模型.eval()
    引 = None
    报 = dict(工具="训练_P6.py 推理", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 项=[])
    with torch.no_grad():
        for 名, 文 in 句表.items():
            keys = 键_of(文, 字表)
            if not keys:
                报["项"].append(dict(样本=名, 失败="无键")); continue
            toks = np.array(keys, np.int64)
            durs = np.full(len(toks), 每音节, np.int64)
            tok, rel, pos, logT = 构造帧输入(toks, durs)
            tokv = torch.as_tensor(tok[None].astype(np.int64), device=设备)
            relv = torch.as_tensor(rel[None], dtype=torch.float32, device=设备)
            posv = torch.as_tensor(pos[None], dtype=torch.float32, device=设备)
            logTv = torch.as_tensor(logT[None], dtype=torch.float32, device=设备)
            A, pf0, plE, pNR = 模型(tokv, relv, posv, logTv)
            A = A[0].cpu().numpy(); pf0 = pf0[0].cpu().numpy()
            plE = plE[0].cpu().numpy(); pNR = pNR[0].cpu().numpy()
            r = np.array(A, dtype=np.float64)
            r[:, 0] = 1.0
            r = np.clip(r, -0.999, 0.999)
            r5 = np.repeat(r, 2, axis=0)
            for i in range(1, len(r5)):
                r5[i] = 0.55 * r5[i] + 0.45 * r5[i - 1]
            A5 = np.stack([levinson(rr, 24) for rr in r5])
            f0 = 300.0 * (2.0 ** (np.repeat(np.clip(pf0, -12, 12), 2) / 12.0))
            f0s = np.copy(f0)
            for i in range(len(f0s)):
                lo = max(0, i - 3); hi = min(len(f0s), i + 4)
                f0s[i] = float(np.median(f0[lo:hi]))
            E = np.exp(np.repeat(np.clip(plE, -8, 0), 2)); E = np.convolve(E, np.ones(3) / 3, mode="same")
            NR = np.clip(np.repeat(pNR, 2), 0.02, 0.99)
            c = dict(T=len(f0s), f0=f0s, 浊=np.ones(len(f0s), bool), F=np.zeros((len(f0s), 4)),
                     B=np.zeros((len(f0s), 4)), E=np.maximum(E, 1e-5), NR=NR, A=A5)
            y = 器.合成(c, 模式="全极点", **参数)
            写(产 / f"{名}.wav", y)
            报["项"].append(dict(样本=名, 文本=文, 路径=str(产 / f"{名}.wav"),
                                秒=round(len(y) / SR, 3), 帧=len(f0s)))
            print("|", 名, round(len(y) / SR, 2), "s", flush=True)
    # 读数
    引 = 量.载引擎()
    from faster_whisper import WhisperModel
    import glob
    snap = glob.glob(str(Path.home() / ".cache" / "huggingface" / "hub" / "models--Systran--faster-whisper-medium" / "snapshots" / "*"))[0]
    wm = WhisperModel(snap, device="cpu", compute_type="int8")
    for x in 报["项"]:
        if "路径" not in x:
            continue
        q = 引["转写"](x["路径"])
        segs, _ = wm.transcribe(x["路径"], language="zh", beam_size=5)
        wh = "".join(y.text for y in segs).strip()
        x.update(转写=q, Whisper=wh, CER=round(量.CER(x["文本"], q), 4))
        print("|", x["样本"], "CER=%.3f" % x["CER"], "|", q[:30], flush=True)
    r0 = [x for x in 报["项"] if "CER" in x]
    报["均值CER"] = round(float(np.mean([x["CER"] for x in r0])), 4) if r0 else None
    (产 / "推理_读数.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print("均值", 报["均值CER"], flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) > 1 and sys.argv[1] == "推理":
        sys.exit(推理())
    sys.exit(训练(int(sys.argv[2]) if len(sys.argv) > 2 else 200))
