# -*- coding: utf-8 -*-
"""meeting67/团队/声学_物理建模/诊断_G0v0失败.py —— G0(v0 等价类) FAIL 的**纯 DSP 机制诊断**（不载 ASR）。

背景：Lead 用真实录音口径跑完 G0：真mel_GL 0.0585（护栏过）、P1 0.3693、**P2 0.4922**、P3 0.9739
      ⇒ **G0-FAIL（P2 > 0.35）**。本件不跑 ASR，只在同一批 D: 盘真 mel 上做量化诊断，回答三个问题：
      Q1 P2 为什么比 P1 更差（声门源谱倾斜？气声噪声？）
      Q2 失败样本的共同点（清擦/塞音段高频能量被吃掉？F0 八度错？清浊误判？）
      Q3 哪些量是可测的、能直接进 G0-v2 的判据

口径：真 mel（meeting66/产物/数据/样本/*.npz，D: 盘）→ GL（mel转波）→ 从 GL 波形提 oracle 控制
      → P1/P2 渲染 → 与 GL 波形逐帧比谱（GL 是「无损参照」，因为它就是 oracle 控制的来源）。
      本件**不做可懂度结论**，只给机制与量级。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AHI4 = HERE.parents[2]
M66 = AHI4 / "meeting66"
样本目录 = M66 / "产物" / "数据" / "样本"
meta路 = M66 / "产物" / "数据" / "meta.json"
出目录 = HERE / "产物"
频带 = [(100, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000)]


def _载(名, 路):
    spec = importlib.util.spec_from_file_location(名, str(路))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def P():
    return _载("探针_渲染上限_只读", HERE / "探针_渲染上限.py")


def 选片(n=6):
    meta = json.loads(meta路.read_text(encoding="utf-8"))
    序 = sorted(meta["样本"], key=lambda r: (int(r.get("帧数", 10 ** 9)), str(r.get("文件", ""))))
    return 序[:n]


def 谱帧(x, n_fft=1024, hop=220, win=551):
    """→ 功率谱帧 [T, 1+n_fft/2]（hann，center 口径近似：两端补零）。"""
    from numpy.lib.stride_tricks import sliding_window_view
    w = np.hanning(win)
    T = 1 + (len(x) - 1) // hop
    xp = np.pad(x, (win // 2, win))
    帧 = np.stack([xp[t * hop: t * hop + win] * w for t in range(T)]) if T else np.zeros((0, win))
    S = np.abs(np.fft.rfft(帧, n_fft)) ** 2
    return S


def ltas斜率(S, sr=22050, n_fft=1024):
    """在 100–8000 Hz 上对 log 功率最小二乘拟合斜率（dB/oct）。"""
    频 = np.fft.rfftfreq(n_fft, 1.0 / sr)
    m = (频 >= 100) & (频 <= 8000)
    p = np.maximum(S[:, m].mean(axis=0), 1e-20)
    lf = np.log2(频[m])
    A = np.vstack([lf, np.ones_like(lf)]).T
    k, _ = np.linalg.lstsq(A, np.log10(p), rcond=None)[0]
    return float(k * 10 * np.log10(2) * 2)      # dB/oct（功率→dB 再 ×2 因 log10(power)）


def 带能比(S, sr=22050, n_fft=1024, 低=(100, 4000), 高=(4000, 8000)):
    频 = np.fft.rfftfreq(n_fft, 1.0 / sr)
    p = np.maximum(S.mean(axis=0), 1e-20)
    e低 = float(p[(频 >= 低[0]) & (频 < 低[1])].sum())
    e高 = float(p[(频 >= 高[0]) & (频 <= 高[1])].sum())
    return float(10 * np.log10((e高 + 1e-20) / (e低 + 1e-20)))


def 平坦度(S, sr=22050, n_fft=1024, 带=(200, 8000)):
    频 = np.fft.rfftfreq(n_fft, 1.0 / sr)
    m = (频 >= 带[0]) & (频 <= 带[1])
    p = np.maximum(S[:, m], 1e-20)
    几 = np.exp(np.mean(np.log(p), axis=1))
    算 = np.mean(p, axis=1)
    return 几 / np.maximum(算, 1e-20)


def 八度跳率(f0, 浊):
    """相邻浊帧 F0 半音差 |Δ|>6 的比例（八度/半八度跳的代理指标）。"""
    v = np.asarray(浊, bool)
    f = np.asarray(f0, float)
    对 = (v[:-1] & v[1:]) & (f[:-1] > 0) & (f[1:] > 0)
    if 对.sum() == 0:
        return None, 0
    半 = 12 * np.log2(f[1:][对] / f[:-1][对])
    return float((np.abs(半) > 6).mean()), int(对.sum())


def 主(n: int = 6) -> dict:
    出目录.mkdir(parents=True, exist_ok=True)
    p = P()
    V = p.量具()
    S = p.声码()
    片s = 选片(n)
    报 = dict(说明="G0(v0) FAIL 的纯 DSP 机制诊断；参照=GL 波形（oracle 控制来源）；不载 ASR",
              逐条=[], 汇总={})
    for i, r in enumerate(片s):
        名 = Path(str(r["文件"])).stem
        mel = np.load(str(样本目录 / str(r["文件"])))["mel"].astype(np.float64)
        gl = np.asarray(S.mel转波(mel), dtype=np.float64)
        控 = p.提取控制(gl)
        nb = len(gl)
        # 三路激励（不同气声增益）：P1 脉冲 / P2 声门 / P2 无声气
        p.气声增益 = 0.25
        p1 = p.合成_臂(控, "P1_脉冲_LPC全极点", nb, 2000 + i)
        p2 = p.合成_臂(控, "P2_声门源_LPC全极点", nb, 2000 + i)
        原气声 = p.气声增益
        p.气声增益 = 0.0
        p2干 = p.合成_臂(控, "P2_声门源_LPC全极点", nb, 2000 + i)
        p.气声增益 = 原气声
        条 = dict(文件=str(r["文件"]), 时长秒=round(nb / p.SR, 2),
                  健康=dict(GL=[bool(np.all(np.isfinite(gl))), round(float(np.sqrt(np.mean(gl ** 2))), 4)],
                          P1=[bool(np.all(np.isfinite(p1))), round(float(np.sqrt(np.mean(p1 ** 2))), 4)],
                          P2=[bool(np.all(np.isfinite(p2))), round(float(np.sqrt(np.mean(p2 ** 2))), 4)]))
        S0, S1, S2, S2干 = 谱帧(gl), 谱帧(p1), 谱帧(p2), 谱帧(p2干)
        T = min(len(S0), len(S1), len(S2))
        S0, S1, S2, S2干 = S0[:T], S1[:T], S2[:T], S2干[:T]
        # Q1 谱倾斜与高频
        条["LTAS斜率dB每倍频程"] = dict(GL=round(ltas斜率(S0), 2), P1=round(ltas斜率(S1), 2),
                                        P2=round(ltas斜率(S2), 2), P2干=round(ltas斜率(S2干), 2))
        条["高频比4k8k对0k1_4k_dB"] = dict(GL=round(带能比(S0), 2), P1=round(带能比(S1), 2),
                                         P2=round(带能比(S2), 2), P2干=round(带能比(S2干), 2))
        频轴 = np.fft.rfftfreq(1024, 1.0 / p.SR)
        p0, p2m = np.maximum(S0.mean(0), 1e-20), np.maximum(S2.mean(0), 1e-20)
        分带 = []
        for b in 频带:
            m = (频轴 >= b[0]) & (频轴 < b[1])
            分带.append(dict(带="%d-%d" % b,
                             GL对P2_dB=round(float(10 * np.log10(p0[m].sum() / p2m[m].sum())), 2)))
        条["分带能量GL对P2_dB"] = 分带
        # Q2 清帧（用 GL 的谱平坦度定义清/浊：前 25% 为清）
        flat = 平坦度(S0)
        阈 = float(np.percentile(flat, 75))
        清帧 = flat >= 阈
        清占比 = float(清帧.mean())
        条["清帧"] = dict(占比=round(清占比, 3), 平坦度阈=round(阈, 3),
                        帧数=int(清帧.sum()))
        if 清帧.sum() > 5:
            频 = np.fft.rfftfreq(1024, 1 / p.SR)
            高带 = (频 >= 2000) & (频 <= 8000)
            低带 = (频 >= 100) & (频 < 2000)
            def 比(Sx):
                e = np.maximum(Sx, 1e-20)
                return float(np.mean(10 * np.log10(e[:, 高带].sum(1) / e[:, 低带].sum(1))))
            条["清帧_高频比2k_8k对0_2k_dB"] = dict(GL=round(比(S0[清帧]), 2),
                                              P1=round(比(S1[清帧]), 2),
                                              P2=round(比(S2[清帧]), 2),
                                              P2干=round(比(S2干[清帧]), 2))
        # Q2 F0/清浊代理
        跳, 对s = 八度跳率(控["f0"], 控["浊"])
        条["F0"] = dict(八度跳率=跳, 相邻浊帧对=对s,
                       浊帧占比=round(float(控["浊"].mean()), 3),
                       F0中位=round(float(np.median(控["f0"][控["浊"]])) if 控["浊"].any() else 0.0, 2))
        # 清浊冲突代理：pyin 浊 但平坦度高（前 25%）
        _m = min(len(控["浊"]), len(清帧))
        _冲突 = float((控["浊"][:_m] & 清帧[:_m]).sum() / max(1, int(控["浊"][:_m].sum())))
        条["清浊冲突率"] = round(_冲突, 3) if 控["浊"].any() else None
        # 气声噪声贡献：P2 与 P2干 的高频差
        条["气声贡献_dB"] = round(float(带能比(S2) - 带能比(S2干)), 2)
        报["逐条"].append(条)
        print("[诊断] %s %.1fs 斜率 GL=%.1f P1=%.1f P2=%.1f dB/oct ｜ 4-8k比 GL=%.1f P2=%.1f 干=%.1f"
              % (名, 条["时长秒"], 条["LTAS斜率dB每倍频程"]["GL"], 条["LTAS斜率dB每倍频程"]["P1"],
                 条["LTAS斜率dB每倍频程"]["P2"], 条["高频比4k8k对0k1_4k_dB"]["GL"],
                 条["高频比4k8k对0k1_4k_dB"]["P2"], 条["高频比4k8k对0k1_4k_dB"]["P2干"]), flush=True)
        if 清帧.sum() > 5:
            print("        清帧%.0f%% 高频比 GL=%.1f P1=%.1f P2=%.1f 干=%.1f dB ｜ 八度跳率=%s 清浊冲突=%s"
                  % (100 * 清占比, 条["清帧_高频比2k_8k对0_2k_dB"]["GL"],
                     条["清帧_高频比2k_8k对0_2k_dB"]["P1"], 条["清帧_高频比2k_8k对0_2k_dB"]["P2"],
                     条["清帧_高频比2k_8k对0_2k_dB"]["P2干"], 跳, 条["清浊冲突率"]), flush=True)
    # 汇总（中位）
    def 中(路):
        vs = []
        for t in 报["逐条"]:
            try:
                v = t
                for k in 路:
                    v = v[k]
                if isinstance(v, (int, float)):
                    vs.append(float(v))
            except (KeyError, TypeError):
                pass
        return round(float(np.median(vs)), 3) if vs else None
    报["汇总"] = dict(
        LTAS斜率_dB每倍频程=dict(GL=中(["LTAS斜率dB每倍频程", "GL"]),
                             P1=中(["LTAS斜率dB每倍频程", "P1"]),
                             P2=中(["LTAS斜率dB每倍频程", "P2"]),
                             P2干=中(["LTAS斜率dB每倍频程", "P2干"])),
        高频比_GL对P2_dB=中(["高频比4k8k对0k1_4k_dB"]) and dict(
            GL=中(["高频比4k8k对0k1_4k_dB", "GL"]), P1=中(["高频比4k8k对0k1_4k_dB", "P1"]),
            P2=中(["高频比4k8k对0k1_4k_dB", "P2"]), P2干=中(["高频比4k8k对0k1_4k_dB", "P2干"])),
        清帧高频比_dB=dict(GL=中(["清帧_高频比2k_8k对0_2k_dB", "GL"]),
                       P1=中(["清帧_高频比2k_8k对0_2k_dB", "P1"]),
                       P2=中(["清帧_高频比2k_8k对0_2k_dB", "P2"]),
                       P2干=中(["清帧_高频比2k_8k对0_2k_dB", "P2干"])),
        八度跳率=中(["F0", "八度跳率"]), 清浊冲突率=中(["清浊冲突率"]), 气声贡献_dB=中(["气声贡献_dB"]),
        清帧占比=中(["清帧", "占比"]), n=len(报["逐条"]))
    (出目录 / "诊断_G0v0失败.json").write_text(json.dumps(报, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    print("\n=== 汇总（中位，n=%d）===" % len(报["逐条"]))
    print(json.dumps(报["汇总"], ensure_ascii=False, indent=1))
    print("[落盘] %s" % (出目录 / "诊断_G0v0失败.json"))
    return 报


if __name__ == "__main__":
    sys.exit(0 if 主() else 0)
