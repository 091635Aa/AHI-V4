# -*- coding: utf-8 -*-
"""meeting59 验收 · 自检（阶段A）：测量脚本自身的分辨率/稳定性

跑法：python 自检.py [--json 输出路径]
内容：
 1) 合成已知 F0（同一谐振器级联架构）→ F0 估计误差
 2) 合成已知共振峰（与 声带.py 同系数：A=1-B-C / C=-exp(-2πBW/FS)）→ F1/F2/F3 误差
 3) 7 带 log 包络余弦的分辨率：同元音/异元音、同 F0/异 F0、已知谱倾斜差
 4) 停顿检测：插入已知 100/140/200/300ms 静音 → 检出时长误差
 5) 回放互相关：自身=1、无关对、真人参考 vs 合成
 6) 两趟 sha256 / 两趟测量逐值同
 7) F0 中位对能量门限的敏感性（干净语音/合成/音乐会参考）
 8) meeting58 既有产物：同档案不同文本波动 vs 跨档案可分（17 维声纹代理）
输出：自测_指标分辨率.json + 屏显表
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import 测量_声学 as M  # noqa: E402

MEET58 = os.path.join(os.path.dirname(os.path.dirname(HERE)), "meeting58")
演示 = os.path.join(MEET58, "产物", "演示")
参考 = os.path.join(MEET58, "产物", "参考")
TMP = os.path.join(HERE, "tmp")
os.makedirs(TMP, exist_ok=True)

角色名 = ("洛天依", "初音未来", "MOSS", "EmoCompanion", "定制")


# ── 合成器（纯 numpy/scipy，与 声带.py 同系数；用于已知真值校准） ──
def 罗森堡脉冲(F0, sr, n):
    T = sr / float(F0)
    φ = (np.arange(n) % T) / T
    Tp, Tn = 0.40, 0.16
    g = np.zeros(n)
    m1 = φ < Tp
    g[m1] = 0.5 * (1.0 - np.cos(np.pi * φ[m1] / Tp))
    m2 = (φ >= Tp) & (φ < Tp + Tn)
    g[m2] = np.cos(np.pi * (φ[m2] - Tp) / (2.0 * Tn))
    return np.diff(g, prepend=0.0)


def 谐振(x, F, BW, sr):
    from scipy.signal import lfilter

    T = 1.0 / sr
    C = -np.exp(-2.0 * np.pi * BW * T)
    B = 2.0 * np.exp(-np.pi * BW * T) * np.cos(2.0 * np.pi * F * T)
    A = 1.0 - B - C
    return lfilter([A], [1.0, -B, -C], x)


def 合成元音(F0, F1, F2, F3, 时长=2.0, sr=M.FS默认):
    n = int(时长 * sr)
    src = 罗森堡脉冲(F0, sr, n)
    y = 谐振(src, F1, 80.0, sr)
    y = 谐振(y, F2, 110.0, sr)
    y = 谐振(y, F3, 170.0, sr)
    fade = max(1, int(0.020 * sr))
    包 = np.ones(n)
    包[:fade] = np.linspace(0.0, 1.0, fade)
    包[-fade:] = np.linspace(1.0, 0.0, fade)
    y = y * 包
    return y / (np.max(np.abs(y)) + 1e-12) * 0.9


def 测(路径):
    return M.汇总(路径)


def 中位(v):
    v = np.asarray([x for x in v if x is not None and math.isfinite(float(x))], dtype=float)
    return float(np.median(v)) if v.size else float("nan")


def main():
    开始 = time.time()
    出 = {"生成时间": time.strftime("%Y-%m-%dT%H:%M:%S"), "项": {}}

    # ── 1) F0 已知真值 ───────────────────────────────────────────
    t0 = time.time()
    f0项 = []
    for F0 in (110.0, 147.0, 196.0, 220.0, 330.0, 440.0):
        y = 合成元音(F0, 700.0, 1220.0, 2600.0, 时长=2.0)
        轨迹 = M.F0轨迹(y)
        统 = M.F0统计(轨迹)
        err = M.半音距(统["中位Hz"], F0)
        f0项.append({"真值": F0, "估计": 统["中位Hz"], "误差st": err,
                    "有声帧比例": 统["有声帧比例"]})
    f0err = [abs(x["误差st"]) for x in f0项 if math.isfinite(x["误差st"])]
    出["项"]["F0已知真值"] = {
        "结果": f0项, "最大绝对误差st": float(max(f0err)) if f0err else float("nan"),
        "中位绝对误差st": float(np.median(f0err)) if f0err else float("nan"),
        "耗时秒": round(time.time() - t0, 2),
    }

    # ── 2) F1/F2/F3 已知真值（同架构级联谐振器） ─────────────────
    t0 = time.time()
    fm项 = []
    for (F1, F2, F3, F0) in ((800.0, 1200.0, 2500.0, 220.0),
                            (300.0, 2200.0, 3000.0, 220.0),
                            (320.0, 700.0, 2500.0, 220.0),
                            (550.0, 1100.0, 2500.0, 180.0),
                            (800.0, 1200.0, 2500.0, 120.0)):
        y = 合成元音(F0, F1, F2, F3, 时长=2.0)
        r = M.倒谱共振峰(y)
        fm项.append({"真值": [F1, F2, F3], "F0": F0,
                    "估计": [r["F1中位"], r["F2中位"], r["F3中位"]],
                    "相对误差": [abs(r["F1中位"] - F1) / F1,
                               abs(r["F2中位"] - F2) / F2,
                               abs(r["F3中位"] - F3) / F3],
                    "F1有效比": r["F1有效比"], "F2有效比": r["F2有效比"]})
    e1 = [x["相对误差"][0] for x in fm项 if math.isfinite(x["相对误差"][0])]
    e2 = [x["相对误差"][1] for x in fm项 if math.isfinite(x["相对误差"][1])]
    出["项"]["共振峰已知真值"] = {
        "结果": fm项,
        "F1最大相对误差": float(max(e1)) if e1 else float("nan"),
        "F2最大相对误差": float(max(e2)) if e2 else float("nan"),
        "耗时秒": round(time.time() - t0, 2),
    }

    # ── 3) 7 带包络余弦分辨率 ────────────────────────────────────
    t0 = time.time()
    包项 = []
    y_a = 合成元音(220.0, 800.0, 1200.0, 2500.0)
    y_i = 合成元音(220.0, 300.0, 2200.0, 3000.0)
    y_a2 = 合成元音(330.0, 800.0, 1200.0, 2500.0)     # 同元音异 F0
    y_a3 = 合成元音(220.0, 880.0, 1320.0, 2750.0)     # 元音整体 +10%
    for 名, y in (("/a/", y_a), ("/i/", y_i), ("/a/ F0 330", y_a2), ("/a/ +10%共振峰", y_a3)):
        包 = M.七带包络(y)
        包项.append({"信号": 名, "带均值dB": [float(v) for v in 包["带均值dB"]],
                    "倾斜": float(包["倾斜dB每倍频程"])})
    cos_同 = M.带余弦(np.array(包项[0]["带均值dB"]), np.array(包项[1]["带均值dB"]))
    cos_f0 = M.带余弦(np.array(包项[0]["带均值dB"]), np.array(包项[2]["带均值dB"]))
    cos_fm = M.带余弦(np.array(包项[0]["带均值dB"]), np.array(包项[3]["带均值dB"]))
    出["项"]["7带包络余弦分辨率"] = {
        "包": 包项, "cos(/a/,/i/)": cos_同, "cos(同元音F0差%dHz)" % (330 - 220): cos_f0,
        "cos(共振峰+10%)": cos_fm, "耗时秒": round(time.time() - t0, 2),
    }

    # ── 4) 停顿检测（插入已知静音） ──────────────────────────────
    t0 = time.time()
    base_p = os.path.join(演示, "角色_EmoCompanion_0.wav")
    停项 = []
    if os.path.isfile(base_p):
        sr, x = M.读wav(base_p)
        x = x[: int(1.2 * sr)] if len(x) > int(1.2 * sr) else x
        半 = len(x) // 2
        for 真ms in (100, 140, 200, 300):
            gap = np.zeros(int(真ms / 1000.0 * sr))
            y = np.concatenate([x[:半], gap, x[半:]])
            路径 = os.path.join(TMP, "自测_停顿_%dms.wav" % 真ms)
            from scipy.io import wavfile
            wavfile.write(路径, sr, (np.clip(y, -1, 1) * 32767).astype("<i2"))
            r = M.汇总(路径)["读数"]
            停 = r["停顿"]
            检出 = [p for p in r["停顿列表"]
                    if abs(p["开始秒"] - 半 / sr) < 0.30]
            最长 = max([p["时长秒"] for p in 检出], default=0.0)
            停项.append({"真值ms": 真ms, "检出段数": len(检出), "检出ms": 1000 * 最长,
                        "误差ms": 1000 * (最长 - 真ms / 1000.0),
                        "检出≥120ms": bool(检出)})
    出["项"]["停顿检测"] = {"结果": 停项, "耗时秒": round(time.time() - t0, 2)}

    # ── 5) 回放互相关 ───────────────────────────────────────────
    t0 = time.time()
    相关项 = {}
    文件 = {
        "EmoCompanion0": os.path.join(演示, "角色_EmoCompanion_0.wav"),
        "洛天依同句": os.path.join(演示, "同句五声_洛天依.wav"),
        "EmoCompanion同句": os.path.join(演示, "同句五声_EmoCompanion.wav"),
        "真人参考_huihui": os.path.join(参考, "参考_定制_huihui.wav"),
        "定制0": os.path.join(演示, "角色_定制_0.wav"),
    }
    波形 = {}
    for k, p in 文件.items():
        if os.path.isfile(p):
            sr, x = M.读wav(p)
            波形[k] = x
    相关项["自身"] = M.最大规整互相关(波形["EmoCompanion0"], 波形["EmoCompanion0"]) if "EmoCompanion0" in 波形 else None
    if "EmoCompanion同句" in 波形 and "洛天依同句" in 波形:
        相关项["同句异声(EmoCompanionvs洛天依)"] = M.最大规整互相关(波形["EmoCompanion同句"], 波形["洛天依同句"])
    if "真人参考_huihui" in 波形 and "定制0" in 波形:
        相关项["真人参考vs合成_定制"] = M.最大规整互相关(波形["真人参考_huihui"], 波形["定制0"])
    出["项"]["回放互相关"] = {"结果": 相关项, "判定阈": 0.99,
                          "耗时秒": round(time.time() - t0, 2)}

    # ── 6) 两趟 sha256 / 两趟测量逐值同 ─────────────────────────
    t0 = time.time()
    sha项 = {}
    p = os.path.join(演示, "角色_EmoCompanion_0.wav")
    if os.path.isfile(p):
        s1 = M.sha256文件(p); s2 = M.sha256文件(p)
        r1 = M.汇总(p)["读数"]; r2 = M.汇总(p)["读数"]
        r1.pop("计时秒", None); r2.pop("计时秒", None)   # 计时不是测量量
        sha项 = {"sha256趟1": s1, "sha256趟2": s2, "sha逐位同": s1 == s2,
                 "测量两趟同": json.dumps(r1, sort_keys=True, ensure_ascii=False) ==
                              json.dumps(r2, sort_keys=True, ensure_ascii=False)}
    出["项"]["确定性"] = {"结果": sha项, "耗时秒": round(time.time() - t0, 2)}

    # ── 7) F0 门限敏感性（分辨率） ──────────────────────────────
    t0 = time.time()
    敏项 = []
    素材目录 = os.path.join(os.path.dirname(HERE), "素材")
    敏文件 = {
        "合成_角色EmoCompanion0": os.path.join(演示, "角色_EmoCompanion_0.wav"),
        "真人参考_huihui": os.path.join(参考, "参考_定制_huihui.wav"),
        "EmoCompanion参考_干净01": os.path.join(素材目录, "EmoCompanion", "ref_EmoCompanion_01.wav"),
        "SpeakerA参考_干净01": os.path.join(素材目录, "SpeakerA", "ref_SpeakerA_01.wav"),
        "洛天依参考_干净说话01": os.path.join(素材目录, "洛天依", "ref_洛天依_语音_01.wav"),
        "洛天依参考_干净清唱01": os.path.join(素材目录, "洛天依", "ref_洛天依_清唱_01.wav"),
        "洛天依参考_BGM测试样本": os.path.join(素材目录, "洛天依", "ref_洛天依_16k.wav"),
        "lead试听(若有)": os.path.join(os.path.dirname(HERE), "产物", "临时", "试听.wav"),
    }
    for 名, p in 敏文件.items():
        if not os.path.isfile(p):
            continue
        sr, x = M.读wav(p)
        if len(x) > 30 * sr:            # 长参考截 30s 控制时间
            x = x[: 30 * sr]
        中 = {}
        for gate in (15, 20, 25, 30, 35, 40):
            轨迹 = M.F0轨迹(x, sr, 能量相对阈dB=gate)
            中[str(gate)] = float(M.F0统计(轨迹)["中位Hz"])
        v = np.array(list(中.values()))
        敏项.append({"文件": 名, "中位Hz": 中,
                    "极差st": M.半音距(v.max(), v.min()) if v.min() > 0 else float("nan"),
                    "门限25_40极差st": M.半音距(中["25"], 中["40"]) if 中["40"] > 0 else float("nan")})
    出["项"]["F0门限敏感性"] = {"结果": 敏项, "耗时秒": round(time.time() - t0, 2)}

    # ── 7b) 参考侧片段间波动（同一角色不同参考片段，F0 中位） ─────
    t0 = time.time()
    片项 = []
    for 子 in ("EmoCompanion", "SpeakerA", "洛天依"):
        d = os.path.join(素材目录, 子)
        if not os.path.isdir(d):
            continue
        ps = sorted(glob.glob(os.path.join(d, "ref_*.wav")))
        # 排除含 BGM 的测试样本（manifest 已声明不参与拟合）
        ps = [p for p in ps if "16k" not in os.path.basename(p)]
        if not ps:
            continue
        值 = []
        for p in ps:
            sr, x = M.读wav(p)
            r = M.F0统计(M.F0轨迹(x, sr))
            值.append({"片段": os.path.basename(p), "时长秒": round(len(x) / sr, 2),
                     "F0中位Hz": round(r["中位Hz"], 2),
                     "跨度st": round(r["跨度st"], 2)})
        med = [v["F0中位Hz"] for v in 值]
        片项.append({"角色": 子, "片段数": len(值), "片段": 值,
                    "F0中位极差st": M.半音距(max(med), min(med)) if min(med) > 0 else None,
                    "F0中位中位": float(np.median(med))})
    出["项"]["参考片段间波动"] = {"结果": 片项, "耗时秒": round(time.time() - t0, 2)}

    # ── 8) meeting58 既有产物：同档案跨文本波动 / 跨档案可分 ─────
    t0 = time.time()
    档案 = {}
    for 名 in 角色名:
        fs = [os.path.join(演示, "角色_%s_%d.wav" % (名, i)) for i in (0, 1)]
        fs = [f for f in fs if os.path.isfile(f)]
        if fs:
            档案[名] = fs
    # 同句五声
    同句 = {}
    for 名 in 角色名:
        p = os.path.join(演示, "同句五声_%s.wav" % 名)
        if os.path.isfile(p):
            同句[名] = p
    汇总 = {}
    for 名, fs in 档案.items():
        for i, f in enumerate(fs):
            汇总["%s_文本%d" % (名, i)] = M.汇总(f)
    for 名, p in 同句.items():
        汇总["%s_同句" % 名] = M.汇总(p)
    # 同档案不同文本：F0 差 / 带余弦 / F1F2 差
    内 = []
    for 名, fs in 档案.items():
        if len(fs) < 2:
            continue
        a = M.汇总(fs[0])["读数"]; b = M.汇总(fs[1])["读数"]
        内.append({"档案": 名,
                  "F0半音差": abs(M.半音距(a["F0"]["中位Hz"], b["F0"]["中位Hz"])),
                  "带余弦": M.带余弦(np.array(a["七带"]["带均值dB"]),
                                  np.array(b["七带"]["带均值dB"])),
                  "F1相对差": abs(a["共振峰"]["F1中位"] - b["共振峰"]["F1中位"]) /
                             max(a["共振峰"]["F1中位"], 1e-9),
                  "F1有效比_0": a["共振峰"]["F1有效比"],
                  "F1有效比_1": b["共振峰"]["F1有效比"]})
    # 跨档案同句：F0 距 / 带余弦
    对比 = []
    名单 = list(同句)
    for i in range(len(名单)):
        for j in range(i + 1, len(名单)):
            a = M.汇总(同句[名单[i]])["读数"]; b = M.汇总(同句[名单[j]])["读数"]
            对比.append({"对": 名单[i] + "|" + 名单[j],
                        "F0半音距": abs(M.半音距(a["F0"]["中位Hz"], b["F0"]["中位Hz"])),
                        "带余弦": M.带余弦(np.array(a["七带"]["带均值dB"]),
                                       np.array(b["七带"]["带均值dB"]))})
    # 17 维声纹代理距离（尺度=同档案跨文本 pooled std）
    def 向量读(读):
        return np.array(读["声纹17"], dtype=float)
    同句向量 = {k: 向量读(v["读数"]) for k, v in 汇总.items() if k.endswith("_同句")}
    差向 = []
    for 名, fs in 档案.items():
        if len(fs) >= 2:
            a = 向量读(M.汇总(fs[0])["读数"]); b = 向量读(M.汇总(fs[1])["读数"])
            差向.append(a - b)
    if 差向:
        尺 = np.sqrt(np.mean(np.stack(差向) ** 2, axis=0)) + 1e-9
    else:
        尺 = np.ones(17)
    d内 = []
    for 名, fs in 档案.items():
        if len(fs) >= 2:
            a = 向量读(M.汇总(fs[0])["读数"]); b = 向量读(M.汇总(fs[1])["读数"])
            d内.append(float(np.linalg.norm((a - b) / 尺)))
    对比列表 = []
    名单 = list(同句向量)
    for i in range(len(名单)):
        for j in range(i + 1, len(名单)):
            a = 同句向量[名单[i]]; b = 同句向量[名单[j]]
            对比列表.append({"对": 名单[i] + "|" + 名单[j],
                          "d17": float(np.linalg.norm((a - b) / 尺))})
    出["项"]["meeting58产物分辨率"] = {
        "同档案不同文本": 内,
        "跨档案同句": 对比,
        "17维近邻对比(同句)": 对比列表,
        "同档跨文本d17": d内,
        "判别比D(跨档d/同档d)": (float(np.median([c["d17"] for c in 对比列表]) /
                                   max(np.median(d内), 1e-9)) if 对比列表 and d内 else None),
        "耗时秒": round(time.time() - t0, 2),
    }

    出["总耗时秒"] = round(time.time() - 开始, 2)
    出["口径"] = ("测量库纯 numpy/scipy 独立实现；合成器仅用于已知真值校准"
                "（罗森伯格脉冲 + 与 声带.py 同系数三级联谐振器）；"
                "不 import lead 拟合/韵律/合成模块；meeting58 产物只读。")
    # ── 打印 ──
    print("== 1) F0 已知真值 ==")
    for x in 出["项"]["F0已知真值"]["结果"]:
        print("  真值 %6.1fHz 估计 %7.2fHz 误差 %+0.2f st  有声 %.0f%%"
              % (x["真值"], x["估计"], x["误差st"], 100 * x["有声帧比例"]))
    print("  最大|误差| %.2f st，中位 %.2f st"
          % (出["项"]["F0已知真值"]["最大绝对误差st"],
             出["项"]["F0已知真值"]["中位绝对误差st"]))
    print("== 2) 共振峰已知真值 ==")
    for x in 出["项"]["共振峰已知真值"]["结果"]:
        print("  真值 %s → 估计 %s  相对误差 %s  有效比 F1 %.0f%% F2 %.0f%%"
              % ([round(v) for v in x["真值"]],
                 [None if not math.isfinite(v) else round(v) for v in x["估计"]],
                 ["%.1f%%" % (100 * v) if math.isfinite(v) else "nan" for v in x["相对误差"]],
                 100 * x["F1有效比"], 100 * x["F2有效比"]))
    print("  F1 最大相对误差 %.1f%%，F2 最大相对误差 %.1f%%（阈值 12%%）"
          % (100 * 出["项"]["共振峰已知真值"]["F1最大相对误差"],
             100 * 出["项"]["共振峰已知真值"]["F2最大相对误差"]))
    print("== 3) 7 带包络余弦分辨率 ==")
    print("  cos(/a/,/i/) = %.4f ; cos(同元音 F0 差 110Hz) = %.4f ; cos(共振峰+10%%) = %.4f"
          % (出["项"]["7带包络余弦分辨率"]["cos(/a/,/i/)"],
             出["项"]["7带包络余弦分辨率"]["cos(同元音F0差110Hz)"],
             出["项"]["7带包络余弦分辨率"]["cos(共振峰+10%)"]))
    print("== 4) 停顿检测 ==")
    for x in 出["项"]["停顿检测"]["结果"]:
        print("  真 %.0fms → 检出 %.0fms（误差 %+.0fms，段数 %d）"
              % (x["真值ms"], x["检出ms"], x["误差ms"], x["检出段数"]))
    print("== 5) 回放互相关 ==")
    for k, v in 出["项"]["回放互相关"]["结果"].items():
        print("  %s = %.4f" % (k, v if v is not None else float("nan")))
    print("== 6) 确定性 ==", 出["项"]["确定性"]["结果"])
    print("== 7) F0 门限敏感性（中位 Hz） ==")
    for x in 出["项"]["F0门限敏感性"]["结果"]:
        print("  %-22s 极差 %.2f st（25–40dB %.2f st）  %s"
              % (x["文件"], x["极差st"], x["门限25_40极差st"],
                 {k: round(v, 1) for k, v in x["中位Hz"].items()}))
    print("== 7b) 参考侧片段间波动（F0 中位） ==")
    for x in 出["项"]["参考片段间波动"]["结果"]:
        print("  %-4s %d 片段 F0 中位极差 %.2f st（中位 %.1f Hz）：%s"
              % (x["角色"], x["片段数"], x["F0中位极差st"], x["F0中位中位"],
                 [(v["片段"], v["F0中位Hz"]) for v in x["片段"]]))
    print("== 8) meeting58 产物分辨率 ==")
    for x in 出["项"]["meeting58产物分辨率"]["同档案不同文本"]:
        print("  同档异文本 %-6s F0差 %.2fst 带余弦 %.4f"
              % (x["档案"], x["F0半音差"], x["带余弦"]))
    for x in 出["项"]["meeting58产物分辨率"]["跨档案同句"]:
        print("  跨档同句 %-16s F0距 %.2fst 带余弦 %.4f" % (x["对"], x["F0半音距"], x["带余弦"]))
    print("  17 维：同档跨文本 d 中位 %s；判别比 D = %s"
          % (round(中位(出["项"]["meeting58产物分辨率"]["同档跨文本d17"]), 3),
             出["项"]["meeting58产物分辨率"]["判别比D(跨档d/同档d)"]))
    print("总耗时 %.1fs" % 出["总耗时秒"])

    out_json = os.path.join(HERE, "自测_指标分辨率.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(出, f, ensure_ascii=False, indent=1)
    print("→", out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
