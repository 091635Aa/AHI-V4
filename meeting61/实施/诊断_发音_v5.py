# -*- coding: utf-8 -*-
"""meeting61/实施/诊断_发音_v5.py —— 发音底层诊断（音韵映射 + 波形包络统计）。

目的：回答「底层发音哪里错」，三条证据：
  1) 拼音→声母/韵母 拆解与目标是否合法（y/w 零声母、舌尖元音、ian/üan 等）；
  2) 合成帧布局：辅音时长/电平 vs 元音，CV 是否有过渡；
  3) 波形统计：合成 vs 参考 的包络分布、清/浊帧电平差、谱平坦度、静音占比。

只读既有 v4 件 + 参考 wav；不改任何 v4 文件。输出 JSON 到 产物/诊断/。
纯 numpy/scipy（C1/C10），确定性（C13）。
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
MEET61 = os.path.dirname(HERE)
AHI4 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AHI4, "meeting32"))

import 声带 as 声带模块  # noqa: E402
import 韵律_v4 as H44  # noqa: E402

FS = 44100
出目录 = os.path.join(MEET61, "产物", "诊断")
os.makedirs(出目录, exist_ok=True)

测试句 = {
    "EmoCompanion": ["你好，我是EmoCompanion。",
             "今天的风很轻，适合慢慢地说一会儿话。",
             "你最近，过得好吗？"],
    "SpeakerA": ["嘿，我是SpeakerA。",
             "别看我平时闹得欢，认真说话的时候，我也能安静下来。",
             "要不要，一起待一会儿？"],
    "洛天依": ["大家好，我是洛天依。",
               "谢谢你愿意听我说话。",
               "希望今天的声音，能让你心里暖一点。"],
}
同句 = "语言不只是声音，它把一个人的温度，送到另一个人的心里。"


def 审计音韵(文本s) -> dict:
    """逐音节：拆解 + 韵母表命中 + 零声母 y/w + 舌尖元音。"""
    from pypinyin import lazy_pinyin, Style
    坏 = []
    音节数 = 0
    见 = {}
    for 文 in 文本s:
        for p in lazy_pinyin(文, style=Style.TONE3, neutral_tone_with_five=True,
                             errors=lambda x: [c for c in x if c.isalnum()]):
            if not p or not p[0].isascii():
                continue
            音节数 += 1
            声, 韵, 调 = 声带模块._拆音节(p)
            原拼 = p
            问题 = []
            if 原拼[0] in "yw":
                问题.append("零声母y/w未归一")
            if 声 == "" and 韵[:1] in tuple("yw"):
                问题.append("声母空且韵母以y/w开头")
            if 韵 not in 声带模块.韵母表:
                问题.append("韵母表未命中")
            if 声 and 声 not in 声带模块.声母表:
                问题.append("声母表未命中")
            目标s, 尾 = 声带模块._查韵母(韵)
            if not 目标s:
                问题.append("无共振峰目标")
            if len(目标s) == 1 and 韵 not in 声带模块._单韵母 and 韵 not in ("er",):
                问题.append("复合/鼻韵母只给单目标")
            if 韵 in ("ian", "uan", "iang", "uang", "ueng"):
                问题.append("介音丢失(表未建介音)")
            见.setdefault(原拼, dict(声=声, 韵=韵, 调=调,
                                     目标=[tuple(round(x) for x in t) for t in 目标s],
                                     尾=尾, 问题=问题))
            if 问题:
                坏.append(dict(拼音=原拼, 声=声, 韵=韵, 问题=问题))
    return dict(音节数=音节数, 问题音节数=len(坏), 问题列表=坏, 逐音节=见)


def 音节全集审计() -> dict:
    """全拼音节表（常用 400 音节的近似）逐个审计，统计结构性问题。"""
    声s = list(声带模块.声母序) + [""]
    韵s = ["a", "o", "e", "i", "u", "v", "ai", "ei", "ao", "ou", "an", "en",
           "ang", "eng", "ong", "er", "ia", "ie", "iao", "iu", "ian", "in",
           "iang", "ing", "iong", "ua", "uo", "uai", "ui", "uan", "un",
           "uang", "ueng", "ve", "van", "vn"]
    零声母y = {"ya": "ia", "ye": "ie", "yao": "iao", "you": "iou", "yan": "ian",
               "yin": "in", "yang": "iang", "ying": "ing", "yong": "iong",
               "yi": "i", "yu": "v", "yue": "ve", "yuan": "van", "yun": "vn",
               "yao2": "iao"}
    零声母w = {"wa": "ua", "wo": "uo", "wai": "uai", "wei": "uei", "wan": "uan",
               "wen": "uen", "wang": "uang", "weng": "ueng", "wu": "u"}
    统计 = {"未命中韵母": [], "零声母y_w": [], "舌尖元音": [],
            "单目标复合韵": [], "介音丢失": []}
    for 声 in 声s:
        for 韵 in 韵s:
            p = 声 + 韵
            if 声 in ("z", "c", "s") and 韵 == "i":
                统计["舌尖元音"].append(p)
            if 声 in ("zh", "ch", "sh", "r") and 韵 == "i":
                统计["舌尖元音"].append(p)
            if 韵 not in 声带模块.韵母表:
                统计["未命中韵母"].append(p)
            else:
                目标s, 尾 = 声带模块.韵母表[韵]
                if len(目标s) == 1 and 韵 not in ("a", "o", "e", "i", "u", "v", "er"):
                    统计["单目标复合韵"].append(p)
    for y in list(零声母y) + list(零声母w):
        统计["零声母y_w"].append(y)
    return 统计


def 帧布局(文本: str, 档) -> dict:
    """跑韵律，输出帧段（声源 type + 幅 + 共振峰）的 run-length。"""
    器 = H44.人类韵律44(档, 300)
    帧 = 器.建帧(文本)
    源 = 帧["声源"]
    幅 = 帧["幅"]
    F1, F2 = 帧["f1"], 帧["f2"]
    runs = []
    i = 0
    while i < len(源):
        j = i
        while j < len(源) and 源[j] == 源[i]:
            j += 1
        seg = dict(起帧=i, 帧数=j - i, 秒=round((j - i) / 100.0, 3), 声源=int(源[i]),
                   幅均=round(float(np.mean(幅[i:j])), 4) if j > i else 0.0,
                   F1均=round(float(np.mean(F1[i:j])), 1),
                   F2均=round(float(np.mean(F2[i:j])), 1))
        runs.append(seg)
        i = j
    清帧 = [r for r in runs if r["声源"] == 0]
    浊帧 = [r for r in runs if r["声源"] == 1]
    静帧 = [r for r in runs if r["声源"] == -1]
    return dict(总帧数=len(源), 段数=len(runs),
                清帧数=sum(r["帧数"] for r in 清帧),
                浊帧数=sum(r["帧数"] for r in 浊帧),
                静帧数=sum(r["帧数"] for r in 静帧),
                清幅均=round(float(np.mean([r["幅均"] for r in 清帧])), 4) if 清帧 else 0.0,
                浊幅均=round(float(np.mean([r["幅均"] for r in 浊帧])), 4) if 浊帧 else 0.0,
                清段数=len(清帧), 浊段数=len(浊帧), 静段数=len(静帧),
                首段=runs[:12])


def 读wav(路径):
    fs, x = wavfile.read(路径)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.max(np.abs(x)) > 1.2:
        x = x / 32768.0
    if fs != FS:
        g = int(np.gcd(fs, FS))
        x = resample_poly(x, FS // g, fs // g)
    return x


def _帧(x, n=1024, hop=256):
    if len(x) < n:
        y = np.zeros(n)
        y[:len(x)] = x
        return y[None, :]
    k = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(k)[:, None]
    return x[idx]


def 波形统计(x: np.ndarray) -> dict:
    """10ms RMS 包络分布 + 清/浊帧电平差 + 谱平坦度 + 静音占比。"""
    n = 441
    k = max(1, len(x) // n)
    e = np.array([np.sqrt(np.mean(x[i * n:(i + 1) * n] ** 2)) for i in range(k)])
    edb = 20.0 * np.log10(np.maximum(e, 1e-7))
    峰 = float(np.max(np.abs(x))) if len(x) else 0.0
    有聲 = edb > (float(np.max(edb)) - 40.0)
    静音占比 = float(np.mean(~有聲)) if k else 0.0
    # 帧级：过零率 + 谱平坦度 → 清/浊分类
    fr = _帧(np.pad(x, (0, max(0, 1024 - len(x)))))[:, :1024]
    zcr = np.mean(np.abs(np.diff(np.sign(fr), axis=1)) > 0, axis=1) / 2.0
    win = np.hanning(1024)
    P = np.abs(np.fft.rfft(fr * win[None, :], axis=1)) ** 2 + 1e-12
    flat = np.exp(np.mean(np.log(P), axis=1)) / np.mean(P, axis=1)
    fdb = 10.0 * np.log10(np.maximum(np.mean(P[:, :200], axis=1), 1e-12))
    清 = (zcr > 0.28) & (flat > 0.12)
    浊 = (zcr < 0.20) & (flat < 0.08) & (fdb > (np.max(fdb) - 45.0))
    清电平 = float(np.median(fdb[清])) if 清.sum() >= 5 else None
    浊电平 = float(np.median(fdb[浊])) if 浊.sum() >= 5 else None
    差 = (清电平 - 浊电平) if (清电平 is not None and 浊电平 is not None) else None
    return dict(
        时长秒=round(len(x) / FS, 3),
        峰=round(峰, 4),
        RMS_dBFS=round(float(20 * np.log10(max(np.sqrt(np.mean(x * x)), 1e-9))), 2),
        包络中位dB=round(float(np.median(edb)), 2),
        包络标准差dB=round(float(np.std(edb)), 2),
        包络动态范围dB=round(float(np.percentile(edb, 95) - np.percentile(edb, 5)), 2),
        静音占比=round(静音占比, 4),
        清帧占比=round(float(清.mean()), 4),
        浊帧占比=round(float(浊.mean()), 4),
        清浊电平差dB=round(差, 2) if 差 is not None else None,
        谱平坦度中位=round(float(np.median(flat)), 4),
    )


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    全句 = [s for v in 测试句.values() for s in v] + [同句]
    报 = {"工具": "诊断_发音_v5.py", "音韵审计": {}, "音节全集": {}, "帧布局": {}, "波形": {}}

    报["音韵审计"] = 审计音韵(全句)
    报["音节全集"] = 音节全集审计()

    sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))
    from 接口契约_角色声线 import 声线档案  # noqa: E402
    with open(os.path.join(MEET61, "产物", "声线档案", "EmoCompanion.json"), encoding="utf-8") as f:
        档 = 声线档案.从dict(json.load(f))
    报["帧布局"] = {"文本": "你好，我是EmoCompanion。", **帧布局("你好，我是EmoCompanion。", 档)}

    # 波形：合成（组C）vs 参考
    pairs = []
    for 名 in ("EmoCompanion", "SpeakerA", "洛天依"):
        p = os.path.join(MEET61, "产物", "测试组", "组C_v4韵律", "角色_%s.wav" % 名)
        if os.path.exists(p):
            pairs.append((名 + "·合成", p))
        rs = sorted(glob.glob(os.path.join(MEET61, "素材", 名, "ref44_*.wav")))
        if rs:
            pairs.append((名 + "·参考", rs[0]))
    for 标签, p in pairs:
        报["波形"][标签] = 波形统计(读wav(p))

    out = os.path.join(出目录, "诊断_发音_v5.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)

    a = 报["音韵审计"]
    print("=" * 78)
    print("音韵审计：%d 音节，问题 %d" % (a["音节数"], a["问题音节数"]))
    from collections import Counter
    c = Counter()
    for it in a["问题列表"]:
        for q in it["问题"]:
            c[q] += 1
    for k, v in c.most_common():
        print("  %-24s %d" % (k, v))
    print("  y/w 例子：", [it["拼音"] for it in a["问题列表"] if "y/w" in str(it["问题"])][:20])
    print("-" * 78)
    print("帧布局：", {k: 报["帧布局"][k] for k in ("总帧数", "清帧数", "浊帧数", "静帧数",
                                                   "清幅均", "浊幅均", "清段数", "浊段数", "静段数")})
    print("-" * 78)
    for k, v in 报["波形"].items():
        print("%-10s 时长%5.2fs RMS %6.2f | 包络σ %5.2f | 静音 %5.1f%% | 清帧 %5.1f%% | 清浊差 %s dB"
              % (k, v["时长秒"], v["RMS_dBFS"], v["包络标准差dB"],
                 v["静音占比"] * 100, v["清帧占比"] * 100, v["清浊电平差dB"]))
    print("JSON:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
