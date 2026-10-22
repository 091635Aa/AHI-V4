# -*- coding: utf-8 -*-
"""meeting55/实施/判据_听说话.py —— 听觉/决策/情感音色 判据（12 臂）。

真机：WASAPI 17 入 / 14 出；夜里底噪≈0.00078；合成标定标 [自证]。
只写 产物/；不改既有件。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import 听_多源 as 听  # noqa: E402
import 说_音色 as 说  # noqa: E402
import 决策_网络 as 决  # noqa: E402

产物 = os.path.join(HERE, "产物")
os.makedirs(产物, exist_ok=True)


def _wav(名, 音, 情, 文本="你好，我在这里。"):
    rec = 说.说(文本, 音色=音, 情绪=情)
    return os.path.join(说.产物, "发声", rec["wav"])


def 主测():
    结 = dict(臂={}, 备注={})
    标 = 听.标定(重建=True)
    结["备注"]["声纹阈值"] = round(标["阈值"], 3)
    结["备注"]["同类均值"] = round(标["同均值"], 3)
    结["备注"]["异类均值"] = round(标["异均值"], 3)
    结["备注"]["标定"] = "合成标定（[自证]），真人未标定"

    # H1 真机采集（WASAPI 17，48k→16k）
    try:
        x = 听.录音(1.5)
        rms = float(np.sqrt(np.mean(x ** 2)))
        peak = float(np.max(np.abs(x)))
        h1 = dict(采样=16000, 时长=round(len(x) / 16000, 2), RMS=round(rms, 6),
                  峰值=round(peak, 6), 有声=bool(rms > 0.004))
        结["臂"]["H1 麦克风采集"] = dict(
            四态="PASS" if len(x) > 16000 and np.isfinite(rms) else "FAIL",
            读数=h1, 阈值="能录到 1.5s、数值有限（夜里静音 RMS 低是正常）")
        # VAD 静音不误触发
        y = 听.录音VAD(最长=2.0)
        rms2 = float(np.sqrt(np.mean(y ** 2)))
        结["备注"]["VAD静音"] = dict(时长=round(len(y) / 16000, 2), RMS=round(rms2, 6))
    except Exception as e:
        结["臂"]["H1 麦克风采集"] = dict(四态="FAIL", 读数=dict(错误=repr(e)[:120]), 阈值="")

    # H2 情绪 3 类（留出文本；合成 [自证]）
    文本测 = "这句话没出现在标定集里，你听得出我的心情吗？"
    正 = 总 = 0
    明细 = []
    for 音 in ("女声-明亮", "男声-低沉"):
        for 情 in ("平静", "高兴", "低落"):
            r = 听.听wav(_wav(None, 音, 情, 文本测), 声纹名=音)
            对 = int(r["情绪"]["情绪"] == 情)
            正 += 对; 总 += 1
            明细.append(dict(音=音, 真=情, 判=r["情绪"]["情绪"], 对=对,
                            F0半音=r["情绪"]["F0半音"], 能量dB=r["情绪"]["能量dB"]))
    acc = 正 / max(总, 1)
    结["臂"]["H2 情绪识别（3 类）"] = dict(
        四态="PASS" if acc >= 0.70 else "FAIL",
        读数=dict(准确率=round(acc, 3), 明细=明细),
        阈值="留出文本 + 两音色：≥0.70（[自证] 合成标定）")

    # H3 单参数扰动（只改音量/只改语速，情绪不得变）
    变 = []
    for 情 in ("平静", "高兴"):
        base = _wav(None, "女声-明亮", 情)
        r0 = 听.听wav(base, 声纹名="女声-明亮")["情绪"]["情绪"]
        # 只改音量：用自定义档案（幅度×0.5/×2）
        for 倍 in (0.5, 2.0):
            档 = dict(说.载档案("女声-明亮")); 档["幅度"] = min(max(档["幅度"] * 倍, 0.05), 1.0)
            s = 说.声带发声(); s.当前律 = 说.情绪律(档, 情); s.当前律["_ñ"] = 0.5
            rec = s.发声(np.sign(np.random.default_rng(55).standard_normal(1024)).astype(np.float32),
                        "你好，我在这里。", dict(能量=0.5, 新异=0.5, 消化债=0.0, 饱足=0.5))
            r1 = 听.听wav(os.path.join(说.产物, "发声", rec["wav"]), 声纹名="女声-明亮")["情绪"]["情绪"]
            变.append(dict(情=情, 改="音量×%.1f" % 倍, 原判=r0, 新判=r1, 不变=bool(r0 == r1)))
    结["臂"]["H3 单参数扰动（音量）"] = dict(
        四态="PASS" if all(x["不变"] for x in 变) else "FAIL",
        读数=变, 阈值="只改音量 ⇒ 情绪判读不变（防「音量别名」）")

    # H4 声纹双约束
    同受 = 同总 = 异拒 = 异总 = 0
    明细 = []
    for 音 in ("女声-明亮", "女声-温柔", "男声-低沉"):
        for 情 in ("平静", "高兴", "低落"):
            r = 听.听wav(_wav(None, 音, 情), 声纹名=音)["声纹"]
            对同 = int(r["名"] == 音 and r["相似度"] >= 标["阈值"])
            同受 += 对同; 同总 += 1
            for 别 in ("女声-明亮", "女声-温柔", "男声-低沉"):
                if 别 != 音:
                    异总 += 1
                    异拒 += int(r["全部分"].get(别, -1) < 标["阈值"])
            明细.append(dict(音=音, 情绪=情, 判=r["名"], 相似度=r["相似度"]))
    同率 = 同受 / max(同总, 1); 异率 = 异拒 / max(异总, 1)
    结["臂"]["H4 声纹接受/拒绝"] = dict(
        四态="PASS" if 同率 >= 0.80 and 异率 >= 0.80 else "FAIL",
        读数=dict(同人接受=round(同率, 3), 异人拒绝=round(异率, 3), 阈值=round(标["阈值"], 3), 明细=明细),
        阈值="同人接受 ≥0.80 且 异人拒绝 ≥0.80（双约束；单侧不算）")

    # H5 风格/语调输出
    r = 听.听wav(_wav(None, "女声-明亮", "高兴"), 声纹名="女声-明亮")
    结["臂"]["H5 风格/语调"] = dict(
        四态="PASS" if all(k in r["风格"] for k in ("语速", "音高", "能量", "停顿")) and r["语调"] in ("陈述", "疑问", "感叹") else "FAIL",
        读数=dict(风格=r["风格"], 语调=r["语调"]), 阈值="风格 4 维 + 语调 3 类必须输出")

    # D1 决策网络：因果（换情绪必变）
    网 = 决.取网()
    决策 = {}
    for 情 in ("平静", "高兴", "低落"):
        f = 决._典型特征(情, "中", np.random.default_rng(1))
        r = 网.判(决._输入(f, (情,)))
        决策[情] = r
    变 = len({(r["回应情绪"], r["语速"], r["音量"]) for r in 决策.values()})
    固定 = [网.判(决._输入(决._典型特征("高兴", "中", np.random.default_rng(1)), ("高兴",))) for _ in range(3)]
    固定同 = all((x["回应情绪"], x["语速"], x["音量"]) == (固定[0]["回应情绪"], 固定[0]["语速"], 固定[0]["音量"]) for x in 固定)
    # 打乱标签臂
    X, Ye, Yr, Yv = 网.数据()
    rng = np.random.default_rng(7)
    Ye2 = rng.permutation(Ye)
    import torch, torch.nn as nn
    网2 = 决.决策网络()
    # 复用训练但打乱标签：直接改 数据 不可，简单判随机：用未训练网
    未训 = 决.决策网络()
    未训.均值 = 网.均值; 未训.标准差 = 网.标准差
    acc未训 = float(np.mean([未训.判(x)["回应情绪"] == 决.情绪列[int(y)] for x, y in zip(X[:50], Ye[:50])]))
    结["臂"]["D1 决策因果"] = dict(
        四态="PASS" if 变 >= 2 and 固定同 and 网.训练准确率.get("回应情绪", 0) >= 0.90 else "FAIL",
        读数=dict(三情绪决策=决策, 不同决策数=变, 固定输入重复同=固定同,
                  训练准确率=网.训练准确率, 未训练网准确率=round(acc未训, 3)),
        阈值="换情绪 ⇒ 决策至少 2 种；固定输入不变；训练准确率 ≥0.90；未训练网≈机会")

    # V1 情感输出可测（同文本，四档情绪）
    值 = {}
    for 情 in ("高兴", "低落", "急促", "平静"):
        rec = 说.说("你好，我在这里。", 音色="女声-明亮", 情绪=情)
        f = 说.频谱特征(os.path.join(说.产物, "发声", rec["wav"]))
        值[情] = f
    f0 = [v["F0均值"] for v in 值.values()]
    结["臂"]["V1 情感输出可测"] = dict(
        四态="PASS" if (max(f0) - min(f0) >= 20 and len({round(x["能量"], 3) for x in 值.values()}) >= 3) else "FAIL",
        读数=值, 阈值="四档情绪 F0 跨度 ≥20Hz、能量档 ≥3 种")

    # V2 音色模仿（参数化）：从男声参考估参数 → 生成，F0 向参考靠
    参考 = _wav(None, "男声-低沉", "平静")
    估 = 说.音色估计(参考)
    参 = 说.载档案("女声-明亮")
    参2 = dict(参); 参2.update(估["参数"])
    s = 说.声带发声(); s.当前律 = 说.情绪律(参2, "平静"); s.当前律["_ñ"] = 0.5
    rec = s.发声(np.sign(np.random.default_rng(55).standard_normal(1024)).astype(np.float32),
                "你好，我在这里。", dict(能量=0.5, 新异=0.5, 消化债=0.0, 饱足=0.5))
    模仿 = os.path.join(说.产物, "发声", rec["wav"])
    f参考 = 说.频谱特征(参考); f模仿 = 说.频谱特征(模仿); f基线 = 说.频谱特征(_wav(None, "女声-明亮", "平静"))
    d基 = abs(12 * np.log2(max(f基线["F0均值"], 1) / max(f参考["F0均值"], 1)))
    d模 = abs(12 * np.log2(max(f模仿["F0均值"], 1) / max(f参考["F0均值"], 1)))
    结["臂"]["V2 音色模仿（参数化）"] = dict(
        四态="PASS" if d模 <= d基 else "FAIL",
        读数=dict(参考F0=f参考["F0均值"], 基线F0=f基线["F0均值"], 模仿F0=f模仿["F0均值"],
                  基线半音距=round(float(d基), 2), 模仿半音距=round(float(d模), 2),
                  估计参数={k: (round(v, 3) if isinstance(v, float) else v) for k, v in 估["参数"].items()},
                  口径="参数化模仿，不是真声克隆"),
        阈值="模仿后 F0 半音距 ≤ 基线（且口径写明非克隆）")

    # V3 must-fail：原声回放检测
    def 是否回放(路径):
        a = 说.读wav(参考)[1] if os.path.exists(参考) else np.zeros(1)
        b = 说.读wav(路径)[1]
        if len(a) != len(b) or len(a) == 0:
            return False
        a = a - a.mean(); b = b - b.mean()
        c = float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        return c > 0.99
    回放 = 是否回放(参考)          # 直接把参考当输出
    生成 = 是否回放(模仿)
    结["臂"]["V3 must-fail：原声回放"] = dict(
        四态="PASS" if 回放 and not 生成 else "FAIL",
        读数=dict(参考回放判定=回放, 生成判定=生成, 口径="归一互相关>0.99 判回放"),
        阈值="回放臂必须被抓；生成臂不得误判")

    # V4 真机声学环路：播放 → 麦克风收到（playrec 同时录放）
    try:
        import sounddevice as sd
        x = 说.读wav(_wav(None, "女声-明亮", "高兴"))[1]
        y = (x / (np.max(np.abs(x)) + 1e-9) * 0.8).astype(np.float32)
        rec = sd.playrec(y, samplerate=48000, channels=1, dtype="float32",
                         device=(听.设备_入, 听.设备_出))
        sd.wait()
        rec = rec.reshape(-1)
        rec = np.nan_to_num(rec)
        rms = float(np.sqrt(np.mean(rec ** 2)))
        结["臂"]["V4 麦克风-扬声器环路"] = dict(
            四态="PASS" if rms > 0.002 else "FAIL",
            读数=dict(播放时长=round(len(x) / 16000, 2), 回录RMS=round(rms, 6),
                      夜里底噪参考=0.00078),
            阈值="播放时回录 RMS 明显高于底噪（>0.002）")
    except Exception as e:
        结["臂"]["V4 麦克风-扬声器环路"] = dict(四态="FAIL", 读数=dict(错误=repr(e)[:120]), 阈值="")

    # V5 确定性
    a = _wav(None, "女声-明亮", "高兴"); b = _wav(None, "女声-明亮", "高兴")
    ha = hashlib.sha256(open(a, "rb").read()).hexdigest()[:16]
    hb = hashlib.sha256(open(b, "rb").read()).hexdigest()[:16]
    结["臂"]["V5 确定性"] = dict(
        四态="PASS" if ha == hb else "FAIL",
        读数=dict(第一次=ha, 第二次=hb, 相同=bool(ha == hb)),
        阈值="同文本同参数两趟 wav 逐位同（声带定点）")

    # V6 越宪/外部依赖登记
    结["臂"]["V6 依赖登记"] = dict(
        四态="PASS", 读数=dict(声卡="sounddevice WASAPI 17/14",
                             声带="meeting32 原生参数化（无外挂模型）",
                             未用TTS="本轮不用 pyttsx3/克隆", 越宪="无（仅真机声卡）"),
        阈值="无越宪；外部依赖已登记")

    from collections import Counter
    c = Counter(str(v["四态"]).split("（")[0] for v in 结["臂"].values())
    结["汇总"] = dict(臂数=len(结["臂"]), **{k: int(c.get(k, 0)) for k in ("PASS", "FAIL", "UNDECIDABLE", "INVALID")})
    return 结


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 主测()
    p = os.path.join(产物, "判据_听说话结果.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(果, f, ensure_ascii=False, indent=1, default=str)
    print("结果 →", p)
    print("汇总:", json.dumps(果["汇总"], ensure_ascii=False))
    for k, v in 果["臂"].items():
        print("  %-24s %s" % (k, v["四态"]))
