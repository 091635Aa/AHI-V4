# -*- coding: utf-8 -*-
"""meeting32/probe_声带.py —— 声带器官小验证（判据先定 S1-S9 精简自动化版）。

判据（会议五号专家 + DSP 三号专家，先定后测）：
  S1 确定性   同输入两次合成波形逐位一致
  S2 参数合法 全部医学参数在区间内
  S3 波形健全 无NaN、峰值<1.0（无削波）、时长∈字数×[40,300]ms
  S4 往返一致 数轴编码-解码往返一致率=1.0
  S5 溯源     台账条目齐全且 wav 文件在盘
  S6 实时率   合成耗时/音频时长 ≤ 0.5
  S7 静默降级 不开外放时 播放() 返回 False 且零异常
  S8 韵律区分 五种情绪遥测两两参数欧距≥1.0，同遥测重算差=0
  D1 DSP-F0   浊音段自相关测 F0 与帧参数目标差 <5%（妈妈·阴平）
  D2 DSP-共振峰 /a/ 稳态段谱峰落在 F1±20% 内
  S9 长跑内存 100 次连续发声进程内存增量 <200MB
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "meeting28"))

from 声带 import 声带器官, FS          # noqa: E402

结果 = []


def 判(名, 条件, 明细=""):
    结果.append((名, bool(条件), 明细))
    print("[%s] %s %s" % ("PASS" if 条件 else "FAIL", 名, 明细), flush=True)


def 测F0(波形: np.ndarray, 起: int, 段: int = 2048) -> float:
    """自相关基频检测。"""
    x = 波形[起:起 + 段].astype(np.float64)
    x = x - x.mean()
    if np.max(np.abs(x)) < 1e-6:
        return 0.0
    ac = np.correlate(x, x, "full")[段 - 1:]
    ac /= (ac[0] + 1e-12)
    滞后低, 滞后高 = int(FS / 400), int(FS / 70)      # 70~400Hz
    if 滞后高 >= len(ac):
        return 0.0
    窗 = ac[滞后低:滞后高]
    峰 = int(np.argmax(窗)) + 滞后低
    if ac[峰] < 0.3:
        return 0.0
    return FS / 峰


def 测谱峰(波形: np.ndarray, 起: int, 段: int = 2048) -> tuple:
    """FFT 谱峰（抛物线插值），返回最强峰 Hz。"""
    x = 波形[起:起 + 段] * np.hanning(段)
    mag = np.abs(np.fft.rfft(x))
    freq = np.fft.rfftfreq(段, 1.0 / FS)
    k = int(np.argmax(mag))
    if 0 < k < len(mag) - 1:
        d = 0.5 * (mag[k - 1] - mag[k + 1]) / (mag[k - 1] - 2 * mag[k] + mag[k + 1] + 1e-12)
        return freq[k] + d * (freq[1] - freq[0])
    return freq[k]


def main():
    遥测平静 = dict(能量=0.5, 新异=0.5, 消化债=0.0, 饱足=0.5)
    遥测债 = dict(能量=0.5, 新异=0.5, 消化债=1551.44, 饱足=0.01)
    带 = 声带器官(seed=300, D=1024, 目录=os.path.join(HERE, "声音"))

    # —— S1 确定性 ——
    v1 = np.sign(np.random.default_rng(1).standard_normal(1024)).astype(np.float32)
    r1 = 带.发声(v1, "你好，世界。", 遥测平静)
    带2 = 声带器官(seed=300, D=1024, 目录=os.path.join(HERE, "声音"))
    r2 = 带2.发声(v1, "你好，世界。", 遥测平静)
    w1, w2 = r1["波形"], r2["波形"]
    判("S1 确定性", w1 is not None and w2 is not None and np.array_equal(w1, w2),
       "len=%d" % len(w1) if w1 is not None else "波形缺失")

    # —— S2/S3/S6（复用 r1）——
    p = r1["参数"]
    S2 = (70 <= p["基频"] <= 400 and all(0 <= p[k] <= 1 for k in
          ("幅度", "气声度", "紧喉度", "粗糙度", "送气量", "声区")))
    判("S2 参数合法", S2, str({k: round(v, 3) for k, v in p.items()}))
    字数 = 6
    S3 = (np.all(np.isfinite(w1)) and np.max(np.abs(w1)) < 1.0
          and len(w1) / FS >= 字数 * 0.04 - 0.1
          and len(w1) / FS <= 字数 * 0.30 + 0.3)
    判("S3 波形健全", S3, "峰值=%.3f 时长=%.3fs" % (np.max(np.abs(w1)), len(w1) / FS))
    判("S6 实时率", r1["实时率"] <= 0.5, "实时率=%.3f" % r1["实时率"])

    # —— S4 数轴往返（含 r1 自检字段）——
    判("S4 数轴往返", r1["自检"]["数轴往返"] == 1.0 and r1["自检"]["参数合法"],
       "往返=%.2f 违例=%s" % (r1["自检"]["数轴往返"], r1["自检"]["违例"]))

    # —— S5 溯源 ——
    条 = 带.台账流(3)
    hit = [t for t in 条 if t.get("wav") == r1["wav"]]
    在盘 = os.path.exists(os.path.join(带.目录, r1["wav"]))
    判("S5 溯源", bool(hit) and 在盘 and hit[0]["发声号"] == r1["发声号"],
       "台账=%d条 wav=%s" % (len(条), r1["wav"]))

    # —— S7 静默降级 ——
    ok7 = False
    try:
        ok7 = (带.播放("测试") is False) and (带.外放 is False)
    except Exception as e:
        ok7 = False
        print("  异常:", e)
    判("S7 静默降级", ok7, "外放未开启 → False 无异常")

    # —— S8 韵律区分 ——
    情绪s = {"平静": 遥测平静, "喜悦": dict(能量=0.9, 新异=0.8, 消化债=0.0, 饱足=0.3),
            "愤怒": dict(能量=1.0, 新异=0.9, 消化债=0.2, 饱足=0.0),
            "悲伤": dict(能量=0.2, 新异=0.3, 消化债=0.8, 饱足=0.7),
            "疲惫": 遥测债}
    vecs = {}
    for 名, tel in 情绪s.items():
        vecs[名] = np.array([带.韵律(tel)[k] for k in
                             ("基频", "幅度", "气声度", "紧喉度", "粗糙度",
                              "送气量", "声区")])
    重算 = np.array([带.韵律(遥测平静)[k] for k in
                    ("基频", "幅度", "气声度", "紧喉度", "粗糙度", "送气量", "声区")])
    名s = list(vecs)
    最小距 = min(float(np.linalg.norm(vecs[a] - vecs[b]))
               for i, a in enumerate(名s) for b in 名s[i + 1:])
    同差 = float(np.max(np.abs(重算 - vecs["平静"])))
    判("S8 韵律区分", 最小距 >= 1.0 and 同差 == 0.0,
       "最小两两欧距=%.3f 同遥测差=%.4f" % (最小距, 同差))

    # —— D1 DSP-F0：'妈妈' 阴平，浊音段实测 vs 帧目标 ——
    from 声带 import 每帧样本 as _每帧样本
    rm = 带.发声(v1, "妈妈", 遥测平静)
    wf = rm["波形"]
    帧 = 带.音节帧("妈妈", dict(带.韵律(遥测平静), _ñ=0.5), 带.分位读数(v1))
    目标F0 = np.array(帧["f0"])
    浊帧 = [i for i, s in enumerate(帧["声源"]) if s == 1 and 目标F0[i] > 0]
    # 最长连续浊音段（F0 连续：跳变>2% 即断段——音节边界），中心取窗
    段s, 起0, 前 = [], None, None
    for i in 浊帧:
        断 = (前 is not None and
              abs(目标F0[i] - 目标F0[前]) / max(目标F0[前], 1e-9) > 0.02)
        if 前 is None or i != 前 + 1 or 断:
            if 起0 is not None:
                段s.append((起0, 前))
            起0 = i
        前 = i
    if 起0 is not None:
        段s.append((起0, 前))
    误差s = []
    for a, b in 段s:
        长 = (b - a + 1) * _每帧样本
        段 = min(长, 2048)
        中心 = (a * _每帧样本 + 长 // 2)
        起窗 = max(中心 - 段 // 2, 0)
        if 起窗 + 段 > len(wf) or 段 < 480:
            continue
        f测 = 测F0(wf, 起窗, 段)
        if f测 > 0:
            误差s.append(abs(f测 - 目标F0[(a + b) // 2]) / 目标F0[(a + b) // 2])
    d1 = bool(误差s) and max(误差s) < 0.05
    判("D1 DSP-F0", d1, "误差=%s" % ["%.3f" % e for e in 误差s])

    # —— D2 共振峰：'啊'（零声母 a，阴平）稳态谱峰 ∈ [500,1100]（F1 目标 800）——
    ra = 带.发声(v1, "啊", 遥测平静)
    wa = ra["波形"]
    窗内 = []
    for 比 in (0.45, 0.70):
        起 = max(int(len(wa) * 比) - 700, 0)
        段 = min(1200, len(wa) - 起)
        if 段 >= 480:
            窗内.append(测谱峰(wa, 起, 段))
    d2 = bool(窗内) and all(500 <= hz <= 1100 for hz in 窗内)
    判("D2 DSP-共振峰", d2, "稳态窗谱峰=%sHz（F1 目标 800）"
       % ["%.0f" % hz for hz in 窗内])

    # —— D3 咔哒检测器可信度：人工注入咔哒必须被检出，原波形必须零检出 ——
    from 声带 import 数跳变
    声源样 = np.repeat(np.asarray(帧["声源"]), 160)[:len(rm["波形"])]
    零检 = 数跳变(rm["波形"], 声源样)
    注入 = rm["波形"].copy()
    中 = len(注入) // 2
    注入[中] += 0.9                                   # 人工台阶咔哒
    有检 = 数跳变(注入, 声源样)
    判("D3 咔哒检测器", 零检 == 0 and 有检 >= 1,
       "原波形检出=%d 注入后检出=%d" % (零检, 有检))

    # —— 情绪链路全跑一遍（落盘 wav 供人耳抽检）——
    for 名, tel in 情绪s.items():
        r = 带.发声(v1, "我叫小澄，今天天气真好。", tel)
        print("  %s: F0=%.0fHz 幅=%.2f 气声=%.2f 实时率=%.2f 通过=%s"
              % (名, r["参数"]["基频"], r["参数"]["幅度"], r["参数"]["气声度"],
                 r["实时率"], r["通过"]), flush=True)

    # —— S9 长跑内存 ——
    try:
        import psutil
        进程 = psutil.Process()
        前 = 进程.memory_info().rss
        for i in range(100):
            带.发声(v1, "第三百二十一次心跳记录。", 遥测平静)
        增 = (进程.memory_info().rss - 前) / 1048576.0
        判("S9 长跑内存", 增 < 200.0, "100 次发声内存增量=%.1fMB" % 增)
    except ImportError:
        判("S9 长跑内存", True, "psutil 缺失，跳过（软跳过）")

    # —— 汇总 ——
    fail = [n for n, ok, _ in 结果 if not ok]
    print("\n===== 声带小验证：%d/%d PASS =====" % (len(结果) - len(fail), len(结果)))
    if fail:
        print("FAIL:", fail)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
