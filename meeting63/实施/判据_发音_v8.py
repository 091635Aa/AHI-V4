# -*- coding: utf-8 -*-
"""meeting63/实施/判据_发音_v8.py —— 咬字修复版发音判据（保号保定义 ＋ 新增「自听」臂）。

纯新增件（C14）：只读 import meeting61/62 既有件与其判据**辅助函数**；只写 `meeting63/产物/判据/`。

保号保定义：本件**沿用 `判据_发音_v7.py` 的量法与阈值**（阈值一字不改）在 **v8 产出**上重跑
  A2/A3、B1–B4、C1–C3、D1–D3、E1–E4、E6、F1–F3、P0（外部真人正对照·准入门）、M1/M2（must-fail）。
本轮**预期变更**（新增件，如实登记，非回退）：`E7_噪帧带分布未变`、`F4_噪声道逐位同v5` 是 v7 为
  「辅音不得改」设的护栏；**本轮的目标恰是改辅音实现** ⇒ 这两条按定义会变 ⇒ 登记为「预期变更」。

新增「自听/反馈」臂（本轮的 参考+反馈学习 落点，全部零外挂）
  · G1_耳内生切段确定性        —— 同波形两趟逐位同
  · G2_皮层可分度              —— HDC 谐振音素格 留一音节 ≥3×chance，含 must-fail（置乱/直读/白噪）
  · G3_波形敏感性              —— 真耳只认波形（时间反转 ⇒ 特征必变）≥0.8
  · G4_零外挂审计              —— 静态 AST＋运行期钩子＋确定性（全过）
  · G5_参考距                  —— 合成辅音区统计 vs 真人参考（耳量出）归一化距离
确定性（C13）：同种子两趟逐位同（E6）。纯 numpy/scipy + 本仓算子（C1/C9/C10）。
用法：`python 判据_发音_v8.py` → 打印 + 写 产物/判据/判据_发音_v8.json
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M63 = os.path.dirname(HERE)
AHI4 = os.path.dirname(M63)
MEET61 = os.path.join(AHI4, "meeting61")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AHI4, "meeting62", "实施"))
sys.path.insert(0, os.path.join(MEET61, "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))

import 音韵_v5 as Y  # noqa: E402
import 韵律_v8 as H8  # noqa: E402
import 引擎_v7 as E7  # noqa: E402
import 引擎_v5 as E5  # noqa: E402
import 韵律_v6 as H6  # noqa: E402
import 判据_发音_v7 as C7  # noqa: E402  只读：复用保号保定义的量法
import 声线覆盖层_v7 as 覆  # noqa: E402
import 耳_内生 as 耳  # noqa: E402
import 皮层_音素格 as 皮  # noqa: E402
import 学_咬字反馈 as 学  # noqa: E402

FS = 44100
单韵 = C7.单韵
出目录 = os.path.join(M63, "产物", "判据")
os.makedirs(出目录, exist_ok=True)
_档 = lambda 名="EmoCompanion": C7.读档(名)


def 音节(拼, 档, 种子=300):
    """与 `判据_发音_v7.合成音节` 同口径：**汉字** → 说()（v7 亦用 说）。"""
    w, 元, 帧 = H8.人类韵律_v8(档, 种子).说(拼)
    return w, 帧


def 句(句文, 档, 种子=300):
    return H8.人类韵律_v8(档, 种子).说(句文)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    np.seterr(all="ignore")
    档 = _档("EmoCompanion")
    臂 = {}

    # ---------------- A 音韵映射（与引擎无关，保号保定义） ----------------
    a2 = {"zhi1": "iii", "chi1": "iii", "shi1": "iii", "ri4": "iii",
          "zi1": "ii", "ci4": "ii", "si1": "ii"}
    ok2 = all(Y.解析音节(k)["韵"] == v for k, v in a2.items())
    f2_shi = Y.解析音节("shi1")["目标s"][0][1]
    f2_i = Y.解析音节("yi1")["目标s"][0][1]
    臂["A2_舌尖元音"] = dict(通过=bool(ok2 and f2_i - f2_shi > 400),
                         读数=dict(映射={k: Y.解析音节(k)["韵"] for k in a2},
                                 F2_shi=f2_shi, F2_i=f2_i))
    文本s = ["你好，我是EmoCompanion。", "今天的风很轻，适合慢慢地说一会儿话。",
             "你最近，过得好吗？", "嘿，我是SpeakerA。",
             "别看我平时闹得欢，认真说话的时候，我也能安静下来。",
             "要不要，一起待一会儿？", "大家好，我是洛天依。",
             "谢谢你愿意听我说话。", "希望今天的声音，能让你心里暖一点。",
             "语言不只是声音，它把一个人的温度，送到另一个人的心里。"]
    审 = Y.审计(文本s)
    臂["A3_全句查表"] = dict(通过=bool(审["问题数"] == 0),
                         读数=dict(音节数=审["音节数"], 问题数=审["问题数"], 问题=审["问题"][:8]))

    # ---------------- B 辅音实现（保号保定义；v8 改辅音，按同阈值重跑） ----------------
    字 = {"sa": "撒", "sha": "沙", "xia": "虾", "fa": "发", "ha": "哈",
         "ma": "妈", "na": "拿", "la": "拉", "ba": "八", "da": "搭", "ga": "嘎",
         "pa": "趴", "ta": "他", "ka": "咖"}
    辅 = {}
    for p in 字:
        w, 帧 = 音节(字[p], 档, 300)
        噪 = C7._辅音样本(w, 帧)
        浊 = C7.分段(w, 帧, {1})
        辅[p] = dict(噪ms=round(C7._辅音帧数(帧) * 10, 1),
                    清浊差dB=round(C7._db(C7._rms(噪), C7._rms(浊)), 2) if len(噪) and len(浊) else None,
                    噪重心Hz=round(C7._谱重心(噪), 0))
    部位序 = [辅[p]["噪重心Hz"] for p in ("sa", "xia", "sha")]
    f弱 = (辅["fa"]["清浊差dB"] < 辅["sha"]["清浊差dB"] - 0.5)
    臂["B1_擦音部位序"] = dict(通过=bool(部位序[0] > 部位序[1] > 部位序[2] and f弱),
                          读数=dict(顺序="s>x>sh", 重心Hz=dict(zip(("sa", "xia", "sha"), 部位序)),
                                   f重心Hz=辅["fa"]["噪重心Hz"], f弱于sh=bool(f弱)))
    强擦 = [辅[p]["清浊差dB"] for p in ("sa", "sha", "xia") if 辅[p]["清浊差dB"] is not None]
    弱擦 = 辅["ha"]["清浊差dB"] if 辅["ha"]["清浊差dB"] is not None else -99
    臂["B2_擦音电平"] = dict(通过=bool(强擦 and all(-16.0 <= d <= -2.0 for d in 强擦) and -24.0 <= 弱擦 <= -5.0),
                         读数=dict(强擦dB=强擦, ha_dB=弱擦))
    浊母 = {}
    for p in ("ma", "na", "la"):
        w, 帧 = 音节(字[p], 档, 300)
        idx = [i for i, t in enumerate(帧["声源"]) if int(t) == 1]
        鼻 = w[idx[0] * 441:(idx[0] + 4) * 441] if len(idx) >= 4 else np.zeros(0)
        元 = w[(idx[0] + 9) * 441:(idx[0] + 14) * 441] if len(idx) >= 14 else np.zeros(0)
        if len(鼻) and len(元):
            浊母[p] = round(C7._db(C7._rms(鼻), C7._rms(元)), 2)
    臂["B3_浊声母电平"] = dict(通过=bool(浊母 and all(-16.0 <= d <= -3.0 for d in 浊母.values())),
                           读数=dict(dB=浊母))
    对 = {"p": "b", "t": "d", "k": "g"}
    asp = {q: 辅[q + "a"]["噪ms"] for q in "ptk"}
    vot差 = {q: 辅[q + "a"]["噪ms"] for q in "bdg"}
    臂["B4_送气对比"] = dict(通过=bool(all(asp[q] >= 2.0 * max(vot差[对[q]], 10.0) for q in "ptk")),
                         读数=dict(不送气ms=vot差, 送气ms=asp))

    # ---------------- C CV 协同发音（保号保定义） ----------------
    期望起 = {"ba": 1027, "pa": 1027, "da": 1522, "ta": 1522, "ga": 1687, "ka": 1687}
    C = {}
    for p in ("ba", "da", "ga", "pa", "ta", "ka"):
        w, 帧 = 音节(字[p], 档, 300)
        浊idx = [i for i, t in enumerate(帧["声源"]) if int(t) == 1]
        if not 浊idx:
            continue
        a0 = 浊idx[0] * 441
        起 = w[a0:a0 + int(0.040 * FS)]
        中 = w[(a0 + int(0.06 * FS)):(a0 + int(0.11 * FS))]
        C[p] = dict(F2起Hz=round(C7._目标峰(起, 期望起[p]), 1),
                    F2中Hz=round(C7._目标峰(中, 1305.0), 1))
    臂["C1_塞音F2轨迹"] = dict(通过=bool(C["ba"]["F2起Hz"] < C["da"]["F2起Hz"] < C["ga"]["F2起Hz"]),
                          读数=dict(顺序="b<d<g", F2起={k: C[k]["F2起Hz"] for k in ("ba", "da", "ga")}))
    转 = max(abs(C[p]["F2起Hz"] - C[p]["F2中Hz"]) for p in C) if C else 0.0
    臂["C2_CV过渡幅度"] = dict(通过=bool(转 >= 80.0), 读数=dict(最大ΔF2Hz=round(转, 1), 逐音节=C))
    调测 = {}
    for 拼 in ("ma1", "ma2", "ma3", "ma4"):
        w, _, _ = H8.人类韵律_v8(档, 300).说音节(拼)
        t = C7._三段F0(w)
        调测[拼] = None if t is None else dict(中st=round(C7._半音(t[1], t[0]), 2),
                                            末st=round(C7._半音(t[2], t[0]), 2))
    四声ok = False
    if all(调测.get(k) for k in ("ma1", "ma2", "ma3", "ma4")):
        四声ok = bool(abs(调测["ma1"]["末st"]) < 0.8 and 调测["ma2"]["末st"] >= 1.2
                      and 调测["ma3"]["中st"] <= -0.4
                      and 调测["ma3"]["末st"] - 调测["ma3"]["中st"] >= 1.2
                      and 调测["ma4"]["末st"] <= -2.0)
    臂["C3_四声轮廓"] = dict(通过=四声ok, 读数=调测)

    # ---------------- D 元音空间（保号保定义；120Hz 派生声线） ----------------
    档低 = 档.派生(基频=120.0)
    V = {}
    for p, 期望, 字p in (("a", (825, 1305), "啊"), ("yi", (330, 2349), "衣"),
                        ("wu", (352, 751), "乌"), ("yu", (330, 1931), "迂")):
        w, 帧 = 音节(字p, 档低, 300)
        浊idx = [i for i, t in enumerate(帧["声源"]) if int(t) == 1]
        a = (浊idx[len(浊idx) // 3] if 浊idx else 0) * 441
        seg = w[a:a + int(0.08 * FS)]
        f1 = C7._目标峰(seg, 期望[0], 0.12)
        f2 = C7._目标峰(seg, 期望[1], 0.12)
        V[p] = dict(F1=round(f1, 1), F2=round(f2, 1),
                    误F1=round(abs(f1 - 期望[0]) / 期望[0], 3),
                    误F2=round(abs(f2 - 期望[1]) / 期望[1], 3))
    误 = max(max(v["误F1"], v["误F2"]) for v in V.values())
    臂["D1_元音峰位"] = dict(通过=bool(误 < 0.15), 读数=dict(测试F0=120, 最大误差=round(误, 3), 元音=V))
    d_ai = math.hypot(V["a"]["F1"] - V["yi"]["F1"], V["a"]["F2"] - V["yi"]["F2"])
    d_au = math.hypot(V["a"]["F1"] - V["wu"]["F1"], V["a"]["F2"] - V["wu"]["F2"])
    d_iu = math.hypot(V["yi"]["F1"] - V["yu"]["F1"], V["yi"]["F2"] - V["yu"]["F2"])
    臂["D2_元音可分"] = dict(通过=bool(d_ai > 600 and d_au > 500 and d_iu > 250),
                         读数=dict(a_i=round(d_ai), a_u=round(d_au), i_v=round(d_iu)))
    N = {}
    for p, 字p in (("an", "安"), ("ang", "昂")):
        w, 帧 = 音节(字p, 档, 300)
        浊idx = [i for i, t in enumerate(帧["声源"]) if int(t) == 1]
        i0 = 浊idx[int(len(浊idx) * 0.72)]
        seg = w[i0 * 441:(i0 * 441) + int(0.06 * FS)]
        N[p] = round(C7._目标峰(seg, 1670.0 if p == "an" else 1148.0), 1)
    臂["D3_鼻尾区别"] = dict(通过=bool(abs(N["an"] - N["ang"]) >= 200),
                         读数=dict(an=N["an"], ang=N["ang"], 差=round(N["an"] - N["ang"], 1)))

    # ---------------- E 语流波形（保号保定义） ----------------
    句s, 句波形 = {}, {}
    for i, s in enumerate(("你好，我是EmoCompanion。", "今天的风很轻，适合慢慢地说一会儿话。",
                           "语言不只是声音，它把一个人的温度，送到另一个人的心里。")):
        w, 元, 帧 = 句(s, 档, 300 + i)
        句波形[s] = w
        n音节 = len(Y.音节列表(s))
        噪 = C7.分段(w, 帧, {0})
        浊 = C7.分段(w, 帧, {1})
        句s[s[:10]] = dict(时长=round(len(w) / FS, 2), 音节数=n音节,
                         音节率=round(n音节 / (len(w) / FS), 2),
                         清浊差dB=round(C7._db(C7._rms(噪), C7._rms(浊)), 2), **C7.句级统计(w, 帧))
    臂["E1_静音占比"] = dict(通过=bool(all(v["静音占比"] < 0.20 for v in 句s.values())),
                         读数={k: v["静音占比"] for k, v in 句s.items()})
    臂["E2_清浊电平差"] = dict(通过=bool(all(-18.0 <= v["清浊差dB"] <= -4.0 for v in 句s.values())),
                           读数={k: v["清浊差dB"] for k, v in 句s.items()})
    臂["E3_包络动态"] = dict(通过=bool(all(v["包络标准差dB"] <= 35.0 for v in 句s.values())),
                         读数={k: v["包络标准差dB"] for k, v in 句s.items()})
    臂["E4_音节率"] = dict(通过=bool(all(3.8 <= v["音节率"] <= 5.8 for v in 句s.values())),
                        读数={k: v["音节率"] for k, v in 句s.items()})
    wa, _, _ = 句("你好，我是EmoCompanion。", 档, 300)
    wb, _, _ = 句("你好，我是EmoCompanion。", 档, 300)
    wc, _, _ = 句("你好，我是EmoCompanion。", 档, 301)
    同 = bool(np.array_equal(wa, wb))
    异 = bool(not np.array_equal(wa, wc))
    臂["E6_确定性"] = dict(通过=bool(同 and 异), 读数=dict(两趟同=同, 异种子异=异))

    # ---------------- F 耳听等价（保号保定义，v6/v7 口径） ----------------
    def _峰与台(x, f1):
        f, env = C7._包络(x, FS, lif=0.0022)
        m = (f >= 150.0) & (f <= 3500.0)
        峰位 = float(f[m][int(np.argmax(env[m]))])
        lo = (f >= 150.0) & (f < f1 * 0.75)
        台 = float(np.median(env[lo])) if np.any(lo) else 0.0
        return 峰位, round(20.0 * math.log10(max(env[m].max(), 1e-12) / max(台, 1e-12)), 1)

    样 = (("啊", "a1"), ("哦", "o1"), ("鹅", "e1"), ("衣", "i1"), ("乌", "u1"),
         ("迂", "v1"), ("儿", "er2"), ("好", "hao3"), ("我", "wo3"), ("是", "shi4"),
         ("天", "tian1"), ("慢", "man4"), ("说", "shuo1"))
    FF = {}
    for 字c, 拼 in 样:
        s = Y.解析音节(拼)
        目s = s["目标s"]
        单元 = len(目s) == 1 and s["尾"] is None
        if 单元:
            f1t, f2t = 目s[0][0] * 档.共振峰缩放[0], 目s[0][1] * 档.共振峰缩放[1]
            窗 = (f1t * 0.85, f2t * 1.15)
        else:
            f1t = min(t[0] for t in 目s) * 档.共振峰缩放[0]
            f2t = max(t[1] for t in 目s) * 档.共振峰缩放[1]
            窗 = (f1t * 0.85, f2t * 1.15)
        帧 = H8.人类韵律_v8(档低, 300).建帧(字c)
        idx = [i for i, t in enumerate(帧["声源"]) if int(t) == 1]
        a0 = (idx[len(idx) // 3] if idx else 0) * 441
        x8, _, _ = 句(字c, 档低, 300)
        x5 = E5.合成(帧, 音色=H6.人类韵律_v5(档低, 300).音色(), 种子=300)[0]
        seg8 = x8[a0:a0 + int(0.12 * FS)]
        seg5 = x5[a0:a0 + int(0.12 * FS)]
        峰8, 突8 = _峰与台(seg8, 目s[0][0] * 档.共振峰缩放[0]) if len(seg8) > 512 else (0.0, 0.0)
        峰5, 突5 = _峰与台(seg5, 目s[0][0] * 档.共振峰缩放[0]) if len(seg5) > 512 else (0.0, 0.0)
        FF[拼] = dict(窗=[round(窗[0], 1), round(窗[1], 1)], 首目标F1=round(目s[0][0] * 档.共振峰缩放[0], 1),
                      v8包络峰=round(峰8, 1), v5包络峰=round(峰5, 1),
                      v8突出dB=突8, v8落位=bool(窗[0] <= 峰8 <= 窗[1]),
                      v5落位=bool(窗[0] <= 峰5 <= 窗[1]))
    全落 = all(v["v8落位"] for v in FF.values())
    落5 = sum(1 for v in FF.values() if v["v5落位"])
    臂["F1_元音谱包络峰位"] = dict(通过=bool(全落 and 落5 < len(FF)),
                              读数=dict(v8落位="%d/%d" % (sum(1 for v in FF.values() if v["v8落位"]), len(FF)),
                                      v5落位="%d/%d" % (落5, len(FF)), 逐音节=FF))
    臂["F2_元音F1F2区突出"] = dict(
        通过=bool(all(v["v8突出dB"] >= 3.0 for v in FF.values() if v["首目标F1"] >= 500.0)),
        读数=dict(v8={k: v["v8突出dB"] for k, v in FF.items()}))

    # ---------------- 本轮预期变更（v7 护栏臂，如实登记，不计入回退） ----------------
    帧缘 = H8.人类韵律_v8(档, 300).建帧("你好，我是EmoCompanion。")
    w缘8 = 句波形["你好，我是EmoCompanion。"]
    w缘6 = E7.合成(帧缘, 音色=H6.人类韵律_v5(档, 300).音色(), 种子=300)[0]
    噪6 = C7._谱带(C7.分段(w缘6, 帧缘, {0}))
    噪8 = C7._谱带(C7.分段(w缘8, 帧缘, {0}))
    噪变 = float(sum(abs(噪6[b] - 噪8[b]) for b in 噪6))
    臂["E7_噪帧带分布未变"] = dict(通过=bool(噪变 < 1e-9),
                               读数=dict(预期="本轮**预期变更**（辅音实现即改动目标）", v6噪=噪6, v8噪=噪8,
                                       噪L1=round(噪变, 6)))
    噪帧 = dict(f0=[180.0] * 20, f1=[300.0] * 20, f2=[1800.0] * 20, f3=[2600.0] * 20,
               f4=[3500.0] * 20, f5=[4500.0] * 20, 幅=[0.8] * 20, 气声=[0.0] * 20,
               声源=[0] * 20, 噪带=[(6800.0, 2200.0)] * 20, 送气=[1.0] * 20,
               紧喉=[0.3] * 20, 粗糙=[0.0] * 20, 噪形=[0] * 20)
    y5 = E5.合成(噪帧, 音色=None, 种子=300)[0]
    y8 = E7.合成(噪帧, 音色=None, 种子=300)[0]
    臂["F4_噪声道逐位同v5"] = dict(通过=bool(np.array_equal(y5, y8)),
                              读数=dict(预期="本轮**预期变更**（噪声道即改动目标）",
                                      逐位同=bool(np.array_equal(y5, y8))))

    # ---------------- must-fail（保号保定义） ----------------
    rng = np.random.default_rng(62019)
    白噪 = rng.standard_normal(int(0.30 * FS))
    静音 = np.zeros(int(0.30 * FS))
    子窗 = {p: (Y.解析音节(p)["目标s"][0][0] * 档.共振峰缩放[0] * 0.85,
               Y.解析音节(p)["目标s"][0][1] * 档.共振峰缩放[1] * 1.15) for _, p in 单韵}

    def sham落位(x):
        起s = C7._voiced起始(x)
        if len(起s) == 0:
            return dict(浊帧=0, 落位=False, 峰=None)
        seg = x[int(起s[0]):int(起s[0]) + int(0.12 * FS)]
        if len(seg) < 512:
            return dict(浊帧=len(起s), 落位=False, 峰=None)
        f, env = C7._包络(seg, FS, lif=0.0022)
        m = (f >= 150.0) & (f <= 3500.0)
        p = float(f[m][int(np.argmax(env[m]))])
        return dict(浊帧=int(len(起s)), 峰=round(p, 1),
                    落位=bool(any(lo <= p <= hi for lo, hi in 子窗.values())))
    m1 = sham落位(白噪)
    m2 = sham落位(静音)
    臂["M1_白噪sham"] = dict(通过=bool(not m1["落位"]), 读数=dict(说明="白噪须 FAIL", **m1))
    臂["M2_静音sham"] = dict(通过=bool(not m2["落位"]), 读数=dict(说明="静音须 FAIL", **m2))
    # M3 v8 波形敏感性（新增 must-fail 口径）
    敏 = 学.波形敏感()
    臂["M3_波形敏感(真耳只认波形)"] = dict(通过=bool(敏["敏感率"] >= 0.80), 读数=敏)

    # ---------------- 准入门：外部真人正对照（保号保定义） ----------------
    真人, 真峰, 真突 = {}, [], []
    import glob
    for 名 in ("EmoCompanion", "SpeakerA", "洛天依"):
        for p in sorted(glob.glob(os.path.join(MEET61, "素材", 名, "ref44_*.wav"))):
            x = C7._读wav(p)
            峰s, 突s = C7.峰位统计(x)
            if not 峰s:
                真人[os.path.basename(p)] = dict(通过=False, 浊帧=0)
                continue
            中位, 突 = float(np.median(峰s)), float(np.median(突s))
            真人[os.path.basename(p)] = dict(通过=bool(len(峰s) >= 30 and 180.0 <= 中位 <= 1300.0 and 突 >= 3.0),
                                          浊帧=len(峰s), 峰位中位Hz=round(中位, 1), F1区突出中位dB=round(突, 2))
            真峰.append(中位); 真突.append(突)
    臂["P0_外部真人正对照"] = dict(通过=bool(真人 and all(v["通过"] for v in 真人.values())),
                              读数=dict(文件数=len(真人), 逐文件=真人))

    # ---------------- 新增「自听/反馈」臂 ----------------
    耳1 = 耳.听(wa, 含段=True)
    耳2 = 耳.听(wa, 含段=True)
    臂["G1_耳内生切段确定性"] = dict(通过=bool(耳1["段"] == 耳2["段"]),
                                读数=dict(段数=耳1["段数"], 清中位ms=耳1["清时长中位ms"],
                                        浊中位ms=耳1["浊时长中位ms"], 清占比=耳1["清总占比"]))
    果皮 = 皮.跑(韵律模块=H8)
    臂["G2_皮层可分度"] = dict(通过=bool(果皮["通过"] and 果皮["P1_准确率达标"]),
                           读数=dict(协议=果皮["协议"], 准确率=果皮["准确率"], chance=果皮["chance"],
                                    混淆=果皮["混淆矩阵"], 置乱=果皮["置乱准确率"],
                                    直读=果皮["直读准确率"], 白噪=果皮["白噪准确率"]))
    学参 = {}
    精 = None
    if os.path.exists(os.path.join(M63, "产物", "读数_学咬字反馈.json")):
        精 = json.load(open(os.path.join(M63, "产物", "读数_学咬字反馈.json"), encoding="utf-8"))
        学参 = dict(基线可分度=精["基线可分度"], 搜索后=精["搜索后可分度"], 学到θ=精["学到θ"],
                   参考=精["参考"], 波形敏感=精["波形敏感"])
    臂["G3_反馈学习(参考+反馈)"] = dict(通过=bool(精 and 精["搜索后可分度"] >= 精["基线可分度"]),
                                读数=学参 or dict(说明="读数_学咬字反馈.json 缺（先跑 学_咬字反馈.py）"))
    import 审计_零外挂 as 审
    静2 = 审.静态扫()
    运2 = 审.运行期审计()
    臂["G4_零外挂审计"] = dict(通过=bool(运2["零外挂读入"] == 0 and 静2["禁词命中数"] == 0
                                    and 静2["禁包命中数"] == 0 and 运2["禁模块命中数"] == 0),
                           读数=dict(静态=静2, 运行期={k: v for k, v in 运2.items() if k != "全部读入"},
                                   全部读入=运2["全部读入"]))
    参 = 学.参考量()
    合清 = 耳.听(句("你好，我是EmoCompanion。", 档, 300)[0])["清时长中位ms"]
    臂["G5_参考距(真人锚定)"] = dict(
        通过=bool(abs(合清 - 参["清时长中位ms"]) / max(参["清时长中位ms"], 1.0) <= 0.60),
        读数=dict(真人参考=参, 合成清中位ms=合清,
                  距离=round(abs(合清 - 参["清时长中位ms"]) / max(参["清时长中位ms"], 1.0), 4)))

    # ---------------- 汇总 ----------------
    # 计分：把「本轮预期变更」两条排除在回退计数外，但如实列出
    预期变更 = ("E7_噪帧带分布未变", "F4_噪声道逐位同v5")
    通过数 = sum(1 for k, v in 臂.items() if v["通过"])
    未过 = [k for k, v in 臂.items() if not v["通过"]]
    报 = dict(工具="判据_发音_v8.py", 版本="v8-44k(咬字修复·学习θ)",
              臂数=len(臂), 通过=通过数, 未过=未过,
              预期变更=[k for k in 预期变更], 臂=臂,
              must_fail=dict(M1=臂["M1_白噪sham"]["通过"], M2=臂["M2_静音sham"]["通过"],
                             M3=臂["M3_波形敏感(真耳只认波形)"]["通过"]),
              准入门=dict(P0=臂["P0_外部真人正对照"]["通过"]),
              句级=句s)
    out = os.path.join(出目录, "判据_发音_v8.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)
    print("=" * 96)
    for k, v in 臂.items():
        print("%-24s %s  %s" % (k, "PASS" if v["通过"] else "CHG/FAIL",
                               json.dumps(v["读数"], ensure_ascii=False)[:120]))
    print("-" * 96)
    print("总计 %d/%d PASS | must-fail M1/M2/M3=%s/%s/%s | 真人准入门 P0=%s"
          % (通过数, len(臂), 臂["M1_白噪sham"]["通过"], 臂["M2_静音sham"]["通过"],
             臂["M3_波形敏感(真耳只认波形)"]["通过"], 臂["P0_外部真人正对照"]["通过"]))
    print("未过/预期变更:", 未过)
    print("JSON:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
