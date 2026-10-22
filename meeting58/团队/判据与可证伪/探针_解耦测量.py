# -*- coding: utf-8 -*-
"""meeting58/团队/判据与可证伪/探针_解耦测量.py —— 「角色复刻效果差」根因可分量测量探针。

口径（跑前冻结，见 R1_立场.md §2）：
  · 只读 import：`meeting55/实施/说_音色.py`、`meeting32/声带.py`、`挂载/角色包.py`、`挂载/说话_连贯.py`；
  · **只写本目录**（wav/台账落在 本目录/探针_产物/，跑完删除）；
  · CPU-only、无 GPU、无声卡、无网络；单次 <10s；
  · 阈值全部写死在代码里（不改结果），结果只报读数，不做 post-hoc 调参。

测四个可分量：
  T 文本层    ：三角色句子池大小 / 12 段×4 句的句重复率下界 / 池命中率
  C 音色层    ：4 档案同文本的 F0 中位、Δ半音、7 带相对谱包络余弦、声纹代理同/异接受率
  D 解耦缺失  ：8 个音色参数逐个 ±Δ 是否改变波形（有效出口数）；换音色时 F1/F2/F3 是否逐位同
  R 扮演崩塌  ：本轮不测（需冲突输入，见 R1 立场 §4 判据设计，未跑标 [提案]）
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
根 = r"<workspace>/03_AHI\AHI4"
sys.path.insert(0, os.path.join(根, "meeting55", "实施"))
sys.path.insert(0, os.path.join(根, "挂载"))
import 说_音色 as T  # noqa: E402
import 角色包 as 包模块  # noqa: E402
import 说话_连贯 as 说话  # noqa: E402

np.seterr(all="ignore")
产物 = os.path.join(HERE, "探针_产物")
os.makedirs(产物, exist_ok=True)
F0阈值_半音 = 2.0          # 冻结：两声纹间 F0 中位差 ≥2 半音才算「音高可分」
声纹阈值 = 0.20            # 冻结：声纹代理距离 d=1−cos，<0.20 判同人（meeting55 A2 同量级）
文本 = "你好，我在这里。"
文本2 = "今天过得怎么样？"
边16 = np.geomspace(80.0, 7600.0, 17)
边7 = np.geomspace(80.0, 7600.0, 8)


class 本器(T.声带发声):
    """把 wav/台账 改写到本目录（不改既有件、不污染 meeting55/产物）。"""

    def __init__(self, seed: int = 300, D: int = 1024):
        super().__init__(seed=seed, D=D)
        self.目录 = 产物
        self.台账路径 = os.path.join(产物, "探针_发声台账.jsonl")
        os.makedirs(self.目录, exist_ok=True)


器 = 本器()
向量 = np.sign(np.random.default_rng(55).standard_normal(1024)).astype(np.float32)
遥测 = dict(能量=0.5, 新异=0.5, 消化债=0.0, 饱足=0.5)


def 发声(档: dict, 文: str, 情绪: str = "平静", 语速=None):
    律 = T.情绪律(档, 情绪)
    if 语速 is not None:
        律["语速"] = float(语速)
    律["_ñ"] = 0.5
    器.当前律 = 律
    rec = 器.发声(向量, 文, 遥测)
    return rec, 律


def 谱带(x, 边):
    sr, n, hop = 16000, int(0.032 * 16000), int(0.016 * 16000)
    if len(x) < n:
        return None
    w = np.hanning(n)
    acc = np.zeros(n // 2 + 1)
    c = 0
    for i in range(0, len(x) - n + 1, hop):
        acc += np.abs(np.fft.rfft(x[i:i + n] * w)) ** 2
        c += 1
    acc /= max(c, 1)
    频 = np.fft.rfftfreq(n, 1.0 / sr)
    return np.array([acc[(频 >= lo) & (频 < hi)].sum() for lo, hi in zip(边[:-1], 边[1:])])


def 声纹(x) -> dict:
    sr = 16000
    带 = 谱带(x, 边7)
    带 = 带 / (带.sum() + 1e-9)
    seg = x[:min(len(x), sr * 2)]
    mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    频 = np.fft.rfftfreq(len(seg), 1.0 / sr)
    重心 = float(np.sum(频 * mag) / (np.sum(mag) + 1e-9))
    平坦 = float(np.exp(np.mean(np.log(mag + 1e-9))) / (np.mean(mag) + 1e-9))
    F0 = T._F0轨迹(x, sr)
    F0v = F0[np.isfinite(F0) & (F0 > 60)]
    return dict(带=带, 重心=重心, 平坦=平坦,
                F0=float(np.median(F0v)) if len(F0v) else 0.0,
                F0std=float(np.std(F0v)) if len(F0v) else 0.0,
                 voiced=float(len(F0v)))


def 距离(a: dict, b: dict) -> float:
    ca = np.concatenate([a["带"], [a["重心"] / 8000.0, a["平坦"] * 5.0, a["F0std"] / 50.0]])
    cb = np.concatenate([b["带"], [b["重心"] / 8000.0, b["平坦"] * 5.0, b["F0std"] / 50.0]])
    return float(1.0 - ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12))


def 半音(f1: float, f2: float) -> float:
    return float(12.0 * np.log2(max(f1, 1e-6) / max(f2, 1e-6)))


from math import gcd as _gcd  # noqa: E402


导入 = dict(音色档案=list(T.音色档案), 情绪=list(T.情绪偏移),
           参数表=list(T.声带模块.参数表) if hasattr(T, "声带模块") else None)
果 = dict(探针="meeting58 E4 根因可分量", 导入=导入,
          冻结阈值=dict(F0阈值_半音=F0阈值_半音, 声纹阈值=声纹阈值, 文本=文本, 文本2=文本2))

# ───────────── T 文本层 ─────────────
包们 = 包模块.全部包()


class _哑LM:
    """占位通用 LM：只提供 阶，`说()` 不调用通用分布（省掉 160KB 语料构建）。"""

    阶 = 5


说话器们 = {名: 说话.说话器(包, _哑LM(), 种子=53001) for 名, 包 in 包们.items()}
rng = np.random.default_rng(2026)
状态 = [np.clip(rng.standard_normal(64) * 0.9, -1.6, 1.6) for _ in range(12)]  # 替身 o用（真值来自 V5；本探针只测文本层）
T层 = {}
总句 = 总唯一 = 命中 = 0
for 名, 说 in 说话器们.items():
    池 = 说.句子池()
    文列 = [说.说(o, 拍=i) for i, o in enumerate(状态)]
    句列 = []
    for 文 in 文列:
        句列 += [s.strip() for s in 文.replace("！", "。").replace("？", "。").split("。") if s.strip()]
    唯一 = len(set(句列))
    步 = 1 if len(池) < 5 else 5
    每段句数 = len(句列) / max(1, len(文列))
    T层[名] = dict(池大小=len(池), 段数=len(文列), 总句数=len(句列), 唯一句=唯一,
                   每段句数=round(每段句数, 2),
                   选句步=步,
                   有效可达句=len(池) // _gcd(步, len(池)) if 池 else 0,
                   句重复率=round(1.0 - 唯一 / max(1, len(句列)), 4),
                   池命中率=round(sum(1 for s in 句列 if (s in 池 or s + "。" in 池)) / max(1, len(句列)), 4),
                   池可支撑不重复句数=len(池),
                   重复下界_48句=round(1.0 - min(len(池), 48) / 48.0, 4))
    总句 += len(句列)
    总唯一 += 唯一
    命中 += sum(1 for s in 句列 if (s in 池 or s + "。" in 池))
T层["汇总"] = dict(总句数=总句, 唯一句=总唯一,
                 句重复率=round(1.0 - 总唯一 / max(1, 总句), 4),
                 池命中率=round(命中 / max(1, 总句), 4))
果["T文本层"] = T层

# ───────────── C 音色层 ─────────────
C层 = dict(每档案={}, 两两={})
for 音, 档 in T.音色档案.items():
    rec1, _ = 发声(档, 文本)
    rec2, _ = 发声(档, 文本2)
    v1, v2 = 声纹(rec1["波形"]), 声纹(rec2["波形"])
    C层["每档案"][音] = dict(设定基频=档["基频"], 实测F0中位=round(v1["F0"], 1),
                            F0std=round(v1["F0std"], 1), 谱重心=round(v1["重心"], 1),
                           谱平坦=round(v1["平坦"], 4), 浊音帧=round(v1["voiced"], 1),
                           时长秒=round(len(rec1["波形"]) / 16000.0, 3),
                           自距_跨文本=round(距离(v1, v2), 4),
                           波形逐位同_重复调用=None)
名列 = list(T.音色档案)
成对 = []
for i in range(len(名列)):
    for j in range(i + 1, len(名列)):
        a, b = 名列[i], 名列[j]
        va, vb = C层["每档案"][a], C层["每档案"][b]
        ra, _ = 发声(T.音色档案[a], 文本)
        rb, _ = 发声(T.音色档案[b], 文本)
        va_, vb_ = 声纹(ra["波形"]), 声纹(rb["波形"])
        成对.append(dict(对=[a, b], Δ半音=round(abs(半音(va_["F0"], vb_["F0"])), 2),
                        d同文本=round(距离(va_, vb_), 4)))
C层["两两"] = 成对
同档 = [C层["每档案"][a]["自距_跨文本"] for a in 名列]
异档 = [x["d同文本"] for x in 成对]
C层["分离"] = dict(
    同档跨文本_d_中位=round(float(np.median(同档)), 4),
    同档跨文本_d_最大=round(float(np.max(同档)), 4),
    异档同文本_d_中位=round(float(np.median(异档)), 4),
    异档同文本_d_最小=round(float(np.min(异档)), 4),
    分离裕度=round(float(np.min(异档)) - float(np.max(同档)), 4),
    同人接受率=round(float(np.mean([d < 声纹阈值 for d in 同档])), 4),
    异人拒绝率=round(float(np.mean([d >= 声纹阈值 for d in 异档])), 4),
    判据="分离裕度>0 ⇒ 可分；meeting55 双约束=同人接受≥0.80 且 异人拒绝≥0.80",
    Δ半音_最小=round(float(np.min([x["Δ半音"] for x in 成对])), 2),
    Δ半音_全部达阈=bool(all(x["Δ半音"] >= F0阈值_半音 for x in 成对)))
果["C音色层"] = C层

# ───────────── D 解耦缺失：参数出口 / 共振峰入口 ─────────────
D层 = dict()
基础 = dict(T.音色档案["女声-明亮"])
基础["语速"] = 3.2
基rec, _ = 发声(基础, 文本)
D层["参数出口"] = {}
扰 = dict(基频=20.0, 语速=0.5, 幅度=0.10, 气声度=0.20, 紧喉度=0.30,
          粗糙度=0.30, 送气量=0.30, 声区=0.20)
有出口 = 0
for k, dv in 扰.items():
    参 = dict(基础)
    参[k] = float(参.get(k, 0.0)) + dv
    if k == "基频":
        参[k] = min(max(参[k], 70.0), 400.0)
    else:
        参[k] = min(max(参[k], 0.0), 1.0)
    r, _ = 发声(参, 文本)
    变 = not np.array_equal(r["波形"], 基rec["波形"])
    有出口 += int(变)
    D层["参数出口"][k] = dict(Δ=dv, 波形改变=bool(变))
D层["有效出口数"] = 有出口
D层["无出口参数"] = [k for k, v in D层["参数出口"].items() if not v["波形改变"]]
D层["音色档案自由度数"] = len(T.音色档案["女声-明亮"])

# 共振峰入口：固定语速后，两个档案的 F1/F2/F3 是否逐位同
qs = 器.分位读数(向量)
律A = T.情绪律(T.音色档案["女声-明亮"], "平静"); 律A["语速"] = 3.2; 律A["_ñ"] = 0.5
律B = T.情绪律(T.音色档案["男声-低沉"], "平静"); 律B["语速"] = 3.2; 律B["_ñ"] = 0.5
帧A = 器.音节帧(文本, 律A, qs)
帧B = 器.音节帧(文本, 律B, qs)
D层["共振峰不变性"] = dict(
    帧数A=len(帧A["f0"]), 帧数B=len(帧B["f0"]),
    F1逐位同=bool(np.array_equal(np.asarray(帧A["f1"]), np.asarray(帧B["f1"]))),
    F2逐位同=bool(np.array_equal(np.asarray(帧A["f2"]), np.asarray(帧B["f2"]))),
    F3逐位同=bool(np.array_equal(np.asarray(帧A["f3"]), np.asarray(帧B["f3"]))),
    F0平均Δ半音=round(float(np.mean(np.abs(半音(np.mean(帧A["f0"]), np.mean(帧B["f0"]))))), 2),
    口径="两档案仅 F0/幅度/气声/声区/送气不同；F1F2F3 来自韵母固定表，无音色缩放入口")

# 同文本换音色 ⇒ 文本层是否被扰动（音节序列 = 文本层独立性的实测锚）
D层["文本不变性"] = dict(
    音节帧数_同=bool(len(帧A["f0"]) == len(帧B["f0"])),
    说明="换音色不改变拼音/音节数 ⇒ 文本层判据可在音频外独立判定；音频内文本判据会被语速污染")

# 确定性
r1, _ = 发声(基础, 文本)
r2, _ = 发声(基础, 文本)
D层["确定性"] = dict(两趟波形逐位同=bool(np.array_equal(r1["波形"], r2["波形"])))

# 送气量：条件性入口（仅送气塞音段）
文本_塞 = "我怕他看。"
r0, _ = 发声(基础, 文本_塞)
参2 = dict(基础)
参2["送气量"] = min(1.0, float(基础["送气量"]) + 0.3)
r1_, _ = 发声(参2, 文本_塞)
D层["送气量_条件入口"] = dict(
    文本=文本_塞, 波形改变=bool(not np.array_equal(r0["波形"], r1_["波形"])),
    说明="送气量只在送气塞音段（怕/他/看）有出口；本探针主文本『你好，我在这里。』无送气塞音 ⇒ 无出口")

# 自证臂：从自家合成音反估参数（音色模仿 PASS 的机器证据）
自证 = {}
for 音 in ("女声-明亮", "男声-低沉"):
    rec, _ = 发声(T.音色档案[音], 文本)
    try:
        est = T.音色估计(os.path.join(产物, rec["wav"]))
        自证[音] = dict(设定基频=T.音色档案[音]["基频"], 估计基频=est["参数"]["基频"],
                        设定语速=T.音色档案[音]["语速"], 估计语速=est["参数"]["语速"],
                        设定紧喉度=T.音色档案[音]["紧喉度"], 估计紧喉度=est["参数"]["紧喉度"],
                        设定送气量=T.音色档案[音]["送气量"], 估计送气量=est["参数"]["送气量"],
                        设定声区=T.音色档案[音]["声区"], 估计声区=est["参数"]["声区"])
    except Exception as e:  # 依赖缺失不阻断主结论
        自证[音] = dict(错误=repr(e)[:80])
D层["自证_参数回估"] = dict(每档=自证,
                           说明="音色估计只估 F0/能量/谱重心/平坦：语速恒 3.2、粗糙度/送气量/声区写死 ⇒ 反向不可辨识（自证循环）")
果["D解耦缺失"] = D层

# ───────────── R 扮演崩塌（本轮不跑） ─────────────
果["R扮演崩塌"] = dict(状态="未测", 原因="需冲突输入下的人设漂移率与 ĝ/ε_self 读数，本轮 CPU 预算内不跑；判据见 R1 立场 §4")

# 清理本目录 wav
for f in os.listdir(产物):
    if f.endswith(".wav"):
        try:
            os.remove(os.path.join(产物, f))
        except OSError:
            pass

p = os.path.join(HERE, "探针_解耦测量结果.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(果, f, ensure_ascii=False, indent=1, default=str)
print("结果 →", p)
print(json.dumps(果, ensure_ascii=False, indent=1, default=str)[:6000])
