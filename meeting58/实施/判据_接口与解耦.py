# -*- coding: utf-8 -*-
"""meeting58/实施/判据_接口与解耦.py —— 前置接口适配 + 两层解耦 + 热插拔 的总判据。

跑前冻结（阈值不许看结果再改；本文件顶部常量即冻结口径）：
  校准集：3 句文本 × 2 个状态 × 6 个声线（用于声线可分性）；判据集：另 1 句 × 2 状态。
  判据分 8 组：G0 契约 / G1 状态真链路 / G2 声线可分 / G3 两层解耦 / G4 热插拔与回滚 /
             G5 must-fail / G6 文本层修复 / G7 预算与确定性。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import 接口契约_角色声线 as C  # noqa: E402
import 声线库  # noqa: E402
import 协议_角色逻辑 as 行层  # noqa: E402
import 协议_状态源 as 状态层  # noqa: E402
import 协议_音色渲染 as 声层  # noqa: E402
from 装配_角色声线 import 装配_角色声线  # noqa: E402
from 注册表_挂载 import 注册表_挂载  # noqa: E402

产物 = os.path.join(HERE, "产物", "判据")
os.makedirs(产物, exist_ok=True)

# ─────────────────────── 冻结阈值（跑前写死） ───────────────────────
阈值 = dict(
    F0最小半音距=3.0,            # 任两档「档案真值 F0」≥3 半音
    包络余弦最大=0.99,            # 任两档 7 带包络余弦 ≤0.99
    包络余弦中位最大=0.96,        # 中位 ≤0.96
    共振峰最小差Hz=120.0,         # 至少 4/6 对 F1 或 F2 差 ≥120Hz
    共振峰过线对数=4,
    判别准确率=0.85,              # 声线最近质心分类（判据集）
    置乱准确率=0.40,              # 标签置乱后 ≤0.40（must-fail）
    状态波形相关最大=0.999,       # 同文本同声线、不同状态 → 相关 <0.999
    状态F0最小半音=0.20,
    每句毫秒=450.0,               # 整句渲染硬上限
    实时率最大=0.15,
    池覆盖最小=10,                # 无台词模板 24 状态至少用到 10 句
    状态依赖改变率=0.50,          # 两条状态轨迹文本差异 ≥50%
    回放包络相关最大=0.50,        # 定制输出 vs 参考录音：必须不像回放
)

结果 = []
def 臂(名, 通过, 实测, 阈值=None, 说明=""):
    结果.append(dict(臂=名, 组=名.split("-")[0], 通过=bool(通过), 实测=实测,
                     阈值=阈值, 说明=说明))
    print(("PASS " if 通过 else "FAIL ") + 名, "|", 实测)


# ─────────────────────── 特征 ───────────────────────
带 = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 8000)]
FS = 16000


def F0中位(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    f0 = []
    n = max(1, (len(x) - int(0.04 * FS)) // int(0.01 * FS))
    for i in range(n):
        seg = x[i * int(0.01 * FS): i * int(0.01 * FS) + int(0.04 * FS)]
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
        lo, hi = int(FS / 500), int(FS / 70)
        if hi >= len(ac):
            continue
        段 = ac[lo:hi]
        mx = float(np.max(段))
        if mx <= 0 or ac[0] <= 0:
            continue
        # 防倍周期（八度下移）错误：取第一个达到 88% 峰值的局部峰（最小 lag）
        阈值 = 0.88 * mx
        cand = [j for j in range(1, len(段) - 1)
                if 段[j] >= 阈值 and 段[j] >= 段[j - 1] and 段[j] >= 段[j + 1]]
        if not cand:
            continue
        lag = lo + cand[0]
        if ac[lag] / (ac[0] + 1e-12) < 0.30:
            continue
        f0.append(FS / lag)
    return float(np.median(f0)) if f0 else 0.0


def 包络7(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    频 = np.fft.rfftfreq(len(x), 1.0 / FS)
    e = []
    for lo, hi in 带:
        m = (频 >= lo) & (频 < hi)
        e.append(np.log(np.mean(spec[m] ** 2) + 1e-12))
    e = np.asarray(e)
    return e - e.mean()


def 谱重心(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    频 = np.fft.rfftfreq(len(x), 1.0 / FS)
    return float(np.sum(频 * spec) / (np.sum(spec) + 1e-12))


def 特征向量(x) -> np.ndarray:
    return np.concatenate([[F0中位(x) / 100.0], 包络7(x), [谱重心(x) / 3000.0]])


def 倒谱共振峰(x) -> list:
    seg = np.asarray(x)[int(0.2 * FS):int(0.2 * FS) + int(0.08 * FS)]
    w = seg * np.hanning(len(seg))
    logmag = np.log(np.abs(np.fft.rfft(w, 2048)) + 1e-9)
    cep = np.fft.irfft(logmag)
    cep[30:-30] = 0.0
    env = np.fft.rfft(cep).real
    频 = np.fft.rfftfreq(2048, 1.0 / FS)
    峰 = []
    for i in range(2, len(env) - 2):
        if env[i] > env[i - 1] and env[i] > env[i + 1] and 200 < 频[i] < 4000:
            峰.append(float(频[i]))
    return 峰[:3]


def 余弦(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def 包络相关(a, b) -> float:
    def env(x):
        x = np.asarray(x, dtype=np.float64)
        win = 160
        n = len(x) // win
        e = np.sqrt(np.mean(x[:n * win].reshape(n, win) ** 2, axis=1) + 1e-12)
        return e
    ea, eb = env(a), env(b)
    n = min(len(ea), len(eb))
    if n < 8:
        return 0.0
    ea, eb = ea[:n] - ea[:n].mean(), eb[:n] - eb[:n].mean()
    return float(np.corrcoef(ea, eb)[0, 1])


# ─────────────────────── 主判据 ───────────────────────
def 跑判据() -> dict:
    器 = 声层.声带渲染器(目录=产物)
    声线名 = list(声线库.全部())
    校准文本 = ["你好，今天过得怎么样？", "我想和你说一件事。", "谢谢你来到这里。"]
    判据文本 = "我愿意把这段声音留给你。"
    状态 = 状态层.合成状态源(拍数=40, 种子=11)
    帧列 = [状态.快照(0), 状态.快照(17)]

    # G0 契约
    c = C.自检()
    臂("G0-契约自检", c["通过"], [x for x in c["项"] if not x["通过"]] or "12 项全过")
    审计 = C.审计文件([
        os.path.join(HERE, f) for f in
        ("声线库.py", "协议_状态源.py", "协议_角色逻辑.py", "协议_音色渲染.py",
         "注册表_挂载.py", "装配_角色声线.py", "工具_确定性.py", "判据_接口与解耦.py")
    ])
    臂("G0-C1C10静态审计", 审计["通过"], 审计["命中"][:4] or "禁词/禁import 0 命中")

    # G1 状态真链路
    a = 声线库.取("EmoCompanion")
    o1 = 器.渲染(判据文本, 声线=a, 状态=帧列[0], 种子=300, 写wav=False)
    o2 = 器.渲染(判据文本, 声线=a, 状态=帧列[1], 种子=300, 写wav=False)
    o1b = 器.渲染(判据文本, 声线=a, 状态=帧列[0], 种子=300, 写wav=False)
    x1, x2 = np.asarray(o1["波形"]), np.asarray(o2["波形"])
    n = min(len(x1), len(x2))
    corr = float(np.corrcoef(x1[:n], x2[:n])[0, 1]) if n > 8 else 1.0
    逐位同 = bool(len(x1) == len(x2) and np.array_equal(x1, x2))
    f0差 = abs(12 * np.log2(max(F0中位(x1), 1) / max(F0中位(x2), 1)))
    臂("G1-状态进声带", (not 逐位同) and corr < 阈值["状态波形相关最大"] and f0差 >= 阈值["状态F0最小半音"],
      dict(相关=round(corr, 4), F0差半音=round(f0差, 3), 逐位同=逐位同))
    臂("G1-状态确定性", np.array_equal(np.asarray(o1b["波形"]), x1), "同状态两趟逐位同")

    # G2 声线可分（固定角色集；「定制」是参考驱动，单列可追溯判据）
    固定集 = ["洛天依", "初音未来", "MOSS", "EmoCompanion", "中性"]
    真值F0 = [(名, 声线库.取(名).基频) for 名 in 固定集]
    最紧 = min((abs(12 * np.log2(a1 / b1)), a0, b0)
               for i, (a0, a1) in enumerate(真值F0) for b0, b1 in 真值F0[i + 1:])
    臂("G2-档案F0间距", 最紧[0] >= 阈值["F0最小半音距"], dict(最小半音=round(最紧[0], 2), 对=(最紧[1], 最紧[2])))

    包络 = {}
    特征 = {}
    for 名 in 固定集:
        a = 声线库.取(名)
        特征[名] = []
        for t in 校准文本:
            for f in 帧列:
                o = 器.渲染(t, 声线=a, 状态=f, 种子=300, 写wav=False)
                特征[名].append(特征向量(o["波形"]))
                if t == 校准文本[0] and f is 帧列[0]:
                    包络[名] = 包络7(o["波形"])
    对 = {}
    名列 = list(包络)
    for i in range(len(名列)):
        for j in range(i + 1, len(名列)):
            对["%s|%s" % (名列[i], 名列[j])] = 余弦(包络[名列[i]], 包络[名列[j]])
    最大 = max(对.values()); 中位 = float(np.median(list(对.values())))
    臂("G2-包络可分", 最大 <= 阈值["包络余弦最大"] and 中位 <= 阈值["包络余弦中位最大"],
      dict(最大=round(最大, 4), 中位=round(中位, 4), 最像=max(对, key=对.get)))

    共振 = {}
    for 名 in 固定集:
        o = 器.渲染("啊啊啊啊啊", 声线=声线库.取(名), 种子=300, 写wav=False)
        共振[名] = 倒谱共振峰(o["波形"])
    共振对 = []
    for i in range(len(名列)):
        for j in range(i + 1, len(名列)):
            a1, b1 = 共振[名列[i]], 共振[名列[j]]
            if a1 and b1:
                共振对.append(max(abs(a1[0] - b1[0]), abs(a1[1] - b1[1]) if len(a1) > 1 and len(b1) > 1 else 0))
    过 = sum(1 for d in 共振对 if d >= 阈值["共振峰最小差Hz"])
    臂("G2-共振峰可分", 过 >= 阈值["共振峰过线对数"],
      dict(过线="%d/%d" % (过, len(共振对)), 最小=max(0.0, round(min(共振对), 1))))

    # 同文本最近质心分类（E2 口径：固定文本/情绪，只让状态变；训练 3 状态，测 2 状态）
    sd = np.std(np.concatenate([np.asarray(v) for v in 特征.values()]), axis=0) + 1e-9
    判据文本列 = ["我愿意把这段声音留给你。", "这句话是留给你的。"]
    训练状态 = [0, 7, 13]
    测试状态 = [17, 29]

    def 预测(向量, 中心):
        d = {k: float(np.linalg.norm((向量 - 中心[k]) / sd)) for k in 中心}
        return min(d, key=d.get)

    对错 = 0; 总 = 0; 混淆 = {}
    for t in 判据文本列:
        中心 = {}
        for 名 in 固定集:
            a = 声线库.取(名)
            vs = [特征向量(器.渲染(t, 声线=a, 状态=状态.快照(s), 种子=302, 写wav=False)["波形"])
                  for s in 训练状态]
            中心[名] = np.mean(vs, axis=0)
        for 名 in 固定集:
            a = 声线库.取(名)
            for s in 测试状态:
                o = 器.渲染(t, 声线=a, 状态=状态.快照(s), 种子=303, 写wav=False)
                p = 预测(特征向量(o["波形"]), 中心)
                混淆.setdefault(名, []).append(p)
                对错 += int(p == 名); 总 += 1
    准确 = 对错 / max(总, 1)
    臂("G2-声线分类", 准确 >= 阈值["判别准确率"], dict(准确率=round(准确, 3), 混淆=混淆))

    # 标签置乱（循环移位=derangement，5 类）必须在同一分类器上失败
    标签 = list(固定集)
    对错2 = 0
    for t in 判据文本列:
        中心 = {}
        for i, 名 in enumerate(固定集):
            a = 声线库.取(标签[(i + 1) % len(标签)])
            vs = [特征向量(器.渲染(t, 声线=a, 状态=状态.快照(s), 种子=302, 写wav=False)["波形"])
                  for s in 训练状态]
            中心[标签[i]] = np.mean(vs, axis=0)
        for 名 in 固定集:
            a = 声线库.取(名)
            for s in 测试状态:
                o = 器.渲染(t, 声线=a, 状态=状态.快照(s), 种子=303, 写wav=False)
                对错2 += int(预测(特征向量(o["波形"]), 中心) == 名)
    准确2 = 对错2 / max(总, 1)
    臂("G2-MF-标签置乱", 准确2 <= 阈值["置乱准确率"], dict(置乱准确率=round(准确2, 3)))

    # 定制音色：只证明「可追溯、非回放」，不主张像参考本人
    定 = 声线库.取("定制")
    臂("G2-定制可追溯", 定.来源 == "外采·参考录音" and len(定.指纹()) == 16
      and ("参考sha256" in 定.备注), dict(来源=定.来源, 指纹=定.指纹(),
                                       备注=定.备注[:40]))

    # G3 解耦（哨兵）
    固定文本 = "这是一个哨兵句。"
    oa = 器.渲染(固定文本, 声线=声线库.取("洛天依"), 状态=帧列[0], 种子=300, 写wav=False)
    ob = 器.渲染(固定文本, 声线=声线库.取("洛天依"), 状态=帧列[1], 种子=300, 写wav=False)
    臂("G3-U1声层哨兵", oa["声线指纹"] == ob["声线指纹"],
      "同声线不同状态 → 声线指纹不变")
    逻辑 = 行层.池角色逻辑()
    无台词 = C.角色模板(名="洛天依池", 版本="1.0", 来源="测试", 角色包名="洛天依",
                       声线名="洛天依", 情绪默认="平静", 句数=3, 腔调=(), 台词=(),
                       备注="池模式（无演示台词）")
    逻辑.装(无台词, 状态, 种子=0)
    序列1 = [逻辑.步(状态.快照(t))["文本"] for t in range(24)]
    状态2 = 状态层.合成状态源(拍数=40, 种子=12)
    逻辑.装(无台词, 状态2, 种子=0)
    序列2 = [逻辑.步(状态2.快照(t))["文本"] for t in range(24)]
    distinct = len(set(序列1))
    臂("G6-池覆盖", distinct >= 阈值["池覆盖最小"], dict(不同句=distinct, 池=逻辑.池大小()))
    改变 = sum(1 for a1, b1 in zip(序列1, 序列2) if a1 != b1) / max(len(序列1), 1)
    臂("G6-状态依赖文本", 改变 >= 阈值["状态依赖改变率"], dict(改变率=round(改变, 3)))

    # G3-U2 换声线必须变、换行层不影响声层
    o洛 = 器.渲染(固定文本, 声线=声线库.取("洛天依"), 状态=帧列[0], 种子=300, 写wav=False)
    oM = 器.渲染(固定文本, 声线=声线库.取("MOSS"), 状态=帧列[0], 种子=300, 写wav=False)
    臂("G3-U2声层可换", o洛["声线指纹"] != oM["声线指纹"] and
      not np.array_equal(np.asarray(o洛["波形"]), np.asarray(oM["波形"])), "换声线 → 指纹/波形都变")
    源码 = {}
    for f in ("协议_角色逻辑.py", "协议_音色渲染.py"):
        with open(os.path.join(HERE, f), encoding="utf-8") as fh:
            源码[f] = fh.read()
    import ast
    行层import = _imports(源码["协议_角色逻辑.py"])
    声层import = _imports(源码["协议_音色渲染.py"])
    臂("G3-静态单向依赖", ("声带" not in 行层import and "协议_音色渲染" not in 行层import
                          and "协议_角色逻辑" not in 声层import),
      dict(行层=sorted(行层import), 声层=sorted(声层import)))

    # G4 热插拔 + 回滚（同一进程）
    器A = 装配_角色声线(状态源=状态层.双通道状态源(种子=53001, 拍数=60), 渲染器=器)
    r1 = 器A.说一句("洛天依", 拍=0)
    r2 = 器A.说一句("MOSS", 拍=1)          # 句边界热切换
    r3 = 器A.说一句("洛天依", 拍=2)        # 切回
    回滚 = 器A.表.回滚("角色逻辑")
    臂("G4-句边界热切换", r1.get("通过") and r2.get("通过") and r3.get("通过")
      and r1["声线"] != r2["声线"] and r3["声线"] == r1["声线"],
      dict(句1=r1["声线"], 句2=r2["声线"], 句3=r3["声线"]))
    轨迹同 = 器A.源.轨迹指纹()
    器A.切换角色("初音未来")
    臂("G4-换角色不重建核", 器A.源.轨迹指纹() == 轨迹同, "状态源轨迹指纹不变")

    # 回滚逐位同：同句再渲染（两个全新装配，同种子）
    器B = 装配_角色声线(状态源=状态层.双通道状态源(种子=53001, 拍数=60), 渲染器=器)
    rr1 = 器B.说一句("洛天依", 拍=0, 写wav=False)
    器C = 装配_角色声线(状态源=状态层.双通道状态源(种子=53001, 拍数=60), 渲染器=器)
    rr2 = 器C.说一句("洛天依", 拍=0, 写wav=False)
    臂("G4-同句逐位同", np.array_equal(rr1["波形"], rr2["波形"]), "两个装配实例同一句 → 波形逐位同")

    # 隔离：一个渲染即抛异常的插件，连续 3 次失败后被隔离，且不拖垮其它插件
    表 = 注册表_挂载()
    class 坏件:
        def 渲染(self, *a, **k):
            raise RuntimeError("坏插件")
    表.注册("音色渲染", "坏", 坏件, 版本="1.0")
    表.试跑("音色渲染", "坏", 探针=lambda p: True)
    表.切换("音色渲染", "坏")
    隔离 = [表.调用("音色渲染", "渲染", "x", 声线=声线库.取("中性")) for _ in range(3)]
    表现 = 表.表现()
    臂("G4-失败隔离", all(r["降级"] for r in 隔离) and 表现["隔离"], dict(三次全降级=True,
                                                              隔离表=list(表现["隔离"])))
    表.注册("音色渲染", "好", lambda: 器, 版本="1.0")
    表.试跑("音色渲染", "好", 探针=lambda p: True)
    表.切换("音色渲染", "好")
    好 = 表.调用("音色渲染", "渲染", "x", 声线=声线库.取("中性"), 写wav=False)
    臂("G4-隔离不拖垮其它", not 好["降级"] and 好["结果"]["通过"], "好插件仍可用")

    # G5 must-fail
    try:
        声线库.取("不存在的声线"); _err = ""
    except C.契约错误 as e:
        _err = e.码
    臂("G5-未知声线被拒", _err == C.E1006, dict(码=_err))
    坏 = C.声线档案(名="坏", 基频=999.0)
    try:
        坏.校验(); _err2 = ""
    except C.契约错误 as e:
        _err2 = e.码
    臂("G5-非法档案被拒", _err2 == C.E1006, dict(码=_err2))
    try:
        注册表_挂载().取("音色渲染", "不存在的插件"); _err3 = ""
    except C.契约错误 as e:
        _err3 = e.码
    臂("G5-未注册被拒", _err3 == C.E1001, dict(码=_err3))
    # 回放 must-fail：定制声线 vs 参考录音
    ref = os.path.join(os.path.dirname(HERE), "产物", "参考", "参考_定制_huihui.wav")
    try:
        import wave as _w
        with _w.open(ref, "rb") as w:
            n = w.getnframes(); sr = w.getframerate()
            raw = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float64) / 32768.0
        orr = 器.渲染("这是一段测试。", 声线=声线库.取("定制"), 种子=300, 写wav=False)
        相关 = abs(包络相关(raw, np.asarray(orr["波形"])))
        臂("G5-MF回放冒充", 相关 <= 阈值["回放包络相关最大"], dict(包络相关=round(相关, 4)))
    except Exception as e:  # noqa: BLE001
        臂("G5-MF回放冒充", False, "异常 %r" % e)

    # G7 预算
    时 = []
    for _ in range(3):
        t0 = time.perf_counter()
        器.渲染(判据文本, 声线=声线库.取("EmoCompanion"), 状态=帧列[0], 种子=300, 写wav=False)
        时.append((time.perf_counter() - t0) * 1000)
    臂("G7-每句预算", max(时) <= 阈值["每句毫秒"], dict(毫秒=[round(x, 1) for x in 时]))
    o = 器.渲染(判据文本, 声线=声线库.取("EmoCompanion"), 状态=帧列[0], 种子=300, 写wav=False)
    臂("G7-实时率", o["实时率"] <= 阈值["实时率最大"], dict(实时率=o["实时率"]))

    # 汇总
    总 = len(结果)
    过 = sum(1 for r in 结果 if r["通过"])
    结 = dict(契约版本=C.契约版本, 阈值=阈值, 臂=结果, 通过数=过, 总数=总,
              总判=(过 == 总), 时间=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(产物, "判据_接口与解耦结果.json"), "w", encoding="utf-8") as f:
        json.dump(结, f, ensure_ascii=False, indent=1, default=str)
    print("=" * 70)
    print("总判：%s（%d/%d）" % (结["总判"], 过, 总))
    return 结


def _imports(src: str) -> set:
    import ast
    out = set()
    try:
        树 = ast.parse(src)
    except SyntaxError:
        return out
    for n in ast.walk(树):
        if isinstance(n, ast.Import):
            out.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module.split(".")[0])
    return out


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    跑判据()
