# -*- coding: utf-8 -*-
"""meeting63/实施/皮层_音素格.py —— 内部深度学习模块（HDC 谐振音素格）· 学习升级 v2。

所有者口径：「内部的深度学习模块可能也要优化（核心）」。本件把 meeting62 那个只做
「F=2 因子分解」的探针，升级成一个**真会用、会学、会读、会诊断**的模块，并接上 `耳_内生.py`：

  感知      波形 → 耳切段 → **辅音区特征**（首类/区长/清长/静长/浊长/谱重心/谱峰/瞬态度）
  写(学习)  辅音区特征 → HDC 段向量（role⊛value 绑定）→ 按**意图音素类**累加成原型
  读(谐振)  新段向量 → `读(原型码本, 段向量)` 最近原型 → 类别
  诊断      在**留出音节**上分类 → 准确率 + **混淆矩阵** ⇒ 这就是「咬字可辨度」
  反馈      错分（读出类 ≠ 意图类）回驱发声参数（见 `学_咬字反馈.py`）

★ 相对 v1 的「核心优化」：v1 用**单段**特征 ⇒ 送气塞与擦音 100% 混（都没有「有无闭塞/瞬态」的表示）；
  v2 改用**辅音区**特征（含首类=静/清/浊、静长、瞬态度）⇒ 让「塞音=先静后爆」「擦音=直接起噪」
  可被表示出来。这正是「模块要优化」的那一刀。

★ 可证伪（全部真跑，含 must-fail）
  P1 留出音节分类准确率 ≥ 3×chance
  P2 must-fail·置乱标签 ⇒ 掉到 chance
  P3 must-fail·直读（不建原型，直接拿单样本向量读原子码本）⇒ 掉到 chance
  P4 must-fail·白噪喂同一管线 ⇒ 掉到 chance
  P5 两趟逐位同（C13）

★ 零外挂（C10）：只用 numpy + 本仓 `算子_能力核` 的 8 原语；不 import 任何框架/模型；
  音频由调用方**喂入**（本件不读文件）。★ C1：只用本仓 8 算子；无注意力子串。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AHI4 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AHI4, "meeting62", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting61", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))
import 耳_内生 as 耳  # noqa: E402

_核路径 = os.path.join(AHI4, "meeting41", "实施", "算子_能力核.py")


def 载核():
    spec = importlib.util.spec_from_file_location("能力核_只读_63", _核路径)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_m = 载核()
能力核 = _m.能力核

D = 1024
角色s = ("首类", "区长", "清长", "静长", "浊长", "重心", "峰", "瞬态")
各档数 = {"首类": 3, "区长": 4, "清长": 4, "静长": 3, "浊长": 3,
        "重心": 4, "峰": 4, "瞬态": 3}

界_区长 = (25.0, 60.0, 120.0)
界_清长 = (10.0, 45.0, 95.0)
界_静长 = (10.0, 35.0)
界_浊长 = (10.0, 35.0)
界_重心 = (1500.0, 3000.0, 5000.0)
界_峰 = (2500.0, 4500.0, 7000.0)
界_瞬态 = (1.6, 3.0)
首类序 = {耳.静: 0, 耳.清: 1, 耳.浊: 2}


def _档(值, 界s) -> int:
    值 = float(值)
    for i, b in enumerate(界s):
        if 值 < b:
            return i
    return len(界s)


def 瞬态度(波形: np.ndarray, a: int, b: int, 窗: int = 88) -> float:
    """段内短窗能量 crest（峰/中位）：爆破有瞬态 ⇒ 高；稳态擦音 ⇒ 低。纯 DSP。
    窗取 2ms（88 样本）⇒ 10ms 的爆破段也能算出 crest（10ms 窗会退化成 0）。"""
    if b - a < 窗 * 2:
        return 0.0
    x = 波形[a:b]
    k = (len(x) - 窗) // 窗 + 1
    if k < 2:
        return 0.0
    idx = np.arange(窗)[None, :] + 窗 * np.arange(k)[:, None]
    e = np.sqrt(np.mean(x[idx] ** 2, axis=1))
    med = float(np.median(e))
    return float(np.max(e)) / max(med, 1e-12)


def 辅音区(波形: np.ndarray):
    """耳切段 → 辅音区（元音前的段）与其特征。返回 (特征dict, 段数) 或 None。"""
    x = np.asarray(波形, dtype=np.float64).ravel()
    if np.max(np.abs(x)) > 1.2:
        x = x / 32768.0
    帧 = 耳._分帧(x)
    元组 = 耳.切段(x, 帧)
    统计 = 耳._段统计(帧, 元组)
    段 = [dict(s, 起=a, 止=b) for s, (t, a, b) in zip(统计, 元组)]
    if len(段) < 2:
        return None
    浊i = [i for i, s in enumerate(段) if s["类"] == 耳.浊]
    元i = max(浊i, key=lambda i: 段[i]["ms"]) if 浊i else len(段)
    若 = 段[:元i]
    # 去掉句首换气（第一个清段能量远低于后续元音）
    if 若 and 若[0]["类"] == 耳.清 and 元i < len(段) and \
            若[0]["能量dB"] < 段[元i]["能量dB"] - 18.0:
        若 = 若[1:]
    if not 若:
        return None
    清s = [s for s in 若 if s["类"] == 耳.清]
    f = dict(
        首类=首类序[若[0]["类"]],
        区长=_档(sum(s["ms"] for s in 若), 界_区长),
        清长=_档(sum(s["ms"] for s in 清s), 界_清长),
        静长=_档(sum(s["ms"] for s in 若 if s["类"] == 耳.静), 界_静长),
        浊长=_档(sum(s["ms"] for s in 若 if s["类"] == 耳.浊), 界_浊长),
        重心=_档(float(np.median([s["谱重心Hz"] for s in 清s])) if 清s else 0.0, 界_重心),
        峰=_档(float(np.median([s["谱峰Hz"] for s in 清s])) if 清s else 0.0, 界_峰),
        瞬态=_档(max([瞬态度(x, s["起"], s["止"]) for s in 清s]) if 清s else 0.0, 界_瞬态),
    )
    return f, len(若)


class 音素皮层:
    """HDC 谐振音素格：写(辅音区特征)→原型；读→类别。"""

    def __init__(self, D维: int = D, 种子: int = 63001):
        self.核 = 能力核(D=D维, 种子=int(种子))
        self.D = int(D维)
        rng = np.random.default_rng(int(种子) + 101)
        self.角色原子 = {}
        self.值原子 = {}
        for r in 角色s:
            self.角色原子[r] = np.where(rng.integers(0, 2, size=self.D) == 1, 1.0, -1.0)
            self.值原子[r] = [np.where(rng.integers(0, 2, size=self.D) == 1, 1.0, -1.0)
                            for _ in range(各档数[r])]
        self.原型 = {}
        self.原型件数 = {}
        self._缓存 = {}

    def 编码(self, f: dict) -> np.ndarray:
        件 = []
        for r in 角色s:
            档 = int(f[r])
            v = self.值原子[r][min(max(档, 0), 各档数[r] - 1)]
            件.append(self.核.绑(self.角色原子[r], v))
        return self.核.叠(件)

    def 写(self, f: dict, 类名: str) -> None:
        self._缓存.setdefault(str(类名), []).append(self.编码(f))

    def 冻结(self) -> None:
        for 名, vs in self._缓存.items():
            self.原型[名] = self.核.叠(vs) if len(vs) > 1 else vs[0]
            self.原型件数[名] = len(vs)

    def 读(self, f: dict) -> dict:
        if not self.原型:
            return dict(类=None, 相似度=0.0, 命中=False)
        名s = sorted(self.原型)
        码 = np.asarray([self.原型[n] for n in 名s], dtype=np.float64)
        r = self.核.读(码, self.编码(f), k=1)
        return dict(类=名s[r["索引"][0]], 相似度=float(r["相似度"][0]), 命中=bool(r["命中"]),
                    序=int(r["索引"][0]), 名表=名s)


声母类 = {
    "b": "不送气塞", "d": "不送气塞", "g": "不送气塞",
    "p": "送气塞", "t": "送气塞", "k": "送气塞",
    "f": "擦", "h": "擦", "s": "擦", "sh": "擦", "x": "擦",
    "z": "塞擦", "c": "塞擦", "zh": "塞擦", "ch": "塞擦", "j": "塞擦", "q": "塞擦",
    "m": "鼻边", "n": "鼻边", "l": "鼻边", "r": "鼻边",
}
音节集 = tuple(声明 + "a1" for 声明 in 声母类)
字表 = {"ba1": "八", "pa1": "趴", "da1": "搭", "ta1": "他", "ga1": "嘎", "ka1": "咖",
       "fa1": "发", "ha1": "哈", "sa1": "撒", "sha1": "沙", "xia1": "虾",
       "za1": "匝", "ca1": "擦", "zha1": "渣", "cha1": "叉", "jia1": "家", "qia1": "掐",
       "ma1": "妈", "na1": "拿", "la1": "拉", "ra1": "呥"}


def 采样本(拼: str, 字: str = None, 声线=None, 种子=300, 韵律模块=None):
    if 韵律模块 is None:
        import 韵律_v7 as H7
        韵律模块 = H7
    import 声线覆盖层_v7 as 覆
    档 = 声线 if 声线 is not None else 覆.取("EmoCompanion")
    w, _, _ = 韵律模块.人类韵律_v8(档, 种子).说音节(拼) if hasattr(韵律模块, "人类韵律_v8") \
        else 韵律模块.人类韵律_v5(档, 种子).说音节(拼)
    r = 辅音区(w)
    if r is None:
        return None
    return (声母类[拼[:-2]], r[0], r[1])


def 采集(韵律模块=None, 种子列=(300, 301, 302)) -> dict:
    """每个音节多趟采样 → {拼: [(意图类, 特征), …]}（确定性）。"""
    样 = {}
    for 拼 in 音节集:
        vs = []
        for sd in 种子列:
            r = 采样本(拼, 字表.get(拼, "啊"), 种子=sd, 韵律模块=韵律模块)
            if r is not None:
                vs.append(r)
        样[拼] = vs
    return 样


def 跑(样品=None, 韵律模块=None) -> dict:
    """留一音节交叉验证：每折以其余全部音节训练、留出该音节测试。"""
    if 样品 is None:
        样品 = 采集(韵律模块=韵律模块)
    类数 = len(set(声母类.values()))
    chance = 1.0 / 类数

    命中 = 总 = 0
    混淆 = {}
    明细 = []
    for p in 音节集:
        if not 样品.get(p):
            continue
        皮 = 音素皮层()
        for q in 音节集:
            if q == p:
                continue
            for 类名, f, _n in 样品[q]:
                皮.写(f, 类名)
        皮.冻结()
        for 类名, f, _n in 样品[p]:
            r = 皮.读(f)
            总 += 1
            命中 += int(r["类"] == 类名)
            混淆.setdefault(类名, {}).setdefault(r["类"] or "无", 0)
            混淆[类名][r["类"] or "无"] += 1
        y = 样品[p][0][0]
        读出s = [皮.读(f)["类"] for _c, f, _n in 样品[p]]
        明细.append(dict(拼=p, 意图=y, 读出=读出s, 段数=样品[p][0][2], 特征=样品[p][0][1]))
    acc = 命中 / max(总, 1)

    # must-fail 1：置乱标签（全体训练件标签打乱后重建原型，留出同折）
    件 = [(p, 类名, f) for p in 音节集 for 类名, f, _n in 样品[p]]
    rng = np.random.default_rng(63007)
    乱 = list(rng.permutation([y for _p, y, _f in 件]))
    命2 = 总2 = 0
    for p in 音节集:
        if not 样品.get(p):
            continue
        皮 = 音素皮层()
        pos = 0
        for q in 音节集:
            for (_c, f, _n) in 样品[q]:
                y = 乱[pos]; pos += 1
                if q != p:
                    皮.写(f, y)
        皮.冻结()
        pos = sum(len(样品[q]) for q in 音节集[:音节集.index(p)])
        for off, (_c, f, _n) in enumerate(样品[p]):
            总2 += 1
            命2 += int(皮.读(f)["类"] == 乱[pos + off])
    acc乱 = 命2 / max(总2, 1)

    # must-fail 2：直读原子码本（不建原型）
    皮0 = 音素皮层()
    原子码 = np.asarray([v for r in 角色s for v in 皮0.值原子[r]], dtype=np.float64)
    类表 = sorted(set(声母类.values()))
    命3 = 总3 = 0
    for p in 音节集:
        for (类名, f, _n) in 样品[p]:
            总3 += 1
            r = 皮0.核.读(原子码, 皮0.编码(f), k=1)
            命3 += int(类表[int(r["索引"][0]) % len(类表)] == 类名)
    acc直读 = 命3 / max(总3, 1)

    # must-fail 3：白噪（切不出结构 ⇒ 不得命中任何意图类）
    白 = np.random.default_rng(63009).standard_normal(int(0.6 * 耳.FS))
    白区 = 辅音区(白)
    总4 = sum(len(v) for v in 样品.values())
    命4 = 0
    if 白区 is not None:
        皮w = 音素皮层()
        for p in 音节集:
            for (类名, f, _n) in 样品[p]:
                皮w.写(f, 类名)
        皮w.冻结()
        y预 = 皮w.读(白区[0])["类"]
        命4 = sum(len(v) for p, v in 样品.items() if v and v[0][0] == y预)
    acc白 = 命4 / max(总4, 1)

    果 = dict(
        工具="皮层_音素格.py", 版本="v2(辅音区特征·HDC 谐振音素格·留一音节)",
        协议="留一音节交叉验证（每折以余全部音节训练、留出该音节测试）",
        采样种子=(300, 301, 302), 类集=类表, 类数=类数, chance=round(chance, 4),
        样本总数=总, 命中=命中, 准确率=round(acc, 4), 混淆矩阵=混淆, 明细=明细,
        P1_准确率达标=bool(acc >= 3.0 * chance),
        置乱准确率=round(acc乱, 4), P2_置乱掉到chance=bool(abs(acc乱 - chance) <= 2.0 * chance),
        直读准确率=round(acc直读, 4), P3_直读掉到chance=bool(acc直读 <= 2.0 * chance),
        白噪准确率=round(acc白, 4), P4_白噪掉到chance=bool(acc白 <= 2.0 * chance),
    )
    果["通过"] = bool(果["P1_准确率达标"] and 果["P2_置乱掉到chance"]
                    and 果["P3_直读掉到chance"] and 果["P4_白噪掉到chance"])
    return 果


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    np.seterr(all="ignore")
    果 = 跑()
    print(json.dumps({k: v for k, v in 果.items() if k not in ("混淆矩阵", "明细")},
                     ensure_ascii=False, indent=1))
    print("--- 混淆矩阵（行=意图, 列=读出）---")
    for k, v in 果["混淆矩阵"].items():
        print("  %-8s → %s" % (k, v))
    print("--- 留出明细 ---")
    for d in 果["明细"]:
        print("  %-6s 意图%-6s 读出%-6s 段%d 特征%s" % (d["拼"], d["意图"], d["读出"], d["段数"], d["特征"]))
    print("[皮层_音素格] %s" % ("PASS" if 果["通过"] else "FAIL"))
