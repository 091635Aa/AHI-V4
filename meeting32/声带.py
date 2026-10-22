# -*- coding: utf-8 -*-
"""meeting32/声带.py —— #300 AHI 声带器官：从向量空间发声（五专家会议综合裁定实现）。

链路：内部状态向量 + 遥测（能量/新异/消化债/饱足）
     → HDC 韵律映射（会议四号专家公式族，消化债经 tanh 压缩）
     → 文本（codec 解码产物）→ pypinyin 拼音 → 音节参数表（医学参数组）
     → 三轨控制层（F0/共振峰/幅度，100Hz 帧）→ Rosenberg 声门源 + Klatt 级联谐振器
     → 16bit PCM wav（确定性逐位一致）。

设计约束（会议裁定）：
- 纯 numpy 确定性：固定 seed 的 rng，同输入两次合成逐位一致（S1）；
- 维度无关：所有维度相关量以 D 变量表达，唯一随维公式 τ=3/√D；
- CPU only（GPU<5%），内存有界（wav FIFO≤32）；
- 默认仅合成不外放（无声卡也不崩，S7）；外放需显式 开外放()；
- 发声台账独立成文件（不扰 meeting31 V5 重放逐位一致）。
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
from pypinyin import lazy_pinyin, Style

try:                                   # meeting58：分块 IIR 加速；无 scipy 时纯 numpy 回退
    from scipy.signal import lfilter as _lfilter
except Exception:  # noqa: BLE001
    _lfilter = None

FS = 16000                 # 采样率
帧率 = 100                  # 控制帧率（100Hz → 每帧 10ms）
每帧样本 = FS // 帧率
WAV上限 = 32               # wav FIFO 上限（内存有界，S9）

# —— 医学参数组（会议一号专家映射表，全部 [0,1] 或物理区间）——
参数表 = ("基频", "幅度", "气声度", "紧喉度", "粗糙度", "送气量", "声区")
基频区间 = (70.0, 400.0)   # Hz（医学：男 85-180 / 女 165-255，全带宽留余量）


def _基向量(键: str, D: int) -> np.ndarray:
    """键→D 维双极基向量（sha256 确定性导出，与 lang_codec 同律）。"""
    h = hashlib.sha256(("声带|%s" % 键).encode("utf-8")).digest()
    buf = b""
    while len(buf) < max(D // 8, 1):
        buf += hashlib.sha256(buf + h).digest()
    bits = np.frombuffer(buf[:D // 8], dtype=np.uint8)
    return ((bits[:, None] >> np.arange(8)[None, :]) & 1).reshape(-1)[:D].astype(np.float32) * 2 - 1


class 数轴:
    """HDC 等级数轴：连续量→档位→等级超向量组合编码；解码=原型链最近邻。
    （修正：双极向量乘法自幂 z⊙z=全 1 会周期坍缩，故采用 level-HDC 档位链
    + 分层双轴组合 f_{k1}⊙c_{k2}，保留相邻档相似度线性衰减与双轴结构。）"""

    def __init__(self, 键: str, D: int, S1: int = 16, S2: int = 16,
                 下界: float = 0.0, 上界: float = 1.0):
        self.D, self.S1, self.S2 = D, S1, S2
        self.下界, self.上界 = float(下界), float(上界)
        self.S = S1 * S2
        # 等级链：相邻档翻转 max(D//(2S1),1) 位（细轴）/ max(D//(2S2),1) 位（粗轴）
        # → 相邻档 cos = 1−2·翻转/D，远档趋 0（确定性 sha256 导出）
        self._f链 = self._等级链("fine|%s" % 键, D, S1 + 1)
        self._c链 = self._等级链("coarse|%s" % 键, D, S2 + 1)
        self._组合: dict = {}                                # k→组合原型缓存

    @staticmethod
    def _等级链(键: str, D: int, 档数: int) -> list:
        h = hashlib.sha256(("levels|%s" % 键).encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        v0 = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=D)
        链 = [v0]
        翻转数 = max(D // (2 * max(档数 - 1, 1)), 1)
        全位 = rng.permutation(D)
        for k in range(1, 档数):
            v = 链[-1].copy()
            位 = 全位[(k - 1) * 翻转数:(k - 1) * 翻转数 + 翻转数]
            v[位] = -v[位]                                   # 确定性位翻转累积
            链.append(v.astype(np.float32))
        return 链

    def 编码档位(self, 值: float) -> int:
        t = (float(值) - self.下界) / max(self.上界 - self.下界, 1e-9)
        return int(min(max(round(min(max(t, 0.0), 1.0) * self.S), 0), self.S))

    def 编码向量(self, 值: float) -> np.ndarray:
        """值→k→(k1,k2) 分层组合向量 = f_{k1} ⊙ c_{k2}（双极闭包）。"""
        k = self.编码档位(值)
        k1, k2 = k % self.S1, k // self.S1
        return (self._f链[k1] * self._c链[k2]).astype(np.float32)

    def _组合原型(self, k: int) -> np.ndarray:
        v = self._组合.get(k)
        if v is None:
            v = (self._f链[k % self.S1] * self._c链[k // self.S1]).astype(np.float32)
            self._组合[k] = v
        return v

    def 解码档位(self, v: np.ndarray) -> int:
        """组合原型链最近邻：全链 257 档扫描（O(S·D)≈26 万次乘加，<2ms）。"""
        vs = float(np.linalg.norm(v)) + 1e-9
        return max(range(self.S + 1),
                   key=lambda k: float(v @ self._组合原型(k)) / vs)


# —— 数轴对象缓存（meeting58 深度优化：构造 55.5ms → <5ms；数轴构造后只读）——
_轴缓存: dict = {}
_往返缓存: dict = {}


def _取轴(键: str, D: int, S1: int = 16, S2: int = 16,
          下界: float = 0.0, 上界: float = 1.0):
    ck = (键, int(D), int(S1), int(S2), float(下界), float(上界))
    a = _轴缓存.get(ck)
    if a is None:
        a = 数轴(键, D, S1=S1, S2=S2, 下界=下界, 上界=上界)
        _轴缓存[ck] = a
    return a


def _音色指纹(音色: dict) -> str:
    """音色档案 → 16 位指纹（确定性；只用于溯源，不参与 DSP）。"""
    if not 音色:
        return ""
    s = json.dumps(音色, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _夹(v, lo, hi):
    return float(min(max(float(v), lo), hi))


# —— 音韵表（会议二号专家：韵母共振峰 / 声母方式×部位 / 四声五度）——# 韵母 → (元音目标序列 [(F1,F2,F3),...], 鼻尾 'n'/'ng'/None)
_单韵母 = {"a": (800, 1200, 2500), "o": (450, 800, 2500), "e": (550, 1100, 2500),
          "i": (300, 2200, 3000), "u": (320, 700, 2500), "v": (300, 1800, 2200)}
_介音 = {"i": (300, 2200, 3000), "u": (320, 700, 2500), "v": (300, 1800, 2200)}
韵母表 = {}
for _r, _t in _单韵母.items():
    韵母表[_r] = ([_t], None)
# 复韵母（双元音滑移）
韵母表["ai"] = ([_单韵母["a"], (450, 2000, 2700)], None)
韵母表["ei"] = ([_单韵母["e"], (400, 2100, 2800)], None)
韵母表["ao"] = ([_单韵母["a"], _单韵母["u"]], None)
韵母表["ou"] = ([_单韵母["o"], _单韵母["u"]], None)
韵母表["er"] = ([_单韵母["e"]], None)
# 鼻韵母（尾段 F2/F3 滑向鼻腔位约 250/1000，会议二号专家）
for _r, _元 in (("an", "a"), ("en", "e"), ("in", "i"), ("un", "u"), ("vn", "v")):
    韵母表[_r] = ([_单韵母[_元]], "n")
for _r, _元 in (("ang", "a"), ("eng", "e"), ("ing", "i"), ("ong", "o"),
                ("uang", "a"), ("ueng", "e")):
    韵母表[_r] = ([_单韵母[_元]], "ng")
for _r, _元 in (("ian", "a"), ("uan", "a"), ("iang", "a")):
    韵母表[_r] = ([_单韵母[_元]], "n" if _r == "ian" else "ng")
韵母表["uan"] = ([_单韵母["a"]], "n")            # uan 与 ian 并列后重设兜底


def _查韵母(韵: str):
    """查韵母表；复合韵拆介音+主体兜底。"""
    if 韵 in 韵母表:
        return 韵母表[韵]
    if len(韵) >= 2 and 韵[0] in _介音:
        主体 = 韵[1:]
        if 主体 in 韵母表:
            目标s, 尾 = 韵母表[主体]
            return ([_介音[韵[0]]] + 目标s, 尾)
    # 最终兜底：逐字符元音串联
    目标s = [_单韵母[c] for c in 韵 if c in _单韵母] or [_单韵母["e"]]
    尾 = "ng" if 韵.endswith("ng") else ("n" if 韵.endswith("n") else None)
    return (目标s, 尾)


# 声母表：方式 与 噪声带（Hz, BW）；塞音附 (爆破带, VOT ms, 送气?)
声母表 = {
    "b": ("塞", 800, 10, False), "p": ("塞", 800, 60, True),
    "d": ("塞", 4000, 10, False), "t": ("塞", 4000, 60, True),
    "g": ("塞", 1800, 10, False), "k": ("塞", 1800, 60, True),
    "f": ("擦", (1500, 3000), 0, False), "h": ("擦", (1200, 2500), 0, False),
    "s": ("擦", (6500, 1800), 0, False), "sh": ("擦", (2800, 1200), 0, False),
    "x": ("擦", (4500, 1500), 0, False), "r": ("浊", None, 0, False),
    "z": ("塞擦", (6500, 1800), 10, False), "c": ("塞擦", (6500, 1800), 60, True),
    "zh": ("塞擦", (2800, 1200), 10, False), "ch": ("塞擦", (2800, 1200), 60, True),
    "j": ("塞擦", (4500, 1500), 10, False), "q": ("塞擦", (4500, 1500), 60, True),
    "m": ("浊", None, 0, False), "n": ("浊", None, 0, False),
    "l": ("浊", None, 0, False),
}
声母序 = ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k",
         "h", "j", "q", "x", "r", "z", "c", "s")

# 四声五度 F0 轮廓：(时间比, 五度值) 折线；五度 d→半音 (d-3)·2 相对基线
声调表 = {1: [(0.0, 5), (1.0, 5)], 2: [(0.0, 3), (1.0, 5)],
          3: [(0.0, 2), (0.6, 1), (1.0, 4)], 4: [(0.0, 5), (1.0, 1)],
          5: [(0.0, 2), (1.0, 2)]}
浊声母表 = {"m": (250, 1100, 2200), "n": (250, 1000, 2200), "l": (350, 1300, 2500),
           "r": (300, 1600, 2500)}


def _折线(轮廓, t: float) -> float:
    """分段折线插值。"""
    for i in range(len(轮廓) - 1):
        t0, v0 = 轮廓[i]
        t1, v1 = 轮廓[i + 1]
        if t <= t1 or i == len(轮廓) - 2:
            w = 0.0 if t1 <= t0 else min(max((t - t0) / (t1 - t0), 0.0), 1.0)
            return v0 + w * (v1 - v0)
    return 轮廓[-1][1]


def _拆音节(拼: str):
    """'zhang3'→('zh','ang',3)。"""
    调 = 5
    if 拼 and 拼[-1].isdigit():
        调 = int(拼[-1])
        拼 = 拼[:-1]
    for s in 声母序:
        if 拼.startswith(s):
            return s, 拼[len(s):], 调
    return "", 拼, 调


class 声带器官:
    """声带：向量空间→发音参数→波形。确定性、维度无关、CPU only。"""

    def __init__(self, seed: int = 300, D: int = 1024, 目录: str = None,
                 台账路径: str = None):
        self.seed = int(seed)
        self.D = int(D)
        self.目录 = 目录 or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "声音")
        os.makedirs(self.目录, exist_ok=True)
        self.台账路径 = 台账路径 or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "发声台账.jsonl")
        self._wav: list = []            # FIFO 路径（≤WAV上限）
        self.发声号 = 0
        self.外放 = False
        self._外放引擎 = None
        self._降级 = False
        # 韵律数轴（会议四号专家：多参数异轴零串扰；meeting58 起按 (键,D,界) 缓存复用）
        self._轴 = {
            "基频": _取轴("基频", D, 下界=基频区间[0], 上界=基频区间[1]),
            "幅度": _取轴("幅度", D), "气声度": _取轴("气声度", D),
            "紧喉度": _取轴("紧喉度", D), "粗糙度": _取轴("粗糙度", D),
            "送气量": _取轴("送气量", D), "声区": _取轴("声区", D),
        }
        self.检索阈值τ = 3.0 / np.sqrt(self.D)   # 唯一随维公式
        # 声区基线：seed 确定音色底（男 110 / 女 210 加权，确定性）
        r = np.random.default_rng(self.seed)
        self.基线F0 = 110.0 + 100.0 * float(r.random())

    # ---------- 韵律：遥测→医学参数组（会议四号专家公式族） ----------
    def 韵律(self, 遥测: dict) -> dict:
        e = float(遥测.get("能量", 0.5) or 0.0)
        ñ = float(遥测.get("新异", 0.5) or 0.0)
        债 = float(遥测.get("消化债", 0.0) or 0.0)
        饱 = float(遥测.get("饱足", 0.5) or 0.0)
        债̃ = float(np.tanh(债 / 300.0))     # 长跑实测债可达 1551 → tanh 压缩（偏离说明见纪要）
        clip = lambda v, lo, hi: float(min(max(v, lo), hi))
        f0 = clip(self.基线F0 + 70.0 * e - 45.0 * 债̃ + 25.0 * ñ,
                  基频区间[0], 基频区间[1])
        语速 = clip(3.2 + 2.2 * e + 1.8 * ñ - 1.4 * 债̃, 1.5, 7.0)
        幅度 = clip(0.30 + 0.45 * e - 0.30 * 债̃ + 0.15 * ñ, 0.1, 1.0)
        气声 = clip(0.15 + 0.35 * 饱 - 0.30 * e, 0.0, 0.8)
        紧喉 = clip(0.30 + 0.30 * e - 0.25 * 饱, 0.0, 1.0)
        粗糙 = clip(0.10 * 债̃ + 0.15 * max(e - 0.8, 0.0), 0.0, 1.0)
        送气 = clip(0.20 + 0.40 * 饱 - 0.10 * e, 0.0, 1.0)
        声区 = clip(0.15 + 0.30 * ñ + 0.20 * 饱, 0.0, 1.0)
        return dict(基频=f0, 语速=语速, 幅度=幅度, 气声度=气声, 紧喉度=紧喉,
                    粗糙度=粗糙, 送气量=送气, 声区=声区)

    # ---------- 意图向量 K 段分位读数（会议四号专家 ΔF0 微调源） ----------
    def 分位读数(self, 向量: np.ndarray, K: int = None) -> np.ndarray:
        D = self.D
        K = K or min(64, max(8, D // 16))
        L = D // K
        段和 = 向量[:K * L].reshape(K, L).sum(axis=1) / max(L, 1)   # ∈[-1,1]
        秩 = np.argsort(np.argsort(段和)).astype(np.float32)
        return 秩 / max(K - 1, 1)                                    # q∈[0,1]

    # ---------- 文本→音节帧参数（三轨控制层） ----------
    def 音节帧(self, 文本: str, 律: dict, qs: np.ndarray) -> dict:
        拼 = lazy_pinyin(文本, style=Style.TONE3, neutral_tone_with_five=True,
                         errors=lambda x: [c for c in x if c.isalnum()])
        音节s = [_拆音节(p) for p in 拼 if p and (p[0].isascii())] [:64]
        if not 音节s:
            音节s = [("", "e", 1)]
        语速 = 律["语速"]
        问句 = 文本.strip().endswith("？") or 文本.strip().endswith("?")
        帧参数 = {k: [] for k in ("f0", "f1", "f2", "f3", "幅", "气声",
                                 "声源", "噪带", "送气")}
        # 声源: 1=浊音(声门源) 0=噪声段 -1=静默
        下滑 = 0.0                                    # 陈述句逐词下滑（半音）
        n = len(音节s)
        for idx, (声, 韵, 调) in enumerate(音节s):
            qi = float(qs[min(idx, len(qs) - 1)]) if len(qs) else 0.5
            ΔF0 = float(np.clip(24.0 * (qi - 0.5) * 2.0 * (1.0 + 律.get("_ñ", 0.5)),
                                -40.0, 40.0))   # 微韵律（±20Hz 内，防音节间断层）
            韵母时长 = (0.10 if 调 == 5 else 0.18) * (3.2 / 语速)   # 秒（基准语速 3.2）
            目标s, 尾 = _查韵母(韵)
            # —— 声母帧 ——
            if 声:
                方式 = 声母表[声][0]
                if 方式 == "塞":
                    vot = 声母表[声][2] / 1000.0
                    帧数 = self._推帧(帧参数, vot, 0, -1, None, 律)
                    帧数 = self._推帧(帧参数, 0.005, 0, 0, (声母表[声][1], 600), 律)
                    送气帧 = 0.025 if not 声母表[声][3] else 0.045
                    帧数 = self._推帧(帧参数, 送气帧, 0, 0,
                                      (声母表[声][1], 1200), 律, 送气=律["送气量"])
                elif 方式 == "擦":
                    self._推帧(帧参数, 0.08 / 语速 * 3.2, 0, 0, 声母表[声][1], 律)
                elif 方式 == "塞擦":
                    self._推帧(帧参数, 0.010, 0, -1, None, 律)
                    self._推帧(帧参数, 0.005, 0, 0, (声母表[声][1][0], 600), 律)
                    self._推帧(帧参数, 0.04 / 语速 * 3.2, 0, 0, 声母表[声][1], 律)
                else:  # 浊声母（m/n/l/r）：低幅浊音+鼻位共振峰
                    目 = 浊声母表.get(声, (250, 1100, 2200))
                    度0 = _折线(声调表.get(调, 声调表[1]), 0.0)   # 协同发音：
                    f0起 = 律["基频"] * 2.0 ** (((度0 - 3.0) * 2.0
                            + ΔF0 / 12.0 - 下滑) / 12.0) * (1.0 + 0.5 * 律["声区"])
                    self._推浊帧(帧参数, 0.05, 律, f0起, 目, 幅倍=0.5)
            # —— 韵母帧：元音目标滑移 + 声调轮廓 ——
            帧数 = max(int(round(韵母时长 * 帧率)), 2)
            for f in range(帧数):
                t = f / max(帧数 - 1, 1)
                # 元音目标序列：多目标按时长等分滑移
                目标数 = len(目标s)
                段 = min(int(t * 目标数), 目标数 - 1)
                段内 = (t * 目标数) - 段
                f1a = np.array(目标s[段], dtype=np.float64)
                f1b = np.array(目标s[min(段 + 1, 目标数 - 1)], dtype=np.float64)
                ff = f1a + (f1b - f1a) * 段内
                if 尾 and t > 0.7:              # 鼻尾滑移（F2→1000, F3→2200）
                    w = (t - 0.7) / 0.3
                    ff = ff + (np.array([ff[0], 1000.0, 2200.0]) - ff) * w
                度 = _折线(声调表.get(调, 声调表[1]), t)
                半音 = (度 - 3.0) * 2.0 + ΔF0 / 12.0 - 下滑
                f0 = 律["基频"] * (2.0 ** (半音 / 12.0))
                # falsetto：声区↑ → F0 整体上移且谐波变纯（紧喉↓）
                f0 *= (1.0 + 0.5 * 律["声区"])
                包络 = min(t / 0.15, 1.0) if t < 0.15 else (
                    min((1.0 - t) / 0.25, 1.0) if t > 0.75 else 1.0)
                if 尾 and t > 0.85:
                    包络 *= max(1.0 - (t - 0.85) / 0.15 * 0.5, 0.5)
                帧参数["f0"].append(f0)
                帧参数["f1"].append(ff[0]); 帧参数["f2"].append(ff[1])
                帧参数["f3"].append(ff[2])
                帧参数["幅"].append(律["幅度"] * 包络)
                帧参数["气声"].append(律["气声度"])
                帧参数["声源"].append(1)
                帧参数["噪带"].append(None)
                帧参数["送气"].append(0.0)
            下滑 += (0.0 if 问句 else 0.15)            # 半音/词，疑问句不降
        if 问句 and 帧参数["f0"]:                       # 是非问句尾上扬 ~2 半音
            尾帧 = max(len(帧参数["f0"]) // 5, 2)
            for i in range(1, 尾帧 + 1):
                帧参数["f0"][-i] *= 2.0 ** ((2.0 * i / 尾帧) / 12.0)
        # —— meeting58：音色档案（共振峰三轴缩放 / 降调）+ 帧级紧喉/粗糙（原死参数接线）——
        音 = 律.get("_音色") or {}
        s = 音.get("共振峰缩放") or (1.0, 1.0, 1.0)
        try:
            s1, s2, s3 = float(s[0]), float(s[1]), float(s[2])
        except Exception:  # noqa: BLE001
            s1 = s2 = s3 = 1.0
        if abs(s1 - 1.0) + abs(s2 - 1.0) + abs(s3 - 1.0) > 1e-9:
            for i in range(len(帧参数["f1"])):
                帧参数["f1"][i] *= s1
                帧参数["f2"][i] *= s2
                帧参数["f3"][i] *= s3
        降 = float(音.get("降调半音", 0.0) or 0.0)
        if abs(降) > 1e-9:
            帧参数["f0"] = [float(f) * (2.0 ** (降 / 12.0)) for f in 帧参数["f0"]]
        帧参数["紧喉"] = [float(律.get("紧喉度", 0.3))] * len(帧参数["f0"])
        帧参数["粗糙"] = [float(律.get("粗糙度", 0.0))] * len(帧参数["f0"])
        return 帧参数

    def _推帧(self, 帧, 时长秒, f0, 声源, 噪带, 律, 送气=0.0) -> int:
        帧数 = max(int(round(时长秒 * 帧率)), 1)
        for _ in range(帧数):
            帧["f0"].append(f0); 帧["f1"].append(500.0); 帧["f2"].append(1500.0)
            帧["f3"].append(2500.0)
            帧["幅"].append(律["幅度"] * (0.4 if 声源 == 0 else 0.0))
            帧["气声"].append(律["气声度"]); 帧["声源"].append(声源)
            帧["噪带"].append(噪带); 帧["送气"].append(送气)
        return 帧数

    def _推浊帧(self, 帧, 时长秒, 律, f0, 目, 幅倍=1.0) -> None:
        for _ in range(max(int(round(时长秒 * 帧率)), 1)):
            帧["f0"].append(f0); 帧["f1"].append(目[0]); 帧["f2"].append(目[1])
            帧["f3"].append(目[2]); 帧["幅"].append(律["幅度"] * 幅倍)
            帧["气声"].append(律["气声度"]); 帧["声源"].append(1)
            帧["噪带"].append(None); 帧["送气"].append(0.0)

    # ---------- DSP：Rosenberg 声门源 + Klatt 级联（会议三号专家闭式） ----------
    @staticmethod
    def _rosenberg表(n: int = 512) -> np.ndarray:
        """单周期 Rosenberg 脉冲查表：Tp=0.4T0，Tn=0.16T0（相位归一）。"""
        φ = np.linspace(0.0, 1.0, n, endpoint=False)
        g = np.zeros_like(φ)
        m1 = φ < 0.4
        m2 = (φ >= 0.4) & (φ < 0.56)
        g[m1] = 0.5 * (1.0 - np.cos(np.pi * φ[m1] / 0.4))
        g[m2] = np.cos(np.pi * (φ[m2] - 0.4) / 0.32)
        return g.astype(np.float32)

    @staticmethod
    def _谐振器系数(F: float, BW: float, fs: int = FS):
        T = 1.0 / fs
        C = -np.exp(-2.0 * np.pi * BW * T)
        B = 2.0 * np.exp(-np.pi * BW * T) * np.cos(2.0 * np.pi * F * T)
        A = 1.0 - B - C
        return float(A), float(B), float(C)

    def 合成波形(self, 帧: dict) -> tuple:
        """三轨帧参数→波形（逐样本确定性）。返回 (波形, 元信息)。"""
        rng = np.random.default_rng(self.seed)          # 固定种子（S1 确定性）
        表 = self._rosenberg表()
        N = len(帧["f0"])
        总样本 = N * 每帧样本
        # 共振峰轨迹单极点低通轻平滑（会议三号专家：消除系数逐帧突变的台阶）
        α = 0.5
        for 键 in ("f1", "f2", "f3"):
            a = np.asarray(帧[键], dtype=np.float64)
            for i in range(1, N):
                a[i] = a[i - 1] + α * (a[i] - a[i - 1])
            帧[键] = a.tolist()
        # 半音域插值 F0（防跳变），幅度 dB 域，共振峰线性
        f0帧 = np.asarray(帧["f0"], dtype=np.float64)
        st帧 = 12.0 * np.log2(np.maximum(f0帧, 1.0) / 55.0)
        st样 = np.interp(np.arange(总样本) / 每帧样本,
                         np.arange(N), st帧)
        f0样 = 55.0 * 2.0 ** (st样 / 12.0)
        幅样 = np.interp(np.arange(总样本) / 每帧样本, np.arange(N),
                         np.asarray(帧["幅"], dtype=np.float64))
        幅样 = 10.0 ** (np.interp(np.arange(总样本) / 每帧样本, np.arange(N),
                                  20.0 * np.log10(np.maximum(
                                      np.asarray(帧["幅"], np.float64), 1e-4))) / 20.0)
        气样 = np.interp(np.arange(总样本) / 每帧样本, np.arange(N),
                         np.asarray(帧["气声"], np.float64))
        送气样 = np.interp(np.arange(总样本) / 每帧样本, np.arange(N),
                           np.asarray(帧["送气"], np.float64))
        声源帧 = np.asarray(帧["声源"])
        噪带帧 = 帧["噪带"]
        白噪 = rng.standard_normal(总样本)
        白噪 = np.diff(白噪, prepend=0.0)               # 一阶差分高通（去隆隆）
        # 段边界 5ms 线性淡入（会议三号专家：防咔哒）+ 谐振器状态切换清零标记
        声源样 = np.repeat(声源帧, 每帧样本)[:总样本]
        段包络 = np.ones(总样本)
        变点 = np.where(np.diff(声源样) != 0)[0] + 1
        for b in 变点:
            段包络[b:b + 40] = np.linspace(0.0, 1.0, min(40, 总样本 - b))
        变点集 = set(int(b) for b in 变点)
        φ = 0.0
        相位样 = np.empty(总样本)
        for i in range(总样本):
            φ += f0样[i] / FS
            φ -= int(φ)
            相位样[i] = φ
        相位idx = (相位样 * 512).astype(np.int64) & 511
        源 = 表[相位idx].astype(np.float64)
        # 粗糙度：±抖动（会议一号专家 jitter 映射黏膜波规则性）
        if 律抖动 > 0.0:
            抖 = 1.0 + 律抖动 * np.sin(2.0 * np.pi * 33.0 * np.arange(总样本) / FS
                                       + 0.7)
            源 *= 抖
        out = np.zeros(总样本)
        # 三级联谐振器状态
        y = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
        yn = (0.0, 0.0)
        每帧系数缓存 = {}
        for i in range(总样本):
            fi = min(i // 每帧样本, N - 1)
            浊 = 声源帧[fi] == 1
            噪带 = 噪带帧[fi]
            if i in 变点集:                              # 声源切换：谐振器状态清零
                y = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
                yn = (0.0, 0.0)
            x = 0.0
            if 浊:
                ah = 气样[i]                              # 气声度：噪声混入
                x = (1.0 - 0.5 * ah) * 源[i] + ah * 0.125 * 白噪[i]
            elif 声源帧[fi] == 0 and 噪带 is not None:
                x = 白噪[i] * 0.25 * (1.0 + 送气样[i])    # ×0.25 防高频支路过响
            x *= 段包络[i]
            if x == 0.0:
                out[i] = 0.0
                continue
            if 浊:
                f1, f2, f3 = (帧["f1"][fi], 帧["f2"][fi], 帧["f3"][fi])
                key = (round(f1), round(f2), round(f3))
                系 = 每帧系数缓存.get(key)
                if 系 is None:
                    系 = [self._谐振器系数(f1, 80.0),
                          self._谐振器系数(f2, 110.0),
                          self._谐振器系数(f3, 170.0)]
                    每帧系数缓存[key] = 系
                s = x
                新y = []
                for (A, B, C), (y1, y2) in zip(系, y):
                    v = A * s + B * y1 + C * y2
                    新y.append((v, y1))
                    s = v
                y = 新y
                out[i] = s
            else:
                Fc, BWc = (噪带 if isinstance(噪带, tuple) and len(噪带) == 2
                           else (噪带, 1000.0))
                key = ("n", round(Fc), round(BWc))
                系 = 每帧系数缓存.get(key)
                if 系 is None:
                    系 = self._谐振器系数(min(Fc, FS / 2 - 200), BWc)
                    每帧系数缓存[key] = 系
                A, B, C = 系
                y1, y2 = yn
                v = A * x + B * y1 + C * y2
                yn = (v, y1)
                out[i] = v
        out *= 幅样
        # 峰值归一化（免 tanh 失真，会议三号专家）
        峰 = float(np.max(np.abs(out))) if 总样本 else 0.0
        if 峰 > 0.9:
            out *= 0.9 / 峰
        元信息 = dict(样本数=总样本, 峰值=峰, 帧数=N, f0样=f0样)
        return out, 元信息

    # ---------- meeting58：音色档案驱动的分块合成（新路径；旧路径保持逐位不变） ----------
    @staticmethod
    def _紧喉表(n: int, 紧喉: float) -> np.ndarray:
        """Rosenberg 声门脉冲：紧喉↑ ⇒ 开相变短、关闭更陡（声门开商下降）。"""
        Tp = _夹(0.40 - 0.16 * float(紧喉), 0.20, 0.40)
        Tn = _夹(0.16 + 0.04 * float(紧喉), 0.12, 0.24)
        φ = np.linspace(0.0, 1.0, n, endpoint=False)
        g = np.zeros_like(φ)
        m1 = φ < Tp
        m2 = (φ >= Tp) & (φ < Tp + Tn)
        g[m1] = 0.5 * (1.0 - np.cos(np.pi * φ[m1] / Tp))
        g[m2] = np.cos(np.pi * (φ[m2] - Tp) / (2.0 * Tn))
        return g.astype(np.float32)

    @staticmethod
    def _滤波块(A: float, B: float, C: float, x: np.ndarray, y1: float, y2: float):
        """二阶 IIR 块滤波：y[n]=A·x[n]+B·y[n-1]+C·y[n-2]；有 scipy 走 lfilter，否则逐样本。"""
        if _lfilter is not None:
            y, _ = _lfilter([A], [1.0, -B, -C], x,
                            zi=np.array([B * y1 + C * y2, C * y1], dtype=np.float64))
            return y, float(y[-1]), float(y[-2] if len(y) > 1 else y[-1])
        y = np.empty_like(x)
        for i in range(len(x)):
            v = A * x[i] + B * y1 + C * y2
            y[i] = v
            y2, y1 = y1, v
        return y, y1, y2

    def 合成波形_快(self, 帧: dict, 音色: dict = None):
        """新路径：音色档案真接线 + 分块 IIR（构造与合成都更快）。

        与旧 `合成波形` 的差别（只在传入音色时启用，旧路径逐位不变）：
          紧喉度 → 声门开商；粗糙度 → 33Hz jitter（不再用常量 0.005）；
          共振峰缩放/带宽缩放/谱倾斜 → 在 音节帧 与谐振器/后处理里；
          幅度/增益 → **后置增益**（修复旧路径被峰值归一化吃掉）。
        """
        音色 = dict(音色 or {})
        N = len(帧["f0"])
        if N == 0:
            return np.zeros(0), dict(样本数=0, 峰值=0.0, 帧数=0, f0样=np.zeros(0))
        总样本 = N * 每帧样本
        # 共振峰轨迹平滑（与旧路径同参数）
        α = 0.5
        f1 = np.asarray(帧["f1"], dtype=np.float64).copy()
        f2 = np.asarray(帧["f2"], dtype=np.float64).copy()
        f3 = np.asarray(帧["f3"], dtype=np.float64).copy()
        for a in (f1, f2, f3):
            for i in range(1, N):
                a[i] = a[i - 1] + α * (a[i] - a[i - 1])
        # 帧 → 样本插值
        f0帧 = np.asarray(帧["f0"], dtype=np.float64)
        st帧 = 12.0 * np.log2(np.maximum(f0帧, 1.0) / 55.0)
        样idx = np.arange(总样本, dtype=np.float64) / 每帧样本
        st样 = np.interp(样idx, np.arange(N), st帧)
        f0样 = 55.0 * 2.0 ** (st样 / 12.0)
        幅帧 = np.asarray(帧["幅"], dtype=np.float64)
        幅样 = 10.0 ** (np.interp(样idx, np.arange(N),
                                  20.0 * np.log10(np.maximum(幅帧, 1e-4))) / 20.0)
        气帧 = np.asarray(帧["气声"], dtype=np.float64)
        气样 = np.interp(样idx, np.arange(N), 气帧)
        送气帧 = np.asarray(帧["送气"], dtype=np.float64)
        送气样 = np.interp(样idx, np.arange(N), 送气帧)
        声源帧 = np.asarray(帧["声源"])
        噪带帧 = 帧["噪带"]
        紧喉帧 = np.asarray(帧.get("紧喉", [0.3] * N), dtype=np.float64)
        粗糙帧 = np.asarray(帧.get("粗糙", [0.0] * N), dtype=np.float64)
        # 噪声（与旧路径同源：一次抽 总样本 后差分；跨路径不要求逐位同）
        rng = np.random.default_rng(self.seed)
        白噪 = np.diff(rng.standard_normal(总样本), prepend=0.0)
        # 段包络（5ms 淡入，同旧）
        声源样 = np.repeat(声源帧, 每帧样本)[:总样本]
        段包络 = np.ones(总样本)
        变点 = np.where(np.diff(声源样) != 0)[0] + 1
        for b in 变点:
            段包络[b:b + 40] = np.linspace(0.0, 1.0, min(40, 总样本 - b))
        # 相位 & 紧喉查表
        φ样 = (np.cumsum(f0样 / FS) + 0.0) % 1.0
        相位idx = (φ样 * 512).astype(np.int64) & 511
        lv = np.clip(np.round(紧喉帧 * 8.0).astype(int), 0, 8)
        表列 = [self._紧喉表(512, k / 8.0) for k in range(9)]
        源 = np.empty(总样本, dtype=np.float64)
        for fi in range(N):
            a, b = fi * 每帧样本, (fi + 1) * 每帧样本
            源[a:b] = 表列[int(lv[fi])][相位idx[a:b]]
        # 粗糙度 → 33Hz AM（接线；旧路径为常量 0.005）
        t样 = np.arange(总样本, dtype=np.float64) / FS
        粗样 = np.interp(样idx, np.arange(N), 粗糙帧)
        源 *= (1.0 + 0.15 * 粗样 * np.sin(2.0 * np.pi * 33.0 * t样 + 0.7))
        # 音色后处理参数
        倾斜 = float(音色.get("谱倾斜", 0.0) or 0.0)
        bw = _夹(float(音色.get("带宽缩放", 1.0) or 1.0), 0.5, 2.0)
        送气增益 = _夹(float(音色.get("送气增益", 0.0) or 0.0), 0.0, 1.0)
        # 逐帧滤波（相位/谐振器状态跨帧保持；声源切换时清状态，同旧）
        输出 = np.zeros(总样本)
        三阶 = [[0.0, 0.0] for _ in range(3)]     # 每个谐振器 (y1,y2)
        一阶 = [0.0, 0.0]
        每帧系数缓存 = {}
        for fi in range(N):
            a, b = fi * 每帧样本, (fi + 1) * 每帧样本
            浊 = 声源帧[fi] == 1
            ah = float(气样[a])
            送 = float(送气样[a])
            if 浊:
                x = (1.0 - 0.5 * ah) * 源[a:b] + ah * 0.125 * 白噪[a:b] \
                    + 送气增益 * 送 * 白噪[a:b]
            elif 声源帧[fi] == 0 and 噪带帧[fi] is not None:
                x = 白噪[a:b] * 0.25 * (1.0 + 送)
            else:
                x = np.zeros(b - a)
            x = x * 段包络[a:b]
            if fi == 0 or 声源帧[fi] != 声源帧[fi - 1]:
                三阶 = [[0.0, 0.0] for _ in range(3)]
                一阶 = [0.0, 0.0]
            if 浊:
                for j, (F, BW0) in enumerate(((float(f1[fi]), 80.0),
                                              (float(f2[fi]), 110.0),
                                              (float(f3[fi]), 170.0))):
                    key = (round(F), round(BW0 * bw))
                    系 = 每帧系数缓存.get(key)
                    if 系 is None:
                        系 = self._谐振器系数(_夹(F, 20.0, FS / 2 - 200), BW0 * bw)
                        每帧系数缓存[key] = 系
                    x, 三阶[j][0], 三阶[j][1] = self._滤波块(系[0], 系[1], 系[2], x,
                                                            三阶[j][0], 三阶[j][1])
            else:
                噪带 = 噪带帧[fi]
                if 噪带 is not None:
                    Fc, BWc = (噪带 if isinstance(噪带, tuple) and len(噪带) == 2
                               else (噪带, 1000.0))
                    key = ("n", round(Fc), round(BWc))
                    系 = 每帧系数缓存.get(key)
                    if 系 is None:
                        系 = self._谐振器系数(min(Fc, FS / 2 - 200), BWc)
                        每帧系数缓存[key] = 系
                    x, 一阶[0], 一阶[1] = self._滤波块(系[0], 系[1], 系[2], x,
                                                      一阶[0], 一阶[1])
            输出[a:b] = x * 幅样[a:b]
        # 谱倾斜（一阶预加重/去加重）
        if abs(倾斜) > 1e-9:
            y = np.empty_like(输出)
            y[0] = 输出[0] * (1.0 - 倾斜)
            y[1:] = 输出[1:] - 倾斜 * 输出[:-1]
            输出 = y / (1.0 + 0.5 * abs(倾斜))
        # 峰值归一 + 后置增益（修复「幅度被归一化吃掉」）
        峰 = float(np.max(np.abs(输出))) if 总样本 else 0.0
        if 峰 > 1e-12:
            输出 = 输出 * (0.9 / 峰)
        幅度 = _夹(float(音色.get("幅度", 0.5) or 0.5), 0.1, 1.0)
        增益 = float(音色.get("增益", 1.0) or 1.0)
        后置 = _夹(0.6 + 0.8 * 幅度, 0.4, 1.5) * 增益
        输出 = np.tanh(1.2 * 输出 * 后置) / np.tanh(1.2)
        元信息 = dict(样本数=总样本, 峰值=float(np.max(np.abs(输出))),
                      帧数=N, f0样=f0样,
                      音色指纹=_音色指纹(音色), 快路径=True)
        return 输出, 元信息

    # ---------- 主入口 ----------
    def 发声(self, 向量: np.ndarray, 文本: str, 遥测: dict = None,
             音色: dict = None) -> dict:
        """向量+文本+遥测 → 发音参数 → wav。返回溯源齐全的结果字典。

        `音色`（meeting58 新增，向后兼容）：
          8 参数（基频/幅度/气声度/紧喉度/粗糙度/送气量/声区）覆盖韵律；
          扩展键：共振峰缩放(3元组)/带宽缩放/谱倾斜/送气增益/降调半音/增益。
          传入音色 ⇒ 走 `合成波形_快`（紧喉/粗糙/幅度/共振峰/带宽/谱倾斜全接线）；
          不传 ⇒ 走旧 `合成波形`（逐位不变，保既有回归）。
        """
        遥测 = 遥测 or {}
        律 = self.韵律(遥测)
        律["_ñ"] = float(遥测.get("新异", 0.5) or 0.0)
        if 音色:
            音色 = dict(音色)
            for k in tuple(参数表) + ("语速",):
                if k in 音色:
                    if k == "基频":
                        律[k] = _夹(音色[k], 基频区间[0], 基频区间[1])
                    elif k == "语速":
                        律[k] = _夹(音色[k], 1.5, 7.0)
                    else:
                        律[k] = _夹(音色[k], 0.0, 1.0)
            律["_音色"] = 音色
        qs = self.分位读数(向量)
        t0 = time.perf_counter()
        帧 = self.音节帧(文本, 律, qs)
        if 音色:
            波形, 元 = self.合成波形_快(帧, 音色)
        else:
            波形, 元 = self.合成波形(帧)
        合成耗时 = time.perf_counter() - t0
        时长 = 元["样本数"] / FS
        # —— 自检（会议四号专家：合法性 + 数轴往返；免人耳）——
        参数s = {k: 律[k] for k in 参数表}
        合法, 违例 = self._参数合法(参数s)
        往返 = self._数轴往返(参数s)
        # 数值健康（DSP 判据 5）：防咔哒=声源切换边界 + 离群台阶检测。
        # 高 F0 大声的连续振荡斜率天然可超 0.5（非缺陷），故叠加局部均值倍率：
        # |Δ|>0.5 且 >8×局部平均|Δ| 才计跳变（真咔哒是离群台阶）。
        无nan = bool(np.all(np.isfinite(波形)))
        声源样 = np.repeat(np.asarray(帧["声源"]), 每帧样本)[:len(波形)]
        跳变 = 数跳变(波形, 声源样, 元.get("f0样"))
        # —— wav 落盘（FIFO 有界，S9）——
        self.发声号 += 1
        名 = "发声_%06d.wav" % self.发声号
        路径 = os.path.join(self.目录, 名)
        self._写wav(路径, 波形)
        self._wav.append(路径)
        while len(self._wav) > WAV上限:
            旧 = self._wav.pop(0)
            try:
                os.remove(旧)
            except OSError:
                pass
        实时率 = 合成耗时 / max(时长, 1e-9)
        rec = dict(发声号=self.发声号, 时刻=time.strftime("%H:%M:%S"),
                   文本=文本, 拼音数=len(帧["f0"]),
                   遥测={k: float(v) for k, v in 遥测.items()
                         if isinstance(v, (int, float))},
                   参数=参数s, 语速=律["语速"],
                   时长秒=round(时长, 3), 实时率=round(实时率, 4),
                   wav=os.path.basename(路径),
                   音色指纹=(_音色指纹(音色) if 音色 else ""),
                   快路径=bool(音色),
                   自检=dict(参数合法=合法, 违例=违例, 数轴往返=往返,
                             无NaN=无nan, 跳变=跳变, 峰值=元["峰值"]),
                   通过=bool(合法 and 往返 == 1.0 and 无nan and 跳变 == 0
                             and 实时率 <= 0.5))
        with open(self.台账路径, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        rec["波形"] = 波形 if len(波形) < 4_000_000 else None
        return rec

    # ---------- meeting58：流式输出入口（整句合成后分块提交；首块延迟≈合成耗时） ----------
    def 发声_流式(self, 向量: np.ndarray, 文本: str, 遥测: dict = None,
                  音色: dict = None, 每块秒: float = 0.10):
        """生成器：逐块产出 (块号, 起始秒, 波形块)。块大小=每块秒；确定性同 `发声`。

        说明：当前实现是「整句一次合成后按 100ms 分块提交」，首块延迟≈合成耗时
        （实测 2.19s 音频约 17ms）；真正的边合成边提交（帧级流水）仍在 P3。
        """
        rec = self.发声(向量, 文本, 遥测, 音色=音色)
        x = rec.get("波形")
        if x is None:
            return
        块 = max(1, int(round(float(每块秒) * FS)))
        for i in range(0, len(x), 块):
            yield dict(块号=i // 块, 起始秒=round(i / FS, 4),
                       波形=x[i:i + 块], 发声号=rec["发声号"],
                       音色指纹=rec.get("音色指纹", ""))

    def _参数合法(self, 参数s: dict) -> tuple:
        合法 = True
        违例 = []
        if not (基频区间[0] <= 参数s["基频"] <= 基频区间[1]):
            合法 = False; 违例.append("基频")
        for k in ("幅度", "气声度", "紧喉度", "粗糙度", "送气量", "声区"):
            if not (0.0 <= 参数s[k] <= 1.0):
                合法 = False; 违例.append(k)
        return 合法, 违例

    def _数轴往返(self, 参数s: dict) -> float:
        """值→数轴编码→解码档位→值→重编码档位 一致率（100% 为过）。
        meeting58：按参数值缓存（同参数重复发声不再重算 7×257 原型扫描）。"""
        键 = tuple(sorted((k, round(float(v), 6)) for k, v in 参数s.items()))
        r = _往返缓存.get(键)
        if r is not None:
            return r
        一致 = 总 = 0
        for 名, 值 in 参数s.items():
            轴 = self._轴[名]
            k = 轴.编码档位(值)
            v = 轴.编码向量(值)
            k2 = 轴.解码档位(v)
            一致 += int(k2 == k); 总 += 1
        r = 一致 / max(总, 1)
        if len(_往返缓存) > 4096:
            _往返缓存.clear()
        _往返缓存[键] = r
        return r

    def _写wav(self, 路径: str, 波形: np.ndarray) -> None:
        import wave
        x = np.clip(波形, -1.0, 1.0)
        pcm = (x * 32767.0).astype("<i2")               # 显式小端（跨平台逐位一致）
        with wave.open(路径, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(FS)
            w.writeframes(pcm.tobytes())

    # ---------- 外放（可选；默认关闭——无声卡零异常，S7） ----------
    def 开外放(self) -> bool:
        if self._降级:
            return False
        try:
            import pyttsx3
            self._外放引擎 = pyttsx3.init()
            self._外放引擎.setProperty("rate", 180)
            self.外放 = True
            return True
        except Exception:
            self._降级 = True
            self.外放 = False
            return False

    def 播放(self, 文本: str) -> bool:
        """外放文本（pyttsx3 兜底驱动，会议五号专家：并存不替换）。"""
        if not self.外放 or self._降级:
            return False
        try:
            self._外放引擎.say(文本)
            self._外放引擎.runAndWait()
            return True
        except Exception:
            self._降级 = True; self.外放 = False        # 单次异常永久降级
            return False

    # ---------- 溯源 ----------
    def 台账流(self, n: int = 10) -> list:
        if not os.path.exists(self.台账路径):
            return []
        with open(self.台账路径, "r", encoding="utf-8") as f:
            行s = f.readlines()
        return [json.loads(l) for l in 行s[-n:]]

    # ---------- 集成挂载（会议五号专家：零侵入旁路协议） ----------
    def 接综合体(self, 桥, 耳=None) -> None:
        """挂到 综合体 上：包一层 桥.步——回复后发声+独立台账（不扰 V5 重放）。
        耳=耳朵器官 时启用自我听觉反馈：说完听自己（说→听→记住自己的声音）。"""
        原步 = 桥.步
        器官 = self

        def 步带声带(问, *a, **kw):
            r = 原步(问, *a, **kw)
            try:
                响 = str(r.get("响应", ""))[:64]
                if 响.strip():
                    遥测 = dict(
                        能量=getattr(桥, "e", None) or 0.5,
                        新异=getattr(桥, "ñ", None) or 0.5,
                        消化债=getattr(桥, "消化债", None) or 0.0,
                        饱足=getattr(桥, "饱足", None) or 0.5)
                    rec = 器官.发声(
                        getattr(桥, "编码", None) or
                        np.sign(np.random.default_rng(
                            器官.seed + 器官.发声号).standard_normal(
                                器官.D)).astype(np.float32),
                        响, 遥测)
                    if 耳 is not None and rec.get("波形") is not None:
                        听到 = 耳.听(rec["波形"])
                        耳.学(听到, 响)             # 自我听觉反馈：记住自己的声音
                        rec["自我听觉"] = True
            except Exception as e:                     # 发声永不拖垮主体（S7 精神）
                with open(os.path.join(os.path.dirname(self.台账路径),
                                       "声带异常.log"), "a",
                          encoding="utf-8") as f:
                    f.write("[%s] %s: %s\n" % (time.strftime("%H:%M:%S"),
                                               type(e).__name__, str(e)[:120]))
            return r

        桥.步 = 步带声带
        桥.声带 = self
        return None


律抖动 = 0.005                   # Rosenberg 粗糙度抖动幅度（±0.5%，会议三号专家）


def 数跳变(波形: np.ndarray, 声源样: np.ndarray, f0样: np.ndarray = None) -> int:
    """防咔哒判据（孤立台阶）：|Δ|>0.5 且 >8×局部平均|Δ| 且在浊音段/类型边界。
    周期性排除：若回看 1~2 个基频周期处有同级尖峰 → 声门闭合脉冲的周期激励
    （生理正确），不计。真咔哒是无周期结构的孤立台阶。"""
    if len(波形) <= 1:
        return 0
    d = np.abs(np.diff(波形))
    局部 = np.convolve(d, np.ones(31) / 31.0, mode="same") + 1e-9
    边界 = (声源样[:-1] != 声源样[1:]) | (声源样[:-1] == 1)
    超限 = (d > 0.5) & (d > 8.0 * 局部) & 边界
    if not 超限.any():
        return 0
    if f0样 is None or len(f0样) < len(波形):
        f0样 = np.full(len(波形), 200.0)
    候选 = np.where(超限)[0]
    计 = 0
    for i in 候选:
        T0 = max(int(FS / max(f0样[min(i, len(f0样) - 1)], 1.0)), 10)
        周期 = False
        for 倍 in (1, 2):
            j = i - 倍 * T0
            if j >= 0 and d[max(j - 3, 0):j + 4].max() > 0.8 * d[i]:
                周期 = True
                break
        if not 周期:
            计 += 1
    return 计
