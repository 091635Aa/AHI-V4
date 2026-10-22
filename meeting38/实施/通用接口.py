# -*- coding: utf-8 -*-
"""通用接口.py —— AHI 通用输入/输出契约 + 全接口预处理 + 维度伸缩原语（Phase2）。

所有者（2026-10-06）：「把引擎底层外部接口学清楚，把通用输入和通用输出做出来——输入输出是
一组向量；扩容向量空间（优化引擎上下文/内部计算空间）；加一个全接口预处理（通用文案+适配+
简单引导），未来就能让 AI 有音频输入/说话的能力。」

本文件只做**纯接口层**，把「任意模态 → 统一内部表示 → 任意模态」这条链固化下来：

  ① 统一输入  任意外部源(text / D 维向量 / 任意维实数特征) → dict(文本, 向量, 模态)
  ② 统一输出  内部文本 → 目标通道(text 直返 / 自定义钩子——音频 TTS 等未来接入点)
  ③ 全接口预处理  任意维实数特征 → D 维 HDC 双极（固定随机投影，零训练、零梯度）
  ④ 维度伸缩原语  升维(桥, 新D)：set lang_codec.D → 重建(载入事件重放) → 记忆逐位再生

铁律（严格）：
  · 眼/耳/手/嘴 **冻结**——本文件不 import、不实例化任何器官（器官_嘴/眼/耳/手/声带/耳朵）。
    音频路径只到「钩子」这一层：注册者可插拔；未注册 → 确定性降级为文本（不报错、不崩）。
  · C1 禁 Transformer（纯 HDC：随机投影 + 字袋 + 双极）；C10 禁外挂模型当器官；
  · C13 确定性可复现：同 seed + 同源 + 同模态 = 逐位一致（无时间、无全局随机态）。

「全接口预处理」是**契约**不是器官：它把「未来耳/嘴」的接入工作压缩成两件小事——
  输入侧：任何前端把音频特征（f 维实数）交给 统一输入.音频特征()，即得可驱引擎的 D 维向量；
  输出侧：任何后端把 TTS 函数注册进 统一输出.注册钩子("音频", f)，即接管「说话」。
两端各自只依赖本契约的签名，不依赖对方实现——这就是「做一些适配 + 简单引导」即可通电。
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AHI = os.path.dirname(os.path.dirname(HERE))                 # AHI3 根（迁移：随文件定位）
for _p in (HERE, AHI, os.path.join(AHI, "meeting28")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def 当前D() -> int:
    """当前向量空间维度（唯一真相源=lang_codec.D；运行期可被维度伸缩改写）。"""
    import lang_codec
    return int(lang_codec.D)


# ══════════════ ★ P0-C13 修复：内建 hash() → sha256 确定性派生 ══════════════
# 病（meeting40 审计 §9-2 / 终稿 §7.7·§8 P0 / §11 风险 6）：`hash((str,int))` 带每进程
#   随机化的字符串哈希盐（PYTHONHASHSEED）⇒ 同一 (模态, f) 在不同进程得到**不同投影阵**，
#   跨进程不可复现，违 C13。旧路径的 `& 0x7FFFFFFF` 语义保留在回退分支里。
# 修：sha256("投影|模态|f") 前 4 字节 → 31 位非负整数，纯确定性、与 PYTHONHASHSEED 无关。
# 回退（逐位回到旧路径，A/B 对照用）：环境变量 AHI_C13_旧哈希=1，或置 `修复C13 = False`。
修复C13 = os.environ.get("AHI_C13_旧哈希", "") not in ("1", "true", "True", "yes")


def _键整数(key) -> int:
    """(模态, f) → 31 位非负整数。默认 sha256 确定性派生；回退=内建 hash（旧行为逐位不变）。"""
    if not 修复C13:
        return hash(key) & 0x7FFFFFFF
    import hashlib
    s = "投影|%s|%d" % (str(key[0]), int(key[1]))
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:4], "big") & 0x7FFFFFFF


# ══════════════ ★ P5：随机投影**保距诊断** + **幅度保留**（默认关 ⇒ 逐位恒等旧路径） ══════════════
# 病（预检）：`统一输入.音频特征/特征` = `sign(Pᵀv)`，sign 丢弃全部幅度；随机投影的保距性
#   （Johnson–Lindenstrauss 式）**从未验证**——属"临时凑数"契约层。
# 修：① 提供 `保距诊断()`：投影前/后两两余弦的一致性（Pearson 相关），真投影相关高、sham 判不过；
#     ② `保幅度`（环境变量 AHI_保幅度=1 开启）：不 sign 量化，改**保方向+保相对幅度**的归一化投影。
# 默认关 ⇒ 全链路逐位恒等旧路径（C14）。
保幅度 = os.environ.get("AHI_保幅度", "") in ("1", "true", "True", "yes")


# ══════════════════════ ① 统一输入（全接口预处理） ══════════════════════

class 统一输入:
    """任意外部源 → 统一内部表示 dict(文本, 向量, 模态)。

    模态（可插拔，签名固定）：
      "文本"      str → (文本=s, 向量=codec.编码(s))
      "向量"      D 维实数 → (文本=None/标签, 向量=归一化后的双极/原向量)  # 直驱引擎
      "音频特征"  f 维实数 → (文本=None, 向量=sign(P_f @ 特征))            # 全接口预处理
      "特征"     同 音频特征（通用实数特征，任意 f）——由 注册适配器() 扩展更多模态

    「扩容向量空间」体现：多模态统一到**同一个 D 维向量空间**；D 由 lang_codec.D 决定，
    升维后本类自动按新 D 建投影（无需改代码）——这是「优化引擎上下文/内部计算空间」的入口。
    """

    def __init__(self, codec, seed: int = 0):
        self.codec = codec
        self.seed = int(seed)
        self._投影: dict = {}                    # (模态, 特征维度) → (f, D) 投影阵
        self._适配器: dict = {}                  # 模态 → callable(源) → dict(文本,向量,模态)
        self._计数: dict = {}                    # 模态 → 接收条数（可审计）

    # ── 全接口预处理：任意维实数特征 → D 维双极（固定随机投影，零训练） ──
    def _投影阵(self, 模态: str, f: int) -> np.ndarray:
        key = (str(模态), int(f))
        P = self._投影.get(key)
        if P is None:
            D = 当前D()
            P = (np.random.default_rng((self.seed, 94001, _键整数(key)))
                 .standard_normal((int(f), D)) / np.sqrt(max(int(f), 1))).astype(np.float32)
            self._投影[key] = P
        return P

    def _双极(self, z: np.ndarray) -> np.ndarray:
        b = np.sign(np.asarray(z, dtype=np.float32)).astype(np.float32)
        b[b == 0] = 1.0
        return b

    def _保距(self, z: np.ndarray) -> np.ndarray:
        """P5 幅度保留（非 sign）：归一化到 L2=√D（与 ±1 向量同量级），**保留方向与相对幅度**。"""
        z = np.asarray(z, dtype=np.float32)
        n = float(np.linalg.norm(z)) + 1e-9
        return (z / n * np.sqrt(float(z.shape[0]))).astype(np.float32)

    def 保距诊断(self, 模态: str = "特征", f: int = 64, 样本: int = 64, seed: int = 1234) -> dict:
        """P5 随机投影保距性诊断（可失败）：样本两两余弦在投影前后的 Pearson 相关。

        真随机投影：相关 ≈ 1（JL 保距）；sham（零阵/置换错配）：相关 ≈ 0。
        纯确定性（default_rng 显式种子），只读，不改任何投影缓存。"""
        rng = np.random.default_rng(int(seed))
        X = rng.standard_normal((int(样本), int(f)))
        P = self._投影阵(str(模态), int(f))
        Z = X @ P                                              # (样本, D) 投影（不量化）

        def 两两余弦(A):
            An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
            C = An @ An.T
            iu = np.triu_indices(A.shape[0], 1)
            return C[iu]

        c前, c后 = 两两余弦(X), 两两余弦(Z)
        r = float(np.corrcoef(c前, c后)[0, 1]) if c前.std() > 0 and c后.std() > 0 else 0.0
        return dict(相关=round(r, 4), 样本=int(样本), f=int(f), D=int(当前D()))

    # ── 三类内置模态 ──
    def 文本(self, s) -> dict:
        t = str(s)
        return dict(文本=t, 向量=self.codec.编码(t), 模态="文本")

    def 向量(self, v, 标签: str = None) -> dict:
        """一组向量直驱：不足 D 维→确定性补零并置符号；超 D 维→截断（有界、可复现）。"""
        D = 当前D()
        a = np.asarray(v, dtype=np.float32).ravel()
        if a.shape[0] < D:
            a = np.concatenate([a, np.zeros(D - a.shape[0], dtype=np.float32)])
        elif a.shape[0] > D:
            a = a[:D]
        return dict(文本=标签, 向量=self._双极(a), 模态="向量")

    def 音频特征(self, 特征, 标签: str = None) -> dict:
        """音频特征（f 维实数）→ D 维。此处不接任何耳器官——纯数学适配。

        P5：默认 `sign(Pᵀv)` 双极（逐位同旧）；`AHI_保幅度=1` 时改 `_保距` 归一化投影（保留幅度）。"""
        v = np.asarray(特征, dtype=np.float32).ravel()
        P = self._投影阵("音频特征", v.shape[0])
        z = P.T @ v
        向量 = self._保距(z) if 保幅度 else self._双极(z)
        return dict(文本=标签, 向量=向量, 模态="音频特征")

    特征 = 音频特征                              # 通用实数特征（同构，任意 f）

    # ── 适配器插拔（新模态只需注册一个纯函数） ──
    def 注册适配器(self, 模态: str, 函数) -> None:
        self._适配器[str(模态)] = 函数

    # ── 统一入口 ──
    def 预处理(self, 源, 模态: str = "文本") -> dict:
        self._计数[模态] = self._计数.get(模态, 0) + 1
        if 模态 in self._适配器:
            d = self._适配器[模态](源)
            d.setdefault("模态", 模态)
            return d
        if 模态 == "文本":
            return self.文本(源)
        if 模态 == "向量":
            return self.向量(源)
        if 模态 in ("音频特征", "特征"):
            return self.音频特征(源)
        raise KeyError("未注册的模态：%r（先 注册适配器()）" % 模态)

    def 计数(self) -> dict:
        return dict(self._计数)


# ══════════════════════ ② 统一输出（可插拔通道） ══════════════════════

class 统一输出:
    """内部文本 → 目标通道。默认通道="文本"（直返文本，供嘴走纯文本流）。

    可插拔：注册钩子(模态, 函数)。未来「说话」= 注册钩子("音频", tts)；本类不 import 声带。
    钩子未注册 → 确定性降级为文本（诚实：没接就是不接，不假装说了）。
    """

    def __init__(self, 默认: str = "文本"):
        self.默认 = str(默认)
        self.钩子: dict = {}
        self._发声数: dict = {}

    def 注册钩子(self, 模态: str, 函数) -> None:
        self.钩子[str(模态)] = 函数

    def 摘钩子(self, 模态: str) -> None:
        self.钩子.pop(str(模态), None)

    def 输出(self, 文本: str, 目标: str = None, **kw) -> dict:
        目标 = str(目标) if 目标 else self.默认
        self._发声数[目标] = self._发声数.get(目标, 0) + 1
        if 目标 in self.钩子:
            try:
                return dict(通道=目标, 结果=self.钩子[目标](str(文本), **kw), 降级=False)
            except Exception as ex:                    # 钩子自身故障 → 降级文本（不让接口崩）
                return dict(通道="文本", 结果=str(文本), 降级=True, 错=repr(ex))
        return dict(通道="文本", 结果=str(文本), 降级=(目标 != "文本"))

    def 计数(self) -> dict:
        return dict(self._发声数)


# ══════════════════════ ④ 维度伸缩原语（升维） ══════════════════════

def 升维(桥, 新D: int) -> "object":
    """把一座对话桥升/换维到 新D：set lang_codec.D → 保存事件史 → 载入重放 → 逐位再生。

    确定性（C13）：同 seed + 同事件序列 → 同新维状态（无随机、无时间）。
    记忆保留（判据）：升维后 codec._段（学过的每一句）逐条仍在，检索阈值口径不变。
    返回**新的桥**（调用方需替换引用；旧桥不销毁，便于对照）。不破既有：档格式向后兼容。
    """
    import tempfile
    import lang_codec
    新D = int(新D)
    旧D = int(lang_codec.D)
    fd, tmp = tempfile.mkstemp(suffix=".升维档.pkl")
    os.close(fd)
    try:
        桥.保存(tmp)                                   # 事件重放式入档（含开放域根/教史）
        lang_codec.D = 新D                             # ① 改唯一真相源
        新桥 = type(桥).载入(tmp, 维度=新D)            # ② 重建 codec@新D + 重建探针 + 重放事件
                                                        #    显式 维度=新D 压过档内旧 D（否则回退）
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return 新桥


def 记忆快照(桥) -> dict:
    """升维前后对照用：记忆段数/账本数/自言数/回复台账数。"""
    return dict(段数=len(getattr(桥.codec, "_段", []) or []),
                账本=len(getattr(桥, "账本", []) or []),
                自言=len(getattr(getattr(桥, "自言", None), "流", []) or []),
                回复台账=len(getattr(桥, "回复台账", []) or []),
                拍=int(getattr(桥, "拍", 0)))