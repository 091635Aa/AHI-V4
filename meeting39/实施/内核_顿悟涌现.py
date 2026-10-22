# -*- coding: utf-8 -*-
"""内核_顿悟涌现.py —— meeting39 步6：深度顿悟涌现（去 if；默认关·只加挂）。

设计出处（meeting39 会议纪要 §2 共识；专家01 §4.2/§4.3；专家05 §2 步6 / §3.3 F2）：
  步6 契约原文：`以临界跃迁/相空间事件替代 if c>=0.8；对齐理论① λ≈0`
              `依赖 步3、步4 ｜ 验收 F2（顿悟事件不落在预注册常量阈值上）｜ 回退 旧阈值路径`。

  病灶（专家01 §4.1，源码级已核实）：`synthesis.py:202 if cosv>=0.8 and 试次>=1`、
  `meeting30/selflearn.py:71 if c>=0.8`——"顿悟"被写成**两个预注册常数阈值的比较**，
  是**程序**（流程图）而非**涌现**（从系统自身状态里长出来）。

  替代机制（专家01 §4.2；纪要 §2）：顿悟 = **自模型失配的相变事件**，判据是**动力学形态**
  而非单一阈值。本实现在**真机实测**上落地为「三者同拍共现」（全部为**窗口局部自归一**比较，
  无任何预注册常数阈值）：
    (a) **失配峰→回落**（ε_self 一阶相变）：前窗 ε 的极差 > 后窗极差（"预测失灵"的峰在前窗）
        ∧ 后窗 ε 均值 < 前窗均值（"回落/恢复"）。
    (b) **主导站固化**（内部状态重构）：后窗"主导站占比" > 前窗占比（吸引子结构在固化）。
    (c) **预测变稳**（σ 下降）：后窗 ε 方差 < 前窗方差。
  三者同拍共现 ⇒ 一条顿悟事件；任一条缺失 ⇒ 不是顿悟（判据**可失败**、不恒真）。
  事件后 **W 拍内抑制**（去抖，结构规则）。

  ⚠ 与专家01 §4.2 的一处**经实测纠正的偏离**（诚实标注）：§4.2.2 建议的"极限环换站相变 =
    换站数从 0 跳到 >0"在**真机文本态不成立**——实测引擎主导站换站率 ≈ 0.68（近随机，
    见 `_诊断_顿悟.py`），不存在"锁死→跳增"。故本实现把该通道**落地为"主导站占比上升"**
    （可测的状态重构相变），而非沿用不成立的"锁死→跳增"。理论① 的 T₂ itinerancy 属思核
    `z` 轨迹（步3，思考态），本步不重复其测法。

  不破既有（C14）：唯一覆盖点 = 综合体.假设环一拍（**只换顿悟分支**）与 _引擎步（**只加挂**观测）；
    开关默认关时 假设环一拍 直接走 super()（逐位恒等），_引擎步 只多一次守卫（不改状态/决策）。
    顿悟账本 = **独立学习存储**（≠ codec._段；记忆≠学习，所有者硬命题）。本文件不 import 任何器官。

诚实边界（不宣称超出本步）：
  1) F2 只能**否证**"顿悟=单点阈值"（若事件分布与阈值触发分布同构即判程序），**不能**正面证明
     "真涌现"——这是**可证伪的不对称性**，如实登记。
  2) "λ≈0 临界"只作**关联登记**：事件拍附近相空间有限时间 λ 与临界带（|λ|<0.05）的接近程度；
     **不声称**因果（对齐理论① L4 的否证性质）。
  3) 本步只替换**顿悟判据**；`_思考一拍` 的常数 0.45/0.6/128 的动力学化（专家01 §4.3）**留待后续**，
     本文件不动它（避免越界、避免破 V2–V5）。

纪律：纯 numpy、零梯度、确定性；只在测试中显式加载内核。
用法（测试）：见 步6_顿悟涌现自证.py。
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 步5 宿主（含步2/3/4）；本步在其之上再加挂顿悟涌现（默认关）
from 内核_自我回路 import 顶覆盖_综合体Z                                    # noqa: E402
from 内核_相空间算子T import 思核, 开关默认 as _步3开关默认                   # noqa: E402

开关默认 = False            # ★保护开关默认关（C14/D1：默认关 ⇒ 既有输出逐位恒等）
窗默认 = 12                 # 结构尺度：滑窗长度（前窗/后窗）+ 事件去抖窗；非判据阈值
λ临界容差 = 0.05            # 对齐理论① L4：|λ| < 0.05 记"临界"（仅关联登记，非本内核判据）


def _单位(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(x))
    return x / n if n > 1e-12 else x


def _主导占比(站段) -> float:
    """窗口内"主导站"占比 = 众数站出现次数 / 窗长（∈[0,1]，与维数/标度无关）。"""
    段 = np.asarray(站段)
    if 段.size == 0:
        return 0.0
    值s, 计s = np.unique(段, return_counts=True)
    return float(计s.max()) / float(段.size)


class 顿悟核:
    """相变式顿悟检测器（零预注册阈值·窗口局部自归一·确定性·纯 numpy）。

    输入三条**自身**流（谁在场就记谁）：
      · ε史 —— 自模型失配 ε_self（步2/步4 产出；判据用它）；
      · 站史 —— 引擎自态在母语空间方向表上的**主导站**（判据用它）；
      · 命史 —— 假设环的预测命中 cos（**仅登记**，恒在，不参与判据以免单调漂移误判）。
    输出：同拍共现 (a)(b)(c) ⇒ 一条顿悟事件，进**独立顿悟账本**（≠ codec._段）。
    """

    def __init__(self, 方向表: dict = None, 窗: int = 窗默认):
        self.窗 = int(窗)
        self.ε史: list = []
        self.站史: list = []
        self.命史: list = []
        self.顿悟账本: list = []        # ★独立学习存储（≠ codec._段）
        self.事件拍s: list = []
        self._上事件拍 = -10 ** 9
        self._核 = None
        if 方向表 is not None:
            self._核 = 思核(方向表)      # 只借用其 差模竞争表 求"主导站"（不跑 T 迭代）

    # ── 主导站：引擎自态 H双极 在方向表（差模）上的 argmax cos（对齐 思核.站点 口径） ──
    def 站点(self, z):
        if self._核 is None:
            return None
        return int(list(self._核.竞争表).index(self._核.站点(_单位(z))))

    # ── 记一拍观测（判=是否本拍做检测；假设环路径只喂"命中"通道时置 判=False） ──
    def 观测(self, 拍: int, ε=None, 站=None, 命=None, z=None, 判: bool = True):
        if ε is not None:
            self.ε史.append(float(ε))
        if 站 is not None:
            self.站史.append(int(站))
        if 命 is not None:
            self.命史.append(float(命))
        if not 判:
            return None
        ev = self._判(拍, z)
        if (ev is not None) and (int(拍) - self._上事件拍 < self.窗):     # 去抖
            ev = None
        if ev is not None:
            self._上事件拍 = int(拍)
            self.顿悟账本.append(ev)
            self.事件拍s.append(int(拍))
        return ev

    # ── 相变共现判定（全部为窗口局部自归一比较；无预注册常量阈值） ──
    def _判(self, 拍: int, z=None):
        W = self.窗
        if len(self.ε史) < 2 * W or len(self.站史) < 2 * W:
            return None
        前e = np.asarray(self.ε史[-2 * W:-W], dtype=np.float64)
        后e = np.asarray(self.ε史[-W:], dtype=np.float64)
        前s = self.站史[-2 * W:-W]
        后s = self.站史[-W:]
        极差前 = float(前e.max() - 前e.min())
        极差后 = float(后e.max() - 后e.min())
        均前, 均后 = float(前e.mean()), float(后e.mean())
        方前, 方后 = float(前e.var()), float(后e.var())
        占比前, 占比后 = _主导占比(前s), _主导占比(后s)

        a = (极差前 > 极差后) and (均后 < 均前)                 # (a) 失配峰→回落
        b = 占比后 > 占比前                                    # (b) 主导站固化
        c = 方后 < 方前                                        # (c) 预测变稳
        if a and b and c:
            return dict(拍=int(拍),
                        ε突降=True, 主导站固化=True, 方差降=True,
                        极差前=round(极差前, 5), 极差后=round(极差后, 5),
                        均前=round(均前, 5), 均后=round(均后, 5),
                        方前=round(方前, 6), 方后=round(方后, 6),
                        占比前=round(占比前, 4), 占比后=round(占比后, 4),
                        z=(None if z is None else np.asarray(z, dtype=np.float32)),
                        命=round(float(self.命史[-1]), 4) if self.命史 else None)
        return None

    def 摘要(self) -> dict:
        return dict(ε史条数=len(self.ε史), 站史条数=len(self.站史), 命史条数=len(self.命史),
                    顿悟条数=len(self.顿悟账本), 事件拍s=list(self.事件拍s), 窗=self.窗)


# ────────────────────────── synthesis 顶覆盖（只加挂；基链一行不改） ──────────────────────────

class 顶覆盖_综合体W(顶覆盖_综合体Z):
    """步6 宿主：在步5 宿主之上**再**加挂顿悟涌现（默认关·只换顿悟分支）。

    · 顿悟开关=False（默认）：假设环一拍 直接走 super()（逐位恒等，D1）；_引擎步 只多守卫。
    · 顿悟开关=True：假设环一拍 复刻基类逻辑，但把 `if cosv>=0.8 and 试次>=1` 的顿悟分支
      **替换**为"相变共现"事件；同时 _引擎步 每拍只读喂 ε_self 与主导站，共现则经既有
      `_自言`（类型∈六类、≥1 账本引用）留痕。**引擎状态推进行一行不改**。
    """

    def __init__(self, seed: int = 310, 私偏κ: float = 10.0, 顿悟开关=None, 顿悟参: dict = None,
                 自我开关=None, 档案seed=None, 学习开关=None, 思核开关=None, 自模型开关=None,
                 学习: bool = True, 状态投影=None, 思核参: dict = None, 自我提问开关=None):
        # 先置属性——super().__init__ 期间 _引擎步 可能已被调用（synthesis.py:89 注释）；
        # 那时 顿悟开关 已就位但 顿悟核=None ⇒ 自然跳过（不改任何状态）。
        self.顿悟开关 = bool(开关默认 if 顿悟开关 is None else 顿悟开关)
        # ★自驱思考（AHI3 接线新增）：默认关（同 开关默认）⇒ 关时逐位恒等（D1/C14）
        self.自我提问开关 = bool(开关默认 if 自我提问开关 is None else 自我提问开关)
        self.顿悟核 = None
        self._顿悟参 = dict(顿悟参) if 顿悟参 else None
        super().__init__(seed=seed, 私偏κ=私偏κ, 自我开关=自我开关, 档案seed=档案seed,
                         学习开关=学习开关, 思核开关=思核开关, 自模型开关=自模型开关,
                         学习=学习, 状态投影=状态投影, 思核参=思核参)
        if self.顿悟开关:
            self._起顿悟核()

    # ── 方向表：引擎动作自由度 → 母语空间方向（与步3 思核 同源；用于"主导站"） ──
    def _起顿悟核(self) -> None:
        if self.顿悟核 is not None:
            return
        Φ = np.asarray(self.eng.hdc.Φ_act, dtype=np.float64)
        方向表 = {"动%d" % j: _单位(Φ[:, j]) for j in range(int(Φ.shape[1]))}
        参 = dict(窗=窗默认)
        if self._顿悟参:
            参.update(self._顿悟参)
        self.顿悟核 = 顿悟核(方向表, 窗=int(参["窗"]))

    # ── 本拍主导站 + 本拍自态 z：引擎自态 H双极 在方向表（差模）上的 argmax cos ──
    def _本拍站(self, rec):
        """返回 (站, z)：站=主导站号（判据 (b) 用）；z=本拍自态单位向量（事件存各自 z 供 F2c λ）。"""
        h = (rec.get("模块") or {}).get("hdc") or {}
        S = h.get("H双极")
        if S is None:
            H = getattr(self.eng.hdc, "H", None)
            if H is None:
                return None, None
            S = np.sign(np.asarray(H, dtype=np.float64))
        S = np.asarray(S, dtype=np.float64)
        return self.顿悟核.站点(S), _单位(S)

    # ── 假设环一拍：只换顿悟分支（关时逐位恒等） ──
    def 假设环一拍(self, 世界):
        if not self.顿悟开关 or self.顿悟核 is None:
            return super().假设环一拍(世界)
        # —— 开：复刻基类逻辑；`if cosv>=0.8` 分支 → 相变事件 ——
        n = self.动作数
        未测 = [i for i in range(n) if i not in self.试次]
        if 未测:
            i = 未测[0]
        else:
            i = max(range(n), key=lambda j: (self.上误差.get(j, 0), -self.试次[j]))
        p = self._预测(i)
        if p is None:
            假设文本 = "动作%d：还没试过，没有可预测的方向。" % i
        else:
            序 = np.argsort(-np.abs(p))[:3]
            假设文本 = "动作%d：我预测 %s。" % (
                i, "、".join("第%d维往%s" % (d + 1, "正" if p[d] > 0 else "负")
                             for d in 序))
        self.账本.append(("假设", 假设文本))
        假设号 = len(self.账本) - 1
        实际 = 世界.行动(i)
        实向 = 实际 / (np.linalg.norm(实际) + 1e-9)
        cosv = None
        if p is not None:
            cosv = float(p @ 实向)
            self.上误差[i] = 1.0 - cosv
            # ★ 顿悟不再用 `if cosv>=0.8`；命中只作登记通道，判据由相变核自身给出
            self.顿悟核.观测(self.拍, 命=cosv, 判=False)
        self.束[i] = self.束.get(i, np.zeros(n)) + np.sign(实际)
        self.试次[i] = self.试次.get(i, 0) + 1
        检验文本 = "动作%d cos=%s" % (i, "未测" if cosv is None else round(cosv, 3))
        self.账本.append(("检验", 检验文本))
        检验号 = len(self.账本) - 1
        rec = self._引擎步(self.感知向量(self.codec.编码("检验 " + 检验文本)))
        self._吃(rec)
        return dict(动作=i, 预测cos=cosv, 假设号=假设号, 检验号=检验号,
                    预测文本=假设文本)

    # ── 每拍只加挂：喂 ε_self 与主导站 → 相变共现则留痕 ──
    def _引擎步(self, u):
        rec = super()._引擎步(u)                  # 基链状态推进 + 步2/3/4/5 旁挂，逐行不变
        # ★AHI3 接线补丁（自驱思考，**默认关**）：自主拍里没有外部提问，思考态永远不会被建立，
        #   于是相空间算子 T 完全没有轨迹。开启后让**它自己**决定停下来想：
        #   门限取 ε_self 相对**自身慢基线**（自归一，零人挑阈值）；
        #   问题文本取它自己最近的自言（复盘/自评）——它想的是**自己刚才在干的事**。
        #   ⚠ 保护开关默认 False：关时本块不执行 ⇒ `_引擎步` 与基类**逐位恒等**（C14/D1）。
        #     （实测教训：一度写成默认 True，D1「关时逐位恒等」立刻 258 条差异 → 回退为默认关。）
        try:
            if getattr(self, "自我提问开关", 开关默认):
                self._自发提问()
        except Exception:
            pass
        if self.顿悟开关 and self.顿悟核 is not None:
            try:
                ε = self._本拍ε()
                站, z = self._本拍站(rec)
                ev = self.顿悟核.观测(int(self.拍), ε=ε, 站=站, z=z)
                if ev is not None and len(self.账本) > 0:
                    文本 = ("顿悟涌现[第%d拍]：ε_self 前窗极差 %.3f→后窗 %.3f、均值 %.3f→%.3f"
                            "（失配峰回落），主导站占比 %.2f→%.2f（固化），方差 %.4f→%.4f"
                            "（预测变稳）——不是阈值越线，是相变共现。" %
                            (ev["拍"], ev["极差前"], ev["极差后"], ev["均前"], ev["均后"],
                             ev["占比前"], ev["占比后"], ev["方前"], ev["方后"]))
                    self._自言("复盘", 文本, ("内生", "顿悟相变"), [len(self.账本) - 1])
            except Exception:
                pass                              # 旁挂失败绝不打断既有路径
        return rec

    # ── 自驱提问：ε_self 高于自身慢基线 ⇒ 它自己开一个「想」 ──
    def _自发提问(self) -> None:
        if getattr(self, "思考中", None) is not None:
            return                                   # 已在想 → 不打断
        drv = getattr(self, "drv", None)
        ε史 = list(getattr(drv, "ε史", []) or [])
        if len(ε史) < 3:
            return
        ε = float(ε史[-1])
        慢 = float(getattr(drv, "ε_slow", 0.0) or 0.0)
        if ε <= 慢:                                  # 相对自身基线不异常 ⇒ 不必想
            return
        try:
            self._延答(self._自问素材())              # 既有思考入口（疑因已由 ε_self 自归一判定）
        except Exception:
            pass

    def _自问素材(self, 上限: int = 60) -> str:
        """它想问什么：优先取自己最近一条自言（复盘/自评/疑问/预测），否则用自指式问句。"""
        try:
            流 = list(getattr(getattr(self, "自言", None), "流", []) or [])
        except Exception:
            流 = []
        for x in reversed(流[-8:]):
            本 = x if isinstance(x, dict) else dict(类型=getattr(x, "类型", ""),
                                                  文本=getattr(x, "文本", ""))
            if str(本.get("类型")) in ("复盘", "自评", "疑问", "预测"):
                t = str(本.get("文本", "")).strip()
                if len(t) >= 6:
                    return t[:上限]
        return "我刚才那一下，自己到底是怎么变的"