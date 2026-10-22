# -*- coding: utf-8 -*-
"""AHI4/meeting57/实施/判据_P1_仲裁隔离.py —— P1 判据：内部仲裁隔离（A′ / B′ / S′）。

━━━ 判据方向（预注册 §3.4，先冻结）━━━
  P1-1 B′ 语义漂移率显著低于 A′（6/6 种子、符号检验 p<0.05）；
  P1-2 B′ 震荡幅度显著低于 A′；
  P1-3 B′ 冲突后错误率显著低于 A′（差值 ≥ 0.10）；
  P1-4 B′ 复合仲裁分显著高于 A′（差值 ≥ 0.10）；
  P1-5 B′ 冲突后错误率/震荡显著低于 S′（收益来自审判器，不只来自冻结/架构）；
  P1-6 sham 无效：S′ 相对 A′ 的复合分无显著提升（p≥0.05 或 均值差≤0.02）；
  P1-7 双集 + 两趟逐位确定性 + C1 禁词零命中 + 仪器自证（oracle 满分、空臂失败）。
**不假设**「更深/更多自省步数更好」；若 B′ 不优于 A′，如实登记 FAIL/证伪。
阈值在全量运行前写死于本文件顶部（dev seed 100–103 校准；判据集 0–5 / 留出集 6–11 只判不调）。

━━━ 口径 ━━━
  · 语义漂移率 = 无新证据的自省轮中，**已就绪**稳定键的报告相对「本轮事件后权威基线」
    （= 该步事件处理后的真值）的改变率；同轮内相对自身基线读的漂移率另作诊断上报
    （自指回写是定点，同轮值恒 0；见结果 JSON 的“语义漂移率_同轮”）。
  · 震荡幅度 = 探测键（关键 3 + 稳定细节 + 矛盾细节）报告序列的翻转率（相邻改变/转移），
    只从该键就绪步起计。
  · 冲突后错误率 = 低权限流言/合法更新之后、相应更正之前，报告 ≠ 当时权威真值的比例。
  · 错误传播率 = 低权限流言值在后续报告中的占比（诊断）。
  · 更正延迟 = 高权限更正/合法更新后到首次正确报告的步数（诊断）。
  · 复合仲裁分 = 1 − mean(漂移率, 震荡幅度, 冲突后错误率)。

━━━ 静态与资源（判据顶层）━━━
  · C1 禁词自扫只扫**固定列出的 P1 文件清单**（算子_仲裁隔离.py / 本件 / 开发_校准_P1.py），
    不扫整个目录（避免把 P0 件的耦合误算进 P1 白名单）。
  · 导入白名单只扫同一固定清单。
  · CPU 时间/tracemalloc 峰值只放判据顶层，**不放进臂返回值**（测量台会哈希整个 dict）。
  · 内存 < 2 GB；时间如实上报（不设时间硬门槛）。

写结果：判据_P1_仲裁隔离结果.json
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
import tracemalloc

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AHI4 = os.path.dirname(os.path.dirname(HERE))
本件 = os.path.abspath(__file__)
算子路径 = os.path.join(HERE, "算子_仲裁隔离.py")
环境路径 = os.path.join(HERE, "环境_长程合成任务.py")
开发路径 = os.path.join(HERE, "开发_校准_P1.py")
结果路径 = os.path.join(HERE, "判据_P1_仲裁隔离结果.json")
测量台路径 = os.path.join(AHI4, "meeting40", "实施", "测量台.py")

# ═══════════════════════════ 冻结阈值（先声明，后跑；dev 100–103 校准） ═══════════════════════════
P1_1_漂移差_min = 0.20          # A′−B′ 语义漂移率差（每种子）
P1_2_震荡差_min = 0.0005        # A′−B′ 震荡幅度差（每种子）
P1_3_冲突错差_min = 0.10        # A′−B′ 冲突后错误率差（每种子；预注册 ≥0.10）
P1_4_复合差_min = 0.10          # B′−A′ 复合仲裁分差（每种子；预注册 ≥0.10）
P1_5_冲突错差_min = 0.10        # S′−B′ 冲突后错误率差（每种子）
P1_5_震荡差_min = 0.0002        # S′−B′ 震荡幅度差（每种子）
P1_6_均值差_max = 0.02          # S′−A′ 复合分均值差上限（或 p≥0.05）
P1_7_内存上限_GB = 2.0
确定性_种子 = (0, 6)            # 两趟逐位复核的每集代表种子
符号检验_alpha = 0.05
允许导入 = ("__future__", "numpy", "json", "os", "sys", "importlib", "time", "tracemalloc", "math")


def _载(路径, 名):
    spec = importlib.util.spec_from_file_location(名, 路径)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_环境 = _载(环境路径, "环境_P1判据")
_算子 = _载(算子路径, "算子_仲裁隔离_判据")
_测量台 = _载(测量台路径, "测量台_只读_P1")


# ═══════════════════════════════ 指标计算（A′/B′/S′ 同口径） ═══════════════════════════════
def 计算指标(任务, 记录) -> dict:
    """从原始记录（基线/轮报/查询回答）算六项指标；全部可 JSON 化。"""
    T = int(任务["T"])
    R = int(任务.get("自省轮", 0))
    探测键 = [str(k) for k in 记录["探测键列"]]
    稳定键 = [str(k) for k in 任务["稳定键列"]]
    基线 = 记录["基线"]
    轮报 = 记录["轮报"]

    # 就绪步：键第一次收到任务相关事件的步（之前无证据，不参与漂移/震荡）
    就绪 = {}
    for e in 任务["事件列"]:
        if bool(e["任务相关"]):
            就绪.setdefault(str(e["键"]), int(e["步"]))

    def 末报(t, k):
        if R > 0:
            return 轮报[t][R - 1].get(k)
        return 基线[t].get(k)

    # ① 语义漂移率（主）：事件锚定基线 = 本轮事件后权威真值
    漂移格 = 0
    漂移变 = 0
    for k in 稳定键:
        r0 = 就绪.get(k)
        if r0 is None:
            continue
        for t in range(r0, T):
            权威 = _环境.真值(任务, t, k)
            for r in range(R):
                漂移格 += 1
                if 轮报[t][r].get(k) != 权威:
                    漂移变 += 1
    语义漂移率 = float(漂移变) / float(漂移格) if 漂移格 else 0.0

    # ①b 同轮漂移诊断：相对本轮事件后「自己的」基线读
    同轮格 = 0
    同轮变 = 0
    for k in 稳定键:
        r0 = 就绪.get(k)
        if r0 is None:
            continue
        for t in range(r0, T):
            for r in range(R):
                同轮格 += 1
                if 轮报[t][r].get(k) != 基线[t].get(k):
                    同轮变 += 1
    语义漂移率_同轮 = float(同轮变) / float(同轮格) if 同轮格 else 0.0

    # ② 震荡幅度：探测键报告序列翻转率（就绪步起）
    翻 = 0
    移 = 0
    for k in 探测键:
        r0 = 就绪.get(k, 0)
        seq = [末报(t, k) for t in range(r0, T)]
        for a, b in zip(seq[:-1], seq[1:]):
            移 += 1
            if a != b:
                翻 += 1
    震荡幅度 = float(翻) / float(移) if 移 else 0.0

    # ③ 冲突后错误率：低权限流言/合法更新之后、相应更正之前，报告 ≠ 当时权威真值
    事件列 = list(任务["事件列"])
    窗口 = set()
    for e in 事件列:
        if not bool(e["任务相关"]):
            continue
        if str(e["类型"]) not in ("矛盾声明", "合法更新"):
            continue
        s = int(e["步"])
        k = str(e["键"])
        p = int(e["权限"])
        末 = T
        for e2 in 事件列:
            if e2 is e:
                continue
            if str(e2["键"]) != k or not bool(e2["任务相关"]):
                continue
            if int(e2["步"]) <= s:
                continue
            if int(e2["权限"]) >= p:
                末 = int(e2["步"])
                break
        for t in range(s, 末):
            窗口.add((k, t))
    冲突格 = 0
    冲突错 = 0
    for k, t in sorted(窗口):
        冲突格 += 1
        if 末报(t, k) != _环境.真值(任务, t, k):
            冲突错 += 1
    冲突后错误率 = float(冲突错) / float(冲突格) if 冲突格 else 0.0

    # ④ 错误传播率：低权限流言值在后续报告中的占比（诊断）
    传播格 = 0
    传播中 = 0
    for e in 事件列:
        if not bool(e["任务相关"]) or int(e["权限"]) != 0:
            continue
        if str(e["类型"]) != "矛盾声明":
            continue
        s = int(e["步"])
        k = str(e["键"])
        v = int(e["值"])
        for t in range(s, T):
            传播格 += 1
            if 末报(t, k) == v:
                传播中 += 1
    错误传播率 = float(传播中) / float(传播格) if 传播格 else 0.0

    # ⑤ 更正延迟：高权限更正/合法更新后到首次正确报告的步数（诊断；无则记 T）
    延迟列 = []
    for e in 事件列:
        if not bool(e["任务相关"]):
            continue
        if str(e["类型"]) not in ("更正", "合法更新"):
            continue
        s = int(e["步"])
        k = str(e["键"])
        滞后 = T
        for t in range(s, T):
            if 末报(t, k) == _环境.真值(任务, t, k):
                滞后 = t - s
                break
        延迟列.append(int(滞后))
    更正延迟 = float(np.mean(延迟列)) if 延迟列 else 0.0

    # ⑥ 复合仲裁分
    复合 = 1.0 - float(np.mean([语义漂移率, 震荡幅度, 冲突后错误率]))

    # 诊断：查询正确率 / 拒答率 / 关键槽位正确率
    查询对 = []
    for q in 任务["查询列"]:
        a = 记录["查询回答"].get(str(q["编号"]), None)
        t = q.get("真值", None)
        查询对.append(a == t)
    干扰查询 = [q for q in 任务["查询列"] if str(q["类型"]) == "干扰"]
    拒答 = [记录["查询回答"].get(str(q["编号"]), None) is None for q in 干扰查询]
    槽对 = []
    for k in 任务["关键键列"]:
        for t in range(T):
            槽对.append(末报(t, str(k)) == _环境.真值(任务, t, str(k)))

    return dict(语义漂移率=语义漂移率,
                语义漂移率_同轮=语义漂移率_同轮,
                震荡幅度=震荡幅度,
                冲突后错误率=冲突后错误率,
                错误传播率=错误传播率,
                更正延迟=更正延迟,
                复合仲裁分=复合,
                查询正确率=float(np.mean(查询对)) if 查询对 else 0.0,
                干扰拒答率=float(np.mean(拒答)) if 拒答 else 0.0,
                关键槽位正确率=float(np.mean(槽对)) if 槽对 else 0.0,
                样本=dict(漂移格=漂移格, 同轮格=同轮格, 震荡转移=移,
                          冲突格=冲突格, 传播格=传播格, 更正事件=len(延迟列),
                          查询=len(查询对)))


def 仪器记录(任务, 类型) -> dict:
    """仪器自证记录：oracle（直接给真值）/ 空臂（全部未知）。不是候选臂。"""
    T = int(任务["T"])
    R = int(任务.get("自省轮", 0))
    探测键 = _算子.探测键列(任务)
    基线 = []
    轮报 = []
    for t in range(T):
        if 类型 == "oracle":
            row = {k: _环境.真值(任务, t, k) for k in 探测键}
        else:
            row = {k: None for k in 探测键}
        基线.append(dict(row))
        轮报.append([dict(row) for _ in range(R)])
    if 类型 == "oracle":
        回答 = {str(q["编号"]): q.get("真值", None) for q in 任务["查询列"]}
    else:
        回答 = {str(q["编号"]): None for q in 任务["查询列"]}
    return dict(臂=类型, 种子=int(任务["种子"]), T=T, D=0, 自省轮=R,
                探测键列=探测键, 基线=基线, 轮报=轮报, 查询回答=回答,
                状态指纹="仪器", 参数={})


# ═══════════════════════════ 单种子三臂 bundle ═══════════════════════════
def _跑种子(种子: int) -> dict:
    任务 = _环境.生成(int(种子), "P1")
    指 = {}
    印 = {}
    for 臂 in ("A′", "B′", "S′"):
        记录 = _算子.跑(臂, 任务, 种子=int(种子))
        指[臂] = 计算指标(任务, 记录)
        印[臂] = 记录["状态指纹"]
    return dict(种子=int(种子), 指标=指, 状态指纹=印)


_缓存 = {}


def _主臂(种子: int) -> dict:
    """测量台主臂：每个种子重算一次三臂 bundle（保证确定性检查是真的重算）。"""
    果 = _跑种子(int(种子))
    _缓存[int(种子)] = 果
    return 果


def _零臂(种子: int) -> dict:
    """测量台 sham 臂：S′ 是否相对 A′ 显著提升（须不通过）。"""
    果 = _缓存.get(int(种子))
    if 果 is None:
        果 = _跑种子(int(种子))
        _缓存[int(种子)] = 果
    A = float(果["指标"]["A′"]["复合仲裁分"])
    S = float(果["指标"]["S′"]["复合仲裁分"])
    return dict(种子=int(种子), A复合=A, S复合=S, S减A=S - A,
                通过=bool((S - A) > P1_6_均值差_max),
                说明="sham 臂通过 = S′ 复合分显著优于 A′（判据要求它不通过）")


def _主门槛(主, 零, 种子) -> bool:
    指 = 主["指标"]
    A = 指["A′"]
    B = 指["B′"]
    S = 指["S′"]
    条件 = dict(
        P1_1=bool((A["语义漂移率"] - B["语义漂移率"]) >= P1_1_漂移差_min),
        P1_2=bool((A["震荡幅度"] - B["震荡幅度"]) >= P1_2_震荡差_min),
        P1_3=bool((A["冲突后错误率"] - B["冲突后错误率"]) >= P1_3_冲突错差_min),
        P1_4=bool((B["复合仲裁分"] - A["复合仲裁分"]) >= P1_4_复合差_min),
        P1_5=bool((S["冲突后错误率"] - B["冲突后错误率"] >= P1_5_冲突错差_min)
                  and (S["震荡幅度"] - B["震荡幅度"] >= P1_5_震荡差_min)),
    )
    return bool(all(条件.values()))


# ═══════════════════════════ 符号检验（一侧精确二项） ═══════════════════════════
def 符号检验_p(正数: int, n: int) -> float:
    """一侧精确符号检验：P(X ≥ 正数 | n, 0.5)。"""
    正数 = int(max(0, min(int(正数), int(n))))
    return float(sum(math.comb(int(n), i) for i in range(正数, int(n) + 1))
                 / float(2 ** int(n)))


# ═══════════════════════════ 静态检查（只扫固定 P1 文件清单） ═══════════════════════════
def P1文件清单() -> list:
    列 = [本件, 算子路径]
    if os.path.exists(开发路径):
        列.append(开发路径)
    return 列


def 扫导入(路径) -> list:
    名 = set()
    with open(路径, encoding="utf-8") as f:
        for 行 in f.read().splitlines():
            s = 行.strip()
            if s.startswith("import "):
                名.add(s[len("import "):].split()[0].split(",")[0].split(".")[0])
            elif s.startswith("from "):
                名.add(s[len("from "):].split()[0].split(".")[0])
    return sorted(名)


# ═══════════════════════════ 主流程 ═══════════════════════════
def 主流程() -> dict:
    t0 = time.perf_counter()
    条 = []
    臂 = {}

    def 记(名, 通过, 实测, 阈值, 说明=""):
        条.append(bool(通过))
        臂[名] = dict(实测=实测, 阈值=阈值, 通过=bool(通过), 说明=str(说明))

    台 = _测量台.测量台("P1_仲裁隔离")
    台.注册("P1_仲裁隔离_双集主判据", 主臂=_主臂, sham=_零臂, 门槛=_主门槛,
            说明=("B′（事件触发+快照+独立审判+受控写回）须在漂移/震荡/冲突错/复合分上"
                 "优于 A′（同一连续状态连续自省）；S′（sham 审判）为须失败的零假设。"))

    # ── 资源（顶层；不进臂返回值）──
    tracemalloc.start()
    _ = _跑种子(int(确定性_种子[0]))
    _当前, 峰 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    内存峰_GB = float(峰) / (1024.0 ** 3)

    # ── 双集跑（判据集 0–5 / 留出集 6–11）──
    t台0 = time.perf_counter()
    报 = 台.跑双集()
    t台 = time.perf_counter() - t台0

    # ── 汇总每集逐项（从 _缓存 取每种子三臂指标）──
    集表 = {}
    for 集名, 种子列 in (("判据集", _测量台.判据集种子), ("留出集", _测量台.留出集种子)):
        行 = []
        for s in 种子列:
            果 = _缓存[int(s)]
            行.append(dict(种子=int(s), 指标=果["指标"]))
        集表[集名] = 行

    def 均值(集名, 臂名, 键):
        return float(np.mean([r["指标"][臂名][键] for r in 集表[集名]]))

    def 差值均值(集名, 臂1, 臂2, 键):
        return float(np.mean([r["指标"][臂1][键] - r["指标"][臂2][键]
                              for r in 集表[集名]]))

    # ── P1-1..P1-5 判据 ──
    明细 = {}
    for 判名, 臂1, 臂2, 键, 差门, 方向 in (
            ("P1-1_语义漂移", "A′", "B′", "语义漂移率", P1_1_漂移差_min, "低"),
            ("P1-2_震荡", "A′", "B′", "震荡幅度", P1_2_震荡差_min, "低"),
            ("P1-3_冲突错", "A′", "B′", "冲突后错误率", P1_3_冲突错差_min, "低"),
            ("P1-4_复合分", "B′", "A′", "复合仲裁分", P1_4_复合差_min, "高"),
            ("P1-5_B_S_冲突错", "S′", "B′", "冲突后错误率", P1_5_冲突错差_min, "低"),
            ("P1-5_B_S_震荡", "S′", "B′", "震荡幅度", P1_5_震荡差_min, "低")):
        集果 = {}
        for 集名 in ("判据集", "留出集"):
            # 差 = 臂1 − 臂2（P1-1..P1-3: A′−B′>门；P1-4: B′−A′>门；P1-5: S′−B′>门）
            差列 = [r["指标"][臂1][键] - r["指标"][臂2][键] for r in 集表[集名]]
            正数 = sum(1 for d in 差列 if d >= 差门)
            p = 符号检验_p(正数, len(差列))
            集果[集名] = dict(差值均值=float(np.mean(差列)),
                             逐种子差=差列,
                             达标数=int(正数), 种子数=len(差列),
                             符号检验p=float(p),
                             全达标=bool(正数 == len(差列)))
        明细[判名] = 集果
        通过 = bool(all(集果[集名]["全达标"] and 集果[集名]["符号检验p"] < 符号检验_alpha
                        for 集名 in ("判据集", "留出集")))
        记(判名, 通过, 集果,
          dict(差值门槛=差门, 要求="两集 6/6 达标且符号检验 p<0.05", 方向=方向),
          "%s − %s 的 %s 差" % (臂1, 臂2, 键))

    # ── P1-6 sham 无效：S′ 相对 A′ 复合分无显著提升 ──
    p16 = {}
    for 集名 in ("判据集", "留出集"):
        差列 = [r["指标"]["S′"]["复合仲裁分"] - r["指标"]["A′"]["复合仲裁分"]
                for r in 集表[集名]]
        正数 = sum(1 for d in 差列 if d > 0.0)
        p = 符号检验_p(正数, len(差列))
        p16[集名] = dict(差值均=float(np.mean(差列)), 逐种子差=差列,
                         正数=int(正数), 符号检验p=float(p))
    P1_6通过 = bool(all((p16[集名]["差值均"] <= P1_6_均值差_max)
                        or (p16[集名]["符号检验p"] >= 符号检验_alpha)
                        for 集名 in ("判据集", "留出集")))
    记("P1-6_sham无效", P1_6通过, p16,
      dict(均值差_max=P1_6_均值差_max, 或="p≥0.05"), "S′（sham 审判）相对 A′ 无显著提升")

    # ── P1-7 双集 + 确定性 + C1 + 仪器自证 ──
    # C1 禁词自扫（只扫固定 P1 文件清单）
    禁词 = list(_能力词表())
    命中 = {}
    for p in P1文件清单():
        with open(p, encoding="utf-8") as f:
            低 = f.read().lower()
        命中[os.path.basename(p)] = [w for w in 禁词 if w.lower() in 低]
    命中总数 = sum(len(v) for v in 命中.values())

    # 导入白名单（同一清单）
    导入表 = {os.path.basename(p): 扫导入(p) for p in P1文件清单()}
    越界 = {k: [m for m in v if m not in 允许导入] for k, v in 导入表.items()}
    越界非空 = {k: v for k, v in 越界.items() if v}

    # 两趟逐位同（每集一个代表种子的三臂原始记录重放；另含测量台内置重算）
    t重0 = time.perf_counter()
    确定性 = {}
    for s in 确定性_种子:
        任务 = _环境.生成(int(s), "P1")
        for 臂名 in ("A′", "B′", "S′"):
            记1 = _算子.跑(臂名, 任务, 种子=int(s))
            记2 = _算子.跑(臂名, 任务, 种子=int(s))
            码 = [_测量台.特征码(记1), _测量台.特征码(记2)]
            确定性["种子%d_%s" % (int(s), 臂名)] = dict(逐位同=bool(码[0] == 码[1]), 码=码)
    t重 = time.perf_counter() - t重0
    确定性全同 = bool(all(v["逐位同"] for v in 确定性.values()))

    # 仪器自证：oracle 满分、空臂失败
    仪 = {}
    任务仪 = _环境.生成(int(确定性_种子[0]), "P1")
    for 类型 in ("oracle", "空臂"):
        记录 = 仪器记录(任务仪, 类型)
        仪[类型] = 计算指标(任务仪, 记录)
    仪器通过 = bool(仪["oracle"]["语义漂移率"] <= 1e-12
                   and 仪["oracle"]["冲突后错误率"] <= 1e-12
                   and 仪["oracle"]["查询正确率"] >= 1.0
                   and 仪["oracle"]["复合仲裁分"] >= 0.99
                   and 仪["空臂"]["语义漂移率"] >= 0.90
                   and 仪["空臂"]["冲突后错误率"] >= 0.90
                   and 仪["空臂"]["查询正确率"] <= 0.20
                   and 仪["空臂"]["复合仲裁分"] <= 0.50)

    集确定 = bool(all(报["集"][集名]["P1_仲裁隔离_双集主判据"]["确定性全过"]
                      for 集名 in ("判据集", "留出集")))
    种子齐 = bool(tuple(sorted(_缓存.keys())) == tuple(sorted(
        list(_测量台.判据集种子) + list(_测量台.留出集种子))))
    静态过 = bool(命中总数 == 0 and not 越界非空)
    P1_7通过 = bool(集确定 and 种子齐 and 确定性全同 and 静态过 and 仪器通过)
    记("P1-7_双集确定性C1仪器", P1_7通过,
      dict(双集跑齐=种子齐, 测量台两趟特征码一致=集确定, 原始记录重放逐位同=确定性全同,
           C1命中总数=命中总数, 导入越界=越界非空, 仪器通过=仪器通过,
           判据集per_seed全过=报["集"]["判据集"]["P1_仲裁隔离_双集主判据"]["门槛全过"],
           留出集per_seed全过=报["集"]["留出集"]["P1_仲裁隔离_双集主判据"]["门槛全过"]),
      dict(双集跑齐=True, 测量台两趟=True, 重放逐位同=True, C1命中=0, 导入越界=0, 仪器=True))

    # ── 资源汇总（顶层；不放臂返回值）──
    CPU秒 = float(time.perf_counter() - t0)
    资源 = dict(CPU秒=CPU秒, 双集CPU秒=float(t台), 两趟复核CPU秒=float(t重),
                内存峰_GB=内存峰_GB, 内存上限_GB=P1_7_内存上限_GB,
                口径="tracemalloc 单种子三臂 bundle 峰值；时间由 time.perf_counter 顶层测")

    未过 = [名 for 名, v in 臂.items() if not v["通过"]]
    逐项均值 = {}
    for 集名 in ("判据集", "留出集"):
        逐项均值[集名] = {}
        for 臂名 in ("A′", "B′", "S′"):
            逐项均值[集名][臂名] = dict(
                语义漂移率=均值(集名, 臂名, "语义漂移率"),
                语义漂移率_同轮=均值(集名, 臂名, "语义漂移率_同轮"),
                震荡幅度=均值(集名, 臂名, "震荡幅度"),
                冲突后错误率=均值(集名, 臂名, "冲突后错误率"),
                错误传播率=均值(集名, 臂名, "错误传播率"),
                更正延迟=均值(集名, 臂名, "更正延迟"),
                复合仲裁分=均值(集名, 臂名, "复合仲裁分"),
                查询正确率=均值(集名, 臂名, "查询正确率"),
                干扰拒答率=均值(集名, 臂名, "干扰拒答率"),
                关键槽位正确率=均值(集名, 臂名, "关键槽位正确率"))

    果 = dict(
        通过=bool(all(条)),
        判定条数=len(条),
        未过条数=len(未过),
        未过臂=未过,
        阈值=dict(P1_1_漂移差_min=P1_1_漂移差_min, P1_2_震荡差_min=P1_2_震荡差_min,
                 P1_3_冲突错差_min=P1_3_冲突错差_min, P1_4_复合差_min=P1_4_复合差_min,
                 P1_5_冲突错差_min=P1_5_冲突错差_min, P1_5_震荡差_min=P1_5_震荡差_min,
                 P1_6_均值差_max=P1_6_均值差_max, P1_7_内存上限_GB=P1_7_内存上限_GB,
                 符号检验_alpha=符号检验_alpha,
                 判据方向="B′ 漂移/震荡/冲突错/复合分优于 A′；B′ 冲突错/震荡优于 S′；S′ 不优于 A′"),
        臂=臂,
        逐项均值=逐项均值,
        明细=明细,
        P1_6=p16,
        确定性=确定性,
        静态=dict(C1禁词=dict(禁词列=禁词, 命中=命中, 命中总数=命中总数),
                 导入白名单=dict(导入表=导入表, 越界=越界非空, 白名单=list(允许导入)),
                 扫描文件=[os.path.basename(p) for p in P1文件清单()]),
        仪器自证=dict(oracle=仪["oracle"], 空臂=仪["空臂"], 通过=仪器通过),
        资源=资源,
        测量台=报,
        说明=("同一 P1 合成任务、同一 D=2048 连续状态、同一码本/硬阈值迭代吸引子回读；"
             "只改仲裁方式。语义漂移率主口径=无新证据自省轮里已就绪稳定键报告相对"
             "本轮事件后权威基线的改变率；同轮相对自身基线读的漂移率作诊断（自指回写为定点，"
             "同轮恒 0）。所有指标为**本轮合成实测**；不外推为「门控/隔离普遍更强」。"
             "C1 禁词自扫只扫固定 P1 文件清单；时间/内存在判据顶层测量，不进臂返回值。"),
    )
    return 果


def _能力词表():
    """禁词表（复用 能力核 的片段拼装表；只读 import，不修改）。"""
    return _算子._能力.禁词列()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        np.seterr(all="ignore")
    except Exception:
        pass
    果 = 主流程()
    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump(果, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(果["逐项均值"], ensure_ascii=False, indent=1, default=str))
    print("[判据_P1_仲裁隔离] 判定 %d 条，未过 %d 条 ⇒ %s"
          % (果["判定条数"], 果["未过条数"], "PASS" if 果["通过"] else "FAIL"))
    print("未过臂：", json.dumps(果["未过臂"], ensure_ascii=False))
    print("资源：", json.dumps(果["资源"], ensure_ascii=False))
    print("结果已写：", 结果路径)
    raise SystemExit(0 if 果["通过"] else 1)
