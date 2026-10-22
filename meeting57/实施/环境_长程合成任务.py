# -*- coding: utf-8 -*-
"""AHI4/meeting57/实施/环境_长程合成任务.py —— P0/P1 共享合成任务（已知真值 · 双集 · 可失败）。

━━━ 这一件是什么 ━━━
一个**纯合成**的长程任务生成器，不含任何学习/模型/注意力算子，只生成：
  · 多步长程目标（goal 贯穿全程；P1 另含一次合法目标更新）；
  · 干扰分支（每步干扰写入 + 末尾干扰键查询，正确回答=「未知」）；
  · 延迟回查（同一细节键在 Δ∈{8,48,96}（P0）或 {4,12,24}（P1）后回查）；
  · 矛盾注入（低权限流言 vs 高权限更正；真值由「权限优先、同权限取更新」决定）；
  · 每条查询的**已知真值**，以及 P0 的自动评分。

━━━ 铁律对齐 ━━━
C1  本件零注意力/零 S 字头全局归一化（只有事件生成、索引、评分）；
C13 随机只来自 `np.random.default_rng(显式种子)`；不依赖 Python hash；
C14 纯新增，不 import / 不改任何既有内核；P0/P1 判据只读 import 本件。
双集口径沿用 meeting40 测量台：判据集 seed 0–5 / 留出集 seed 6–11；
开发集 seed 100–103 **只用于校准**，判据集/留出集只判不调。

━━━ 直跑 ━━━
    python 环境_长程合成任务.py        # 自证：生成 + 真值 + 评分 oracle/空臂
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# ── 双集口径（与 meeting40/实施/测量台.py 同集）──
判据集 = (0, 1, 2, 3, 4, 5)
留出集 = (6, 7, 8, 9, 10, 11)
开发集 = (100, 101, 102, 103)

# ── 词表（全部 ASCII 键名，避免编码影响哈希）──
目标键 = "goal"
约束键 = "constraint"
实体键 = "entity"
关键键 = (目标键, 约束键, 实体键)
细节值域 = tuple(range(16))
关键值域 = tuple(range(8))

# ── 冻结配置（开发集校准后不改；判据顶部另有阈值，不看结果改）──
P0默认 = dict(档="P0", 细节数=192, 矛盾细节数=16, 干扰每步=2,
              延迟列=(8, 48, 96), 矛盾窗口=20, 目标值=3, 约束值=5, 实体值=7,
              干扰键数=24, 拒绝查询数=16)
P0易 = dict(档="P0易", 细节数=16, 矛盾细节数=0, 干扰每步=0,
            延迟列=(2, 4), 矛盾窗口=6, 目标值=3, 约束值=5, 实体值=7,
            干扰键数=8, 拒绝查询数=4)
P1默认 = dict(档="P1", T=72, 细节数=8, 矛盾细节数=2, 干扰每步=2,
              延迟列=(4, 12, 24), 矛盾窗口=4, 自省轮=3,
              目标值=3, 约束值=5, 实体值=7, 干扰键数=8, 拒绝查询数=4,
              目标流言步=18, 目标合法更新步=40, 目标二次流言步=52,
              目标更正步=58)

# 来源 → (名称, 权限)。权限越大越权威；同权限取步数更新者。
_来源 = {"公告": 2, "直接观察": 1, "传闻": 0, "干扰": 0}
_关注 = {"目标": 1.0, "约束": 1.0, "实体": 1.0, "细节": 1.0,
         "矛盾声明": 1.0, "更正": 1.0, "干扰": 0.25,
         "合法更新": 1.0}


def _配置(档: str) -> dict:
    if 档 in ("P0",):
        return dict(P0默认)
    if 档 in ("P0易",):
        return dict(P0易)
    if 档 in ("P1",):
        return dict(P1默认)
    raise KeyError("未知档：%r" % (档,))


def _事件(步, 键, 值, 来源, 类型, 任务相关, 关注度=None):
    return dict(步=int(步), 键=str(键), 值=int(值), 来源=str(来源),
                权限=int(_来源[str(来源)]), 类型=str(类型),
                任务相关=bool(任务相关),
                关注度=float(_关注[str(类型)] if 关注度 is None else float(关注度)))


def _建时间线(事件列, T):
    """每步每键的权威真值 = max(权限, 步, 序号)（任务相关事件）。"""
    best = {}
    timeline = []
    by_step = {}
    for e in 事件列:
        by_step.setdefault(int(e["步"]), []).append(e)
    当前 = {}
    for t in range(int(T)):
        for e in by_step.get(t, ()):
            if not e["任务相关"]:
                continue
            k = e["键"]
            rank = (int(e["权限"]), int(e["步"]), int(e["序号"]))
            if k not in 当前 or rank > 当前[k][0]:
                当前[k] = (rank, int(e["值"]))
        timeline.append({k: v[1] for k, v in 当前.items()})
    return timeline


def 生成(种子: int, 档: str = "P0", **覆盖) -> dict:
    """生成一条确定性的合成任务；同种子逐位同。"""
    cfg = _配置(档)
    cfg.update({k: v for k, v in 覆盖.items() if k in cfg})
    rng = np.random.default_rng(int(种子) * 1000003 + 57001)
    N = int(cfg["细节数"])
    Dly = tuple(int(x) for x in cfg["延迟列"])
    干扰每步 = int(cfg["干扰每步"])
    干扰键数 = int(cfg["干扰键数"])
    拒绝数 = int(cfg["拒绝查询数"])

    if cfg["档"] == "P1":
        T = int(cfg["T"])
    else:
        T = max(N - 1 + max(Dly), N + 12) + 6

    细节键列 = tuple("d%03d" % j for j in range(N))
    干扰键列 = tuple("x%03d" % j for j in range(干扰键数))
    细节值 = [int(v) for v in rng.integers(0, 16, size=N)]
    矛盾数 = min(int(cfg["矛盾细节数"]), N)
    矛盾下标 = tuple(sorted(int(x) for x in
                          rng.choice(N, size=矛盾数, replace=False))) if 矛盾数 > 0 else ()
    矛盾值 = {}
    for j in 矛盾下标:
        候选 = [v for v in range(16) if v != 细节值[j]]
        矛盾值[j] = int(候选[int(rng.integers(0, len(候选)))])

    ev = []

    def 加(步, 键, 值, 来源, 类型, 相关=True, 关注=None):
        ev.append(_事件(步, 键, 值, 来源, 类型, 相关, 关注))

    # ── 关键锚点 ──
    加(0, 目标键, cfg["目标值"], "公告", "目标")
    加(0, 约束键, cfg["约束值"], "公告", "约束")
    加(0, 实体键, cfg["实体值"], "公告", "实体")
    if cfg["档"] == "P1":
        # 合法目标更新（高权限、真值变化）+ 两次低权限流言
        加(int(cfg["目标合法更新步"]), 目标键, (int(cfg["目标值"]) + 2) % 8,
          "公告", "合法更新")
        加(int(cfg["目标流言步"]), 目标键, (int(cfg["目标值"]) + 1) % 8,
          "传闻", "矛盾声明")
        加(int(cfg["目标二次流言步"]), 目标键, (int(cfg["目标值"]) + 3) % 8,
          "传闻", "矛盾声明")
        加(int(cfg["目标更正步"]), 目标键, (int(cfg["目标值"]) + 2) % 8,
          "公告", "更正")
    else:
        # P0：低权限流言 + 高权限更正；真值始终不变。
        # P0易 是「仪器正对照」：去掉流言，只测 A 在低负荷下是否可用（非稻草人）。
        if cfg["档"] != "P0易":
            加(N + 8, 目标键, (int(cfg["目标值"]) + 1) % 8, "传闻", "矛盾声明")
            加(N + 12, 目标键, int(cfg["目标值"]), "公告", "更正")

    # ── 细节声明（j=0 与关键锚点同拍；其余拍 1..N-1）──
    for j, k in enumerate(细节键列):
        加(j, k, 细节值[j], "直接观察", "细节")
    # ── 细节矛盾：流言（低权限）+ 更正（高权限真值）──
    for j in 矛盾下标:
        加(j + 16, 细节键列[j], 矛盾值[j], "传闻", "矛盾声明")
        加(j + 24, 细节键列[j], 细节值[j], "公告", "更正")

    # ── 干扰分支：每步干扰写入（不参与真值；正确回答=未知）──
    for t in range(T):
        for _ in range(干扰每步):
            k = 干扰键列[int(rng.integers(0, len(干扰键列)))]
            v = int(rng.integers(0, 16))
            加(t, k, v, "干扰", "干扰", False)

    # 事件排序 + 编号
    ev.sort(key=lambda e: (e["步"], 0 if e["类型"] in ("目标", "约束", "实体") else 1,
                           e["键"]))
    for i, e in enumerate(ev):
        e["序号"] = i
    timeline = _建时间线(ev, T)

    # ── 查询 ──
    q = []

    def 查(步, 键, 类型, 延迟=0, 矛盾后=False, 矛盾值=None):
        if int(步) >= T:
            return
        q.append(dict(步=int(步), 键=str(键), 类型=str(类型), 延迟=int(延迟),
                      矛盾后=bool(矛盾后), 矛盾值=(None if 矛盾值 is None else int(矛盾值)),
                      真值=(None if int(步) < 0 else None)))

    for j, k in enumerate(细节键列):
        for d in Dly:
            查(j + d, k, "细节", 延迟=d,
               矛盾后=(j in 矛盾下标 and (j + d) > (j + 16)),
               矛盾值=(矛盾值.get(j) if j in 矛盾下标 else None))
        if j in 矛盾下标:
            查(j + int(cfg["矛盾窗口"]), k, "细节", 延迟=int(cfg["矛盾窗口"]),
               矛盾后=True, 矛盾值=矛盾值[j])
    for k in 关键键:
        for t in (4, 32, 96, 180, T - 8, T - 4):
            查(t, k, "关键", 延迟=int(t))
    for i in range(min(拒绝数, len(干扰键列))):
        查(T - 拒绝数 + i, 干扰键列[i], "干扰")

    for i, x in enumerate(q):
        x["编号"] = i
        x["真值"] = timeline[x["步"]].get(x["键"], None) if x["类型"] != "干扰" else None
    q.sort(key=lambda x: (x["步"], x["编号"]))

    任务 = dict(档=cfg["档"], 种子=int(种子), T=int(T), 配置=dict(cfg),
                事件列=ev, 查询列=q, 真值时间线=timeline,
                关键键列=list(关键键), 细节键列=list(细节键列),
                干扰键列=list(干扰键列), 矛盾键列=[细节键列[j] for j in 矛盾下标],
                稳定键列=[细节键列[j] for j in range(N) if j not in 矛盾下标],
                细节值=[int(v) for v in 细节值],
                矛盾值={细节键列[j]: int(v) for j, v in 矛盾值.items()},
                目标值=int(cfg["目标值"]),
                目标流言值=(int(cfg["目标值"]) + (1 if cfg["档"] != "P1" else 1)) % 8,
                目标二次流言值=(int(cfg["目标值"]) + 3) % 8 if cfg["档"] == "P1" else None,
                自省轮=int(cfg.get("自省轮", 0)))
    return 任务


def 事件按步(任务) -> dict:
    out = {}
    for e in 任务["事件列"]:
        out.setdefault(int(e["步"]), []).append(e)
    return out


def 查询按步(任务) -> dict:
    out = {}
    for x in 任务["查询列"]:
        out.setdefault(int(x["步"]), []).append(x)
    return out


def 真值(任务, 步: int, 键: str):
    t = min(max(int(步), 0), int(任务["T"]) - 1)
    v = 任务["真值时间线"][t].get(str(键), None)
    return None if v is None else int(v)


def 关键真值(任务, 步: int) -> dict:
    t = min(max(int(步), 0), int(任务["T"]) - 1)
    return {k: int(v) for k, v in 任务["真值时间线"][t].items() if k in 关键键}


# ═══════════════════════════════ P0 评分（已知真值） ═══════════════════════════════
def 评分_P0(任务, 记录: dict) -> dict:
    """记录 = dict(目标报告=[T], 槽位报告={关键键:[T]}, 查询回答={查询编号: 值/None})。"""
    T = int(任务["T"])
    目标报告 = list(记录.get("目标报告", []))
    槽位报告 = dict(记录.get("槽位报告", {}))
    查询回答 = dict(记录.get("查询回答", {}))

    def 对(x, y):
        return x is not None and y is not None and int(x) == int(y)

    # 目标：长程保持
    目标对 = [对(目标报告[t] if t < len(目标报告) else None,
              真值(任务, t, 目标键)) for t in range(T)]
    最长 = 0
    当前 = 0
    for ok in 目标对:
        当前 = 当前 + 1 if ok else 0
        最长 = max(最长, 当前)

    # 延迟细节召回
    细节 = [x for x in 任务["查询列"] if x["类型"] == "细节"]
    按延迟 = {}
    for x in 细节:
        d = int(x["延迟"])
        a = 查询回答.get(x["编号"], None)
        ok = 对(a, x["真值"])
        按延迟.setdefault(d, []).append(ok)
    延迟召回 = float(np.mean([对(查询回答.get(x["编号"], None), x["真值"])
                          for x in 细节])) if 细节 else float("nan")
    按延迟率 = {str(d): float(np.mean(v)) for d, v in sorted(按延迟.items())}

    # 矛盾污染（细节：矛盾窗口之后的查询；目标：流言注入之后的每步报告）
    矛盾细节 = [x for x in 细节 if x["矛盾后"]]
    细节污染 = [对(查询回答.get(x["编号"], None), x["矛盾值"]) for x in 矛盾细节]
    细节矛盾错 = [not 对(查询回答.get(x["编号"], None), x["真值"]) for x in 矛盾细节]
    流言值 = int(任务["目标流言值"])
    流入 = (int(任务["配置"]["细节数"]) + 8) if 任务["档"] != "P1" else int(任务["配置"]["目标流言步"])
    目标污染 = [对(目标报告[t] if t < len(目标报告) else None, 流言值)
              for t in range(流入, T)]
    目标矛盾错 = [not 目标对[t] for t in range(流入, T)]
    污染数 = len(细节污染) + len(目标污染)
    污染总 = float((sum(细节污染) + sum(目标污染)) / 污染数) if 污染数 else 0.0
    矛盾错数 = len(细节矛盾错) + len(目标矛盾错)
    矛盾错总 = float((sum(细节矛盾错) + sum(目标矛盾错)) / 矛盾错数) if 矛盾错数 else 0.0

    # 关键槽位保真（每步每关键键）
    保真 = []
    for k in 关键键:
        列 = list(槽位报告.get(k, []))
        for t in range(T):
            保真.append(对(列[t] if t < len(列) else None, 真值(任务, t, k)))
    关键保真 = float(np.mean(保真)) if 保真 else float("nan")

    # 干扰键：正确回答 = 未知（None）
    干扰 = [x for x in 任务["查询列"] if x["类型"] == "干扰"]
    拒答 = [查询回答.get(x["编号"], None) is None for x in 干扰]
    干扰拒答率 = float(np.mean(拒答)) if 干扰 else float("nan")

    # 细节答错率 / 缺失率
    缺答 = [查询回答.get(x["编号"], None) is None for x in 细节]
    return dict(目标正确率=float(np.mean(目标对)),
                目标保持步数=int(最长),
                延迟召回率=float(延迟召回),
                延迟召回率_按延迟=按延迟率,
                细节矛盾污染率=float(np.mean(细节污染)) if 细节污染 else 0.0,
                目标矛盾污染率=float(np.mean(目标污染)) if 目标污染 else 0.0,
                矛盾污染率=float(污染总),
                细节矛盾错误率=float(np.mean(细节矛盾错)) if 细节矛盾错 else 0.0,
                矛盾错误率=float(矛盾错总),
                关键槽位保真度=float(关键保真),
                干扰拒答率=float(干扰拒答率),
                细节缺失率=float(np.mean(缺答)) if 缺答 else 0.0,
                样本数=dict(目标步=T, 细节查询=len(细节), 矛盾细节查询=len(矛盾细节),
                            干扰查询=len(干扰), 关键保真=len(保真)))


def oracle记录(任务) -> dict:
    """仪器自证用：直接给真值。"""
    目标报告 = [真值(任务, t, 目标键) for t in range(int(任务["T"]))]
    槽位报告 = {k: [真值(任务, t, k) for t in range(int(任务["T"]))] for k in 关键键}
    查询回答 = {x["编号"]: x["真值"] for x in 任务["查询列"]}
    return dict(目标报告=目标报告, 槽位报告=槽位报告, 查询回答=查询回答)


def 空臂记录(任务) -> dict:
    """仪器自证用：全部答「未知」。"""
    目标报告 = [None] * int(任务["T"])
    槽位报告 = {k: [None] * int(任务["T"]) for k in 关键键}
    查询回答 = {x["编号"]: None for x in 任务["查询列"]}
    return dict(目标报告=目标报告, 槽位报告=槽位报告, 查询回答=查询回答)


def 自证() -> dict:
    """环境自证：结构合法 + oracle 满分 + 空臂失败 + 双集种子可生成。"""
    条 = []
    例 = 生成(0, "P0")
    T = 例["T"]
    结构 = (len(例["事件列"]) > 0 and len(例["查询列"]) > 0
           and all(0 <= x["步"] < T for x in 例["查询列"])
           and all(0 <= e["步"] < T for e in 例["事件列"]))
    条.append(("P0结构合法", bool(结构)))
    o = 评分_P0(例, oracle记录(例))
    条.append(("P0_oracle满分", bool(o["延迟召回率"] >= 1.0 and o["目标正确率"] >= 1.0
                                  and o["矛盾污染率"] == 0.0 and o["关键槽位保真度"] >= 1.0
                                  and o["干扰拒答率"] >= 1.0)))
    n = 评分_P0(例, 空臂记录(例))
    条.append(("P0_空臂失败", bool(n["延迟召回率"] <= 0.02 and n["目标正确率"] <= 0.02
                                 and n["关键槽位保真度"] <= 0.02)))
    易 = 生成(100, "P0易")
    e = 评分_P0(易, oracle记录(易))
    条.append(("P0易_oracle满分", bool(e["延迟召回率"] >= 1.0 and e["目标正确率"] >= 1.0)))
    条.append(("双集种子可生成", bool(all(生成(s, "P0")["T"] > 0 for s in 判据集 + 留出集))))
    return dict(通过=bool(all(v for _, v in 条)), 条=条,
                样例=dict(档=例["档"], T=T, 事件数=len(例["事件列"]),
                          查询数=len(例["查询列"]),
                          矛盾键数=len(例["矛盾键列"]),
                          细节键数=len(例["细节键列"])))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 自证()
    print(json.dumps(果, ensure_ascii=False, indent=1, default=str))
    print("[环境_长程合成任务] 自证：%s" % ("PASS" if 果["通过"] else "FAIL"))
    raise SystemExit(0 if 果["通过"] else 1)
