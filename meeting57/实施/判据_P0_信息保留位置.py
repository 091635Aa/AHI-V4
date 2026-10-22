# -*- coding: utf-8 -*-
"""AHI4/meeting57/实施/判据_P0_信息保留位置.py —— P0 判据：信息保留位置消融（A/B/C/S）。

四臂（阈值先冻结；开发集 100–103 只校准，判据集 0–5 / 留出集 6–11 只判不调）：
  A 窄递归状态（D=1024）：叠加态 + 解绑 + 硬阈值迭代吸引子回读；
  B 窄状态 + 离散锚点/账本旁路：关键事件原样入账，受控回读；
  C 宽状态/无瓶颈上界：每键独立 1024 维槽位，精确回读（不是目标架构）；
  S B + sham：随机锚点/打乱账本/惰性回读（零假设对照，不得优于 A）。

判据（见 00_预注册_P0P1探针.md §2.4）：
  P0-1 B 延迟召回显著优于 A；P0-2 B ≥ 0.80·C；P0-3 B 目标/关键保真显著优于 A；
  P0-4 B 污染显著低于 A；P0-5 A 长程（Δ=96）显著退化；P0-6 S 相对 A 无显著提升；
  P0-7 双集都过；P0-8 两趟逐位同；P0-9 C1 源码自扫/迭代吸引子审查；
  P0-10 仪器自证（oracle/空臂/易任务/容量扫描/测量台自证）；P0-11 峰值内存 < 2GB。

直跑：python 判据_P0_信息保留位置.py  → 写 判据_P0_信息保留位置结果.json
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import tracemalloc

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
本件 = os.path.abspath(__file__)
AHI4 = os.path.dirname(os.path.dirname(HERE))
环境路径 = os.path.join(HERE, "环境_长程合成任务.py")
载体路径 = os.path.join(HERE, "算子_信息保留位置.py")
测量台路径 = os.path.join(AHI4, "meeting40", "实施", "测量台.py")
能力核路径 = os.path.join(AHI4, "meeting41", "实施", "算子_能力核.py")
结果路径 = os.path.join(HERE, "判据_P0_信息保留位置结果.json")


def _载(路径, 名):
    spec = importlib.util.spec_from_file_location(名, 路径)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E = _载(环境路径, "环境_P0_判据")
M = _载(载体路径, "载体_P0_判据")
T = _载(测量台路径, "测量台_P0_判据")
K = _载(能力核路径, "能力核_P0_判据")
np.seterr(all="ignore")

# ═══════════════════════════════ 冻结阈值（先声明，后跑；不看结果改） ═══════════════════════════════
B减A_召回_min = 0.15
B比C_召回_min = 0.80
B减A_目标_min = 0.15
B减A_保真_min = 0.15
B污染_max = 0.05
B污染比A低_min = 0.05
A长程掉_min = 0.10
S比A_召回_max = 0.05
S比A_复合_max = 0.02
B复合减A_min = 0.15
B绝对召回_min = 0.80
显著性_p_max = 0.05
内存上限MB = 2048.0
允许导入 = ("__future__", "numpy", "json", "os", "sys", "importlib",
            "time", "tracemalloc", "math", "hashlib")


# ═══════════════════════════════ 跑臂（不含时间/内存，保证可哈希确定性） ═══════════════════════════════
def 跑臂(臂: str, 任务: dict, 种子: int) -> dict:
    体 = M.建(臂, 任务, 种子=int(种子))
    记录 = dict(目标报告=[], 槽位报告={k: [] for k in 任务["关键键列"]}, 查询回答={})
    事件步 = E.事件按步(任务)
    查询步 = E.查询按步(任务)
    for t in range(int(任务["T"])):
        for e in 事件步.get(t, ()):
            体.写(e)
        体.步进(t)
        记录["目标报告"].append(体.读(E.目标键))
        for k in 任务["关键键列"]:
            记录["槽位报告"][k].append(体.读(k))
        for q in 查询步.get(t, ()):
            记录["查询回答"][q["编号"]] = 体.读(q["键"])
    return E.评分_P0(任务, 记录)


def 跑全臂(种子: int) -> dict:
    任务 = E.生成(int(种子), "P0")
    return {臂: 跑臂(臂, 任务, int(种子)) for 臂 in ("A", "B", "C", "S")}


def 复合分(果: dict) -> float:
    return float(np.mean([果["延迟召回率"], 果["目标正确率"],
                          果["关键槽位保真度"], 1.0 - 果["矛盾污染率"]]))


def 判据_单种子(果: dict) -> tuple:
    A, B, C, S = 果["A"], 果["B"], 果["C"], 果["S"]
    检查 = {
        "B减A_召回": float(B["延迟召回率"] - A["延迟召回率"]),
        "B比C_召回": float(B["延迟召回率"] / max(1e-12, C["延迟召回率"])),
        "B减A_目标": float(B["目标正确率"] - A["目标正确率"]),
        "B减A_保真": float(B["关键槽位保真度"] - A["关键槽位保真度"]),
        "A污染减B污染": float(A["矛盾污染率"] - B["矛盾污染率"]),
        "A8减A96": float(A["延迟召回率_按延迟"]["8"] - A["延迟召回率_按延迟"]["96"]),
        "S减A_召回": float(S["延迟召回率"] - A["延迟召回率"]),
        "B复合减A": float(复合分(B) - 复合分(A)),
        "S复合减A": float(复合分(S) - 复合分(A)),
        "B绝对召回": float(B["延迟召回率"]),
    }
    条件 = {
        "B减A_召回": 检查["B减A_召回"] >= B减A_召回_min,
        "B比C_召回": 检查["B比C_召回"] >= B比C_召回_min,
        "B减A_目标": 检查["B减A_目标"] >= B减A_目标_min,
        "B减A_保真": 检查["B减A_保真"] >= B减A_保真_min,
        "B污染": (B["矛盾污染率"] <= B污染_max
                 and 检查["A污染减B污染"] >= B污染比A低_min),
        "A长程掉": 检查["A8减A96"] >= A长程掉_min,
        "S不优于A召回": 检查["S减A_召回"] <= S比A_召回_max,
        "B绝对召回": 检查["B绝对召回"] >= B绝对召回_min,
    }
    return bool(all(条件.values())), dict(实测=检查, 条件=条件, 复合={k: 复合分(果[k]) for k in 果})


def 主臂(种子: int) -> dict:
    果 = 跑全臂(int(种子))
    通过, 检查 = 判据_单种子(果)
    return dict(通过=bool(通过), 臂=果, 检查=检查)


def 零臂(种子: int) -> dict:
    """sham 臂=S：若 S 也达到 B 的绝对门槛，则 sham 未失效 ⇒ 测量台判 FAIL。"""
    任务 = E.生成(int(种子), "P0")
    果 = 跑臂("S", 任务, int(种子))
    达标 = bool(果["延迟召回率"] >= B绝对召回_min
              and 果["目标正确率"] >= 0.80
              and 果["关键槽位保真度"] >= 0.80
              and 果["矛盾污染率"] <= B污染_max)
    return dict(通过=达标, 臂=果)


def 门槛(主: dict, 零: dict, 种子: int) -> bool:
    return bool(主.get("通过")) and not bool((零 or {}).get("通过"))


# ═══════════════════════════════ 统计（无 scipy；精确符号检验） ═══════════════════════════════
def 符号p(列) -> float:
    """单侧符号检验：H1=正向；p=P(X≥正向数)，X~Bin(n,0.5)。"""
    列 = [float(x) for x in 列]
    n = len(列)
    if n <= 0:
        return 1.0
    k = sum(1 for x in 列 if x > 0)
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / float(2 ** n))


def 汇总结论(报: dict) -> dict:
    集报 = {}
    for 集名 in ("判据集", "留出集"):
        逐 = 报["集"][集名]["P0_主判据"]["逐种子"]
        行 = [x["主"]["臂"] for x in 逐]
        零 = [x["零"] for x in 逐]
        A = [r["A"] for r in 行]
        B = [r["B"] for r in 行]
        C = [r["C"] for r in 行]
        S = [r["S"] for r in 行]
        列 = dict(
            B减A_召回=[b["延迟召回率"] - a["延迟召回率"] for a, b in zip(A, B)],
            B比C_召回=[b["延迟召回率"] / max(1e-12, c["延迟召回率"]) for b, c in zip(B, C)],
            B减A_目标=[b["目标正确率"] - a["目标正确率"] for a, b in zip(A, B)],
            B减A_保真=[b["关键槽位保真度"] - a["关键槽位保真度"] for a, b in zip(A, B)],
            A污染减B污染=[a["矛盾污染率"] - b["矛盾污染率"] for a, b in zip(A, B)],
            A8减A96=[a["延迟召回率_按延迟"]["8"] - a["延迟召回率_按延迟"]["96"] for a in A],
            S减A_召回=[s["延迟召回率"] - a["延迟召回率"] for s, a in zip(S, A)],
            B复合减A=[复合分(b) - 复合分(a) for a, b in zip(A, B)],
            S复合减A=[复合分(s) - 复合分(a) for s, a in zip(S, A)],
        )
        p = {k: 符号p(v) for k, v in 列.items()}
        均 = {k: float(np.mean(v)) for k, v in 列.items()}
        条 = {
            "B减A_召回显著": bool(均["B减A_召回"] >= B减A_召回_min and p["B减A_召回"] < 显著性_p_max),
            "B比C_召回": bool(min(列["B比C_召回"]) >= B比C_召回_min),
            "B减A_目标显著": bool(均["B减A_目标"] >= B减A_目标_min and p["B减A_目标"] < 显著性_p_max),
            "B减A_保真显著": bool(均["B减A_保真"] >= B减A_保真_min and p["B减A_保真"] < 显著性_p_max),
            "B污染显著低": bool(均["A污染减B污染"] >= B污染比A低_min and p["A污染减B污染"] < 显著性_p_max),
            "A长程显著退": bool(均["A8减A96"] >= A长程掉_min and p["A8减A96"] < 显著性_p_max),
            "S不优于A_召回": bool(均["S减A_召回"] <= S比A_召回_max
                              and (p["S减A_召回"] >= 显著性_p_max
                                   or 均["S减A_召回"] <= S比A_复合_max)),
            "S不优于A_复合": bool(均["S复合减A"] <= S比A_复合_max
                              and (p["S复合减A"] >= 显著性_p_max
                                   or 均["S复合减A"] <= S比A_复合_max)),
            "B复合显著优A": bool(均["B复合减A"] >= B复合减A_min and p["B复合减A"] < 显著性_p_max),
            "sham全不达标": bool(all(not bool(z.get("通过")) for z in 零)),
            "确定性全过": bool(all(bool(x["确定性"]) for x in 逐)),
        }
        集报[集名] = dict(通过=bool(all(条.values())), 条=条,
                          均值=均, p=p, 列=列,
                          阈值=dict(B减A_召回_min=B减A_召回_min,
                                    B比C_召回_min=B比C_召回_min,
                                    B减A_目标_min=B减A_目标_min,
                                    B减A_保真_min=B减A_保真_min,
                                    B污染比A低_min=B污染比A低_min,
                                    A长程掉_min=A长程掉_min,
                                    S比A_召回_max=S比A_召回_max,
                                    S比A_复合_max=S比A_复合_max,
                                    B复合减A_min=B复合减A_min,
                                    显著性_p_max=显著性_p_max))
    return dict(通过=bool(集报["判据集"]["通过"] and 集报["留出集"]["通过"]),
                集=集报, 说明="精确符号检验（n=6/集，6/6 正向时 p=1/64≈0.0156）；双集都过才算过")


# ═══════════════════════════════ 静态：C1 / 导入 / 回读式 / 只读件哈希 ═══════════════════════════════
def 扫导入(路径: str) -> list:
    名 = set()
    with open(路径, encoding="utf-8") as f:
        for 行 in f.read().splitlines():
            s = 行.strip()
            if s.startswith("import "):
                名.add(s[len("import "):].split()[0].split(",")[0].split(".")[0])
            elif s.startswith("from "):
                名.add(s[len("from "):].split()[0].split(".")[0])
    return sorted(名)


def _哈希(路径: str) -> str:
    with open(路径, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def 静态检查() -> dict:
    # 只扫 P0 自己的文件（不把 P1 文件耦合进 P0 的白名单/禁词口径）
    候选 = [环境路径, 载体路径, 本件,
            os.path.join(HERE, "开发_校准.py"),
            os.path.join(HERE, "开发_校准_rho.py"),
            os.path.join(HERE, "开发_校准_清理.py"),
            os.path.join(HERE, "开发_校准_消融.py")]
    新文件 = [p for p in 候选 if os.path.exists(p)]
    禁词 = list(K.禁词列())
    命中 = {}
    for p in 新文件:
        with open(p, encoding="utf-8") as f:
            低 = f.read().lower()
        命中[os.path.basename(p)] = [w for w in 禁词 if w.lower() in 低]
    命中总数 = sum(len(v) for v in 命中.values())
    导入表 = {os.path.basename(p): 扫导入(p) for p in 新文件}
    越界 = {k: [m for m in v if m not in 允许导入] for k, v in 导入表.items()}
    越界 = {k: v for k, v in 越界.items() if v}
    with open(载体路径, encoding="utf-8") as f:
        载体原文 = f.read()
    回读式 = bool("吸引子清理" in 载体原文 and "吸引子步数" in 载体原文
                and "值码 @ x" in 载体原文 and int(getattr(M, "吸引子步数", 0)) >= 1)
    只读哈希 = {
        "测量台": _哈希(测量台路径),
        "能力核": _哈希(能力核路径),
    }
    通过 = bool(命中总数 == 0 and not 越界 and 回读式)
    return dict(通过=通过, 禁词命中=命中, 禁词命中总数=命中总数,
                导入表=导入表, 导入越界=越界, 回读式=回读式,
                只读既有件哈希=只读哈希,
                新文件=sorted(os.path.basename(p) for p in 新文件),
                说明="C1：新 .py 禁词零命中 + 回读走硬阈值迭代吸引子；导入白名单；"
                     "既有件只读（哈希记录，git diff 另行复核）")


# ═══════════════════════════════ 仪器自证（先仪器后对象） ═══════════════════════════════
def 仪器自证() -> dict:
    条 = {}
    环境自证 = E.自证()
    条["环境自证"] = bool(环境自证["通过"])

    任务 = E.生成(100, "P0")
    o = E.评分_P0(任务, E.oracle记录(任务))
    条["oracle满分"] = bool(o["延迟召回率"] >= 1.0 and o["目标正确率"] >= 1.0
                         and o["关键槽位保真度"] >= 1.0 and o["矛盾污染率"] == 0.0
                         and o["干扰拒答率"] >= 1.0)
    n = E.评分_P0(任务, E.空臂记录(任务))
    条["空臂失败"] = bool(n["延迟召回率"] <= 0.02 and n["目标正确率"] <= 0.02
                        and n["关键槽位保真度"] <= 0.02)

    易 = []
    for s in E.开发集:
        易.append(跑臂("A", E.生成(s, "P0易"), s))
    条["易任务A可用"] = bool(np.mean([x["延迟召回率"] for x in 易]) >= 0.95
                          and np.mean([x["目标正确率"] for x in 易]) >= 0.95
                          and np.mean([x["关键槽位保真度"] for x in 易]) >= 0.95)

    扫 = {}
    for N in (32, 192):
        行 = []
        for s in E.开发集:
            任务 = E.生成(s, "P0", 细节数=N, 矛盾细节数=min(16, N))
            行.append(跑臂("A", 任务, s))
        扫[str(N)] = dict(延迟召回=float(np.mean([x["延迟召回率"] for x in 行])),
                          延迟8=float(np.mean([x["延迟召回率_按延迟"]["8"] for x in 行])),
                          延迟96=float(np.mean([x["延迟召回率_按延迟"]["96"] for x in 行])),
                          目标正确率=float(np.mean([x["目标正确率"] for x in 行])),
                          关键保真=float(np.mean([x["关键槽位保真度"] for x in 行])))
    条["容量扫描可分辨"] = bool(扫["192"]["延迟96"] <= 扫["32"]["延迟96"] - 0.10
                          and 扫["192"]["延迟8"] >= 0.90)

    R = 跑臂("R", E.生成(100, "P0"), 100)
    可失败, _ = 判据_单种子(dict(A=R, B=R, C=R, S=R))
    条["判据可失败"] = bool(not 可失败)

    台 = T.本台自证()
    条["测量台自证"] = bool(台["通过"])
    return dict(通过=bool(all(条.values())), 条=条, 环境自证=环境自证,
                oracle=o, 空臂=n, 易任务=dict(
                    延迟召回=float(np.mean([x["延迟召回率"] for x in 易])),
                    目标正确率=float(np.mean([x["目标正确率"] for x in 易])),
                    关键保真=float(np.mean([x["关键槽位保真度"] for x in 易]))),
                容量扫描=扫, 测量台自证=台,
                说明="oracle 满分 / 空臂≈机会 / 易任务 A 高（非稻草人）/ "
                     "容量扫描显示 192 事实的 Δ=96 显著低于 32 事实 / 判据对随机臂可判 FAIL / 测量台自证 PASS")


# ═══════════════════════════════ 主流程 ═══════════════════════════════
def 主流程() -> dict:
    静态 = 静态检查()
    自证 = 仪器自证()

    台 = T.测量台("P0_信息保留位置")
    台.注册("P0_主判据", 主臂=主臂, sham=零臂, 门槛=门槛,
            说明="A窄状态 / B账本旁路 / C宽状态上界 / S sham；双集 seed 0–5 / 6–11")

    tracemalloc.start()
    t0 = time.process_time()
    报 = 台.跑双集()
    秒 = time.process_time() - t0
    当前, 峰值 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    资源 = dict(CPU秒=float(秒), 峰值内存MB=float(峰值) / (1024.0 * 1024.0),
                当前内存MB=float(当前) / (1024.0 * 1024.0),
                通过=bool(峰值 / (1024.0 * 1024.0) < 内存上限MB),
                口径="tracemalloc Python 分配峰值；非 RSS（诚实边界）")

    汇总 = 汇总结论(报)
    未过 = []
    if not 静态["通过"]:
        未过.append("静态_C1/导入/回读式")
    if not 自证["通过"]:
        未过.append("仪器自证")
    if not 报.get("总过"):
        未过.append("测量台双集/确定性/sham")
    if not 汇总["通过"]:
        未过.append("汇总显著性/S不优于A")
    if not 资源["通过"]:
        未过.append("内存<2GB")
    结论 = dict(通过=bool(not 未过), 未过=未过,
                判据集=bool(汇总["集"]["判据集"]["通过"]),
                留出集=bool(汇总["集"]["留出集"]["通过"]),
                测量台总过=bool(报.get("总过")),
                静态=bool(静态["通过"]), 仪器自证=bool(自证["通过"]),
                资源=bool(资源["通过"]))

    报 = dict(报, 汇总=汇总, 静态=静态, 仪器自证=自证, 资源=资源,
              结论=结论, 阈值=dict(B减A_召回_min=B减A_召回_min,
                                   B比C_召回_min=B比C_召回_min,
                                   B减A_目标_min=B减A_目标_min,
                                   B减A_保真_min=B减A_保真_min,
                                   B污染_max=B污染_max,
                                   B污染比A低_min=B污染比A低_min,
                                   A长程掉_min=A长程掉_min,
                                   S比A_召回_max=S比A_召回_max,
                                   S比A_复合_max=S比A_复合_max,
                                   B复合减A_min=B复合减A_min,
                                   B绝对召回_min=B绝对召回_min,
                                   显著性_p_max=显著性_p_max,
                                   内存上限MB=内存上限MB),
              说明=("P0 探索性探针：合成任务 + 双集 + sham + 两趟确定性。"
                   "B 为离散锚点/账本旁路；C 为宽状态上界（不是目标架构）；"
                   "A 是固定码叠加的窄状态载体，不是任意 1024 维可训练 RNN 的上界。"))
    报["本台特征码"] = T.特征码(报)
    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1, default=str)
    return 报


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    报 = 主流程()
    结 = 报["结论"]
    print(json.dumps(dict(通过=结["通过"], 未过=结["未过"],
                          判据集=结["判据集"], 留出集=结["留出集"],
                          静态=结["静态"], 仪器自证=结["仪器自证"],
                          资源=结["资源"], 测量台总过=结["测量台总过"],
                          CPU秒=报["资源"]["CPU秒"],
                          峰值内存MB=报["资源"]["峰值内存MB"]),
                     ensure_ascii=False, indent=1, default=str))
    for 集名 in ("判据集", "留出集"):
        s = 报["汇总"]["集"][集名]
        print("[%s] 通过=%s 条=%s" % (集名, s["通过"],
              json.dumps(s["条"], ensure_ascii=False)))
        print("  均值=%s" % json.dumps(s["均值"], ensure_ascii=False))
        print("  p=%s" % json.dumps(s["p"], ensure_ascii=False))
    print("结果已写：", 结果路径)
    raise SystemExit(0 if 结["通过"] else 1)
