# -*- coding: utf-8 -*-
"""meeting41/实施/主程序_第七轮.py —— 第七轮总装（Lead 件）。

一条龙把所有者这一轮要的四件事跑出来（**全部只走既有件与 meeting41 新件，不碰器官**）：

    L1 能力核     → 上下文条件计算（非注意力：码本索引／逐元素门控／定点迭代）
    L2 技能训练   → 15→10→5→3 秒台阶，允许反弹，反弹加练
    L3 同时进行   → 思考链 ⇄ 手 ⇄ 正文 同拍交替，三者不断
    L4 判据       → 判据_三层（8 条，双集 + 反例臂）；判据_效率（E1/E2/E3）
    L5 真机探头   → 演练档自证；真机档需所有者在场（本程序默认不开真机）

用法（在 AHI3 目录下）：
    python meeting41\\实施\\主程序_第七轮.py              # 一条龙演示
    python meeting41\\实施\\主程序_第七轮.py --训练 4      # 只看 4 轮训练台账
    python meeting41\\实施\\主程序_第七轮.py --能力         # 只看能力核自证
    python meeting41\\实施\\主程序_第七轮.py --同时 80      # 跑 80 拍的"边想边做边说"
    python meeting41\\实施\\主程序_第七轮.py --判据         # 跑 判据_三层 双集总判
    python meeting41\\实施\\主程序_第七轮.py --真机演练     # 真机探头演练档（不动鼠标键盘）

诚实边界：`--真机` 会**真的动鼠标键盘**，必须所有者在场并显式确认；本程序不自动开真机。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AHI3 = os.path.dirname(os.path.dirname(HERE))
for _p in (HERE,
           os.path.join(AHI3, "meeting40", "实施"),
           os.path.join(AHI3, "meeting38", "实施")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

产物目录 = os.path.join(HERE, "产物")


def _试导入(名):
    try:
        return __import__(名), None
    except Exception as ex:                                   # noqa: BLE001
        return None, "%s: %s" % (type(ex).__name__, ex)


def _标题(s):
    print("\n" + "=" * 72)
    print("  " + s)
    print("=" * 72)


# ══════════════════════════════════════════════════════════════════════
# L1 能力核
# ══════════════════════════════════════════════════════════════════════

def 跑能力核(种子: int = 0) -> dict:
    _标题("L1 能力核 —— 非注意力，能力对标（线性时间 / 长程依赖 / 任意训练）")
    m, 因 = _试导入("算子_能力核")
    if m is None:
        print("  [缺件] 算子_能力核.py 不可用：%s" % 因)
        return dict(通过=False, 缺件="算子_能力核", 原因=因)
    try:
        核 = m.能力核(D=1024, 种子=种子)
        基 = 核.参数基线()
        print("  维数 D=%s｜算子数=%s｜参数总量=%s" % (基.get("维数"), 基.get("算子数"),
                                                     基.get("参数总量")))
        print("  W码 = %s" % str(基.get("W码"))[:32])
        # 线性时间
        计1, 计4 = 核.计费(1024), 核.计费(4096)
        比 = float(计4.get("乘加数", 0)) / max(1.0, float(计1.get("乘加数", 1)))
        print("  线性时间：计费(1024)=%s  计费(4096)=%s  →  比值=%.3f （<4.5 为线性）"
              % (计1.get("乘加数"), 计4.get("乘加数"), 比))
        # 未定义令牌（任意训练）
        import numpy as np
        码本 = [np.asarray(v, dtype=float) for v in getattr(核, "码本", [])] if hasattr(核, "码本") else []
        rng = np.random.default_rng(seed=种子)
        未 = []
        for i in range(3):
            令 = rng.normal(size=1024)
            前 = 核.前向([令], 上下文=(核.前向([令], 上下文=None)["终态"] if i else None))
            未.append(前)
        非零 = sum(1 for r in 未 if float(np.abs(np.asarray(r["输出"])).max()) > 1e-12)
        两两不同 = len({json.dumps(np.round(np.asarray(r["输出"]).ravel(), 9).tolist())
                       for r in 未}) == 3
        有限 = all(bool(np.isfinite(np.asarray(r["输出"])).all()) for r in 未)
        print("  未定义令牌：非零 %d/3｜两两不同=%s｜全部有限=%s｜如实上报=%s"
              % (非零, 两两不同, 有限, [bool(r.get("未定义令牌")) for r in 未]))
        过 = bool(比 < 4.5 and 非零 >= 2 and 两两不同 and 有限)
        print("  → L1 %s" % ("PASS" if 过 else "FAIL"))
        return dict(通过=过, 维数=基.get("维数"), 参数总量=基.get("参数总量"),
                    W码=基.get("W码"), 比值=比, 未定义非零=非零,
                    未定义两两不同=两两不同, 未定义有限=有限)
    except Exception as ex:                                   # noqa: BLE001
        print("  [异常] %r" % (ex,))
        return dict(通过=False, 异常=repr(ex)[:300])


# ══════════════════════════════════════════════════════════════════════
# L2 技能训练
# ══════════════════════════════════════════════════════════════════════

def 跑训练(种子: int = 0, 轮数: int = 6) -> dict:
    _标题("L2 技能训练 —— 台阶 15→10→5→3 秒（允许反弹，反弹即加练）")
    TL, 因 = _试导入("训练循环")
    if TL is None:
        print("  [缺件] 训练循环：%s" % 因)
        return dict(通过=False, 缺件="训练循环", 原因=因)
    try:
        # ★直接驱动**已验证**的训练闭环（技能原子 × 技能调度 × 技能库），
        #   而不是在这里另造一把小技能——否则主程序演示的台阶与判据量的台阶不是同一条曲线。
        报 = TL.跑训练(种子=种子, 轮数=int(轮数))
        print("  基准(该局面第一次操作) = %.1f ms" % float(报["基准毫秒"]))
        print("  轮次 | 次数 |   最好ms |   中位ms |   末次ms | 反弹 | 加练 | 台阶 | 档 | 达3秒")
        print("  " + "-" * 88)
        for r in 报["轮列"]:
            print("  %4d | %4d | %8.1f | %8.1f | %8.1f | %4s | %4s | %4s | %2d | %s"
                  % (r["轮次"], r["次数"], r["最好毫秒"], r["中位毫秒"], r["末次毫秒"],
                     r["反弹"], r["加练"], r["台阶"], r["档"], r["达3秒"]))
        最好 = float(报["最好毫秒"])
        缩比 = 报["缩比"]
        print("\n  最快 = %.1f ms｜缩比 = %.3f（所有者口径 3s/15s = 0.200）｜台阶序列 = %s"
              % (最好, 缩比, 报["台阶序列"]))
        反弹轮 = 报["反弹轮"]
        print("  到达 3 秒：%s%s" % (报["达3秒"],
                                  "" if 报["达3秒"] else "（该局面档3 不可达，见 证据_三层.md §诚实边界）"))
        print("  出现反弹的轮 = %s（反弹 ⇒ 加练，下一轮次数更多）" % (反弹轮 or "无"))
        # 通过口径与总判据一致：机制真的在变快（缩比 ≤0.5）+ 至少走 2 级台阶
        过 = bool(缩比 is not None and float(缩比) <= 0.5 and len({s for s in 报["台阶序列"] if s >= 0}) >= 2)
        print("  → L2 %s（缩比 %.3f ≤0.5 且台阶≥2级）" % ("PASS" if 过 else "FAIL", 缩比))
        return dict(通过=过, 轮列=报["轮列"], 最好毫秒=最好, 基准毫秒=报["基准毫秒"],
                    缩比=缩比, 达3秒=bool(报["达3秒"]), 台阶序列=报["台阶序列"],
                    反弹轮=反弹轮, 库台账=报.get("库台账"))
    except Exception as ex:                                   # noqa: BLE001
        print("  [异常] %r" % (ex,))
        return dict(通过=False, 异常=repr(ex)[:300])


# ══════════════════════════════════════════════════════════════════════
# L3 同时进行
# ══════════════════════════════════════════════════════════════════════

def 跑同时(种子: int = 0, 拍数: int = 60) -> dict:
    _标题("L3 同时进行 —— 边思考、边操作、边输出正文（三者不断）")
    流, 因1 = _试导入("输出流")
    契, 因2 = _试导入("事件契约_操作")
    时, 因3 = _试导入("时基")
    for 名, 因 in (("输出流", 因1), ("事件契约_操作", 因2), ("时基", 因3)):
        if 因 is not None:
            print("  [缺件] %s：%s" % (名, 因))
            return dict(通过=False, 缺件=名, 原因=因)
    try:
        核 = None
        m核, _ = _试导入("算子_能力核")
        if m核 is not None and hasattr(m核, "能力核"):
            try:
                核 = m核.能力核(D=1024, 种子=种子)
            except Exception:                                 # noqa: BLE001
                核 = None
        o = 流.输出流(能力核=核, 钟=时.拍钟(), 种子=种子)
        for i in range(int(拍数)):
            o.一拍(输入s=dict(拍=i))
        链, 指标 = o.链(), o.链指标()
        ev = 契.事件契约_操作()
        绑 = ev.绑(链["动作列"], 链["正文项列"], 链["思考项列"])
        断 = ev.不断证(链["思考项列"], int(拍数))
        print("  思考项 %s｜动作 %s｜正文分片 %s" % (指标.get("思考数"), 指标.get("动作数"),
                                                 指标.get("正文数")))
        print("  最大空隙拍 = %s（≤2 = 思考链不断）" % 断.get("最大空隙拍"))
        print("  思考跨操作 = %s（>0 = 有动作的拍里也在思考）" % 指标.get("思考跨操作"))
        print("  孤立动作数 = %s（=0 = 每个动作都有正文/思考伴着）" % 绑.get("孤立动作数"))
        print("  覆盖率 = %s｜字速 = %s｜首字延迟 = %s ms"
              % (绑.get("覆盖率"), 指标.get("字速"), 指标.get("首字延迟毫秒")))
        样 = o.正文全文()
        print("\n  ── 它自己说出来的正文（前 300 字）──")
        print("  " + (样[:300].replace("\n", "\n  ") if 样 else "（空）"))
        过 = bool(int(断.get("最大空隙拍", 9)) <= 2
                  and int(指标.get("思考跨操作", 0)) > 0
                  and int(绑.get("孤立动作数", 9)) == 0)
        print("\n  → L3 %s" % ("PASS" if 过 else "FAIL"))
        return dict(通过=过, 指标=指标, 绑=绑, 断=断, 正文=样[:800])
    except Exception as ex:                                   # noqa: BLE001
        print("  [异常] %r" % (ex,))
        return dict(通过=False, 异常=repr(ex)[:300])


# ══════════════════════════════════════════════════════════════════════
# L4 判据
# ══════════════════════════════════════════════════════════════════════

def 跑判据() -> dict:
    _标题("L4 总判据 —— 判据_三层（双集 + 反例臂）")
    m, 因 = _试导入("判据_三层")
    if m is None:
        print("  [缺件] 判据_三层：%s" % 因)
        return dict(通过=False, 缺件="判据_三层", 原因=因)
    报 = m.跑并存()
    for k, v in 报["判据"].items():
        print("  %-18s %s" % (k, "PASS" if v["全过"] else "FAIL"))
        if not v["全过"]:
            for 集名, 集 in v["集"].items():
                坏 = [x for x in 集["逐种子"] if not x["通过"]]
                if 坏:
                    print("      [%s] 首个不过：seed=%s %s" % (
                        集名, 坏[0]["种子"],
                        json.dumps(坏[0]["主"], ensure_ascii=False)[:220]))
                    break
    print("\n  总判：%s｜特征码 %s" % (m.决策(报), 报["本台特征码"][:24]))
    print("  结果落盘：%s" % 报["_路径"])
    return dict(通过=bool(报["总过"]), 逐条=报["总过_逐条"], 特征码=报["本台特征码"],
                路径=报["_路径"])


def 跑效率判据() -> dict:
    _标题("L4b 效率判据 —— E1 线性时间 / E2 长程依赖 / E3 任意培训")
    m, 因 = _试导入("判据_效率")
    if m is None:
        print("  [缺件] 判据_效率：%s" % 因)
        return dict(通过=False, 缺件="判据_效率", 原因=因)
    try:
        for 名 in ("跑双集", "跑", "主"):
            f = getattr(m, 名, None)
            if callable(f):
                报 = f()
                print(json.dumps(报, ensure_ascii=False, indent=1, default=str)[:2500])
                return dict(通过=bool((报 or {}).get("总过", (报 or {}).get("通过", False))),
                            报=报)
        print("  [缺件] 判据_效率 没有可调用的入口（跑双集/跑/主）")
        return dict(通过=False, 缺件="判据_效率.入口")
    except Exception as ex:                                   # noqa: BLE001
        print("  [异常] %r" % (ex,))
        return dict(通过=False, 异常=repr(ex)[:300])


# ══════════════════════════════════════════════════════════════════════
# L5 真机探头（演练档）
# ══════════════════════════════════════════════════════════════════════

def 跑真机演练() -> dict:
    _标题("L5 真机探头 —— 演练档（绝不驱动真鼠标键盘）")
    m, 因 = _试导入("真机探头")
    if m is None:
        print("  [缺件] 真机探头：%s" % 因)
        return dict(通过=False, 缺件="真机探头", 原因=因)
    try:
        # ★真件入口是 `真机探头.主(演练=True, ...)`；本程序**只调演练档**。
        #   真机档（演练=False）会真动鼠标键盘，**必须所有者在场**，本程序不自动开。
        #   `真任务=False`：主程序这条一条龙**不跑**真任务链（那一步会真执行既有三个任务件、
        #   含解析与落盘，很重且与 L5 的验收目的无关）。完整演练档请单独跑：
        #       python meeting41\实施\真机探头.py --演练
        #   本程序只验**三件安全物 + 演练档不碰真设备 + 演练档确定性**。
        主 = getattr(m, "主", None)
        if not callable(主):
            print("  [缺件] 真机探头 没有 主() 入口")
            return dict(通过=False, 缺件="真机探头.主")
        报 = 主(演练=True, 已确认=False, 真任务=False, 截图=True)
        汇总 = (报 or {}).get("判定") or (报 or {})
        # `_汇总判定` 不返回 `通过` 键；L5 的通过口径 = 三件安全物全过 ∧ 原子全放行
        #   ∧ 有脱靶对照（证明"打不中"能被抓到）∧ 截图前后都落盘 ∧ 演练档两遍净台账逐位同。
        安全物 = (报 or {}).get("安全物") or {}
        过 = bool(汇总.get("安全物全过") and 汇总.get("原子全放行")
                  and 汇总.get("有脱靶对照") and 汇总.get("截图前后落盘")
                  and 汇总.get("演练档两遍净台账逐位同"))
        print("  安全物：①令牌闸=%s｜②演练/真机双档=%s｜③现场确认=%s"
              % ((安全物.get("①许可令牌闸") or {}).get("通过"),
                 (安全物.get("②演练真机双档") or {}).get("通过"),
                 (安全物.get("③现场确认") or {}).get("通过")))
        print("  原子全放行=%s｜命中 %s/%s｜有脱靶对照=%s｜截图前后落盘=%s｜两遍净台账逐位同=%s"
              % (汇总.get("原子全放行"), 汇总.get("命中数"), 汇总.get("原子数"),
                 汇总.get("有脱靶对照"), 汇总.get("截图前后落盘"),
                 汇总.get("演练档两遍净台账逐位同")))
        print("  真机动作数 = %s（★本程序只跑演练档；真机档需所有者在场）"
              % (报 or {}).get("真机动作数", "?"))
        print("  → L5 %s" % ("PASS" if 过 else "FAIL"))
        return dict(通过=过, 档=(报 or {}).get("档"),
                    真机动作数=(报 or {}).get("真机动作数"), 安全物=安全物, 判定=汇总)
    except Exception as ex:                                   # noqa: BLE001
        print("  [异常] %r" % (ex,))
        return dict(通过=False, 异常=repr(ex)[:300])


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

def 主() -> int:
    ap = argparse.ArgumentParser(description="AHI V3 第七轮：能力核 / 操作熟练 / 同时进行")
    ap.add_argument("--种子", type=int, default=0)
    ap.add_argument("--训练", type=int, metavar="轮数", help="只跑 L2 训练，给轮数")
    ap.add_argument("--同时", type=int, metavar="拍数", help="只跑 L3，给拍数")
    ap.add_argument("--能力", action="store_true", help="只跑 L1 能力核")
    ap.add_argument("--判据", action="store_true", help="只跑 L4 总判据（双集）")
    ap.add_argument("--效率", action="store_true", help="只跑 L4b 效率判据")
    ap.add_argument("--真机演练", action="store_true", help="只跑 L5 真机探头演练档")
    a = ap.parse_args()

    果 = dict(种子=a.种子)
    if a.能力:
        果["L1"] = 跑能力核(a.种子)
    elif a.训练 is not None:
        果["L2"] = 跑训练(a.种子, a.训练)
    elif a.同时 is not None:
        果["L3"] = 跑同时(a.种子, a.同时)
    elif a.判据:
        果["L4"] = 跑判据()
    elif a.效率:
        果["L4b"] = 跑效率判据()
    elif a.真机演练:
        果["L5"] = 跑真机演练()
    else:
        print("AHI V3 · 第七轮总装 —— 能力核 / 操作熟练 / 同时进行")
        print("目录：%s" % HERE)
        果["L1"] = 跑能力核(a.种子)
        果["L2"] = 跑训练(a.种子, 6)
        果["L3"] = 跑同时(a.种子, 60)
        果["L4b"] = 跑效率判据()
        果["L5"] = 跑真机演练()
        果["L4"] = 跑判据()

    逐 = {k: bool((v or {}).get("通过")) for k, v in 果.items() if k != "种子"}
    if 逐:
        _标题("总汇")
        for k in sorted(逐):
            print("  %-6s %s" % (k, "PASS" if 逐[k] else "FAIL"))
        print("  → %s" % ("全部 PASS" if all(逐.values()) else
                          "存在 FAIL：%s" % "、".join(k for k in sorted(逐) if not 逐[k])))
    os.makedirs(产物目录, exist_ok=True)
    路径 = os.path.join(产物目录, "主程序_第七轮_结果.json")
    with open(路径, "w", encoding="utf-8") as f:
        json.dump(果, f, ensure_ascii=False, indent=1, default=str)
    print("\n落盘：%s" % 路径)
    return 0 if all(逐.values()) else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                         # noqa: BLE001
        pass
    raise SystemExit(主())
