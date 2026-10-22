# -*- coding: utf-8 -*-
"""AHI4/meeting57/实施/开发_校准_P1.py —— P1 开发集校准（seed 100–103，**只校准、不进判据**）。

只做两件事：
  ① 在冻结参数下测 A′/B′/S′ 的六项指标（与判据同一 `计算指标` 口径）；
  ② 扫「自省强度」（A′ 的自指回写权重）对 A′ 指标的影响，用于说明最终冻结值 1.0 的依据。
判据集 seed 0–5 / 留出集 6–11 不在这里出现；阈值由本脚本的 dev 结果决定后写死在判据顶部。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AHI4 = os.path.dirname(os.path.dirname(HERE))
判据路径 = os.path.join(HERE, "判据_P1_仲裁隔离.py")
结果路径 = os.path.join(HERE, "开发_校准_P1结果.json")
开发集 = (100, 101, 102, 103)


def _载(路径, 名):
    spec = importlib.util.spec_from_file_location(名, 路径)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_判据 = _载(判据路径, "判据_P1_开发校准")
_环境 = _判据._环境
_算子 = _判据._算子

指标键 = ("语义漂移率", "语义漂移率_同轮", "震荡幅度", "冲突后错误率",
          "错误传播率", "更正延迟", "复合仲裁分", "查询正确率", "干扰拒答率",
          "关键槽位正确率")
结果_判据路径 = os.path.join(HERE, "判据_P1_仲裁隔离结果.json")


# ═══════════════════ post-hoc 公平基线（诊断，不是新判据）═══════════════════
class A有据臂(_算子.A自指臂):
    """A′_有据：与 A′ 相同（事件直接写入 + 每轮把读回写回），但**只对已收到过
    任务相关事件的键**做自省回写；不对从未写入的键写回幻读。用于 post-hoc 诊断
    「A′ 的崩溃/幻读固化」占 P1-6 失败的多大权重。"""
    名 = "A′_有据"

    def __init__(self, 任务, 种子=0, 参数=None):
        super().__init__(任务, 种子=种子, 参数=参数)
        self.已见 = set()

    def 收事件(self, 事件列):
        for e in 事件列:
            if bool(e.get("任务相关", True)):
                self.已见.add(str(e["键"]))
        super().收事件(事件列)

    def 自省轮(self, 轮, 探测键):
        for k in 探测键:
            if str(k) not in self.已见:
                continue
            v = self.状态.读(k)
            if v is not None:
                self.状态.自省回写(k, v, 强度=float(self.参数["自省强度"]))


_臂列诊断 = ("A′", "A′_有据", "B′", "S′")


def 公平基线():
    """dev 上跑 A′ / A′_有据 / B′ / S′，给四臂逐种子与均值指标 + 差值汇总。"""
    _算子._臂表["A′_有据"] = A有据臂
    表 = {}
    for s in 开发集:
        任务 = _环境.生成(int(s), "P1")
        行 = {}
        for 臂 in _臂列诊断:
            记录 = _算子.跑(臂, 任务, 种子=int(s))
            行[臂] = _判据.计算指标(任务, 记录)
        表[int(s)] = 行
    均 = {臂: {k: float(np.mean([表[s][臂][k] for s in 开发集])) for k in 指标键}
          for 臂 in _臂列诊断}
    逐 = {臂: {k: [float(表[s][臂][k]) for s in 开发集]
               for k in ("语义漂移率", "震荡幅度", "冲突后错误率", "复合仲裁分")}
          for 臂 in _臂列诊断}
    差 = {}
    for s in 开发集:
        A0, A1, B, S = 表[s]["A′"], 表[s]["A′_有据"], 表[s]["B′"], 表[s]["S′"]
        差[int(s)] = dict(
            A有据减A_复合=A1["复合仲裁分"] - A0["复合仲裁分"],
            A有据减A_漂移=A1["语义漂移率"] - A0["语义漂移率"],
            A有据减A_冲突错=A1["冲突后错误率"] - A0["冲突后错误率"],
            B减A有据_漂移=A1["语义漂移率"] - B["语义漂移率"],
            B减A有据_震荡=A1["震荡幅度"] - B["震荡幅度"],
            B减A有据_冲突错=A1["冲突后错误率"] - B["冲突后错误率"],
            B减A有据_复合=B["复合仲裁分"] - A1["复合仲裁分"],
            S减A有据_复合=S["复合仲裁分"] - A1["复合仲裁分"],
            S减A有据_冲突错=A1["冲突后错误率"] - S["冲突后错误率"],
        )
    汇总 = {}
    for k in 差[开发集[0]]:
        列 = [差[s][k] for s in 开发集]
        汇总[k] = dict(均值=float(np.mean(列)), 逐种子=[float(x) for x in 列],
                       最小=float(np.min(列)), 最大=float(np.max(列)),
                       正数=sum(1 for x in 列 if x > 0.0))
    return dict(臂列=list(_臂列诊断), 逐种子=表, 均值=均, 逐项=逐, 差值=差, 差值汇总=汇总,
                说明=("post-hoc 诊断（仅 dev 100–103）：A′_有据 = A′ 同构，但自省回写只对"
                      "『已收到过任务相关事件』的键生效，避免对从未写入的键写回幻读。"
                      "不替换主判据；主判据仍按原冻结口径 FAIL。"))


def 口径对照():
    """读主判据结果 JSON（只读），并列「事件锚定漂移」与「严格同轮漂移」两个口径。"""
    if not os.path.exists(结果_判据路径):
        return None
    with open(结果_判据路径, encoding="utf-8") as f:
        果 = json.load(f)
    out = {}
    全零 = True
    for 集名 in ("判据集", "留出集"):
        条 = 果["测量台"]["集"][集名]["P1_仲裁隔离_双集主判据"]
        逐 = {}
        for 臂 in ("A′", "B′", "S′"):
            列 = [float(r["主"]["指标"][臂]["语义漂移率_同轮"]) for r in 条["逐种子"]]
            主口径列 = [float(r["主"]["指标"][臂]["语义漂移率"]) for r in 条["逐种子"]]
            逐[臂] = dict(严格同轮_逐种子=列, 严格同轮_均值=float(np.mean(列)),
                          严格同轮_最大=float(np.max(列)),
                          事件锚定_均值=float(np.mean(主口径列)))
            if max(列) > 0.0:
                全零 = False
        out[集名] = 逐
    out["严格同轮三臂全零"] = bool(全零)
    out["结论"] = ("主判据 P1-1 用的是「报告 ≠ 本轮事件后权威基线」的事件锚定口径；"
                "严格按预注册字面「相对本轮事件后自身基线读」的同轮口径时，"
                "三臂在判据集/留出集所有种子上都为 0.0000（自指回写是定点：写回=读回），"
                "因此严格同轮口径下 P1-1 不能成立（0 对 0，无可检验差）⇒ 会判 FAIL。"
                "两口径并列上报；主判据判定不变。")
    return out



def _行(种子, 臂, 参数=None):
    任务 = _环境.生成(int(种子), "P1")
    记录 = _算子.跑(臂, 任务, 种子=int(种子), 参数=参数)
    return _判据.计算指标(任务, 记录)


def 主表():
    表 = {}
    for s in 开发集:
        表[int(s)] = {}
        for 臂 in ("A′", "B′", "S′"):
            表[int(s)][臂] = _行(s, 臂)
        A, B, S = 表[int(s)]["A′"], 表[int(s)]["B′"], 表[int(s)]["S′"]
        表[int(s)]["差值"] = dict(
            A减B_漂移=A["语义漂移率"] - B["语义漂移率"],
            A减B_震荡=A["震荡幅度"] - B["震荡幅度"],
            A减B_冲突错=A["冲突后错误率"] - B["冲突后错误率"],
            B减A_复合=B["复合仲裁分"] - A["复合仲裁分"],
            S减B_冲突错=S["冲突后错误率"] - B["冲突后错误率"],
            S减B_震荡=S["震荡幅度"] - B["震荡幅度"],
            S减A_复合=S["复合仲裁分"] - A["复合仲裁分"],
        )
    return 表


def 均值表(表):
    均 = {}
    for 臂 in ("A′", "B′", "S′"):
        均[臂] = {k: float(np.mean([表[s][臂][k] for s in 开发集])) for k in 指标键}
    差键 = list(表[开发集[0]]["差值"].keys())
    均["差值"] = {k: float(np.mean([表[s]["差值"][k] for s in 开发集])) for k in 差键}
    均["差值_逐种子"] = {k: [float(表[s]["差值"][k]) for s in 开发集] for k in 差键}
    均["差值_最小"] = {k: float(np.min(均["差值_逐种子"][k])) for k in 差键}
    return 均


def 自省强度扫描(强度列=(0.1, 0.5, 1.0, 2.0)):
    扫 = {}
    for 强度 in 强度列:
        行 = {k: [] for k in 指标键}
        for s in 开发集:
            指 = _行(s, "A′", 参数=dict(自省强度=float(强度)))
            for k in 指标键:
                行[k].append(float(指[k]))
        扫[str(强度)] = {k: float(np.mean(v)) for k, v in 行.items()}
    return 扫


def 主流程():
    表 = 主表()
    均 = 均值表(表)
    扫 = 自省强度扫描()
    公 = 公平基线()
    口径 = 口径对照()
    果 = dict(开发集=list(开发集), 冻结参数=dict(_算子.默认参数),
              主表=表, 均值=均, 自省强度扫描=扫,
              公平基线=公, 口径对照=口径,
              说明=("dev seed 100–103 只用于校准阈值与自省强度；判据集/留出集只判不调。"
                    "语义漂移率主口径=已就绪稳定键的报告相对本轮事件后权威基线的改变率；"
                    "同轮漂移率相对自身基线读。公平基线与口径对照是 post-hoc 诊断，"
                    "不替换主判据。"))
    return 果


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
    均 = 果["均值"]
    print("=== dev 100–103 均值（冻结参数）===")
    print("臂    漂移   同轮   震荡   冲突错  传播   延迟   复合   查询对  拒答  槽位")
    for 臂 in ("A′", "B′", "S′"):
        v = 均[臂]
        print("%-4s %6.4f %6.4f %6.4f %7.4f %6.4f %6.2f %6.4f %6.4f %5.3f %6.4f"
              % (臂, v["语义漂移率"], v["语义漂移率_同轮"], v["震荡幅度"],
                 v["冲突后错误率"], v["错误传播率"], v["更正延迟"], v["复合仲裁分"],
                 v["查询正确率"], v["干扰拒答率"], v["关键槽位正确率"]))
    print("=== dev 差值均值 / 最小 ===")
    for k in 均["差值"]:
        print("%-16s 均=%+.5f 最小=%+.5f 逐种子=%s"
              % (k, 均["差值"][k], 均["差值_最小"][k],
                 ["%.4f" % x for x in 均["差值_逐种子"][k]]))
    print("=== A′ 自省强度扫描（dev 均值）===")
    for 强, v in 果["自省强度扫描"].items():
        print("强度=%-4s 漂移=%.4f 震荡=%.4f 冲突错=%.4f 复合=%.4f 查询对=%.4f"
              % (强, v["语义漂移率"], v["震荡幅度"], v["冲突后错误率"],
                 v["复合仲裁分"], v["查询正确率"]))
    公 = 果["公平基线"]
    print("=== post-hoc 公平基线（dev 均值）===")
    print("臂        漂移   震荡   冲突错  复合   查询对")
    for 臂 in 公["臂列"]:
        v = 公["均值"][臂]
        print("%-8s %6.4f %6.4f %7.4f %6.4f %6.4f"
              % (臂, v["语义漂移率"], v["震荡幅度"], v["冲突后错误率"],
                 v["复合仲裁分"], v["查询正确率"]))
    print("--- 公平基线差值汇总（均 / 最小 / 正数）---")
    for k, v in 公["差值汇总"].items():
        print("%-18s 均=%+.5f 最小=%+.5f 正数=%d/%d 逐种子=%s"
              % (k, v["均值"], v["最小"], v["正数"], len(开发集),
                 ["%+.4f" % x for x in v["逐种子"]]))
    口径 = 果["口径对照"]
    if 口径 is not None:
        print("=== 口径对照（读主判据结果 JSON）===")
        for 集名 in ("判据集", "留出集"):
            for 臂 in ("A′", "B′", "S′"):
                v = 口径[集名][臂]
                print("%s %-3s 严格同轮 均值=%.6f 最大=%.6f ｜事件锚定 均值=%.4f"
                      % (集名, 臂, v["严格同轮_均值"], v["严格同轮_最大"], v["事件锚定_均值"]))
        print("严格同轮三臂全零 =", 口径["严格同轮三臂全零"])
    print("结果已写：", 结果路径)
