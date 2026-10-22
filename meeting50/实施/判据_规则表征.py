# -*- coding: utf-8 -*-
"""AHI4/meeting50/实施/判据_规则表征.py —— P0 判据：规则表征（后继表征 + 三因子）。

四臂（阈值先冻结，写死；不看结果改）：
  ① 规则重组：训练 w={0,2 地标} → 新目标 {1,3 地标}；M 不动，只改 w。
     对照：Q 基线零样本、打乱 M、随机策略。
  ② 未见组合：训练禁用 5 个起点 → 测试只用这 5 个起点。
  ③ 区室特异误差：A（自我/目标）通道错误 vs B（世界/观测）通道错误，须可分离。
  ④ 三因子消融：完整 / 常数η / 无第三因子 / 无资格迹。
静态：C1 禁词自扫、导入白名单、两趟逐位确定性。

直跑：python 判据_规则表征.py  → 写 判据_规则表征结果.json
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
本件 = os.path.abspath(__file__)
算子路径 = os.path.join(HERE, "算子_规则表征.py")
结果路径 = os.path.join(HERE, "判据_规则表征结果.json")

# ── 冻结阈值（先声明，后跑）──
已知_min = 0.90
新目标_min = 0.80
上界_min = 0.90
Q零样本_max = 0.20
SR减Q_min = 0.50
打乱M_max = 0.30
未见起点_min = 0.75
随机_max = 0.40
A误差_权重_min = 0.50
A误差_上界_min = 0.90
B误差_权重_max = 0.05
B误差_成功_max = 0.50
三因子_完整已知_min = 0.90
三因子_完整误差_max = 0.15
确定性_容差 = 0.0
允许导入 = ("__future__", "numpy", "hashlib", "json", "os", "sys", "importlib")


def _载(路径, 名):
    spec = importlib.util.spec_from_file_location(名, 路径)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _载(算子路径, "算子_规则表征_判据")


def 指纹(体, w):
    return hashlib.sha256(np.ascontiguousarray(体.M).tobytes()
                          + np.ascontiguousarray(w).tobytes()).hexdigest()


def 跑全部(种子):
    环境 = M.网格环境()
    全部 = M.全部起点()
    训练起 = M.训练起点()

    # ── ① 主臂：训练（已知目标）──
    体 = M.规则表征核(环境, 种子=种子).训练(M.w训, 起点池=训练起)
    w训后 = 体.w.copy()
    M训后 = 体.M.copy()
    已知, 已知步 = M.评估(体, 全部, w训后, M.w训)
    未见, 未见步 = M.评估(体, list(M.保留起点), w训后, M.w训)
    指纹主 = 指纹(体, w训后)

    # ── ② Q 基线（同环境/同预算/同奖赏）──
    q = M.Q基线(环境, 种子=种子).训练(M.w训, 起点池=训练起)
    q零 = q.评估(全部, M.w新)

    # ── ③ 新目标重组：M 不动，只改 w ──
    体.重配目标(正列=(1, 3), 负列=(0, 2))
    w新后 = 体.w.copy()
    新目标, 新步 = M.评估(体, 全部, w新后, M.w新)
    上界, 上界步 = M.评估(体, 全部, M.w新, M.w新)

    # ── ④ 打乱 M 对照（规则被破坏 ⇒ 必须失败）──
    打乱体 = M.规则表征核(环境, 种子=种子)
    打乱体.M = np.roll(M训后, 7, axis=0)
    打乱体.w = w新后.copy()
    打乱, _ = M.评估(打乱体, 全部, w新后, M.w新)

    # ── ⑤ 随机策略基线 ──
    随机 = M.随机基线(环境, 全部, M.w新, 种子=种子)

    # ── ⑥ 区室特异误差 ──
    rng = np.random.default_rng(int(种子) + 8801)
    噪声 = rng.standard_normal(len(M.地标))
    w_A = np.clip(w新后 + 0.80 * 噪声, -0.5, 1.5)
    A误差_权重 = float(np.linalg.norm(w_A - M.w新))
    A误差_路径, _ = M.评估(体, 全部, w_A, M.w新)
    A误差_上界, _ = M.评估(体, 全部, M.w新, M.w新)
    B误差_成功, _ = M.评估(体, 全部, w新后, M.w新, 观测置换=True)
    B误差_权重 = float(np.linalg.norm(w新后 - M.w新))

    # ── ⑦ 三因子消融 ──
    消融 = {}
    for 模式 in M.模式列:
        b = M.规则表征核(环境, 种子=种子, 模式=模式).训练(M.w训, 起点池=训练起)
        w1 = b.w.copy()
        已知b, _ = M.评估(b, 全部, w1, M.w训)
        误1 = float(np.linalg.norm(w1 - M.w训))
        b.重配目标(正列=(1, 3), 负列=(0, 2))
        新b, _ = M.评估(b, 全部, b.w, M.w新)
        误2 = float(np.linalg.norm(b.w - M.w新))
        消融[模式] = dict(已知成功率=已知b, 新目标成功率=新b,
                         训练权重误差=误1, 新目标权重误差=误2,
                         w训后=w1.tolist(), w新后=b.w.tolist())

    return dict(
        主臂=dict(已知成功率=已知, 已知平均步=已知步,
                  未见起点成功率=未见, 未见起点平均步=未见步,
                  新目标成功率=新目标, 新目标平均步=新步,
                  上界_直接给w=上界, 上界平均步=上界步,
                  w训后=w训后.tolist(), w新后=w新后.tolist(),
                  训练权重误差=float(np.linalg.norm(w训后 - M.w训)),
                  新目标权重误差=float(np.linalg.norm(w新后 - M.w新))),
        对照=dict(Q零样本=q零, 打乱M=打乱, 随机策略=随机,
                  随机减Q=float(随机 - q零)),
        区室=dict(A误差=dict(目标权重误差=A误差_权重, 路径成功率=A误差_路径, 上界=A误差_上界),
                  B误差=dict(目标权重误差=B误差_权重, 路径成功率=B误差_成功)),
        三因子=消融,
        指纹=指纹主,
    )


def 扫导入(路径):
    名 = set()
    with open(路径, encoding="utf-8") as f:
        for 行 in f.read().splitlines():
            s = 行.strip()
            if s.startswith("import "):
                名.add(s[len("import "):].split()[0].split(",")[0].split(".")[0])
            elif s.startswith("from "):
                名.add(s[len("from "):].split()[0].split(".")[0])
    return sorted(名)


def 多种子复核(种子列):
    """稳健性诊断（不入主判据）：主臂在 5 个种子上的 known/held/new/oracle/Q 零样本。"""
    行 = []
    for s in 种子列:
        环境 = M.网格环境()
        体 = M.规则表征核(环境, 种子=s).训练(M.w训, 起点池=M.训练起点())
        已知, _ = M.评估(体, M.全部起点(), 体.w, M.w训)
        未见, _ = M.评估(体, list(M.保留起点), 体.w, M.w训)
        体.重配目标(正列=(1, 3), 负列=(0, 2))
        新目标, _ = M.评估(体, M.全部起点(), 体.w, M.w新)
        上界, _ = M.评估(体, M.全部起点(), M.w新, M.w新)
        q = M.Q基线(环境, 种子=s).训练(M.w训, 起点池=M.训练起点())
        零样本 = q.评估(M.全部起点(), M.w新)
        行.append(dict(种子=int(s), 已知=已知, 未见起点=未见, 新目标=新目标,
                      上界=上界, Q零样本=零样本))
    return dict(行=行,
                已知均值=float(np.mean([r["已知"] for r in 行])),
                未见起点均值=float(np.mean([r["未见起点"] for r in 行])),
                新目标均值=float(np.mean([r["新目标"] for r in 行])),
                新目标最小=float(np.min([r["新目标"] for r in 行])),
                上界均值=float(np.mean([r["上界"] for r in 行])),
                Q零样本均值=float(np.mean([r["Q零样本"] for r in 行])))


def 主流程():
    条 = []
    臂 = {}

    def 记(名, 通过, 实测, 阈值, 说明=""):
        条.append(bool(通过))
        臂[名] = dict(实测=实测, 阈值=阈值, 通过=bool(通过), 说明=说明)

    种子 = M.种子默认
    果1 = 跑全部(种子)
    果2 = 跑全部(种子)
    主 = 果1["主臂"]

    # ① 规则重组
    条件1 = (主["已知成功率"] >= 已知_min and 主["新目标成功率"] >= 新目标_min
             and 主["上界_直接给w"] >= 上界_min and 果1["对照"]["Q零样本"] <= Q零样本_max
             and (主["新目标成功率"] - 果1["对照"]["Q零样本"]) >= SR减Q_min
             and 果1["对照"]["打乱M"] <= 打乱M_max)
    记("规则重组_新目标零样本", 条件1,
       dict(已知成功率=主["已知成功率"], 新目标成功率=主["新目标成功率"],
            新目标上界=主["上界_直接给w"], Q零样本=果1["对照"]["Q零样本"],
            SR减Q=主["新目标成功率"] - 果1["对照"]["Q零样本"],
            打乱M=果1["对照"]["打乱M"],
            w训后=主["w训后"], w新后=主["w新后"],
            训练权重误差=主["训练权重误差"], 新目标权重误差=主["新目标权重误差"]),
       dict(已知_min=已知_min, 新目标_min=新目标_min, 上界_min=上界_min,
            Q零样本_max=Q零样本_max, SR减Q_min=SR减Q_min, 打乱M_max=打乱M_max),
       "M 为后继表征（与奖赏无关），新目标只改 w；Q 基线同预算但无特征结构")

    # ② 未见起点
    条件2 = (主["未见起点成功率"] >= 未见起点_min
             and (主["未见起点成功率"] - 果1["对照"]["随机策略"]) >= 0.30)
    记("未见起点_组合泛化", 条件2,
       dict(未见起点成功率=主["未见起点成功率"], 随机策略=果1["对照"]["随机策略"],
            差=主["未见起点成功率"] - 果1["对照"]["随机策略"]),
       dict(未见起点_min=未见起点_min, 相对随机_min=0.30),
       "训练禁用 5 个起点；测试只用这 5 个")

    # ③ 区室特异误差
    A = 果1["区室"]["A误差"]
    B = 果1["区室"]["B误差"]
    条件3 = (A["目标权重误差"] >= A误差_权重_min and A["上界"] >= A误差_上界_min
             and B["目标权重误差"] <= B误差_权重_max and B["路径成功率"] <= B误差_成功_max)
    记("区室特异误差_双分离", 条件3,
       dict(A误差=A, B误差=B),
       dict(A权重误差_min=A误差_权重_min, A上界_min=A误差_上界_min,
            B权重误差_max=B误差_权重_max, B成功率_max=B误差_成功_max),
       "A 通道（自我/目标）坏 ⇒ 目标表征错、但 M 上界仍高；"
       "B 通道（世界/观测）坏 ⇒ 目标表征对、但路径执行塌")

    # ④ 三因子消融
    完 = 果1["三因子"]["完整"]
    条件4 = (完["已知成功率"] >= 三因子_完整已知_min
             and 完["训练权重误差"] <= 三因子_完整误差_max
             and 完["训练权重误差"] < 果1["三因子"]["无第三因子"]["训练权重误差"]
             and 完["训练权重误差"] < 果1["三因子"]["无资格迹"]["训练权重误差"])
    记("三因子_消融", 条件4,
       dict(各模式=果1["三因子"],
            常数η减无第三因子=abs(果1["三因子"]["常数η"]["训练权重误差"]
                                - 果1["三因子"]["无第三因子"]["训练权重误差"])),
       dict(完整已知_min=三因子_完整已知_min, 完整误差_max=三因子_完整误差_max,
            须优于=("无第三因子", "无资格迹")),
       "诊断项：若 常数η 与 无第三因子 的误差几乎相同，说明门控只是换了名字的常数")

    # ── 真接线：规则表征核继承两区室核，第八算子「符」已登记，前向走 软限(A−B) ──
    核 = M.规则表征核(M.网格环境(), D=M.D默认, 种子=种子)
    算子表 = 核.算子表()
    前 = 核.前向_规则(6)                                 # 6 = 地标0；非地标 φ=0 ⇒ 门值应为 0
    费比 = float(核.计费(4096)["乘加数"]) / max(1.0, float(核.计费(1024)["乘加数"]))
    真接通过 = bool(isinstance(核, M.两区室核) and "符" in 算子表 and len(算子表) == 8
                   and 前["输出"].shape == (M.D默认,) and 前["门值"] > 0.0
                   and 3.95 <= 费比 <= 4.05)
    记("真接线_规则表征核", 真接通过,
       dict(继承两区室核=isinstance(核, M.两区室核), 算子表=算子表, 算子数=len(算子表),
            输出形状=list(前["输出"].shape), 门值=前["门值"], 计费比=费比),
       dict(继承两区室核=True, 符已登记=True, 算子数=8, 输出形状=[M.D默认], 计费比区间=[3.95, 4.05]),
       "规则表征核是两区室核的子类；符走继承的第八算子；整合仍是 软限(A−B)；计费线性")

    # ── 静态 1：C1 禁词自扫 ──
    禁词 = list(M.禁词列())
    命中 = {}
    for p in (算子路径, 本件):
        with open(p, encoding="utf-8") as f:
            低 = f.read().lower()
        命中[os.path.basename(p)] = [w for w in 禁词 if w.lower() in 低]
    命中总数 = sum(len(v) for v in 命中.values())
    记("静态_C1禁词自扫", 命中总数 == 0, dict(命中=命中, 命中总数=命中总数),
       dict(命中总数=0), "复用 能力核.禁词列")

    # ── 静态 2：导入白名单 ──
    导入表 = {os.path.basename(p): 扫导入(p) for p in (算子路径, 本件)}
    越界 = {k: [m for m in v if m not in 允许导入] for k, v in 导入表.items()}
    越界非空 = {k: v for k, v in 越界.items() if v}
    记("静态_导入白名单", not 越界非空, dict(导入表=导入表, 越界=越界非空),
       dict(白名单=list(允许导入)), "只允许 numpy/hashlib/json/os/sys/importlib/__future__")

    # ── 静态 3：两趟逐位确定性 ──
    决 = bool(果1["指纹"] == 果2["指纹"]
             and abs(主["新目标成功率"] - 果2["主臂"]["新目标成功率"]) <= 确定性_容差
             and abs(主["已知成功率"] - 果2["主臂"]["已知成功率"]) <= 确定性_容差)
    记("静态_两趟确定性", 决,
       dict(指纹=[果1["指纹"], 果2["指纹"]],
            Acc甲=[主["已知成功率"], 主["新目标成功率"]],
            Acc乙=[果2["主臂"]["已知成功率"], 果2["主臂"]["新目标成功率"]]),
       dict(逐位同=True), "同种子两遍：M/w 指纹与关键 Acc 逐位同")

    未过 = [名 for 名, v in 臂.items() if not v["通过"]]
    果 = dict(通过=bool(all(条)), 判定条数=len(条), 未过条数=len(未过), 未过臂=未过,
              阈值=dict(已知_min=已知_min, 新目标_min=新目标_min, 上界_min=上界_min,
                        Q零样本_max=Q零样本_max, SR减Q_min=SR减Q_min, 打乱M_max=打乱M_max,
                        未见起点_min=未见起点_min, A误差_权重_min=A误差_权重_min,
                        A误差_上界_min=A误差_上界_min, B误差_权重_max=B误差_权重_max,
                        B误差_成功_max=B误差_成功_max,
                        三因子_完整已知_min=三因子_完整已知_min,
                        三因子_完整误差_max=三因子_完整误差_max),
              臂=臂,
              多种子复核=多种子复核(tuple(range(种子, 种子 + 5))),
              说明=("训练 300 回合；M 为后继表征（与奖赏无关），w 由三因子门控学习。"
                   "新目标只改 w、不动 M。Q 基线同环境/同预算/同奖赏，但没有特征结构。"
                   "random 只来自 default_rng(显式种子)。"))
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
    print(json.dumps(果["臂"], ensure_ascii=False, indent=1, default=str)[:6000])
    print("[判据_规则表征] 判定 %d 条，未过 %d 条 ⇒ %s"
          % (果["判定条数"], 果["未过条数"], "PASS" if 果["通过"] else "FAIL"))
    print("未过臂：", json.dumps(果["未过臂"], ensure_ascii=False))
    print("结果已写：", 结果路径)
    raise SystemExit(0 if 果["通过"] else 1)
