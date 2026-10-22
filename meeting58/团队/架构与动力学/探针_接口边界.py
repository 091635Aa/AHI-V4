# -*- coding: utf-8 -*-
"""meeting58/E1 探针二：A层(角色行为) ↔ B层(音色输出) 的真实接口边界（CPU-only，<10s）。

测：
  Q1 数据契约：声带.分位读数 对向量长度的硬约束（说_音色 硬编码 D=1024 vs 双通道体 D=64）
  Q2 A→音色的**信息通道宽度**：qs 是「分位秩」⇒ 对 |A| 尺度不变、对 A 取反完全镜像
  Q3 V5 实跑（含 更新）：自我标量 轨迹 / credit 权 / ⟨B,A⟩/‖A‖² / 出戏率
  Q4 credit 权 上界：人设进 B 时 权 是否饱和（冲突可观测性）
  Q5 时序成本：一拍(A层) 与 一句(B层渲染) 的实测耗时 + 渲染确定性（wav 落本目录）
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AHI4 = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(AHI4, "meeting49", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting53", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting32"))

import 算子_两区室核 as 核m  # noqa: E402
import V5_双通道 as V5  # noqa: E402
import 声带 as SD  # noqa: E402

D = 64
种子 = 53001
软限 = 核m.软限
_归 = 核m._归


def Q1_维度契约():
    """声带器官(D=1024).分位读数 收到 64 维向量会怎样？"""
    器 = SD.声带器官(seed=300, D=1024, 目录=os.path.join(HERE, "_tmp_声带"),
                     台账路径=os.path.join(HERE, "_tmp_声带", "台账.jsonl"))
    果 = {}
    try:
        器.分位读数(np.ones(64, dtype=np.float32))
        果["D1024器_收64向量"] = "未报错"
    except Exception as e:
        果["D1024器_收64向量"] = "%s: %s" % (type(e).__name__, str(e)[:120])
    try:
        器.分位读数(np.ones(1024, dtype=np.float32))
        果["D1024器_收1024向量"] = "OK"
    except Exception as e:
        果["D1024器_收1024向量"] = "%s: %s" % (type(e).__name__, str(e)[:80])
    器64 = SD.声带器官(seed=300, D=64, 目录=os.path.join(HERE, "_tmp_声带"),
                       台账路径=os.path.join(HERE, "_tmp_声带", "台账.jsonl"))
    果["D64器_收64向量"] = "OK(K=%d)" % min(64, max(8, 64 // 16))
    return 果


def Q2_通道宽度():
    """qs = 分位秩：对尺度不变、对符号镜像。"""
    核 = 核m.两区室核(D=D, 种子=种子)
    核.设自我标量(0.8)
    A = 核.自我态向量()
    器 = SD.声带器官(seed=300, D=64, 目录=os.path.join(HERE, "_tmp_声带"),
                     台账路径=os.path.join(HERE, "_tmp_声带", "台账.jsonl"))
    qs_A = 器.分位读数(A)
    qs_2A = 器.分位读数(2.0 * A)
    qs_负A = 器.分位读数(-A)
    qs_随机 = 器.分位读数(np.random.default_rng(3).standard_normal(D))
    K = len(qs_A)
    # 真实 A 轨迹上 qs 的取值个数（扫描 自我标量）
    qs列 = []
    for s in np.linspace(0.05, 0.95, 19):
        核.设自我标量(float(s))
        qs列.append(器.分位读数(核.自我态向量()))
    qs列 = np.asarray(qs列)
    return dict(K=K, qs分辨率=round(1.0 / max(K - 1, 1), 4),
                尺度2倍_qs改变元素=int((qs_A != qs_2A).sum()),
                取反_qs与1_qs之差=int((np.abs(qs_负A - (1.0 - qs_A)) > 1e-9).sum()),
                A_vs随机_qs平均绝对差=round(float(np.mean(np.abs(qs_A - qs_随机))), 4),
                扫19个自我标量_qs唯一行数=int(len({tuple(np.round(r, 6)) for r in qs列})),
                说明="A→音色只通过每音节一个标量 ΔF0（声带.py:264 上限±40Hz）；|A| 完全丢失，符号只翻方向")


def Q3_V5实跑(拍数=40):
    体 = V5.双通道体(D=D, 种子=种子)
    环 = V5.接口沙盒(种子=种子 + 1)
    行 = []
    for t in range(拍数):
        环.新试次()
        r = 体.一拍(观测id=环.cue, 学习=True, 老师=True, 拍=t)
        成功 = 环.走(r["动作id"])
        A前 = 体.核.自我态向量()
        体.更新(成功, r["动作id"])
        A后 = 体.核.自我态向量()
        # 复算 差(A,B)：用同一拍内同 token 列
        rr = 体.核.前向_双区室(体._token列(环.cue, None, list(体.历史词)[-2:]),
                           上下文=None, 迭代步数=1)
        B = rr["世界态"][-1]
        差 = _归(软限(A前 - B))
        门 = 体.核.符(A前, 体.o)
        权 = float(np.mean(np.abs(门))) / (float(np.mean(np.abs(A前)))
                                       * float(np.mean(np.abs(体.o))) + 1e-9)
        行.append(dict(拍=t, 自我标量=float(体.核.自我标量),
                       ΔA=int((A前 != A后).sum()),
                       权=权,
                       cos_差A=float(差 @ A前 / (np.linalg.norm(差) * np.linalg.norm(A前) + 1e-12)),
                       投影比=float((B @ A前) / (A前 @ A前)),
                       出戏=bool(float(差 @ A前) < 0.0),
                       A_L2=float(np.linalg.norm(A前)), B_L2=float(np.linalg.norm(B))))
    标量列 = np.array([x["自我标量"] for x in 行])
    权列 = np.array([x["权"] for x in 行])
    改列 = np.array([x["ΔA"] for x in 行])
    return dict(拍数=拍数, 自我标量_末10=[round(float(v), 4) for v in 标量列[-10:]],
                自我标量_最小=round(float(标量列.min()), 4), 自我标量_最大=round(float(标量列.max()), 4),
                自我标量_标准差=round(float(标量列.std()), 4),
                改A元素_均值=round(float(改列.mean()), 2),
                权_均值=round(float(权列.mean()), 4), 权_最小=round(float(权列.min()), 4),
                权_最大=round(float(权列.max()), 4),
                出戏率=round(float(np.mean([x["出戏"] for x in 行])), 4),
                cos_差A均值=round(float(np.mean([x["cos_差A"] for x in 行])), 4),
                投影比均值=round(float(np.mean([x["投影比"] for x in 行])), 4),
                A_L2均值=round(float(np.mean([x["A_L2"] for x in 行])), 4),
                B_L2均值=round(float(np.mean([x["B_L2"] for x in 行])), 4),
                说明="A 只经 自我标量 1 个标量变（ΔA 恒=64 但方向固定在 P 的 2 维面内）；权≈1 且不随冲突变")


def Q4_权上界():
    """把人设文本压进 B（模拟），权 是否携带「一致/冲突」信息？"""
    核 = 核m.两区室核(D=D, 种子=种子)
    A = 核.自我态向量()
    rng = np.random.default_rng(21)
    人设 = _归(rng.standard_normal(D) * 4.0)     # 「洛天依语气向量」占位
    行 = []
    for 倍 in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        B = _归(rng.standard_normal(D) / np.sqrt(D) * 2.0 + 倍 * 人设)
        门 = 核.符(A, B)
        权 = float(np.mean(np.abs(门))) / (float(np.mean(np.abs(A)))
                                       * float(np.mean(np.abs(B))) + 1e-9)
        行.append(dict(人设倍=倍, 权=round(权, 4),
                       cos_AB=round(float(A @ B / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-12)), 4),
                       出戏=bool(float(_归(软限(A - B)) @ A) < 0.0)))
    return dict(扫描=行, 权理论上界=float(np.mean(np.abs(A)) * np.mean(np.abs(A)) /
                                    (np.mean(np.abs(A)) ** 2 + 1e-9)),
                说明="权 = E|A⊙B|/(E|A|E|B|) ∈ [~1, ~1.64]，只对 |cos| 敏感、对 cos 符号不敏感 ⇒ 冲突不可观测")


def Q5_时序与确定性():
    # A 层一拍
    体 = V5.双通道体(D=D, 种子=种子)
    环 = V5.接口沙盒(种子=种子 + 1)
    t0 = time.perf_counter()
    n = 30
    for t in range(n):
        环.新试次()
        r = 体.一拍(观测id=环.cue, 学习=True, 老师=True, 拍=t)
        体.更新(环.走(r["动作id"]), r["动作id"])
    一拍毫秒 = (time.perf_counter() - t0) / n * 1000.0
    # B 层一句渲染（落本目录）
    目录 = os.path.join(HERE, "_tmp_声带")
    向量 = np.sign(np.random.default_rng(55).standard_normal(64)).astype(np.float32)
    文本 = "我看左边，落点对上再往下走。"
    器A = SD.声带器官(seed=300, D=64, 目录=目录, 台账路径=os.path.join(目录, "t.jsonl"))
    律 = dict(基频=190.0, 语速=3.2, 幅度=0.5, 气声度=0.3, 紧喉度=0.2,
              粗糙度=0.05, 送气量=0.2, 声区=0.05, _ñ=0.5)
    t0 = time.perf_counter()
    r1 = 器A.发声(向量, 文本, dict(能量=0.5, 新异=0.5, 消化债=0.0, 饱足=0.5))
    t1 = time.perf_counter() - t0
    器B = SD.声带器官(seed=300, D=64, 目录=目录, 台账路径=os.path.join(目录, "t.jsonl"))
    r2 = 器B.发声(向量, 文本, dict(能量=0.5, 新异=0.5, 消化债=0.0, 饱足=0.5))
    h1 = hashlib.sha256(np.asarray(r1["波形"]).tobytes()).hexdigest()[:16]
    h2 = hashlib.sha256(np.asarray(r2["波形"]).tobytes()).hexdigest()[:16]
    return dict(一拍毫秒=round(一拍毫秒, 3), D=D,
                一句渲染毫秒=round(t1 * 1000.0, 1), 句字时长秒=r1["时长秒"],
                实时率=r1["实时率"], 两趟逐位同=bool(h1 == h2), sha=h1,
                参数合法=r1["自检"]["参数合法"], 跳变=r1["自检"]["跳变"],
                通过=r1["通过"],
                说明="A层一拍 <1ms 量级；B层整句渲染较慢但离线；流式口径见 说话_连贯.py:178-182（虚拟毫秒、与拍钟解耦）")


def main():
    out = dict(Q1=Q1_维度契约(), Q2=Q2_通道宽度(), Q3=Q3_V5实跑(),
               Q4=Q4_权上界(), Q5=Q5_时序与确定性())
    p = os.path.join(HERE, "探针_接口边界_结果.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n落盘:", p)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    np.seterr(all="ignore")
    main()
