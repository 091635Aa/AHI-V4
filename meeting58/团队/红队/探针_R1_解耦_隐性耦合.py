# -*- coding: utf-8 -*-
"""meeting58 · 红队探针1：攻击「解耦」——把最可能被误认为解耦的隐性耦合量出来。

CPU-only，只读 import 既有件（不改），产物落本目录 产物/解耦/。目标 ≤10s。
探的六件事：
  D1 音色档案 = 封闭 4 档；未知音色名静默回落「女声-明亮」⇒ 换角色=换字符串，音色不变
  D2 状态向量→声音 的真实通道：只经 分位读数→ΔF0；维度不匹配（V5 的 D=64 进不来 D=1024）
  D3 遥测（能量/新异/消化债/饱足）在 `说` 路径是否死掉（子类覆盖 韵律）
  D4 音色档案里混进「语速」⇒ 音色与时长共线；单参数（F0）就能 4/4 分档
  D5 单例 `_器` + WAV上限=32 ⇒ 跨角色 FIFO 互相删文件
  D6 基线F0（seed 随机标量）在 `说` 路径是否被绕过（同一 seed 的隐性音色底）
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # …/meeting58
AHI4 = os.path.dirname(ROOT)                           # …/AHI4
sys.path.insert(0, os.path.join(AHI4, "meeting55", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting32"))

import 说_音色 as SY  # noqa: E402

产物 = os.path.join(HERE, "产物", "解耦")
os.makedirs(产物, exist_ok=True)
SY.产物 = 产物
SY.音色目录 = os.path.join(产物, "音色档案")
os.makedirs(SY.音色目录, exist_ok=True)

文本 = "你好，我是天依。今天想听我唱歌吗？"
B = lambda x: hashlib.sha256(np.asarray(x, dtype=np.float32).tobytes()).hexdigest()[:16]


def 音色头(rec: dict) -> dict:
    r = SY.频谱特征(os.path.join(SY.产物, "发声", rec["wav"]))
    return dict(wav=rec["wav"], 波形sha=B(rec["波形"]), F0中位=r["F0均值"],
                时长=r["时长"], 能量=r["能量"])


def D1_未知音色回落() -> dict:
    参默认 = SY.载档案("女声-明亮")
    参未知 = SY.载档案("洛天依")          # 没有这个档案
    参大小写 = SY.载档案("女声 明亮")      # 近似名也不认
    r默认 = SY.说(文本, 音色="女声-明亮", 情绪="平静")
    r未知 = SY.说(文本, 音色="洛天依", 情绪="平静")
    r名字乱写 = SY.说(文本, 音色="洛天依-官方-真声克隆", 情绪="平静")
    return dict(未知名读数=参未知, 与女声明亮参数相同=bool(参未知 == 参默认),
                近似名是否回落=bool(参大小写 == 参默认),
                wav逐位同_默认vs未知=bool(音色头(r默认)["波形sha"] == 音色头(r未知)["波形sha"]),
                三段sha=[音色头(r默认)["波形sha"], 音色头(r未知)["波形sha"],
                        音色头(r名字乱写)["波形sha"]],
                结论="音色档案是封闭 4 档白名单；换角色名/换角色包/写错名字都不换声音，只换档案字符串才换")


def D2_状态向量通道() -> dict:
    器 = SY.取器()
    D = 器.D
    列 = {}
    for 名, v in (
        ("全零(D=1024)", np.zeros(D)),
        ("默认随机(seed55)", None),
        ("全+1", np.ones(D)),
        ("前32+1后-1", np.concatenate([np.ones(D // 2), -np.ones(D // 2)])),
    ):
        rec = SY.说(文本, 音色="女声-明亮", 情绪="平静", 向量=v)
        列[名] = 音色头(rec)
    # 维度不匹配：把 V5 的真实状态（D=64）直接喂进来
    错 = None
    try:
        SY.说(文本, 音色="女声-明亮", 情绪="平静", 向量=np.zeros(64))
        错 = "未报错（意外）"
    except Exception as e:
        错 = "%s: %s" % (type(e).__name__, str(e)[:120])
    # 状态→F0 的可动幅度：对 4 种向量取 音节帧 的 ΔF0 范围
    律 = SY.情绪律(SY.载档案("女声-明亮"), "平静")
    律["_ñ"] = 0.5
    ΔF0 = {}
    for 名, v in (("全零", np.zeros(D)), ("默认随机(seed55)", None), ("全+1", np.ones(D))):
        if v is None:
            v = np.sign(np.random.default_rng(55).standard_normal(D)).astype(np.float32)
        qs = 器.分位读数(v)
        ΔF0[名] = dict(q区间=[round(float(np.min(qs)), 3), round(float(np.max(qs)), 3)],
                        ΔF0区间Hz=[round(24.0 * (float(np.min(qs)) - 0.5) * 2 * 1.5, 1),
                                  round(24.0 * (float(np.max(qs)) - 0.5) * 2 * 1.5, 1)])
    return dict(四种状态向量=列, V5的D64状态直喂=错,
                **{"状态→ΔF0": ΔF0}, 器D=器.D,
                结论="状态只经 分位读数 改 ΔF0（≤±36Hz，与男女声差同级）；且 D=64 的状态向量直接喂会崩 ⇒ 两层之间没有维度契约")


def D3_遥测是否死() -> dict:
    器 = SY.声带发声(seed=300, D=1024)
    器.当前律 = SY.情绪律(SY.载档案("女声-明亮"), "平静")
    v = np.sign(np.random.default_rng(55).standard_normal(1024)).astype(np.float32)
    基 = dict(能量=0.5, 新异=0.5, 消化债=0.0, 饱足=0.5)
    他 = dict(能量=1.0, 新异=0.5, 消化债=1551.0, 饱足=1.0)
    新 = dict(能量=0.5, 新异=1.0, 消化债=0.0, 饱足=0.5)
    零 = dict(能量=0.5, 新异=0.0, 消化债=0.0, 饱足=0.5)
    ra = 器.发声(v, 文本, 基)
    rb = 器.发声(v, 文本, 他)
    rc = 器.发声(v, 文本, 新)
    rd = 器.发声(v, 文本, 零)
    return dict(只改能量/债/饱=dict(sha_a=B(ra["波形"]), sha_b=B(rb["波形"]),
                                    逐位同=bool(B(ra["波形"]) == B(rb["波形"]))),
                只改新异=dict(sha_基=B(ra["波形"]), sha_新=B(rc["波形"]), sha_零=B(rd["波形"]),
                              新异0vs1逐位同=bool(B(rd["波形"]) == B(rc["波形"]))),
                说路径固定遥测="说_音色.py:112 硬编码 {能量:0.5, 新异:0.5, 消化债:0, 饱足:0.5}",
                结论="`说` 路径上 遥测 只剩『新异→_ñ』一条微扰通道；能量/消化债/饱足全死（子类覆盖 韵律 直接返回 当前律）。⇒ 情绪/能量不经过 AHI 状态进入声音，音色层与核心零耦合；唯一状态入口是 向量→分位读数")


def D4_语速共线() -> dict:
    行 = {}
    for 名 in SY.音色档案:
        r = SY.说(文本, 音色=名, 情绪="平静")
        行[名] = 音色头(r)
    for 情 in ("高兴", "低落", "急促"):
        r = SY.说(文本, 音色="女声-明亮", 情绪=情)
        行["女声-明亮|" + 情] = 音色头(r)
    F0s = [行[k]["F0中位"] for k in SY.音色档案]
    时长s = [行[k]["时长"] for k in SY.音色档案]
    return dict(行=行,
                四档F0中位=F0s, 最小F0间隔=round(min(abs(F0s[i] - F0s[j])
                                                for i in range(4) for j in range(i + 1, 4)), 1),
                四档时长=时长s, 时长比=round(max(时长s) / min(时长s), 3),
                情绪时长比=round(max(行["女声-明亮|高兴"]["时长"], 行["女声-明亮|低落"]["时长"]) /
                               min(行["女声-明亮|高兴"]["时长"], 行["女声-明亮|低落"]["时长"]), 3),
                结论="档案里含 语速 ⇒ 「音色」与「句长」共线（24%），情绪又改语速（30%）⇒ 任何用时长/能量判音色的指标都能被语速/情绪冒充；F0 单参数即可 4/4 分档")


def D5_单例与WAV删除() -> dict:
    器a = SY.取器()
    器b = SY.取器()
    # 交替两个角色，超过 WAV上限(=32) 后先录的 wav 是否被删
    件 = []
    短文本 = "啊"
    for i in range(34):
        名 = "女声-明亮" if i % 2 == 0 else "男声-低沉"
        rec = SY.说(短文本, 音色=名, 情绪="平静")
        件.append(os.path.join(SY.产物, "发声", rec["wav"]))
    存在 = [os.path.exists(p) for p in 件]
    目录 = os.path.join(SY.产物, "发声")
    实际文件 = sorted(os.listdir(目录))
    return dict(同一器=bool(器a is 器b),
                生成数=34, WAV上限=32,
                前2个是否被删=存在[:2], 最后2个是否存在=存在[-2:],
                实际文件数=len([f for f in 实际文件 if f.endswith(".wav")]),
                结论="`_器` 单例 + FIFO 32：换角色不重启，但角色 B 的发声会删掉角色 A 的产物文件 ⇒ 回放/复现/审计按角色隔离失效")


def D6_基线F0() -> dict:
    列 = {}
    for seed in (300, 301, 302):
        器 = SY.声带发声(seed=seed, D=1024)
        列["seed=%d" % seed] = round(float(器.基线F0), 2)
    器 = SY.声带发声(seed=300, D=1024)
    器.当前律 = SY.情绪律(SY.载档案("女声-明亮"), "平静")
    v = np.ones(1024)
    rec = 器.发声(v, "啊", dict(能量=0.5, 新异=0.5, 消化债=0.0, 饱足=0.5))
    rec_low = None
    return dict(不同seed的基线F0=列,
                说路径实际基频=round(float(rec["参数"]["基频"]), 1),
                说明="基线F0 只在主路 韵律() 里用；`说` 走 声带发声 子类覆盖 ⇒ seed 的随机音色底被绕过；若混用主路/子类，音色底来自 seed 而不是角色（隐性耦合）",
                结论="同一 seed 的基线F0 是隐藏音色底：主路测试会带上它，子类测试不会——两套口径不可比")


def main():
    t0 = time.perf_counter()
    出 = dict(耗时秒=None,
              D1=D1_未知音色回落(), D2=D2_状态向量通道(), D3=D3_遥测是否死(),
              D4=D4_语速共线(), D5_单例与WAV删除=D5_单例与WAV删除(), D6=D6_基线F0())
    出["耗时秒"] = round(time.perf_counter() - t0, 3)
    p = os.path.join(HERE, "探针_R1_解耦_结果.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(出, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: v for k, v in 出.items() if k in ("耗时秒",)},
                     ensure_ascii=False))
    for k in ("D1", "D2", "D3", "D4", "D5_单例与WAV删除", "D6"):
        v = 出[k]
        print("=" * 20, k)
        print(json.dumps({kk: vv for kk, vv in v.items() if kk not in ("行", "四种状态向量")},
                         ensure_ascii=False)[:1500])
    print("落盘:", p)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    np.seterr(all="ignore")
    main()
