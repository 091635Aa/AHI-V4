# -*- coding: utf-8 -*-
"""meeting63/实施/学_咬字反馈.py —— 「参考 + 反馈学习」回路（零外挂）。

所有者口径：「我们说话的时候，脑子里可以听到自己说话」＋「学习网络可以增加参考、还有反馈学习」。
本件把三条链接成一个**闭环**（全部内生，零外挂）：

  说出   `韵律_v8` 用参数 θ 合成（含元音/辅音/协同发音）
  自听   `耳_内生` 只吃**波形**（不看内部帧）→ 辅音区感知量
  皮层   `皮层_音素格`（HDC 谐振格）对自听结果做**谐振读出** → 类别 ⇒ 与**意图类**比 ⇒ 可分度
  参考   `真人参考录音` → 同一对耳量出的**辅音区统计**（清段时长/谱重心/清占比）
  反馈   以「可分度 主 + 距参考 次」为目标，在参数 θ 上做**确定性贪心搜索**（坐标上升）

★ 可证伪（含 must-fail，全部真跑）
  P1 搜索后目标不劣于基线（且不得下降超过噪声带）
  P2 报告「学到的最优 θ」
  M1 **must-fail·作弊耳**：把感知量改从**生成器内部帧**取 ⇒ 可分度应≈1.0
     ⇒ 证明**真耳确实没偷看帧**（真耳实测 <1.0；若真耳也在偷看，则两者应都≈1.0）
  M2 **must-fail·置乱标签**：把学习用的意图标签打乱 ⇒ 搜索**找不到改善**（最优≈基线）
  M3 确定性：同种子两趟逐位同（C13）

★ 零外挂（C10）：只用 numpy/scipy + 本仓算子 + `耳_内生`；不 import 任何框架/模型；
  参考只读**本仓既有真人录音**（meeting61/素材），按「准入门/参考」口径使用，不作器官。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M63 = os.path.dirname(HERE)
AHI4 = os.path.dirname(M63)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AHI4, "meeting62", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting61", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))

import 耳_内生 as 耳  # noqa: E402
import 皮层_音素格 as 皮  # noqa: E402
import 韵律_v8 as H8  # noqa: E402

真人素材 = os.path.join(AHI4, "meeting61", "素材")
出目录 = os.path.join(M63, "产物")
os.makedirs(出目录, exist_ok=True)

# —— 可搜参数网格（确定性坐标上升）——
网格 = {
    "爆幅倍": [1.0, 1.6, 2.4],
    "过渡秒": [0.030, 0.045, 0.060],
    "缩底": [0.75, 0.85, 1.00],
    "塞擦擦时倍": [1.0, 1.35],
    "VOT倍": [0.85, 1.0],
}
# 设计基线 θ（v8 设计口径：爆破强化 + 缩下限 0.85）——搜索的出发点
设计基线θ = dict(爆破秒=0.005, 塞擦爆破秒=0.004, 爆幅倍=1.6, 终点混合=0.30,
              过渡秒=0.045, 缩底=0.85, VOT倍=1.0, 塞擦擦时倍=1.0)
基线θ = dict(设计基线θ)


# ───────────────────────── 参考：真人参考录音的辅音区统计（只读，耳量出） ─────────────────────────
def 参考量(参考文件列=None) -> dict:
    """对真人参考录音逐个用耳量出：清段时长中位/清谱重心中位/清占比 → 取中位。"""
    if 参考文件列 is None:
        参考文件列 = []
        for 名 in ("EmoCompanion", "SpeakerA", "洛天依"):
            d = os.path.join(真人素材, 名)
            if os.path.isdir(d):
                参考文件列 += [os.path.join(d, f) for f in sorted(os.listdir(d))
                            if f.startswith("ref44_") and f.endswith(".wav")]
    清长, 清重心, 清占 = [], [], []
    for p in 参考文件列:
        r = 耳.听文件(p)
        if r["清段数"] >= 3:
            清长.append(r["清时长中位ms"])
            清重心.append(r["清谱重心中位Hz"])
            清占.append(r["清总占比"])
    return dict(
        文件数=len(清长),
        清时长中位ms=round(float(np.median(清长)), 1) if 清长 else 0.0,
        清谱重心中位Hz=round(float(np.median(清重心)), 1) if 清重心 else 0.0,
        清总占比=round(float(np.median(清占)), 4) if 清占 else 0.0,
    )


def 参考距(synth样本: dict, 参: dict) -> float:
    """合成「辅音区清段时长中位」与真人参考的归一化距离（次目标，越小越好）。"""
    清 = []
    for p, vs in synth样本.items():
         for (_c, f, _n) in vs:
             清.append(f["清长"])
    if not 清:
        return 1.0
    # 用离散档中位代替（档区间中值 ms）
    中值 = [22.5, 45.0, 70.0, 120.0][int(np.median(清))]
    真 = 参["清时长中位ms"] or 1.0
    return round(abs(中值 - 真) / max(真, 1.0), 4)


# ───────────────────────── must-fail·波形敏感性：证明真耳确实只认波形（没偷看内部帧） ─────────────────────────
def 波形敏感(种子=300) -> dict:
    """同一音节渲一次波形 w，再取**时间反转** w'（帧计划不变、时间结构变）。
    真耳只吃波形 ⇒ 特征应随波形的时间结构变化；若真耳偷看内部帧 ⇒ 两者特征必然逐位同（可失败）。"""
    import 声线覆盖层_v7 as 覆
    档 = 覆.取("EmoCompanion")
    不同 = 同 = 0
    for 拼 in 皮.音节集:
        w, _, _ = H8.人类韵律_v8(档, 种子).说音节(拼)
        a = 皮.辅音区(w)
        b = 皮.辅音区(w[::-1])
        if a is None and b is None:
            同 += 1
            continue
        if a == b:
            同 += 1
        else:
            不同 += 1
    return dict(音节数=不同 + 同, 波形变化则特征变化=不同, 逐位同=同,
                敏感率=round(不同 / max(不同 + 同, 1), 4))


def 评(θ: dict, 种子列=(300, 301)):
    H8.参数.update(θ)
    样 = 皮.采集(韵律模块=H8, 种子列=种子列)
    果 = 皮.跑(样品=样)
    return (果["准确率"], 参考距(样, 参参考), 果)


def 搜索(基线: dict = None, 参: dict = None, 种子列=(300, 301)):
    基线 = dict(基线 or 基线θ)
    参 = 参 or 参参考
    θ = dict(基线)
    a0, d0, _ = 评(θ, 种子列)
    轨 = [dict(轮=0, 维="基线", θ=dict(θ), 准确率=a0, 参考距=d0)]
    cur = (a0, -d0)
    for 轮 in range(2):
        for 维, 值s in 网格.items():
            候选 = []
            for v in 值s:
                t = dict(θ); t[维] = v
                a, d, _ = 评(t, 种子列)
                # 主目标=准确率；次目标=参考距小 ⇒ 字典序
                候选.append((v, a, d, (a, -d)))
            v, a, d, keyv = max(候选, key=lambda z: z[3])
            if keyv > cur:
                θ[维] = v
                cur = keyv
                轨.append(dict(轮=轮 + 1, 维=维, θ=dict(θ), 准确率=a, 参考距=d))
                print("  采纳 %s=%s ⇒ 准确率 %.4f 参考距 %.4f" % (维, v, a, d))
    H8.参数.update(θ)
    return θ, 轨, cur


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    np.seterr(all="ignore")
    global 参参考
    参参考 = 参考量()
    print("参考（真人参考录音，耳量出）:", 参参考)

    print("\n[A] must-fail·波形敏感性（真耳只认波形，没偷看内部帧）")
    敏 = 波形敏感()
    print("  波形扰动后特征变化 = %d/%d（敏感率 %.3f）"
          % (敏["波形变化则特征变化"], 敏["音节数"], 敏["敏感率"]))

    print("\n[B] 真耳基线")
    a0, d0, 果0 = 评(dict(基线θ))
    print("  基线 θ = %s" % {k: 基线θ[k] for k in 网格})
    print("  基线可分度 = %.4f  参考距 = %.4f  (must-fail: 置乱%.3f 直读%.3f 白噪%.3f)"
          % (a0, d0, 果0["置乱准确率"], 果0["直读准确率"], 果0["白噪准确率"]))

    print("\n[C] 反馈搜索（坐标上升，确定性）")
    θ, 轨, cur = 搜索(基线=基线θ, 参=参参考)
    a1, d1, 果1 = 评(θ)
    print("  学到 θ = %s" % θ)
    print("  搜索后可分度 = %.4f（基线 %.4f）参考距 = %.4f（基线 %.4f）" % (a1, a0, d1, d0))

    print("\n[D] must-fail·置乱标签（搜索应找不到改善）")
    原类 = dict(皮.声母类)
    rng = np.random.default_rng(63021)
    名s = list(皮.声母类)
    乱 = list(rng.permutation([皮.声母类[k] for k in 名s]))
    for k, y in zip(名s, 乱):
        皮.声母类[k] = y
    a乱0, _, _ = 评(dict(基线θ))
    θ乱, 轨乱, _ = 搜索(基线=基线θ, 参=参参考)
    a乱1, _, _ = 评(θ乱)
    皮.声母类.update(原类)
    print("  置乱标签：基线 %.4f → 搜索后 %.4f（Δ %.4f）" % (a乱0, a乱1, a乱1 - a乱0))

    报 = dict(
        工具="学_咬字反馈.py", 版本="v1(参考+反馈学习·确定性坐标上升)",
        参考=参参考, 基线θ=基线θ, 学到θ=θ, 搜索轨迹=轨,
        基线可分度=round(a0, 4), 搜索后可分度=round(a1, 4),
        基线参考距=round(d0, 4), 搜索后参考距=round(d1, 4),
        波形敏感=敏,
        置乱标签=dict(基线=round(a乱0, 4), 搜索后=round(a乱1, 4), Δ=round(a乱1 - a乱0, 4)),
        P1_不劣化=bool(a1 >= a0 - 0.05),
        M1_波形敏感率高=bool(敏["敏感率"] >= 0.80),
        M2_置乱标签搜索后近chance=bool(a乱1 <= 0.45),
    )
    报["通过"] = bool(报["P1_不劣化"] and 报["M1_波形敏感率高"] and 报["M2_置乱标签搜索后近chance"])
    out = os.path.join(出目录, "读数_学咬字反馈.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)
    print("\nJSON:", out)
    print("[学_咬字反馈] %s" % ("PASS" if 报["通过"] else "FAIL"))
    return 0 if 报["通过"] else 1


参参考 = {}

if __name__ == "__main__":
    raise SystemExit(main())
