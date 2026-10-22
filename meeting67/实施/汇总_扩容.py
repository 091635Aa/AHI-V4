# -*- coding: utf-8 -*-
"""meeting67/实施/汇总_扩容.py —— 数据扩容 v13 结果汇总（按 15 预注册判据；只读 JSON）。

比较：A1（合并数据）vs A0（既有 2.77h），e0150 报告点；Δ = A1 − A0（负=改善）。
判据（15 §4）：
  有效（主）  ：A1 双集均低于 A0，且至少一集下降 >0.03，另一集不升 >0.01；
  强数据效应  ：A1 双集均下降 >0.10；
  部分有效    ：仅一集下降 >0.03，另一集不升 >0.01（且不满足主判据）；
  无效        ：其余（无下降，或任一集升 >0.02）。
护栏：D-1 阳性对照 ≤0.10；D-2 sham ≥0.9；D-3 两趟一致；D-4 e0150 存在/有限；D-6 轮数 ≥100。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

根 = Path(r"<workspace>/03_AHI\AHI4\meeting67\产物\扩容_v13")
臂s = ("A0", "A1")


def 读(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def 取(名, 点="e0150"):
    判 = 读(根 / 名 / f"判据_扩容_{名}_{点}.json")
    if not 判:
        return None
    双 = 判.get("双集", {})
    历 = 双.get("历史", {})
    留 = 双.get("留出", {})
    训 = 读(根 / 名 / "训练曲线.json") or {}
    信 = 训.get("训练信息", {})
    阳 = 判.get("仪器阳性对照", {})
    sham = 判.get("sham", {})
    return dict(
        历史=历.get("CER均值"), 留出=留.get("CER均值"),
        两趟一致=bool(历.get("两趟一致")) and bool(留.get("两趟一致")),
        阳性=阳.get("CER"), sham白噪=sham.get("白噪"), sham静音=sham.get("静音"),
        轮=信.get("轮"), 训练秒=信.get("耗时秒"), 完成=信.get("完成"),
        设备=信.get("设备"), 时间=信.get("时间戳"),
        训练样本数=信.get("训练样本数"), 验证样本数=信.get("验证样本数"),
    )


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    报 = {"预注册": "meeting67/15", "报告点": "ckpt_e0150；推理 4轮/阻尼0.5；S1–S12 只报一次"}
    d = {}
    for 名 in 臂s:
        x = 取(名)
        d[名] = x
        报[名] = x
    if not d.get("A0") or not d.get("A1"):
        报["判定"] = {"结论": "未完成（缺 A0 或 A1 的 e0150 判据 JSON）"}
        (根 / "扩容结果.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
        print("未完成：缺读数")
        return 0
    Δ历 = d["A1"]["历史"] - d["A0"]["历史"]
    Δ留 = d["A1"]["留出"] - d["A0"]["留出"]
    Δs = {"Δ历史": Δ历, "Δ留出": Δ留}
    有效 = (Δ历 < 0 and Δ留 < 0 and min(Δ历, Δ留) < -0.03 and max(Δ历, Δ留) <= 0.01)
    强 = (Δ历 <= -0.10 and Δ留 <= -0.10)
    部分 = ((Δ历 < -0.03 and Δ留 <= 0.01) or (Δ留 < -0.03 and Δ历 <= 0.01)) and not 有效
    无效 = not (有效 or 部分)
    if 强:
        判定 = "强数据效应：A1 双集均下降 >0.10 ⇒ 数据量是主瓶颈（与 meeting66 P0 同向）"
    elif 有效:
        判定 = "数据有效：A1 双集均低于 A0，至少一集下降 >0.03，另一集不升 >0.01"
    elif 部分:
        判定 = "部分有效：仅一集下降 >0.03，另一集不升 >0.01（不宣称）"
    else:
        判定 = "无效：无下降或任一集升 >0.02 ⇒ 数据不是唯一瓶颈（与 V 点一致）"
    护栏 = {}
    for 名 in 臂s:
        x = d[名]
        护栏[名] = {
            "D-1 阳性≤0.10": (x["阳性"] is not None and x["阳性"] <= 0.10),
            "D-2 sham≥0.9": (x["sham白噪"] is not None and x["sham白噪"] >= 0.9 and x["sham静音"] >= 0.9),
            "D-3 两趟一致": x["两趟一致"],
            "D-4 训练完成": bool(x["完成"]),
            "D-6 轮数≥100": (x["轮"] is not None and x["轮"] >= 100),
        }
        护栏[名]["全过"] = all(护栏[名].values())
    报["Δ"] = Δs
    报["判定"] = {"结论": 判定, "判据": dict(有效=有效, 强=强, 部分=部分, 无效=无效)}
    报["护栏"] = 护栏
    (根 / "扩容结果.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print("=" * 78)
    print("臂 | 历史 CER | 留出 CER | 轮 | 训练秒 | 阳性 | sham | 护栏全过")
    for 名 in 臂s:
        x = d[名]
        print(f"{名} | {x['历史']:.4f} | {x['留出']:.4f} | {x['轮']} | {x['训练秒']:.0f} | {x['阳性']} | {x['sham白噪']} | {护栏[名]['全过']}")
    print(f"Δ(A1-A0)：历史 {Δ历:+.4f} ｜ 留出 {Δ留:+.4f}")
    print("判定：", 判定)
    print("已写：", 根 / "扩容结果.json")
    return 0


if __name__ == "__main__":
    sys.exit(主())
