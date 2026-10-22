# meeting67 / 实施 / 汇总_Vv2.py
# 用途：把 V-v2 四档判据 JSON + 训练曲线 JSON 汇总为 产物/V曲线_v2/V曲线结果.json，并打印表/判读。
# 预注册：meeting67/04 §7（判据先定后跑）。
# 只读现有 JSON；不改任何读数。
import json
import math
import sys
from pathlib import Path

根 = Path(r"<workspace>/03_AHI\AHI4\meeting67\产物\V曲线_v2")
数据分钟 = {"D1": 15.0, "D2": 30.0, "D3": 83.0, "D4": 162.0}
子集名 = {"D1": "子集_D1_le15min", "D2": "子集_D2_le30min",
         "D3": "子集_D3_1-2_83min", "D4": "子集_D4_全量"}


def 读(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def 斜率(xs, ys):
    """最小二乘 y = a + b x；返回 (b, r2)。"""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else float("nan")
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return b, r2


def 主():
    档 = {}
    for tag in ("D1", "D2", "D3", "D4"):
        判 = 读(根 / tag / f"判据_V_v2_{tag}.json")
        训 = 读(根 / tag / "训练曲线.json")
        if not 判 or not 训:
            档[tag] = {"状态": "缺读数", "判据存在": bool(判), "训练曲线存在": bool(训)}
            continue
        双 = 判.get("双集", {})
        历 = 双.get("历史", {})
        留 = 双.get("留出", {})
        信 = 训.get("训练信息", {})
        阳 = 判.get("仪器阳性对照", {})
        sham = 判.get("sham", {})
        自检 = 判.get("自检", {})
        档[tag] = {
            "数据分钟": 数据分钟[tag],
            "训练轮数": 信.get("轮"),
            "训练耗时秒": 信.get("耗时秒"),
            "提前停": 信.get("提前停"),
            "完成": 信.get("完成"),
            "历史CER": 历.get("CER均值"),
            "留出CER": 留.get("CER均值"),
            "双集均值": None if 历.get("CER均值") is None or 留.get("CER均值") is None
            else (历["CER均值"] + 留["CER均值"]) / 2,
            "两趟一致": bool(历.get("两趟一致")) and bool(留.get("两趟一致")),
            "阳性对照CER": 阳.get("CER"),
            "sham白噪": sham.get("白噪"),
            "sham静音": sham.get("静音"),
            "自检PASS": 自检.get("PASS"),
            "脚本sha256": 信.get("脚本sha256"),
            "设备": 信.get("设备"),
            "时间戳": 信.get("时间戳"),
        }
        # 护栏 V-1..V-6（可自动判定的部分）
        护栏 = {}
        if 档[tag]["阳性对照CER"] is not None:
            护栏["V-1 阳性对照≤0.10"] = 档[tag]["阳性对照CER"] <= 0.10
        if 档[tag]["sham白噪"] is not None:
            护栏["V-2 sham≥0.9"] = (档[tag]["sham白噪"] >= 0.9) and (档[tag]["sham静音"] >= 0.9)
        if 档[tag]["两趟一致"] is not None:
            护栏["V-3 两趟逐字一致"] = 档[tag]["两趟一致"]
        if 档[tag]["训练轮数"] is not None:
            护栏["V-4 轮数≥100"] = 档[tag]["训练轮数"] >= 100
        护栏["V-5 子集索引存在"] = (根.parents[1] / "数据" / 子集名[tag] / "meta.json").exists()
        if 档[tag]["自检PASS"] is not None:
            护栏["V-6 确定性/自检"] = bool(档[tag]["自检PASS"])
        档[tag]["护栏"] = 护栏
        档[tag]["护栏全过"] = all(护栏.values()) if 护栏 else False

    有 = [t for t in ("D1", "D2", "D3", "D4") if 档.get(t, {}).get("历史CER") is not None]
    结果 = {"预注册": "meeting67/04 §7", "生成时间": "", "档": 档, "可用档": 有}
    if 有:
        xs = [math.log2(档[t]["数据分钟"]) for t in 有]
        ys历 = [档[t]["历史CER"] for t in 有]
        ys留 = [档[t]["留出CER"] for t in 有]
        b历, r2历 = 斜率(xs, ys历)
        b留, r2留 = 斜率(xs, ys留)
        结果["斜率"] = {"历史": {"b_per_doubling": b历, "r2": r2历},
                       "留出": {"b_per_doubling": b留, "r2": r2留}}
        if "D4" in 有:
            基历, 基留 = 档["D4"]["历史CER"], 档["D4"]["留出CER"]
            knee = None
            for t in 有:
                if abs(档[t]["历史CER"] - 基历) <= 0.05 and abs(档[t]["留出CER"] - 基留) <= 0.05:
                    knee = t
                    break
            结果["knee"] = {"对D4差≤0.05的最小档": knee,
                            "D4历史": 基历, "D4留出": 基留}
        bmax = max(abs(b历), abs(b留))
        if bmax < 0.05:
            结论 = "数据不敏感（8 min 固定墙钟下）：斜率 |b|<0.05/倍 ⇒ 「最小数据」命题在该算力下成立"
        elif bmax >= 0.10:
            结论 = "数据敏感：斜率 |b|≥0.10/倍 ⇒ 数据量仍是主瓶颈（与 meeting66 P0 同向）"
        else:
            结论 = "弱/中间敏感（0.05≤|b|<0.10）：不足以单独判定，按 knee 登记"
        结果["判定"] = {"结论": 结论, "bmax": bmax}
    # 写盘
    结果["生成时间"] = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    出 = 根 / "V曲线结果.json"
    出.write_text(json.dumps(结果, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已写：", 出)
    # 打印表
    print("档 | 数据min | 轮 | 训练秒 | 历史CER | 留出CER | 阳性对照 | sham | 护栏全过")
    for t in ("D1", "D2", "D3", "D4"):
        d = 档.get(t, {})
        if d.get("历史CER") is None:
            print(f"{t} | 缺读数")
            continue
        print("{} | {:.0f} | {} | {:.0f} | {:.4f} | {:.4f} | {} | {} | {}".format(
            t, d["数据分钟"], d["训练轮数"], d["训练耗时秒"] or -1,
            d["历史CER"], d["留出CER"], d["阳性对照CER"], d["sham白噪"], d["护栏全过"]))
    if "斜率" in 结果:
        print("斜率/倍: 历史 %.4f (R2 %.3f) ｜ 留出 %.4f (R2 %.3f)" % (
            结果["斜率"]["历史"]["b_per_doubling"], 结果["斜率"]["历史"]["r2"],
            结果["斜率"]["留出"]["b_per_doubling"], 结果["斜率"]["留出"]["r2"]))
        print("判读：", 结果["判定"]["结论"])
    return 0


if __name__ == "__main__":
    sys.exit(主())
