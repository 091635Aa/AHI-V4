# meeting67/实施/V曲线_汇总.py —— 汇总 V-v2 四档评测（预注册 e0150 + 探索性末轮），算斜率/knee
# 只读 产物/V曲线_v2；只写 产物/V曲线_v2/V曲线结果.json；不改任何既有件。
# 用法：& $py meeting67\实施\V曲线_汇总.py
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
M67 = HERE.parent
V2 = M67 / "产物" / "V曲线_v2"
子集索引 = M67 / "数据" / "子集索引.json"

档 = ["D1", "D2", "D3", "D4"]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def 读档(tag: str) -> dict:
    d = V2 / tag
    报 = dict(tag=tag)
    for 名, 文件 in (("预注册_e0150", "判据_V_v2_%s.json" % tag), ("探索性_末轮", "判据_V_v2_末_%s.json" % tag)):
        p = d / 文件
        if not p.exists():
            报[名] = None
            continue
        j = json.loads(p.read_text(encoding="utf-8"))
        v = j.get("判定", {})
        报[名] = dict(
            历史=float(v["历史"]["CER均值"]), 留出=float(v["留出"]["CER均值"]),
            均值=round(float((v["历史"]["CER均值"] + v["留出"]["CER均值"]) / 2), 4),
            sham=bool(j.get("sham", {}).get("双达标")), 阳性对照=j.get("仪器阳性对照", {}).get("CER"),
            逐句=j.get("判定", {}).get("历史", {}).get("逐句"),
        )
    # 训练信息
    cp = d / "训练曲线.json"
    if cp.exists():
        ti = (json.loads(cp.read_text(encoding="utf-8")).get("训练信息") or {})
        报["训练"] = dict(轮=ti.get("轮"), 耗时秒=ti.get("耗时秒"), 早停=ti.get("提前停"))
    return 报


def 斜率(xs: list, ys: list) -> float:
    """最小二乘 CER ~ log2(minutes)，返回 b（每翻倍的数据量 CER 变化）。"""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else float("nan")


def knee(rows: list) -> str:
    """与 D4 双集差 ≤0.05 的最小档（两个 ckpt 口径分别算）。"""
    out = {}
    for 名 in ("预注册_e0150", "探索性_末轮"):
        full = next(r[名] for r in rows if r["tag"] == "D4")
        k = None
        for r in rows:
            v = r[名]
            if not v:
                continue
            if abs(v["历史"] - full["历史"]) <= 0.05 and abs(v["留出"] - full["留出"]) <= 0.05:
                k = r["tag"]
                break
        out[名] = k
    return out


def 主() -> int:
    索引 = json.loads(子集索引.read_text(encoding="utf-8"))
    分钟 = {}
    for r in 索引.get("档", []):
        tag = r.get("tag")
        if tag in 档 and r.get("训练秒"):
            分钟[tag] = round(float(r["训练秒"]) / 60.0, 3)
    rows = [读档(t) for t in 档]
    for r in rows:
        r["数据分钟"] = 分钟.get(r["tag"])
    log2 = [math.log2(分钟[t]) for t in 档 if 分钟.get(t)]
    汇总 = {}
    for 名 in ("预注册_e0150", "探索性_末轮"):
        hs = [r[名]["历史"] for r in rows if r[名]]
        ls = [r[名]["留出"] for r in rows if r[名]]
        if len(hs) == len(档):
            汇总[名] = dict(
                历史CER={r["tag"]: r[名]["历史"] for r in rows},
                留出CER={r["tag"]: r[名]["留出"] for r in rows},
                斜率_历史_每翻倍=round(斜率(log2, hs), 4),
                斜率_留出_每翻倍=round(斜率(log2, ls), 4),
                knee=knee(rows),
            )
    报 = dict(
        脚本="meeting67/实施/V曲线_汇总.py",
        脚本sha256=_sha256(Path(__file__)),
        时间戳=time.strftime("%Y-%m-%d %H:%M:%S"),
        口径=dict(训练秒转分钟=分钟, 斜率定义="CER ~ log2(数据分钟) 最小二乘"),
        逐档=rows,
        汇总=汇总,
        说明="预注册=ckpt_e0150（与 V-v1 同一 ckpt，读数应逐位相同）；探索性=ckpt_末（8 min 墙钟末轮，非预注册）",
    )
    输出 = V2 / "V曲线结果.json"
    输出.write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print("=== V-v2 汇总 ===")
    for r in rows:
        print("%-3s 数据 %6.2f min 轮 %4s | e0150 历史 %.4f 留出 %.4f | 末轮 历史 %s 留出 %s"
              % (r["tag"], r["数据分钟"] or -1, (r.get("训练") or {}).get("轮"),
                 (r["预注册_e0150"] or {}).get("历史", float("nan")),
                 (r["预注册_e0150"] or {}).get("留出", float("nan")),
                 ("%.4f" % r["探索性_末轮"]["历史"]) if r["探索性_末轮"] else "未测",
                 ("%.4f" % r["探索性_末轮"]["留出"]) if r["探索性_末轮"] else "未测"))
    for 名, v in 汇总.items():
        print("[%s] 斜率 历史 %+.4f/倍 留出 %+.4f/倍 knee=%s"
              % (名, v["斜率_历史_每翻倍"], v["斜率_留出_每翻倍"], v["knee"][名]))
    print("[落盘] %s" % 输出)
    return 0


if __name__ == "__main__":
    raise SystemExit(主())
