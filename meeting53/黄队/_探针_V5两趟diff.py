# 黄队 R2：V5 两趟逐位 diff（只读两趟 JSON，不跑判据）
import json, hashlib, sys

P = r"<workspace>/03_AHI\AHI4\meeting53\实施\产物"
a = json.load(open(P + r"\判据_V5_双通道结果.json", encoding="utf-8"))
try:
    b = json.load(open(P + r"\判据_V5_双通道结果_第二趟.json", encoding="utf-8"))
except FileNotFoundError:
    print(json.dumps({"错误": "第二趟 JSON 不存在"}, ensure_ascii=False)); sys.exit(0)


def 剥(d):
    """剥掉非确定性字段（墙钟/峰值内存），只留可判内容"""
    d = json.loads(json.dumps(d))  # deep copy
    d.get("备注", {}).pop("墙钟秒", None)
    d.get("备注", {}).pop("峰值内存MB", None)
    return d


def 指纹(d):
    return hashlib.sha256(json.dumps(剥(d), sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:32]


def 逐臂比(a, b):
    rows = []
    for 名 in sorted(set(a.get("臂", {})) | set(b.get("臂", {}))):
        ra, rb = a["臂"].get(名), b["臂"].get(名)
        if ra is None or rb is None:
            rows.append((名, "缺臂", None)); continue
        sa = json.dumps({k: v for k, v in ra.items() if k != "四态"}, sort_keys=True, ensure_ascii=False)
        sb = json.dumps({k: v for k, v in rb.items() if k != "四态"}, sort_keys=True, ensure_ascii=False)
        rows.append((名, "同" if sa == sb else "异", ra.get("四态")))
    return rows


ra = 逐臂比(a, b)
差 = [x for x in ra if x[1] == "异"]
print(json.dumps({
    "臂数_A": len(a.get("臂", {})), "臂数_B": len(b.get("臂", {})),
    "逐臂读数不同数": len(差), "不同臂": [x[0] for x in 差],
    "剥墙钟后内容指纹_A": 指纹(a), "剥墙钟后内容指纹_B": 指纹(b),
    "剥墙钟后逐位同": 指纹(a) == 指纹(b),
    "基线指纹_A": a.get("备注", {}).get("基线指纹"),
    "基线指纹_B": b.get("备注", {}).get("基线指纹"),
    "基线指纹同": a.get("备注", {}).get("基线指纹") == b.get("备注", {}).get("基线指纹"),
    "墙钟_A": a.get("备注", {}).get("墙钟秒"), "墙钟_B": b.get("备注", {}).get("墙钟秒"),
    "峰值内存_A": a.get("备注", {}).get("峰值内存MB"), "峰值内存_B": b.get("备注", {}).get("峰值内存MB"),
    "汇总_A": a.get("汇总", {}).get("PASS"), "汇总_B": b.get("汇总", {}).get("PASS"),
}, ensure_ascii=False, indent=1))
