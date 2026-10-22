# 黄队 R3：核对 held-out 汇总计数与证据 §2 表是否一致
import json

P = r"<workspace>/03_AHI\AHI4\meeting53\实施\产物"
h = json.load(open(P + r"\判据_heldout.json", encoding="utf-8"))
print("汇总:", json.dumps(h["汇总"], ensure_ascii=False)[:200])
print("臂数:", len(h["臂"]))
for k, v in h["臂"].items():
    print("%-22s %s" % (k, v.get("四态")))
a = json.load(open(P + r"\判据_A.json", encoding="utf-8"))
print("A_汇总:", json.dumps(a["汇总"], ensure_ascii=False)[:160], "| A_臂数:", len(a["臂"]))
