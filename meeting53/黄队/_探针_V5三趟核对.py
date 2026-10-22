# 黄队 R2：抽取两趟/三趟的判别性读数，核对「原趟 vs 第二趟」是否内容逐位同
import json

P = r"<workspace>/03_AHI\AHI4\meeting53\实施\产物"
b = json.load(open(P + r"\判据_V5_双通道结果_第二趟.json", encoding="utf-8"))
c = json.load(open(P + r"\判据_V5_双通道结果.json", encoding="utf-8"))

def pick(d):
    g = lambda n: d["臂"].get(n, {})
    return {
        "汇总_PASS": d["汇总"]["PASS"], "汇总_FAIL": d["汇总"]["FAIL"],
        "码_甲": g("G-01 确定性")["读数"].get("码_甲"),
        "T1-04_可解码": [round(x["可解码率"], 4) for x in g("T1-04 正文可解码")["读数"]],
        "T1-02_置换": [round(x["置换状态改变率"], 3) for x in g("T1-02 状态依赖")["读数"]],
        "T2-02_动作改变": [round(x["同扰动下动作改变率"], 3) for x in g("T2-02 共同状态")["读数"]],
        "T3-02_正文改变": [round(x["正文改变率"], 3) for x in g("T3-02 因果三档")["读数"]],
        "T3-06_spearman": [round(x["spearman"], 4) for x in g("T3-06 难度→步数")["读数"]],
        "T3-05_掉分": [round(x["掉分"], 3) for x in g("T3-05 F5 不可重演")["读数"]],
        "T4-01": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in g("T4-01 通信因果")["读数"].items()},
        "T4-03": g("T4-03 归属门")["读数"],
        "基线指纹": d["备注"]["基线指纹"],
        "墙钟": d["备注"]["墙钟秒"], "峰值MB": d["备注"]["峰值内存MB"],
    }

pb, pc = pick(b), pick(c)
print("=== 第二趟 ===")
print(json.dumps(pb, ensure_ascii=False, indent=1))
print("=== 当前 结果.json（22:45 第三趟） ===")
print(json.dumps(pc, ensure_ascii=False, indent=1))
print("=== 差异键 ===")
print([k for k in pb if pb[k] != pc[k]])
