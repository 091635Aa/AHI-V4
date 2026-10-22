# -*- coding: utf-8 -*-
"""meeting31/probe_数据集.py —— #274 三舱摄取与验证：情商/对话/心理学。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")

from synthesis import 综合体

数据目录 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "数据集")
行上限 = 2500


def 抽取字符串(obj, 池):
    """递归抽取 JSON 中的字符串（10–200 字）。"""
    if isinstance(obj, str):
        s = obj.strip()
        if 10 <= len(s) <= 200:
            池.append(s)
    elif isinstance(obj, list):
        for v in obj:
            抽取字符串(v, 池)
    elif isinstance(obj, dict):
        for v in obj.values():
            抽取字符串(v, 池)


def 摄取(桥, 域名, 文件名, 舱名):
    路径 = os.path.join(数据目录, 域名, 文件名)
    池 = []
    if 文件名.endswith((".json", ".jsonl")):
        sep = "\n" if 文件名.endswith(".jsonl") else None
        with open(路径, encoding="utf-8", errors="ignore") as f:
            if sep:
                for 行 in f:
                    if 行.strip():
                        try:
                            抽取字符串(json.loads(行), 池)
                        except Exception:
                            pass
            else:
                抽取字符串(json.load(f), 池)
    else:                                             # csv/txt：按行
        with open(路径, encoding="utf-8", errors="ignore") as f:
            for 行 in f.read().splitlines():
                if 10 <= len(行.strip()) <= 200:
                    池.append(行.strip())
    去重 = list(dict.fromkeys(池))[:行上限]
    舱 = 桥.舱s[舱名]
    for s in 去重:
        舱.存行(s)
    return len(去重)


def main():
    桥 = 综合体(seed=311)
    前核心段 = len(桥.codec._段)
    for 域名 in ("情商", "对话", "心理学"):
        桥.挂舱(域名)
    桥.设开放域([], [])                     # 空域：纯舱检索（无导航资源）
    print("核心库段数（摄取前）：%d" % 前核心段)

    测试 = []
    for 域名, 文件名 in (("情商", "ESConv.json"),
                        ("对话", "dailydialog_dummy.json"),
                        ("心理学", "counseling.jsonl")):
        路径 = os.path.join(数据目录, 域名, 文件名)
        n = 摄取(桥, 域名, 文件名, 域名)
        print("【摄取】%s 舱 ← %s：%d 行（语料采样 %s…）"
              % (域名, 文件名, n, (桥.舱s[域名].行s[0][:24] if 桥.舱s[域名].行s else "-")))
        测试.append((域名, 桥.舱s[域名].行s[len(桥.舱s[域名].行s) // 2][:16]))
    print("核心库段数（摄取后）：%d（不变=隔离 PASS）" % len(桥.codec._段))

    print("\n【可达性】（延答机制：问题入思考队列，取证后投递）")
    for 域名, 素材 in 测试:
        r = 桥.步("%s里提到『%s』了吗？" % (域名, 素材))
        while 桥.思考中 is not None:
            桥.断供一拍()                                # 思考态取证（隔离下应找不到）
        响应 = 桥.回复台账[-1]["响应"]
        ok = 域名 in 响应
        print("  [%s] %s" % (域名, "命中舱（%s）" % 响应[:52] if ok else "未命中（%s）" % 响应[:44]))
    print("\n【卸载后不可达】")
    桥.卸舱("情商")
    r = 桥.步("情商里提到『%s』了吗？" % 测试[0][1])
    while 桥.思考中 is not None:
        桥.断供一拍()
    响应 = 桥.回复台账[-1]["响应"]
    print("  [情商] 卸舱后同问 → %s" % ("已不可达 PASS" if "《情商》" not in 响应 else "FAIL"))
    print("自言审计：", 桥.自言.审计(桥.账本)["通过"])
    print("自言审计：", 桥.自言.审计(桥.账本)["通过"])


if __name__ == "__main__":
    main()
