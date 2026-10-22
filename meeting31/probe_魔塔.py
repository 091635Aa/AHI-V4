# -*- coding: utf-8 -*-
"""meeting31/probe_魔塔.py —— #275 魔塔三舱：下载→解压→门控深度摄取→归纳→问答验证。"""
import io
import json
import os
import re
import sys
import tarfile
import time
import urllib.request
import zipfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")

from synthesis import 综合体

HDR = {"User-Agent": "Mozilla/5.0"}
数据目录 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "数据集", "魔塔")
每舱上限 = 600

源 = [
    ("情商", "OpenDataLab/EmpatheticDialogues",
     "raw/empatheticdialogues.tar.gz", "tar"),
    ("对话", "OmniData/DailyDialog",
     "raw/ijcnlp_dailydialog.zip", "zip"),
    ("心理学", "wangerzi/psychology-10k-zh",
     "Psychology-10K-ZH.json", "json"),
]


def 下载(ds: str, 路径: str, 目标: str):
    url = ("https://modelscope.cn/api/v1/datasets/%s/repo?FilePath=%s&Revision=master"
           % (ds, urllib.parse.quote(路径)))
    req = urllib.request.Request(url, headers=HDR)
    with urllib.request.urlopen(req, timeout=300) as r, open(目标, "wb") as f:
        f.write(r.read())
    return os.path.getsize(目标)


def 取行(路径: str, 种类: str, 上限: int) -> list:
    行s = []
    if 种类 == "tar":
        with tarfile.open(路径) as tf:
            for m in tf.getmembers():
                if m.name.endswith("train.csv"):
                    文本 = tf.extractfile(m).read().decode("utf-8", "ignore")
                    break
            else:
                文本 = ""
        for 行 in 文本.splitlines()[1:]:
            parts = 行.split(",")
            if len(parts) >= 5 and len(行) >= 20:
                行s.append(parts[4] + "。" + parts[5])
    elif 种类 == "zip":
        with zipfile.ZipFile(路径) as zf:
            for 名 in zf.namelist():
                if "dialog" in 名.lower() and 名.endswith(".txt"):
                    块s = zf.read(名).decode("utf-8", "ignore").split("__eou__")
                    for 块 in 块s:
                        s = 块.strip().replace("\n", " ")
                        if 20 <= len(s) <= 200:
                            行s.append(s)
    else:
        with open(路径, encoding="utf-8", errors="ignore") as f:
            抽 = []
            def rec(o):
                if isinstance(o, str):
                    s = o.strip()
                    if 10 <= len(s) <= 200:
                        抽.append(s)
                elif isinstance(o, list):
                    for v in o: rec(v)
                elif isinstance(o, dict):
                    for v in o.values(): rec(v)
            rec(json.load(f))
            行s = list(dict.fromkeys(抽))
    return 行s[:上限]


def main():
    os.makedirs(数据目录, exist_ok=True)
    桥 = 综合体(seed=311)
    桥.设开放域([], [])                                  # 思考态需要（无导航资源）
    前核心段 = len(桥.codec._段)
    报告 = []
    for 域名, ds, 路径, 种类 in 源:
        目标 = os.path.join(数据目录, 路径.replace("/", "_"))
        if not os.path.exists(目标):
            print("【下载】%s ← %s" % (域名, 路径))
            n = 下载(ds, 路径, 目标)
            print("   %d 字节" % n)
        else:
            print("【下载】%s 已存在" % 目标)
        行s = 取行(目标, 种类, 每舱上限)
        桥.挂舱(域名)
        # 门控深度摄取：预测不了的才入舱（不是随便投喂）
        新学 = 0
        t0 = time.time()
        for 行 in 行s:
            h = 桥.当前舱.命中(行)
            if h < 0.5:
                桥.当前舱.存行(行)
                新学 += 1
        tok = Counter(re.findall(r"[A-Za-z_]{4,}|[\u4e00-\u9fff]{2}", " ".join(
            桥.舱s[域名].行s)))
        要点 = "、".join("%s(×%d)" % (w, c) for w, c in tok.most_common(3))
        报告.append("【%s】语料 %d 行、门控新学 %d 行、归纳要点：%s（耗时 %.1fs）"
                    % (域名, len(行s), 新学, 要点, time.time() - t0))
        print(报告[-1])

    print("\n核心库段数（全程）：%d（舱隔离=核心纯净）" % len(桥.codec._段))
    print("\n【可达性】（每舱随机抽一行问一次）")
    for 域名, ds, 路径, 种类 in 源:
        舱 = 桥.舱s[域名]
        if not 舱.行s:
            continue
        素材 = 舱.行s[len(舱.行s) // 2][:14]
        桥.步("%s里提到『%s』了吗？" % (域名, 素材))
        while 桥.思考中 is not None:
            桥.断供一拍()
        print("  [%s] %s" % (域名, 桥.回复台账[-1]["响应"][:70]))
    print("\n【卸载验证】卸掉情商舱后：")
    桥.卸舱("情商")
    桥.步("情商里提到什么？")
    while 桥.思考中 is not None:
        桥.断供一拍()
    print("  ", 桥.回复台账[-1]["响应"][:70])
    print("自言审计：", 桥.自言.审计(桥.账本)["通过"])


if __name__ == "__main__":
    main()
