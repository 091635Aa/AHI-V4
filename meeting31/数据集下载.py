# -*- coding: utf-8 -*-
"""meeting31/数据集下载.py —— #274 三域数据集下载器（情商/对话/心理学）。

多候选 URL 逐一尝试（GitHub raw / HF 镜像），超时/失败如实报告；
产物落 数据集/<域名>/。域→舱摄取见 probe_数据集.py。
"""
from __future__ import annotations

import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
数据目录 = os.path.join(HERE, "数据集")

域 = {
    "情商": [
        ("https://raw.githubusercontent.com/facebookresearch/empathetic_dialogues/master/data/train.csv",
         "empathetic_train.csv"),
        ("https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/main/ESConv.json",
         "ESConv.json"),
    ],
    "对话": [
        ("https://hf-mirror.com/datasets/ConvLab/dailydialog/resolve/main/dummy_data.json",
         "dailydialog_dummy.json"),
    ],
    "心理学": [
        ("https://hf-mirror.com/datasets/Amod/mental_health_counseling_conversations/resolve/main/combined_dataset.json",
         "counseling.json"),
        ("https://hf-mirror.com/datasets/MaggiePai/mental_health_counseling_conversations/resolve/main/combined_dataset.jsonl",
         "counseling.jsonl"),
    ],
}


def 下载(url: str, 目标: str, 超时: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=超时) as r:
        数据 = r.read(50 * 1024 * 1024)
    with open(目标, "wb") as f:
        f.write(数据)
    return "%d 字节" % len(数据)


def main():
    os.makedirs(数据目录, exist_ok=True)
    for 域名, 候选s in 域.items():
        域目录 = os.path.join(数据目录, 域名)
        os.makedirs(域目录, exist_ok=True)
        成功 = False
        for url, 名 in 候选s:
            目标 = os.path.join(域目录, 名)
            if os.path.exists(目标) and os.path.getsize(目标) > 1000:
                print("[%s] 已有：%s" % (域名, 名))
                成功 = True
                break
            try:
                print("[%s] 尝试 %s …" % (域名, url))
                info = 下载(url, 目标)
                print("[%s] 成功：%s（%s）" % (域名, 名, info))
                成功 = True
                break
            except Exception as e:
                print("[%s] 失败：%s（%s）" % (域名, type(e).__name__, e))
        if not 成功:
            print("[%s] 全部候选失败——如实记录" % 域名)


if __name__ == "__main__":
    main()
