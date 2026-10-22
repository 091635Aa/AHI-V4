# -*- coding: utf-8 -*-
"""meeting31/modelscope_找数据.py —— 魔塔社区数据集探测（#275）。"""
import json
import urllib.request
import urllib.parse

HDR = {"User-Agent": "Mozilla/5.0"}


def get_json(url):
    req = urllib.request.Request(url, headers=HDR)
    return json.load(urllib.request.urlopen(req, timeout=30))


def 搜(关键词, 页=1, 数=8):
    q = urllib.parse.quote(关键词)
    url = ("https://modelscope.cn/api/v1/datasets?PageSize=%d&PageNumber=%d&Query=%s"
           % (数, 页, q))
    try:
        r = get_json(url)
        data = r.get("Data")
        rows = data if isinstance(data, list) else (data or {}).get("Datasets") or []
        out = []
        for x in rows[:数]:
            ns = x.get("Namespace") or x.get("namespace") or "?"
            名 = x.get("Name") or x.get("name") or "?"
            out.append("%s/%s" % (ns, 名))
        return out
    except Exception as e:
        return ["✗ %s: %s" % (type(e).__name__, str(e)[:50])]


for kw in ("empathetic", "dailydialog", "psychology", "counseling",
           "情感对话", "心理咨询", "情商"):
    print("『%s』→" % kw)
    for id_ in 搜(kw):
        print("   ", id_)
