# -*- coding: utf-8 -*-
"""meeting31/搜索工具.py —— #277 自主搜索工具（DuckDuckGo HTML 端点，无键、确定性解析）。"""
from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request

HDR = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 Chrome/120 Safari/537.36")}


def _清(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s).strip()


def 搜索(查询: str, 条数: int = 8, 超时: int = 30) -> list:
    """返回 [(标题, 摘要)]——浏览器搜索的文本模态（视觉模态见 #278 预登记）。
    国内可达端点：cn.bing.com；失败回退 www.bing.com。"""
    页 = ""
    for 主机 in ("cn.bing.com", "www.bing.com"):
        try:
            url = ("https://%s/search?q=" % 主机) + urllib.parse.quote(查询)
            req = urllib.request.Request(url, headers=HDR)
            页 = urllib.request.urlopen(req, timeout=超时).read().decode(
                "utf-8", "ignore")
            if 页:
                break
        except Exception:
            continue
    块s = re.findall(r'<li class="b_algo".*?</li>', 页, re.S)
    出 = []
    for 块 in 块s[:条数]:
        标m = re.search(r"<h2[^>]*>.*?<a[^>]*>(.*?)</a>", 块, re.S)
        摘m = re.search(r"<p[^>]*>(.*?)</p>", 块, re.S)
        标 = _清(标m.group(1)) if 标m else ""
        摘 = _清(摘m.group(1)) if 摘m else ""
        if 标:
            出.append((标, 摘))
    return 出
