# -*- coding: utf-8 -*-
"""test_模块三_看网页.py —— V3 模块三判据：打开浏览器、看清网页内容。

做法：写一个**本地已知内容**的 HTML → 用浏览器打开 → 置前景 → UIA 读结构 + 视觉块交叉验证
     → 断言能拿到控件、正文文本（含已知关键词）、导航区；截图与结果落盘（证明"它真的在看"）。

判据：
  M3-1 能打开浏览器并置前景（找到标题匹配的窗口）。
  M3-2 UIA 可用：控件数 > 0（系统自报结构成功）。
  M3-3 看清正文：正文文本非空且含页面里的已知关键词。
  M3-4 分出导航区：导航控件数 ≥ 1。
  M3-5 双通道与交叉验证有产出（视觉块数 ≥ 0 且有帧）。
  M3-6 正文入记忆 M：存入记忆的 向量 或 记录 非空。
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
M38 = os.path.dirname(HERE)
AHI3 = os.path.dirname(M38)                            # .../AHI3
OUT = os.path.join(AHI3, "测试记录")
for p in (HERE, M38):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── 出口编码自愈（C13/CI）：默认 GBK 控制台下 print ✅/❌ 会抛 UnicodeEncodeError 并
#    让进程 exit=1（判据全过却被判失败，见 V3审计报告 P0-3）。此处就地重配 stdout/stderr，
#    不改任何既有输出内容，只保证非 utf-8 控制台也能安全写出。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from 视觉_看网页 import 网页眼, 打开网页                       # noqa: E402

标记 = "AHI-V3-看网页测试页-%d" % int(time.time())     # 唯一标题 → 精确定位本次新窗口
关键词 = ["阿希", "网页视觉", "自指引擎", "正文段落", "导航"]


def _写测试页() -> str:
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "test_page.html")
    html = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>%s</title></head><body>
<nav id="nav"><a href="#a">首页</a> <a href="#b">文档</a> <a href="#c">关于</a></nav>
<main>
<h1>阿希的网页视觉</h1>
<p>这是一个给自指引擎看的正文段落。HDC/VSA 双极超向量，不用 Transformer。</p>
<p>第二段正文段落：眼睛用 UIA 系统自报结构加视觉块交叉验证，不 OCR。</p>
<h2>导航</h2>
<p>导航区标记为可点击，不急于点。</p>
</main></body></html>""" % 标记
    with open(p, "w", encoding="utf-8") as f:
        f.write(html)
    return p


def _置前景(子串: str) -> bool:
    try:
        import win32gui
        hits = []
        def cb(h, _):
            if win32gui.IsWindowVisible(h):
                t = win32gui.GetWindowText(h)
                if 子串 in t:
                    hits.append(h)
        win32gui.EnumWindows(cb, None)
        if not hits:
            return False
        h = hits[0]
        try:
            win32gui.ShowWindow(h, 3)                # SW_MAXIMIZE
        except Exception:
            win32gui.ShowWindow(h, 9)
        win32gui.SetForegroundWindow(h)
        return True
    except Exception:
        return False


def _存截图(路径):
    try:
        import mss
        import mss.tools
        with mss.mss() as sct:
            img = sct.grab(sct.monitors[1])
            mss.tools.to_png(img.rgb, img.size, output=路径)
        return True
    except Exception:
        return False


def main():
    页 = _写测试页()
    url = "file:///" + 页.replace("\\", "/")
    档案 = os.path.join(OUT, "edge_profile")
    ok_开 = 打开网页(url, 用edge=True, 参数=[
        "--user-data-dir=" + 档案, "--no-first-run", "--new-window",
        "--window-size=1600,1000", "--window-position=20,20"])
    time.sleep(5.0)
    置 = _置前景(标记)
    if not 置:
        time.sleep(2.0)
        置 = _置前景(标记) or _置前景("Edge") or _置前景("Chrome") or _置前景("浏览器")
    time.sleep(1.0)

    # 真编码器（UIA 正文 → HDC 文本向量）+ 真记忆（容量桥 FSW 长时库）
    编码 = None
    记忆 = None
    try:
        from 共存态 import 共存态
        编码 = 共存态(种子=3800, 用gpu=False).编码
    except Exception:
        编码 = None
    try:
        from 容量桥 import 容量桥
        记忆 = 容量桥(种子=3800)
    except Exception:
        记忆 = None

    眼 = 网页眼(种子=3800, 编码器=编码, 记忆=记忆)
    r = 眼.看(存记忆=True)
    最佳 = r
    最好命中 = len([k for k in 关键词 if k in ((r.get("布局") or {}).get("正文预览") or "")])
    for _ in range(5):                                # 可访问性懒加载 → 多次重试取最佳
        if 最好命中 >= 2:
            break
        time.sleep(1.5)
        r = 眼.看(存记忆=True)
        正r = (r.get("布局") or {}).get("正文预览") or ""
        命r = len([k for k in 关键词 if k in 正r])
        if 命r > 最好命中 or (命r == 最好命中 and r["控件数"] > 最佳["控件数"]):
            最佳, 最好命中 = r, 命r
    r = 最佳

    正文 = ""
    try:
        正文 = (r.get("布局") or {}).get("正文预览") or ""
    except Exception:
        正文 = ""
    命中 = [k for k in 关键词 if k in 正文]

    截图 = os.path.join(OUT, "看网页_截图.png")
    有图 = _存截图(截图)

    M3_1 = bool(置)
    M3_2 = bool(r["控件数"] > 0)
    M3_3 = bool(正文 and len(命中) >= 2)
    M3_4 = bool((r.get("布局") or {}).get("导航数", 0) >= 1)
    M3_5 = bool((r.get("视觉通道") or {}).get("有帧"))
    M3_6 = bool((r.get("存入记忆") or {}).get("存"))
    # M3-7 ★审计 P1-4 修复判据：正文里**不得混入浏览器外壳**（地址栏/工具栏/标签页）
    外壳首 = str(正文).lstrip().startswith(("地址和搜索栏", "地址栏", "搜索栏",
                                          "Address and search bar", "工具栏", "标签页"))
    M3_7 = bool((not 外壳首) and (r.get("布局") or {}).get("外壳剔除数", 0) >= 1)

    报 = dict(
        M3_1_置前景=dict(过=M3_1, 打开=ok_开),
        M3_2_UIA可用=dict(过=M3_2, 控件数=r["控件数"], UIA可用=r["UIA可用"]),
        M3_3_看清正文=dict(过=M3_3, 命中关键词=命中, 正文预览=正文[:160]),
        M3_4_导航区=dict(过=M3_4, 导航数=(r.get("布局") or {}).get("导航数")),
        M3_5_双通道=dict(过=M3_5, 视觉=r.get("视觉通道"),
                        交叉验证=r.get("交叉验证")),
        M3_6_正文入记忆=dict(过=M3_6, 存入=r.get("存入记忆")),
        M3_7_外壳不入正文=dict(过=M3_7, 外壳开头=外壳首,
                          外壳剔除数=(r.get("布局") or {}).get("外壳剔除数", 0),
                          正文首40字=str(正文)[:40]),
        窗口=r.get("窗口"), 控件样本=r.get("控件样本"),
        截图=截图 if 有图 else None,
    )
    全过 = all(报[k]["过"] for k in 报 if k.startswith("M3_"))
    报["全过"] = bool(全过)
    with open(os.path.join(OUT, "看网页_结果.json"), "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)
    with open(os.path.join(HERE, "test_模块三_结果.json"), "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)
    print(json.dumps(报, ensure_ascii=False, indent=1))
    return 全过


if __name__ == "__main__":
    ok = main()
    print("\n[模块三] %s" % ("全过 ✅" if ok else "未全过 ❌（详见 测试记录/看网页_结果.json）"))