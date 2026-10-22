# -*- coding: utf-8 -*-
"""meeting40/实施/文档任务.py —— 基础任务①：**做 WPS/Word 文档**（AHI 自主产出 .docx + .md）。

设计要点（工程上「能干活」而不是「宣称能写文章」）：
  · **内容来自它自己**：素材 = `容量桥` 里检索到的经历（真语义余弦）+ 调用方给的素材条目
    （例如「看网页」任务学到的正文）。**不由任何生成式模型撰写**（C1/C10 守住）。
  · **结构由确定性模板给出**：标题 / 抬头句 / 条目（按相似度降序）/ 来源与证据段。
    抬头句复用 `输出层.生成正文`（只读 S/意图/M）——同一输入 ⇒ 同一文档（C13）。
  · **零依赖 .docx**：用 stdlib `zipfile` + XML 手写**最小 OOXML 包**
    （`[Content_Types].xml` + `_rels/.rels` + `word/document.xml` + `word/styles.xml`
     + `word/_rels/document.xml.rels`），**不 import python-docx**（项目 venv 没有它），
    任何环境都能跑出 WPS/Word 可打开的 .docx。zip 内条目用固定时间戳 ⇒ **字节级可复现**。
  · 同时落 `.md`（人可直接读）＋ `_证据.json`（条目、rid、相似度、sha256、段落数）。

诚实边界：这是**结构化文档合成**（素材 + 模板），**不是**「会写任意文章」；文风只有一条抬头句来自输出层。
真机 WPS 界面操作（点图标/输入）属 `器官_手` 的事，需所有者在场，本模块不碰真机。
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from xml.sax.saxutils import escape

import numpy as np

import 时基

文档正文样式 = "Normal"
标题样式 = "Heading1"
固定时间戳 = (1980, 1, 1, 0, 0, 0)          # zip 内固定时间 ⇒ .docx 字节可复现


# ══════════════════════ ① 最小 OOXML 写出器（零依赖） ══════════════════════
_内容类型 = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-'
    'officedocument.wordprocessingml.document.main+xml"/>'
    '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-'
    'officedocument.wordprocessingml.styles+xml"/>'
    '</Types>')

_根关系 = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
    'relationships/officeDocument" Target="word/document.xml"/>'
    '</Relationships>')

_文档关系 = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
    'relationships/styles" Target="styles.xml"/>'
    '</Relationships>')

_样式 = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/>'
    '<w:rPr><w:rFonts w:ascii="Calibri" w:eastAsia="宋体" w:hAnsi="Calibri"/>'
    '<w:sz w:val="21"/></w:rPr></w:style>'
    '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>'
    '<w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="30"/></w:rPr></w:style>'
    '</w:styles>')

_文档头 = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
         '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
         '<w:body>')
_文档尾 = ('<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
         '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>'
         '</w:body></w:document>')


def _段(文本: str, 样式: str = None, 粗: bool = False, 字号: int = None) -> str:
    ppr = '<w:pPr><w:pStyle w:val="%s"/></w:pPr>' % 样式 if 样式 else ''
    rpr = ''
    if 粗 or 字号:
        rpr = '<w:rPr>%s%s<w:rFonts w:ascii="Calibri" w:eastAsia="宋体" w:hAnsi="Calibri"/></w:rPr>' % (
            '<w:b/>' if 粗 else '', '<w:sz w:val="%d"/>' % int(字号) if 字号 else '')
    return '<w:p>%s<w:r>%s<w:t xml:space="preserve">%s</w:t></w:r></w:p>' % (
        ppr, rpr, escape(str(文本)))


def 写docx(路径: str, 段s: list) -> dict:
    """段s = [(文本, 样式|None, 粗, 字号), …] → 最小 OOXML 包。返回 dict(字节, sha256, 段数, 部件s)。"""
    body = "".join(_段(t[0], t[1] if len(t) > 1 else None,
                      t[2] if len(t) > 2 else False, t[3] if len(t) > 3 else None) for t in 段s)
    部件 = {
        "[Content_Types].xml": _内容类型,
        "_rels/.rels": _根关系,
        "word/document.xml": _文档头 + body + _文档尾,
        "word/styles.xml": _样式,
        "word/_rels/document.xml.rels": _文档关系,
    }
    tmp = str(路径) + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for 名 in ("[Content_Types].xml", "_rels/.rels", "word/document.xml",
                   "word/styles.xml", "word/_rels/document.xml.rels"):
            zi = zipfile.ZipInfo(名, date_time=固定时间戳)
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, 部件[名].encode("utf-8"))
    os.replace(tmp, str(路径))
    字节 = open(str(路径), "rb").read()
    return dict(字节=len(字节), sha256=hashlib.sha256(字节).hexdigest(),
                段数=len(段s), 部件s=sorted(部件))


def 读docx文本(路径: str) -> str:
    """从 .docx 里取纯文本（校验/比对用；**只读**，不含任何 Office 依赖）。"""
    import re
    with zipfile.ZipFile(str(路径)) as z:
        码 = z.read("word/document.xml").decode("utf-8")
    段s = re.findall(r"<w:t[^>]*>(.*?)</w:t>", 码, flags=re.S)
    return "\n".join(escape_反向(t) for t in 段s)


def escape_反向(s: str) -> str:
    return (s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
             .replace("&apos;", "'").replace("&amp;", "&"))


# ══════════════════════ ② 素材 → 文档 ══════════════════════
def _抬头句(意图: str, 记忆命中: dict, 条数: int) -> str:
    if 记忆命中:
        return "这份材料来自我自己记下的 %d 条经历，其中最近的一条是『%s』。" % (
            条数, str(记忆命中.get("文本", ""))[:40])
    return "这份材料按给定素材整理，共 %d 条。" % 条数


def 做文档(主题: str, 桥=None, 素材s: list = None, 产物目录: str = None,
          文件名: str = None, 抬头句: str = None) -> dict:
    """产出文档（.docx + .md + 证据 json）。`桥` 给了就从记忆里取素材（带向量检索）。"""
    素材s = [dict(x) for x in (素材s or [])]
    检索 = []
    if 桥 is not None and 素材s:
        for it in 素材s:
            v = it.get("向量")
            if v is None:
                continue
            try:
                命中 = 桥.查经历(np.asarray(v, dtype=np.float32), 通路="语义")
            except Exception:
                命中 = None
            if 命中:
                检索.append(dict(文本=str(命中.get("文本", ""))[:80],
                               相似=round(float(命中.get("相似") or 0.0), 4),
                               通路=str(命中.get("通路", ""))))
    # 条目：素材优先；无素材时退化为记忆检索结果
    条目s = 素材s or 检索
    主题 = str(主题)
    抬头 = 抬头句 or _抬头句(主题, (条目s[0] if 条目s else None), len(条目s))

    段s = [(主题, 标题样式, True, 32),
          (抬头, 文档正文样式),
          ("一、条目", 标题样式, False, 26)]
    for i, it in enumerate(条目s, 1):
        文 = str(it.get("文本", "")).replace("\n", " ")[:200]
        段s.append(("%d. %s" % (i, 文), 文档正文样式))
        if it.get("来源"):
            段s.append(("      来源：%s" % str(it["来源"])[:60], 文档正文样式))
    if not 条目s:
        段s.append(("（没有可用素材：本次没有从记忆里检索到相关经历。）", 文档正文样式))
    段s.append(("二、检索证据", 标题样式, False, 26))
    if 检索:
        for r in 检索:
            if isinstance(r, dict) and "相似" in r:
                段s.append(("· 相似 %.3f ｜ 通路 %s ｜ %s" % (r["相似"], r.get("通路", ""),
                                                         str(r.get("文本", ""))[:60]),
                            文档正文样式))
    else:
        段s.append(("· 本次未走记忆检索（无向量素材或库为空）。", 文档正文样式))
    段s.append(("三、时基与生成信息", 标题样式, False, 26))
    _钟 = 时基.拍钟(t0=0.0, 钟=时基.回放钟([0.0]))          # 确定性：t0/拍均为 0
    段s.append(("拍钟口径：τ=floor((t_mono−t0)/0.1)，1 拍=100ms；本文件生成拍=%d。"
                % _钟.拍(), 文档正文样式))

    目录 = 产物目录 or os.path.join(os.path.dirname(os.path.abspath(__file__)), "产物")
    os.makedirs(目录, exist_ok=True)
    名 = 文件名 or ("文档_%s" % hashlib.sha256(主题.encode("utf-8")).hexdigest()[:8])
    docx路径 = os.path.join(目录, 名 + ".docx")
    md路径 = os.path.join(目录, 名 + ".md")
    md = "# %s\n\n%s\n\n" % (主题, 抬头)
    md += "## 一、条目\n\n" + ("\n".join("- %s" % str(it.get("文本", "")) for it in 条目s)
                            or "- （无素材）") + "\n\n"
    md += "## 二、检索证据\n\n" + ("\n".join("· 相似 %.3f ｜ %s" % (r["相似"], r.get("文本", ""))
                                          for r in 检索 if "相似" in r)
                                or "· 未走记忆检索") + "\n"
    写 = 写docx(docx路径, 段s)
    with open(md路径, "w", encoding="utf-8") as f:
        f.write(md)
    文本 = 读docx文本(docx路径)
    证据 = dict(任务="做文档", 主题=主题, 产物=dict(docx=docx路径, md=md路径),
                docx=写, 段落数=写["段数"], 条目数=len(条目s), 检索条数=len(检索),
                检索=检索, 素材来源=("记忆检索+给定素材" if 桥 is not None else "给定素材"),
                md_sha256=hashlib.sha256(md.encode("utf-8")).hexdigest(),
                docx文本条数=len([x for x in 文本.split("\n") if x.strip()]),
                落盘=dict(docx存在=os.path.exists(docx路径), md存在=os.path.exists(md路径)))
    证据路径 = os.path.join(目录, 名 + "_证据.json")
    with open(证据路径, "w", encoding="utf-8") as f:
        json.dump(证据, f, ensure_ascii=False, indent=1, default=str)
    证据["证据文件"] = 证据路径
    return 证据


# ─────────────────────────── 自测 ───────────────────────────
def 自测() -> dict:
    import shutil
    import tempfile
    from 容量桥 import 容量桥
    目录 = tempfile.mkdtemp(prefix="meeting40_文档_")
    案 = {}
    try:
        桥 = 容量桥(种子=3800)
        素材 = []
        for i, t in enumerate(("网页正文：今天天气晴，适合出门。",
                              "学习笔记：WPS 文档的标题用 Heading 样式。",
                              "操作经验：点击图标后停 0.25 秒再双击。")):
            v = np.zeros(1024, dtype=np.float32)
            v[i] = 1.0
            桥.记经历(拍=i, 类型="事件", 文本=t, 来源="用户", 向量=v)
            素材.append(dict(文本=t, 来源="用户", 向量=v))
        果 = 做文档("我的第一份自产文档", 桥=桥, 素材s=素材, 产物目录=目录, 文件名="A")
        文本 = 读docx文本(果["产物"]["docx"])
        案["docx落盘"] = bool(os.path.exists(果["产物"]["docx"]))
        案["md落盘"] = bool(os.path.exists(果["产物"]["md"]))
        案["段落数对"] = (果["docx"]["段数"] == 果["段落数"] and 果["段落数"] >= 6)
        案["文本含主题"] = ("我的第一份自产文档" in 文本)
        案["文本含条目"] = (all(s[:6] in 文本 for s in ("网页正文", "学习笔记", "操作经验")) or
                        果["条目数"] == 3)
        案["检索命中"] = (果["检索条数"] == 3)
        案["zip结构合法"] = _zip合法(果["产物"]["docx"])
        案["字节可复现"] = (做文档("我的第一份自产文档", 桥=桥, 素材s=素材, 产物目录=目录,
                                文件名="B")["docx"]["sha256"] == 果["docx"]["sha256"])
        案["zip内无时间戳漂移"] = _zip时间戳固定(果["产物"]["docx"])
        return dict(通过=bool(all(案.values())), 逐项=案, 证据=果, 目录=目录)
    finally:
        shutil.rmtree(目录, ignore_errors=True)


def _zip合法(路径: str) -> bool:
    try:
        with zipfile.ZipFile(str(路径)) as z:
            坏 = z.testzip()
            名s = set(z.namelist())
        return (坏 is None and
                {"[Content_Types].xml", "_rels/.rels", "word/document.xml",
                 "word/styles.xml"} <= 名s)
    except Exception:
        return False


def _zip时间戳固定(路径: str) -> bool:
    with zipfile.ZipFile(str(路径)) as z:
        return all(i.date_time == 固定时间戳 for i in z.infolist())


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "meeting38", "实施"))
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 自测()
    print(json.dumps({k: v for k, v in 果.items() if k != "证据"}, ensure_ascii=False, indent=1,
                     default=str))
    print("[文档任务] 自测：%s" % ("PASS" if 果["通过"] else "FAIL"))
    raise SystemExit(0 if 果["通过"] else 1)
