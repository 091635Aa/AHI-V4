# -*- coding: utf-8 -*-
"""E4 只读工具：把 LST ICLR2026 PDF 抽成文本，供反方审查核对原文（不改既有件）。"""
import os
from pypdf import PdfReader

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(os.path.dirname(os.path.dirname(HERE)), "产物", "文献", "LST_ICLR2026.pdf")
OUT = os.path.join(HERE, "_lst_text.txt")

r = PdfReader(PDF)
print("pages", len(r.pages))
parts = []
for i, p in enumerate(r.pages):
    try:
        parts.append("\n\n=== PAGE %d ===\n" % (i + 1) + (p.extract_text() or ""))
    except Exception as e:  # noqa: BLE001
        parts.append("\n\n=== PAGE %d ERR %s ===\n" % (i + 1, e))
txt = "".join(parts)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(txt)
print("chars", len(txt), "->", OUT)
