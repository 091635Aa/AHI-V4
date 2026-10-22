import sys, pypdf
src, dst = sys.argv[1], sys.argv[2]
r = pypdf.PdfReader(src)
parts=[]
for i,p in enumerate(r.pages):
    try: t=p.extract_text() or ""
    except Exception as e: t=""
    parts.append(f"\n===== PAGE {i+1} =====\n"+t)
txt="".join(parts)
open(dst,"w",encoding="utf-8").write(txt)
print("pages",len(r.pages),"chars",len(txt))
