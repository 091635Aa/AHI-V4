# -*- coding: utf-8 -*-
"""附加探针 5：S0 合成刺激是否被改（重跑到临时目录后逐字节比 47 个 wav + manifest）。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

M69 = Path(r"<workspace>/03_AHI\AHI4\meeting69")
复核 = M69 / "产物" / "S0.1" / "复核"
tmp = 复核 / "S0刺激重跑"
(tmp / "产物" / "S0").mkdir(parents=True, exist_ok=True)

s = importlib.util.spec_from_file_location("合0", M69 / "实施" / "合成_S0刺激.py")
合 = importlib.util.module_from_spec(s); s.loader.exec_module(合)
合.M68 = tmp
合.出 = tmp / "产物" / "S0" / "刺激"
合.主()

原 = M69 / "产物" / "S0"
同 = {}; 全同 = True
for p in sorted((原 / "刺激").glob("*.wav")):
    q = 合.出 / p.name
    ok = q.exists() and q.read_bytes() == p.read_bytes()
    同[p.name] = ok
    全同 = 全同 and ok
真同 = (原 / "manifest.json").read_text(encoding="utf-8") == (tmp / "产物" / "S0" / "manifest.json").read_text(encoding="utf-8")
出 = dict(wav数=len(同), 全逐字节同=bool(全同), 不同文件=[k for k, v in 同.items() if not v],
        manifest文本同=bool(真同),
        说明="同源合成器重跑；逐字节同 ⇒ 47 个冻结刺激未被改动")
(复核 / "复核_S0刺激.json").write_text(json.dumps(出, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(出, ensure_ascii=False, indent=1))
