# -*- coding: utf-8 -*-
"""meeting67/实施/标注_变长片段.py —— 变长片段离线 ASR 伪标签（Paraformer-seaco-large；C10 离线仪器）。

- 带时间戳（`return_timestamp=True`）：token 级 [起ms, 止ms]，供 npz 对齐用；
- 只产出数据标签，文本不得进入生成路径/器官回路（C9/C10）；
- 断点续跑：已标注文件跳过（按 manifest 的 sha256 去重）；
- 单进程：一次只许一个 ASR 进程。

输入：`过滤_变长片段.py` 的 manifest_通过.jsonl
输出：标注.jsonl（逐条：文件/文本/tokens/引擎/rtf/耗时/音频sha256/时长）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def 音频sha(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def 主(argv=None):
    ap = argparse.ArgumentParser(description="变长片段 ASR 伪标签（带时间戳）")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--出", required=True)
    ap.add_argument("--上限", type=int, default=0, help="只标前 N 条（0=全部）")
    参 = ap.parse_args(argv)
    出 = Path(参.出); 出.mkdir(parents=True, exist_ok=True)
    标路 = 出 / "标注.jsonl"
    已有 = set()
    if 标路.exists():
        for 行 in 标路.read_text(encoding="utf-8").splitlines():
            try:
                已有.add(json.loads(行)["音频sha256"])
            except Exception:
                pass
    rows = [json.loads(x) for x in Path(参.manifest).read_text(encoding="utf-8").splitlines() if x.strip()]
    if 参.上限:
        rows = rows[: 参.上限]
    from funasr import AutoModel
    t0 = time.time()
    m = AutoModel(model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                  disable_update=True, trust_remote_code=True, device="cpu")
    完成 = 0
    with 标路.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            p = Path(r["文件"])
            if not p.exists():
                continue
            sha = 音频sha(p)
            if sha in 已有:
                continue
            t1 = time.time()
            try:
                res = m.generate(input=str(p), batch_size_s=300, return_timestamp=True)
                if not res:
                    continue
                文本 = (res[0].get("text") or "").strip()
                ts = res[0].get("timestamp") or []
                rec = {"序": r.get("序"), "文件": str(p), "文件名": r.get("文件名"),
                       "起秒": r.get("起秒"), "止秒": r.get("止秒"), "时长秒": r.get("时长秒"),
                       "音频sha256": sha, "文本": 文本, "tokens": ts,
                       "引擎": "Paraformer-seaco-large(zh)", "耗时秒": round(time.time() - t1, 2)}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                已在 = True
                完成 += 1
                if 完成 % 20 == 0:
                    print(f"[标注] {完成} 条，已用 {time.time()-t0:.0f}s")
            except Exception as e:  # noqa: BLE001
                fh.write(json.dumps({"文件": str(p), "音频sha256": sha,
                                     "错误": f"{type(e).__name__}: {e}"}, ensure_ascii=False) + "\n")
                fh.flush()
    print(f"[标注] 完成 {完成} 条，用时 {time.time()-t0:.0f}s → {标路}")
    return 0


if __name__ == "__main__":
    sys.exit(主())
