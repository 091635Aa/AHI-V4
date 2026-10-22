# -*- coding: utf-8 -*-
"""meeting67/实施/合并数据_扩容.py —— 既有 2.77h npz + 新片段 npz → 合并训练集目录（meta 兼容）。

只读既有件（C14）：不改 meeting66/产物/数据，只复制/新写合并目录。
词表/mel 统计沿用既有 meta（两臂归一化口径一致）；划分：旧 train + 全部新样本；val 保持旧 val。

输出：<出>/样本/*.npz + meta.json（loaders 兼容 meeting66 模型_学说话.py）
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def 主(argv=None):
    ap = argparse.ArgumentParser(description="合并既有数据 + 扩容新数据")
    ap.add_argument("--既有", required=True, help="既有数据目录（含 样本/ 与 meta.json）")
    ap.add_argument("--新", required=True, help="建数据_扩容.py 输出目录（含 样本/ 与 meta_扩容.json）")
    ap.add_argument("--出", required=True)
    参 = ap.parse_args(argv)
    既有 = Path(参.既有); 新 = Path(参.新); 出 = Path(参.出)
    (出 / "样本").mkdir(parents=True, exist_ok=True)
    m0 = json.loads((既有 / "meta.json").read_text(encoding="utf-8"))
    m1 = json.loads((新 / "meta_扩容.json").read_text(encoding="utf-8"))
    旧样本 = list(m0.get("样本", []))
    n0 = len(旧样本)
    # 1) 复制既有 npz
    for i, r in enumerate(旧样本):
        src = 既有 / "样本" / r["文件"]
        dst = 出 / "样本" / f"{i:04d}.npz"
        if not dst.exists():
            shutil.copy2(src, dst)
    样本 = []
    for i, r in enumerate(旧样本):
        样本.append({**r, "id": i, "文件": f"{i:04d}.npz"})
    # 2) 复制新 npz + 追加样本记录
    新记录 = [r for r in m1.get("记录", [])]
    for j, r in enumerate(新记录):
        src = 新 / "样本" / r["文件"]
        dst = 出 / "样本" / f"{n0 + j:04d}.npz"
        if not dst.exists():
            shutil.copy2(src, dst)
        样本.append({**r, "id": n0 + j, "文件": f"{n0 + j:04d}.npz"})
    # 3) 划分：旧 train + 全部新；val 保持旧 val
    划分0 = m0.get("划分", {}) or {}
    旧train = list(划分0.get("train", []) or [])
    旧val = list(划分0.get("val", []) or [])
    新ids = list(range(n0, n0 + len(新记录)))
    train = 旧train + 新ids
    val = 旧val
    # 4) meta
    meta = dict(m0)
    meta["样本"] = 样本
    meta["划分"] = dict(划分0)
    meta["划分"]["train"] = train
    meta["划分"]["train数"] = len(train)
    meta["划分"]["val"] = val
    meta["划分"]["val数"] = len(val)
    meta["划分"]["说明"] = "旧 train + 全部新样本；val 保持旧 val（两臂同验证集）"
    meta["版本"] = "数据_SpeakerA+扩容 v1"
    meta["来源"] = {**dict(m0.get("来源", {})), "既有": str(既有), "扩容": str(新),
                    "合并时间": __import__("time").strftime("%Y-%m-%d %H:%M:%S")}
    (出 / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    报 = dict(旧条数=n0, 新条数=len(新记录), 合并条数=len(样本),
              train=len(train), val=len(val), 出=str(出))
    (出 / "合并报告.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(报, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(主())
