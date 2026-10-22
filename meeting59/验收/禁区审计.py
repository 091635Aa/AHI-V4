# -*- coding: utf-8 -*-
"""meeting59 静态审计（qa 独立）

三项（简报 §4.5 + §1 硬约束）：
 1) 禁 import 扫描：meeting59/**/*.py 实际 import 的模块名，与禁名单交集；
 2) 产物禁区：meeting58/产物/视频/ 目录 mtime 与逐文件 sha256 必须与基线一致，
    meeting58 下已有 .py、meeting32/声带.py sha256 必须一致（C14 只读）；
 3) 无视频新增：meeting59 内不得出现视频后缀文件；视频目录不得新增/删除文件。

用法：
  python 禁区审计.py --基线      # 开跑前记录一次（qa 阶段A已执行）
  python 禁区审计.py [--json 路径]  # 比对并输出结论（阶段B）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

本目录 = os.path.dirname(os.path.abspath(__file__))
meeting59 = os.path.dirname(本目录)
AHI4 = os.path.dirname(meeting59)
视频目录 = os.path.join(AHI4, "meeting58", "产物", "视频")
声带路径 = os.path.join(AHI4, "meeting32", "声带.py")
基线路径 = os.path.join(本目录, "基线_产物禁区.json")
禁import = ("torch", "transformers", "rwkv", "pyttsx3", "onnxruntime", "tts",
           "tensorflow", "jax", "paddle", "whisper", "funasr", "rvc", "fairseq",
           "espeak", "edge_tts", "sovits", "vits", "hifigan", "vocoder")
视频后缀 = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".wmv", ".m4v")


def sha256文件(路径):
    h = hashlib.sha256()
    with open(路径, "rb") as f:
        for 块 in iter(lambda: f.read(1 << 20), b""):
            h.update(块)
    return h.hexdigest()


def 目录快照(目录, 含哈希=True, 后缀=None):
    out = {"存在": os.path.isdir(目录), "目录": 目录,
           "目录mtime": None, "文件": []}
    if not out["存在"]:
        return out
    out["目录mtime"] = os.stat(目录).st_mtime
    for 根, _, 文件 in os.walk(目录):
        for 名 in sorted(文件):
            if 后缀 and not 名.lower().endswith(后缀):
                continue
            p = os.path.join(根, 名)
            st = os.stat(p)
            条 = {"相对": os.path.relpath(p, 目录), "大小": st.st_size,
                  "mtime": st.st_mtime}
            if 含哈希:
                条["sha256"] = sha256文件(p)
            out["文件"].append(条)
    return out


def py快照(目录):
    out = {}
    if not os.path.isdir(目录):
        return out
    for 根, _, 文件 in os.walk(目录):
        for 名 in sorted(文件):
            if 名.endswith(".py"):
                p = os.path.join(根, 名)
                out[os.path.relpath(p, 目录)] = sha256文件(p)
    return out


def 记录基线():
    b = {
        "记录时间": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "视频目录": 目录快照(视频目录, 含哈希=True),
        "meeting58_py": py快照(os.path.join(AHI4, "meeting58")),
        "声带py": ({"sha256": sha256文件(声带路径),
                   "大小": os.path.getsize(声带路径),
                   "mtime": os.stat(声带路径).st_mtime} if os.path.isfile(声带路径) else {}),
        "meeting59_py": py快照(meeting59),
    }
    with open(基线路径, "w", encoding="utf-8") as f:
        json.dump(b, f, ensure_ascii=False, indent=1)
    print("基线已写入 %s（视频文件 %d 个）" % (基线路径, len(b["视频目录"]["文件"])))
    return b


def 扫禁import():
    import ast

    命中 = []
    扫描 = []
    for 根, _, 文件 in os.walk(meeting59):
        for 名 in sorted(文件):
            if not 名.endswith(".py"):
                continue
            p = os.path.join(根, 名)
            rel = os.path.relpath(p, meeting59)
            try:
                src = open(p, "r", encoding="utf-8", errors="replace").read()
                src = src.lstrip("\ufeff")          # 容忍 Windows 工具生成的 BOM
                tree = ast.parse(src)
            except Exception as e:  # noqa: BLE001
                命中.append({"文件": rel, "错误": "解析失败 %s" % e})
                continue
            模块 = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        模块.append(a.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module:
                        模块.append(node.module.split(".")[0])
                    elif node.level > 0:
                        for a in node.names:
                            模块.append(a.name)
            模块 = sorted(set(模块))
            扫描.append({"文件": rel, "import": 模块})
            for m in 模块:
                if m.lower() in 禁import:
                    命中.append({"文件": rel, "禁模块": m})
    return {"命中": 命中, "扫描文件": 扫描,
            "PASS": len(命中) == 0}


def 检查视频新增():
    视频后缀命 = []
    for 根, _, 文件 in os.walk(meeting59):
        for 名 in 文件:
            if 名.lower().endswith(视频后缀):
                视频后缀命.append(os.path.relpath(os.path.join(根, 名), meeting59))
    return {"meeting59_视频文件": 视频后缀命,
            "PASS": len(视频后缀命) == 0}


def 比对基线(b):
    结果 = {"项": [], "PASS": True}

    def 加项(名, 通过, 读数):
        结果["项"].append({"名": 名, "通过": bool(通过), "读数": 读数})
        if not 通过:
            结果["PASS"] = False

    # 1) 视频目录 mtime
    now = 目录快照(视频目录, 含哈希=False)
    if not b["视频目录"].get("存在"):
        加项("视频目录存在", False, {"基线存在": False, "现状": now["存在"]})
    else:
        加项("视频目录mtime未变", now["目录mtime"] == b["视频目录"]["目录mtime"],
             {"基线": b["视频目录"]["目录mtime"], "现状": now["目录mtime"]})
        # 2) 文件集合 + 哈希
        基文件 = {f["相对"]: f for f in b["视频目录"]["文件"]}
        现文件 = {f["相对"]: f for f in 目录快照(视频目录, 含哈希=True)["文件"]}
        缺 = sorted(set(基文件) - set(现文件))
        新 = sorted(set(现文件) - set(基文件))
        变 = sorted(k for k in 基文件 if k in 现文件 and 基文件[k]["sha256"] != 现文件[k]["sha256"])
        加项("视频文件无新增/删除/改写", not (缺 or 新 or 变),
             {"新增": 新, "删除": 缺, "改写": 变})
    # 3) meeting58 既有 .py
    现py = py快照(os.path.join(AHI4, "meeting58"))
    基py = b.get("meeting58_py", {})
    新py = sorted(set(现py) - set(基py))
    缺py = sorted(set(基py) - set(现py))
    改py = sorted(k for k in 基py if k in 现py and 基py[k] != 现py[k])
    加项("meeting58既有.py未改", not (缺py or 改py or 新py),
         {"新增": 新py, "删除": 缺py, "改写": 改py})
    # 4) 声带.py
    now声 = {"sha256": sha256文件(声带路径), "大小": os.path.getsize(声带路径),
             "mtime": os.stat(声带路径).st_mtime} if os.path.isfile(声带路径) else {}
    加项("meeting32/声带.py未改", now声.get("sha256") == b.get("声带py", {}).get("sha256"),
         {"基线sha256": b.get("声带py", {}).get("sha256"),
          "现状sha256": now声.get("sha256")})
    return 结果


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--基线", action="store_true")
    ap.add_argument("--json", default=os.path.join(本目录, "审计_静态.json"))
    args = ap.parse_args(argv)
    if args.基线:
        记录基线()
        return 0
    b = json.load(open(基线路径, "r", encoding="utf-8"))
    禁 = 扫禁import()
    视频 = 检查视频新增()
    比对 = 比对基线(b)
    禁["PASS"] = 禁["PASS"] and 视频["PASS"]
    总PASS = 禁["PASS"] and 视频["PASS"] and 比对["PASS"]
    出 = {"总PASS": 总PASS, "禁import": 禁, "视频新增": 视频, "基线比对": 比对,
          "基线记录时间": b.get("记录时间")}
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(出, f, ensure_ascii=False, indent=1)
    print("禁 import: %s（%d 文件）" % ("PASS" if 禁["PASS"] else "FAIL", len(禁["扫描文件"])))
    for h in 禁["命中"]:
        print("  !! %s" % h)
    print("无视频新增: %s %s" % ("PASS" if 视频["PASS"] else "FAIL", 视频["meeting59_视频文件"]))
    for 项 in 比对["项"]:
        print("  %s: %s %s" % (项["名"], "PASS" if 项["通过"] else "FAIL", 项["读数"]))
    print("总: %s → %s" % ("PASS" if 总PASS else "FAIL", args.json))
    return 0 if 总PASS else 1


if __name__ == "__main__":
    sys.exit(main())
