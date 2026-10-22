# -*- coding: utf-8 -*-
"""步0 修复脚本：把 backup #244 中缺失的路线函数区恢复进 verify_all.py。

发现过程：meeting38 容量方案 步0 基线冻结时，run_verify.py 报
ImportError: cannot import name 'ALL' —— verify_all.py 只剩辅助函数+_CHANGELOG，
路线函数区（_route / verify_P1..P8 / N/W/S/M/I/K 组 / ALL 注册表）整体缺失。
恢复来源：backup_20261004_v10自言体/ahi_theory_verify_all_#244.py 第5170-7223行
（原样，一字不改）；当前文件的 _CHANGELOG（含 2026-10-04 v10.x 条目）不动，仅按
本文件纪律追加一条修复留痕。
"""
import io

CUR = r"<workspace>/03_AHI/AHI2.1/ahi_theory/verify_all.py"
BAK = r"<workspace>/03_AHI/AHI2.1/backup_20261004_v10自言体/ahi_theory_verify_all_#244.py"

cur = io.open(CUR, encoding="utf-8").read()
bak = io.open(BAK, encoding="utf-8").read()

assert cur.rstrip().endswith("]"), "当前文件结尾不是变更日志闭合 ]"
blines = bak.split("\n")
# 备份第5169行(1基)=变更日志闭合 ]；其后的 5170-7223 行即缺失区
missing = "\n".join(blines[5169:]).rstrip() + "\n"
assert missing.lstrip().startswith("def _route"), missing[:80]
assert "ALL = [(" in missing and "K_GROUP = [(" in missing, "恢复区不完整"

entry = (
    '    ("2026-10-05", '
    '"修复：本文件曾只剩辅助函数+变更日志，路线函数区（_route/verify_P1..P8/'
    'N/W/S/M/I/K 组/ALL 注册表）缺失；从 backup_20261004_v10自言体/'
    'ahi_theory_verify_all_#244.py 第5170-7223行原样恢复，未改动变更日志区与任何判据。'
    '发现于 meeting38 容量方案 步0 基线冻结。",\n'
    '     "恢复路线区"),\n'
)

lines = cur.rstrip("\r\n").split("\n")
assert lines[-1].strip() == "]", "最后一行不是变更日志闭合 ]"
# 在闭合 ] 前插入修复留痕条目（保持该行原有缩进风格：顶格 ]）
lines.insert(len(lines) - 1, entry.rstrip("\n"))
cur2 = "\n".join(lines)

out = (cur2.rstrip()
       + "\n\n# ===== meeting38 步0 修复：以下路线区自 backup #244 第5170-7223行原样恢复 =====\n"
       + missing)

io.open(CUR, "w", encoding="utf-8", newline="\n").write(out)
print("written chars:", len(out))
