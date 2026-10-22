# -*- coding: utf-8 -*-
"""AHI4/V4主程序.py —— AHI V4 根启动器（多核心贯通）。

用法（在 AHI4 目录下）：
    python V4主程序.py                  # 贯通演示：六内核宿主 × 两区室核，跑 120 拍
    python V4主程序.py 贯通 300          # 跑 300 拍
    python V4主程序.py 判据              # V4 总判据（16 臂）
    python V4主程序.py 贯通判据          # 多核心贯通判据
    python V4主程序.py 规则              # 规则表征（后继表征 + 三因子）冒烟
    python V4主程序.py 规则判据          # 规则表征判据（8 臂）
    python V4主程序.py 底座 ...          # 转 V3 底座主程序（四模块/刷网页/交互/自主）

V4 = AHI3（复制，未改）+ 两区室真接线 + 六内核贯通 + meeting50 规则表征旁挂。详 [AGENTS.md](AGENTS.md)。
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
M49 = os.path.join(HERE, "meeting49", "实施")
M50 = os.path.join(HERE, "meeting50", "实施")
M38 = os.path.join(HERE, "meeting38", "实施")
for p in (M50, M49, M38, os.path.join(HERE, "meeting38"), os.path.join(HERE, "meeting31"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def _贯通(拍数: int = 120) -> int:
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("贯通_多核心", os.path.join(M49, "贯通_多核心.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    双 = m.造双区室()
    果 = m.跑(双, 拍数)
    print(json.dumps(dict(已跑=果["已跑"], 异常=果["异常"], 内核态=果["内核态"],
                          自我态有限=果["自我态有限"], 自我态界=round(果["自我态界"], 6)),
                     ensure_ascii=False, indent=1, default=str))
    print("[V4主程序] 多核心贯通：%s" % ("OK" if 果["异常"] is None else "FAIL"))
    return 0 if 果["异常"] is None else 1


def main(argv) -> int:
    a = list(argv)
    if not a:
        return _贯通(120)
    头 = a[0]
    if 头 in ("贯通", "贯通演示"):
        拍数 = int(a[1]) if len(a) > 1 and a[1].isdigit() else 120
        return _贯通(拍数)
    if 头 == "判据":
        runpy.run_path(os.path.join(M49, "判据_V4_两区室核.py"), run_name="__main__")
        return 0
    if 头 == "贯通判据":
        runpy.run_path(os.path.join(M49, "判据_V4_贯通.py"), run_name="__main__")
        return 0
    if 头 == "规则":
        runpy.run_path(os.path.join(M50, "算子_规则表征.py"), run_name="__main__")
        return 0
    if 头 == "规则判据":
        runpy.run_path(os.path.join(M50, "判据_规则表征.py"), run_name="__main__")
        return 0
    if 头 == "底座":
        runpy.run_path(os.path.join(M38, "V3主程序.py"), run_name="__main__")
        return 0
    print("未知子命令：%s（可用：贯通 [N] / 判据 / 贯通判据 / 规则 / 规则判据 / 底座 ...）" % 头)
    return 2


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main(sys.argv[1:]))
