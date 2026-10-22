# -*- coding: utf-8 -*-
"""复核探针 A：引擎本体（千层核 D=64/256）逐数组 nbytes + 分类汇总 + 参数/定点换算 + RSS。
只读：不写任何文件；python -B 运行（不落 __pycache__）。
复跑：$env:PYTHONIOENCODING='utf-8'; python -B 黄队\\_探针\\A_引擎本体.py
"""
import sys, time, json
import psutil, numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, r"<workspace>/03_AHI\AHI4\meeting51\实施")
p = psutil.Process()


def rss(tag):
    mi = p.memory_info()
    print("%-16s rss_MB %6.1f  peak_wset_MB %6.1f" %
          (tag, mi.rss / 1048576, mi.peak_wset / 1048576))
    return mi.rss / 1048576, mi.peak_wset / 1048576


MI = 1048576.0
rss("base")
import 算子_千层骨 as q  # noqa: E402
rss("+import")
print("轴默认", q.轴默认, "环路", q.开关默认["环路"], "读出层档", q.开关默认["读出层档"])

参数类 = {"g", "边权", "W_in", "W_in全", "P_入"}
拓扑类 = {"入源", "入类", "分区", "层", "区室", "时档", "外单元", "外分区",
          "读单元", "读单元平", "外分区全", "最近池", "最近活性"}
状态类 = {"A慢", "B快", "输出"}
最近类 = {"最近快", "最近输出", "最近汇"}
读出类 = {"读出", "读出偏", "均", "方"}

for D in (64, 256):
    t0 = time.perf_counter()
    k = q.千层核(D=D, 种子=7)
    dt = time.perf_counter() - t0
    rss("built D=%d" % D)
    groups = {}
    for objname in ("容器", "记忆"):
        obj = getattr(k, objname)
        for name, v in vars(obj).items():
            if isinstance(v, np.ndarray):
                groups[objname + "." + name] = int(v.nbytes)
    for name, v in vars(k).items():
        if isinstance(v, np.ndarray):
            groups[name] = int(v.nbytes)

    def cat_of(name):
        short = name.split(".", 1)[-1]
        if short in 参数类:
            return "1_参数"
        if short in 状态类:
            return "2_状态"
        if short in 最近类:
            return "3_最近scratch"
        if short in 读出类:
            return "4_读出/归一"
        if name.startswith("记忆."):
            return "5_记忆"
        if short in 拓扑类:
            return "6_拓扑/索引"
        return "7_其他"

    分类 = {}
    for name, b in groups.items():
        分类[cat_of(name)] = 分类.get(cat_of(name), 0) + b
    tot = sum(groups.values())
    print("\n== D=%d  build_s %.3f  n_arrays %d  总 nbytes %.3f MiB (%.3f MB十进制)" %
          (D, dt, len(groups), tot / MI, tot / 1e6))
    for c in sorted(分类):
        print("   %-12s %8.3f MiB" % (c, 分类[c] / MI))
    print("   最大 6 个:", [(n, round(b / MI, 3)) for n, b in
                        sorted(groups.items(), key=lambda x: -x[1])[:6]])
    # 黄队 R1 §1.3 口径复算（参数 8.49 + 状态 A/B/输出 7.59 + 拓扑 0.08 + 记忆 0.11）
    r1口径 = (分类.get("1_参数", 0) + 分类.get("2_状态", 0)
              + 分类.get("6_拓扑/索引", 0) + 分类.get("5_记忆", 0))
    print("   R1 §1.3 同口径(参数+状态AB输出+拓扑+记忆) = %.3f MiB" % (r1口径 / MI))
    print("   R1 遗漏项(最近scratch+读出/归一+其他) = %.3f MiB" %
          ((分类.get("3_最近scratch", 0) + 分类.get("4_读出/归一", 0)
            + 分类.get("7_其他", 0)) / MI))
    print("   summary", json.dumps(k.容器.摘要(), ensure_ascii=False))
    print("   params(参数总量) %d  -> int32 打包 %.3f MB十进制 / %.3f MiB" %
          (k.参数总量(), k.参数总量() * 4 / 1e6, k.参数总量() * 4 / MI))
    print("   参数数组 float64 实占 %.3f MiB；状态 float64 %.3f MiB" %
          (分类.get("1_参数", 0) / MI, 分类.get("2_状态", 0) / MI))
    del k
    rss("del D=%d" % D)
rss("end")
