# -*- coding: utf-8 -*-
"""meeting31/写作模式.py —— #253 长时持续输出：读写交替循环（写→耗语料→读→补语料→写）。

用法：python 写作模式.py [--分钟 0] [--拍率 6.0] [--seed 311]
  --分钟 0 = 无限（24 小时不间断输出测试用）
产物：输出/作品.md（持续追加）、写作日志.log（每 30s 进度）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"<workspace>/03_AHI\AHI2.1"
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from synthesis import 综合体
from writer import 写作体

备份目录 = os.path.join(ROOT, "backup_20261004_v10自言体")
屏蔽令牌 = "backup_20261004"
指示词 = ("刚才", "刚刚", "你说", "你之前", "上面")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--分钟", type=float, default=0.0, help="0=无限")
    ap.add_argument("--拍率", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=311)
    args = ap.parse_args()
    t0 = time.time()
    print("=" * 72)
    print("#253 写作模式：读写交替持续输出 ｜ %s ｜ %.1f 拍/秒" %
          ("无限" if args.分钟 <= 0 else "%.0f 分钟" % args.分钟, args.拍率))
    print("=" * 72)

    盘s = [d for d in "EFGHIJKLMN" if os.path.exists(d + ":\\")]
    桥 = 综合体(seed=args.seed)
    桥.设开放域([d + ":\\" for d in 盘s] + [r"<workspace>/"], [备份目录])

    输出目录 = os.path.join(HERE, "输出")
    os.makedirs(输出目录, exist_ok=True)
    作品 = os.path.join(输出目录, "作品.md")
    写 = 写作体(桥, 作品)

    # 看门狗
    import subprocess
    st = os.path.join(HERE, "status", "status.json")
    estop = os.path.join(HERE, "status", "ESTOP")
    os.makedirs(os.path.dirname(st), exist_ok=True)
    if os.path.exists(estop):
        os.remove(estop)
    wd = subprocess.Popen([sys.executable, os.path.join(ROOT, "meeting24", "watchdog.py"),
                           "--status", st, "--pid", str(os.getpid())],
                          cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("【看门狗】挂载（pid=%d）" % os.getpid())

    import threading
    import json as _json
    _锁 = threading.Lock()
    _态 = dict(拍=0)

    def _心跳():
        while True:
            try:
                with _锁:
                    拍号, 最近动作 = _态["拍"], list(桥.动作史[-100:])
                tmp = st + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    _json.dump(dict(拍号=拍号, 引导拍=桥.引导拍, κ=桥.私偏κ,
                                    最近动作=最近动作), f, ensure_ascii=False)
                try:
                    os.replace(tmp, st)
                except PermissionError:
                    pass
            except Exception:
                pass
            time.sleep(1.0)

    threading.Thread(target=_心跳, daemon=True).start()

    日志 = os.path.join(HERE, "写作日志.log")
    拍间 = 1.0 / args.拍率
    下拍 = time.time()
    下一点 = 30.0
    静默 = 0
    补读 = 0
    while args.分钟 <= 0 or time.time() - t0 < args.分钟 * 60.0:
        el = time.time() - t0
        # 读写交替：写枯竭（静默≥4）→ 补 6 拍阅读（语料再生+轮转游标跳进）→ 恢复写作
        if 补读 > 0:
            桥.断供一拍()                     # 探索拍（读，补语料）
            补读 -= 1
            if 补读 == 0:
                静默 = 0
                写.题材游标 += 13              # 题材轮标跳进（换一批语料）
        else:
            n = 写.写一句()                    # 写作拍（输出）
            if n == 0:
                静默 += 1
                if 静默 >= 4:
                    补读 = 6
            else:
                静默 = 0
        with _锁:
            _态["拍"] = _态["拍"] + 1
        下拍 += 拍间
        剩 = 下拍 - time.time()
        if 剩 > 0:
            time.sleep(剩)
        else:
            下拍 = time.time()
        if el >= 下一点:
            下一点 += 30.0
            行 = ("[%02d:%02d] 句 %d ｜ 字 %d ｜ 唯一 %d ｜ 静默 %d ｜ 细读语料 %d 行" %
                  (el // 60, el % 60, 写.句数, 写.字数, len(写.唯一句),
                   静默, sum(1 for t, _ in 桥.账本 if t == "教材")))
            print(行)
            with open(日志, "a", encoding="utf-8") as f:
                f.write(行 + "\n")

    try:
        wd.terminate()
    except Exception:
        pass
    print("结束：句 %d ｜ 字 %d ｜ 唯一 %d" % (写.句数, 写.字数, len(写.唯一句)))


if __name__ == "__main__":
    main()
