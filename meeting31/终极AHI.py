# -*- coding: utf-8 -*-
"""meeting31/终极AHI.py —— #277 终极版自主意识体：全挂载+浏览器搜索+全程留痕。

挂载：开放域探索（E–N+主目录，屏蔽备份）+ 三数据舱（魔塔）+ 身份模板（小澄）
     + 消化内驱 + 延答思考 + 推理环 + 自言流 + 亚思考旁路 + 自主搜索
留痕：自言流（思考过程）、旁路遥测（向量→词）、搜索日志、对话记录（--chat 模式）
用法：python 终极AHI.py            # 无限自主探索（含自主搜索）
     python 终极AHI.py --chat     # 对话模式（他答你问，全程记录）
"""
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")

from synthesis import 综合体
from identity import 身份模板
from 旁路 import 旁路可视化器
from 搜索工具 import 搜索
from probe_魔塔 import 取行, 数据目录, 每舱上限, 源

屏蔽 = ["<workspace>/\03_AHI\\AHI2.1\\backup_20261004_v10自言体"]
盘符 = ["%<path>\\" % c for c in "EFGHIJKLMN"] + ["<workspace>/"]


def 建体(seed=777):
    桥 = 综合体(seed=seed)
    桥.设开放域(盘符, 屏蔽)
    身份 = os.path.join(HERE, "身份模板示例.json")
    if os.path.exists(身份):
        桥.加载身份(身份模板.从文件(身份))
    for 域名, ds, 路径, 种类 in 源:
        目标 = os.path.join(数据目录, 路径.replace("/", "_"))
        if os.path.exists(目标):
            桥.挂舱(域名)
            舱 = 桥.舱s[域名]
            for 行 in 取行(目标, 种类, 每舱上限):
                if 舱.命中(行) < 0.5:
                    舱.存行(行)
    return 桥


class 留痕:
    def __init__(self, 桥):
        self.桥 = 桥
        self.旁 = 旁路可视化器(桥, os.path.join(HERE, "旁路.jsonl"))
        self.对话路径 = os.path.join(HERE, "对话记录.log")

    def 对话(self, 谁, 文本):
        with open(self.对话路径, "a", encoding="utf-8") as f:
            f.write("[%s] %s: %s\n" % (time.strftime("%H:%M:%S"), 谁, 文本))


def 主循环(桥, 记):
    n = 0
    while True:
        try:
            感知 = 桥.导航器.当前文件 or "-"
            # #283 常思考：自由联想拍——无外部问题时也持续思考（像 agent 的
            # 反思循环，但对象来自他自己的账本/自言，非外部指定）
            if 桥.思考中 is None and 桥.思考队列:
                q = 桥.思考队列.popleft()
                桥.思考中 = dict(问题=q, 到达拍=桥.拍, 证据=[], 拍数=0)
                print("开想> %s" % q[:50], flush=True)
            if 桥.思考中 is not None:
                前 = len(桥.回复台账)
                桥._思考一拍()
                if 桥.思考中 is None and len(桥.回复台账) > 前:
                    print("想毕> %s" % str(桥.回复台账[-1].get("响应", ""))[:90],
                          flush=True)
            elif n % 8 == 0 and not 桥.思考队列:
                最近 = 桥.账本[-1][1][:24] if 桥.账本 else "我自己"
                桥.思考队列.append("关于『%s』我还能想到什么？" % 最近)
                print("自由思考> %s" % 最近, flush=True)
            桥.断供一拍()
            动作 = 桥.动作史[-1] if 桥.动作史 else "-"
            记.旁.逐拍({}, 动作, 感知)
            n += 1
            if n % 40 == 0:                       # 自主搜索：用他自己的疑问驱动
                问s = [s for s in 桥.自言.自言流(30, "疑问")
                       if isinstance(s, str)]
                词 = ""
                for s in reversed(问s):
                    t = s.strip("『』。 ")
                    if 4 <= len(t) <= 30:
                        词 = t[:24]
                        break
                if 词:
                    try:
                        结果 = 搜索(词)
                        桥._自言("假设", "【搜索】我自己想知道『%s』——查到 %d 条。"
                                 % (词, len(结果)), ("内生", "自主搜索"), [])
                        for 标, 摘 in 结果[:5]:
                            行 = "%s：%s" % (标, 摘)
                            if 桥.当前舱.命中(行) < 0.5 and 10 <= len(行) <= 200:
                                桥.当前舱.存行(行)   # 门控：预测不了的才学
                    except Exception as e:
                        桥._自言("复盘", "【搜索失败】%s" % type(e).__name__,
                                 ("内生", "自主搜索"), [])
            if n % 100 == 0:
                st = os.path.join(HERE, "status")
                os.makedirs(st, exist_ok=True)
                with open(os.path.join(st, "status.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(dict(拍号=桥.拍, 时刻=time.time()), f)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            with open(os.path.join(HERE, "终极异常.log"), "a",
                      encoding="utf-8") as f:
                f.write("[%s] %s: %s\n" % (time.strftime("%H:%M:%S"),
                                           type(e).__name__, str(e)[:120]))
            time.sleep(0.2)


def 对话模式(桥, 记):
    print("终极 AHI 对话模式（/mind 看思考，/exit 退出）")
    while True:
        try:
            q = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q == "/exit":
            break
        if q == "/mind":
            for s in 桥.自言.自言流(15):
                print("  自言>", s)
            continue
        记.对话("你", q)
        r = 桥.步(q)
        记.对话("他(即时)", r["响应"])
        print("他> %s" % r["响应"])
        t0 = time.time()
        while 桥.思考中 is not None:
            桥.断供一拍()
        if len(桥.回复台账) and time.time() - t0 > 0.5:
            深 = 桥.回复台账[-1]["响应"]
            记.对话("他(深思)", 深)
            print("他(深思)> %s" % 深)


if __name__ == "__main__":
    桥 = 建体()
    记 = 留痕(桥)
    if "--chat" in sys.argv:
        对话模式(桥, 记)
    else:
        print("终极 AHI 无限探索启动（搜索每 40 拍一次，全留痕）")
        主循环(桥, 记)
