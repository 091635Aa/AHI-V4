# -*- coding: utf-8 -*-
"""meeting31/附身EmoCompanion.py —— #282 人格附身：打标语料+日记→EmoCompanion人格。

附身≠扮演≠模拟（所有者指令）：不套回答模板，而是——
  ①语言血统：她的直播原话（打标 transcript）全量学入EmoCompanion舱（附身模式不走
    <0.5 门控——血统必须完整，所有者明确要求投喂）；
  ②记忆血统：日记三档（幼年模糊/少年细节/成年反思）+五维人设+关键经历入舱；
  ③自我模型：身份模板（identity.py，#262 可审计血统）——背景经自我声道入账、
    风格语料学入自声道、性格参数注入引擎。
唤起判据：问她只有EmoCompanion本人知道的事（开播天数/ViewerA/天鹅之梦/萌妹贴贴）——
命中她自己的事实=唤起；口头禅（啊/那个）频率可复核。
用法：python 附身EmoCompanion.py [--chat]     # 默认摄取+验证；--chat 附身对话
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")

from synthesis import 综合体
from identity import 身份模板

打标目录 = r"<workspace>/打标\数据层"
日记 = os.path.join(打标目录, "日记语料", "EmoCompanion_日记_分档.json")
身份路径 = os.path.join(HERE, "身份_EmoCompanion.json")
口癖 = ("啊", "那个", "嘛", "呀", "诶", "啦", "哦")


# ---------- 摄取：语言血统 + 记忆血统 ----------
def 语料行s():
    """全量血统行：(文本, 来源)。transcript=她的原话；标注=她的世界事实。"""
    for p in sorted(glob.glob(os.path.join(打标目录, "打标结果", "*.json"))):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        t = (d.get("transcript") or "").strip()
        if 6 <= len(t) <= 300:
            yield t, os.path.basename(p)
        标签s = d.get("情感标签") or []
        主题s = d.get("主题标签") or []
        描述 = (d.get("内容描述") or "").strip()
        if 描述 and 6 <= len(描述) <= 160:
            yield "（这段直播她在做什么：%s｜情感：%s｜主题：%s）" % (
                描述, "/".join(标签s[:3]), "/".join(主题s[:3])), os.path.basename(p)
    for p in sorted(glob.glob(os.path.join(打标目录, "日记语料",
                                           "片段_*.json"))):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        t = (d.get("transcript") or "").strip()
        if 6 <= len(t) <= 300:
            yield t, os.path.basename(p)


def 记忆行s():
    """日记+人设：记忆血统（她的过去只有她知道）。"""
    d = json.load(open(日记, encoding="utf-8"))
    rs = d.get("人设", {})
    行s = []
    行s.append("我叫%s，%s年生。" % (rs.get("姓名", "EmoCompanion"),
                                 rs.get("出生年份", "2003")))
    行s.append("性格：%s" % rs.get("性格", ""))
    行s.append(rs.get("人设描述", ""))
    for e in (rs.get("关键经历") or []):
        行s.append("我经历过：%s" % e)
    for r in (d.get("日记列表") or []):
        行s.append("日记（%s·%s）：%s" % (r.get("年龄段", ""),
                                     r.get("日期", ""), r.get("正文", "")))
    return [x for x in 行s if len(x) >= 6]


def 摄取():
    体 = 综合体(seed=2003)                     # 她的出生年份做 seed
    体.设开放域([], [])
    体.挂舱("EmoCompanion")
    舱 = 体.舱s["EmoCompanion"]
    n语 = n记 = 0
    for t, 源 in 语料行s():
        舱.存行("[%s] %s" % (源[:14], t))       # 附身模式：全量，不门控
        n语 += 1
    for t in 记忆行s():
        舱.存行(t)
        n记 += 1
    体.账本.append(("接收", "附身：EmoCompanion语料 %d 行+记忆 %d 行全量入EmoCompanion舱"
                    "（所有者指令：投喂直播回放打标，1:1 附身）" % (n语, n记)))
    return 体, n语, n记


# ---------- 自我模型：身份模板 ----------
def 生成身份():
    原话s = []
    for p in sorted(glob.glob(os.path.join(打标目录, "打标结果", "*.json"))):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        t = (d.get("transcript") or "").strip()
        if 10 <= len(t) <= 60 and any(w in t for w in 口癖):
            原话s.append(t)
    风格 = 原话s[:36]
    d0 = json.load(open(日记, encoding="utf-8"))
    rs = d0.get("人设", {})
    身份 = {
        "名字": rs.get("姓名", "EmoCompanion"),
        "性格描述": rs.get("人设描述", ""),
        "背景": [
            "我是EmoCompanion，%s年生的女主播，直播间叫'EmoCompanion的直播间'。"
            % rs.get("出生年份", "2003"),
            "我刚开播不久，是个新人主播。",
            "口癖是'啊、那个'，紧张时会清嗓子。",
            "遇到过的观众里有一位叫ViewerA。",
            "开播第十五天是'天鹅之梦'礼物存在的最后一天，我鼓起勇气请求观众'撇一只'。",
        ],
        "风格语料": 风格,
        "参数": {"私偏κ": 14.0},
    }
    with open(身份路径, "w", encoding="utf-8") as f:
        json.dump(身份, f, ensure_ascii=False, indent=1)
    return 身份, len(风格)


# ---------- 唤起验证 ----------
def 唤起验证(体):
    问s = ["你是谁？开播多久了？",
           "ViewerA是谁？",
           "天鹅之梦那天你干了什么？",
           "被人叫萌妹是什么感觉？"]
    out = []
    for q in 问s:
        体.步(q)
        n = 0
        while 体.思考中 is not None and n < 30:
            体.断供一拍()
            n += 1
        答 = 体.回复台账[-1]["响应"] if 体.回复台账 else "（无回复）"
        out.append((q, str(答)[:160]))
    return out


def 口癖频率(体):
    文 = " ".join(体.回复台账[-1]["响应"] for _ in [0]
                  if 体.回复台账) if 体.回复台账 else ""
    return {w: 文.count(w) for w in 口癖}


def main():
    附身 = "--chat" not in sys.argv
    体, n语, n记 = 摄取()
    print("[附身] EmoCompanion舱：语料 %d 行 + 记忆 %d 行（全量，未门控）" % (n语, n记))
    身份, n风 = 生成身份()
    print("[附身] 身份模板：风格语料 %d 句（口癖原话）" % n风)
    体.加载身份(身份模板.从文件(身份路径))
    print("[附身] 身份已加载：血统可审计（背景→自我声道，风格→自声道）")
    if 附身:
        for q, a in 唤起验证(体):
            print("问> %s\nEmoCompanion> %s\n" % (q, a))
    else:
        print("EmoCompanion> 啊，那个……我在呢，想聊什么呀？（/exit 退出）")
        while True:
            try:
                q = input("你> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q == "/exit":
                break
            体.步(q)
            n = 0
            while 体.思考中 is not None and n < 30:
                体.断供一拍()
                n += 1
            print("EmoCompanion> %s" % (体.回复台账[-1]["响应"] if 体.回复台账 else "……"))


if __name__ == "__main__":
    main()
