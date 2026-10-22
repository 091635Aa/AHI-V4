# -*- coding: utf-8 -*-
"""meeting55/实施/闭环_听决策说.py —— 端到端：听（情绪/风格/语调/声纹）→ 决策网络 → 情感音色说。

真机：麦克风 WASAPI 17 / 扬声器 14（调用一次，夜里音量适中）。
合成用户：男声-低沉 四种情绪作为输入（[自证]）；回复文本用本地 Qwen（越宪对照，标来源）。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "meeting54", "实施"))
import 听_多源 as 听  # noqa: E402
import 说_音色 as 说  # noqa: E402
import 决策_网络 as 决  # noqa: E402
import 语言器官_LLM as LLM  # noqa: E402  (meeting54)

产物 = os.path.join(HERE, "产物")
os.makedirs(产物, exist_ok=True)
用户音色 = "女声-明亮"
助手音色 = "女声-温柔"


def _回复文本(器, 听事件, 决策):
    提示 = ("用户刚才的情绪是%s，语速%s。你决定用%s的情绪、%s语速、%s音量回应。"
            "请只说 1 到 2 句自然的话，不要括号动作/表情，不要解释你是 AI。"
            % (听事件["情绪"]["情绪"], 听事件["风格"]["语速"],
               决策["回应情绪"], 决策["语速"], 决策["音量"]))
    try:
        r = 器.说(dict(角色="EmoCompanion", 人设="温暖、爱聊天、记得观众", 记忆条目列=[]), "看到第1路；内心步数=1")
        # 直接走对话接口更稳：用提示作为用户话
        return 器.说(dict(角色="EmoCompanion", 人设="温暖、爱聊天、记得观众", 记忆条目列=[]),
                    "看到第1路；内心步数=1")["文"]
    except Exception:
        return "我听见你了，慢一点也没关系。"


def 主跑():
    记录 = dict(回合=[], 真机={}, 备注={})
    器 = LLM.语言器官()
    for 情 in ("高兴", "低落", "平静", "急促"):
        t0 = time.time()
        # 1) 造一个「用户」输入（合成，[自证]）
        u = 说.说("这句话没出现在标定集里，你听得出我的心情吗？", 音色=用户音色, 情绪=情)
        用户wav = os.path.join(说.产物, "发声", u["wav"])
        # 2) 听
        ev = 听.听wav(用户wav, 声纹名=用户音色)
        # 3) 决策
        dec = 决.决策(ev, 声纹名=用户音色)
        # 4) 回复文本（Qwen，越宪对照；把决策写进场景）
        try:
            txt = 器.说(
                dict(角色="EmoCompanion", 人设="温暖、爱聊天、记得观众", 记忆条目列=[]),
                "看到第1路；内心步数=1",
                场景="用户情绪=%s，语速=%s。你决定用%s的情绪、%s语速、%s音量回应，只说 1 到 2 句。"
                     % (ev["情绪"]["情绪"], ev["风格"]["语速"], dec["回应情绪"], dec["语速"], dec["音量"]))["文"]
        except Exception:
            txt = "我听见你了，慢慢说，我在。"
        # 5) 情感音色说
        速倍 = {"慢": 0.85, "中": 1.0, "快": 1.15}[dec["语速"]]
        量倍 = {"低": 0.8, "中": 1.0, "高": 1.15}[dec["音量"]]
        a = 说.说(txt, 音色=助手音色, 情绪=dec["回应情绪"], 语速倍=速倍, 音量倍=量倍)
        用时 = time.time() - t0
        记录["回合"].append(dict(
            用户情绪=情, 听=dict(情绪=ev["情绪"], 风格=ev["风格"], 语调=ev["语调"], 声纹=ev["声纹"]),
            决策=dec, 回复=txt, 助手情绪=dec["回应情绪"], 语速倍=速倍, 音量倍=量倍,
            音频=os.path.basename(a["wav"]), 用时秒=round(用时, 2)))
    # 6) 真机：录一段时间（夜里应无声）+ 播放一句（调用扬声器）
    try:
        x = 听.录音(1.5)
        rms = float(np.sqrt(np.mean(x ** 2)))
        记录["真机"]["麦克风"] = dict(RMS=round(rms, 6), 时长=round(len(x) / 16000, 2),
                                    有声=bool(rms > 0.004))
    except Exception as e:
        记录["真机"]["麦克风"] = dict(错误=repr(e)[:120])
    try:
        播 = 说.说("我在听，你先睡吧。", 音色=助手音色, 情绪="平静")
        记录["真机"]["扬声器"] = dict(wav=os.path.basename(播["wav"]),
                                    已播放=bool(说.播放(os.path.join(说.产物, "发声", 播["wav"]))))
    except Exception as e:
        记录["真机"]["扬声器"] = dict(错误=repr(e)[:120])
    记录["备注"] = dict(模型=器.台账["模型"], 权重码=器.权重码, 越宪="Qwen 仅作回复文本对照",
                       声纹="合成标定[自证]；真人未注册", 用户音色=用户音色, 助手音色=助手音色)
    return 记录


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 主跑()
    with open(os.path.join(产物, "闭环结果.json"), "w", encoding="utf-8") as f:
        json.dump(果, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(产物, "闭环对话.txt"), "w", encoding="utf-8") as f:
        for r in 果["回合"]:
            f.write("用户（%s）→ 听：%s/%s/%s 声纹=%s\n" % (
                r["用户情绪"], r["听"]["情绪"]["情绪"], r["听"]["风格"], r["听"]["语调"], r["听"]["声纹"]["名"]))
            f.write("  决策：%s / 语速%s / 音量%s\n" % (r["决策"]["回应情绪"], r["决策"]["语速"], r["决策"]["音量"]))
            f.write("  回复：%s\n  音频：%s（用时 %.2fs）\n\n" % (r["回复"], r["音频"], r["用时秒"]))
        f.write("真机：%s\n" % json.dumps(果["真机"], ensure_ascii=False))
    print(json.dumps(果, ensure_ascii=False, indent=1)[:3000])
