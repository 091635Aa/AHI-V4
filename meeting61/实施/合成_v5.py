# -*- coding: utf-8 -*-
"""meeting61/实施/合成_v5.py —— v5 试听交付（音韵/协同发音修复版）。

组：
  V5_三角色      EmoCompanion/SpeakerA/洛天依 各三句（与 v4 同文本，便于 A/B）
  V5_同句三声    同一句三声
  V5_辅音表      塞/擦/塞擦/浊声母最小对（CV 音节）
  V5_元音表      a o e i u ü er ai ei ao ou an en ang eng
  V5_四声        妈麻马骂
  V5_参考对照    参考原声 → v5 合成（同角色）
主交付：产物/修复版v5/修复版v5_三角色+同句.wav（含全部试听素材的总装另见各文件）

复用 v4 的响度/拼接/参考片段工具，不改 v4 任何文件。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEET61 = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import 韵律_v5 as H5  # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402
from 合成_v4 import 静音, 拼, 响度, 写wav, 参考片段  # noqa: E402

FS = H5.FS
产物组 = os.path.join(MEET61, "产物", "测试组")
产物主 = os.path.join(MEET61, "产物", "修复版v5")
产物档案 = os.path.join(MEET61, "产物", "声线档案")
os.makedirs(产物组, exist_ok=True)
os.makedirs(产物主, exist_ok=True)

顺序 = ("EmoCompanion", "SpeakerA", "洛天依")
种子 = {"EmoCompanion": 300, "SpeakerA": 317, "洛天依": 334}
角色文本 = {
    "EmoCompanion": ["你好，我是EmoCompanion。",
             "今天的风很轻，适合慢慢地说一会儿话。",
             "你最近，过得好吗？"],
    "SpeakerA": ["嘿，我是SpeakerA。",
             "别看我平时闹得欢，认真说话的时候，我也能安静下来。",
             "要不要，一起待一会儿？"],
    "洛天依": ["大家好，我是洛天依。",
               "谢谢你愿意听我说话。",
               "希望今天的声音，能让你心里暖一点。"],
}
同句 = "语言不只是声音，它把一个人的温度，送到另一个人的心里。"
辅音表 = ["ba1", "pa1", "ma1", "fa1", "da1", "ta1", "na1", "la1",
        "ga1", "ka1", "ha1", "sa1", "sha1", "xia1", "ji1", "qi1", "xi1",
        "zhi1", "chi1", "shi1", "ri4", "zi1", "ci4", "si1"]
元音表 = ["a1", "o1", "e1", "i1", "u1", "v1", "er2", "ai1", "ei1",
        "ao1", "ou1", "an1", "en1", "ang1", "eng1"]
四声表 = ["ma1", "ma2", "ma3", "ma4"]


def 读档案(名: str) -> 声线档案:
    with open(os.path.join(产物档案, 名 + ".json"), encoding="utf-8") as f:
        return 声线档案.从dict(json.load(f))


def 渲染角色(档, 名, 种子偏移=0) -> np.ndarray:
    ws = []
    for i, 句 in enumerate(角色文本[名]):
        x, _, _ = H5.人类韵律_v5(档, 种子[名] + 种子偏移 + i * 11).说(句)
        ws.append(x)
    return 响度(拼(ws, 0.30), 0.06)


def 渲染音节表(档, 拼s, 间隔=0.17, 遍=2) -> np.ndarray:
    ws = []
    for _ in range(遍):
        for p in 拼s:
            x, _, _ = H5.人类韵律_v5(档, 300).说音节(p)
            ws.append(响度(x, 0.06))
    return 响度(拼(ws, 间隔), 0.06)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    档案 = {名: 读档案(名) for 名 in 顺序}
    组 = {}

    d = os.path.join(产物组, "组V5_三角色")
    os.makedirs(d, exist_ok=True)
    角色音 = {}
    for 名 in 顺序:
        角色音[名] = 渲染角色(档案[名], 名)
        写wav(os.path.join(d, "角色_%s.wav" % 名), 角色音[名])
    主三角 = 拼([角色音[名] for 名 in 顺序], 0.55)
    写wav(os.path.join(产物组, "组V5_三角色.wav"), 主三角)
    组["组V5_三角色"] = "v5 主交付：音韵重建 + CV 协同发音 + 声门流源/单次唇辐射"

    同句段 = []
    d = os.path.join(产物组, "组V5_同句三声")
    os.makedirs(d, exist_ok=True)
    for 名 in 顺序:
        x, _, _ = H5.人类韵律_v5(档案[名], 种子[名] + 97).说(同句)
        x = 响度(x, 0.06)
        同句段.append(x)
        写wav(os.path.join(d, "角色_%s.wav" % 名), x)
    同句混 = 响度(拼(同句段, 0.55), 0.06)
    写wav(os.path.join(产物组, "组V5_同句三声.wav"), 同句混)
    组["组V5_同句三声"] = "同一句三声（文本/韵律骨架/时长一致，只换声线）"

    辅 = 渲染音节表(档案["EmoCompanion"], 辅音表)
    写wav(os.path.join(产物组, "组V5_辅音表.wav"), 辅)
    组["组V5_辅音表"] = "CV 最小对：b/p/m/f d/t/n/l g/k/h s/sh/x j/q/x zh/ch/sh/r z/c/s"

    元 = 渲染音节表(档案["EmoCompanion"], 元音表, 遍=2)
    写wav(os.path.join(产物组, "组V5_元音表.wav"), 元)
    组["组V5_元音表"] = "元音/韵母表：a o e i u ü er ai ei ao ou an en ang eng"

    调 = 渲染音节表(档案["EmoCompanion"], 四声表, 间隔=0.30, 遍=3)
    写wav(os.path.join(产物组, "组V5_四声.wav"), 调)
    组["组V5_四声"] = "妈麻马骂（同音节四声，检验声调轮廓）"

    g段 = []
    for 名 in 顺序:
        r = 参考片段(名, 9.0)
        if r.size:
            g段.extend([r, 静音(0.40), 角色音[名][:int(9.5 * FS)], 静音(0.7)])
    if g段:
        写wav(os.path.join(产物组, "组V5_参考对照.wav"), 响度(np.concatenate(g段), 0.06))
        组["组V5_参考对照"] = "每角色：44.1k 参考原声 → v5 合成（A/B）"

    主 = 响度(np.concatenate([主三角, 静音(0.9), 同句混]), 0.06)
    主路径 = os.path.join(产物主, "修复版v5_三角色+同句.wav")
    写wav(主路径, 主)
    写wav(os.path.join(产物主, "修复版v5_辅音表.wav"), 辅)
    写wav(os.path.join(产物主, "修复版v5_元音表.wav"), 元)
    写wav(os.path.join(产物主, "修复版v5_四声.wav"), 调)
    写wav(os.path.join(产物主, "修复版v5_三角色.wav"), 主三角)
    写wav(os.path.join(产物主, "修复版v5_同句三声.wav"), 同句混)
    if g段:
        写wav(os.path.join(产物主, "参考A_B_v5.wav"), 响度(np.concatenate(g段), 0.06))

    with open(os.path.join(产物主, "清单_v5.json"), "w", encoding="utf-8") as f:
        json.dump({"口径": "参数化模仿（非克隆）；纯 numpy/scipy；无 TTS/RVC/外部模型",
                   "采样率": FS, "组": 组,
                   "主文件": os.path.basename(主路径), "主时长秒": round(len(主) / FS, 2)},
                  f, ensure_ascii=False, indent=1)
    for k, v in 组.items():
        print("[组]", k, v)
    print("[主]", 主路径, "%.2fs" % (len(主) / FS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
