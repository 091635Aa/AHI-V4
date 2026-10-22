# -*- coding: utf-8 -*-
"""生成人耳听测包：v8.1 的 6 句 × 3 角色（随机顺序、去标签）+ 3 条真人参考。

只读 meeting63 产物/素材；只写 meeting64/产物/听测/（C14）。
确定性：固定 seed=641，同输入同输出。
"""
import json
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M64 = os.path.dirname(HERE)
AHI4 = os.path.dirname(M64)
M63 = os.path.join(AHI4, "meeting63")
M61 = os.path.join(AHI4, "meeting61")

源目录 = os.path.join(M63, "产物", "ASR临时", "多句")
出目录 = os.path.join(M64, "产物", "听测")
os.makedirs(出目录, exist_ok=True)

句集 = [
    ("S1", "今天的风很轻，适合慢慢地说一会儿话。"),
    ("S2", "我们明天早上八点在南门见面。"),
    ("S3", "他把三本书放在窗台上。"),
    ("S4", "小猫从桌上跳下来，碰到杯子。"),
    ("S5", "春天来了，山上的雪开始融化。"),
    ("S6", "请问这条路怎么走？"),
]
角色s = ("EmoCompanion", "SpeakerA", "洛天依")

rng = np.random.default_rng(641)
序 = list(range(18))
rng.shuffle(序)

键 = []
for 新号, 原号 in enumerate(序, 1):
    句号, 句 = 句集[原号 // 3]
    角色 = 角色s[原号 % 3]
    源 = os.path.join(源目录, "发声_%06d.wav" % (原号 + 1))
    目标 = os.path.join(出目录, "听测_%02d.wav" % 新号)
    assert os.path.exists(源), 源
    shutil.copyfile(源, 目标)
    键.append(dict(文件="听测_%02d.wav" % 新号, 类型="v8.1合成", 句号=句号, 句=句, 角色=角色))

对照 = [
    ("EmoCompanion", os.path.join(M61, "素材", "EmoCompanion", "ref44_EmoCompanion_01.wav")),
    ("SpeakerA", os.path.join(M61, "素材", "SpeakerA", "ref44_SpeakerA_01.wav")),
    ("洛天依", os.path.join(M61, "素材", "洛天依", "ref44_洛天依_清唱_01.wav")),
]
rng2 = np.random.default_rng(642)
序2 = list(range(3))
rng2.shuffle(序2)
for 新号, 原号 in enumerate(序2, 1):
    角色, 源 = 对照[原号]
    目标 = os.path.join(出目录, "对照_%02d.wav" % 新号)
    assert os.path.exists(源), 源
    shutil.copyfile(源, 目标)
    键.append(dict(文件="对照_%02d.wav" % 新号, 类型="真人参考", 角色=角色,
                  注="歌唱/朗读素材，内容与上表不同，只作音质锚点"))

with open(os.path.join(出目录, "答案键.json"), "w", encoding="utf-8") as f:
    json.dump(dict(说明="先打分，再对答案", 条目=键), f, ensure_ascii=False, indent=1)

行 = ["| 文件 | 0 完全听不懂 | 1 只猜到大意 | 2 听懂一半 | 3 基本听懂 | 4 完全听懂 | 备注 |",
      "|---|---|---|---|---|---|---|"]
for 条 in 键:
    行.append("| %s | | | | | | |" % 条["文件"])
with open(os.path.join(出目录, "听测_说明.md"), "w", encoding="utf-8") as f:
    f.write("# meeting64 · 人耳听测（v8.1）\n\n")
    f.write("> 目的：给「听清楚」补**人耳锚点**。ASR CER=0.5292 只是外部仪器；本轮发现它与「按声学教科书修保真」不同向。\n")
    f.write("> 用法：按文件名顺序听，**每句只听 1 遍**，在 0–4 打分；打完再看 `答案键.json`。\n\n")
    f.write("## 打分表（18 条合成 + 3 条真人对照）\n\n")
    f.write("\n".join(行) + "\n\n")
    f.write("## 对照说明\n\n")
    f.write("- `听测_01..18` = v8.1 合成（6 句 × 3 角色，顺序已随机、标签已去）。\n")
    f.write("- `对照_01..03` = 真人参考（EmoCompanion/SpeakerA/洛天依，歌唱/朗读片段；**内容与合成句不同**，只作「清晰的人声是什么样」的锚点）。\n")
    f.write("- 真值文本见 `答案键.json`。\n\n")
    f.write("## 记完分之后\n\n")
    f.write("1) 算合成 18 条的**人耳均值**；2) 与 ASR CER 0.5292 并排；3) 若两者明显不一致（例如人耳觉得「基本听懂」而 CER 0.53），\n")
    f.write("   则下一轮目标函数改为「人耳评分（主）+ ASR（辅）」，并按 `05_会议纪要.md` §5 开 A 原型（≤1 人日）。\n")

print("听测包 →", 出目录)
print("文件数：", len(键), "（18 合成 + 3 对照）")
