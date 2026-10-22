# -*- coding: utf-8 -*-
"""meeting28/chat_smoke.py —— #216 对话体判据冒烟（①往返 ②检索命中 ③状态响应 ④确定性 ⑤持久化）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lang_codec import 语言编解码器
from dialogue_bridge import 对话桥

结果 = {}

# ① 编解码往返
c = 语言编解码器(seed=216)
v = c.编码("今天天气不错")
结果["①编码确定"] = bool((c.编码("今天天气不错") == v).all())
v2 = c.编码("明天天气不错")
结果["①相近文本相似度"] = round(c.相似度(v, v2), 3)
v3 = c.编码("量子计算机的纠错码")
结果["①无关文本相似度"] = round(c.相似度(v, v3), 3)

# ② 检索命中（改述查询 hit@1）
c2 = 语言编解码器(seed=216)
情节 = ["我最喜欢的颜色是蓝色", "我养了一只猫叫小咪", "我在杭州工作",
        "我周末喜欢爬山", "我的生日在七月", "我不吃香菜",
        "我的工作是程序员", "我最近在学吉他", "我家住在六楼",
        "我每天早上喝咖啡", "我妹妹在上大学", "我讨厌堵车"]
for p in 情节:
    c2.注册段(p)
改述 = {"最喜欢什么颜色": "我最喜欢的颜色是蓝色", "猫叫什么": "我养了一只猫叫小咪",
        "在哪里上班": "我在杭州工作", "周末干嘛": "我周末喜欢爬山"}
命中 = 0
for 查询, 期望 in 改述.items():
    if c2.解码(查询) == 期望:
        命中 += 1
结果["②检索命中"] = "%d/%d" % (命中, len(改述))

# ③ 状态响应性 + ④ 确定性 + ⑤ 持久化
def 脚本(路径=None):
    桥 = 对话桥(seed=216)
    对话 = ["你好", "我叫小明", "我最喜欢的颜色是蓝色", "你还记得我叫什么吗",
            "我养了一只猫", "它叫什么来着", "今天有点累"]
    outs = []
    for t in 对话:
        r = 桥.步(t)
        outs.append((r["动作"], r["响应"]))
    if 路径:
        桥.保存(路径)
    return outs, 桥

oA, 桥A = 脚本()
oB, _ = 脚本()
结果["④确定性"] = oA == oB
结果["③动作分布"] = str({a: sum(1 for x in oA if x[0] == a) for a in set(x[0] for x in oA)})
结果["③样例响应"] = oA[-1][1][:60]

pkl = os.path.join(HERE, "chat_smoke.pkl")
桥A.保存(pkl)
桥C = 对话桥.载入(pkl)
rC = 桥C.步("我最喜欢的颜色是什么")
结果["⑤持久化后记忆连续"] = "蓝色" in rC["响应"] or "蓝色" in rC["检索"]
os.remove(pkl)

print(json.dumps(结果, ensure_ascii=False, indent=1) if False else 结果)
import json
print(json.dumps(结果, ensure_ascii=False, indent=1))
