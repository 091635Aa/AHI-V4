# -*- coding: utf-8 -*-
"""本文件由 AHI（meeting40 代码任务）依据**结构化规格**合成，非人手编写。

规格指纹：1a50b153fa64a841
操作：计数
说明：给一屏网页的正文段列表，返回段数
自证方式：运行本文件，会逐条执行内嵌用例并打印 JSON（ok=全部通过）。
"""
from __future__ import annotations

import json

def 段数(xs):
    """返回元素个数。"""
    return len(xs)

_用例s = [{"输入": ["段1", "段2", "段3"], "期望": 3}, {"输入": [], "期望": 0}]


if __name__ == "__main__":
    果 = []
    for i, c in enumerate(_用例s):
        try:
            得 = 段数(c["输入"])
            ok = (得 == c["期望"])
        except Exception as ex:
            得, ok = "异常:%r" % (ex,), False
        果.append(dict(序号=i, 期望=c["期望"], 实得=得, ok=bool(ok)))
    print(json.dumps(dict(用例s=果, ok=all(x["ok"] for x in 果)), ensure_ascii=False))
