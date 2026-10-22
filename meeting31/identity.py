# -*- coding: utf-8 -*-
"""meeting31/identity.py —— #262 身份模板：显式、可审计的角色定义。

身份=JSON 定义（名字/性格描述/背景事实/风格语料/性格参数）。
加载=背景事实经自我声道入账（账本可审计）+风格语料学入自声道+性格参数注入引擎动态。
换模板=换角色；血统完整（哪句设定来自哪次加载可查）。
"""
from __future__ import annotations

import json


class 身份模板:
    def __init__(self, 名字: str, 性格描述: str = "", 背景: list = None,
                 风格语料: list = None, 参数: dict = None):
        self.名字 = 名字
        self.性格描述 = 性格描述
        self.背景 = list(背景 or [])
        self.风格语料 = list(风格语料 or [])
        self.参数 = dict(参数 or {})

    @classmethod
    def 从文件(cls, 路径: str) -> "身份模板":
        with open(路径, encoding="utf-8") as f:
            d = json.load(f)
        return cls(名字=d["名字"], 性格描述=d.get("性格描述", ""),
                   背景=d.get("背景", []), 风格语料=d.get("风格语料", []),
                   参数=d.get("参数", {}))
