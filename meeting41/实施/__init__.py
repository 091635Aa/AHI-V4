# -*- coding: utf-8 -*-
"""meeting41/实施/__init__.py —— 第七轮实施包（纯新增；不改既有件）。

把本目录加进 sys.path 后即可 `import 接口契约`、`import 算子_能力核` 等。
本包不 import 任何器官；器官只能由主程序注入（与 V3 既有纪律一致）。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
