# -*- coding: utf-8 -*-
"""一次性：由 训练_P6v4.py 生 训练_P6v5.py（只改数据/输出路径，不动逻辑）。"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
s = (HERE / "训练_P6v4.py").read_text(encoding="utf-8")
s = s.replace('P6 = E / "P6v4"', 'P6 = E / "P6v5"')
s = s.replace('with open(E / "P6v4_数据.pkl", "rb")', 'with open(E / "P6v5_数据.pkl", "rb")')
s = s.replace('训练_P6v4.py', '训练_P6v5.py')
s = s.replace('P6 v4：扩数据', 'P6 v5：全量扩数据')
(HERE / "训练_P6v5.py").write_text(s, encoding="utf-8")
print("written", (HERE / "训练_P6v5.py").stat().st_size)
