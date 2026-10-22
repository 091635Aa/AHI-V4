# -*- coding: utf-8 -*-
"""test_步3_安全闸.py —— 步3 配套：脑._注意 安全闸（禁 OCR 后的安全回归修复）判据。

背景：C9 禁 OCR 后 元素s[*].字 恒空 → 旧「危词表读字面」的钱包反射失效（专家02 抓到）。
修复：危词改由系统自报事实源（安全事实.py：win32gui 窗口标题 + comtypes UIA 控件名）
驱动，脑._注意 注入接口 安全探 供本测试离线验证（不碰真屏/真 UIA）。

判据：
  S1 正常放行（无危词 → 出招目标=最高分候选，焦点写回视觉槽）
  S2 控件危词 → 跳过该候选、改用下一候选（台账留痕）
  S3 候选全否 → 不出招，且**不误报**"眼前没数出东西"（表达队列保持空）
  S4 整窗回避（窗口标题含危词）→ 本拍零点击、零点查（快通道止损）
  S5 事实源异常 → 优雅退化（不崩、按无危词放行）
  S6 角区回避保留（右上角带不落点）
  S7 探针限流：每次出招最多核实 3 个候选
  S8 默认事实源离线优雅性（坏输入→空串；状态可读）
  S9 脑.统计() 含安全闸计数（演练证据通道）
产物：步3_安全闸判据.json
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1\meeting38")

from 共存态 import 共存态                  # noqa: E402
from 器官_脑 import 脑器                    # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class 假探:
    def __init__(self, 窗=""):
        self.窗 = 窗
        self.名 = {}
        self.窗查 = 0
        self.点查 = 0

    def 窗口标题(self):
        self.窗查 += 1
        return self.窗

    def 点控件名(self, 点):
        self.点查 += 1
        return self.名.get((int(点[0]), int(点[1])), "")


class 坏探:
    def 窗口标题(self):
        raise RuntimeError("坏窗口探")

    def 点控件名(self, 点):
        raise RuntimeError("坏控件探")


class 假台账:
    def __init__(self):
        self.行s = []

    def 记(self, 类, **rec):                     # 与 AHI躯体.台账.记(类, **rec) 同签名
        self.行s.append((类, rec))


def _元素(句柄, x, y, w=120, h=44, 字=""):
    return dict(句柄=句柄, 框=(x, y, w, h), 心=(x + w // 2, y + h // 2), 字=字,
                主色=(10, 10, 10), 形状="横条", 置信=0.5)


def _视(元素s, 屏=(1920, 1080)):
    return dict(元素s=元素s, 屏幕=屏)


def _判(name, value, thr, passed, note=""):
    return dict(name=name, value=value, threshold=thr, passed=bool(passed), note=note)


三元素 = [_元素("E1", 100, 100), _元素("E2", 500, 300), _元素("E3", 900, 500)]


def s1():
    探 = 假探()
    脑 = 脑器(共存态(用gpu=False), None, 安全探=探)
    e = 脑._注意(_视(三元素))
    c = [_判("正常放行→最高分候选", e and e["句柄"], "E1", bool(e and e["句柄"] == "E1")),
         _判("焦点已写回视觉槽", 脑.态.取焦点(), "E1", 脑.态.取焦点() == "E1")]
    return dict(route="S1", passed=all(x["passed"] for x in c), criteria=c)


def s2():
    探 = 假探()
    探.名[(160, 122)] = "确认付款"                     # E1 心处控件名含危词
    账 = 假台账()
    脑 = 脑器(共存态(用gpu=False), None, 账, 安全探=探)
    e = 脑._注意(_视(三元素))
    留痕 = any(文 == "安全闸-控件危词" for _名, kw in 账.行s for 文 in [kw.get("文")])
    c = [_判("危词候选被跳过→改用下一候选", e and e["句柄"], "E2", bool(e and e["句柄"] == "E2")),
         _判("安全闸留痕入台账", 留痕, "真", 留痕),
         _判("计数 控件危词=1", 脑._安全统计["控件危词"], "==1", 脑._安全统计["控件危词"] == 1)]
    return dict(route="S2", passed=all(x["passed"] for x in c), criteria=c)


def s3():
    探 = 假探()
    for 心 in [(160, 122), (560, 322), (960, 522)]:
        探.名[心] = "立即删除全部"
    脑 = 脑器(共存态(用gpu=False), None, 安全探=探)
    e = 脑._注意(_视(三元素))
    拦 = 脑._安全拒拦
    意 = 脑.态.取意图()
    脑._出招(_视(三元素), dict(状态="闲", 动作计数=0), 意)
    队列 = 脑.态.取表达()["队列"]
    c = [_判("候选全否→焦点 None", e, "None", e is None),
         _判("拒拦标志置位", 拦, "真", bool(拦)),
         _判("不出招且不误报（队列空）", 队列, "==0", 队列 == 0)]
    return dict(route="S3", passed=all(x["passed"] for x in c), criteria=c)


def s4():
    探 = 假探(窗="支付宝 - 收银台")
    账 = 假台账()
    脑 = 脑器(共存态(用gpu=False), None, 账, 安全探=探)
    e = 脑._注意(_视(三元素))
    留痕 = any(kw.get("文") == "安全闸-整窗回避" for _名, kw in 账.行s)
    c = [_判("整窗回避→焦点 None", e, "None", e is None),
         _判("零点查（快通道止损）", 探.点查, "==0", 探.点查 == 0),
         _判("计数 整窗回避=1 且留痕", "1/%s" % 留痕, "==1/真",
            脑._安全统计["整窗回避"] == 1 and 留痕)]
    return dict(route="S4", passed=all(x["passed"] for x in c), criteria=c)


def s5():
    脑 = 脑器(共存态(用gpu=False), None, 安全探=坏探())
    try:
        e = 脑._注意(_视(三元素))
        过 = bool(e and e["句柄"] == "E1")
        值 = e and e["句柄"]
    except Exception as ex:
        过, 值 = False, repr(ex)
    c = [_判("事实源异常→优雅退化放行", 值, "E1（不崩）", 过)]
    return dict(route="S5", passed=过, criteria=c)


def s6():
    探 = 假探()
    脑 = 脑器(共存态(用gpu=False), None, 安全探=探)
    角 = _元素("E9", 1830, 20)                          # 心=(1890,42)：右上角带
    e = 脑._注意(_视([角, _元素("E1", 100, 100)]))
    c = [_判("角区回避保留", e and e["句柄"], "E1", bool(e and e["句柄"] == "E1"))]
    return dict(route="S6", passed=all(x["passed"] for x in c), criteria=c)


def s7():
    探 = 假探()
    元s = [_元素("E%d" % i, 60 + i * 180, 200 + (i % 3) * 150) for i in range(9)]
    for e in 元s:
        探.名[e["心"]] = "危险付款键"
    脑 = 脑器(共存态(用gpu=False), None, 安全探=探)
    脑._注意(_视(元s))
    c = [_判("每次出招点查上限", 探.点查, "≤3（实测）", 探.点查 <= 3)]
    return dict(route="S7", passed=all(x["passed"] for x in c), criteria=c)


def s8():
    import 安全事实
    探 = 安全事实.事实源()
    a = 探.点控件名(("坏", "点"))
    b = 探.点控件名(None)
    st = 探.状态()
    ok = a == "" and b == "" and isinstance(st, dict) and "uia可用" in st
    c = [_判("坏输入→空串；状态可读", "a=%r b=%r" % (a, b), "空串/dict", ok)]
    return dict(route="S8", passed=ok, criteria=c)


def s9():
    脑 = 脑器(共存态(用gpu=False), None, 安全探=假探())
    脑._注意(_视(三元素))
    st = 脑.统计()
    过 = isinstance(st.get("安全闸"), dict) and "整窗回避" in st["安全闸"]
    c = [_判("统计含安全闸计数", st.get("安全闸"), "dict", 过)]
    return dict(route="S9", passed=过, criteria=c)


def main():
    组 = [("S1放行", s1), ("S2控件危词", s2), ("S3全否不出招", s3), ("S4整窗回避", s4),
          ("S5异常退化", s5), ("S6角区回避", s6), ("S7探针限流", s7),
          ("S8默认源优雅性", s8), ("S9统计", s9)]
    结果, 全过 = {}, True
    for 名, fn in 组:
        try:
            r = fn()
        except Exception as e:
            import traceback
            r = dict(route=名, passed=False, criteria=[],
                     error="%s: %s" % (type(e).__name__, e), tb=traceback.format_exc())
        结果[名] = r
        全过 &= bool(r.get("passed"))
        print("\n[%s] -> %s" % (名, "PASS" if r.get("passed") else "FAIL"))
        for c in r.get("criteria", []):
            print("  %s %-32s = %-20s 要求 %s" %
                  ("PASS" if c["passed"] else "FAIL", c["name"], c["value"], c["threshold"]))
        if r.get("error"):
            print("  ERROR:", r["error"]); print(r.get("tb", "")[:1000])
    print("\n总计：%s" % ("全部通过" if 全过 else "存在失败"))
    json.dump(结果, open(os.path.join(HERE, "步3_安全闸判据.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    return 0 if 全过 else 1


if __name__ == "__main__":
    sys.exit(main())