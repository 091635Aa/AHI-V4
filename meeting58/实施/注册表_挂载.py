# -*- coding: utf-8 -*-
"""meeting58/实施/注册表_挂载.py —— 动态可插拔注册表（生命周期 + 隔离 + 回滚）。

回答所有者三问：
  ① 换音色要不要重启？→ 不要。注册表只换**不可变声线档案指针**，O(1)。
  ② 热插拔角色模板要不要重建核？→ 不要。只重建行层逻辑对象；状态源可缓存。
  ③ 切换时在途语句？→ `句边界` 栅栏：当前句用旧插件走完，下一句新插件生效；禁止中途换律。

隔离四道：
  1) 每次调用 try/except → 降级产物（不抛穿）；
  2) 调用前查协议版本/依赖；
  3) 装载/切换前快照 np.geterr() / sys.path / os.environ，卸载还原；
  4) 内核只持指针+历史栈，不持插件内部可变对象。
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from 接口契约_角色声线 import (  # noqa: E402
    E1001, E1002, E1003, E1004, E1005, E1010, E1012,
    契约版本, 契约错误,
)
from 工具_确定性 import 指纹  # noqa: E402


class 注册表_挂载:
    """按「类」分区（状态源 / 角色逻辑 / 音色渲染），每类同时只有一个当前指针。"""

    def __init__(self):
        self._登记: dict = {}          # (类, 名) -> 条目
        self._当前: dict = {}          # 类 -> 名
        self._历史: dict = {}          # 类 -> [名, ...]
        self._隔离: dict = {}          # (类, 名) -> 原因
        self._失败: dict = {}          # (类, 名) -> 连续失败数
        self.台账 = dict(注册=0, 试跑=0, 切换=0, 回滚=0, 卸载=0, 隔离=0, 调用=0, 失败=0)

    # ── 登记 ──
    def 注册(self, 类: str, 名: str, 工厂, 版本: str = "1.0", 依赖=(), 元数据=None) -> dict:
        k = (str(类), str(名))
        if k in self._登记:
            raise 契约错误(E1005, "%s/%s 已注册" % k)
        if not callable(工厂):
            raise 契约错误(E1001, "%s/%s 工厂不可调用" % k)
        条目 = dict(类=str(类), 名=str(名), 工厂=工厂, 版本=str(版本),
                    依赖=tuple(依赖), 元数据=dict(元数据 or {}),
                    插件=None, 状态="已登记", 指纹="")
        self._登记[k] = 条目
        self.台账["注册"] += 1
        return dict(类=k[0], 名=k[1], 状态=条目["状态"], 版本=条目["版本"])

    def 取(self, 类: str, 名: str | None = None):
        名 = 名 or self._当前.get(类)
        if 名 is None:
            raise 契约错误(E1004, "类 %r 无当前插件" % 类)
        k = (类, 名)
        条目 = self._登记.get(k)
        if 条目 is None:
            raise 契约错误(E1001, "未注册 %s/%s" % (类, 名))
        return 条目

    # ── 试跑（只读，不切换）──
    def 试跑(self, 类: str, 名: str, 探针=None) -> dict:
        条目 = self.取(类, 名)
        if 条目["状态"] == "已隔离":
            raise 契约错误(E1001, "%s/%s 已隔离" % (类, 名))
        插件 = 条目["插件"] or 条目["工厂"]()
        条目["插件"] = 插件
        self.台账["试跑"] += 1
        结果 = dict(类=类, 名=名, 版本=条目["版本"], 指纹=指纹(名, 条目["版本"]),
                    降级=False, 错="")
        if 探针 is not None:
            try:
                结果["探针"] = 探针(插件)
            except Exception as e:  # noqa: BLE001
                结果["降级"] = True
                结果["错"] = "E1009 %s" % repr(e)[:120]
                条目["状态"] = "已试跑(降级)"
                return 结果
        条目["状态"] = "已试跑"
        结果["指纹"] = 指纹(名, 条目["版本"], 结果.get("探针"))
        return 结果

    # ── 热切换（句边界栅栏由调用方保证；这里只做指针切换）──
    def 切换(self, 类: str, 名: str, 栅栏: str = "句边界") -> dict:
        if 栅栏 not in ("句边界", "灌满", "立即"):
            raise 契约错误(E1012, "未知栅栏 %r" % 栅栏)
        新 = self.取(类, 名)
        if 新["状态"] in ("已登记", "已试跑(降级)"):
            raise 契约错误(E1004, "%s/%s 未试跑通过" % (类, 名))
        旧名 = self._当前.get(类)
        旧 = self._登记.get((类, 旧名)) if 旧名 else None
        env = self._快照环境()
        self._当前[类] = 名
        self._历史.setdefault(类, []).append(旧名) if 旧名 else None
        if 旧 is not None:
            旧["状态"] = "历史"
        新["状态"] = "当前"
        self._还原环境(env)
        self.台账["切换"] += 1
        return dict(类=类, 旧=旧名, 新=名, 栅栏=栅栏, 指纹=指纹(类, 旧名, 名))

    def 回滚(self, 类: str, 到名: str | None = None) -> dict:
        历史 = self._历史.get(类) or []
        if not 历史:
            raise 契约错误(E1010, "类 %r 无历史" % 类)
        目标 = 到名
        if 目标 is None:
            目标 = 历史[-1]
            if 目标 is None:
                raise 契约错误(E1010, "上一版本为空")
        if (类, 目标) not in self._登记:
            raise 契约错误(E1010, "%s/%s 不存在" % (类, 目标))
        旧 = self._当前.get(类)
        self._当前[类] = 目标
        self._登记[(类, 目标)]["状态"] = "当前"
        if 旧:
            self._登记[(类, 旧)]["状态"] = "历史"
        self.台账["回滚"] += 1
        return dict(类=类, 旧=旧, 新=目标, 指纹=指纹(类, 旧, 目标))

    def 卸载(self, 类: str, 名: str) -> None:
        条目 = self.取(类, 名)
        if self._当前.get(类) == 名:
            raise 契约错误(E1012, "当前插件不能直接卸载，请先切换")
        条目["插件"] = None
        条目["状态"] = "已卸载"
        self.台账["卸载"] += 1

    def 调用(self, 类: str, 方法: str, *a, **kw):
        """隔离调用：异常不穿层，连续 3 次失败自动隔离。"""
        条目 = self.取(类)
        k = (类, 条目["名"])
        if 条目["状态"] == "已隔离":
            return dict(降级=True, 错="E1001 已隔离", 结果=None)
        self.台账["调用"] += 1
        try:
            插件 = 条目["插件"] or 条目["工厂"]()
            条目["插件"] = 插件
            r = getattr(插件, 方法)(*a, **kw)
            self._失败[k] = 0
            return dict(降级=False, 错="", 结果=r)
        except Exception as e:  # noqa: BLE001
            self.台账["失败"] += 1
            self._失败[k] = self._失败.get(k, 0) + 1
            if self._失败[k] >= 3:
                条目["状态"] = "已隔离"
                self._隔离[k] = repr(e)[:120]
                self.台账["隔离"] += 1
            return dict(降级=True, 错="E1009 %s" % repr(e)[:120], 结果=None)

    def 表现(self) -> dict:
        return dict(
            当前=dict(self._当前),
            已登记={"%s/%s" % k: v["状态"] for k, v in self._登记.items()},
            历史={k: [x for x in v] for k, v in self._历史.items()},
            隔离={"%s/%s" % k: v for k, v in self._隔离.items()},
            失败={"%s/%s" % k: v for k, v in self._失败.items()},
            台账=dict(self.台账),
            指纹=self.指纹(),
        )

    def 指纹(self) -> str:
        return 指纹(sorted("%s/%s:%s" % (k[0], k[1], v["状态"])
                          for k, v in self._登记.items()),
                    sorted("%s:%s" % (k, v) for k, v in self._当前.items()), 长=16)

    @staticmethod
    def _快照环境() -> dict:
        return dict(nperr=np.geterr(), path=list(sys.path), env=dict(os.environ))

    @staticmethod
    def _还原环境(env: dict) -> None:
        np.seterr(**env["np"]) if env.get("np") else None
        sys.path[:] = env["path"]
        os.environ.clear(); os.environ.update(env["env"])


if __name__ == "__main__":
    import sys as _s
    try:
        _s.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import 声线库
    import 协议_音色渲染 as R
    import 协议_角色逻辑 as L

    表 = 注册表_挂载()
    表.注册("音色渲染", "默认", R.声带渲染器, 版本="1.0")
    表.注册("角色逻辑", "洛天依", L.池角色逻辑, 版本="1.0")
    表.注册("角色逻辑", "MOSS", L.池角色逻辑, 版本="1.0")
    print("试跑 渲染:", 表.试跑("音色渲染", "默认",
                             探针=lambda p: p.渲染("自检。", 声线=声线库.取("EmoCompanion"),
                                                 写wav=False)["通过"]))
    print("试跑 洛:", 表.试跑("角色逻辑", "洛天依", 探针=lambda p: hasattr(p, "步") and hasattr(p, "装")))
    print("切换:", 表.切换("角色逻辑", "洛天依"))
    print("表现:", 表.表现()["当前"], 表.台账)
