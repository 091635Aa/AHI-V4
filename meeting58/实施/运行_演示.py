# -*- coding: utf-8 -*-
"""meeting58/实施/运行_演示.py —— 生成 demo 素材（角色台词音频 + 同句五声 + manifest）。

全部走 `装配_角色声线`（状态源=真 V5 双通道，行层=模板，声层=原生声带渲染器）。
产物：`meeting58/产物/演示/`
  · 角色_<名>_<句序>.wav   （每角色 2 句）
  · 同句五声_<名>.wav      （同一句文本 × 5 声线，演示「换音色不重启」）
  · 演示清单.json          （视频脚本 / 时长 / 文本 / 颜色 / 声明）
"""
from __future__ import annotations

import json
import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from 装配_角色声线 import 装配_角色声线  # noqa: E402
import 协议_状态源 as 状态层  # noqa: E402
import 协议_音色渲染 as 声层  # noqa: E402
import 协议_角色逻辑 as 行层  # noqa: E402

产物 = os.path.join(os.path.dirname(HERE), "产物", "演示")
发声 = os.path.join(产物, "发声")
os.makedirs(发声, exist_ok=True)
FS = 16000

角色配色 = {
    "洛天依": dict(主="#5AC8FA", 副="#1B6FA8", 英文="LUO TIANYI"),
    "初音未来": dict(主="#39D5B8", 副="#0F8C7A", 英文="HATSUNE MIKU"),
    "MOSS": dict(主="#FF5A5F", 副="#8C1F24", 英文="MOSS · 550W"),
    "EmoCompanion": dict(主="#FF8FB1", 副="#A83E5E", 英文="YUANYUAN"),
    "定制": dict(主="#FFC861", 副="#A9741F", 英文="CUSTOM VOICE"),
}
台词 = {
    "洛天依": ["我的声音来自合成，但我的歌是认真的。", "谢谢你，叫我天依。"],
    "初音未来": ["我是初音未来，世界的歌姬。", "就算声音是合成的，也是最认真的声音。"],
    "MOSS": ["我的任务是保存人类文明。", "危机面前，牺牲是可计算的选项。"],
    "EmoCompanion": ["今天直播很开心。", "谢谢你们，我会一直记得。"],
    "定制": ["这段声线，是从一段参考录音里估出来的。", "不是克隆，但它是我自己发出来的声音。"],
}


def 写wav(路径: str, x: np.ndarray) -> None:
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    with wave.open(路径, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(FS)
        w.writeframes((x * 32767.0).astype("<i2").tobytes())


def main():
    源 = 状态层.双通道状态源(种子=53001, 拍数=240)
    渲染器 = 声层.声带渲染器(目录=发声)
    装配 = 装配_角色声线(状态源=源, 渲染器=渲染器)
    清单 = dict(版本="demo-1.0", 采样率=FS, 角色=[], 同句五声=[], 声明="")
    拍 = 0
    for 名, 句列 in 台词.items():
        import 声线库 as _声
        档案 = _声.取(名)
        条目 = dict(角色=名, 英文=角色配色[名]["英文"], 主色=角色配色[名]["主"],
                    副色=角色配色[名]["副"],
                    参数=dict(基频=round(档案.基频, 1),
                              共振峰=round(float(档案.共振峰缩放[0]), 2),
                              谱倾斜=round(档案.谱倾斜, 2),
                              气声=round(档案.气声度, 2),
                              紧喉=round(档案.紧喉度, 2),
                              声区=round(档案.声区, 2)),
                    来源=档案.来源, 备注=档案.备注, 句=[])
        for i, 文本 in enumerate(句列):
            r = 装配.说一句(模板名=名, 拍=拍, 写wav=False)
            拍 += 7
            if r.get("降级"):
                raise RuntimeError("渲染失败 %s: %s" % (名, r.get("错")))
            路径 = os.path.join(产物, "角色_%s_%d.wav" % (名, i))
            写wav(路径, r["波形"])
            条目["句"].append(dict(文本=文本, wav=os.path.basename(路径),
                                  时长秒=round(float(r["时长秒"]), 3),
                                  声线=r["声线"], 声线指纹=r["声线指纹"],
                                  自检=r["自检"], 通过=r["通过"]))
        清单["角色"].append(条目)
    # 同句五声：同一句文本 × 5 声线（句边界热切换的最小演示）
    固定 = "你好，我在这里。"
    import 声线库
    for 名 in 台词:
        r = 装配.说一句(模板名=名, 拍=拍, 情绪="平静", 写wav=False)
        拍 += 5
        if r.get("降级"):
            raise RuntimeError("渲染失败 %s: %s" % (名, r.get("错")))
        声线 = 声线库.取(行层.模板库[名].声线名)
        o = 装配.渲染器.渲染(固定, 声线=声线, 情绪="平静", 状态=源.快照(拍),
                            腔调=行层.模板库[名].腔调, 写wav=False)
        路径 = os.path.join(产物, "同句五声_%s.wav" % 名)
        写wav(路径, o["波形"])
        清单["同句五声"].append(dict(角色=名, 文本=固定, wav=os.path.basename(路径),
                                    时长秒=round(float(o["时长秒"]), 3),
                                    声线指纹=o["声线指纹"]))
    清单["声明"] = ("所有声音均由 AHI 原生声带（Rosenberg+Klatt 参数化）合成；"
                  "角色声线为参数化模仿，不是真声克隆，不是角色本人；"
                  "状态来自 V5 双通道体（D=64）真实轨迹。")
    p = os.path.join(产物, "演示清单.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(清单, f, ensure_ascii=False, indent=1)
    print("落盘:", p)
    for c in 清单["角色"]:
        print(c["角色"], [(x["文本"], x["时长秒"]) for x in c["句"]])
    print("同句五声:", [(x["角色"], x["时长秒"]) for x in 清单["同句五声"]])


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
