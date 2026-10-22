# meeting67/实施/建子集数据.py —— 由 meeting66 对齐数据 + 语料席最小子集规则，物化 v13 训练用数据子集
# 只读 meeting66 产物；只写 meeting67/数据/；硬链接（同盘零拷贝，不改原文件）
# 用法：
#   $py meeting67\实施\建子集数据.py            # 建 4 档（D0 5min / D1 15min / D2 30min / D3 83min） + 全量索引
#   $py meeting67\实施\建子集数据.py --档 D2     # 只建一档
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
M67 = HERE.parent
AHI4 = M67.parent
M66 = AHI4 / "meeting66"
源数据 = M66 / "产物" / "数据_对齐asr"
子集json = M67 / "团队" / "语料_适配" / "选子集.json"
出根 = M67 / "数据"

# 选子集.json 的档名 → 输出 tag
档映射 = {
    "≤5_min": ("D0", "≤5min"),
    "≤15_min": ("D1", "≤15min"),
    "≤30_min": ("D2", "≤30min"),
    "1/2": ("D3", "1/2=83min"),
    "全量": ("D4", "全量"),
}


def 载入():
    meta = json.loads((源数据 / "meta.json").read_text(encoding="utf-8"))
    档s = json.loads(子集json.read_text(encoding="utf-8"))["档"]
    return meta, 档s


def 建一档(meta: dict, 档: dict, tag: str, 说明: str) -> dict:
    目录 = 出根 / ("子集_%s_%s" % (tag, 说明.replace("≤", "le").replace("/", "-").replace("=", "_")))
    样本目录 = 目录 / "样本"
    样本目录.mkdir(parents=True, exist_ok=True)
    # 源 meta 的 id → 条目 / 文件名
    条目 = {}
    for 项 in meta.get("样本", []):
        try:
            条目[int(项["id"])] = 项
        except (KeyError, TypeError, ValueError):
            continue
    ids = []
    for 片 in 档["片段"]:
        i = int(片["id"])
        if i in 条目:
            ids.append(i)
    保留 = set(ids)
    新样本 = []
    for i in ids:
        项 = deepcopy(条目[i])
        名 = 项.get("文件") or ("%04d.npz" % i)
        源 = 源数据 / "样本" / 名
        目标 = 样本目录 / 名
        if not 目标.exists():
            try:
                os.link(源, 目标)                 # 硬链接：同盘、零拷贝
            except OSError:
                shutil.copy2(源, 目标)            # 跨盘兜底（会慢）
        新样本.append(项)
    # 划分：源划分 ∩ 子集；验证空则取贪心序末 5%（确定性）
    旧划分 = meta.get("划分", {})
    旧训 = 旧划分.get("train") or 旧划分.get("训练") or []
    旧验 = 旧划分.get("val") or 旧划分.get("验证") or []
    训 = [int(x) for x in 旧训 if int(x) in 保留]
    验 = [int(x) for x in 旧验 if int(x) in 保留]
    if not 验:
        k = max(1, len(ids) // 20)
        验 = ids[-k:]
    if not 训:
        训 = [i for i in ids if i not in set(验)]
    if not 训:                                  # 极端小档：留 1 条训练
        训 = ids[: max(1, len(ids) - len(验))]
    # 新 meta：复制统计/词表，替换样本清单与划分
    新meta = deepcopy(meta)
    新meta["样本"] = 新样本
    新meta["划分"] = {"train": 训, "val": 验, "训练": 训, "验证": 验}
    新meta["子集说明"] = {
        "tag": tag, "说明": 说明, "片段数": len(ids), "来源": str(源数据),
        "规则": "meeting67 团队/语料_适配/选子集.json（确定性贪心，见其 规则 字段）",
        "训练秒": 档.get("训练秒"), "音节数": 档.get("音节数"),
        "音节型": 档.get("音节型"), "音节型覆盖": 档.get("音节型覆盖"),
    }
    (目录 / "meta.json").write_text(json.dumps(新meta, ensure_ascii=False, indent=1), encoding="utf-8")
    # 自检：数据集类可加载
    报 = dict(tag=tag, 说明=说明, 目录=str(目录), 片段=len(ids),
              训练=len(训), 验证=len(验), npz=len(list(样本目录.glob("*.npz"))),
              训练秒=档.get("训练秒"), 音节=档.get("音节数"),
              音节型覆盖=档.get("音节型覆盖"))
    print("[建子集] %s %s：片段 %d / 训练 %d / 验证 %d / npz %d / 训练秒 %s"
          % (tag, 说明, 报["片段"], 报["训练"], 报["验证"], 报["npz"], 报["训练秒"]), flush=True)
    return 报


def 主(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--档", default="", help="只建某 tag（D0/D1/D2/D3/D4）；空=全部")
    ap.add_argument("--自检载入", action="store_true", help="建完用 模型_学说话.数据集 试载（只读）")
    a = ap.parse_args(argv)
    meta, 档s = 载入()
    出根.mkdir(parents=True, exist_ok=True)
    报s = []
    for 档 in 档s:
        名 = 档.get("名", "")
        if 名 not in 档映射:
            continue
        tag, 说明 = 档映射[名]
        if a.档 and tag != a.档:
            continue
        报s.append(建一档(meta, 档, tag, 说明))
    if (not a.档 or a.档 == "D4") and not any(r["tag"] == "D4" for r in 报s):
        全量档 = {
            "片段": [{"id": int(项["id"])} for 项 in meta.get("样本", [])],
            "训练秒": sum(float(项.get("帧数", 0)) for 项 in meta.get("样本", [])) / 100.0,
            "音节数": sum(int(项.get("音节数", 0)) for 项 in meta.get("样本", [])),
            "音节型": None,
            "音节型覆盖": 1.0,
        }
        报s.append(建一档(meta, 全量档, "D4", "全量162min"))
    (M67 / "数据" / "子集索引.json").write_text(
        json.dumps({"版本": "meeting67 子集 v1", "档": 报s}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("[落盘] %s" % (M67 / "数据" / "子集索引.json"), flush=True)
    if a.自检载入:
        sys.path.insert(0, str(M66 / "实施"))
        import importlib
        模 = importlib.import_module("模型_学说话")
        for r in 报s:
            ds = 模.数据集(Path(r["目录"]), 前N=None, 打印=False)
            print("[试载] %s 训练 %d / 验证 %d / 全部 %d"
                  % (r["tag"], len(ds.训练ids), len(ds.验证ids), len(ds.全部ids)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(主())
