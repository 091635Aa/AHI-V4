# -*- coding: utf-8 -*-
"""meeting63/实施/学_咬字反馈_ASR.py —— 用**外部 ASR（可懂度）**当 reward 的离线 θ 搜索件。

口径（先钉边界，不许含糊）：
  · 裁判 = `判据_可懂度_ASR.载引擎()`（Paraformer-seaco-large 中文）。
    ★ **只作外部 reward / 裁判**，**不在生成路径**（C10：生成路径零外挂由 `审计_零外挂.py` 独立证明）。
  · 目标 = **筛选集**（S1/S3/S6 × EmoCompanion/SpeakerA = 6 样本）的 CER 均值（越小越可懂）。
    复评集（6 句 × 3 角色 = 18 样本）只在最后评一次，不参与选择（防止选择偏差）。
  · 约束 = **皮层可分度 ≥ 0.75**（`皮层_音素格`，内生、零外挂）⇒ 别把咬字的「可分性」换掉了。
  · 协议 = **一轮坐标上升**：每维独立地从当前 θ 出发试两个网格值，取筛选 CER 更低者；
    最后把各维最优**组合**起来再评一次；**组合不比基线好 ⇒ 不采纳**（保留基线 θ）。
  · must-fail = **置乱真值**（确定性 rng，种子 63099）：把 6 句真值的字序打乱（保持每句长度），
    用「搜索前 θ / 搜索后 θ」分别对置乱真值算 CER 均值；要求 Δ = 前 − 后 ≤ 0.02
    （即搜索在**无意义真值**上找不到实质改善）。Δ > 0.02 ⇒ 如实写 must_fail_置乱失败=true。
  · C13 = 网格固定、无随机抽取；最终 θ 合成筛选集**两趟**，wav sha256 必须逐位同。
  · ★ 已知坑：ASR 可能按**文件路径**缓存/读滞后 ⇒ 每次转写都用**唯一的文件名**
    （含角色、句号、全局计数器），绝不复用同一路径。
  · 预算：ASR 推理 ≤ 240 次，每次调用计数并写进台账（JSON 的 ASR调用次数）。

用法：
  $env:PYTHONIOENCODING="utf-8"; & "<workspace>/\03_AHI\\.venv\\Scripts\\python.exe" `
      "<workspace>/\03_AHI\\AHI4\\meeting63\\实施\\学_咬字反馈_ASR.py"
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

这里 = os.path.dirname(os.path.abspath(__file__))
M63 = os.path.dirname(这里)
AHI4 = os.path.dirname(M63)
sys.path.insert(0, 这里)
sys.path.insert(0, os.path.join(AHI4, "meeting62", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting61", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))

import 判据_可懂度_ASR as 判   # noqa: E402  只读：外部裁判（载引擎 / CER / 正规）
import 声带后端_v8 as 后8      # noqa: E402  只读：`_档案` 造档 + `_写wav` 落盘
import 声线覆盖层_v7 as 覆     # noqa: E402  只读：三角色声线
import 皮层_音素格 as 皮       # noqa: E402  只读：内生可分度（约束）
import 韵律_v8 as H8           # noqa: E402  只读：被搜的合成器（θ 只在本进程内临时覆盖）

出目录 = os.path.join(M63, "产物")
os.makedirs(出目录, exist_ok=True)
临时 = os.path.join(出目录, "ASR临时", "学咬字反馈_ASR")
os.makedirs(临时, exist_ok=True)
出JSON = os.path.join(出目录, "读数_学咬字反馈_ASR.json")

# ───────────────────────── 句集 / 样本集 ─────────────────────────
句集 = (
    ("S1", "今天的风很轻，适合慢慢地说一会儿话。"),
    ("S2", "我们明天早上八点在南门见面。"),
    ("S3", "他把三本书放在窗台上。"),
    ("S4", "小猫从桌上跳下来，碰到杯子。"),
    ("S5", "春天来了，山上的雪开始融化。"),
    ("S6", "请问这条路怎么走？"),
)
句表 = dict(句集)
复评角色 = ("EmoCompanion", "SpeakerA", "洛天依")
# 筛选集：S1/S3/S6 × EmoCompanion/SpeakerA = 6 样本（内层搜索，省 ASR 调用）
筛选句号 = ("S1", "S3", "S6")
筛选角色 = ("EmoCompanion", "SpeakerA")
筛选对 = tuple((号, 名) for 号 in 筛选句号 for 名 in 筛选角色)
# 复评集：6 句 × 3 角色 = 18 样本（最终）
复评对 = tuple((号, 名) for 号, _ in 句集 for 名 in 复评角色)

# ───────────────────────── 可搜参数（只用这些网格） ─────────────────────────
# 键顺序 = 坐标上升的搜索顺序（固定 ⇒ C13 确定性）
网格 = {
    "过渡秒": [0.030, 0.040],
    "爆幅倍": [1.6, 2.0],
    "缩底": [0.85, 0.95],
    "塞擦擦时倍": [1.35, 1.55],
    "鼻尾幅": [0.40, 0.44],
    "鼻尾帧数": [8, 7],
}
# 当前值即基线（鼻尾帧数句中=7 不在网格内，全程固定不动）
基线θ = dict(过渡秒=0.030, 爆幅倍=1.6, 缩底=0.85, 塞擦擦时倍=1.35,
            鼻尾幅=0.40, 鼻尾帧数=8)
可分度下限 = 0.75
预算 = 240

# ───────────────────────── 运行期状态（确定性） ─────────────────────────
_默认θ = dict(H8.参数)          # 导入时的完整参数（含未被搜的维）
_序号 = [0]                     # 全局合成计数器（唯一文件名用）
调用 = dict(次数=0, 上限=int(预算))
音色s = {名: 覆.取(名).转dict() for 名 in 复评角色}
档s = {名: 后8.声带V8._档案(音色s[名], 名) for 名 in 复评角色}
FS = 44100


# ───────────────────────── 基础：θ 覆盖 / 合成 / 转写 ─────────────────────────
def 设θ(θ: dict) -> None:
    """把 θ 覆盖到 `韵律_v8.参数`（每次先复位到默认全量，避免维间串味）。"""
    H8.参数.clear()
    H8.参数.update(_默认θ)
    H8.参数.update(dict(θ))


def 合(对, 标签: str = "") -> list:
    """按当前 `参数`（调用方已 `设θ`）合成 → 落盘唯一 wav。返回 [(号, 角色, 路径, 样本数, 元)]。"""
    出 = []
    for 号, 名 in 对:
        w, 元, _帧 = H8.人类韵律_v8(档s[名], 300).说(句表[号])
        _序号[0] += 1
        # ★ 唯一文件名：全局计数 + 角色 + 句号（防 ASR 路径缓存/读滞后）
        路径 = os.path.join(临时, "%04d_%s_%s%s.wav" % (_序号[0], 名, 号, 标签))
        后8.声带V8._写wav(路径, w)
        出.append((号, 名, 路径, int(w.size), 元))
    return 出


def 转写(引: dict, 路径: str) -> str:
    """外部 ASR 转写（**计数**，超预算立即报错，不许悄悄超）。"""
    调用["次数"] += 1
    if 调用["次数"] > 调用["上限"]:
        raise RuntimeError("ASR 调用超预算：%d > %d" % (调用["次数"], 调用["上限"]))
    return 引["转写"](路径)


def 可分度(θ: dict) -> float:
    """内生可分度（皮层_音素格，零外挂）：留一音节准确率。"""
    设θ(θ)
    样 = 皮.采集(韵律模块=H8, 种子列=(300, 301))
    果 = 皮.跑(样品=样)
    return float(果["准确率"])


def 评(对, θ: dict, 引: dict) -> dict:
    """一个 θ 在某样本集上的读数：先算可分度约束（不耗 ASR），再合成→转写→CER。"""
    可分 = round(可分度(θ), 4)
    if 可分 < 可分度下限:
        # 约束不过：**不花 ASR 预算**，直接丢弃该候选
        return dict(CER=None, 可分度=可分, 约束丢弃=True, 样本=[])
    设θ(θ)
    逐 = []
    for 号, 名, 路径, 样本数, _元 in 合(对):
        t = 转写(引, 路径)
        逐.append(dict(号=号, 句=句表[号], 角色=名, wav=os.path.basename(路径),
                       时长秒=round(样本数 / float(FS), 3), 转写=t,
                       CER=判.CER(句表[号], t)))
    cer = round(float(np.mean([x["CER"] for x in 逐])), 4)
    return dict(CER=cer, 可分度=可分, 约束丢弃=False, 样本=逐)


def 置乱真值() -> dict:
    """must-fail 用：确定性 rng(63099) 打乱每句字序（保持每句长度）。"""
    rng = np.random.default_rng(63099)
    出 = {}
    for 号, 句 in 句集:
        字 = list(句)
        序 = rng.permutation(len(字))
        出[号] = "".join(字[int(i)] for i in 序)
    return 出


def 置乱CER(样本: list, 乱句: dict) -> float:
    """用同一批转写结果对**置乱真值**算 CER 均值（不再调 ASR，省预算且同音频可比）。"""
    if not 样本:
        return None
    return round(float(np.mean([判.CER(乱句[x["号"]], x["转写"]) for x in 样本])), 4)


def _sha256(路径: str) -> str:
    with open(路径, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ───────────────────────── 主流程 ─────────────────────────
def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    np.seterr(all="ignore")

    print("=" * 78)
    print("[学_咬字反馈_ASR] 外部 ASR 作 reward 的离线 θ 搜索（★不在生成路径，C10）")
    print("筛选集 %d 样本：%s ｜ 复评集 %d 样本" % (len(筛选对), 筛选对, len(复评对)))
    print("基线 θ = %s ｜ 可分度下限 = %.2f ｜ ASR 预算 = %d" % (基线θ, 可分度下限, 预算))

    引 = 判.载引擎()
    print("ASR 引擎：%s" % 引["名"], flush=True)

    # —— 1) 基线（筛选集）——
    基筛 = 评(筛选对, 基线θ, 引)
    print("\n[基线] 筛选 CER = %s ｜ 可分度 = %.4f ｜ 已用 ASR %d 次"
          % (基筛["CER"], 基筛["可分度"], 调用["次数"]), flush=True)

    # —— 2) 一轮坐标上升（每维两值，取筛选 CER 更低者；平局保留当前值）——
    print("\n[搜索] 一轮坐标上升（确定性网格）")
    θ = dict(基线θ)
    轨 = []
    for 维, 值s in 网格.items():
        记录s = []
        for v in 值s:
            t = dict(θ)
            t[维] = v
            r = 评(筛选对, t, 引)
            记录s.append(dict(维=维, 取值=v, CER=r["CER"], 可分度=r["可分度"],
                             约束丢弃=bool(r["约束丢弃"]), 采纳=False, θ=dict(t)))
            print("  试 %-6s = %-6s ⇒ 筛选CER %-6s 可分度 %.4f%s（ASR 累计 %d）"
                  % (维, v, r["CER"], r["可分度"],
                     "  ★约束丢弃" if r["约束丢弃"] else "", 调用["次数"]), flush=True)
        有效 = [x for x in 记录s if x["CER"] is not None]
        if 有效:
            # 键 = (CER 升, 是否偏离当前值) ⇒ 平局保当前值（不改就不改，C13 可复现）
            选 = min(有效, key=lambda x: (x["CER"], 0 if x["取值"] == θ[维] else 1))
            采纳 = bool(选["取值"] != θ[维])
            选["采纳"] = 采纳
            θ = dict(选["θ"])
            print("  ⇒ %s 选 %s（%s）" % (维, 选["取值"], "改动" if 采纳 else "保持当前值"),
                  flush=True)
        else:
            print("  ⇒ %s 两值均被可分度约束丢弃 ⇒ 保持当前值" % 维, flush=True)
        轨.extend(记录s)

    # —— 3) 各维最优组合 → 复评一次 ——
    组 = 评(筛选对, θ, 引)
    组合采纳 = bool(组["CER"] is not None and 基筛["CER"] is not None
                and 组["CER"] < 基筛["CER"] - 1e-12)
    学θ = dict(θ) if 组合采纳 else dict(基线θ)
    print("\n[组合] 组合筛选 CER = %s ｜ 基线筛选 CER = %s ⇒ %s"
          % (组["CER"], 基筛["CER"], "采纳" if 组合采纳 else "不采纳（保留基线 θ）"), flush=True)

    # —— 4) 复评集（18 样本）——
    基复 = 评(复评对, 基线θ, 引)
    if 学θ == 基线θ:
        学复 = 基复
        复评复用 = True
    else:
        学复 = 评(复评对, 学θ, 引)
        复评复用 = False
    学可 = round(可分度(学θ), 4)
    print("\n[复评] 基线 %.4f ｜ 搜索后 %.4f ｜ 可分度 基线 %.4f → 学后 %.4f"
          % (基复["CER"], 学复["CER"], 基筛["可分度"], 学可), flush=True)

    # —— 4b) 迁移判定（留出检验，不参与选择）——
    #   筛选集小（6 样本）⇒ 选出来的改动可能是拟合噪声；复评集（18 样本）是**留出**检验。
    #   复评不改善 ⇒ 判定「无可迁移增益」，最终 θ 保留基线（不硬凑）。
    复评Δ = round(float(学复["CER"]) - float(基复["CER"]), 4)
    可迁移 = bool(复评Δ <= 0.0)
    最终θ = dict(学θ) if 可迁移 else dict(基线θ)
    结论 = ("有可迁移增益 ⇒ 采用学到 θ" if 可迁移
            else "无可迁移增益，保留基线 θ")
    print("[迁移] 复评 Δ = %+.4f ⇒ %s ｜ 最终 θ = %s" % (复评Δ, 结论, 最终θ), flush=True)

    # —— 5) must-fail·置乱真值（同一批转写，换无意义真值再评）——
    乱句 = 置乱真值()
    前 = 置乱CER(基复["样本"], 乱句)
    后 = 置乱CER(学复["样本"], 乱句)
    Δ = round(float(前) - float(后), 4)
    置乱通过 = bool(Δ <= 0.02 + 1e-12)
    print("\n[must-fail·置乱] 前 %.4f → 后 %.4f ｜ Δ = %.4f ｜ 通过 = %s"
          % (前, 后, Δ, 置乱通过), flush=True)

    # —— 6) 确定性：最终 θ 合成筛选集两趟，wav sha256 逐位同 ——
    设θ(最终θ)
    趟1 = 合(筛选对, "_趟1")
    趟2 = 合(筛选对, "_趟2")
    哈1 = [_sha256(p) for _号, _名, p, _n, _元 in 趟1]
    哈2 = [_sha256(p) for _号, _名, p, _n, _元 in 趟2]
    两趟逐位同 = bool(哈1 == 哈2 and len(哈1) == len(筛选对))
    print("\n[C13] 两趟逐位同 = %s ｜ sha256 首段 %s" % (两趟逐位同, 哈1[0][:16]), flush=True)

    # —— 7) 台账 ——
    注 = ("★ ASR 只作**外部 reward/裁判**，**不在生成路径**（C10；生成路径零外挂由 审计_零外挂.py 独立证明）；"
          "θ 需由**主控冻结进 `韵律_v8.参数`**（本件只在运行期临时覆盖，不改文件）。"
          "搜索=一轮坐标上升（确定性网格，无随机）；筛选集=S1/S3/S6×EmoCompanion/SpeakerA（6 样本），"
          "复评集=6 句×3 角色（18 样本，不参与选择）；平局保留当前值；"
          "约束=皮层可分度≥0.75（不达标即丢弃，且不花 ASR 预算）；"
          "置乱 must-fail 复用同一批转写（同音频、换真值）⇒ 不额外耗预算；"
          "复评集（18 样本）**不参与选择**，只作留出检验：复评不改善 ⇒ 判定「无可迁移增益」、最终 θ 保留基线（不硬凑）；"
          "鼻尾帧数句中=7 不在网格内、全程固定；每次转写用唯一文件名（角色+句号+全局计数）防路径缓存。")

    报 = dict(
        工具="学_咬字反馈_ASR.py", 版本="v1（外部ASR可懂度 · 一轮坐标上升 · 筛选/复评分离）",
        引擎=引["名"],
        筛选集=[dict(号=号, 角色=名, 句=句表[号]) for 号, 名 in 筛选对],
        复评集=[dict(号=号, 角色=名, 句=句表[号]) for 号, 名 in 复评对],
        网格=网格, 可分度下限=可分度下限,
        基线θ=dict(基线θ), 学到θ=dict(学θ), 组合θ=dict(θ), 组合采纳=组合采纳,
        搜索轨迹=[dict(x) for x in 轨],
        基线筛选CER=基筛["CER"], 组合筛选CER=组["CER"],
        基线可分度=基筛["可分度"], 组合可分度=组["可分度"], 学后可分度=学可,
        基线筛选逐样本=基筛["样本"], 组合筛选逐样本=组["样本"],
        基线复评CER=基复["CER"], 搜索后复评CER=学复["CER"],
        复评Δ=复评Δ, 可迁移=可迁移, 最终θ=dict(最终θ), 结论=结论,
        复评沿用基线=bool(复评复用),
        基线复评逐样本=基复["样本"], 搜索后复评逐样本=学复["样本"],
        置乱=dict(真值=乱句, 种子=63099, 前=前, 后=后, Δ=Δ, 通过=置乱通过),
        must_fail_置乱失败=bool(not 置乱通过),
        两趟逐位同=两趟逐位同,
        两趟sha256=dict(趟1=哈1, 趟2=哈2),
        ASR调用次数=int(调用["次数"]), 预算=int(预算),
        注=注,
    )
    with open(出JSON, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)

    # —— 8) 控制台汇总 ——
    print("-" * 78)
    print("每维选择：")
    for x in 轨:
        print("  %-6s 试 %-6s CER %-6s 可分度 %.4f%s%s"
              % (x["维"], x["取值"], x["CER"], x["可分度"],
                 " ★丢弃" if x["约束丢弃"] else "", "  ⇒采纳" if x["采纳"] else ""))
    print("学到 θ = %s" % 学θ)
    print("最终 θ = %s（%s）" % (最终θ, 结论))
    print("筛选 CER：基线 %s → 组合 %s" % (基筛["CER"], 组["CER"]))
    print("复评 CER：基线 %s → 搜索后 %s（Δ %+.4f）%s"
          % (基复["CER"], 学复["CER"], 复评Δ, "（未采纳改动，同一批）" if 复评复用 else ""))
    print("置乱 Δ = %.4f（前 %.4f → 后 %.4f）通过 = %s" % (Δ, 前, 后, 置乱通过))
    print("ASR 调用 %d / 预算 %d ｜ 两趟逐位同 = %s" % (调用["次数"], 预算, 两趟逐位同))
    print("JSON:", 出JSON)
    print("[学_咬字反馈_ASR] 完成")
    H8.参数.clear()
    H8.参数.update(_默认θ)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
