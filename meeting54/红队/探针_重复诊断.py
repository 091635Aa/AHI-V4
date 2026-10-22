# -*- coding: utf-8 -*-
"""红队 R1 诊断探针（meeting54）：重复到底有多严重 + 根因。只读，不写既有件。<30s。"""
import sys, os, re, json, collections, itertools

挂 = r"<workspace>/03_AHI\AHI4\挂载"
sys.path.insert(0, 挂)
import 角色包 as 包模块

独白 = r"<workspace>/03_AHI\AHI4\挂载\产物\说话_三角色连续独白.txt"
文本 = open(独白, encoding="utf-8", errors="ignore").read()

# ── 解析：角色 → 段列 ──
角色段 = {}
当前 = None
for 行 in 文本.splitlines():
    m = re.match(r"^=+ (.+)$", 行.strip())
    if m:
        当前 = m.group(1).strip()
        角色段[当前] = []
        continue
    m2 = re.match(r"^\s*第\d+段：(.*)$", 行)
    if m2 and 当前:
        角色段[当前].append(m2.group(1).strip())

import 说话_连贯 as 连

class _假LM:
    阶 = 5
    def 分布(self, *a, **k):
        return {}
_假LM实例 = _假LM()

别名 = {'洛天依':'洛天依', 'MOSS（550W）':'MOSS', 'EmoCompanion（打标日记）':'日记角色'}
def 找包(名):
    名 = 别名.get(名, 名)
    全部 = 包模块.全部包()
    if 名 in 全部: return 全部[名], 名
    for k in 全部:
        if k in 名 or 名 in k or (k == 'MOSS' and 'MOSS' in 名): return 全部[k], k
    for k in 全部:
        if k.startswith(名[:2]): return 全部[k], k
    raise KeyError(名)

def 切句(s):
    return [x.strip() for x in re.split(r"[。！？；]", s) if x.strip()]

def distinct_n(序列, n):
    if not 序列: return 0.0
    g = set(tuple(序列[i:i+n]) for i in range(len(序列)-n+1))
    tot = max(1, len(序列)-n+1)
    return len(g)/tot

def selfbleu(段列, n=3):
    """简化 self-BLEU-n：每段 n-gram 对「其余段池」的最大计数精确率，取平均（越低越重复）。"""
    池 = collections.Counter()
    for s in 段列:
        for i in range(len(s)-n+1): 池[s[i:i+n]] += 1
    分数 = []
    for s in 段列:
        cnt = collections.Counter(s[i:i+n] for i in range(len(s)-n+1))
        if not cnt: continue
        命 = 0
        for g, c in cnt.items():
            可用 = 池[g] - c if (池[g] - c) > 0 else 0
            命 += min(c, 可用)
        分数.append(命/sum(cnt.values()))
    return sum(分数)/len(分数) if 分数 else 0.0

汇总 = {}
for 名, 段列 in 角色段.items():
    句列 = [j for s in 段列 for j in 切句(s)]
    全字 = "".join(段列)
    # 池（不含末句号）
    池 = [x.rstrip("。") for x in 连.说话器(找包(名)[0], _假LM实例).句子池()] if False else None
    try:
        包 = 找包(名)[0]
        器 = 连.说话器(包, _假LM实例)   # 通用LM=None 只用于 句子池()
        池 = [x.rstrip("。") for x in 器.句子池()]
        池源 = "句子池()"
    except Exception as e:
        池, 池源 = [], "ERR:%r" % (e,)
    # 逐句来源：独白句是否逐字出现在池里
    命中 = sum(1 for j in 句列 if any(j == p or j in p for p in 池))
    # 段级重复
    段唯一 = len(set(段列)); 段重复率 = 1 - 段唯一/max(1, len(段列))
    # 相邻段句集重叠
    ov = []
    for a, b in zip(段列, 段列[1:]):
        A, B = set(切句(a)), set(切句(b))
        ov.append(len(A & B)/max(1, len(A | B)))
    # 每段句数
    句数 = [len(切句(s)) for s in 段列]
    汇总[名] = dict(
        段数=len(段列), 句数=int(len(句列)), 总字=int(len(全字)),
        池大小=len(池), 池来源=池源,
        段唯一=段唯一, 段重复率=round(段重复率, 3),
        句唯一=int(len(set(句列))), 句重复率=round(1 - len(set(句列))/max(1, len(句列)), 3),
        逐句池命中率=round(命中/max(1, len(句列)), 3),
        distinct_1字=round(distinct_n(全字, 1), 3),
        distinct_2字=round(distinct_n(全字, 2), 3),
        distinct_2句=round(distinct_n(句列, 2), 3),
        selfbleu3=round(selfbleu(段列, 3), 3),
        相邻段句集重叠均值=round(sum(ov)/len(ov), 3) if ov else 0.0,
        每段句数_min=min(句数), 每段句数_max=max(句数),
        句长均值=round(sum(len(j) for j in 句列)/max(1, len(句列)), 1),
        最长重复句=collections.Counter(句列).most_common(1)[0] if 句列 else None,
        最高重复次数=collections.Counter(句列).most_common(1)[0][1] if 句列 else 0,
    )

# ── 机制复算：按 说() 的选句公式复现，看能不能逐字重现独白 ──
def 复现(名, 拍, 句数=4, 桶=0):
    包 = 找包(名)[0]
    器 = 连.说话器(包, _假LM实例)
    池 = 器.句子池()
    n = max(1, min(int(句数), len(池)))
    步 = 1 if len(池) < 5 else 5
    选, 已, i = [], set(), 0
    while len(选) < n and i < len(池)*4:
        idx = (桶 + i*步 + int(拍)) % len(池)
        if idx not in 已:
            已.add(idx); 选.append(池[idx])
        i += 1
    return "".join(选)

机制核对 = {}
for 名, 段列 in 角色段.items():
    能复现 = 0; 试 = 0; 命中拍 = []
    for t, s in enumerate(段列):
        for 桶 in range(0, 60):
            for 句数 in (3, 4, 5):
                试 += 1
                if 复现(名, t, 句数, 桶) == s:
                    能复现 += 1; 命中拍.append((t, 桶, 句数)); break
            else: continue
            break
    机制核对[名] = dict(段可复现=能复现, 段数=len(段列), 命中样例=命中拍[:4])

print(json.dumps(dict(汇总=汇总, 机制核对=机制核对), ensure_ascii=False, indent=1))

