# meeting65 · E1：LST 技术解剖与可迁移要素

> **本件只新增，未改既有件。**
> 席：E1（外部方案解剖）｜日期：2026-10-20｜交付物：本文件 + `meeting65/产物/E1/LST_事实卡.json`（读数卡）
> 一手依据：ICLR 2026 相机稿 PDF 全文（`meeting65/产物/文献/LST_全文.txt`，974 行，62,589 字符；PDF 3,167,035 B，sha256 `60381BAF44BDBFA59719B1AC44BF6C10ED1EB966B03D3ACE894E7E9FF81A8FAB`）。
> **本文件正文里所有 `LST_全文.txt:NNN` 行号均指该抽取文本**，可逐条复核（复核命令见 §12）。
> 本件不含任何 AHI4 侧运行；凡涉及 AHI4 的落点、收益、代价，一律标 `[提案]`/`[推断]`/`未测`，不得读作结论。
> 证据标签按简报 §6：`[文献-原文]`＝相机稿抽取文本；`[文献]`＝ICLR 摘要页；`[文献-二手]`＝检索摘要/二手笔记（未取正文）；`[源码]`＝本仓源码；`[实测]`＝本机真跑真读；`[推断]`/`[提案]`/`未测` 同简报。

---

## 0. 结论先行（给 Lead 的五条）

**K1｜LST 不是新架构，是 BLT 的"语音版迁移 + 多模态验证"。** 它的三段式（patch encoder → global transformer → patch decoder）整套来自 byte-latent transformer（Pagnoni et al., 2024），语音 token 来自 HuBERT（501 码本 @25Hz），文本 token 来自 Llama-2 BPE（32K），交错预训练范式来自 SpiritLM（Nguyen et al., 2025）。LST 的**增量**集中在三处：①把 patching 只施加在 speech span 上；②提出并比较 **static / aligned(词级) / mixed / curriculum** 四种切分策略；③在 speech-text 双向（理解基准 + ASR/TTS 迁移）上把这条机制的证据做实。其余（不用 hash embedding、speech-BPE 无效、scaling 曲线、词级聚类诊断）属**负结果与证据性增量**，逐条见 §4 边界表。`[文献-原文]`（L103-108 / L240-249 / L250-262 / L769-793 / L382-396）

**K2｜"latent speech patch" 的定义是纯切分问题，不是表征学习问题。** patch = 一段连续帧的下标集合 `Pi`，`zi = PatchEnc(X_Pi)`；**怎么切**（static p=4 / 词级对齐 / 混合 / 课程）是论文的主变量，**怎么聚合**（1 层 1024 维滑窗自注意力+交叉注意力，查询由段内 token 隐状态池化初始化）是 BLT 的既有件。端到端只用 token 级似然训练，**没有 patch 级辅助损失**。`[文献-原文]`（L263-270 / L244-245 / L769-780 / L804-809）

**K3｜+6.5% 已核实，但必须带口径。** compute-controlled（同迭代数、同"单位"预算）HellaSwag S→S：39.0 → 45.5（**+6.5**，来自 Table 3 的 Base SpeechLLM 与 LST(Curriculum) 两行）；data-controlled（同数据量）：40.2 → 45.5（**+5.3**）。摘要里那句 "+6.5% absolute gain on speech HellaSwag in compute-controlled (+5.3% in data-controlled)" 与表格逐位对得上。`[文献-原文]`（L404-421 / L428-444）`[文献]`（摘要页同句）

**K4｜它省算力的真实来源是"序列变短"，不是"模型变聪明"。** 同算力下内容吞吐 ×≈4；序列长度 3.00 → 2.42（约 20% FLOPs，按注意力二次依赖口径）；训练算力省 19.3–19.7%（Table 4 的 Compute Savings 列）；TTS 自回归步数 ≈÷4、ASR 上下文单位 ≈×0.25。而 T→T 也涨（+5.2 / +2.6）这件事，主要来自**预算在两种模态间的重新分配**，不宜读成"语音知识迁移到了文本"。`[文献-原文]`（L938-963 / L414-421 / L452-461 / L428-436）`[推断]`

**K5｜对 AHI4 而言，LST 的"省算力"卖点在本地几乎没有消费者。** AHI4 发声链路是参数化物理声道（实时率 0.0451），算力不是瓶颈；AHI4 的韵律 θ 本来就工作在音节/音素粒度，**LST 花外部对齐器换来的"单位边界"，AHI4 天然免费拥有**。因此真正可能拿走的不是"patch 化省算力"，而是三件轻活：**单元/边界设计经验（含静音独立成单元）、单元级诊断口径、patch 级"信息瓶颈"式反馈信号**；而"patch 作为自回归单元"这一 LST 的承重机制，离开 Transformer 就没有可分离载体。`[源码]`（`meeting64/实施/韵律_v9.py:44-45,122-123`；`meeting63/实施/耳_内生.py:29-33`）`[推断]`

---

## 1. 取件与网络登记（`[实测]`，2026-10-20 09:30–10:0x）

| # | 目标 | 手段 | 结果 | 处置 |
|---|---|---|---|---|
| R1 | `proceedings.iclr.cc` 摘要页 | web_fetch | **HTTP 200**，取到标题/作者/完整摘要/"Code is available at" | 用 `[文献]` |
| R2 | 会议相机稿 PDF `…-Paper-Conference.pdf` | web_fetch | HTTP 200 但 `unsupported content type "application/pdf"`（工具不解码 PDF） | — |
| R3 | 同上 PDF | pwsh `Invoke-WebRequest` | **HTTP 200，3,167,035 B**，sha256 `60381BAF…81A8FAB` | 下载成功（临时目录）；**与 Lead 已落盘的 `meeting65/产物/文献/LST_ICLR2026.pdf` 同哈希** |
| R4 | 全文文本 | 读 Lead 抽取件 | `meeting65/产物/文献/LST_全文.txt`，63,983 B / 974 行 / 62,589 字符，含 `===== PAGE 1..18 =====` 分页标记 | **本件一手依据** |
| R5 | `github.com/facebookresearch/lst`（README） | pwsh HEAD + web_fetch | pwsh 12s **超时**；web_fetch `TypeError: fetch failed` | **代码未取到** |
| R6 | `raw.githubusercontent.com`（xht252/paper-notes、zhaoyang97/Paper-Notes-en 两篇 ICLR2026 笔记） | web_fetch | 两次均 `TypeError: fetch failed` | 二手笔记**未取到正文**；只在检索结果里看到标题行 |
| R7 | `arxiv.org/abs/…` | pwsh HEAD | 23.2s **超时** | 未取 |
| R8 | `browse-export.arxiv.org/pdf/2510.06195` | web_fetch | 可达，但同样 `unsupported content type "application/pdf"` | 仅证明该主机可达；**未核对正文**，arXiv 号存疑（见 §6 的 E19） |
| R9 | `mlanthology.org` | — | 简报 §3 登记失败；**E1 未复测** | 不采信 |

判定：**原文（相机稿全文）已到手，代码与其 README 未到手，二手笔记正文未到手。** 因此本件所有机制级结论标 `[文献-原文]`；凡属"代码里才有的实现细节"（课程切换的具体 step 数 τ1/τ2、patch encoder 初始化常数、TTS 解码的 token→波形环节）一律写"未取到/原文未述"，不补白。`[实测]`

---

## 2. 一句话架构：BLT 三段式，语音替换文本 patch

`[文献-原文]`（L103-111 / L240-249 / 图 2 见 L183-186）

```
                        ┌──────────────────── 全局 speech-transformer（25L, d=2048, block-causal RoPE）────────────────────┐
文本 BPE tokens ────────►│  AR 建模对象 = 文本 token 与 speech patch 的交错序列                                        │
                        └────────────────────────────────────────────────────────────────────────────────────────────┘
                                    ▲                                                    │
                    patch encoder（1L, d=1024, 滑窗 512）                  patch decoder（9L, d=1024, NTP loss）
                    聚合「一段语音 token」→ 1 个潜变量 z                   每层：对"已生成 patch+文本"交叉注意力
                                    ▲                                     + 对"过去 512 token"滑窗自注意力
                         HuBERT tokens（501 码本 @25Hz）                              │
                                    ▲                                     → 动态长度 speech tokens（~4/ patch）
                            原始波形（S→S 任务）/ 文本（T→S 任务）                   → （论文未交代的）波形回放
```

论文原文对核心思想的一句话：*"The core idea of the LST architecture is to auto-regressively model latent patches of tokens (using a global transformer), rather than individual tokens, similar in spirit to BLT"*（L240-242）。`[文献-原文]`

**LST 相对 BLT 的两处显式差异**（论文自己列的，L776-780）：
1. LST **只对 speech span** 做 patching（文本仍逐 token）；
2. LST **不用 hash embedding**（消融无收益）；
3. 另加一条：不做 speech-BPE（L253-256 / L779-780，复现 Cuervo & Marxer 的负结果）。

**未述项**：BLT 用"下一 patch 熵"做动态边界预测；LST 全文（§3、§3.1、§4.2、Table 7、A.2.1）列举的四种切分策略里**没有 entropy 边界预测器**，论文也未说明是否去掉。此点只能记 `[推断]`（依据：策略枚举穷尽 + A.2.1 只列了 (i)(ii) 两条差异），不计作已核实。`[推断]`

---

## 3. 技术栈还原

### 3.1 语音分词器：离散 token 从哪来、什么频率

| 项 | 事实 | 定位 |
|---|---|---|
| 来源 | **HuBERT**（Hsu et al., 2021）语义 token：帧表示做 k-means 聚类得到的 cluster id；沿用 Hassid et al. 2023 / Nguyen et al. 2025 的设置 | `[文献-原文]` L194-201 |
| 码本 | **501 个 token** | `[文献-原文]` L199-200 |
| 频率 | **25 Hz**（即 40 ms/token） | `[文献-原文]` L200 |
| 去重 | **不去重**（与 Nguyen et al. 2025 不同）：原文称"这由 LST 架构自然处理"（指向 patch 化） | `[文献-原文]` L200-201 |
| 文本分词器 | **Llama-2 SentencePiece BPE，32K 词表** | `[文献-原文]` L216-217 / L347-348 |
| 对照组 | BPE SpeechLLM：对语音 token 再训 1K SentencePiece BPE（100k 随机语音序列）；5K/10K 无收益 | `[文献-原文]` L393-396 / L400 脚注 |
| 语音数据规模 | 4 套语料，**合计 164,783 小时 / 13.7B HuBERT token**（Table 1：LibriLight 44,174h/3.7B；People's Speech 14,699h/1.2B；MLS 50,601h/4.2B；Spotify 55,309h/4.6B；语音被复用约 6 个 epoch） | `[文献-原文]` L357-362 / L472 |
| 文本数据规模 | Llama-2 语料子集，**1.8T token** | `[文献-原文]` L346-348 |

**内部不一致（原文级）**：§4.1 正文写 "LibriLight (60k hours), People's Speech (30k hours), Multilingual LibriSpeech (50k hours), Spotify (60k hours)"（L350-351），与 Table 1 的 44,174 / 14,699 / 50,601 / 55,309 对不上。本件采信 **Table 1 逐项值**（可相加校验），并把 §4.1 记为正文口误候选。`[文献-原文]` `[推断]`

> 对本会议的含义：**语音侧的"离散 token"完全来自外部神经权重**（HuBERT 编码器 + k-means 码本），文本侧来自外部词表（Llama-2 BPE）。这两件事分别是 C10（生成路径外挂模型）与 C1（Transformer）的直接冲突面，E3 需要逐条对。`[推断]`

### 3.2 latent speech patch：怎么定义、怎么学、粒度、是否与文本单位对齐

**(a) 定义（数学形式，原文照抄）** `[文献-原文]`（L263-270）

令 `X = [x0…xT] ∈ R^{T×d}` 为语音 token 的嵌入序列（`xi` = 查表嵌入），切分是对下标集合的分片 `{Pi}`，`Pi ⊆ {0..T}`，patch 嵌入为

```
zi = PatchEnc(X_Pi)          # 一段连续帧 → 一个潜变量
```

"不同 patch 策略 = 不同 `{Pi}`"。**这就是 LST 的全部"新变量"**：切分方案，而不是聚合网络。

**(b) 四种切分策略** `[文献-原文]`（L271-274 / L318-342 / L382-390）

| 策略 | 定义 | 论文实测的粒度 | 备注 |
|---|---|---|---|
| Static | `Pi = {ip, …, min((i+1)p−1, T)}`，非重叠定长 | **p = 4 HuBERT token**（=160 ms），与对齐无关、训练/推理一致 | 类似 Megabyte 在文本上的做法（Yu et al., 2023） |
| Aligned | 用 Wav2Vec2+CTC 强制对齐拿到文本单位 `[bk,ek]`，`Pk={bk..ek}`；**文本区间外的帧（静音）各自独立成 patch** | 词级：词平均 **5.8** token（≈232 ms）；静音段平均 **3.7** token | 需要推理期也有对齐器（外部模型） |
| Mixed | 每条序列随机选 static 或 aligned | 同上 | 兼顾鲁棒与细粒度同步 |
| Curriculum | 训练按概率 `P(u)` 从 aligned 线性过渡到 static：`P(u)=1 (u<τ1)`；`1−(u−τ1)/(τ2−τ1)`；`0 (u≥τ2)`；实验设定为**前 1/3 aligned、中段 mixed、末段 static** | 训练早期 ≈5.8，评估期 = 4 | **推理时不需要对齐器** |

**(c) 怎么学（聚合器与训练信号）** `[文献-原文]`（L250-262 / L769-793 / L804-809）

- Patch Encoder：**1 层** Transformer，d=1024，16 头，窗口 512；交替的滑窗自注意力 + 交叉注意力。交叉注意力里，**查询由段内语音 token 的隐状态池化初始化**，键/值是段内 token 特征（=BLT 的 latent aggregation 范式）。
- Patch Decoder：**9 层**轻量 Transformer，d=1024，NTP 损失；每层含 (i) 对"最近 512 个已生成 token"的因果自注意力；(ii) 交叉注意力：当前 token 隐状态作 query，**已生成的 speech patch + 文本 token** 作 key/value。
- 全局 Transformer：**25 层**，d=2048，16 头，block-causal，RoPE θ=5×10⁵。
- 训练目标：*"the entire model is trained end-to-end using the same token-level likelihood as before"*（L244-245）→ **没有 patch 级 / 对比 / 重建辅助损失**。
- 作者自述的机制直觉：局部编解码器构成 *"token → latent patch → token information bottleneck"*（L789-793）；patching 是 *"bounded-context temporal aggregation, preserving phonetic discriminability while reducing sequence-level redundancy"*（L247-249）。

**(d) 是否与文本单位对齐？** 分三档回答：

- **Aligned/Curriculum 训练期的确是词（或 BPE）级对齐**：一个 patch = 一个文本单位的语音跨度，静音单独成 patch。`[文献-原文]`（L321-331 / L385-386）
- **Static 不对齐**，只是定长压缩；论文实测 **aligned 在评估时反而更差**（Table 3：Aligned S→S 42.7 vs Static 44.3），机制解释是"变长词 patch 平均更长 ⇒ 同算力下评估的内容更少"。`[文献-原文]`（L433-435）
- **Curriculum 是"训练对齐、推理不对齐"的折中**，也是全表最优：Table 3 的 S→S/T→T = 45.5/52.2。`[文献-原文]`（L413）

**(e) 粒度换算（`[推导]`，算式给出）**

- 25 Hz × 40 ms：1 个 HuBERT token = 40 ms；static p=4 → **1 patch = 160 ms = 6.25 patch/s**。
- 词级 patch 5.8 token ≈ **232 ms**；静音 patch 3.7 token ≈ 148 ms。
- 对照 AHI4：`韵律_v9` 帧率 100（10 ms/帧），一个音节 ≈ 200 ms ≈ **20 帧**；`耳_内生` 帧长 20 ms / 跳 10 ms。`[源码]`（`meeting64/实施/韵律_v9.py:44-45`、`meeting63/实施/耳_内生.py:29-33`）
- 于是：**AHI4 的一个音节在时长上 ≈ LST 的一个词级 patch（~232 ms）；AHI4 的一个音素 ≈ 1–2 个 HuBERT token。** 换言之 AHI4 的"原生单位"（音节）已经比 LST 的 token 粗 5 倍，比 LST 的 static patch 还粗——**AHI4 不缺"压缩"，缺的是"在单元上做分布/序列建模的消费者"。** `[推断]` `[源码]`
- 旁证（**他席读数，E1 未复算**，仅供交叉核对）：E2 同轮探针 `meeting65/产物/E2/读数_序列基座.json` 实测 3 句共 931 帧 / 46 音节 → **帧每音节均值 20.92**、帧每耳段均值 13.04、粗游程压缩比均值 0.0939；与本席的"音节≈21 帧≈210 ms"换算一致。`[实测-他席]`

### 3.3 interleaved speech-text 预训练：数据语法

`[文献-原文]`（L211-221 / L741-757 / L398-426）

- 语料：并行（语音, 转录）→ Wav2Vec2+CTC 对齐 → 得到"文本 token ↔ 语音 token 跨度"。
- 每条训练样本：**随机选一段连续词，用文本 token 替换这段语音；紧接着再取约一半长度的跨度保留为语音 token**。
- 插入模态标记 `<t>`（文本段开始）/`<s>`（语音段开始）。
- 交错序列**训练时动态生成**（每 epoch 不同交错模式）。
- 比例：**语音 token 约占总训练数据 1/3，纯文本 2/3**；交错部分内按词比 1:2（speech:text）。论文与 SpiritLM 的 33/33/33 对比后说"从零训练时 33% 交错 + 66% 文本更好"（L422-426）。
- 效果：同一模型统一支持 **S→S / S→T / T→S / T→T**。

### 3.4 ASR 推理图

`[文献-原文]`（L452-461 / L477-486）

```
波形 → HuBERT(25Hz,501) → patch encoder（curriculum 模型推理期=static 4:1）→ 有效上下文单位 ×0.25
     → 全局 Transformer（AR 出文本 token）→ 文本
微调：LibriSpeech clean，1k / 2k / 4k iters，batch 4，seq len 4096
读数（1B 模型，Table 5，WER% clean/other）：
  Baseline  1k: 140 / 202    2k: 44.7 / 73.2    4k: 20.7 / 42.4
  LST       1k: 6.8 / 10.4  （ctx units 0.25×）  2k: 6.0 / 13.3
```

**必须一起读的三点诚实**：
1. baseline 在 1k 步是崩的（140%/202%，原文说它"频繁幻觉、停止行为不可靠"），因此 6.8 vs 140 这个差**不是稳定解码能力的对比**；到 4k 步 baseline 仍有 >20%/>40%。`[文献-原文]`（L479-482）
2. LST 自己的 1k→2k 在 other 上是 **10.4 → 13.3（变差）**，论文未解释这条非单调。所谓 "stabilizes ASR adaptation" 是**相对 baseline 崩坏**而言，不是绝对稳定。`[文献-原文]`（L460-461）`[推断]`
3. "省 0.25× 上下文"与"WER 更好"是两件事，论文没有做"同上下文长度"的等算力对照，**无法把 WER 改善单独归因给 patching**。`[推断]`（依据：Table 5 没有 patched-baseline-equal-context 臂）

### 3.5 TTS 推理图

`[文献-原文]`（L452-461 / L482-486 / L938-963）

```
文本 token → 全局 Transformer（每步 AR 生成 1 个 speech patch 的潜变量）
          → patch decoder：以该潜变量为条件，展开 ~4 个 HuBERT token（长度动态）
          → （论文未交代的 token→波形 环节）
          → Whisper 转写 → CER
微调：20k 步
读数（Table 5）：Baseline CER 14.1 / 15.1（gen units 1.0×）；LST 14.1 / 16.2（0.25×）
```

**关键未述/要害**：
- 全文只有 1 处出现 "vocoder"（在 §2 背景里泛泛提到 HiFi-GAN，L188-193），**TTS 评估路径的 token→波形环节、以及该环节是否复用 HuBERT 的逆映射/声码器，原文（正文 + 附录）没有交代**。`[文献-原文]` `[推断]`
- 所谓 "without degrading reconstruction quality"（摘要 L31-32）在本表里的实测其实是：**clean 项持平（14.1 vs 14.1），other 项 +1.1 变差（15.1 → 16.2）**，而且"重建质量"是用 **Whisper 文本转写 CER** 代理的，**不是波形保真/MOS/谱失真**。这是本次解剖里最需要 Lead 注意的一处"摘要措辞强于表格读数"。`[文献-原文]`（L452-461 / L484-485）`[推断]`

### 3.6 规模与结果（数字逐条核实）

**训练配置（1B 主模型）** `[文献-原文]`（L811-819）：AdamW（β=0.9/0.95, wd=0.1），lr 4e-4 cosine，warmup 2,000，bf16，32×H100，per-GPU batch 4×4096 units → 总 batch 0.5M units，200k 步 ≈ 1T units，**约 17 小时**。

**理解基准（HellaSwag 语音版 + StoryCloze/TopicStoryCloze）**

| 设置 | 关键读数 | 定位 |
|---|---|---|
| compute-controlled（同迭代数、同"单位"预算） | HellaSwag S→S 39.0→**45.5**（+6.5）；T→T 47.0→52.2（+5.2）；SC S→S 59.1→61.2；TSC S→S 87.5→87.9 | `[文献-原文]` L404-413 |
| data-controlled（同 token 量） | S→S 40.2→**45.5**（+5.3）；T→T 49.6→52.2（+2.6）；**训练算力省 19.7%**（Static 19.3%） | `[文献-原文]` L414-421 |
| 模态差 | 语音-文本性能差 **9.4 → 6.7** | `[文献-原文]` L442-444 |
| 规模（compute-optimal 420M→1.8B） | 420M：语音 29.2 vs 28.4、文本 31.6 vs 31.1；1.8B：语音 39.0 vs 35.3、文本 46.3 vs 45.7 | `[文献-原文]` L462-466 |
| 规模（7B，固定 70B token 预算，低于最优 ≈140B） | 语音 44.2 vs 42.0；文本 55.3 vs 54.8 | `[文献-原文]` L827-833 |
| 稳定性（3 次重复） | HellaSwag：Curriculum S→S 41.4±0.13、Static 40.9±0.67、Baseline 36.5±0.22；SC/TSC 标准差常 >1.5 | `[文献-原文]` L847-856 |
| 序列/算力账 | 序列 3.00 → 2.42（0.81×）≈ **20% FLOPs**；TTS AR 步数 ÷4；ASR 编码上下文 ÷4 | `[文献-原文]` L938-963 |
| 语言结构不退化 | sWUGGY 72.8 vs 72.5；sBLIMP 59.0 vs 58.9（平价） | `[文献-原文]` L925-937 |
| 词级 patch 语义结构 | 同词相似度 ~0.87、异词 ~0.43、silhouette 0.65–0.68（t-SNE，描述性） | `[文献-原文]` L487-495 |

> **口径警告（给 E4 做对抗审查用）**：Table 3（主表，1:2 语音比例，108B 交错 token）与 Table 6（消融，patch size 匹配对照）**不是同一设置**：同一 Curriculum 策略在 Table 3 是 45.5、在 Table 6 是 41.3。任何跨表拼接都是错的。`[文献-原文]`（L404-413 / L553-565 / L902-911）

---

## 4. LST 机制贡献 vs 既有件：边界表

| # | 组件 | 出处 | 是否 LST 增量 | 判定依据 |
|---|---|---|---|---|
| B1 | 离散语音 token（HuBERT 501 @25Hz） | Hsu 2021 / Hassid 2023 / Nguyen 2025 | **既有** | §2 明写沿用；L199-201 |
| B2 | 文本 BPE（Llama-2, 32K） | Touvron 2023 | **既有** | L216-217 |
| B3 | decoder-only 因果自注意力 + NTP 似然 | Vaswani 2017 / 文本 LLM 惯例 | **既有** | L188-209（式 1） |
| B4 | 交错 speech-text 预训练（模态标记、span 替换） | SpiritLM（Nguyen 2025）直接沿用 | **既有** | L211-221 / L741-757；论文自述 "we follow" |
| B5 | **patch encoder（滑窗自注意力+交叉注意力，段内池化查询）** | **BLT（Pagnoni 2024）** | **既有（整套搬）** | L103-108 / L250-256 / L769-780 |
| B6 | **patch decoder（轻量 Transformer，NTP，逐层交叉注意力到 patch）** | **BLT** | **既有** | L257-262 / L781-788 |
| B7 | "按 patch 而非 token 自回归"这一粒度思想 | Megabyte（Yu 2023）/ BLT | **既有** | L240-249 |
| B8 | 用 patch 把多模态序列"信息密度拉平" | BLT 的思想在语音-文本上的首次系统化 | **LST 增量（迁移与验证）** | 摘要 L15-23；L101-111 |
| B9 | **只对 speech span 做 patching** | — | **LST 增量** | L776-777 显式列为 "Unlike BLT" |
| B10 | **四种 speech 切分策略（static / aligned 词级 / mixed / curriculum）+ 课程过渡公式** | — | **LST 增量（核心）** | L271-342 / L382-390 |
| B11 | **对齐 patch 把"静音"独立成单元** | — | **LST 增量（次级）** | L321-331 / L326-329 |
| B12 | **去掉 BLT 的 hash embedding**（消融无收益） | — | **LST 增量（负结果）** | L253-256 / L778-779 |
| B13 | **speech-BPE 不管用**（复现 Cuervo & Marxer） | — | **LST 增量（负结果，与既有结论一致）** | L253-256 / L438-440 |
| B14 | 算力/数据双控下的收益量化 + 420M→1.8B→7B 的 scaling 证据 | — | **LST 增量（证据）** | L428-475 / L820-833 |
| B15 | 词级 patch 嵌入的聚类诊断（t-SNE） | — | **LST 增量（诊断）** | L487-495 |
| B16 | ASR/TTS 下游迁移与"序列变短"的工程交代 | — | **LST 增量（证据，但对照不严）** | L452-486 / L938-963 |
| B17 | 全双工 / inner-monologue / 指令微调 | — | **未做（Limitations 自陈）** | L578-587 |

**一句话边界（回答 Lead 的 Q1 前半）：** 把 BLT 的 text patch 换成 speech patch 的**架构搬用**是既有件；LST 的增量是**"切分策略空间 + 对齐/课程 + 多模态双向验证"**，外加两条负结果（hash embedding、speech-BPE 无效）。它没有提出新的表示学习目标，也没有新的分词器。`[文献-原文]` + `[推断]`

---

## 5. 论文三条主张的可拆性（E1 的独立判断）

论文把三件事捆在一起说成"patching 有效"，本席认为应当拆开：

| 主张 | 支撑读数 | 强度 | 备注 |
|---|---|---|---|
| A. 压缩 → 同算力下内容更多 | Table 3/4/13、Compute Savings 19.3–19.7%、TTS÷4、ASR ctx×0.25 | **强**（口径清楚、可复算） | 本质是吞吐，不是"理解变强" |
| B. 粒度对齐 → 跨模态迁移更好 | Table 6 同 patch 尺寸下 aligned > static（如 SC S→S 60.3 vs 58.7）；t-SNE 同词/异词可分；Table 3 T→T 也涨 | **中** | aligned 在 Table 3 评估反而更差（42.7<44.3）；真正优势只在 curriculum。t-SNE 是描述性证据、无因果臂 |
| C. 下游 ASR/TTS 迁移 | Table 5 | **弱-中** | baseline 崩坏式对照、无等上下文臂、TTS 只报 CER、other CER 反而变差 |

`[文献-原文]`（各定位见 §3.6）`[推断]`：因此**复用 LST 时最小可证伪对象是 B（切分/对齐）**，A 属通用工程账、C 属工程迁移证据；C 不能作为"机制正确"的独立证据。

---

## 6. 证据等级表（含未取到）

| 编号 | 命题 | 等级 | 来源/取件 |
|---|---|---|---|
| E1 | 架构=patch encoder + global transformer + patch decoder（三段式，承自 BLT） | `[文献-原文]` | L103-111 / L240-249 / L769-793 / 图2 L183-186 |
| E2 | 语音 token = HuBERT，501 码本，25 Hz，不去重 | `[文献-原文]` | L194-201 |
| E3 | 文本 token = Llama-2 BPE，32K；语音约为总量 1/3 | `[文献-原文]` | L216-217 / L346-348 / L398-400 |
| E4 | static patch = 4 HuBERT token（160 ms） | `[文献-原文]` | L383-384 / L271-274 |
| E5 | aligned patch = 词级，词均 5.8 token、静音 3.7 token，静音独立成 patch | `[文献-原文]` | L321-331 / L553-565 脚注 L565 |
| E6 | curriculum = 训练从 aligned 线性过渡到 static；前三段式设定 | `[文献-原文]` | L332-342 / L387-390 |
| E7 | 端到端只用 token 级似然，无 patch 级辅助损失 | `[文献-原文]` | L244-245 |
| E8 | 各模块层数/维度/头数（1/25/9；1024/2048/1024；16 头；窗口 512；RoPE 5e5） | `[文献-原文]` | Table 7, L804-809 |
| E9 | compute-controlled：HellaSwag S→S 39.0→45.5（+6.5）、T→T 47.0→52.2 | `[文献-原文]` | Table 3, L404-413 |
| E10 | data-controlled：S→S 40.2→45.5（+5.3）、T→T 49.6→52.2、算力省 19.7% | `[文献-原文]` | Table 4, L414-421 |
| E11 | 模态差 9.4→6.7 | `[文献-原文]` | L442-444 |
| E12 | 规模 420M→1.8B；7B 固定预算 44.2/55.3 vs 42.0/54.8 | `[文献-原文]` | L462-466 / Table 8, L827-833 |
| E13 | ASR：LST 6.8/10.4 @1k（ctx 0.25×），baseline 140/202 @1k；4k 仍 20.7/42.4 | `[文献-原文]` | Table 5, L452-461 |
| E14 | TTS：CER 14.1/16.2 vs 14.1/15.1，生成单位 0.25× | `[文献-原文]` | Table 5, L452-461 |
| E15 | 序列长度 3.00→2.42（≈20% FLOPs）；TTS 步数 ÷4 | `[文献-原文]` | Table 13, L938-963 |
| E16 | 对齐器边界扰动 ±20–40 ms → 最敏感设置退化 ≤~1% | `[文献-原文]` | L964-971 |
| E17 | sWUGGY/sBLIMP 与 baseline 平价（72.8/72.5；59.0/58.9） | `[文献-原文]` | L925-937 |
| E18 | 词级 patch 嵌入：同词 0.87 / 异词 0.43 / silhouette 0.65–0.68 | `[文献-原文]` | L487-495 |
| E19 | arXiv 编号 2510.06195 | `[文献-二手]` **未核实** | 检索结果里出现 `browse-export.arxiv.org/pdf/2510.06195`；正文未取到（PDF 不解码） |
| E20 | "ICLR 2026 **Oral**" | `[文献-二手]` **未核实** | proceedings 页本身只标 "Conference"（`[文献]`）；"Oral" 只见于二次笔记标题的检索摘要（xht252/paper-notes），**笔记正文未取到** |
| E21 | 代码实现（τ1/τ2 具体 step、patch encoder 初始化、TTS token→波形） | **未取到** | github/raw 均 fetch failed（R5/R6） |
| E22 | 作者单位 = JHU CLSP + Meta Superintelligence Labs；代码 URL | `[文献-原文]` | L5-9 / L34 |
| E23 | 原文未述：TTS 的回放环节与词表外 token 处理 | **未述** | 全文仅 1 处 "vocoder"（L192，背景泛述） |
| E24 | 原文内部不一致：§4.1 语音小时数（60k/30k/50k/60k）vs Table 1（44,174/14,699/50,601/55,309） | `[文献-原文]`+`[推断]` | L350-351 vs L357-362 |
| E25 | 原文未解释：LST ASR 1k→2k 的 other WER 10.4→13.3 非单调 | `[文献-原文]` | L460-461 |
| E26 | 原文自相矛盾候选：Limitations 说"未探索下游适配"（L582-583），但 §5.3 做了 ASR/TTS fine-tune（L477-486） | `[文献-原文]`+`[推断]` | 两处对照 |

---

## 7. 可迁移要素表（≥5 条，每条带证伪条件）

> 判定口径按简报 §7：**可迁移** = 用本仓 8 算子 + numpy/scipy 能实现，且不引入外部权重/第三方推理栈，且两趟逐位同；**需改造** = 机制成立但要换载体；**不可用** = 离开 Transformer/预训练权重就没有可分离机制。
> 三列"落点"仅指 `[提案]` 候选，E2 负责落到文件/函数并给判据；本席**没有跑任何 AHI4 侧实验**。

| 编号 | 要素（LST 侧用法） | 能否离开 Transformer/外部权重 | 证据（等级+定位） | 复现难度 | 潜在 AHI4 落点 | **证伪条件（出现即判此要素无效）** |
|---|---|---|---|---|---|---|
| T-1 | **有界上下文聚合**：把一段连续帧聚成一个"单元级"表征（LST：`zi=PatchEnc(X_Pi)`，段内均值池化初始化 + 注意力聚合） | **能**（需改造）：把 learned PatchEnc 换成**确定性段内池化**（mean/std/斜率 + HDC bind/bundle）——8 算子已够 | `[文献-原文]` L247-249 / L263-270 / L769-780（"保留音素可辨性"只在其 learned 版实测：sWUGGY/sBLIMP 平价 L925-937） | **低**（AHI4 侧纯 numpy；~1 天量级） | 韵律层（`韵律_v9.py` 的音节/音素分段，帧率 100）；音素格层（`皮层_音素格.编码(f)` 已是段级特征，只需把"段"上提为"patch"） | AHI4 用确定性聚合实现后，`判据_发音_v9` 的 **G2_皮层可分度** 劣于现状（留一音节准确率 0.8571，chance 0.2，置乱 0.1111，白噪 0.0 —— `[实测]` 读自 `meeting64/产物/判据/判据_发音_v9.json`），**且**无可懂度/CER 改善 ⇒ 判"聚合粒度"在 AHI4 无价值 |
| T-2 | **单元边界对齐文本单位**：patch 边界取词/子词边界（LST：Wav2Vec2+CTC 强制对齐）；且**静音独立成段** | **能，且 AHI4 侧成本≈0**：AHI4 的 θ 自带音素/音节序列与时长，边界是**已知量**，不需要外部对齐器（LST 必须外挂 Wav2Vec2+CTC） | `[文献-原文]` L321-331 / L326-329 / L385-386 / L964-971（±20–40 ms 扰动 ≤1%）+ `[源码]` `韵律_v9.py:44-45,122-123`（`秒×帧率` 直接生成 n 帧） | **极低**（AHI4 侧：边界现成） | 韵律层（音节格/音素格边界、停顿）；听觉层（`耳_内生.切段` 已产 静/清/浊 段） | AHI4 用"θ 边界聚合"与"定长 20 帧聚合"在同一判据上无可分差异（建议阈值：可分度差 <2σ 或 CER 差 <0.02）⇒ 判"边界信息"在 AHI4 无增量（那就只需 T-1，不需要 T-2） |
| T-3 | **课程式切分**：训练早期用对齐单元、后期退化为定长/无对齐；推理免对齐 | **能**（纯调度，无参数） | `[文献-原文]` L332-342 / L387-390 / L553-565（Table 6：Curriculum 41.3 vs Static4 40.5，+0.8；Table 3 里 Curriculum 45.5 vs Static 44.3，+1.2） | **低** | 咬字反馈课程（`meeting63/实施/学_咬字反馈.py` 的 θ 坐标上升：先按已知真值段边界优化，再退化为盲段） | AHI4 里"课程 vs 单阶段（纯按 θ 边界）"在 **P0_外部真人正对照** 与 G2 可分度上无差（差 <2σ）⇒ 判课程无增量，撤回 |
| T-4 | **静音/重复声学模式作为独立单元**，压缩掉低信息帧 | **能** | `[文献-原文]` L326-329（silence 单独成 patch）/ L565（sil 平均 3.7 token）/ 摘要 L22-23（"compactly capture recurring acoustic patterns such as silence"）/ L553-565（Align sil sep 60.3 vs static6 58.7, StoryCloze S→S） | **极低**（`耳_内生` 已切 静/清/浊） | 听觉层（`耳_内生.py` 切段）+ 韵律层（句间停顿/犹豫时长） | 把静音独立成单元后，`判据_发音_v9` 的 **E1_静音占比 / E4_音节率** 与可懂度均无改善 ⇒ 判该要素只是记账、不带来机制（注意：LST 里它带来的是算力，而 AHI4 不缺算力，见 T-5） |
| T-5 | **序列/算力预算账**：4→1 patch ⇒ 序列 3.00→2.42（≈20% FLOPs）、TTS AR 步数 ÷4、ASR ctx ×0.25 | **能**（纯算术，无需任何模型） | `[文献-原文]` L938-963 / Table 4 L414-421（19.3–19.7%）/ Table 5 L452-461 | **极低**（算术） | **当前无算力落点**：AHI4 引擎 v6 实时率 **0.0451**（10 s 合成 0.45 s，简报 §2 `[实测]`），算力不是瓶颈 ⇒ 该要素对 AHI4 的直接价值≈0；只能作"若将来上规模"的记账模板 | 若实测 AHI4 发声/听觉链路耗时主项不在"帧数/单元数"（而在 scipy 滤波、共振峰内循环、Python 层循环），则"压缩 4:1 省 20% 算力"在 AHI4 **无对应收益** ⇒ 判无效。**本席预判此条已不成立**（依据：实时率 0.0451），仍请 E2 用真实计时给 must-fail 读数 |
| T-6 | **token→patch→token 信息瓶颈**作为自监督/重构目标（LST：patch decoder 以 NTP 重建 token；作者称其为 information bottleneck） | **部分能**（需改造）：AHI4 无神经 decoder，可用**确定性逆映射**（patch→段统计→帧参数）或"从 patch 能否重建音素格读出"作替代目标 | `[文献-原文]` L257-262 / L789-793 / L482-486（TTS "reconstruction" 实际只由 Whisper CER 代理，非波形保真） | **中**（要设计确定性逆映射与误差可比性，0.5–2 天） | 咬字反馈（`学_咬字反馈.py` 的误差信号：现为可分度误差，可加 patch 级重构误差）；音素格读出侧 | 用 patch 级重构误差替代/叠加现误差后，`判据_发音_v9` 的 **G3_反馈学习** 与 **P0_外部真人正对照** 不优于现状（或两趟不逐位同 / 引入第三方权重）⇒ 撤回 |

**被本席判为"不可迁移"的候选（不进上表）**：

- **"以 patch 为自回归单元"本身**：它是 LST 的承重机制，但载体必须是 AR 全局 Transformer（25L/2048d，C1）。离开它只剩"切分"，见 N-3/N-4。
- **交错 speech-text 预训练语法**：数据构造规则本身可离开 Transformer（span 替换 + `<t>/<s>` + 动态采样，L741-757），但 AHI4 **没有 NTP 消费者**（既不训文本 LM 也不训语音 LM）⇒ 语法无处落地（N-9）。
- **t-SNE 词级聚类诊断**：可作离线仪器（L1 级），但 AHI4 已有同构且更硬的指标（G2 留一音节可分度 + 置乱/白噪/直读三 sham），**不是新机制**，见 §8。

---

## 8. 不可迁移 / 不可复现项

| 编号 | 项 | 不可迁移理由 | 相关红线 | 证据 |
|---|---|---|---|---|
| N-1 | HuBERT 语音分词器（501 码本 @25Hz） | 外部神经权重 + 聚类码本；进生成路径即外挂模型。AHI4 无"声学 token id"概念（音素格是 HDC 连续码） | **C10**（生成路径） | `[文献-原文]` L194-201 |
| N-2 | Llama-2 SentencePiece BPE（32K） | 外部词表 + 外部分词规则；AHI4 已有 `音韵_v5` 拼音→声母/韵母 | C10（弱，词表非权重） | `[文献-原文]` L216-217 |
| N-3 | 全局 speech-transformer（25 层、d=2048、block-causal、RoPE 5e5） | Transformer 本体 | **C1** | `[文献-原文]` Table 7 L804-809 |
| N-4 | Patch Encoder / Patch Decoder（注意力式，learned） | Transformer 层 + 预训练权重；本地无权重、无训练算力，C13 不可复现 | **C1 + C13** | `[文献-原文]` L250-262 / L769-788 |
| N-5 | 训练规模（1B/7B、32×H100、1.8T 文本 token、164,783 h 语音、~1T units、17 h/次） | 硬件/数据/许可三重不可得；任何"复现 LST"主张都不成立 | **C13** | `[文献-原文]` L346-362 / L811-819 |
| N-6 | Wav2Vec2+CTC 强制对齐器 | 外部神经模型；*生成方向* AHI4 不需要（θ 自带边界），*听觉方向* 无权重 → 两头都不可用 | **C10** | `[文献-原文]` L328-331 |
| N-7 | Kokoro-82M（造 sHellaSwag 语音）+ Whisper（CER） | 外部模型，仅评测仪器用途；AHI4 已确立"离线仪器可"的口径（meeting63 §5，ASR 只算 CER）→ 属 L1 仪器层，不是机制；且本机网络取不到权重 | C10（仪器豁免面）+ 网络 | `[文献-原文]` L366-375 / L484-485；`[文献]` 简报 §1 |
| N-8 | BLT 的 hash embedding / entropy 边界预测 | hash embedding 论文明确未用（L778-779）；entropy 边界**原文未述**（`[推断]`），无据可借 | — | `[文献-原文]` / `[推断]` |
| N-9 | 交错数据语法的消费者（NTP AR LM） | 语法可离开 Transformer，但 AHI4 无 NTP 训练对象（既不训文本 LM 也不训语音 LM） | C1 | `[文献-原文]` L741-757 + `[推断]` |
| N-10 | 全双工 / inner-monologue / 指令微调 | 论文自陈未做（"not yet address full-duplex"; "without exploring instruction fine-tuning"） | — | `[文献-原文]` L578-587 |
| N-11 | 原样复现 Table 3/5/6 的**任何**数字 | 依赖 N-1..N-6 全套；即便拿到代码也无权重/数据/算力 | C13 | `[实测]` R5/R6 取不到代码 |
| N-12 | 把 +6.5% 当作"AHI4 可期待收益" | 口径不同：LST 的收益是"同算力下吞吐 ×4"，AHI4 的瓶颈是**可懂度/自然度**（meeting62 G1–G9、v8 CER 0.771、v9 CER 未测），不是吞吐 | — | 简报 §2 `[实测]`；`[推断]` |

---

## 9. 未测项（空着＝未测，不写"预计/应该"）

| 编号 | 未测项 | 归属 | 为什么未测 / 怎么才能测 |
|---|---|---|---|
| U-1 | 把 T-1..T-6 任何一条接进 AHI4 后，对 CER/可懂度/可分度/韵律的影响 | E1→E2 | 本轮纪律=纯解剖，零运行；需 E2/E4 预注册探针 |
| U-2 | **确定性聚合能否替代 learned PatchEnc 并保持"音素可辨性"** | 机制核心未知 | 论文只测了 learned 版（sWUGGY/sBLIMP 平价 L925-937）；没有"确定性池化"臂 ⇒ 这是跨方案的**外推**，必须做 must-fail 探针 |
| U-3 | LST 在**中文/声调语言**上的表现 | 原论文 | 训练数据 4 套全英文（LibriLight/People's Speech/MLS 英文部分/Spotify），评估 3 套全英文；声调、音节结构、停顿形态与 AHI4（中文）差异大 |
| U-4 | LST 对**韵律/情感/音色**的表现 | 原论文 | 全文没有任何韵律/情感/说话人指标；AHI4 最关心的三项恰是 LST 未测的三项 |
| U-5 | 代码级细节：课程 τ1/τ2 具体 step、patch encoder 初始化、动态 patch 长度上限、loss 权重 | 代码 | github/raw 取不到（R5/R6）；论文只给公式形状与"前 1/3"这类相对刻度 |
| U-6 | **TTS 的 token→波形 环节** | 原论文 | 全文（正文+附录）未交代；只有 Whisper-CER 代理。"不降重建质量"因此**没有波形级证据** |
| U-7 | patch 尺寸对**静音占比**的敏感性（中文停顿多） | 跨方案 | 论文只报 sil 平均 3.7 token；不同语速/停顿分布下的最优 p 未测 |
| U-8 | "同算力更多内容" vs "粒度更对齐"的**独立归因** | 原论文 | 论文两种控制都报，但没有"同内容同算力、只变粒度"的第三臂 ⇒ A/B 两主张仍有耦合 |
| U-9 | AHI4 侧"若为省算力而 patch 化"是否有意义 | AHI4 | 实时率 0.0451 已说明算力不紧；但"链路上限"与"将来规模化"未测 |
| U-10 | AHI4 若接 L1 离线仪器（Whisper/Paraformer 类 CER、t-SNE 类诊断）的合规边界 | E3 | 已有 meeting63 口径（ASR 只算 CER），但"LST 式 patch 诊断"是否算仪器需 E3 判 |

---

## 10. 给 Lead / E2 / E3 / E4 的交接（含 Q1、Q2 的 E1 输入）

### 10.1 回答 Lead 收口问题 Q1（"真正值得拿的是不是只有 patch 聚合 + 对齐"）

**E1 判定：比"一条"更窄，也比"一条"更宽。** `[推断]`

- **更窄**：LST 的"聚合"是 BLT 既有件，AHI4 真正能拿走的是**"切分/边界的设计空间"**（含静音独立、课程过渡），不是聚合网络；而"以 patch 为 AR 单元"是承重件、不可拿（C1）。
- **更宽**：另外还有两件非机制性但有用的东西——①**单元级诊断口径**（同词/异词可分、silhouette 型指标；AHI4 的 G2 已同构，可直接借用其"报告口径"），②**"信息瓶颈"式反馈信号**的候选（T-6，唯一可能改变学习信号的一条）。
- **明确不成立**：+6.5% / 省 20% 算力 **不能**作为 AHI4 的收益预期（N-12；AHI4 不提吞吐）。

### 10.2 回答 Lead 收口问题 Q2（落点是韵律层还是音素格层？能否同时？）

`[提案]`（E1 建议，需 E2 落到文件）

- **主要落点＝韵律层**：因为 AHI4 的"单元"定义权在 `韵律_v9.py`（音节/音素序列 + 时长 + F0），T-1/T-2/T-3/T-4 都是"单元怎么切、要不要对齐、要不要课程"的问题，天然属于韵律层。
- **次要落点＝音素格层**：只限"读出侧聚合"（把 `皮层_音素格.编码(f)` 的段级特征上提为 patch 级），以及 T-6 的"从 patch 重建读出"目标。**音素格层不能改变切分**（切分在韵律层），所以两层**可以同时接，但不共享同一套证据**：韵律层的证伪看发音判据（`判据_发音_v9` 的 A–F 臂 + P0），音素格层的证伪看 G2 可分度（0.8571/chance 0.2/置乱 0.1111/白噪 0.0）与 G3 反馈学习。
- **关键提醒**：AHI4 缺"patch 级序列模型"这个消费者。所以两层都接，收益上界也只是"分段更合手 / 读出更稳"，不是"LST 式理解力"。`[推断]`

### 10.3 给 E3 的红线输入（只给"哪条要素需要哪一级"，不代判）

| 要素 | 不接外部权重即可实现？ | 是否需要动红线 |
|---|---|---|
| T-1 有界上下文聚合（确定性版） | 是（numpy/scipy + 8 算子） | **L0 即可，一条红线都不用改** |
| T-2 边界对齐（用 θ 的时长） | 是 | L0 |
| T-3 课程 | 是 | L0 |
| T-4 静音独立单元 | 是 | L0 |
| T-5 算力记账 | 是 | L0（且无收益） |
| T-6 信息瓶颈式反馈（确定性逆映射版） | 是 | L0 |
| 若有人主张"直接跑 LST / 用其权重做教师" | 否（需 1B+ 权重 + HuBERT/Windows 生态 + GPU） | **≥L2，且本机网络取不到权重（R5/R6）、C13 无法保证** |

结论要点：**E1 侧没有一条"值得拿"的要素需要放宽 C1/C10**；所有放宽诉求都来自"LST 整体当教师/当骨架"，而那部分在 E1 看来连可复现性都不成立。`[推断]`

### 10.4 给 E4 的对抗审查提示（本席自认为最可能错在哪）

1. **最可能高估**：T-2 的"AHI4 免费获得对齐"——这只在**生成方向**成立；在**听觉/理解方向** AHI4 同样没有对齐器，必须重新挣。若 E2 把 T-2 当成"双向免费"，就错了。
2. **最可能低估**：T-6。信息瓶颈是本席看到的唯一"可能改变学习信号"的要素，若探针把"确定性逆映射"设计得太弱（例如只用线性回归），失败会被误读成"机制无效"而非"载体太弱"。**must-fail 臂必须包含"同预算但无瓶颈"的 sham。**
3. **最可能被误引**：+6.5%。若有人在纪要里写"借 LST 机制可期 +6.5%"，请 E4 直接按本件 §3.6 的口径反驳（那是同算力吞吐口径，不是 AHI4 指标）。
4. **最可能漏掉**：aligned patch **单独用会更差**（Table 3：42.7 < 44.3）。迁移时若只拿"对齐"不拿"课程/静态推理"，可能先吃一次负收益，然后被误判为"方向错"。

---

## 11. 引用与失败登记

**一手（已取到）**
- ICLR 2026 proceedings 摘要页：<https://proceedings.iclr.cc/paper_files/paper/2026/hash/6276ec46607b50a014c0db6525e72f71-Abstract-Conference.html>（`[文献]`，web_fetch HTTP 200）
- 相机稿 PDF：<https://proceedings.iclr.cc/paper_files/paper/2026/file/6276ec46607b50a014c0db6525e72f71-Paper-Conference.pdf>（`[实测]`，3,167,035 B，sha256 `60381BAF44BDBFA59719B1AC44BF6C10ED1EB966B03D3ACE894E7E9FF81A8FAB`）
- 本机抽取全文：`meeting65/产物/文献/LST_全文.txt`（Lead 落盘，974 行；与本席下载件**同哈希**，可判定同源）

**二手（仅检索摘要，正文未取到）**
- xht252/paper-notes 的 ICLR2026 笔记（标题行可见 "\[ICLR 2026 Oral\]\[音频/语音\]\[speech-text modeling\]"）：`github.com/xht252/paper-notes/blob/main/docs/ICLR2026/audio_speech/latent_speech_text_transformer.md` —— web_fetch / raw 均失败，**未取正文**
- zhaoyang97/Paper-Notes-en 同名笔记：同上，**未取正文**
- arXiv 镜像 `browse-export.arxiv.org/pdf/2510.06195`：主机可达但 PDF 不解码，**未核对**，编号存疑

**失败登记（`[实测]`）**
- `github.com/facebookresearch/lst`：pwsh HEAD 12 s 超时；web_fetch README `TypeError: fetch failed`。**代码与 README 均未取到。**
- `raw.githubusercontent.com/*`：web_fetch `TypeError: fetch failed`（两次）。
- `arxiv.org/abs/…`：pwsh HEAD 23.2 s 超时。
- `mlanthology.org`：简报 §3 登记失败，E1 未复测。

**引用纪律声明**：本件**没有**使用任何二手笔记的实质内容下结论；二手来源只用于两件事——(a) 发现"Oral"这一未核实属性（E20），(b) 记录检索路径。凡与二手冲突处，一律以相机稿全文为准。

---

## 12. 附：复核指引（给 Lead/E4 逐条验伪）

```powershell
$p = '<workspace>/03_AHI\AHI4\meeting65\产物\文献\LST_全文.txt'
# 1) 架构/分词器
Select-String $p -Pattern 'byte-latent|501|25Hz|Patch Encoder|Patch Decoder' | Select -First 12
# 2) 切分策略与课程公式
Select-String $p -Pattern 'Static Patching|Alignment Patching|Curriculum Patching|4 HuBERT'
# 3) 主结果（+6.5 / +5.3）
Select-String $p -Pattern '45.5|39.0|40.2|52.2|49.6'
# 4) 下游与算力账
Select-String $p -Pattern '6.8|10.4|140|14.1|16.2|2.42|19.7'
# 5) 我对原文的两处挑刺
Select-String $p -Pattern '60k hours|44,174|not yet address full-duplex|instruction fine-tuning'
```

AHI4 侧读数（本件引用到的两条，均在原 json 内）：
```powershell
$j = Get-Content '<workspace>/03_AHI\AHI4\meeting64\产物\判据\判据_发音_v9.json' -Raw -Encoding UTF8 | ConvertFrom-Json
$j.臂.G2_皮层可分度.读数     # 准确率 0.8571 / chance 0.2 / 置乱 0.1111 / 白噪 0.0
$j.臂数; $j.通过; $j.未过      # 30 / 29 / E2_清浊电平差
```

---

*meeting65 · 01_LST技术解剖 · E1 · 2026-10-20 · 只新增本件与 `meeting65/产物/E1/LST_事实卡.json`，未改既有件*
