# Fusion 深度调研与可落地方案

## 一、OpenRouter Fusion 到底是什么

### 1.1 核心机制：Panel + Judge，不是模型合并

OpenRouter Fusion 的官方描述非常清楚：**它不是把多个模型的输出 merge 在一起，而是让一个 Judge 模型去比较（对比、而不是合并）Panel 的输出，产出结构化分析，再由外层模型基于这个分析写出最终答案。**

官方原话：
> "The judge compares the panel responses rather than merging them: it treats what all or most models agree on as higher-confidence consensus, surfaces contradictions, preserves unique insights from individual models, and flags blind spots none of them addressed. The outer model writes the final answer from that analysis — so the result isn't a simple majority vote."

流程图：
```
request → 你的模型 → 决定调用 openrouter:fusion
  → Panel (最多8个模型, 各自带 web_search + web_fetch) 并行回答
  → Judge (比较所有panel回答, 产出结构化JSON分析)
  → 你的模型接收分析结果, 写最终答案
```

### 1.2 Judge 产出的结构化分析（5个维度）

```json
{
  "result": {
    "consensus": [
      "所有或大多数 panel 模型都同意的点 —— 高置信度"
    ],
    "contradictions": [
      {
        "claim": "矛盾的论点",
        "models": ["model_a", "model_b"],
        "resolution": "可能的调和方式或 Judge 的判断"
      }
    ],
    "partial_coverage": [
      {
        "topic": "某主题",
        "covered_by": ["model_a"],
        "missed_by": ["model_b", "model_c"]
      }
    ],
    "unique_insights": [
      {
        "insight": "只有某个模型提出的独特洞见",
        "from": "model_a"
      }
    ],
    "blind_spots": [
      "所有 panel 模型都没有涉及到的盲区"
    ],
    "panel_responses": [
      // 每个 panel 模型的原始回答（用于溯源）
    ]
  }
}
```

### 1.3 关键设计决策

| 设计点 | OpenRouter Fusion 的选择 | 理由 |
|--------|------------------------|------|
| 融合方式 | Judge 比较而非 merge | 避免简单平均化损失精度 |
| 外层模型角色 | 接收分析后自己写最终答案 | 保持外层模型的主体性和连贯性 |
| Panel 增强 | 每个 panel 模型都可以 web_search + web_fetch | 让各模型能拉取最新信息，减少信息不对称 |
| 递归保护 | 单层，不允许递归 fusion | 控制延迟和成本 |
| 调用决策 | 由外层模型自己决定是否调用 fusion | 不是每个问题都值得融合 |

---

## 二、学术界的融合方案全景对比

### 2.1 MoA (Mixture-of-Agents) — Together AI

**论文**: [arXiv 2406.04692](https://arxiv.org/abs/2406.04692)
**代码**: [togethercomputer/MoA](https://github.com/togethercomputer/MoA) (2.9k stars)
**效果**: 纯开源模型在 AlpacaEval 2.0 上达到 65.1%，超过 GPT-4 Omni 的 57.5%

**机制**：
```
Layer 1: N个"proposer"模型并行回答同一个prompt
  → Aggregator模型接收所有回答, 合成一个更好的回答
Layer 2 (可选): 上一层的输出作为新的proposer输入
  → 再次aggregate
```

核心代码就 50 行，精髓在 **aggregator 的 system prompt**:
```
"You have been provided with a set of responses from various open-source models.
Your task is to synthesize these responses into a single, high-quality response.
It is crucial to critically evaluate the information, recognizing that some may
be biased or incorrect. Your response should not simply replicate the given
answers but should offer a refined, accurate, and comprehensive reply."
```

**与 Fusion 的差异**：
- MoA 是 "合成"（synthesize）—— 让 aggregator 直接产出融合答案
- Fusion 是 "比较"（compare）—— 让 judge 产出结构化分析，外层模型自己写答案
- MoA 可以多层迭代，Fusion 限制单层
- MoA 没有结构化的 consensus/contradiction/blind_spots 分析

### 2.2 LLM-Blender — ACL 2023

**论文**: [arXiv 2306.02561](https://arxiv.org/abs/2306.02561)
**代码**: [yuchenlin/LLM-Blender](https://github.com/yuchenlin/LLM-Blender) (983 stars)

**机制（两阶段）**：
1. **PairRanker**: 对所有候选回答做 pairwise 比较，排序出最佳回答
2. **GenFuser**: 取 top-K 回答，融合生成最终输出

**与 Fusion 的差异**：
- LLM-Blender 先排序再融合，Fusion 不排序，而是分类分析
- LLM-Blender 需要训练 PairRanker（虽然也有 zero-shot 模式），Fusion 全用 prompt 工程
- LLM-Blender 的 GenFuser 是 merge 模式，Fusion 是 compare-then-write 模式

### 2.3 FrugalGPT — Stanford

**论文**: [arXiv 2304.04675](https://arxiv.org/abs/2304.04675)

**机制**：LLM cascade —— 先用便宜模型，不确定时升级到贵模型。更偏路由而非融合。

### 2.4 OptiLLM

**代码**: [algorithmicsuperintelligence/optillm](https://github.com/algorithmicsuperintelligence/optillm)

OpenAI-compatible proxy，内置多种 test-time compute 策略（best-of-n, self-consistency, MCTS 等）。更偏推理增强而非多模型融合。

---

## 三、我们应该怎么做：可行的 Fusion 方案

### 3.1 方案选择：Hybrid Fusion —— 吸取三家之长

我们不照搬任何一个方案，而是组合最优部分：

| 组件 | 来源 | 为什么选它 |
|------|------|-----------|
| Panel 并行调用 | Fusion + MoA | 基础能力，所有方案都用 |
| Judge 结构化分析 | **Fusion** | 这是 Fusion 的核心差异化优势 —— 5维度结构化分析比 MoA 的"合成"更有信息量 |
| 最终答案由外层模型写 | **Fusion** | 保持主体性，外层模型可以用自己的风格和判断 |
| 可选的多层迭代 | **MoA** | 对高价值请求允许二次/多次聚合，但默认单层 |
| Judge prompt 工程 | **自研** | 结合 Fusion 的分析维度和 MoA 的批判性思维 prompt |

### 3.2 架构设计

```
                          ┌─────────────────────┐
                          │  你的 Fusion API     │
                          │  (OpenAI-compatible) │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  请求分析层（可选）   │
                          │  判断是否值得 fusion  │
                          └──────────┬──────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
              ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
              │ Panel #1  │   │ Panel #2  │   │ Panel #N  │
              │ (你的API1)│   │ (你的API2)│   │ (你的API3)│
              └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
                    │                │                │
                    └────────────────┼────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   Judge 模型         │
                          │   产出结构化分析JSON  │
                          │   (5维度)            │
                          └──────────┬──────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
              ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
              │ 直接返回   │   │ 二次聚合   │   │ 外层模型   │
              │ 分析+答案  │   │ (可选MoA)  │   │ 重写答案   │
              └───────────┘   └───────────┘   └───────────┘
```

### 3.3 Judge 的 System Prompt（核心）

这是整个方案最关键的 prompt 工程。基于 Fusion 5维度 + MoA 的批判性思维：

```
You are a multi-model analysis judge. You will receive responses from {N}
different AI models to the same user query. Your job is NOT to merge or
synthesize them, but to produce a structured comparative analysis.

For each of the following dimensions, provide specific, actionable findings:

1. **consensus** (string[]): Points where all or most models agree.
   These should be treated as higher-confidence conclusions.
   For each point, cite which models support it.

2. **contradictions** (object[]): Points where models disagree.
   For each contradiction, specify:
   - claim: the disputed claim
   - models_supporting: which models take which position
   - judge_assessment: your assessment of which position is better supported,
     and why

3. **partial_coverage** (object[]): Important topics that some models covered
   but others missed.
   For each topic, specify:
   - topic: the topic
   - covered_by: which models addressed it
   - missed_by: which models didn't
   - what_was_missed: what important aspect the missing models overlooked

4. **unique_insights** (object[]): Valuable points raised by only one model.
   For each insight, specify:
   - insight: the unique point
   - from: which model
   - why_valuable: why this adds something the others didn't

5. **blind_spots** (string[]): Important aspects of the query that NO model
   addressed. These are gaps the final answer should fill.

CRITICAL RULES:
- Be specific, not generic. Cite actual claims and model names.
- Don't summarize the responses — analyze them.
- Flag when a model appears to hallucinate or give factually wrong info.
- If all models essentially agree, say so — don't fabricate contradictions.
- Your analysis will be used by another model to write the final answer,
  so focus on actionable guidance.
```

### 3.4 外层模型的 Final Answer Prompt

```
You are a helpful AI assistant. Below is a structured analysis comparing
responses from multiple models to the user's query, followed by the raw
responses for reference.

## User Query
{user_prompt}

## Structured Analysis
{judge_analysis_json}

## Raw Panel Responses
{panel_responses}

Using this analysis, write a comprehensive, accurate final answer.
Guidelines:
- Consensus points should form the backbone of your answer
- Address contradictions explicitly — explain which position is better supported
- Incorporate unique insights that add value
- Cover blind spots that were missed
- Do NOT simply repeat any single model's response
- If the analysis reveals errors or hallucinations, correct them
```

### 3.5 成本控制策略

| 策略 | 说明 |
|------|------|
| **按需 fusion** | 不是所有请求都用 fusion。默认让外层模型判断（类似 OpenRouter），或者根据 prompt 复杂度做规则判断 |
| **Panel 大小可配** | 2-8 个模型可配，默认 3 个。对简单问题用 2 个即可 |
| **轻量 Judge** | Judge 不需要是最强的模型。用一个中等模型（如 GPT-4o-mini 级别）做 judge 就够了，因为它只做比较分析，不做内容生成 |
| **Panel 用便宜模型** | Panel 里的模型可以是你的低成本 API，Judge 用一个稍强的。这样总成本远低于所有请求都走最强模型 |
| **缓存** | 相同/相似 prompt 的 fusion 结果可以缓存 |
| **超时/降级** | 如果某个 panel 模型超时，用剩余模型继续；如果所有 panel 都超时，回退到单模型直接回答 |

### 3.6 实现路线图

**Phase 1: Core Fusion（1-2天）**
- [ ] 实现 Panel 并行调用（asyncio）
- [ ] 实现 Judge 结构化分析
- [ ] 实现外层模型最终答案生成
- [ ] OpenAI-compatible API 接口
- [ ] 基本错误处理（超时、API 故障）

**Phase 2: 增强（1-2天）**
- [ ] 可选的二次 MoA 聚合层
- [ ] 智能 fusion 触发判断（规则 or 外层模型决策）
- [ ] Panel 配置（模型列表、数量）
- [ ] 流式输出支持

**Phase 3: 生产就绪（1-2天）**
- [ ] 成本追踪和报告
- [ ] 结果缓存
- [ ] 多 provider 支持（你的不同 API）
- [ ] 监控和降级策略

---

## 四、与竞品的差异化优势

| 维度 | OpenRouter Fusion | MoA | 我们的方案 |
|------|------------------|-----|-----------|
| 分析深度 | 5维度结构化分析 | 无结构化分析，直接合成 | **5维度 + 可选多层** |
| 最终答案质量 | 取决于外层模型 | 取决于 aggregator | **外层模型有完整分析上下文** |
| 可解释性 | 高（可追溯每个结论来源） | 低（黑盒合成） | **高** |
| 多 provider 支持 | 仅 OpenRouter | 仅 Together | **你的任意 API** |
| 成本可见 | OpenRouter 定价 | Together 定价 | **你完全掌控** |
| 自定义能力 | 有限 | 有限 | **完全可控** |

---

## 五、总结

**Fusion 的本质不是"多个模型投票出结果"，而是"让一个模型去比较其他模型的回答，产出结构化差异分析，再让主模型基于这个分析写出更全面的答案"。**

这个设计比 MoA 的"合成"更聪明的地方在于：
1. 保留了每个模型的独立视角，而不是模糊地融合
2. 矛盾被显式标记，而不是被抹平
3. 盲区被显式指出，而不是被忽略
4. 外层模型不只是复读机，而是有分析上下文的决策者

我们的方案在这个基础上加入 MoA 的可选多层迭代能力，形成一个既深思熟虑又可渐进增强的融合引擎。
