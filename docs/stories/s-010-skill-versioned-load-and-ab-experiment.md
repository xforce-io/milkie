---
id: s-010
title: Skill loaded at epoch boundary, A/B experiment on skill version
status: draft
kind: scenario
subsystems:
  - agent-runtime
  - agent-trace
  - evolution
capability: skill-load-and-ab
requires:
  - FSM Core
  - working context
  - Skill epoch loading
  - Evolution: Experiment Registry
owner: "@xupeng"
created: 2026-05-23
tests:
  - tests/e2e/s-010-skill-versioned-load-and-ab-experiment.e2e.test.ts
related:
  - ARCHITECTURE.md#evolution
  - docs/superpowers/specs/2026-05-16-agent-e2e-scenarios.md
---

## 场景叙事

两个子场景，围绕 skill 的版本化加载与对照实验：

**(a) Skill epoch 边界生效**：agent 起初 instructions 不含 research
skill。在 turn 2 LLM 调 `skill_request('research')` 主动请求加载；
**当前 turn 不切换**（instructions bucket 在同一 turn 内冻结，避免
invalidate cache），turn 结束时 Runtime 切 `contextEpoch` 并把
research skill 的 instructions 加入 bucket，turn 3 起 LLM 才看到 skill。

**(b) A/B 版本对比**：两个 agent 配置（`skill-tester-v1`、
`skill-tester-v2`）唯一区别是 pin 的 research skill 版本（1.0.0 vs
1.1.0）。运行同一 goal，trajectory diff 显示仅 `skills.research`
版本不同，且 v1.1.0 在输出中体现了 skill 1.1 新增的 "至少 2 个引用源"
要求（可观测行为差异）。可据此构造一个 Evolution 的 Experiment 对象。

> **说明**：本 story 同时覆盖 epoch 边界生效（机制）和 A/B 版本对比
> （评估）。两者紧密关联——前者保证 cache 不被错误 invalidate、版本
> 切换有明确边界；后者验证版本化 + 精确归因可观测。讨论后可决定是否
> 拆分成两个 story。

## 关键交互流

```
[(a) Skill epoch 边界]

skill-tester-v1 (research skill pinned to 1.0.0 in AgentConfig)
  ├─ turn 1: web_search('TypeScript 5.0')
  │            [instructions 不含 research]
  ├─ turn 2: LLM 调用 skill_request('research')
  │            [当前 turn 继续以原 instructions 执行]
  │            [Runtime: pending skill = research，本 turn 不生效]
  │  turn 2 结束 → contextEpoch: 0→1, instructions 加入 research@1.0.0
  └─ turn 3: [instructions 已含 research skill]
              web_search('TypeScript 5.0 features') → 输出风格符合 skill 引导

[(b) A/B 版本对比]

并行运行：
  milkie.invoke({ agentId: 'skill-tester-v1', goal })   // skill=1.0.0
  milkie.invoke({ agentId: 'skill-tester-v2', goal })   // skill=1.1.0

diff(trajectoryA.resolvedManifest, trajectoryB.resolvedManifest)
  → 仅 skills.research 版本不同 (1.0.0 vs 1.1.0)
  → agentVersion 不同 (1.1.0 vs 1.2.0)
v1.1.0 输出含 "来源 / source / 引用" 关键词，v1.0.0 不含
```

## 验收准则

**Skill epoch 边界：**
- [ ] turn 2 的 `llm.call.loadedSkills` 不含 `research`
- [ ] turn 3 的 `llm.call.loadedSkills` 含 `research`
- [ ] turn 3 结束时 checkpoint 的 `context.contextEpoch == 1`
- [ ] turn 3 的 `cacheBreakpoint2Hash` 与 turn 2 不同（instructions 变化）

**A/B 版本对比：**
- [ ] 两 trajectory 的 `resolvedManifest` 差异**仅在** `skills.research`
- [ ] 差异值为 `['1.0.0', '1.1.0']`
- [ ] `agentVersion` 分别为 `1.1.0` 和 `1.2.0`
- [ ] v1.2 trajectory 的最终 LLM output 含引用源相关关键词，v1.1 不含
- [ ] 可构造 `Experiment` 对象（id、goal、variants、trajectoryIds 字段齐全）

## 不在此 story 范围

- **A/B 实验的流量切分 / 显著性判定** → 未来的 Evolution 实验 story
- **Skill 卸载（unload）/ 版本回退** → 未来
- **Skill 之间的依赖解析** → 未来
- **prefix cache 命中的细粒度行为**（cache_control 标记的内部实现）→ 不属于用户场景
