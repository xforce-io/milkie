---
title: CLI surface design — agent-facing protocol facade
date: 2026-05-24
status: draft
phase: cross-cutting
subsystems:
  - agent-runtime
  - agent-trace
unblocks:
  - s-001 through s-015 (every story's CLI entry point — `agent run` underpins all others)
  - examples/ scaffold (all examples will route through CLI verbs)
  - Future UI projection (UI consumes CLI output, not a parallel API)
---

# CLI surface design — `milkie`

This spec defines the **agent-facing CLI facade** that ARCHITECTURE.md
invariants 12–13 require. It enumerates the verb surface, args, and output
contracts; implementation details live in code.

## 1. 目标与边界

**目标.** 提供一个 stable、machine-readable、composable 的命令集，作为
agent consumers（meta-agents、sub-agents、external scripts）访问 milkie
全部能力的唯一通道。同时是人类用户的入门入口（CLI → SDK → API 学习路径
的第一层）。

**契约层级.**
- 这里定义的是 **verb 名称、参数集、输出 schema** ——是 protocol surface
- 不定义具体实现（哪个 process、是否走 daemon、怎么 invoke library）
- 不定义美化渲染（人友好的 pretty print 是可选 projection）

**显式不在范围内.**
- ❌ Daemon / 长进程架构（实现策略，未来文档）
- ❌ Pretty print 细节（projection 层，可独立演进）
- ❌ Auth / 权限模型（部署 concern）
- ❌ Remote CLI / agent-to-CLI over network（先做本地）
- ❌ Agent registration 机制（registry 怎么 populate 是单独 design doc 的事；CLI 假定 registry 已存在并通过 `milkie agent list` 查询）

## 2. 设计原则（per invariant 13）

每条命令必须满足：

1. **Machine-readable by default.** 非 TTY 环境下，stdout 默认 JSONL；TTY 环境下可 pretty-print，但加 `--format json` 一定能拿到结构化输出
2. **Non-interactive.** 不读 stdin（除非显式 `--stdin` / `--input-stdin`）、不弹 prompt、不需要 tty
3. **Id-oriented.** 接收和返回的是 opaque ids（`runId` / `eventId` / `suiteId` / `contextId`），不要求 caller 解析其内部结构
4. **Bounded.** 默认有 limit（`--limit`，默认 100），不返回无限流；超出时报告 truncation
5. **Stdout = data, stderr = diagnostics.** 任何 progress、warning、debug 输出走 stderr；stdout 只放 caller 程序化消费的数据
6. **Exit code = outcome.** 0 = 成功；非 0 表示具体错误类（详见 §7 错误约定）
7. **Composable.** 输出可作为下一条命令的输入（id 形式或 JSONL 流），shell pipe 友好

## 3. 顶层结构

```
milkie <domain> <verb> [args] [flags]
```

**Domain（一级）：**

| Domain | 范围 | 状态 |
|---|---|---|
| `agent` | agent 执行入口（run / resume / interrupt / list） | 本 spec |
| `trace` | 对单个 run 的操作 | 本 spec |
| `suite` | 对 saved run 集合的操作 | 本 spec |
| `experiment` | Evolution 操作（Experiment Registry / promotion） | 留待 Evolution phase |
| `cache` | cache 视图与 GC | 内部 ops，本 spec 不展开 |

本 spec 聚焦 `agent` / `trace` / `suite`——对应 agent 执行入口 + 6-capability
surface + suite-level batch + runtime trace consumption。`agent` 在前因为它是
所有其他命令的前置：trace / suite / experiment 全部依赖一个先存在的 run。

## 4. `milkie agent` —— 执行入口

agent 命令是 CLI 的最基础动作：所有 trace / suite / experiment 操作都
依赖它产出 `runId`。按 user-facing surfaces 的学习路径，这是新用户第一
条会用的命令。

### 4.1 `milkie agent run <agentName>` — 执行 agent

| Arg / Flag | 说明 |
|---|---|
| `<agentName>` | 位置参数，已注册的 agent 名称 |
| `--input <text>` | inline 输入 |
| `--input-file <path>` | 从文件读输入（与 `--input` 互斥） |
| `--input-stdin` | 从 stdin 读输入（与上两者互斥） |
| `--context-id <id>` | 可选；不传则由 CLI 生成。后续 resume / interrupt 用这个 id |
| `--tag <tag>` | 可重复，给本 run 打标签 |
| `--timeout <duration>` | 最大执行时间（如 `30s` / `5m`） |
| `--format json\|text` | 默认 json |

**Output.** JSON 单对象：
```json
{
  "runId": "...",
  "contextId": "...",
  "status": "succeeded" | "failed" | "interrupted" | "paused",
  "lastOutput": "...",
  "error": { ... }
}
```

**注册前提.** `<agentName>` 必须是已注册的 agent。CLI 通过 `milkie agent
list` 暴露 registry 的快照；registry 怎么 populate（启动时 config 扫描 /
显式注册调用 / file-system convention）由独立 design doc 决定，不在本
spec 范围。

**Story.** 当前 15 个 stories 都隐式需要 `agent run`（产出待 inspect /
replay / fork / 加入 suite 的 run）；s-001 / s-011 / s-009 等 active
stories 已有等价 SDK 调用。

### 4.2 `milkie agent resume <contextId>` — 从 checkpoint 恢复

把一个 `paused` 状态的 agent 继续跑下去（s-008 的核心入口）。

| Arg / Flag | 说明 |
|---|---|
| `<contextId>` | 位置参数 |
| `--format json\|text` | 默认 json |

**Output.** JSON 单对象，与 `run` 同 schema。
**Story.** s-008

### 4.3 `milkie agent interrupt <contextId>` — 中断运行中的 agent

向运行中的 agent 写入 interrupt 信号；agent 在下一个 turn boundary 触
发 `paused` 转换，写出 checkpoint。本命令立即返回，**不阻塞等待 agent
实际 pause**——caller 需要的话可以再调 `agent inspect <contextId>` 或
等待 `paused` 事件出现在 trace。

| Arg / Flag | 说明 |
|---|---|
| `<contextId>` | 位置参数 |

**Output.** JSON：`{ contextId, status: "interrupt-signaled" }`。
**Story.** s-008

### 4.4 `milkie agent list` — 列已注册 agent

| Arg / Flag | 说明 |
|---|---|
| `--format json\|jsonl\|text` | 默认 jsonl |

**Output.** JSONL：每行 `{ agentName, description?, registeredAt?, metadata? }`。

## 5. `milkie trace` —— 6-capability surface

### 5.1 `milkie trace inspect <runId>` — observable

返回 run 的事件时间线。

| Arg / Flag | 说明 |
|---|---|
| `<runId>` | 位置参数 |
| `--event-type <type>` | 可重复，过滤事件类型（如 `llm.requested`） |
| `--since <ts>` / `--until <ts>` | 时间窗 |
| `--limit <n>` | 默认 100 |
| `--format json\|jsonl\|text` | 输出格式 |

**Output.** JSONL：每行一个 event（`{ id, type, ts, payload, causedBy }`）。
**Story.** s-002

### 5.2 `milkie trace explain <runId> --event <eventId>` — diagnosable

返回某个 event 发生时刻的完整推理材料。

| Arg / Flag | 说明 |
|---|---|
| `<runId>` | 位置参数 |
| `--event <eventId>` | 必填，要解释的事件 |
| `--include working-context\|prompt\|response\|capabilities` | 可重复，默认全部 |

**Output.** JSON 单对象：`{ event, workingContext, prompt, response, capabilities }`。
**Story.** s-003

### 5.3 `milkie trace lineage <runId> --artifact <artifactId>` — lineage forward

artifact → source 的正向追溯。

| Arg / Flag | 说明 |
|---|---|
| `<runId>` | 位置参数 |
| `--artifact <artifactId>` | 必填，要追溯的产出物 |
| `--depth <n>` | 追溯深度上限 |
| `--format json\|dot` | dot 输出可用 graphviz 渲染 |

**Output.** JSON：因果链节点列表 + 边。
**Story.** s-004

### 5.4 `milkie trace lineage-search --source <sourceId>` — lineage reverse

source → all dependents 的反向引用查询。

| Arg / Flag | 说明 |
|---|---|
| `--source <sourceId>` | 必填，content hash / doc 版本 / 实体 id |
| `--since <ts>` / `--until <ts>` | 时间窗 |
| `--tag <tag>` | 可重复，run 标签过滤 |
| `--limit <n>` | 默认 100 |

**Output.** JSONL：每行一个 `{ runId, eventIds, surfaceMetadata }`。
**Story.** s-014

### 5.5 `milkie trace replay <runId>` — replay

确定性 replay。

| Arg / Flag | 说明 |
|---|---|
| `<runId>` | 位置参数 |
| `--output <newRunId>` | 可选，给 replay 出的新 run 起 id |
| `--verify-byte-identical` | 严格 byte-identical 模式（Phase 4 后） |
| `--strict` | cache miss 立即报错（默认行为） |

**Output.** JSON：`{ newRunId, status: "equivalent" | "diverged", divergenceAt? }`。
**Story.** s-005

### 5.6 `milkie trace fork <runId> --at <eventId>` — fork

从指定 event 分叉。

| Arg / Flag | 说明 |
|---|---|
| `<runId>` | 位置参数 |
| `--at <eventId>` | 必填，分叉点 |
| `--override <key>=<value>` | 可重复，配置覆盖 |
| `--override-file <path>` | 替代方案：从文件读 overrides |
| `--output <newRunId>` | 可选 |

**Output.** JSON：`{ newRunId, sharedPrefixLength, divergedFromEvent }`。
**Story.** s-006

### 5.7 `milkie trace diff <runIdA> <runIdB>` — diff

两 run 结构化对比。

| Arg / Flag | 说明 |
|---|---|
| `<runIdA> <runIdB>` | 两个位置参数 |
| `--scope events\|artifacts\|outcomes` | 对比维度，默认 events |
| `--format json\|patch` | patch 输出类似 unified diff 但作用在 event 树 |

**Output.** JSON：`{ identical: bool, divergences: [...] }`。
**Stories.** s-012（批量场景）、所有需要 diff 的场景

## 6. `milkie suite` —— 批量 + 跨 run 操作

### 6.1 `milkie suite create <name> --runs <file>`

从 run id 列表定义一个 suite。

| Arg / Flag | 说明 |
|---|---|
| `<name>` | suite 名 |
| `--runs <file>` | 每行一个 runId 的文件 |
| `--description <text>` | 可选元数据 |

**Output.** JSON：`{ suiteId, name, runCount }`。

### 6.2 `milkie suite list`

列出已保存的 suite。

**Output.** JSONL：每行 `{ suiteId, name, runCount, createdAt }`。

### 6.3 `milkie suite replay <suiteId>`

批量 replay。

| Arg / Flag | 说明 |
|---|---|
| `<suiteId>` | 位置参数 |
| `--branch <ref>` | 可选，指明 replay 时的代码版本（仅记录到 metadata） |
| `--parallel <n>` | 并行度 |

**Output.** JSONL：每行 `{ originalRunId, replayedRunId, status: "equivalent" | "diverged", divergenceAt? }`。
**Story.** s-012

### 6.4 `milkie suite diff <suiteId>`

对 suite 的批量 replay 结果做聚合 diff。

| Arg / Flag | 说明 |
|---|---|
| `<suiteId>` | 位置参数 |
| `--baseline <suiteReplayId>` | 哪一次 replay 作为 baseline |
| `--against <suiteReplayId>` | 跟谁比 |

**Output.** JSONL：每行一条 diverged run 的 structural diff 摘要。
**Story.** s-012、s-013

## 7. 通用约定

### 7.1 Output formats

| Mode | 用法 | 默认情况 |
|---|---|---|
| `json` | 单对象输出 | agent run/resume/interrupt、explain、replay、fork、diff（单次结果） |
| `jsonl` | 每行一对象 | agent list、inspect、lineage-search、suite list/replay/diff（流式 / 列表） |
| `text` | 人友好的 pretty print | TTY 环境下的默认 fallback |

`--format` 可覆盖默认。非 TTY 环境强制结构化（即便用户不传 `--format`）。

### 7.2 Exit codes

| Code | 含义 |
|---|---|
| 0 | 成功 |
| 1 | Generic error（未分类） |
| 2 | Invalid args（参数校验失败） |
| 3 | Not found（runId / eventId / suiteId / contextId / agentName 不存在） |
| 4 | Replay divergence（cache miss / strict mode） |
| 5 | I/O error（存储 / 网络） |
| 6 | Agent execution failed（agent run 内部异常） |
| 7 | Timeout（agent run / suite replay 超时） |

### 7.3 Error rendering

错误一律写 stderr，格式：

```json
{ "error": { "code": "REPLAY_DIVERGENCE", "message": "...", "context": {...} } }
```

JSON-on-stderr 保证 agent consumer 也能程序化解析错误。

### 7.4 Id schema

- `runId`、`eventId`、`suiteId`、`contextId`、`agentName` 都是 opaque string，caller 不应解析
- `agentName` 由 registry 命名空间约束（人可读，但 caller 应当 treat as opaque）
- 其他 id 是 CLI/SDK 生成，对调用方完全 opaque
- 每个输出的 id 都能作为另一条命令的输入（id 是引用，不是数据）

## 8. CLI ↔ SDK 映射

每个 CLI 命令必须有等价的 SDK 调用。表对照（仅示意，实际签名以 SDK 源
码为准）：

| CLI | SDK |
|---|---|
| `milkie agent run <name> --input <text>` | `Milkie.invoke({ agentName, input, contextId? })` |
| `milkie agent resume <contextId>` | `Milkie.resume(contextId)` |
| `milkie agent interrupt <contextId>` | `Milkie.interrupt(contextId)` |
| `milkie agent list` | `Milkie.listAgents()` |
| `milkie trace inspect <runId>` | `trace.inspect(runId, opts)` |
| `milkie trace replay <runId>` | `Milkie.replay(runId, opts)` |
| `milkie trace fork <runId> --at <eventId>` | `trace.fork(runId, eventId, overrides)` |
| `milkie suite replay <suiteId>` | `suite.replay(suiteId, opts)` |

实现策略：CLI 是 SDK 的薄 wrapper（argparse + 输出格式化），所有 logic
在 SDK，CLI 不持有 state。这样 CLI / SDK / UI 永远是同一行为，不会三路
漂移。

## 9. Open questions

- **Agent registration mechanism.** `milkie agent run <name>` 假定底层
  有 agent registry，但 registry 怎么 populate（启动时 config 扫描 / 显式
  `register()` 调用 / file-system convention / plugin discovery）需要单独
  design doc 决定。CLI 的 contract **不应**假定特定 registry 机制——它
  只通过 `agent list` 查询 registry 快照。这是 baseline (A) 选定后必须紧
  跟的下一份 design。
- **`agent run` 与 in-flight inspection.** `agent run` 在 agent 还没跑完时
  是否暴露增量 output？三种选择：同步阻塞返回 final、立刻返回 runId
  让 caller 自己 `trace inspect`、返回一个 stream cursor。关联 s-015。
- **In-flight semantics.** `inspect` / `lineage` 在 run 还在跑时的语义需要
  明确：返回截至调用时刻的快照？阻塞到 run 结束？给 cursor 让 caller
  轮询？关联 s-015。
- **Sub-agent trace 拼接.** 一个 run 包含 sub-agent 时，`inspect` 默认是
  否展开 nested traces，还是只返回顶层 + sub-agent invocation events？
- **Fork override schema.** `--override key=value` 的 key 命名空间需要单
  独定义（哪些参数允许 override、override 作用到 fork 后哪些 event）。
- **Suite mutation 语义.** `suite create` 之后 suite 是否 immutable，怎么
  对一个 suite 加 / 删 run。
- **Replay side effects.** 跟 ARCHITECTURE.md 的 open question 一样，
  replay 时遇到记录的 tool I/O，是 skip / mock / reinvoke 需要 CLI flag
  暴露给用户。

## 10. 下一步

1. 在此 spec 通过后，把每个 verb 落到 `packages/cli/`（或当前等价位置）
   的脚手架；先建 contract，不必所有 verb 同时实现
2. 优先级：
   - **P0** `agent run / resume / interrupt`（无此 CLI 等于没入口）
   - **P0** `trace inspect / replay`（已 implemented 的能力直接 wrap）
   - **P1** `agent list`（依赖 registry design 完成）
   - **P1** `trace fork`（依赖 Fork primitive 实现）
   - **P2** `suite *`、`trace diff / lineage*`（依赖 Phase 5/6 实现）
3. 用 `examples/` 验证 contract：每个 example 同时给 CLI 调用和 SDK 调
   用，对比两者输出一致
4. **并行启动一份 agent registration design doc**，作为 baseline (A) 的
   必要补全。无 registry 设计，`agent run / list` 就只是壳
