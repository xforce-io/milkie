# milkie — E2E 验证场景

Date: 2026-05-16  
Status: Draft  
Related: [agent-system-design.md](./2026-05-16-agent-system-design.md)

---

## 概述

本文档定义 milkie Agent 框架的端到端验证场景。与单元测试互补：

| | 单元测试 | E2E 场景 |
|--|---------|---------|
| 验证对象 | 单个组件的接口契约 | 跨组件的集成语义和数据流 |
| 依赖方式 | mock 外部依赖 | 真实 LLM、真实存储、真实协议 |
| 运行频率 | 每次 commit | PR merge / 夜间 |
| 断言粒度 | 返回值、调用次数 | 状态变化、Trajectory 内容、文件输出 |

**真实执行原则**：不 mock LLM（使用低成本模型 claude-haiku）、不 mock 存储（SQLite / Redis）。工具层可用 fixture 控制输入，但执行路径真实。

---

## 测试环境

### LLM

```yaml
provider: anthropic
model: claude-haiku-4-5-20251001
adapter: anthropic
```

使用 haiku 控制成本。Goal 和 fixture 工具输出设计为确定性强的场景，减少 LLM 随机性对断言的影响。对输出内容的断言只做结构性检查（包含关键词、JSON 可解析），不做全文匹配。

### 存储

| 场景 | State Store | Trajectory Store |
|------|------------|-----------------|
| Case 1 | MemoryStore（默认） | JSONLRecorder（写 `./test-output/trajectories/`） |
| Case 2 | MemoryStore（默认） | JSONLRecorder |
| Case 3 | SQLiteStore（`./test-output/state.db`） | JSONLRecorder |
| Case 4 | RedisStore（`localhost:6379/db=15`，测试专用 db） | JSONLRecorder |
| Case 5 | MemoryStore | JSONLRecorder |
| Case 6 | RedisStore（`localhost:6379/db=15`，测试专用 db） | JSONLRecorder |

每个 case 在 `beforeEach` 清空对应存储；`afterAll` 保留 JSONL 文件供事后分析。

---

## Case 1：Plan-and-Act — 竞品分析报告生成

### 目标

验证：react FSM、Cognitive toolbox（create_plan / update_step，intra-agent stateless handler / stateful workingMemory）、intra-agent 并行（单 Agent 单次响应多 tool_use）、toolbox 直接调用、Trajectory。

**模式说明**：Plan-and-Act 是单 Agent 内部的推理结构，不涉及 sub-agent。LLM 在同一个 Agent 内先用 `create_plan` 制定计划（写入 workingMemory），再直接调 toolbox tools 执行，每完成一步调 `update_step` 标记进度（"做一个划掉一个"）。并行由 LLM 在**单次响应中输出多个 tool_use block** 触发，Runtime 检测到多个 → allSettled 并发执行。cognitive toolbox handler 是纯函数（stateless），状态全部在 `ctx.workingMemory`（Context Layer 管理，随 checkpoint 持久化）。

### Agent 配置

```yaml
# analyst/agent.md frontmatter
version: "1.0.0"
fsm:
  states:
    - name: react
      type: llm
      max_iterations: 8
toolboxes:
  cognitive:  "1.0.0"   # create_plan / update_step / think → 读写 workingMemory
  search:     "1.0.0"   # web_search(query) → fixture 返回预置内容
  filesystem: "1.0.0"   # write_file(path, content) → 写真实文件
model:
  provider: anthropic
  model: claude-haiku-4-5-20251001
  adapter: anthropic
```

```
# analyst system prompt
你是一个竞品分析师，使用 cognitive toolbox 管理分析进度，用 web_search 和 write_file 完成报告。
步骤：
1. 调用 create_plan 列出所有步骤（搜索 A、搜索 B、搜索 C、写报告），计划写入 workingMemory
2. 在一次响应中同时调用 web_search 三次（分别搜索 Product A、B、C），框架并发执行
3. 三次搜索全部返回后，调用 update_step 将搜索步骤标记为 done，再调用 write_file 写入报告
4. 报告写完后调用 update_step 将报告步骤标记为 done
```

### 工具 Fixture

```typescript
// test/fixtures/search.ts — web_search 按 query 关键词返回预置内容
export const searchFixtures: Record<string, string> = {
  'Product A': '## Product A\n核心功能：实时协作编辑，定价 $20/mo，支持 API...',
  'Product B': '## Product B\n核心功能：离线优先，定价 $15/mo，支持插件...',
  'Product C': '## Product C\n核心功能：AI 辅助写作，定价 $25/mo，支持团队空间...',
}
```

`write_file` 写入真实文件系统 `./test-output/case1/`，供断言读取。

### 执行流

```
测试进程
  └─ milkie.invoke({
       agentId: 'analyst',
       goal: '分析 Product A/B/C 的核心功能差异',
       input: '输出 Markdown 报告到 ./test-output/case1/report.md'
     })

Analyst（react FSM，单 Agent）
  │
  ├─ [LLM 第 1 次响应] — plan 阶段 ─────────────────────────────
  │    输出：1 个 tool_use block
  │      └─ create_plan({ steps: ['搜索 Product A/B/C', '对比分析', '写报告'] })
  │    Runtime：执行 create_plan（intra-agent tool）
  │      → workingMemory.plan = {
  │          steps: [
  │            { id: 0, desc: '搜索 Product A/B/C', status: 'pending' },
  │            { id: 1, desc: '对比分析',            status: 'pending' },
  │            { id: 2, desc: '写报告',              status: 'pending' },
  │          ]
  │        }
  │    handler 本身无状态，状态全在 workingMemory（Context Layer 管理）
  │    FSM：进入下一 turn，LLM 下轮可读到当前计划状态
  │
  ├─ [LLM 第 2 次响应] — act 阶段，intra-agent 并行 ────────────
  │    输出：3 个 tool_use block（同一 LLM 响应内）
  │      ├─ web_search({ query: 'Product A features pricing' })  ┐
  │      ├─ web_search({ query: 'Product B features pricing' })  ├─ allSettled 并发
  │      └─ web_search({ query: 'Product C features pricing' })  ┘
  │    Runtime：3 个 tool 同进程内并发执行，全部 settled 后合并 observation
  │    （无 sub-agent，无跨进程通信，无独立 FSM）
  │    FSM：进入下一 turn
  │
  ├─ [LLM 第 3 次响应] — 更新进度 + 写报告 ─────────────────────
  │    输出：2 个 tool_use block
  │      ├─ update_step({ stepId: 0, status: 'done' })   ← 搜索步骤完成，划掉
  │      └─ write_file({
  │           path: './test-output/case1/report.md',
  │           content: '# 竞品分析\n## Product A\n...'
  │         })
  │    Runtime：update_step 更新 workingMemory.plan.steps[0].status = 'done'
  │             write_file 写入真实文件系统
  │    FSM：进入下一 turn
  │
  └─ [LLM 第 4 次响应] — 完成 ────────────────────────────────
       输出：1 个 tool_use block + text
         └─ update_step({ stepId: 2, status: 'done' })   ← 报告步骤完成，划掉
       text：'分析报告已生成：./test-output/case1/report.md'
       react FSM 终止条件满足
```

### 断言

```typescript
describe('Case 1: Plan-and-Act — 竞品分析', () => {
  let result: AgentInvokeResponse
  let trajectory: Trajectory

  beforeAll(async () => {
    result = await milkie.invoke({
      agentId: 'analyst',
      goal: '分析 Product A/B/C 的核心功能差异',
      input: '输出 Markdown 报告到 ./test-output/case1/report.md',
    })
    trajectory = await trajectoryStore.getByRunId(result.agentRunId)
  })

  it('报告文件包含三个产品内容', () => {
    const report = fs.readFileSync('./test-output/case1/report.md', 'utf8')
    expect(report).toContain('Product A')
    expect(report).toContain('Product B')
    expect(report).toContain('Product C')
    expect(report.length).toBeGreaterThan(200)
  })

  it('create_plan 在 web_search 之前被调用（plan 先于 act）', () => {
    const spans = trajectory.spans.filter(s => s.name === 'tool.call')
    const planIdx = spans.findIndex(s => s.attributes.toolName === 'create_plan')
    const firstSearchIdx = spans.findIndex(s => s.attributes.toolName === 'web_search')
    expect(planIdx).toBeGreaterThanOrEqual(0)
    expect(planIdx).toBeLessThan(firstSearchIdx)
  })

  it('create_plan 写入 workingMemory，初始所有步骤为 pending', () => {
    const planSpan = trajectory.spans.find(
      s => s.name === 'tool.call' && s.attributes.toolName === 'create_plan'
    )
    expect(planSpan).toBeDefined()
    const plan = planSpan!.attributes.output as any
    expect(plan.steps).toHaveLength(3)
    expect(plan.steps.every((s: any) => s.status === 'pending')).toBe(true)
  })

  it('update_step 将搜索步骤和报告步骤标记为 done（checklist 模式）', () => {
    const updateSpans = trajectory.spans.filter(
      s => s.name === 'tool.call' && s.attributes.toolName === 'update_step'
    )
    expect(updateSpans.length).toBeGreaterThanOrEqual(2)
    const doneUpdates = updateSpans.filter(s => (s.attributes.input as any).status === 'done')
    expect(doneUpdates.length).toBeGreaterThanOrEqual(2)
  })

  it('3 个 web_search 在同一 turn 内并发执行（intra-agent 并行）', () => {
    const searchSpans = trajectory.spans.filter(
      s => s.name === 'tool.call' && s.attributes.toolName === 'web_search'
    )
    expect(searchSpans).toHaveLength(3)
    // 同一 turn 内并发：3 个 span 的 turn 编号相同
    const turns = searchSpans.map(s => s.attributes.turn as number)
    expect(new Set(turns).size).toBe(1)
    // 时间上有重叠（并发）：最晚开始的 < 最早结束的
    const starts = searchSpans.map(s => s.startTime)
    const ends = searchSpans.map(s => s.startTime + (s.duration ?? 0))
    expect(Math.max(...starts)).toBeLessThan(Math.min(...ends))
  })

  it('无 agent.spawn span（单 Agent，无 sub-agent）', () => {
    const spawnSpans = trajectory.spans.filter(s => s.name === 'agent.spawn')
    expect(spawnSpans).toHaveLength(0)
  })

  it('llm.call span 记录 provider 和 model', () => {
    const llmSpans = trajectory.spans.filter(s => s.name === 'llm.call')
    expect(llmSpans.length).toBeGreaterThan(0)
    for (const span of llmSpans) {
      expect(span.attributes.provider).toBe('anthropic')
      expect(span.attributes.model).toContain('haiku')
    }
  })
})
```

---

## Case 2：Inter-Agent 并行 — 多角色代码审查

### 目标

验证：Named sub-agent tools、inter-agent 并行（allSettled join）、Context 隔离（各 sub-agent 独立 FSM + Context）、TaskResult、ResolvedManifest.subAgents、agent.spawn span。

**模式说明**：与 Case 1 的 intra-agent 并行不同，这里 orchestrator 调用的每个 reviewer 都是**独立的 Agent 实例**，拥有独立的 FSM 和 Context，可在独立进程中运行。适合子任务本身需要多步推理的场景。

### Agent 配置

```yaml
# review-orchestrator/agent.md frontmatter
version: "1.0.0"
fsm:
  states:
    - name: react
      type: llm
      max_iterations: 4
sub_agents:
  security-reviewer: "1.0.0"
  perf-reviewer:     "1.0.0"
  style-checker:     "1.0.0"
model:
  provider: anthropic
  model: claude-haiku-4-5-20251001
  adapter: anthropic
```

```
# review-orchestrator system prompt
你是代码审查协调员。对提交的代码同时启动三个专项审查：
在一次响应中同时调用 security-reviewer、perf-reviewer、style-checker，
收到全部结果后汇总为最终审查报告。
```

```yaml
# security-reviewer/agent.md frontmatter
version: "1.0.0"
fsm:
  states:
    - name: react
      type: llm
      max_iterations: 5
toolboxes:
  filesystem: "1.0.0"   # read_file(path) → 读源文件
```

```yaml
# perf-reviewer/agent.md（同结构，system prompt 侧重性能）
# style-checker/agent.md（同结构，system prompt 侧重代码风格）
```

### 工具 Fixture

三个 reviewer 通过 `read_file` 读取真实 fixture 文件：

```typescript
// test/fixtures/code/target.ts — 含已知问题的示例代码
// 安全问题：SQL 拼接；性能问题：N+1 查询；风格问题：命名不一致
```

### 执行流

```
测试进程
  └─ milkie.invoke({
       agentId: 'review-orchestrator',
       goal: '审查 ./test/fixtures/code/target.ts',
       input: '请并行启动三个专项审查'
     })

Review-Orchestrator（react FSM）
  │
  ├─ [LLM 第 1 次响应] ──────────────────────────────────────────
  │    输出：3 个 tool_use block（inter-agent 并行）
  │      ├─ security-reviewer({ file: './test/fixtures/code/target.ts' })  ┐
  │      ├─ perf-reviewer({ file: './test/fixtures/code/target.ts' })      ├─ allSettled
  │      └─ style-checker({ file: './test/fixtures/code/target.ts' })      ┘
  │
  │    Framework 拦截（named sub-agent tools）：
  │      每个调用 → AgentInvokeRequest → 独立 Agent 实例
  │
  │    Security-Reviewer          Perf-Reviewer           Style-Checker
  │    [独立 FSM + Context]       [独立 FSM + Context]    [独立 FSM + Context]
  │    └─ read_file(target.ts)    └─ read_file(target.ts) └─ read_file(target.ts)
  │    └─ 分析安全问题             └─ 分析性能问题          └─ 分析风格问题
  │    └─ TaskResult.success      └─ TaskResult.success   └─ TaskResult.success
  │
  │    allSettled join → 3 条 TaskResult 合并为 observations
  │
  ├─ [LLM 第 2 次响应] ──────────────────────────────────────────
  │    输出：text（汇总三份审查意见）→ react FSM 终止
  │    "安全：发现 SQL 注入风险；性能：N+1 查询；风格：命名规范不一致"
```

### 断言

```typescript
describe('Case 2: Inter-Agent 并行 — 多角色代码审查', () => {
  let result: AgentInvokeResponse
  let trajectory: Trajectory

  beforeAll(async () => {
    result = await milkie.invoke({
      agentId: 'review-orchestrator',
      goal: '审查 ./test/fixtures/code/target.ts',
      input: '请并行启动三个专项审查',
    })
    trajectory = await trajectoryStore.getByRunId(result.agentRunId)
  })

  it('生成 3 个 agent.spawn span（inter-agent 并行）', () => {
    const spawnSpans = trajectory.spans.filter(s => s.name === 'agent.spawn')
    expect(spawnSpans).toHaveLength(3)
    const childAgents = spawnSpans.map(s => s.attributes.childAgentId as string)
    expect(childAgents).toContain('security-reviewer')
    expect(childAgents).toContain('perf-reviewer')
    expect(childAgents).toContain('style-checker')
  })

  it('3 个 sub-agent 的 traceId 各自独立（Context 隔离）', () => {
    const spawnSpans = trajectory.spans.filter(s => s.name === 'agent.spawn')
    const childTraceIds = spawnSpans.map(s => s.attributes.childTraceId as string)
    expect(new Set(childTraceIds).size).toBe(3)
  })

  it('3 个 sub-agent 在同一 turn 内并发启动', () => {
    const spawnSpans = trajectory.spans.filter(s => s.name === 'agent.spawn')
    const turns = spawnSpans.map(s => s.attributes.turn as number)
    expect(new Set(turns).size).toBe(1)   // 同一 turn
  })

  it('所有 TaskResult 均为 success', () => {
    const spawnSpans = trajectory.spans.filter(s => s.name === 'agent.spawn')
    for (const span of spawnSpans) {
      expect(span.attributes.resultStatus).toBe('success')
    }
  })

  it('orchestrator output 包含三类审查发现', () => {
    expect(result.output).toMatch(/SQL|注入|安全/i)
    expect(result.output).toMatch(/N\+1|性能|查询/i)
    expect(result.output).toMatch(/命名|风格|规范/i)
  })

  it('ResolvedManifest 记录三个 sub-agent 版本', () => {
    const { resolvedManifest } = trajectory
    expect(resolvedManifest.subAgents['security-reviewer'].version).toBe('1.0.0')
    expect(resolvedManifest.subAgents['perf-reviewer'].version).toBe('1.0.0')
    expect(resolvedManifest.subAgents['style-checker'].version).toBe('1.0.0')
  })
})
```

---

## Case 3：长任务中断与恢复

### 目标

验证：Interrupt / yield point、Checkpoint 写入（SQLite）、Resume 语义、Tool 幂等键、Supervisor Tree 传播、TaskResult.interrupted、pendingEvents 保存与恢复、Trajectory append-only。

### Agent 配置

```yaml
# analyst/agent.md frontmatter
version: "1.0.0"
fsm:
  states:
    - name: react
      type: llm
      max_iterations: 20
toolboxes:
  data-processor: "1.0.0"    # 提供 process_chunk(chunkId) tool，parallelSafe: false
state_store: sqlite
model:
  provider: anthropic
  model: claude-haiku-4-5-20251001
  adapter: anthropic
```

```
# analyst system prompt
你是一个数据分析师。将数据集分成 10 个 chunk，依次调用 process_chunk 处理每个 chunk，
最后汇总结果。chunk id 从 1 到 10。
```

### 工具 Fixture

`process_chunk(chunkId)` 工具真实执行（写 SQLite 记录），每次调用耗时约 200ms，返回 `{ chunkId, result: 'processed' }`。chunkId 作为幂等键存储，重复调用直接返回已有结果。

```typescript
// data-processor toolbox 实现
async function process_chunk({ chunkId }: { chunkId: number }): Promise<ChunkResult> {
  const existing = await db.get('SELECT * FROM chunks WHERE chunk_id = ?', chunkId)
  if (existing) return { chunkId, result: existing.result, fromCache: true }

  await sleep(200)
  const result = `processed-${chunkId}`
  await db.run('INSERT INTO chunks VALUES (?, ?)', chunkId, result)
  return { chunkId, result, fromCache: false }
}
```

### 执行流

#### 主流：单 Agent 中断恢复

```
测试进程
  └─ 启动 analyst（goal: '处理 dataset-42 的 10 个 chunk'）

Analyst（react FSM）
  ├─ turn 1: 制定计划，决定逐 chunk 处理
  ├─ turn 2: process_chunk(1) → ok
  ├─ turn 3: process_chunk(2) → ok
  ├─ turn 4: process_chunk(3) → ok
  │            ↑ tool 执行完成后 yield point
  │
测试进程 ──→ eventQueue.push({ type: 'interrupt' })
  │
  ├─ yield point 检测到 interrupt
  ├─ saveCheckpoint()                    → SQLite checkpoint:latest
  ├─ FSM → paused
  └─ throw InterruptSignal

测试进程：验证 checkpoint，然后 resume
  └─ milkie.resume(checkpointId)

Analyst（从 checkpoint 恢复）
  ├─ turn 5: process_chunk(4) → ok（fromCache: false）
  ├─ ...
  ├─ turn 11: process_chunk(10) → ok
  └─ 输出汇总结果
```

#### 子流：Supervisor Tree 中断传播

```
测试进程
  └─ 启动 orchestrator（含两个并发 sub-agent: worker-a, worker-b）

Orchestrator
  ├─ worker-a(task='A') ─┐ allSettled join（均在执行中）
  └─ worker-b(task='B') ─┘

测试进程 ──→ orchestrator.eventQueue.push({ type: 'interrupt' })

Orchestrator（yield point）
  ├─ 将 interrupt 写入 worker-a.eventQueue
  ├─ 将 interrupt 写入 worker-b.eventQueue
  └─ 等待 allSettled

Worker-a（下一个 yield point）          Worker-b（下一个 yield point）
  ├─ saveCheckpoint()                     ├─ saveCheckpoint()
  └─ TaskResult: { status:'interrupted',  └─ TaskResult: { status:'interrupted',
       checkpointId: 'wk-a-cp-1' }              checkpointId: 'wk-b-cp-1' }

Orchestrator（收到 allSettled 结果）
  ├─ saveCheckpoint()   ← children 记录 wk-a 和 wk-b 的 checkpointId
  └─ FSM → paused
```

### 断言

```typescript
describe('Case 3: 长任务中断与恢复', () => {
  describe('主流：单 Agent 中断恢复', () => {
    let checkpointId: string
    let agentRunId: string

    it('中断后 FSM 状态为 paused，sequence = 3', async () => {
      // 启动 analyst（后台执行）
      const runPromise = milkie.invoke({
        agentId: 'analyst',
        goal: '处理 dataset-42 的 10 个 chunk',
        input: 'chunk 数量：10',
      })

      // 等待第 3 个 process_chunk 完成（通过 tool 调用计数信号）
      await waitForToolCalls('process_chunk', 3)

      // 注入中断
      await milkie.interrupt(analystInstanceId)

      // 等待 run 完成（以 InterruptSignal 结束）
      await expect(runPromise).rejects.toMatchObject({ type: 'InterruptSignal' })

      const cp = await stateStore.get(`agent:${analystId}:checkpoint:latest`)
      checkpointId = cp.checkpointId
      agentRunId = cp.meta.agentRunId

      expect(cp.fsm.currentState).toBe('paused')
      expect(cp.sequence).toBe(3)
      expect(cp.goal).toBe('处理 dataset-42 的 10 个 chunk')
      expect(cp.currentTurn).toBeDefined()
    })

    it('Checkpoint 包含 pendingEvents（空，中断前无积压事件）', async () => {
      const cp = await stateStore.get(`agent:${analystId}:checkpoint:latest`)
      expect(cp.pendingEvents).toEqual([])
    })

    it('Resume 后 tool 总调用 10 次，无重复 toolCallId', async () => {
      const resumeResult = await milkie.resume(checkpointId)
      const trajectory = await trajectoryStore.getByRunId(agentRunId)

      const toolSpans = trajectory.spans.filter(
        s => s.name === 'tool.call' && s.attributes.toolName === 'process_chunk'
      )
      expect(toolSpans).toHaveLength(10)

      const toolCallIds = toolSpans.map(s => s.attributes.toolCallId as string)
      expect(new Set(toolCallIds).size).toBe(10)   // 无重复
    })

    it('Resume 后不重复执行 chunk 1-3（checkpoint 恢复，LLM 从历史感知进度，不重放）', async () => {
      // chunk 1-3 在中断前首次真实执行，from_cache = false
      const preInterruptChunks = await db.all(
        'SELECT * FROM chunks WHERE chunk_id <= 3 ORDER BY chunk_id'
      )
      expect(preInterruptChunks).toHaveLength(3)
      expect(preInterruptChunks.every((r: any) => r.from_cache === 0)).toBe(true)

      // trajectory 中每个 chunkId 只出现一次（resume 后不重调 1-3）
      const trajectory = await trajectoryStore.getByRunId(agentRunId)
      const chunkSpans = trajectory.spans.filter(
        s => s.name === 'tool.call' && s.attributes.toolName === 'process_chunk'
      )
      const calledIds = chunkSpans.map(s => (s.attributes.input as any).chunkId as number)
      expect(new Set(calledIds).size).toBe(calledIds.length)                          // 无重复
      expect(calledIds.sort((a, b) => a - b)).toEqual([1,2,3,4,5,6,7,8,9,10])       // 共 10 次
    })

    it('最终 Trajectory status = completed，跨中断连续', async () => {
      const trajectory = await trajectoryStore.getByRunId(agentRunId)
      expect(trajectory.status).toBe('completed')
      // Trajectory 中断前后的 span 连续，无 gap
      const spans = trajectory.spans.sort((a, b) => a.startTime - b.startTime)
      expect(spans[0].name).toBe('agent.run')
      expect(spans[spans.length - 1].name).toBe('agent.run')  // root span 最后关闭
    })
  })

  describe('子流：Supervisor Tree 中断传播', () => {
    it('父 Agent 中断后，children checkpoint 记录两个子 Agent 的 checkpointId', async () => {
      // 启动含两个 sub-agent 的 orchestrator
      const runPromise = milkie.invoke({ agentId: 'orchestrator', goal: '...', input: '...' })

      await waitForSubAgentsRunning(2)
      await milkie.interrupt(orchestratorInstanceId)
      await expect(runPromise).rejects.toMatchObject({ type: 'InterruptSignal' })

      const parentCp = await stateStore.get(`agent:${orchestratorId}:checkpoint:latest`)
      expect(parentCp.children).toHaveLength(2)
      expect(parentCp.children.every(c => c.status === 'interrupted')).toBe(true)
      expect(parentCp.children.every(c => c.checkpointId != null)).toBe(true)
    })
  })
})
```

---

## Case 4：多轮对话与错误恢复

### 目标

验证：多轮对话（type: llm，无 on.DONE → 等待用户）、contextId 复用（history 跨 invoke 保留）、Goal 不变性 vs current_turn 多轮变化、State Store（Redis）、Error handling FSM 转移（tool retryable 错误自动重试）。

### Agent 配置

```yaml
# order-analyst/agent.md frontmatter
version: "1.0.0"
fsm:
  states:
    - name: analyze
      type: llm
      # 无 on.DONE → LLM 输出后等待下一条用户消息（多轮对话）
toolboxes:
  database: "1.0.0"     # 提供 query_orders(orderId) tool
state_store: redis
model:
  provider: anthropic
  model: claude-haiku-4-5-20251001
  adapter: anthropic
```

```
# order-analyst system prompt
你是一个订单异常分析师。使用 query_orders 查询订单详情，结合提供的信息给出分析判断。
```

### 工具 Fixture

```typescript
// database toolbox 实现
let callCount = 0

async function query_orders({ orderId }: { orderId: string }): Promise<OrderRecord> {
  callCount++
  // 第 1 次调用模拟超时（触发 error_handling 重试）
  if (callCount === 1) {
    await sleep(5000)
    throw Object.assign(new Error('Connection timeout'), { retryable: true })
  }
  return {
    orderId,
    amount: 15000,
    threshold: 5000,
    customerHistory: 'new_customer',
    flagged: true,
  }
}
```

### 执行流

```
[第 1 次 invoke]
milkie.invoke({ agentId: 'order-analyst', goal: '分析订单 #12345', input: '金额超阈值 3 倍' })

  → analyze state
  → LLM: calls query_orders('12345') → 超时，retryable: true
  → error_handling: 自动重试 query_orders → 成功
  → LLM: 输出初步分析
  → DONE → 无 on.DONE → 返回 run1，等待下一条消息

[第 2 次 invoke，同 contextId，同 goal]
milkie.invoke({ ..., input: '客户历史消费记录显示为正常季节性采购', contextId: run1.contextId })

  → analyze state（history 含第 1 轮对话）
  → LLM: 综合历史 + 新 input，输出最终判断
```

### 断言

```typescript
describe('Case 4: 多轮对话与错误恢复', () => {
  let run1: AgentInvokeResponse
  let run2: AgentInvokeResponse
  let trajectory: Trajectory
  let run1Cp: AgentCheckpoint
  let run2Cp: AgentCheckpoint

  beforeAll(async () => {
    run1 = await milkie.invoke({
      agentId: 'order-analyst',
      goal: '分析订单 #12345 的异常原因',
      input: '订单金额超出阈值 3 倍',
    })
    run1Cp = await stateStore.get(`agent:order-analyst:checkpoint:latest`) as AgentCheckpoint

    run2 = await milkie.invoke({
      agentId: 'order-analyst',
      goal: '分析订单 #12345 的异常原因',
      input: '客户历史消费记录显示为正常季节性采购',
      contextId: run1.contextId,
    })
    run2Cp = await stateStore.get(`agent:order-analyst:checkpoint:latest`) as AgentCheckpoint
    trajectory = await trajectoryStore.getByAgentId('order-analyst')
  })

  afterAll(async () => {
    await redis.flushdb()
  })

  it('goal 在两次 invoke 中保持不变', () => {
    expect(run1Cp.goal).toBe('分析订单 #12345 的异常原因')
    expect(run1Cp.goal).toBe(run2Cp.goal)
  })

  it('两次 invoke 使用同一 contextId', () => {
    expect(run1.contextId).toBe(run2.contextId)
  })

  it('第 2 次 invoke 的 context history 包含第 1 轮对话', () => {
    const hasRound1 = run2Cp.context.history.some(
      m => JSON.stringify(m).includes('query_orders')
    )
    expect(hasRound1).toBe(true)
  })

  it('error_handling FSM 转移存在，且成功恢复', () => {
    const fsmSpans = trajectory.spans.filter(s => s.name === 'fsm.transition')
    const toError   = fsmSpans.find(s => s.attributes.toState   === 'error_handling')
    const fromError = fsmSpans.find(s => s.attributes.fromState === 'error_handling')
    expect(toError).toBeDefined()
    expect(fromError).toBeDefined()
  })

  it('query_orders 被调用 2 次（第 1 次超时，第 2 次成功）', () => {
    const querySpans = trajectory.spans.filter(
      s => s.name === 'tool.call' && s.attributes.toolName === 'query_orders'
    )
    expect(querySpans).toHaveLength(2)
  })

  it('run2 output 包含最终判断', () => {
    expect(run2.output).toMatch(/正常|异常|判断|结论/i)
  })
})
```

---

## Case 5：Skill 渐进加载与 A/B 版本对比

### 目标

验证：react FSM、skill_request + epoch 边界生效、instructions bucket 单 turn 内冻结、contextEpoch 递增、Trajectory 版本 diff、A/B Test / Experiment 模型、ResolvedManifest 精确归因。

### Agent 配置

```yaml
# skill-tester-v1/agent.md
version: "1.1.0"
fsm:
  states:
    - name: react
      type: llm
      max_iterations: 6
toolboxes:
  search: "1.0.0"
skills:
  research: "1.0.0"    # pin 在 config 里；skill_request 激活时加载此版本，不触发版本解析
model:
  provider: anthropic
  model: claude-haiku-4-5-20251001
  adapter: anthropic

# skill-tester-v2/agent.md
version: "1.2.0"
fsm:
  states:
    - name: react
      type: llm
      max_iterations: 6
toolboxes:
  search: "1.0.0"
skills:
  research: "1.1.0"    # 升级到 1.1.0，产生新 agentVersion，A/B 归因于此
model:
  provider: anthropic
  model: claude-haiku-4-5-20251001
  adapter: anthropic
```

**Skill 版本解析语义**：skill 版本在 `AgentConfig.skills` 中 pin，构建时锁定 sha 写入 resolvedManifest。`skill_request('research')` 只负责激活（epoch 边界生效），不触发版本解析。两个 agent 加载的是各自 config pin 的不同版本，A/B diff 精确归因于 `skills.research` 版本变化。

### Skill 定义

```yaml
# skills/research/v1.0.0/skill.yaml
name: research
version: "1.0.0"
requires_toolboxes:
  - search
instructions: |
  ## Research Guidelines (v1.0)
  Use web_search to find information. Summarize findings concisely.
  Focus on factual accuracy.

# skills/research/v1.1.0/skill.yaml
name: research
version: "1.1.0"
requires_toolboxes:
  - search
instructions: |
  ## Research Guidelines (v1.1)
  Use web_search to find information. Summarize findings with citations.
  Focus on factual accuracy and source credibility.
  Always include at least 2 sources per claim.   ← 新增要求，可观测行为差异
```

### 执行流

#### Skill epoch 验证

```
Agent（初始：无 skill）
  ├─ turn 1: web_search('TypeScript 5.0') → fixture 结果
  │           [instructions 不含 research skill]
  ├─ turn 2: skill_request('research')   ← LLM 发现需要 skill
  │           [当前 turn 继续以原 instructions 执行]
  │           [Runtime 记录 pending skill = research，本 turn 不生效]
  └─ turn 2 结束，Runtime 切换 epoch：
       contextEpoch: 0 → 1
       instructions bucket 加入 research@1.0.0 instructions
  ├─ turn 3: [instructions 已含 research skill]
  │           web_search('TypeScript 5.0 features') → 输出含 citation 格式（1.1.0）或不含（1.0.0）
  └─ 输出分析结果
```

### 断言

```typescript
describe('Case 5: Skill 渐进加载与 A/B 版本对比', () => {
  describe('Skill epoch 边界生效', () => {
    let trajectory: Trajectory

    beforeAll(async () => {
      const result = await milkie.invoke({
        agentId: 'skill-tester-v1',
        goal: '分析 TypeScript 5.0 的主要新特性',
        input: '请使用 research skill 进行深入调研',
      })
      trajectory = await trajectoryStore.getByRunId(result.agentRunId)
    })

    it('turn 2 的 llm.call span 不含 research skill', () => {
      const turn2 = trajectory.spans.find(
        s => s.name === 'llm.call' && s.attributes.turn === 2
      )
      expect(turn2?.attributes.loadedSkills).not.toContain('research')
    })

    it('turn 3 的 llm.call span 含 research skill', () => {
      const turn3 = trajectory.spans.find(
        s => s.name === 'llm.call' && s.attributes.turn === 3
      )
      expect(turn3?.attributes.loadedSkills).toContain('research')
    })

    it('contextEpoch 在 turn 3 递增为 1', async () => {
      const cpAfterTurn3 = await stateStore.getLatestForTurn(agentId, 3)
      expect(cpAfterTurn3.context.contextEpoch).toBe(1)
    })

    it('turn 2 的 cache breakpoint 2 在 epoch 切换后重建', () => {
      const turn2LlmSpan = trajectory.spans.find(
        s => s.name === 'llm.call' && s.attributes.turn === 2
      )
      const turn3LlmSpan = trajectory.spans.find(
        s => s.name === 'llm.call' && s.attributes.turn === 3
      )
      // turn 3 的 cacheBreakpoint2Hash 与 turn 2 不同（instructions 变化）
      expect(turn3LlmSpan?.attributes.cacheBreakpoint2Hash).not.toBe(
        turn2LlmSpan?.attributes.cacheBreakpoint2Hash
      )
    })
  })

  describe('A/B 版本对比', () => {
    let t1: Trajectory
    let t2: Trajectory
    const goal = '分析 TypeScript 5.0 的主要新特性'

    beforeAll(async () => {
      const [r1, r2] = await Promise.all([
        milkie.invoke({ agentId: 'skill-tester-v1', goal, input: goal }),
        milkie.invoke({ agentId: 'skill-tester-v2', goal, input: goal }),
      ])
      ;[t1, t2] = await Promise.all([
        trajectoryStore.getByRunId(r1.agentRunId),
        trajectoryStore.getByRunId(r2.agentRunId),
      ])
    })

    it('两个 trajectory 的 resolvedManifest 只有 research skill 版本不同', () => {
      const diff = diffManifests(t1.resolvedManifest, t2.resolvedManifest)
      expect(Object.keys(diff)).toEqual(['skills.research'])
      expect(diff['skills.research']).toEqual(['1.0.0', '1.1.0'])
    })

    it('agentVersion 不同（1.1.0 vs 1.2.0）', () => {
      expect(t1.resolvedManifest.agentVersion).toBe('1.1.0')
      expect(t2.resolvedManifest.agentVersion).toBe('1.2.0')
    })

    it('v1.2 输出包含 citation 相关内容（skill 1.1.0 新增要求的可观测效果）', () => {
      // v1.1.0 skill 要求 "at least 2 sources per claim"
      // 通过 Trajectory 的 llm.call output 检查（不全文匹配，只检查结构特征）
      const lastLlmSpan = [...t2.spans]
        .filter(s => s.name === 'llm.call')
        .sort((a, b) => b.startTime - a.startTime)[0]
      const output = lastLlmSpan.attributes.output as string
      // v1.1 skill 引导 LLM 输出 citation，v1.0 不要求
      expect(output.match(/来源|source|引用|reference/i)).not.toBeNull()
    })

    it('可构造 Experiment 对象用于后续分析', () => {
      const experiment: Experiment = {
        id: 'research-skill-upgrade-ts50',
        goal,
        variants: [
          { name: 'research-v1.0', agentVersion: '1.1.0', trajectoryIds: [t1.traceId] },
          { name: 'research-v1.1', agentVersion: '1.2.0', trajectoryIds: [t2.traceId] },
        ],
      }
      expect(experiment.variants).toHaveLength(2)
      expect(experiment.variants[0].trajectoryIds[0]).toBe(t1.traceId)
    })
  })
})
```

---

## Case 6：客服意图路由与多轮槽填充

### 目标

验证：多 state FSM 自定义拓扑（type: llm + type: action 组合）、意图识别硬转移（tool handler ctx.emit）、多轮槽填充（working_memory 累积，完整性检查在 handler）、用户确认流程、低置信度升级（terminal escalated state）。

### 业务场景

客服 Agent 接收用户消息，完成以下链式流程：

```
意图识别 → 分流路由 → 槽填充（多轮） → 用户确认 → 执行
                ↓ 置信度不足
            升级人工
```

设计三条测试路径：
- **Path A（主路径）**：意图清晰 → 取消订单 → 收集槽位 → 用户确认 → 执行
- **Path B（澄清路径）**：意图模糊 → Agent 追问 → 意图明确 → 路由到账单查询子 Agent
- **Path C（升级路径）**：置信度不足 → 直接升级人工，不进入槽填充

### FSM 拓扑设计

```yaml
# customer-service/agent.md frontmatter
version: "1.0.0"
fsm:
  states:
    - name: intent_classification
      type: llm
      tools: [classify_intent]
      on:
        INTENT_CANCELLATION: collecting_slots
        INTENT_BILLING:      routing_to_specialist
        INTENT_UNCLEAR:      clarifying
        ESCALATE:            escalated

    - name: clarifying
      type: llm
      tools: [classify_intent]
      on:
        INTENT_CANCELLATION: collecting_slots
        INTENT_BILLING:      routing_to_specialist
        ESCALATE:            escalated

    - name: collecting_slots
      type: llm
      tools: [collect_slot]
      on:
        SLOTS_COMPLETE:      confirming
        ESCALATE:            escalated

    - name: routing_to_specialist
      type: action
      handler: spawnBillingSpecialist
      on:
        DONE: completed

    - name: confirming
      type: llm
      tools: [confirm_action]
      on:
        USER_CONFIRMED:      executing
        USER_REJECTED:       collecting_slots

    - name: executing
      type: action
      handler: spawnCancellationExecutor
      on:
        DONE:  completed
        ERROR: escalated

    - name: escalated
      type: llm
      terminal: true
      instructions: "向用户说明已转接人工客服，给出预计等待时间，保持礼貌。"

    - name: completed
      terminal: true

sub_agents:
  cancellation-executor: "1.0.0"   # 执行取消逻辑
  billing-specialist:    "1.0.0"   # 账单查询专家
state_store: redis
model:
  provider: anthropic
  model: claude-haiku-4-5-20251001
  adapter: anthropic
```

```
# customer-service system prompt
你是一个客服助手。首先识别用户意图，然后根据意图采取对应行动：
- 取消订单：收集 orderId、reason、preferRefund，然后请用户确认
- 账单查询：转交给 billing-specialist 处理
- 意图不明确：礼貌追问，最多追问 1 次，仍不明确则升级人工

识别意图时调用 classify_intent 工具，置信度 < 0.75 时直接 ESCALATE。
收集每个槽位时调用 collect_slot 工具。所有槽位收集完后调用 present_confirmation。
用户回复确认时调用 confirm_action。
```

### 生命周期钩子

置信度检查和槽位完整性检查已移入 tool handler（确定性逻辑不依赖 LLM）。
生命周期钩子只处理状态进入时的初始化：

```typescript
// customer-service/hooks.ts

milkie.registerAction('on_enter_intent_classification', async (ctx) => {
  ctx.workingMemory.set('intentConfidence', null)
  ctx.workingMemory.set('intent', null)
})

milkie.registerAction('on_enter_collecting_slots', async (ctx) => {
  if (!ctx.workingMemory.has('collectedSlots')) {
    ctx.workingMemory.set('collectedSlots', {})
  }
})
```

### 工具定义

工具 handler 通过 `ctx.emit()` 触发 FSM 状态转移，确定性逻辑（置信度检查、完整性检查）在 handler 内完成，不依赖 LLM 判断。

```typescript
// classify_intent tool
registerTool({
  name: 'classify_intent',
  inputSchema: {
    intent:     { type: 'string', enum: ['cancellation', 'billing', 'unclear'] },
    confidence: { type: 'number' },   // LLM 自评估，0.0 ~ 1.0
  },
  handler: async ({ intent, confidence }, ctx) => {
    ctx.workingMemory.set('intent', intent)
    ctx.workingMemory.set('intentConfidence', confidence)

    if (confidence < 0.75) {
      ctx.emit('ESCALATE')
      return { accepted: false, reason: 'low_confidence' }
    }

    const eventMap: Record<string, string> = {
      cancellation: 'INTENT_CANCELLATION',
      billing:      'INTENT_BILLING',
      unclear:      'INTENT_UNCLEAR',
    }
    ctx.emit(eventMap[intent])
    return { accepted: true, intent }
  }
})

// collect_slot tool
registerTool({
  name: 'collect_slot',
  inputSchema: {
    name:  { type: 'string', enum: ['orderId', 'reason', 'preferRefund'] },
    value: { type: 'string' },
  },
  handler: async ({ name, value }, ctx) => {
    const slots = ctx.workingMemory.get('collectedSlots') as Record<string, unknown>
    slots[name] = value
    ctx.workingMemory.set('collectedSlots', slots)

    const allFilled = ['orderId', 'reason', 'preferRefund'].every(k => slots[k] !== undefined)
    if (allFilled) ctx.emit('SLOTS_COMPLETE')
    return { collected: slots, complete: allFilled }
  }
})

// confirm_action tool
registerTool({
  name: 'confirm_action',
  inputSchema: {
    confirmed:   { type: 'boolean' },
    userMessage: { type: 'string' },
  },
  handler: async ({ confirmed }, ctx) => {
    ctx.emit(confirmed ? 'USER_CONFIRMED' : 'USER_REJECTED')
  }
})
```

### Path A：主路径（意图清晰 → 槽填充 → 确认 → 执行）

```
用户              customer-service Agent              cancellation-executor
  │                        │                                  │
  │ '我想取消订单'          │                                  │
  ├────────────────────────→│                                  │
  │                 intent_classification state                │
  │                        │── classify_intent(               │
  │                        │     intent:'cancellation',       │
  │                        │     confidence:0.92              │
  │                        │   ) → emit INTENT_CANCELLATION   │
  │                 collecting_slots state                     │
  │                        │                                  │
  │ '订单号 ORD-456'        │ [Agent 追问：请提供订单号]        │
  ├────────────────────────→│                                  │
  │                        │── collect_slot(orderId,'ORD-456')│
  │                        │                                  │
  │ '收到货损坏了'          │ [Agent 追问：取消原因]            │
  ├────────────────────────→│                                  │
  │                        │── collect_slot(reason,'damaged') │
  │                        │                                  │
  │ '要退款'               │ [Agent 追问：是否退款]            │
  ├────────────────────────→│                                  │
  │                        │── collect_slot(preferRefund,true)│
  │                        │   → emit SLOTS_COMPLETE          │
  │                 confirming state                           │
  │                        │                                  │
  │ '确认'                 │ [Agent 展示确认摘要]              │
  ├────────────────────────→│                                  │
  │                        │── confirm_action(confirmed:true) │
  │                        │   → emit USER_CONFIRMED          │
  │                 executing state                            │
  │                        │── cancellation-executor(         │──→ 执行取消
  │                        │     orderId:'ORD-456',           │    写入 DB
  │                        │     reason:'damaged',            │←── TaskResult.success
  │                        │     preferRefund:true            │
  │                        │   ) → emit EXEC_DONE             │
  │                 completed state                            │
  │←────────────────────── '取消成功，退款将在 3-5 个工作日...'│
```

### Path B：澄清路径（意图模糊 → 追问 → 路由账单专家）

```
用户              customer-service Agent              billing-specialist
  │                        │                                  │
  │ '我的账有问题'          │                                  │
  ├────────────────────────→│                                  │
  │                 intent_classification                      │
  │                        │── classify_intent(               │
  │                        │     intent:'unclear',            │
  │                        │     confidence:0.55              │
  │                        │   ) → emit INTENT_UNCLEAR        │
  │                 clarifying state                           │
  │                        │                                  │
  │ '上个月扣了两次钱'      │ [Agent：请问是账单疑问还是...]   │
  ├────────────────────────→│                                  │
  │                 intent_classification (重新进入)           │
  │                        │── classify_intent(               │
  │                        │     intent:'billing',            │
  │                        │     confidence:0.91              │
  │                        │   ) → emit INTENT_BILLING        │
  │                 routing_to_specialist state                │
  │                        │── billing-specialist(            │──→ 查询账单记录
  │                        │     query:'重复扣费 ORD-xxx'     │    返回账单详情
  │                        │   ) → emit SPECIALIST_DONE       │←── TaskResult.success
  │                 completed state                            │
  │←────────────────────── '查到您上月有一笔重复扣款...'       │
```

### Path C：升级路径（置信度不足 → 直接升级）

```
用户              customer-service Agent
  │                        │
  │ '我要投诉你们！'        │
  ├────────────────────────→│
  │                 intent_classification
  │                        │── classify_intent(
  │                        │     intent:'unclear',
  │                        │     confidence:0.48
  │                        │   ) → emit ESCALATE
  │                 escalated state（type:llm, terminal）
  │                        │── LLM 生成人工客服转接说明
  │←────────────────────── '已为您转接人工客服，预计等待...'
```

### 断言

```typescript
describe('Case 6: 客服意图路由与多轮槽填充', () => {
  afterAll(async () => {
    await redis.flushdb()
  })

  describe('Path A: 主路径 — 取消订单', () => {
    let trajectory: Trajectory
    const contextId = uuid()
    const goal = '处理客户服务请求'

    async function sendTurn(input: string) {
      return milkie.invoke({ agentId: 'customer-service', goal, input, contextId })
    }

    beforeAll(async () => {
      await sendTurn('我想取消订单')
      await sendTurn('订单号 ORD-456')
      await sendTurn('收到货损坏了')
      await sendTurn('要退款')
      await sendTurn('确认')
      trajectory = await trajectoryStore.getByContextId(contextId)
    })

    it('FSM 经历完整状态序列', () => {
      const fsmSpans = trajectory.spans
        .filter(s => s.name === 'fsm.transition')
        .map(s => s.attributes.toState)
      expect(fsmSpans).toEqual([
        'collecting_slots',    // classify_intent → INTENT_CANCELLATION
        'confirming',          // collect_slot × 3 → SLOTS_COMPLETE
        'executing',           // confirm_action → USER_CONFIRMED
        'completed',           // EXEC_DONE
      ])
    })

    it('classify_intent 置信度 ≥ 0.75，走正常路径', () => {
      const classifySpan = trajectory.spans.find(
        s => s.name === 'tool.call' && s.attributes.toolName === 'classify_intent'
      )
      expect(classifySpan?.attributes.input).toMatchObject({ intent: 'cancellation' })
      expect(classifySpan?.attributes.input.confidence).toBeGreaterThanOrEqual(0.75)
    })

    it('collect_slot 被调用 3 次，槽位完整', () => {
      const slotSpans = trajectory.spans.filter(
        s => s.name === 'tool.call' && s.attributes.toolName === 'collect_slot'
      )
      expect(slotSpans).toHaveLength(3)
      const names = slotSpans.map(s => (s.attributes.input as any).name)
      expect(names).toContain('orderId')
      expect(names).toContain('reason')
      expect(names).toContain('preferRefund')
    })

    it('working_memory 在 collecting_slots 阶段累积槽位', async () => {
      // 取第 2 次 invoke 完成后的 checkpoint（收集了 orderId 后）
      const cp = await stateStore.getByContextAndTurn(contextId, 2)
      const slots = cp.context.workingMemory as any
      expect(slots.collectedSlots.orderId).toBe('ORD-456')
      expect(slots.collectedSlots.reason).toBeUndefined()  // 尚未收集
    })

    it('USER_REJECTED 时 FSM 退回 collecting_slots（确认被拒子场景）', async () => {
      // 独立子测试：第 4 轮用户说"不对，修改原因"，FSM 应退回
      const subContextId = uuid()
      await milkie.invoke({ agentId: 'customer-service', goal, input: '我想取消订单', contextId: subContextId })
      await milkie.invoke({ agentId: 'customer-service', goal, input: 'ORD-789', contextId: subContextId })
      await milkie.invoke({ agentId: 'customer-service', goal, input: '商品损坏', contextId: subContextId })
      await milkie.invoke({ agentId: 'customer-service', goal, input: '不退款', contextId: subContextId })
      await milkie.invoke({ agentId: 'customer-service', goal, input: '取消，我要修改', contextId: subContextId })  // 拒绝确认

      const subTraj = await trajectoryStore.getByContextId(subContextId)
      const fsmStates = subTraj.spans
        .filter(s => s.name === 'fsm.transition')
        .map(s => s.attributes.toState)
      expect(fsmStates).toContain('collecting_slots')  // 退回修改
      // collecting_slots 出现两次（第一次填槽 + 被拒后退回）
      expect(fsmStates.filter(s => s === 'collecting_slots').length).toBe(2)
    })

    it('cancellation-executor sub-agent 被调用，TaskResult.success', () => {
      const spawnSpan = trajectory.spans.find(s => s.name === 'agent.spawn')
      expect(spawnSpan?.attributes.childAgentId).toContain('cancellation-executor')
      expect(spawnSpan?.attributes.resultStatus).toBe('success')
    })
  })

  describe('Path B: 澄清路径 — 意图模糊后再识别', () => {
    let trajectory: Trajectory
    const contextId = uuid()
    const goal = '处理客户服务请求'

    beforeAll(async () => {
      await milkie.invoke({ agentId: 'customer-service', goal, input: '我的账有问题', contextId })
      await milkie.invoke({ agentId: 'customer-service', goal, input: '上个月扣了两次钱', contextId })
      trajectory = await trajectoryStore.getByContextId(contextId)
    })

    it('FSM 经历 clarifying → intent_classification → routing_to_specialist', () => {
      const fsmStates = trajectory.spans
        .filter(s => s.name === 'fsm.transition')
        .map(s => s.attributes.toState)
      expect(fsmStates).toContain('clarifying')
      expect(fsmStates).toContain('routing_to_specialist')
      expect(fsmStates).toContain('completed')
    })

    it('classify_intent 被调用 2 次（第 1 次 unclear，第 2 次 billing）', () => {
      const classifySpans = trajectory.spans.filter(
        s => s.name === 'tool.call' && s.attributes.toolName === 'classify_intent'
      )
      expect(classifySpans).toHaveLength(2)
      expect((classifySpans[0].attributes.input as any).intent).toBe('unclear')
      expect((classifySpans[1].attributes.input as any).intent).toBe('billing')
    })

    it('billing-specialist sub-agent 被调用', () => {
      const spawnSpan = trajectory.spans.find(s => s.name === 'agent.spawn')
      expect(spawnSpan?.attributes.childAgentId).toContain('billing-specialist')
    })

    it('goal 在两次 invoke 中保持不变', async () => {
      const [cp1, cp2] = await Promise.all([
        stateStore.getByContextAndTurn(contextId, 1),
        stateStore.getByContextAndTurn(contextId, 2),
      ])
      expect(cp1.goal).toBe(cp2.goal)
    })
  })

  describe('Path C: 升级路径 — 置信度不足直接升级', () => {
    let result: AgentResult
    let trajectory: Trajectory

    beforeAll(async () => {
      result = await milkie.invoke({
        agentId: 'customer-service',
        goal: '处理客户服务请求',
        input: '我要投诉你们！',
      })
      trajectory = await trajectoryStore.getByRunId(result.agentRunId)
    })

    it('FSM 直接从 intent_classification 转移到 escalated', () => {
      const fsmStates = trajectory.spans
        .filter(s => s.name === 'fsm.transition')
        .map(s => s.attributes.toState)
      expect(fsmStates).toEqual(['escalated'])
      expect(fsmStates).not.toContain('collecting_slots')
    })

    it('classify_intent 置信度 < 0.75', () => {
      const classifySpan = trajectory.spans.find(
        s => s.name === 'tool.call' && s.attributes.toolName === 'classify_intent'
      )
      expect((classifySpan?.attributes.input as any).confidence).toBeLessThan(0.75)
    })

    it('escalated state 输出人工客服转接说明', () => {
      expect(result.output).toMatch(/人工|客服|转接/)
    })

    it('不触发槽填充工具', () => {
      const slotSpans = trajectory.spans.filter(
        s => s.name === 'tool.call' && s.attributes.toolName === 'collect_slot'
      )
      expect(slotSpans).toHaveLength(0)
    })
  })
})
```

---

## 能力覆盖矩阵

| 能力点 | Case 1 | Case 2 | Case 3 | Case 4 | Case 5 | Case 6 |
|--------|:------:|:------:|:------:|:------:|:------:|:------:|
| type: llm 单 state（LLM loop） | ✓ | ✓ | ✓ | | ✓ | |
| type: llm 无 on.DONE（等待用户） | | | | ✓ | | |
| **多 state FSM（type: llm + type: action）** | | | | | | ✓ |
| **Cognitive toolbox（create_plan / update_step）** | ✓ | | | | | |
| **workingMemory plan state（checklist 模式）** | ✓ | | | | | |
| **intra-agent tool（handler stateless / workingMemory stateful）** | ✓ | | | | | ✓ |
| **intra-agent 并行（单次响应多 tool_use）** | ✓ | | | | | |
| **Named sub-agent tools（inter-agent tool）** | | ✓ | | | | ✓ |
| **inter-agent 并行（allSettled join）** | | ✓ | | | | |
| **Context 隔离（独立 FSM+Context）** | | ✓ | | | | |
| **意图识别硬转移（classify_intent → FSM event）** | | | | | | ✓ |
| **置信度检查（handler 内部判断 → ctx.emit）** | | | | | | ✓ |
| **多轮槽填充（working_memory 累积）** | | | | | | ✓ |
| **用户确认流程（event_transition）** | | | | | | ✓ |
| **FSM 状态回退（USER_REJECTED → collecting_slots）** | | | | | | ✓ |
| **低置信度升级（ESCALATE terminal）** | | | | | | ✓ |
| skill_request + epoch 边界 | | | | | ✓ | |
| instructions bucket 单 turn 内冻结 | | | | | ✓ | |
| contextEpoch 递增 | | | | | ✓ | |
| prefix cache breakpoint 重建 | | | | | ✓ | |
| Model Gateway（Anthropic adapter）| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Toolbox 直接调用（search / filesystem）| ✓ | | | | ✓ | |
| Toolbox（parallelSafe: false）| | | ✓ | ✓ | | |
| State Store（MemoryStore）| ✓ | ✓ | | | ✓ | |
| State Store（SQLiteStore）| | | ✓ | | | |
| State Store（RedisStore）| | | | ✓ | | ✓ |
| Checkpoint + sequence | | | ✓ | | | |
| Resume | | | ✓ | | | |
| Tool 幂等键（fromCache 验证）| | | ✓ | | | |
| Interrupt / yield point | | | ✓ | | | |
| Supervisor Tree（父→子中断传播）| | | ✓ | | | |
| TaskResult.interrupted | | | ✓ | | | |
| TaskResult.success | | ✓ | | | | ✓ |
| pendingEvents 保存与恢复 | | | ✓ | | | |
| Error handling FSM 转移 | | | | ✓ | | |
| TaskResult.error + retryable | | | | ✓ | | |
| Goal 不变性 vs current_turn 多轮变化 | | | | ✓ | | ✓ |
| contextId 复用 | | | | ✓ | | ✓ |
| Trajectory + spans | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| ResolvedManifest | ✓ | ✓ | | | ✓ | |
| A/B Test / Experiment 模型 | | | | | ✓ | |
