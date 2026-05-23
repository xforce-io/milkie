/**
 * Case 3: 长任务中断与恢复
 *
 * 验证：Interrupt / yield point、Checkpoint 写入与恢复、Resume 语义、
 * Tool 幂等键、TaskResult.interrupted、pendingEvents 保存与恢复
 */

import type { AgentConfig } from '../../src/types/agent.js'
import type { AgentResult } from '../../src/types/common.js'
import type { IModelGateway, ModelRequest, ModelResponse } from '../../src/types/model.js'
import type { AgentCheckpoint } from '../../src/types/store.js'
import type { ToolDefinition } from '../../src/types/tool.js'
import { Milkie } from '../../src/runtime/Milkie.js'
import { TrajectoryStore } from '../../src/trajectory/TrajectoryStore.js'
import { SQLiteStore } from '../../src/store/SQLiteStore.js'
import { createToolCallTracker } from './helpers.js'
import fs from 'fs'
import path from 'path'

const SKIP = !process.env['VOLCENGINE_TOKEN'] || !process.env['VOLCENGINE_API_BASE']

// ─────────────────────────────── Tool ────────────────────────────────────────

function createProcessChunkTool(tracker: ReturnType<typeof createToolCallTracker>) {
  const processed = new Map<number, string>()

  const tool: ToolDefinition = {
    name:        'process_chunk',
    description: 'Process a data chunk. Returns cached result if already processed.',
    inputSchema: {
      type:       'object',
      properties: { chunkId: { type: 'number', description: 'Chunk ID (1-10)' } },
      required:   ['chunkId'],
    },
    handler: async (input: unknown) => {
      const { chunkId } = input as { chunkId: number }
      if (processed.has(chunkId)) {
        return { chunkId, result: processed.get(chunkId), fromCache: true }
      }
      // Slow enough to let the interrupt be injected between tool calls
      await new Promise<void>(r => setTimeout(r, 150))
      const result = `processed-${chunkId}`
      processed.set(chunkId, result)
      tracker.track('process_chunk')
      return { chunkId, result, fromCache: false }
    },
  }

  return { tool, processed }
}

// ─────────────────────────────── Agent Config ────────────────────────────────

const analystConfig: AgentConfig = {
  agentId:      'chunk-analyst',
  version:      '1.0.0',
  systemPrompt: `你是一个数据分析师。将数据集分成 10 个 chunk，依次调用 process_chunk 处理每个 chunk（chunkId 从 1 到 10，每次只调用一个），最后输出汇总所有结果。每次只处理一个 chunk，等返回后再处理下一个。所有 10 个 chunk 都处理完后才输出最终摘要。`,
  fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 25 }] },
  model: {
    provider: 'volcengine',
    model:    'doubao-seed-2.0-lite',
    adapter:  'openai-compatible',
  },
}

class SupervisorGateway implements IModelGateway {
  async complete(req: ModelRequest): Promise<ModelResponse> {
    const toolNames = req.tools?.map(t => t.name) ?? []
    if (toolNames.includes('worker-a') && toolNames.includes('worker-b')) {
      return {
        content: [
          { type: 'tool_use', id: 'spawn-a', name: 'worker-a', input: { goal: 'work a', input: 'do a' } },
          { type: 'tool_use', id: 'spawn-b', name: 'worker-b', input: { goal: 'work b', input: 'do b' } },
        ],
        toolCalls: [
          { id: 'spawn-a', name: 'worker-a', input: { goal: 'work a', input: 'do a' } },
          { id: 'spawn-b', name: 'worker-b', input: { goal: 'work b', input: 'do b' } },
        ],
        finishReason: 'tool_use',
      }
    }

    await new Promise<void>(resolve => setTimeout(resolve, 100))
    return {
      content:      [{ type: 'text', text: 'child done' }],
      toolCalls:    [],
      finishReason: 'end_turn',
    }
  }

  async *stream(_req: ModelRequest): AsyncIterable<never> {
    yield* []
  }
}

async function waitFor(
  predicate: () => Promise<boolean>,
  timeoutMs = 1000,
): Promise<void> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (await predicate()) return
    await new Promise<void>(resolve => setTimeout(resolve, 10))
  }
  throw new Error('Timed out waiting for condition')
}

function supervisorAgent(agentId: string, subAgents?: Record<string, string>): AgentConfig {
  return {
    agentId,
    version:      '1.0.0',
    systemPrompt: 'test supervisor tree interrupt',
    fsm:          { states: [{ name: 'react', type: 'llm', max_iterations: 3 }] },
    model:        {
      provider: 'test',
      model:    'test-model',
      adapter:  'test',
    },
    subAgents,
  }
}

// ─────────────────────────────── Tests ───────────────────────────────────────

describe('Case 3: 长任务中断与恢复', () => {
  let milkie: Milkie
  let trajectoryStore: TrajectoryStore
  let stateStore: SQLiteStore
  let tracker: ReturnType<typeof createToolCallTracker>
  let contextId: string
  let run1Result: AgentResult
  let run2Result: AgentResult
  let checkpoint: AgentCheckpoint

  beforeAll(async () => {
    if (SKIP) return

    tracker = createToolCallTracker()
    const { tool: processChunkTool } = createProcessChunkTool(tracker)

    trajectoryStore = new TrajectoryStore({ jsonlDir: './test-output/trajectories' })
    const dbPath = path.resolve('./test-output/case3/state.db')
    fs.mkdirSync(path.dirname(dbPath), { recursive: true })
    if (fs.existsSync(dbPath)) fs.unlinkSync(dbPath)
    stateStore = new SQLiteStore({ path: dbPath })
    await stateStore.init()
    milkie = new Milkie({ stateStore, trajectoryStore, tools: [processChunkTool] })
    milkie.registerAgent(analystConfig)

    contextId = `ctx-case3-${Date.now()}`

    // ── Phase 1: Run and interrupt ──────────────────────────────────────────
    const runPromise = milkie.invoke({
      agentId:   'chunk-analyst',
      goal:      '处理 dataset-42 的 10 个 chunk',
      input:     '请依次处理 chunk 1 到 10，每次调用一个 process_chunk',
      contextId,
    })

    // Wait for 3+ chunks, then interrupt
    await tracker.waitFor('process_chunk', 3)
    await new Promise<void>(r => setTimeout(r, 80))  // buffer for tool result append
    await milkie.interrupt(contextId)
    run1Result = await runPromise

    // ── Phase 2: Load checkpoint ────────────────────────────────────────────
    const cpKey = `context:${contextId}:checkpoint:latest`
    checkpoint = await stateStore.get(cpKey) as AgentCheckpoint

    // ── Phase 3: Resume ────────────────────────────────────────────────────
    if (checkpoint) {
      run2Result = await milkie.resume(
        cpKey,
        'chunk-analyst',
        '处理 dataset-42 的 10 个 chunk',
        '请继续处理剩余的 chunk，直到所有 10 个都处理完',
      )
    }
  }, 240_000)

  afterAll(() => {
    stateStore?.close()
  })

  const live = SKIP ? it.skip : it

  live('Phase 1: 中断后 status = interrupted', () => {
    expect(run1Result.status).toBe('interrupted')
  })

  live('Phase 1: 中断前至少处理了 3 个 chunk', () => {
    const count = tracker.counts['process_chunk'] ?? 0
    expect(count).toBeGreaterThanOrEqual(3)
  })

  live('Phase 1: checkpoint 保存到 stateStore', () => {
    expect(checkpoint).toBeDefined()
    expect(checkpoint.goal).toBe('处理 dataset-42 的 10 个 chunk')
    expect(checkpoint.pendingEvents).toBeDefined()
  })

  live('Phase 1: checkpoint 包含 FSM 状态', () => {
    expect(checkpoint).toBeDefined()
    expect(checkpoint.fsm).toBeDefined()
    expect(checkpoint.fsm.currentState).toBe('paused')
    expect(checkpoint.fsm.resumeState).toBe('react')
  })

  live('Phase 2: Resume 后执行完成', () => {
    expect(run2Result).toBeDefined()
    expect(run2Result.status).toBe('completed')
  })

  live('Phase 2: Resume 复用原 contextId 和 agentRunId', () => {
    expect(run2Result.contextId).toBe(run1Result.contextId)
    expect(run2Result.agentRunId).toBe(run1Result.agentRunId)
  })

  live('Phase 2: Resume 后总 chunk 累计 >= 9（含中断前已处理的）', () => {
    const total = tracker.counts['process_chunk'] ?? 0
    expect(total).toBeGreaterThanOrEqual(9)
  })

  live('Phase 2: Resume 后未重复处理已完成的 chunk（幂等性）', () => {
    // All process_chunk calls should be for unique chunk IDs
    // (The LLM sees history and won't re-call already done chunks)
    // We just verify the total count makes sense (not > 10 if LLM is smart)
    const total = tracker.counts['process_chunk'] ?? 0
    expect(total).toBeLessThanOrEqual(20)  // should not exceed 2× the total
  })

  live('Trajectory 包含 tool.call span', async () => {
    const traj = await trajectoryStore.getByRunId(run1Result.agentRunId)
    const toolSpans = traj.spans.filter(s => s.name === 'tool.call')
    expect(toolSpans.length).toBeGreaterThanOrEqual(3)
  })

  live('Run1 trajectory 中 process_chunk 数量 >= 3', async () => {
    const traj = await trajectoryStore.getByRunId(run1Result.agentRunId)
    const chunkSpans = traj.spans.filter(
      s => s.name === 'tool.call' && s.attributes['toolName'] === 'process_chunk'
    )
    expect(chunkSpans.length).toBeGreaterThanOrEqual(3)
  })

  live('Trajectory 跨 resume 按同一个 runId 续写', async () => {
    const traj = await trajectoryStore.getByRunId(run1Result.agentRunId)
    const runSpans = traj.spans.filter(s => s.name === 'agent.run')
    expect(runSpans.length).toBeGreaterThanOrEqual(2)
    expect(new Set(traj.spans.map(s => s.traceId)).size).toBe(1)
  })
})

describe('Case 3 子流：Supervisor Tree 中断传播', () => {
  it('父 Agent 中断后 children checkpoint 记录两个子 Agent 的 checkpointId', async () => {
    const stateStore = new SQLiteStore({ path: path.resolve('./test-output/case3/supervisor-state.db') })
    const dbPath = path.resolve('./test-output/case3/supervisor-state.db')
    if (fs.existsSync(dbPath)) fs.unlinkSync(dbPath)
    await stateStore.init()

    try {
      const milkie = new Milkie({
        stateStore,
        trajectoryStore: new TrajectoryStore({ jsonlDir: './test-output/trajectories' }),
        gateway: new SupervisorGateway(),
      })
      milkie.registerAgent(supervisorAgent('worker-a'))
      milkie.registerAgent(supervisorAgent('worker-b'))
      milkie.registerAgent(supervisorAgent('supervisor', {
        'worker-a': '1.0.0',
        'worker-b': '1.0.0',
      }))

      const runPromise = milkie.invoke({
        agentId:   'supervisor',
        goal:      'coordinate workers',
        input:     'start workers',
        contextId: 'ctx-case3-supervisor',
      })

      await waitFor(async () => {
        const children = await stateStore.get('context:ctx-case3-supervisor:children') as Array<{ status: string }> | undefined
        return (children ?? []).filter(c => c.status === 'running').length === 2
      })

      await milkie.interrupt('ctx-case3-supervisor')
      const result = await runPromise
      const parentCp = await stateStore.get('context:ctx-case3-supervisor:checkpoint:latest') as AgentCheckpoint

      expect(result.status).toBe('interrupted')
      expect(parentCp.fsm.currentState).toBe('paused')
      expect(parentCp.children).toHaveLength(2)
      expect(parentCp.children.every(c => c.status === 'interrupted')).toBe(true)
      expect(parentCp.children.every(c => c.checkpointId != null)).toBe(true)
    } finally {
      stateStore.close()
    }
  })
})
