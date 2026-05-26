/**
 * Case 4: 多轮对话与错误恢复
 *
 * 验证：多轮对话（type: llm，无 on.DONE → 等待用户）、contextId 复用
 * （history 跨 invoke 保留）、Goal 不变性、Error handling FSM 转移（retryable 错误自动重试）
 */

import type { AgentConfig } from '../../src/types/agent.js'
import type { AgentResult } from '../../src/types/common.js'
import type { AgentCheckpoint, IStateStore } from '../../src/types/store.js'
import type { Trajectory } from '../../src/types/trajectory.js'
import type { ToolDefinition } from '../../src/types/tool.js'
import { Milkie } from '../../src/runtime/Milkie.js'
import { TrajectoryStore } from '../../src/trajectory/TrajectoryStore.js'
import { MemoryStore } from '../../src/store/MemoryStore.js'

const SKIP = !process.env['VOLCENGINE_TOKEN'] || !process.env['VOLCENGINE_API_BASE']

// ─────────────────────────────── Tool ────────────────────────────────────────

function createQueryOrdersTool() {
  let callCount = 0

  const tool: ToolDefinition = {
    name:        'query_orders',
    description: 'Query order details by order ID.',
    inputSchema: {
      type:       'object',
      properties: { orderId: { type: 'string', description: 'Order ID to query' } },
      required:   ['orderId'],
    },
    handler: async (input: unknown) => {
      const { orderId } = input as { orderId: string }
      callCount++

      // First call simulates a timeout / retryable error
      if (callCount === 1) {
        const err = new Error('Connection timeout') as Error & { retryable: boolean }
        err.retryable = true
        throw err
      }

      return {
        orderId,
        amount:          15000,
        threshold:       5000,
        customerHistory: 'new_customer',
        flagged:         true,
      }
    },
  }

  return { tool, getCallCount: () => callCount }
}

// ─────────────────────────────── Agent Config ────────────────────────────────

const orderAnalystConfig: AgentConfig = {
  agentId:      'order-analyst',
  version:      '1.0.0',
  systemPrompt: `你是一个订单异常分析师。使用 query_orders 查询订单详情，结合提供的信息给出分析判断。

每次用户输入后，分析订单情况并给出回复。如果需要查询订单，调用 query_orders。
输出你的分析后等待用户的下一条消息。`,
  fsm: {
    states: [{
      name: 'analyze',
      type: 'llm',
      // 无 on.DONE → 输出后等待用户消息（多轮对话模式）
    }],
  },
  model: {
    provider: 'volcengine',
    model:    'doubao-seed-2.0-lite',
    adapter:  'openai-compatible',
  },
}

// ─────────────────────────────── Tests ───────────────────────────────────────

describe('Case 4: 多轮对话与错误恢复', () => {
  let milkie: Milkie
  let trajectoryStore: TrajectoryStore
  let stateStore: IStateStore
  let queryOrdersTool: ToolDefinition
  let getCallCount: () => number

  let run1: AgentResult
  let run2: AgentResult
  let run1Cp: AgentCheckpoint | null
  let run2Cp: AgentCheckpoint | null
  let trajectory: Trajectory

  const goal = '分析订单 #12345 的异常原因'

  beforeAll(async () => {
    if (SKIP) return

    const { tool, getCallCount: getCount } = createQueryOrdersTool()
    queryOrdersTool = tool
    getCallCount = getCount

    trajectoryStore = new TrajectoryStore({ jsonlDir: './test-output/trajectories' })
    stateStore = new MemoryStore()
    milkie = new Milkie({ stateStore, trajectoryStore, tools: [queryOrdersTool] })
    milkie.registerAgent(orderAnalystConfig)

    // First turn: initial problem statement
    run1 = await milkie.invoke({
      agentId: 'order-analyst',
      goal,
      input:   '订单 #12345 金额超出阈值 3 倍，请分析异常原因',
    })

    run1Cp = await stateStore.get(`context:${run1.contextId}:checkpoint:latest`) as AgentCheckpoint | null

    // Second turn: provide additional context, same contextId
    run2 = await milkie.invoke({
      agentId:   'order-analyst',
      goal,
      input:     '客户历史消费记录显示为正常季节性采购，请综合判断',
      contextId: run1.contextId,
    })

    run2Cp = await stateStore.get(`context:${run2.contextId}:checkpoint:latest`) as AgentCheckpoint | null
    trajectory = await trajectoryStore.getByAgentId('order-analyst')
  }, 120_000)

  const live = SKIP ? it.skip : it

  live('两次 invoke 均成功完成', () => {
    expect(run1.status).toBe('completed')
    expect(run2.status).toBe('completed')
  })

  live('两次 invoke 使用同一 contextId', () => {
    expect(run1.contextId).toBe(run2.contextId)
  })

  live('goal 在两次 invoke 中保持不变', () => {
    expect(run1Cp?.goal).toBe(goal)
    expect(run2Cp?.goal).toBe(goal)
  })

  live('第 2 次 invoke 的 context history 包含第 1 轮对话', () => {
    // New substrate: history is stored as regions with section === 'history',
    // not in a flat context.history array. Each turn crystallises into a
    // (user, finalAssistant) pair region persisted across turns.
    const regions = run2Cp?.context?.regions?.regions ?? []
    const historyRegions = regions.filter((r: { section: string }) => r.section === 'history')
    expect(historyRegions.length).toBeGreaterThan(0)
    // Sanity: the history pair content should be non-trivial
    const historyStr = JSON.stringify(historyRegions)
    expect(historyStr.length).toBeGreaterThan(100)
  })

  live('query_orders 被调用至少 2 次（第 1 次超时 retryable，第 2 次成功）', () => {
    expect(getCallCount()).toBeGreaterThanOrEqual(2)
  })

  live('error_handling FSM 转移 span 存在（retryable 错误重试）', () => {
    const fsmSpans = trajectory.spans.filter(s => s.name === 'fsm.transition')
    const toError   = fsmSpans.find(s => s.attributes['toState'] === 'error_handling')
    const fromError = fsmSpans.find(s => s.attributes['fromState'] === 'error_handling')
    expect(toError).toBeDefined()
    expect(fromError).toBeDefined()
  })

  live('run2 output 包含综合判断', () => {
    expect(run2.output.length).toBeGreaterThan(20)
    expect(run2.output).toMatch(/正常|异常|判断|结论|分析|季节性/i)
  })

  live('trajectory 中 query_orders tool.call span 数量 >= 2', () => {
    const querySpans = trajectory.spans.filter(
      s => s.name === 'tool.call' && s.attributes['toolName'] === 'query_orders'
    )
    expect(querySpans.length).toBeGreaterThanOrEqual(2)
  })
})
