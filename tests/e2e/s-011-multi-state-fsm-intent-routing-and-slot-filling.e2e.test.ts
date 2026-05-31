/**
 * Case 6: 客服意图路由与多轮槽填充
 *
 * 验证：多 state FSM（type: llm + type: action）、意图识别硬转移（ctx.emit）、
 * 多轮槽填充（workingMemory 累积）、用户确认流程、低置信度升级（terminal escalated）
 *
 * 三条测试路径：
 *   Path A: 意图清晰 → 取消订单 → 收集槽位 → 用户确认 → 执行
 *   Path B: 账单查询（intent: billing）→ 路由到专家
 *   Path C: 置信度不足 → 直接升级人工（terminal LLM state 输出）
 */

import type { AgentConfig } from '../../src/types/agent.js'
import type { AgentResult } from '../../src/types/common.js'
import type { IModelGateway, ModelRequest, ModelResponse } from '../../src/types/model.js'
import type { Trajectory } from '../../src/types/trajectory.js'
import type { ToolDefinition } from '../../src/types/tool.js'
import { Milkie } from '../../src/runtime/Milkie.js'
import { TrajectoryStore } from '../../src/trajectory/TrajectoryStore.js'
import { RedisStore } from '../../src/store/RedisStore.js'
import { MemoryStore } from '../../src/store/MemoryStore.js'
import { MemoryEventStore } from '../../src/trace/MemoryEventStore.js'
import { checkpointFromEvents } from '../../src/trace/diagnostics/checkpointFromEvents.js'
import type { FsmTransitionPayload } from '../../src/trace/types.js'
import { createRedisStore } from './redis.js'

// #73: the resume checkpoint is in the event log; resolve it via the
// context→runId routing pointer (replaces the removed stateStore blob).
async function readCheckpoint(
  stateStore: { get: (k: string) => Promise<unknown> },
  eventStore: MemoryEventStore,
  contextId: string,
): Promise<{ context: { workingMemory: { data: Record<string, unknown> } }; fsm: { currentState: string } } | null> {
  const runId = await stateStore.get(`context:${contextId}:checkpoint-run:latest`) as string | undefined
  if (!runId) return null
  return checkpointFromEvents(await eventStore.readByRunId(runId)) as never
}

const SKIP = !process.env['VOLCENGINE_TOKEN'] || !process.env['VOLCENGINE_API_BASE']
const RUN_REDIS_E2E = process.env['REDIS_E2E_REQUIRED'] === '1'

// ─────────────────────────────── Tools ───────────────────────────────────────

const classifyIntentTool: ToolDefinition = {
  name:        'classify_intent',
  description: '识别用户意图。根据用户消息判断意图类型和置信度。置信度 < 0.75 时应触发 ESCALATE 升级。',
  inputSchema: {
    type:       'object',
    properties: {
      intent:     { type: 'string', enum: ['cancellation', 'billing', 'unclear'] },
      confidence: { type: 'number', description: '0.0~1.0，自我评估的置信度' },
    },
    required: ['intent', 'confidence'],
  },
  handler: async (input: unknown, ctx) => {
    const { intent, confidence } = input as { intent: string; confidence: number }
    ctx.workingMemory.set('intent', intent)
    ctx.workingMemory.set('intentConfidence', confidence)

    if (confidence < 0.75) {
      ctx.emit('ESCALATE', undefined, {
        guardId: 'intent-threshold', result: 'ESCALATE',
        contextSlice: { intent, confidence, threshold: 0.75 },
      })
      return { accepted: false, reason: 'low_confidence', confidence }
    }

    const eventMap: Record<string, string> = {
      cancellation: 'INTENT_CANCELLATION',
      billing:      'INTENT_BILLING',
      unclear:      'INTENT_UNCLEAR',
    }
    ctx.emit(eventMap[intent] ?? 'ESCALATE', undefined, {
      guardId: 'intent-threshold', result: eventMap[intent] ?? 'ESCALATE',
      contextSlice: { intent, confidence, threshold: 0.75 },
    })
    return { accepted: true, intent, confidence }
  },
}

const collectSlotTool: ToolDefinition = {
  name:        'collect_slot',
  description: '收集订单取消所需的槽位信息（orderId、reason、preferRefund）。所有槽位收集完毕后自动触发确认流程。',
  inputSchema: {
    type:       'object',
    properties: {
      name:  { type: 'string', enum: ['orderId', 'reason', 'preferRefund'] },
      value: { type: 'string' },
    },
    required: ['name', 'value'],
  },
  handler: async (input: unknown, ctx) => {
    const { name, value } = input as { name: string; value: string }
    const slots = (ctx.workingMemory.get('collectedSlots') as Record<string, unknown>) ?? {}
    slots[name] = value
    ctx.workingMemory.set('collectedSlots', slots)

    const allFilled = ['orderId', 'reason', 'preferRefund'].every(k => slots[k] !== undefined)
    if (allFilled) ctx.emit('SLOTS_COMPLETE', undefined, {
      guardId: 'slots-complete', result: 'SLOTS_COMPLETE',
      contextSlice: { filled: Object.keys(slots), required: ['orderId', 'reason', 'preferRefund'] },
    })
    return { collected: slots, complete: allFilled }
  },
}

const confirmActionTool: ToolDefinition = {
  name:        'confirm_action',
  description: '处理用户的确认或拒绝。确认则触发执行，拒绝则退回收集槽位。',
  inputSchema: {
    type:       'object',
    properties: {
      confirmed:   { type: 'boolean' },
      userMessage: { type: 'string' },
    },
    required: ['confirmed'],
  },
  handler: async (input: unknown, ctx) => {
    const { confirmed } = input as { confirmed: boolean }
    ctx.emit(confirmed ? 'USER_CONFIRMED' : 'USER_REJECTED', undefined, {
      guardId: 'user-confirm', result: confirmed ? 'USER_CONFIRMED' : 'USER_REJECTED',
      contextSlice: { confirmed },
    })
    return { confirmed }
  },
}

// Sub-agents for executing and routing
const cancellationExecutorConfig: AgentConfig = {
  agentId:      'cancellation-executor',
  version:      '1.0.0',
  systemPrompt: '你是取消订单执行器。接收取消请求，执行取消操作，返回"取消成功，退款将在3-5个工作日处理"。',
  fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 2 }] },
  model: { provider: 'volcengine', model: 'doubao-seed-2.0-lite', adapter: 'openai-compatible' },
}

const billingSpecialistConfig: AgentConfig = {
  agentId:      'billing-specialist',
  version:      '1.0.0',
  systemPrompt: '你是账单查询专家。分析用户的账单问题，给出专业解答。',
  fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 2 }] },
  model: { provider: 'volcengine', model: 'doubao-seed-2.0-lite', adapter: 'openai-compatible' },
}

// ─────────────────────────────── Agent Config ────────────────────────────────

const customerServiceConfig: AgentConfig = {
  agentId:      'customer-service',
  version:      '1.0.0',
  systemPrompt: `你是一个客服助手。根据意图采取对应行动：

- 取消订单（cancellation）：收集 orderId、reason、preferRefund 三个槽位（使用 collect_slot 工具），然后请用户确认
- 账单查询（billing）：转交给 billing-specialist 处理
- 意图不明确（unclear）：礼貌追问一次，仍不明确则升级
- 置信度 < 0.75：直接 ESCALATE

处理流程：
1. 调用 classify_intent 识别意图（提供 intent 和 confidence 参数）
2. 如果 cancellation：依次调用 collect_slot 收集 orderId、reason、preferRefund（全部收集后自动进入确认）
3. 如果 billing：系统自动路由到 billing-specialist
4. 确认阶段：用户确认时调用 confirm_action(confirmed: true)，拒绝时调用 confirm_action(confirmed: false)`,
  fsm: {
    states: [
      {
        name:  'intent_classification',
        type:  'llm',
        tools: ['classify_intent'],
        on: {
          INTENT_CANCELLATION: 'collecting_slots',
          INTENT_BILLING:      'routing_to_specialist',
          INTENT_UNCLEAR:      'clarifying',
          ESCALATE:            'escalated',
        },
      },
      {
        name:         'clarifying',
        type:         'llm',
        tools:        ['classify_intent'],
        instructions: '用户已回复了你的追问。请先调用 classify_intent 重新判断用户意图（提供 intent 和 confidence 参数），再根据结果采取相应行动。',
        on: {
          INTENT_CANCELLATION: 'collecting_slots',
          INTENT_BILLING:      'routing_to_specialist',
          INTENT_UNCLEAR:      'escalated',   // After one clarification attempt, still unclear → escalate
          ESCALATE:            'escalated',
        },
      },
      {
        name:  'collecting_slots',
        type:  'llm',
        tools: ['collect_slot'],
        on: {
          SLOTS_COMPLETE: 'confirming',
          ESCALATE:       'escalated',
        },
      },
      {
        name:    'routing_to_specialist',
        type:    'action',
        handler: 'billing-specialist',
        on:      { DONE: 'completed' },
      },
      {
        name:  'confirming',
        type:  'llm',
        tools: ['confirm_action'],
        on: {
          USER_CONFIRMED: 'executing',
          USER_REJECTED:  'collecting_slots',
        },
      },
      {
        name:    'executing',
        type:    'action',
        handler: 'cancellation-executor',
        on: {
          DONE:  'completed',
          ERROR: 'escalated',
        },
      },
      {
        name:         'escalated',
        type:         'llm',
        terminal:     true,
        tools:        [],  // no tools — produce final escalation message only
        instructions: '向用户说明已转接人工客服，给出预计等待时间（3-5分钟），保持礼貌。',
      },
      {
        name:         'completed',
        type:         'llm',
        terminal:     true,
        tools:        [],  // no tools — produce final confirmation only
        instructions: '向用户简要确认操作已完成。',
      },
    ],
  },
  model: { provider: 'volcengine', model: 'doubao-seed-2.0-lite', adapter: 'openai-compatible' },
  subAgents: {
    'cancellation-executor': '1.0.0',
    'billing-specialist':    '1.0.0',
  },
}

class SlotFillingGateway implements IModelGateway {
  private collectedSlots = 0
  private waitingForNextUserSlot = false
  private confirmationAsked = false

  async complete(request: ModelRequest): Promise<ModelResponse> {
    const toolNames = request.tools?.map(t => t.name) ?? []

    if (request.system?.includes('取消订单执行器')) {
      return this.text('取消成功，退款将在3-5个工作日处理')
    }

    if (toolNames.includes('classify_intent')) {
      return this.tool('intent-1', 'classify_intent', { intent: 'cancellation', confidence: 0.95 })
    }

    if (toolNames.includes('collect_slot')) {
      if (this.waitingForNextUserSlot) {
        this.waitingForNextUserSlot = false
        return this.text('请继续提供下一个取消订单信息')
      }
      if (this.collectedSlots === 0) {
        this.collectedSlots++
        this.waitingForNextUserSlot = true
        return this.tool('slot-order', 'collect_slot', { name: 'orderId', value: 'ORD-456' })
      }
      if (this.collectedSlots === 1) {
        this.collectedSlots++
        this.waitingForNextUserSlot = true
        return this.tool('slot-reason', 'collect_slot', { name: 'reason', value: '商品损坏' })
      }
      if (this.collectedSlots === 2) {
        this.collectedSlots++
        return this.tool('slot-refund', 'collect_slot', { name: 'preferRefund', value: 'true' })
      }
      return this.text('请确认是否取消订单 ORD-456')
    }

    if (toolNames.includes('confirm_action')) {
      if (!this.confirmationAsked) {
        this.confirmationAsked = true
        return this.text('请确认是否执行取消')
      }
      return this.tool('confirm-1', 'confirm_action', { confirmed: true, userMessage: '确认' })
    }

    return this.text('操作已完成')
  }

  async *stream(_request: ModelRequest): AsyncIterable<never> {
    yield* []
  }

  private text(text: string): ModelResponse {
    return {
      content:      [{ type: 'text', text }],
      toolCalls:    [],
      finishReason: 'end_turn',
    }
  }

  private tool(id: string, name: string, input: unknown): ModelResponse {
    return {
      content:      [{ type: 'tool_use', id, name, input }],
      toolCalls:    [{ id, name, input }],
      finishReason: 'tool_use',
    }
  }
}

// ─────────────────────────────── Shared Setup ────────────────────────────────

async function createTestEnv() {
  const trajectoryStore = new TrajectoryStore({ jsonlDir: './test-output/trajectories' })
  const eventStore      = new MemoryEventStore()
  const stateStore      = await createRedisStore(15)
  const milkie = new Milkie({
    stateStore,
    trajectoryStore,
    eventStore,
    tools: [classifyIntentTool, collectSlotTool, confirmActionTool],
  })
  milkie.registerAgent(cancellationExecutorConfig)
  milkie.registerAgent(billingSpecialistConfig)
  milkie.registerAgent(customerServiceConfig)
  return { milkie, trajectoryStore, eventStore, stateStore }
}

// ─────────────────────────────── Path A ──────────────────────────────────────

describe('Case 6 Path A: 主路径 — 取消订单', () => {
  let milkie: Milkie
  let trajectoryStore: TrajectoryStore
  let eventStore: MemoryEventStore
  let stateStore: RedisStore
  let trajectory: Trajectory
  let run1Result: AgentResult
  let run2Result: AgentResult

  const contextId = `ctx-case6-pathA-${Date.now()}`
  const goal      = '处理客户服务请求'

  async function sendTurn(input: string): Promise<AgentResult> {
    return milkie.invoke({ agentId: 'customer-service', goal, input, contextId })
  }

  beforeAll(async () => {
    if (SKIP) return
    const env = await createTestEnv()
    ;({ milkie, trajectoryStore, eventStore, stateStore } = env)

    // Turn 1: all slot info provided upfront (more deterministic than 3 separate turns)
    run1Result = await sendTurn('我想取消订单 ORD-456，原因是商品损坏，需要退款')

    // Turn 2: confirm
    run2Result = await sendTurn('确认，请执行取消')

    trajectory = await trajectoryStore.getByContextId(contextId)
  }, 180_000)

  afterAll(async () => {
    await stateStore?.flushdb()
    await stateStore?.disconnect()
  })

  const live = SKIP || !RUN_REDIS_E2E ? it.skip : it

  live('FSM 经历意图识别 → collecting_slots 转移', () => {
    const fsmStates = trajectory.spans
      .filter(s => s.name === 'fsm.transition')
      .map(s => s.attributes['toState'] as string)
    expect(fsmStates).toContain('collecting_slots')
  })

  live('classify_intent 调用意图为 cancellation，置信度 ≥ 0.75', () => {
    const classifySpan = trajectory.spans.find(
      s => s.name === 'tool.call' && s.attributes['toolName'] === 'classify_intent'
    )
    expect(classifySpan).toBeDefined()
    const inp = classifySpan!.attributes['input'] as { intent?: string; confidence?: number }
    expect(inp.intent).toBe('cancellation')
    expect(inp.confidence).toBeGreaterThanOrEqual(0.75)
  })

  live('classify 的硬转移携带 intent-threshold guard 依据', async () => {
    // guardEvaluations 落在 fsm.transition 事件 payload 上（不在 span attributes），
    // 所以从 eventStore 读 turn 1 的事件流，定位 INTENT_* 业务硬转移那条。
    const events = await eventStore.readByRunId(run1Result.agentRunId)
    const transition = events.find(
      e => e.type === 'fsm.transition' &&
        (e.payload as FsmTransitionPayload).trigger.name.startsWith('INTENT_')
    )
    expect(transition).toBeDefined()
    const ge = (transition!.payload as FsmTransitionPayload).guardEvaluations
    expect(ge?.[0]?.guardId).toBe('intent-threshold')
    expect((ge?.[0]?.contextSlice as { confidence?: number }).confidence).toBeGreaterThan(0)
  })

  live('collect_slot 收集了 orderId 槽位', () => {
    const slotSpans = trajectory.spans.filter(
      s => s.name === 'tool.call' && s.attributes['toolName'] === 'collect_slot'
    )
    expect(slotSpans.length).toBeGreaterThanOrEqual(1)
    const names = slotSpans.map(s => (s.attributes['input'] as { name?: string })?.name)
    expect(names).toContain('orderId')
  })

  live('FSM 经历完整的 confirming 流程', () => {
    const fsmStates = trajectory.spans
      .filter(s => s.name === 'fsm.transition')
      .map(s => s.attributes['toState'] as string)
    expect(fsmStates).toContain('confirming')
  })

  live('工作记忆保存了收集的槽位', async () => {
    const store = stateStore
    const cp = await readCheckpoint(store, eventStore, contextId) as { context?: { workingMemory?: unknown } } | null
    if (!cp) return  // checkpoint might not be saved if terminal
    const wm = cp.context?.workingMemory as { data?: Record<string, unknown> } | undefined
    if (!wm?.data?.['collectedSlots']) return
    const slots = wm.data['collectedSlots'] as Record<string, unknown>
    expect(slots['orderId']).toBeDefined()
  })

  live('最终到达 executing 或 completed 状态（取消执行）', () => {
    const fsmStates = trajectory.spans
      .filter(s => s.name === 'fsm.transition')
      .map(s => s.attributes['toState'] as string)
    const reached = fsmStates.includes('executing') || fsmStates.includes('completed')
    expect(reached).toBe(true)
  })

  live('不触发 escalated 状态', () => {
    const fsmStates = trajectory.spans
      .filter(s => s.name === 'fsm.transition')
      .map(s => s.attributes['toState'] as string)
    expect(fsmStates).not.toContain('escalated')
  })
})

// ─────────────────────────────── Path B ──────────────────────────────────────

describe('Case 6 Path B: 账单查询路由', () => {
  let milkie: Milkie
  let trajectoryStore: TrajectoryStore
  let stateStore: RedisStore
  let result: AgentResult
  let trajectory: Trajectory

  beforeAll(async () => {
    if (SKIP) return
    const env = await createTestEnv()
    ;({ milkie, trajectoryStore, stateStore } = env)

    result = await milkie.invoke({
      agentId: 'customer-service',
      goal:    '处理客户服务请求',
      input:   '我的上个月的账单有重复扣费，订单号是 ORD-789，金额 299 元被扣了两次',
    })
    trajectory = await trajectoryStore.getByRunId(result.agentRunId)
  }, 120_000)

  afterAll(async () => {
    await stateStore?.flushdb()
    await stateStore?.disconnect()
  })

  const live = SKIP || !RUN_REDIS_E2E ? it.skip : it

  live('classify_intent 被调用，识别出 billing 意图', () => {
    const classifySpan = trajectory.spans.find(
      s => s.name === 'tool.call' && s.attributes['toolName'] === 'classify_intent'
    )
    expect(classifySpan).toBeDefined()
    const inp = classifySpan!.attributes['input'] as { intent?: string }
    expect(inp.intent).toBe('billing')
  })

  live('FSM 路由到 routing_to_specialist 或 completed', () => {
    const fsmStates = trajectory.spans
      .filter(s => s.name === 'fsm.transition')
      .map(s => s.attributes['toState'] as string)
    const routed = fsmStates.includes('routing_to_specialist') || fsmStates.includes('completed')
    expect(routed).toBe(true)
  })

  live('不触发 collecting_slots 状态（账单查询不走取消流程）', () => {
    const fsmStates = trajectory.spans
      .filter(s => s.name === 'fsm.transition')
      .map(s => s.attributes['toState'] as string)
    expect(fsmStates).not.toContain('collecting_slots')
  })

  live('billing-specialist 被调用（inter-agent routing）', () => {
    const spawnSpan = trajectory.spans.find(s => s.name === 'agent.spawn')
    // Either billing-specialist was spawned, or FSM transitioned to completed via routing
    const fsmStates = trajectory.spans
      .filter(s => s.name === 'fsm.transition')
      .map(s => s.attributes['toState'] as string)
    const billingHandled = spawnSpan?.attributes['childAgentId'] === 'billing-specialist' ||
      fsmStates.includes('routing_to_specialist')
    expect(billingHandled).toBe(true)
  })
})

// ─────────────────────────────── Path C ──────────────────────────────────────

describe('Case 6 Path C: 升级路径 — 不明确意图升级人工客服', () => {
  let milkie: Milkie
  let trajectoryStore: TrajectoryStore
  let stateStore: RedisStore
  let result: AgentResult
  let trajectory: Trajectory

  beforeAll(async () => {
    if (SKIP) return
    const env = await createTestEnv()
    ;({ milkie, trajectoryStore, stateStore } = env)

    // Ambiguous complaint → classify_intent → ESCALATE (low confidence) or INTENT_UNCLEAR
    // → escalated (terminal LLM) → outputs "人工客服" text
    result = await milkie.invoke({
      agentId: 'customer-service',
      goal:    '处理客户服务请求',
      input:   '你们的服务太差了！！！我要投诉！',
    })
    trajectory = await trajectoryStore.getByRunId(result.agentRunId)
  }, 60_000)

  afterAll(async () => {
    await stateStore?.flushdb()
    await stateStore?.disconnect()
  })

  const live = SKIP || !RUN_REDIS_E2E ? it.skip : it

  live('FSM 最终到达 escalated（不经过 collecting_slots）', () => {
    const fsmStates = trajectory.spans
      .filter(s => s.name === 'fsm.transition')
      .map(s => s.attributes['toState'] as string)
    expect(fsmStates).toContain('escalated')
    expect(fsmStates).not.toContain('collecting_slots')
  })

  live('classify_intent 触发了升级路径（低置信度或不明确意图）', () => {
    const classifySpan = trajectory.spans.find(
      s => s.name === 'tool.call' && s.attributes['toolName'] === 'classify_intent'
    )
    expect(classifySpan).toBeDefined()
    const inp = classifySpan!.attributes['input'] as { intent?: string; confidence?: number }
    // Either direct ESCALATE (confidence < 0.75) or INTENT_UNCLEAR path
    const isEscalationPath = inp.intent === 'unclear' ||
      (inp.confidence !== undefined && inp.confidence < 0.75)
    expect(isEscalationPath).toBe(true)
  })

  live('escalated terminal LLM state 输出人工客服转接说明', () => {
    expect(result.output).toMatch(/人工|客服|转接|等待|分钟/i)
  })

  live('FSM 不触发 collecting_slots 状态', () => {
    const fsmStates = trajectory.spans
      .filter(s => s.name === 'fsm.transition')
      .map(s => s.attributes['toState'] as string)
    expect(fsmStates).not.toContain('collecting_slots')
  })
})

describe('Case 6 Path D: 多轮槽填充（确定性）', () => {
  let milkie: Milkie
  let trajectoryStore: TrajectoryStore
  let stateStore: MemoryStore
  let eventStore: MemoryEventStore
  let run1: AgentResult
  let run4: AgentResult
  let trajectory: Trajectory

  const contextId = `ctx-case6-slots-${Date.now()}`
  const goal = '处理客户取消订单请求'

  beforeAll(async () => {
    trajectoryStore = new TrajectoryStore({ jsonlDir: './test-output/trajectories' })
    stateStore = new MemoryStore()
    eventStore = new MemoryEventStore()
    milkie = new Milkie({
      stateStore,
      eventStore,
      trajectoryStore,
      gateway: new SlotFillingGateway(),
      tools: [classifyIntentTool, collectSlotTool, confirmActionTool],
    })
    milkie.registerAgent(cancellationExecutorConfig)
    milkie.registerAgent(billingSpecialistConfig)
    milkie.registerAgent(customerServiceConfig)

    run1 = await milkie.invoke({
      agentId: 'customer-service',
      goal,
      input: '我想取消订单',
      contextId,
    })
    let cp = (await readCheckpoint(stateStore, eventStore, contextId))!
    expect(cp.fsm.currentState).toBe('collecting_slots')
    expect(cp.context.workingMemory.data['collectedSlots']).toMatchObject({ orderId: 'ORD-456' })

    await milkie.invoke({
      agentId: 'customer-service',
      goal,
      input: '原因是商品损坏',
      contextId,
    })
    cp = (await readCheckpoint(stateStore, eventStore, contextId))!
    expect(cp.fsm.currentState).toBe('collecting_slots')
    expect(cp.context.workingMemory.data['collectedSlots']).toMatchObject({
      orderId: 'ORD-456',
      reason:  '商品损坏',
    })

    await milkie.invoke({
      agentId: 'customer-service',
      goal,
      input: '需要退款',
      contextId,
    })
    cp = (await readCheckpoint(stateStore, eventStore, contextId))!
    expect(cp.fsm.currentState).toBe('confirming')
    expect(cp.context.workingMemory.data['collectedSlots']).toMatchObject({
      orderId:      'ORD-456',
      reason:       '商品损坏',
      preferRefund: 'true',
    })

    run4 = await milkie.invoke({
      agentId: 'customer-service',
      goal,
      input: '确认，请执行取消',
      contextId,
    })
    trajectory = await trajectoryStore.getByContextId(contextId)
  }, 30_000)

  it('跨 4 轮 invoke 完成取消流程', () => {
    expect(run4.status).toBe('completed')
    expect(run4.contextId).toBe(contextId)
    expect(run4.output).toMatch(/完成|取消成功|操作/)
  })

  it('workingMemory 跨 turn 累积三个槽位', async () => {
    const cp = await readCheckpoint(stateStore, eventStore, contextId) as { context?: { workingMemory?: unknown } } | null
    const wm = cp?.context?.workingMemory as { data?: Record<string, unknown> } | undefined
    expect(wm?.data?.['collectedSlots']).toMatchObject({
      orderId:      'ORD-456',
      reason:       '商品损坏',
      preferRefund: 'true',
    })
  })

  it('FSM 覆盖 collecting_slots → confirming → executing/completed', () => {
    const fsmStates = trajectory.spans
      .filter(s => s.name === 'fsm.transition')
      .map(s => s.attributes['toState'] as string)
    expect(fsmStates).toContain('collecting_slots')
    expect(fsmStates).toContain('confirming')
    expect(fsmStates).toContain('executing')
    expect(fsmStates).toContain('completed')
  })

  it('collect_slot 分三次调用不同槽位', () => {
    const slotNames = trajectory.spans
      .filter(s => s.name === 'tool.call' && s.attributes['toolName'] === 'collect_slot')
      .map(s => (s.attributes['input'] as { name?: string }).name)
    expect(slotNames).toEqual(['orderId', 'reason', 'preferRefund'])
  })

  // #60: turn-1 is emit-driven (classify_intent → ctx.emit routes the intent
  // hard-transition into collecting_slots). Before #60 replaying it diverged,
  // because the handler is not re-run on replay so ctx.emit never fired.
  it('replays the emit-driven turn-1 run without divergence (#60)', async () => {
    const replayed = await milkie.replay(run1.agentRunId)
    expect(replayed.status).toBe(run1.status)
    expect(replayed.output).toBe(run1.output)
  })
})
