/**
 * s-011 — MIGRATION NOTE (#175 de-core multi-state FSM)
 * ──────────────────────────────────────────────────────────────────────────────
 * This story originally exercised a developer-authored MULTI-STATE business FSM
 * (intent routing + DialogFlow-style state topology): `type: llm` + `type: action`
 * states wired by `on:` business transitions, tool handlers firing hard
 * transitions through `ctx.emit`, `fsm.transition` events, guard evaluations, and
 * sub-agent action states (`cancellation-executor` / `billing-specialist`).
 *
 * #175 removed the multi-state business FSM from core (see
 * docs/design/175-decore-multistate-fsm.md — D1/D2 and §6 "结构化对话分档").
 * What was deleted from core:
 *   - the FSMEngine business-topology layer (`states` map / `on:` / `pendingEvent`
 *     / emit→transition),
 *   - `ctx.emit` hard transitions (#60 `tool.emitted` / `replayEmits`),
 *   - `#31` guard evaluations,
 *   - `fsm.transition` business events.
 * The lower run-lifecycle SM (RunLifecycle: running/paused/…) is kept; the upper
 * business-topology SM is now a USERLAND composition concern, not a core feature.
 *
 * Consequently the three original LIVE paths (A intent→slots→confirm→execute,
 * B billing routing to a specialist sub-agent, C low-confidence escalation to a
 * terminal `escalated` state) can NO LONGER be expressed as a core multi-state
 * FSM, and are intentionally NOT carried over — preserving them would re-introduce
 * the exact constructs #175 deleted, leaving this e2e red against current core.
 *
 * Per the design's §6 lightweight tier, slot-filling is re-expressed WITHOUT a
 * state machine: a SINGLE-state `type: llm` agent whose `collect_slot` tool
 * accumulates slots in workingMemory (a plain WM phase/slot variable + tool param
 * schema as the hard floor). No `ctx.emit`, no business `on:` topology, no
 * `fsm.transition`. The deterministic test below pins that this userland
 * slot-filling expression survives the de-core: the autonomous loop keeps running
 * while the tool returns calls (the `continue` signal) and completes on the final
 * text (the `done` signal), with WM accumulating across loop iterations.
 *
 * The full multi-state DialogFlow → slot-filling + action-precondition migration
 * POC lives in examples/repair-ticketing/ (design §10 slice 5), not here.
 */

import type { AgentConfig } from '../../src/types/agent.js'
import type { AgentResult } from '../../src/types/common.js'
import type { IModelGateway, ModelRequest, ModelResponse } from '../../src/types/model.js'
import type { Trajectory } from '../../src/types/trajectory.js'
import type { ToolDefinition } from '../../src/types/tool.js'
import { Milkie } from '../../src/runtime/Milkie.js'
import { TrajectoryStore } from '../../src/trajectory/TrajectoryStore.js'
import { MemoryStore } from '../../src/store/MemoryStore.js'
import { MemoryEventStore } from '../../src/trace/MemoryEventStore.js'

// ─────────────────────────────── Tools ───────────────────────────────────────

// Userland slot-filling: the tool's job is a pure, deterministic side effect —
// accumulate the named slot into workingMemory and report progress. The handler
// does NOT drive any state transition (no `ctx.emit`); whether the loop keeps
// going is decided by the agent loop itself (more tool calls vs final text).
const collectSlotTool: ToolDefinition = {
  name:        'collect_slot',
  description: '收集订单取消所需的槽位信息（orderId、reason、preferRefund），累积进 workingMemory。',
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

    const required = ['orderId', 'reason', 'preferRefund']
    const complete = required.every(k => slots[k] !== undefined)
    return { collected: slots, complete }
  },
}

// ─────────────────────────────── Agent Config ────────────────────────────────

// Single-state autonomous agent (no `on:` topology, no action sub-states). The
// loadout focuses the LLM on slot-filling; the tool param schema is the hard
// floor (#175 §6). Completion is reached when the model emits final text.
const customerServiceConfig: AgentConfig = {
  agentId:      'customer-service',
  version:      '1.0.0',
  systemPrompt: '你是一个客服助手，负责取消订单。依次使用 collect_slot 工具收集 orderId、reason、preferRefund 三个槽位；全部收集完毕后，向用户确认操作已完成。',
  fsm: { states: [{ name: 'react', type: 'llm', tools: ['collect_slot'] }] },
  model: { provider: 'volcengine', model: 'doubao-seed-2.0-lite', adapter: 'openai-compatible' },
}

// Deterministic gateway: drive collection of the three slots, then a final text.
// Each tool call → the loop continues; the final text → the run completes.
class SlotFillingGateway implements IModelGateway {
  private collectedSlots = 0

  async complete(_request: ModelRequest): Promise<ModelResponse> {
    if (this.collectedSlots === 0) {
      this.collectedSlots++
      return this.tool('slot-order', 'collect_slot', { name: 'orderId', value: 'ORD-456' })
    }
    if (this.collectedSlots === 1) {
      this.collectedSlots++
      return this.tool('slot-reason', 'collect_slot', { name: 'reason', value: '商品损坏' })
    }
    if (this.collectedSlots === 2) {
      this.collectedSlots++
      return this.tool('slot-refund', 'collect_slot', { name: 'preferRefund', value: 'true' })
    }
    return this.text('已收集全部槽位，取消订单操作已完成。')
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

// ─────────────────────────────── Test ────────────────────────────────────────

describe('s-011 (#175 migrated): userland slot-filling, single-state agent', () => {
  let milkie: Milkie
  let trajectoryStore: TrajectoryStore
  let result: AgentResult
  let trajectory: Trajectory

  const contextId = `ctx-s011-slots-${Date.now()}`
  const goal = '处理客户取消订单请求'

  beforeAll(async () => {
    trajectoryStore = new TrajectoryStore({ jsonlDir: './test-output/trajectories' })
    milkie = new Milkie({
      stateStore:  new MemoryStore(),
      eventStore:  new MemoryEventStore(),
      trajectoryStore,
      gateway:     new SlotFillingGateway(),
      tools:       [collectSlotTool],
    })
    milkie.registerAgent(customerServiceConfig)

    result = await milkie.invoke({
      agentId: 'customer-service',
      goal,
      input:   '我想取消订单 ORD-456，原因是商品损坏，需要退款',
      contextId,
    })
    trajectory = await trajectoryStore.getByContextId(contextId)
  }, 30_000)

  it('run completes via the autonomous loop (done signal on final text)', () => {
    expect(result.status).toBe('completed')
    expect(result.contextId).toBe(contextId)
    expect(result.output).toMatch(/完成|取消|操作/)
  })

  it('collect_slot is invoked once per slot, accumulating in order', () => {
    const slotNames = trajectory.spans
      .filter(s => s.name === 'tool.call' && s.attributes['toolName'] === 'collect_slot')
      .map(s => (s.attributes['input'] as { name?: string }).name)
    expect(slotNames).toEqual(['orderId', 'reason', 'preferRefund'])
  })

  it('the tool reports all three slots collected (workingMemory accumulation)', () => {
    const lastSlotSpan = [...trajectory.spans]
      .reverse()
      .find(s => s.name === 'tool.call' && s.attributes['toolName'] === 'collect_slot')
    expect(lastSlotSpan).toBeDefined()
    const output = lastSlotSpan!.attributes['output'] as
      { collected?: Record<string, unknown>; complete?: boolean } | undefined
    expect(output?.complete).toBe(true)
    expect(output?.collected).toMatchObject({
      orderId:      'ORD-456',
      reason:       '商品损坏',
      preferRefund: 'true',
    })
  })

  // #175 de-core invariant: no business multi-state FSM survives. The single-state
  // autonomous loop must NOT emit any business-topology `fsm.transition` span.
  it('records no business fsm.transition spans (core is business-state agnostic)', () => {
    const businessTransitions = trajectory.spans.filter(s => s.name === 'fsm.transition')
    expect(businessTransitions).toHaveLength(0)
  })
})
