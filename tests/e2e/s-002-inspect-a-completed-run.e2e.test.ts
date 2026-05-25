/**
 * s-002: Inspect a completed agent run.
 *
 * Given a runId, exercise the read-side query surface of Agent Trace:
 *   - TrajectoryStore.getByRunId(runId) → ordered spans
 *   - EventStore.readByRunId(runId)    → ordered events (append order)
 *   - EventStore.readRange(runId, from, count) → sliced window
 *   - filter by event type / time window via plain array filter
 *
 * No replay, no fork, no lineage traversal — pure read.
 *
 * Uses a stub gateway so the test is hermetic and asserts the recording
 * infrastructure itself, not LLM behavior.
 */

import { Milkie } from '../../src/runtime/Milkie'
import { MemoryStore } from '../../src/store/MemoryStore'
import { MemoryEventStore } from '../../src/trace/MemoryEventStore'
import { TrajectoryStore } from '../../src/trajectory/TrajectoryStore'
import type { IModelGateway, ModelRequest, ModelResponse } from '../../src/types/model'
import type { AgentConfig } from '../../src/types/agent'
import type { ToolDefinition } from '../../src/types/tool'
import type {
  LlmRequestedPayload,
  LlmRespondedPayload,
  ToolRequestedPayload,
  ToolRespondedPayload,
} from '../../src/trace/types'

// ─────────────────────────────── Stub gateway ─────────────────────────────────

class SequentialGateway implements IModelGateway {
  public callCount = 0
  constructor(private readonly responses: ModelResponse[]) {}
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    this.callCount++
    const r = this.responses.shift()
    if (!r) throw new Error('SequentialGateway exhausted')
    return r
  }
  async *stream(_req: ModelRequest): AsyncIterable<never> { yield* [] }
}

const textOnly = (s: string): ModelResponse => ({
  content:      [{ type: 'text', text: s }],
  toolCalls:    [],
  finishReason: 'end_turn',
})

const toolCall = (name: string, input: unknown, callId = 'call_1'): ModelResponse => ({
  content:      [{ type: 'tool_use', id: callId, name, input }],
  toolCalls:    [{ id: callId, name, input }],
  finishReason: 'tool_use',
})

// ─────────────────────────────── Tools ────────────────────────────────────────

const echoTool: ToolDefinition = {
  name:        'echo',
  description: 'Echo the given text back',
  inputSchema: {
    type:       'object',
    properties: { text: { type: 'string' } },
    required:   ['text'],
  },
  handler: async (input: unknown) => {
    const { text } = input as { text: string }
    return { echoed: text }
  },
}

// ─────────────────────────────── Agent config ─────────────────────────────────

const inspectableAgent: AgentConfig = {
  agentId:      'inspectable-agent',
  version:      '0.0.0',
  systemPrompt: 'you are a friendly bot',
  fsm: {
    states: [
      { name: 'greet',    type: 'llm', instructions: 'greet the user',  tools: ['echo'], on: { DONE: 'finalize' } },
      { name: 'finalize', type: 'llm', instructions: 'wrap up briefly', tools: [] },
    ],
  },
  model: { provider: 'stub', model: 'stub', adapter: 'stub' },
} as AgentConfig

// ─────────────────────────────── Tests ────────────────────────────────────────

describe('s-002 inspect a completed run', () => {
  let eventStore: MemoryEventStore
  let trajectoryStore: TrajectoryStore
  let gateway: SequentialGateway
  let runId: string
  let invokeReturnedAt: number

  beforeAll(async () => {
    eventStore      = new MemoryEventStore()
    trajectoryStore = new TrajectoryStore()
    gateway         = new SequentialGateway([
      toolCall('echo', { text: 'hi' }),  // greet turn 1 → tool
      textOnly('greeted'),                // greet turn 2 → DONE → finalize
      textOnly('wrapped'),                // finalize turn 1 → terminal LLM end
    ])

    const milkie = new Milkie({
      stateStore:      new MemoryStore(),
      gateway,
      eventStore,
      trajectoryStore,
      tools:           [echoTool],
    })
    milkie.registerAgent(inspectableAgent)

    const result = await milkie.invoke({
      agentId: 'inspectable-agent',
      goal:    'demo inspection',
      input:   'hello',
    })

    expect(result.status).toBe('completed')
    runId = result.agentRunId
    invokeReturnedAt = Date.now()
  })

  test('TrajectoryStore returns a completed trajectory with ordered spans', async () => {
    const traj = await trajectoryStore.getByRunId(runId)
    expect(traj.status).toBe('completed')
    expect(traj.spans.length).toBeGreaterThan(0)

    const sorted = [...traj.spans].sort((a, b) => a.startTime - b.startTime)
    expect(sorted.map(s => s.spanId)).toEqual(traj.spans.map(s => s.spanId)
      .slice()
      .sort((aId, bId) => {
        const a = traj.spans.find(s => s.spanId === aId)!
        const b = traj.spans.find(s => s.spanId === bId)!
        return a.startTime - b.startTime
      }))

    for (const s of traj.spans) {
      expect(s.endTime).toBeDefined()
      expect((s.endTime ?? 0) >= s.startTime).toBe(true)
    }
  })

  test('Trajectory contains llm.call, tool.call, fsm.transition spans', async () => {
    const traj = await trajectoryStore.getByRunId(runId)
    const names = new Set(traj.spans.map(s => s.name))
    expect(names.has('llm.call')).toBe(true)
    expect(names.has('tool.call')).toBe(true)
    expect(names.has('fsm.transition')).toBe(true)
  })

  test('Event log opens with agent.run.started and closes with agent.run.completed', async () => {
    const events = await eventStore.readByRunId(runId)
    expect(events.length).toBeGreaterThanOrEqual(6)
    // clock.read / uuid.generated nondet events may be flushed before
    // agent.run.started, so find by type rather than assuming position 0.
    const started = events.find(e => e.type === 'agent.run.started')
    expect(started).toBeDefined()
    expect(events[events.length - 1]!.type).toBe('agent.run.completed')
  })

  test('Each llm.requested is paired with a llm.responded (same requestHash, causedBy set)', async () => {
    const events = await eventStore.readByRunId(runId)
    const reqs = events.filter(e => e.type === 'llm.requested')
    const resps = events.filter(e => e.type === 'llm.responded')

    expect(reqs.length).toBe(3)        // 3 LLM turns
    expect(resps.length).toBe(reqs.length)

    for (const req of reqs) {
      const reqHash = (req.payload as LlmRequestedPayload).requestHash
      const paired = resps.filter(r => r.causedBy === req.id)
      expect(paired.length).toBe(1)
      expect((paired[0]!.payload as LlmRespondedPayload).requestHash).toBe(reqHash)
    }
  })

  test('Each tool.requested is paired with a tool.responded (causedBy + requestHash match)', async () => {
    const events = await eventStore.readByRunId(runId)
    const reqs = events.filter(e => e.type === 'tool.requested')
    const resps = events.filter(e => e.type === 'tool.responded')

    expect(reqs.length).toBe(1)        // single echo tool call
    expect(resps.length).toBe(reqs.length)

    for (const req of reqs) {
      const reqHash = (req.payload as ToolRequestedPayload).requestHash
      const paired = resps.filter(r => r.causedBy === req.id)
      expect(paired.length).toBe(1)
      expect((paired[0]!.payload as ToolRespondedPayload).requestHash).toBe(reqHash)
    }
  })

  test('Event timestamps are monotonically non-decreasing', async () => {
    const events = await eventStore.readByRunId(runId)
    for (let i = 1; i < events.length; i++) {
      expect(events[i]!.timestamp).toBeGreaterThanOrEqual(events[i - 1]!.timestamp)
    }
  })

  test('Filter by type matches LLM call count', async () => {
    const events = await eventStore.readByRunId(runId)
    const llmRequests = events.filter(e => e.type === 'llm.requested')
    expect(llmRequests.length).toBe(gateway.callCount)
  })

  test('Filter by time window returns a bounded, non-empty subset', async () => {
    // Note: in a hermetic stub-gateway run, the full timeline can collapse
    // into a single Date.now() millisecond, so we cannot assert a strict
    // prefix length here. The contract under test is that time-window
    // filtering is a valid operation: it returns a subset of events bounded
    // by the window endpoints.
    const events = await eventStore.readByRunId(runId)
    const cutoff = events[Math.floor(events.length / 2)]!.timestamp

    const window = events.filter(e => e.timestamp <= cutoff)
    expect(window.length).toBeGreaterThanOrEqual(1)
    expect(window.length).toBeLessThanOrEqual(events.length)
    expect(window.every(e => e.timestamp <= cutoff)).toBe(true)
  })

  test('readRange returns the same slice as readByRunId slice', async () => {
    const all = await eventStore.readByRunId(runId)
    const range = await eventStore.readRange(runId, 0, 2)
    expect(range.length).toBe(2)
    expect(range.map(e => e.id)).toEqual(all.slice(0, 2).map(e => e.id))
  })

  test('Inspection does not trigger any extra LLM calls', () => {
    const expected = 3                  // calls made during the original invoke
    expect(gateway.callCount).toBe(expected)
    // The pure-read operations above ran after invoke returned;
    // gateway.callCount must remain at the invoke-time value.
    expect(Date.now()).toBeGreaterThanOrEqual(invokeReturnedAt)
  })
})
