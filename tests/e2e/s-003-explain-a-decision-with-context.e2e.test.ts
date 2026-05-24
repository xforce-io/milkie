/**
 * s-003: Explain an agent decision with its full context.
 *
 * For a chosen LLM decision in a recorded run, reconstruct the moment:
 *   - the prompt sent (system, messages = working context projection, tools)
 *   - the response received (content + structured toolCalls)
 *   - the tool calls that decision induced (matched by toolName + input)
 *   - the matching tool.responded for each
 *
 * Pure read, no replay. Uses a stub gateway so the test is hermetic.
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
  public lastRequest: ModelRequest | null = null
  constructor(private readonly responses: ModelResponse[]) {}
  async complete(req: ModelRequest): Promise<ModelResponse> {
    this.callCount++
    this.lastRequest = req
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

const searchCall = (query: string, callId = 'call_search_1'): ModelResponse => ({
  content:      [{ type: 'tool_use', id: callId, name: 'web_search', input: { query } }],
  toolCalls:    [{ id: callId, name: 'web_search', input: { query } }],
  finishReason: 'tool_use',
})

// ─────────────────────────────── Tools ────────────────────────────────────────

const webSearchTool: ToolDefinition = {
  name:        'web_search',
  description: 'Search the web for a given query',
  inputSchema: {
    type:       'object',
    properties: { query: { type: 'string' } },
    required:   ['query'],
  },
  handler: async (input: unknown) => {
    const { query } = input as { query: string }
    return { results: [`stub result for "${query}"`] }
  },
}

// ─────────────────────────────── Agent config ─────────────────────────────────

const planAgent: AgentConfig = {
  agentId:      'plan-agent',
  version:      '0.0.0',
  systemPrompt: 'you are a research assistant',
  fsm: {
    states: [
      { name: 'plan',  type: 'llm', instructions: 'pick a search to run', tools: ['web_search'], on: { DONE: 'close' } },
      { name: 'close', type: 'llm', instructions: 'wrap up briefly',      tools: [] },
    ],
  },
  model: { provider: 'stub', model: 'stub', adapter: 'stub' },
} as AgentConfig

// ─────────────────────────────── Tests ────────────────────────────────────────

describe('s-003 explain a decision with context', () => {
  let eventStore: MemoryEventStore
  let trajectoryStore: TrajectoryStore
  let gateway: SequentialGateway
  let runId: string
  let callCountAtInvokeEnd: number

  const searchQuery = 'milkie agent framework'

  beforeAll(async () => {
    eventStore      = new MemoryEventStore()
    trajectoryStore = new TrajectoryStore()
    gateway         = new SequentialGateway([
      searchCall(searchQuery),  // plan turn 1 → web_search tool
      textOnly('search done'),  // plan turn 2 → DONE → close
      textOnly('all wrapped'),  // close turn 1 → terminal end
    ])

    const milkie = new Milkie({
      stateStore:      new MemoryStore(),
      gateway,
      eventStore,
      trajectoryStore,
      tools:           [webSearchTool],
    })
    milkie.registerAgent(planAgent)

    const result = await milkie.invoke({
      agentId: 'plan-agent',
      goal:    'research milkie',
      input:   'please look it up',
    })

    expect(result.status).toBe('completed')
    runId = result.agentRunId
    callCountAtInvokeEnd = gateway.callCount
  })

  test('Decision event has a non-empty messages array including a user message', async () => {
    const events = await eventStore.readByRunId(runId)
    const firstLlmReq = events.find(e => e.type === 'llm.requested')!
    const req = (firstLlmReq.payload as LlmRequestedPayload).request

    expect(req.messages.length).toBeGreaterThan(0)
    const hasUser = req.messages.some(m => m.role === 'user')
    expect(hasUser).toBe(true)
  })

  test('Decision event lists web_search in the available tools', async () => {
    const events = await eventStore.readByRunId(runId)
    const firstLlmReq = events.find(e => e.type === 'llm.requested')!
    const req = (firstLlmReq.payload as LlmRequestedPayload).request

    expect(req.tools).toBeDefined()
    expect(req.tools!.length).toBeGreaterThan(0)
    expect(req.tools!.some(t => t.name === 'web_search')).toBe(true)
  })

  test('Decision event has exactly one paired llm.responded (causedBy + requestHash)', async () => {
    const events = await eventStore.readByRunId(runId)
    const firstLlmReq = events.find(e => e.type === 'llm.requested')!
    const reqHash = (firstLlmReq.payload as LlmRequestedPayload).requestHash

    const paired = events.filter(e =>
      e.type === 'llm.responded' && e.causedBy === firstLlmReq.id
    )
    expect(paired.length).toBe(1)
    expect((paired[0]!.payload as LlmRespondedPayload).requestHash).toBe(reqHash)
  })

  test('The response carries the structured tool call (web_search)', async () => {
    const events = await eventStore.readByRunId(runId)
    const firstLlmReq = events.find(e => e.type === 'llm.requested')!
    const resp = events.find(e =>
      e.type === 'llm.responded' && e.causedBy === firstLlmReq.id
    )!
    const response = (resp.payload as LlmRespondedPayload).response

    expect(response.toolCalls.length).toBeGreaterThan(0)
    expect(response.toolCalls[0]!.name).toBe('web_search')
    expect((response.toolCalls[0]!.input as { query: string }).query).toBe(searchQuery)
  })

  test('Tool call induced by the decision has matching toolName + input + paired response', async () => {
    const events = await eventStore.readByRunId(runId)
    const firstLlmReq = events.find(e => e.type === 'llm.requested')!
    const llmResp = events.find(e =>
      e.type === 'llm.responded' && e.causedBy === firstLlmReq.id
    )!
    const respIdx = events.indexOf(llmResp)

    // The tool.requested triggered by this decision must come after the response
    const tail = events.slice(respIdx + 1)
    const toolReq = tail.find(e => e.type === 'tool.requested')!
    expect(toolReq).toBeDefined()

    const toolPayload = toolReq.payload as ToolRequestedPayload
    expect(toolPayload.toolName).toBe('web_search')
    expect((toolPayload.input as { query: string }).query).toBe(searchQuery)

    const toolResp = events.find(e =>
      e.type === 'tool.responded' && e.causedBy === toolReq.id
    )!
    expect(toolResp).toBeDefined()
    const toolRespPayload = toolResp.payload as ToolRespondedPayload
    expect(toolRespPayload.output).toBeDefined()
    expect(toolRespPayload.error).toBeUndefined()
  })

  test('Trajectory contains an llm.call span aligned with the decision turn', async () => {
    const traj = await trajectoryStore.getByRunId(runId)
    const llmSpans = traj.spans.filter(s => s.name === 'llm.call')
    expect(llmSpans.length).toBeGreaterThan(0)
    // The first llm.call span belongs to the first decision (plan turn 1)
    const firstSpan = [...llmSpans].sort((a, b) => a.startTime - b.startTime)[0]!
    expect(firstSpan.attributes['turn']).toBeDefined()
  })

  test('Explaining a decision does not trigger any extra LLM calls', () => {
    expect(gateway.callCount).toBe(callCountAtInvokeEnd)
  })
})
