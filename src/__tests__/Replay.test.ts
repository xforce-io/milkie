import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { ReplayError } from '../trace/ReplayError'
import { ReplayDivergenceError } from '../trace/ReplayDivergenceError'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { AgentConfig } from '../types/agent'
import type { ToolDefinition } from '../types/tool'

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

const text = (s: string): ModelResponse => ({
  content: [{ type: 'text', text: s }], toolCalls: [], finishReason: 'end_turn',
})

const oneShotAgent = (agentId = 'a1'): AgentConfig => ({
  agentId,
  version: '0.0.0',
  systemPrompt: 'sys',
  fsm: {
    states: [
      { name: 'react', type: 'llm', instructions: 'say hi', tools: [] },
    ],
  },
  model: { provider: 'stub', model: 'stub', adapter: 'stub' },
})

describe('Milkie.replay', () => {
  it('replays a recorded run with identical result and zero LLM calls', async () => {
    const store = new MemoryEventStore()
    const replayGateway = new SequentialGateway([text('this would be wrong')])
    // First run records
    const recordGateway = new SequentialGateway([text('hello world')])
    const recordMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: recordGateway, eventStore: store })
    recordMilkie.registerAgent(oneShotAgent())
    const original = await recordMilkie.invoke({ agentId: 'a1', goal: 'g', input: 'i' })

    // Replay reuses the same store + agent config but a different gateway
    const replayMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: replayGateway, eventStore: store })
    replayMilkie.registerAgent(oneShotAgent())
    const replayed = await replayMilkie.replay(original.agentRunId)

    expect(replayed.status).toBe(original.status)
    expect(replayed.output).toBe(original.output)
    expect(replayGateway.callCount).toBe(0)  // cache served everything
  })

  it('throws ReplayError when runId has no events', async () => {
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway:    new SequentialGateway([]),
      eventStore: new MemoryEventStore(),
    })
    await expect(milkie.replay('nonexistent')).rejects.toBeInstanceOf(ReplayError)
  })

  it('throws ReplayError when run has no lifecycle start (Phase 2 run)', async () => {
    const store = new MemoryEventStore()
    // Manually append only an llm.responded — no agent.run.started
    await store.append({
      id: 'e1', runId: 'r-old', type: 'llm.responded', actor: 'runtime', timestamp: 1,
      payload: { response: text('x'), requestHash: 'h' },
    })
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway:    new SequentialGateway([]),
      eventStore: store,
    })
    await expect(milkie.replay('r-old')).rejects.toThrow(/no lifecycle start/)
  })

  it('throws ReplayError when agentId is not registered', async () => {
    const store = new MemoryEventStore()
    const recordMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: new SequentialGateway([text('x')]), eventStore: store })
    recordMilkie.registerAgent(oneShotAgent('a1'))
    const original = await recordMilkie.invoke({ agentId: 'a1', goal: 'g', input: 'i' })

    const replayMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: new SequentialGateway([]), eventStore: store })
    // intentionally do NOT register a1
    await expect(replayMilkie.replay(original.agentRunId)).rejects.toThrow(/not registered/)
  })

  it('throws ReplayDivergenceError when replay agent diverges from recorded I/O', async () => {
    const store = new MemoryEventStore()
    const recordGateway = new SequentialGateway([text('original')])
    const recordMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: recordGateway, eventStore: store })
    recordMilkie.registerAgent(oneShotAgent())
    const original = await recordMilkie.invoke({ agentId: 'a1', goal: 'g', input: 'i' })

    // Replay registers a *changed* agent so the LLM request differs
    const replayMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: new SequentialGateway([]), eventStore: store })
    const mutated = oneShotAgent()
    ;(mutated.fsm.states[0] as { instructions?: string }).instructions = 'say goodbye'  // changes ModelRequest → hash mismatch
    replayMilkie.registerAgent(mutated)

    await expect(replayMilkie.replay(original.agentRunId)).rejects.toBeInstanceOf(ReplayDivergenceError)
  })

  it('does not write new events to the event store during replay (I7)', async () => {
    const store = new MemoryEventStore()
    const recordMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: new SequentialGateway([text('once')]), eventStore: store })
    recordMilkie.registerAgent(oneShotAgent())
    const original = await recordMilkie.invoke({ agentId: 'a1', goal: 'g', input: 'i' })

    const before = (await store.readByRunId(original.agentRunId)).length
    const replayMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: new SequentialGateway([]), eventStore: store })
    replayMilkie.registerAgent(oneShotAgent())
    await replayMilkie.replay(original.agentRunId)
    const after = (await store.readByRunId(original.agentRunId)).length

    expect(after).toBe(before)
  })

  // ── Test #1: Tool retry equivalence (I8 at integration level) ────────────
  //
  // Verifies that if the recorded run's tool threw a retryable error and the
  // AgentRuntime retry loop retried and succeeded, the replayed run takes the
  // same retry path — driven by the cached error payload carrying retryable:true.
  // The live tool handler must never be invoked during replay.
  it('replays a tool-retry run: retryable error then success served from cache (I8)', async () => {
    let liveCallCount = 0

    // The live tool: fails on first call (retryable), succeeds on second call.
    const retryableTool: ToolDefinition = {
      name: 'flaky_tool',
      description: 'A tool that fails once then succeeds',
      parallelSafe: false,
      inputSchema: {
        type: 'object',
        properties: { value: { type: 'string' } },
        required: ['value'],
      },
      handler: async (_input: unknown) => {
        liveCallCount++
        if (liveCallCount === 1) {
          const err = new Error('transient failure') as Error & { retryable: boolean }
          err.retryable = true
          throw err
        }
        return 'tool-success'
      },
    }

    // Agent: one LLM state that calls the tool then receives a text response.
    // The LLM emits toolCalls first, then a plain text response after tool result.
    const agentWithTool: AgentConfig = {
      agentId: 'retry-agent',
      version: '0.0.0',
      systemPrompt: 'sys',
      fsm: {
        states: [
          { name: 's0', type: 'llm', instructions: 'use flaky_tool', tools: ['flaky_tool'], on: { DONE: 'end' } },
          { name: 'end', type: 'action', terminal: true },
        ],
      },
      model: { provider: 'stub', model: 'stub', adapter: 'stub' },
    }

    // LLM sequence: (1) call tool, (2) text response after tool result
    const toolCallResponse: ModelResponse = {
      content: [],
      toolCalls: [{ id: 'tc-retry', name: 'flaky_tool', input: { value: 'x' } }],
      finishReason: 'tool_use',
    }
    const finalTextResponse: ModelResponse = text('retry-success-output')

    const store = new MemoryEventStore()
    const recordGateway = new SequentialGateway([toolCallResponse, finalTextResponse])
    const recordMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: recordGateway, eventStore: store, tools: [retryableTool] })
    recordMilkie.registerAgent(agentWithTool)
    const original = await recordMilkie.invoke({ agentId: 'retry-agent', goal: 'g', input: 'i' })

    expect(original.status).toBe('completed')
    expect(original.output).toBe('retry-success-output')
    // The live tool was called twice (retryable fail + success)
    expect(liveCallCount).toBe(2)

    // Verify the event log has two tool.responded events for flaky_tool:
    // first with error (retryable), second with output.
    const events = await store.readByRunId(original.agentRunId)
    const toolResponded = events.filter(e => e.type === 'tool.responded')
    expect(toolResponded.length).toBe(2)
    const firstToolEvent = toolResponded[0]!.payload as { error?: { retryable?: boolean }; output?: unknown }
    const secondToolEvent = toolResponded[1]!.payload as { error?: { retryable?: boolean }; output?: unknown }
    expect(firstToolEvent.error?.retryable).toBe(true)
    expect(secondToolEvent.output).toBe('tool-success')

    // Replay: replace the live tool with a stub that THROWS if called — proving
    // replay serves everything from cache and never invokes the live handler.
    const replayStub: ToolDefinition = {
      ...retryableTool,
      handler: async () => { throw new Error('live tool invoked during replay — must not happen') },
    }

    const replayGateway = new SequentialGateway([text('this-would-be-wrong')])
    const replayMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: replayGateway, eventStore: store, tools: [replayStub] })
    replayMilkie.registerAgent(agentWithTool)
    const replayed = await replayMilkie.replay(original.agentRunId)

    expect(replayed.status).toBe(original.status)
    expect(replayed.output).toBe(original.output)
    expect(replayGateway.callCount).toBe(0)
  }, 10_000)  // allow up to 10s for the 500ms retry delay × 2 runs

  // ── Test #2: Same-hash multi-response served in FIFO order ───────────────
  //
  // Verifies that when the same LLM request hash appears multiple times in
  // the event log, replay serves them in FIFO order (not by overwriting).
  // We construct this scenario directly at the boundary: manually append two
  // llm.responded events with identical requestHash to the store, then replay.
  it('serves same-hash LLM responses in FIFO order during replay', async () => {
    const store = new MemoryEventStore()

    // First do a normal record run so we get a valid agent.run.started event
    // and a known agentId/goal/input in the snapshot.
    const recordGateway = new SequentialGateway([text('first-response')])
    const recordMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: recordGateway, eventStore: store })
    recordMilkie.registerAgent(oneShotAgent('fifo-agent'))
    const original = await recordMilkie.invoke({ agentId: 'fifo-agent', goal: 'fifo-goal', input: 'fifo-input' })

    // Capture the requestHash from the recorded llm.responded event
    const events = await store.readByRunId(original.agentRunId)
    const firstLlmResponded = events.find(e => e.type === 'llm.responded')!
    const originalHash = (firstLlmResponded.payload as { requestHash: string }).requestHash

    // Now inject a second llm.responded with the SAME hash but DIFFERENT response
    // (after the existing one) — simulates a same-hash second entry in the FIFO queue.
    // We do this by appending directly to the store.
    await store.append({
      id: 'injected-1',
      runId: original.agentRunId,
      type: 'llm.responded',
      actor: 'test',
      timestamp: Date.now(),
      payload: {
        response: text('second-response'),
        requestHash: originalHash,
      },
    })

    // The store now has two llm.responded events for the same hash:
    //   [0] → text('first-response')  (recorded)
    //   [1] → text('second-response') (injected)
    //
    // Replay should consume index [0] first (FIFO), returning 'first-response'.
    const replayGateway = new SequentialGateway([])
    const replayMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: replayGateway, eventStore: store })
    replayMilkie.registerAgent(oneShotAgent('fifo-agent'))
    const replayed = await replayMilkie.replay(original.agentRunId)

    // FIFO: first response wins
    expect(replayed.status).toBe('completed')
    expect(replayed.output).toBe('first-response')
    expect(replayGateway.callCount).toBe(0)
  })

  // ── Test #3: LLM response with toolCalls is fully reconstructed in replay ─
  //
  // Verifies that ReplayingIOPort correctly serves a cached ModelResponse
  // whose toolCalls array is non-empty. The full toolCalls structure must be
  // reconstructed from the cache (not silently dropped or truncated).
  it('replays a run where the cached LLM response carries a non-empty toolCalls array', async () => {
    // A simple tool that returns a fixed value
    const simpleTool: ToolDefinition = {
      name: 'mytool',
      description: 'A simple test tool',
      parallelSafe: false,
      inputSchema: {
        type: 'object',
        properties: { x: { type: 'number' } },
        required: ['x'],
      },
      handler: async (_input: unknown) => {
        return 'tool-output-42'
      },
    }

    const agentWithTool: AgentConfig = {
      agentId: 'toolcall-agent',
      version: '0.0.0',
      systemPrompt: 'sys',
      fsm: {
        states: [
          { name: 's0', type: 'llm', instructions: 'use mytool then respond', tools: ['mytool'], on: { DONE: 'end' } },
          { name: 'end', type: 'action', terminal: true },
        ],
      },
      model: { provider: 'stub', model: 'stub', adapter: 'stub' },
    }

    // LLM returns toolCalls first, then a plain text response after the tool result
    const toolCallResponse: ModelResponse = {
      content: [],
      toolCalls: [{ id: 'tc1', name: 'mytool', input: { x: 1 } }],
      finishReason: 'tool_use',
    }
    const finalText: ModelResponse = text('toolcall-final-output')

    const store = new MemoryEventStore()
    const recordGateway = new SequentialGateway([toolCallResponse, finalText])
    const recordMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: recordGateway, eventStore: store, tools: [simpleTool] })
    recordMilkie.registerAgent(agentWithTool)
    const original = await recordMilkie.invoke({ agentId: 'toolcall-agent', goal: 'g', input: 'i' })

    expect(original.status).toBe('completed')
    expect(original.output).toBe('toolcall-final-output')

    // Replay: gateway and tool stub both throw if called — everything must come from cache
    const replayGateway = new SequentialGateway([text('wrong')])
    const replayToolStub: ToolDefinition = {
      ...simpleTool,
      handler: async () => { throw new Error('live tool invoked during replay — must not happen') },
    }

    const replayMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: replayGateway, eventStore: store, tools: [replayToolStub] })
    replayMilkie.registerAgent(agentWithTool)
    const replayed = await replayMilkie.replay(original.agentRunId)

    expect(replayed.status).toBe(original.status)
    expect(replayed.output).toBe(original.output)
    expect(replayGateway.callCount).toBe(0)
  })
})
