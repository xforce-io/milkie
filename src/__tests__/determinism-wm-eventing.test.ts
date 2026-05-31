// #73 Stage 3: tool working-memory side-effects are event-sourced (wm.mutated)
// so replay — which does NOT re-run tool handlers — reconstructs the exact WM
// state at every LLM call. Comprehensive matrix over WM-mutation patterns.
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { cognitiveTools } from '../tools/cognitive'
import type { ToolDefinition } from '../types/tool'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { AgentConfig } from '../types/agent'

type Step = { tool: string; input: unknown } | { done: string }

// Drives a scripted ReAct run: each step is one LLM response (a tool call or
// the final answer). A step may carry MULTIPLE tool calls (parallel batch).
type Multi = { tools: Array<{ tool: string; input: unknown }> }
class ScriptedGateway implements IModelGateway {
  private i = 0
  constructor(private readonly steps: Array<Step | Multi>) {}
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    const step = this.steps[this.i++]
    if (!step || 'done' in step) {
      return { content: [{ type: 'text', text: step && 'done' in step ? step.done : 'done' }], toolCalls: [], finishReason: 'end_turn' }
    }
    const calls = 'tools' in step
      ? step.tools.map((t, j) => ({ id: `c${this.i}_${j}`, name: t.tool, input: t.input }))
      : [{ id: `c${this.i}`, name: step.tool, input: step.input }]
    return {
      content:   calls.map(c => ({ type: 'tool_use' as const, id: c.id, name: c.name, input: c.input })),
      toolCalls: calls,
      finishReason: 'tool_use',
    }
  }
  async *stream(_r: ModelRequest): AsyncIterable<never> { yield* [] }
}

const agent: AgentConfig = {
  agentId: 'a', version: '1.0.0', systemPrompt: 'sys',
  fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 20 }] },
  model: { provider: 'test', model: 'test', adapter: 'test' },
}

async function runAndReplay(steps: Array<Step | Multi>, extraTools: ToolDefinition[] = []) {
  const eventStore = new MemoryEventStore()
  const milkie = new Milkie({
    stateStore: new MemoryStore(),
    eventStore,
    gateway: new ScriptedGateway(steps),
    tools: [...cognitiveTools, ...extraTools],
  })
  milkie.registerAgent(agent)
  const run = await milkie.invoke({ agentId: 'a', goal: 'g', input: 'go', contextId: 'c' })
  const events = await eventStore.readByRunId(run.agentRunId)
  const replayed = await milkie.replay(run.agentRunId)  // throws ReplayDivergenceError on any divergence
  return { run, replayed, events }
}

describe('#73 Stage 3: WM side-effects are event-sourced for deterministic replay', () => {
  it('create_plan + think replays deterministically', async () => {
    const { run, replayed } = await runAndReplay([
      { tool: 'create_plan', input: { steps: ['a', 'b'] } },
      { tool: 'think', input: { thoughts: 'reasoning here' } },
      { done: 'final' },
    ])
    expect(run.status).toBe('completed')
    expect(replayed.output).toBe(run.output)
  })

  it('update_step (in-place plan mutation) replays deterministically', async () => {
    const { replayed, run } = await runAndReplay([
      { tool: 'create_plan', input: { steps: ['x', 'y'] } },
      { tool: 'update_step', input: { stepId: 0, status: 'done' } },
      { done: 'ok' },
    ])
    expect(replayed.output).toBe(run.output)
  })

  it('multiple think calls (growing log) replay deterministically', async () => {
    const { replayed, run } = await runAndReplay([
      { tool: 'think', input: { thoughts: 'one' } },
      { tool: 'think', input: { thoughts: 'two' } },
      { tool: 'think', input: { thoughts: 'three' } },
      { done: 'done' },
    ])
    expect(replayed.output).toBe(run.output)
  })

  it('a run whose tools do NOT touch WM still replays deterministically', async () => {
    const pure: ToolDefinition = {
      name: 'echo', description: 'echo', inputSchema: { type: 'object', properties: { x: { type: 'string' } }, required: ['x'] },
      handler: async (input: unknown) => ({ echoed: (input as { x: string }).x }),
    }
    const { replayed, run } = await runAndReplay([
      { tool: 'echo', input: { x: 'hi' } },
      { done: 'done' },
    ], [pure])
    expect(replayed.output).toBe(run.output)
  })

  it('parallel WM-writing tools (a parallel-safe batch) replay deterministically', async () => {
    // `think` is parallelSafe → two thinks in one LLM response run as a batch.
    const { replayed, run } = await runAndReplay([
      { tools: [{ tool: 'think', input: { thoughts: 'alpha' } }, { tool: 'think', input: { thoughts: 'beta' } }] },
      { done: 'done' },
    ])
    expect(replayed.output).toBe(run.output)
  })

  it('records one wm.mutated event per tool call, each a frozen snapshot of WM at that point', async () => {
    const { events } = await runAndReplay([
      { tool: 'create_plan', input: { steps: ['a'] } },  // wm now has plan
      { tool: 'think', input: { thoughts: 't' } },         // wm now has plan + log
      { done: 'd' },
    ])
    const wm = events.filter(e => e.type === 'wm.mutated').map(e => (e.payload as { snapshot: { data: Record<string, unknown>; log: unknown[] } }).snapshot)
    expect(wm).toHaveLength(2)
    // Snapshot #1 (after create_plan): plan present, log still empty — proves
    // the snapshot was frozen and did NOT pick up the later think write.
    expect(wm[0]!.data).toHaveProperty('plan')
    expect(wm[0]!.log).toHaveLength(0)
    // Snapshot #2 (after think): plan still present, log now has the thought.
    expect(wm[1]!.data).toHaveProperty('plan')
    expect(wm[1]!.log).toHaveLength(1)
  })
})
