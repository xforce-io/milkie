import { makeTraceTools } from '../tools/trace-tools'
import { MemoryEventStore } from '../../../src/trace/MemoryEventStore'
import { MemoryTraceObjectStore } from '../../../src/trace/TraceObjectStore'
import type { ToolContext } from '../../../src/types/tool'

const CTX = {} as ToolContext  // these tools do not use ctx

async function seedRun(store: MemoryEventStore, runId: string) {
  await store.append({ id: 's', runId, type: 'agent.run.started', actor: 'a', timestamp: 1,
    payload: { agentId: 'x', goal: 'g', input: '曹操爸爸是谁', contextId: runId } })
  await store.append({ id: 'c', runId, type: 'agent.run.completed', actor: 'a', timestamp: 9,
    payload: { status: 'completed', lastTextOutput: '赤壁之战发生在公元208年。' } })
}

describe('makeTraceTools: get_run_io', () => {
  it('returns the user question and final answer of a run', async () => {
    const store = new MemoryEventStore()
    await seedRun(store, 'target-1')
    const tools = makeTraceTools(store, new MemoryTraceObjectStore())
    const getRunIo = tools.find(t => t.name === 'get_run_io')!
    const out = await getRunIo.handler({ runId: 'target-1' }, CTX)
    expect(out).toEqual({ question: '曹操爸爸是谁', finalAnswer: '赤壁之战发生在公元208年。' })
  })
})

describe('makeTraceTools: get_execution', () => {
  it('returns the execution projection (steps with tool query) of a run', async () => {
    const store = new MemoryEventStore()
    const runId = 'target-2'
    await store.append({ id: 's', runId, type: 'agent.run.started', actor: 'a', timestamp: 1,
      payload: { agentId: 'x', goal: 'g', input: '曹操爸爸是谁', contextId: runId } })
    await store.append({ id: 'lq', runId, type: 'llm.requested', actor: 'a', timestamp: 2,
      payload: { request: { model: 'm', messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }] }, requestHash: 'h1' } })
    await store.append({ id: 'lr', runId, type: 'llm.responded', actor: 'a', timestamp: 3, causedBy: 'lq',
      payload: { response: { content: [], toolCalls: [], finishReason: 'tool_use' }, requestHash: 'h1' } })
    await store.append({ id: 'tq', runId, type: 'tool.requested', actor: 'a', timestamp: 4,
      payload: { toolName: 'grep', input: { pattern: '赤壁' }, requestHash: 'h2' } })
    await store.append({ id: 'tr', runId, type: 'tool.responded', actor: 'a', timestamp: 5, causedBy: 'tq',
      payload: { toolName: 'grep', output: { matches: [] }, requestHash: 'h2' } })

    const tools = makeTraceTools(store, new MemoryTraceObjectStore())
    const getExec = tools.find(t => t.name === 'get_execution')!
    const proj = await getExec.handler({ runId }, CTX) as { steps: Array<{ kind: string; tool?: { name: string; input: unknown } }> }
    const toolStep = proj.steps.find(s => s.kind === 'tool')
    expect(toolStep?.tool).toMatchObject({ name: 'grep', input: { pattern: '赤壁' } })
  })
})
