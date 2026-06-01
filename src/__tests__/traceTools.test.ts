import { makeTraceTools } from '../tools/trace'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { MemoryTraceObjectStore } from '../trace/TraceObjectStore'
import type { ToolContext } from '../types/tool'

const CTX = {} as ToolContext

async function seed(store: MemoryEventStore, runId: string) {
  await store.append({ id: 's', runId, type: 'agent.run.started', actor: 'a', timestamp: 1,
    payload: { agentId: 'x', goal: 'g', input: 'Q', contextId: runId } })
  await store.append({ id: 'lq', runId, type: 'llm.requested', actor: 'a', timestamp: 2,
    payload: { request: { model: 'm', messages: [] }, requestHash: 'h1' } })
  await store.append({ id: 'lr', runId, type: 'llm.responded', actor: 'a', timestamp: 3, causedBy: 'lq',
    payload: { response: { content: [], toolCalls: [], finishReason: 'end_turn' }, requestHash: 'h1' } })
  await store.append({ id: 'c', runId, type: 'agent.run.completed', actor: 'a', timestamp: 9,
    payload: { status: 'completed', lastTextOutput: 'A' } })
}

describe('makeTraceTools (src)', () => {
  it('get_run_io returns question + finalAnswer', async () => {
    const store = new MemoryEventStore(); await seed(store, 'r1')
    const t = makeTraceTools(store, new MemoryTraceObjectStore()).find(t => t.name === 'get_run_io')!
    expect(await t.handler({ runId: 'r1' }, CTX)).toEqual({ question: 'Q', finalAnswer: 'A' })
  })

  it('get_execution tolerates an undefined objectStore and still returns steps', async () => {
    const store = new MemoryEventStore(); await seed(store, 'r1')
    const t = makeTraceTools(store, undefined).find(t => t.name === 'get_execution')!
    const proj = await t.handler({ runId: 'r1' }, CTX) as { steps: unknown[] }
    expect(Array.isArray(proj.steps)).toBe(true)
    expect(proj.steps.length).toBeGreaterThan(0)
  })
})
