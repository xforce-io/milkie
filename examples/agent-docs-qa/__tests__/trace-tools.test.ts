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
