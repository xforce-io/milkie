// src/__tests__/traceTools.selfOnly.test.ts
// #196 follow-up (P1 security): the GENERIC registration (serve/constructor) must
// expose only a self-view. Reading an arbitrary runId is the diagnoser's privilege,
// granted via loadStandardAgents(), not something every eventStore-enabled agent
// should get — otherwise a shared serve instance leaks other sessions' I/O.
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { makeTraceTools } from '../tools/trace'
import type { Event } from '../trace/types'

function startedEvent(runId: string, previousRunId?: string): Event {
  return {
    id: `${runId}-s`, runId, type: 'agent.run.started', actor: 'test', timestamp: 0,
    payload: { agentId: 'a', goal: 'g', input: 'VICTIM QUESTION', contextId: 'other',
               ...(previousRunId ? { previousRunId } : {}) },
  } as Event
}
function llmReq(runId: string): Event {
  return {
    id: `${runId}-llm`, runId, type: 'llm.requested', actor: 'test', timestamp: 1,
    payload: { requestHash: 'h1', request: { system: 'SECRET PROMPT', messages: [], tools: [] } },
  } as Event
}
function completed(runId: string): Event {
  return {
    id: `${runId}-c`, runId, type: 'agent.run.completed', actor: 'test', timestamp: 4,
    payload: { lastTextOutput: 'VICTIM ANSWER' },
  } as Event
}

describe('makeTraceTools selfOnly mode (generic/serve registration)', () => {
  it('omits get_run_io entirely (it has no self meaning)', () => {
    const names = makeTraceTools(new MemoryEventStore(), undefined, { selfOnly: true }).map(t => t.name)
    expect(names).not.toContain('get_run_io')
    expect(names).toEqual(expect.arrayContaining(['get_execution', 'get_lineage']))
  })

  it('get_execution drops runId from its inputSchema', () => {
    const tool = makeTraceTools(new MemoryEventStore(), undefined, { selfOnly: true })
      .find(t => t.name === 'get_execution')!
    expect((tool.inputSchema as { properties: Record<string, unknown> }).properties.runId).toBeUndefined()
  })

  it('get_execution ignores a foreign runId — returns self window, never the target run', async () => {
    const store = new MemoryEventStore()
    for (const e of [startedEvent('victim'), llmReq('victim'), completed('victim')]) await store.append(e)
    const tool = makeTraceTools(store, undefined, { selfOnly: true }).find(t => t.name === 'get_execution')!
    // Attacker passes the victim's runId AND has no previousRunId of their own.
    const res = await tool.handler({ runId: 'victim' }, {} as never) as { turns?: unknown[]; steps?: unknown }
    expect(res.turns).toEqual([])      // self window only; first turn → empty
    expect(res.steps).toBeUndefined()  // NOT the full projection
    expect(JSON.stringify(res)).not.toContain('SECRET PROMPT')
  })

  it('get_lineage ignores a foreign runId — searches only the self window', async () => {
    const store = new MemoryEventStore()
    for (const e of [startedEvent('victim'), completed('victim')]) await store.append(e)
    const tool = makeTraceTools(store, undefined, { selfOnly: true }).find(t => t.name === 'get_lineage')!
    const res = await tool.handler({ runId: 'victim', query: 'VICTIM' }, {} as never) as { matches: unknown[] }
    expect(res.matches).toEqual([])
  })
})

describe('makeTraceTools full mode (diagnoser, default) is unchanged', () => {
  it('still exposes get_run_io and honours an explicit runId', async () => {
    const store = new MemoryEventStore()
    for (const e of [startedEvent('target'), completed('target')]) await store.append(e)
    const names = makeTraceTools(store).map(t => t.name)
    expect(names).toEqual(expect.arrayContaining(['get_run_io', 'get_execution', 'get_lineage']))
    const tool = makeTraceTools(store).find(t => t.name === 'get_run_io')!
    const res = await tool.handler({ runId: 'target' }, {} as never) as { question: string; finalAnswer: string }
    expect(res.question).toBe('VICTIM QUESTION')
    expect(res.finalAnswer).toBe('VICTIM ANSWER')
  })
})
