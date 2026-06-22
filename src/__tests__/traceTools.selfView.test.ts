import { MemoryEventStore } from '../trace/MemoryEventStore'
import { makeTraceTools } from '../tools/trace'
import type { Event } from '../trace/types'

function startedEvent(runId: string, previousRunId?: string): Event {
  return {
    id: `${runId}-s`, runId, type: 'agent.run.started', actor: 'test', timestamp: 0,
    payload: { agentId: 'a', goal: 'g', input: 'i', contextId: 'c',
               ...(previousRunId ? { previousRunId } : {}) },
  } as Event
}
function llmReq(runId: string): Event {
  return {
    id: `${runId}-llm`, runId, type: 'llm.requested', actor: 'test', timestamp: 1,
    payload: { requestHash: 'h1', request: { system: 'SECRET PROMPT', messages: [], tools: [] } },
  } as Event
}
function toolReq(runId: string, name: string): Event {
  return {
    id: `${runId}-tr`, runId, type: 'tool.requested', actor: 'test', timestamp: 2,
    payload: { requestHash: 'h2', toolName: name, input: { query: 'q' } },
  } as Event
}
function toolResp(runId: string): Event {
  return {
    id: `${runId}-tp`, runId, type: 'tool.responded', actor: 'test', timestamp: 3,
    payload: { requestHash: 'h2', toolName: 'fetch_news', output: { hits: 37 } },
  } as Event
}

describe('#189 get_execution self view', () => {
  it('runId omitted → windowed tool steps, no prompt bodies', async () => {
    const store = new MemoryEventStore()
    for (const e of [startedEvent('run1'), llmReq('run1'), toolReq('run1', 'fetch_news'), toolResp('run1')]) await store.append(e)
    const tool = makeTraceTools(store).find(t => t.name === 'get_execution')!
    const res = await tool.handler({}, { previousRunId: 'run1' } as never) as { turns: { runId: string; toolSteps: unknown[]; llmStepCount: number }[] }
    expect(res.turns).toHaveLength(1)
    const turn0 = res.turns[0]!
    expect(turn0.runId).toBe('run1')
    expect(turn0.toolSteps).toHaveLength(1)
    expect(turn0.llmStepCount).toBe(1)
    expect(JSON.stringify(turn0)).not.toContain('SECRET PROMPT')
  })

  it('explicit runId → full projection unchanged (diagnoser path)', async () => {
    const store = new MemoryEventStore()
    for (const e of [startedEvent('run1'), llmReq('run1')]) await store.append(e)
    const tool = makeTraceTools(store).find(t => t.name === 'get_execution')!
    const res = await tool.handler({ runId: 'run1' }, {} as never) as { steps: unknown[] }
    expect(res.steps).toBeDefined()
    expect(JSON.stringify(res)).toContain('SECRET PROMPT') // 诊断路径保留全量
  })

  it('no previousRunId (first turn) → empty turns', async () => {
    const store = new MemoryEventStore()
    const tool = makeTraceTools(store).find(t => t.name === 'get_execution')!
    const res = await tool.handler({}, {} as never) as { turns: unknown[] }
    expect(res.turns).toEqual([])
  })
})
