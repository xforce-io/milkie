// src/__tests__/traceTools.deliveredDeref.test.ts
// #200 C: selfOnly 的 runId 轴由"全砍"改为"投递域白名单解引用"。
// 句柄 = 本会话被投递过的 sourceRunId(ctx.deliveredRunIds);消费侧可解引用
// "投递给我的那条产出"的执行/血缘/IO,但任意未投递 runId 仍被拒(保留 #196 安全属性)。
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { makeTraceTools } from '../tools/trace'
import type { Event } from '../trace/types'

function startedEvent(runId: string, input = 'REPORT QUESTION'): Event {
  return { id: `${runId}-s`, runId, type: 'agent.run.started', actor: 'test', timestamp: 0,
    payload: { agentId: 'a', goal: 'g', input, contextId: 'job' } } as Event
}
function llmReq(runId: string): Event {
  return { id: `${runId}-llm`, runId, type: 'llm.requested', actor: 'test', timestamp: 1,
    payload: { requestHash: 'h1', request: { system: 'SECRET PROMPT', messages: [], tools: [] } } } as Event
}
function objCreated(runId: string, objectId: string, type: string, meta?: Record<string, unknown>): Event {
  return { id: `${objectId}-c`, runId, type: 'object.created', actor: 'test', timestamp: 2,
    payload: { objectId, type, producerEventId: 'p', ...(meta ? { meta } : {}) } } as Event
}
function citesRel(runId: string, from: string, to: string): Event {
  return { id: `${from}-${to}`, runId, type: 'relation.created', actor: 'test', timestamp: 3,
    payload: { relationId: `${from}-${to}`, type: 'cites', fromObjectId: from, toObjectId: to, causedByEventId: 'x' } } as Event
}
function completed(runId: string): Event {
  return { id: `${runId}-c`, runId, type: 'agent.run.completed', actor: 'test', timestamp: 4,
    payload: { lastTextOutput: 'REPORT ANSWER' } } as Event
}

const delivered = (...runIds: string[]) => ({ deliveredRunIds: runIds } as never)

describe('#200 C: selfOnly delivered-runId gated dereference', () => {
  it('get_execution honours a DELIVERED runId — returns that run\'s full projection', async () => {
    const store = new MemoryEventStore()
    for (const e of [startedEvent('job-run-1'), llmReq('job-run-1'), completed('job-run-1')]) await store.append(e)
    const tool = makeTraceTools(store, undefined, { selfOnly: true }).find(t => t.name === 'get_execution')!
    const res = await tool.handler({ runId: 'job-run-1' }, delivered('job-run-1')) as { steps?: unknown[] }
    expect(res.steps).toBeDefined()                       // full projection, not self-window turns
    expect(JSON.stringify(res)).toContain('SECRET PROMPT') // the delivered report run is reachable
  })

  it('get_execution IGNORES a non-delivered runId — falls back to self window (no leak)', async () => {
    const store = new MemoryEventStore()
    for (const e of [startedEvent('victim'), llmReq('victim'), completed('victim')]) await store.append(e)
    const tool = makeTraceTools(store, undefined, { selfOnly: true }).find(t => t.name === 'get_execution')!
    // 'victim' was never delivered to this session.
    const res = await tool.handler({ runId: 'victim' }, delivered('job-run-1')) as { turns?: unknown[]; steps?: unknown }
    expect(res.turns).toEqual([])
    expect(res.steps).toBeUndefined()
    expect(JSON.stringify(res)).not.toContain('SECRET PROMPT')
  })

  it('get_lineage honours a DELIVERED runId — surfaces that run\'s claim→source graph', async () => {
    const store = new MemoryEventStore()
    for (const e of [startedEvent('job-run-1'),
      objCreated('job-run-1', 'src1', 'shell:stdout', { source: 'cnbc' }),
      objCreated('job-run-1', 'claim1', 'claim', { text: '某信号' }),
      citesRel('job-run-1', 'claim1', 'src1')]) await store.append(e)
    const tool = makeTraceTools(store, undefined, { selfOnly: true }).find(t => t.name === 'get_lineage')!
    const res = await tool.handler({ runId: 'job-run-1' }, delivered('job-run-1')) as
      { matches: { runId: string; sources: { objectId: string }[] }[] }
    expect(res.matches).toHaveLength(1)
    expect(res.matches[0]!.runId).toBe('job-run-1')
    expect(res.matches[0]!.sources[0]!.objectId).toBe('src1')
  })

  it('get_lineage IGNORES a non-delivered runId', async () => {
    const store = new MemoryEventStore()
    for (const e of [startedEvent('victim'),
      objCreated('victim', 'claim1', 'claim', { text: '机密' }),
      objCreated('victim', 'src1', 'passage'),
      citesRel('victim', 'claim1', 'src1')]) await store.append(e)
    const tool = makeTraceTools(store, undefined, { selfOnly: true }).find(t => t.name === 'get_lineage')!
    const res = await tool.handler({ runId: 'victim' }, delivered('job-run-1')) as { matches: unknown[] }
    expect(res.matches).toEqual([])
  })

  it('get_run_io is registered in selfOnly mode and returns a DELIVERED run\'s I/O', async () => {
    const store = new MemoryEventStore()
    for (const e of [startedEvent('job-run-1'), completed('job-run-1')]) await store.append(e)
    const tools = makeTraceTools(store, undefined, { selfOnly: true })
    expect(tools.map(t => t.name)).toContain('get_run_io')
    const tool = tools.find(t => t.name === 'get_run_io')!
    const res = await tool.handler({ runId: 'job-run-1' }, delivered('job-run-1')) as
      { question?: string; finalAnswer?: string; error?: string }
    expect(res.question).toBe('REPORT QUESTION')
    expect(res.finalAnswer).toBe('REPORT ANSWER')
  })

  it('get_run_io REFUSES a non-delivered runId (no cross-session I/O leak)', async () => {
    const store = new MemoryEventStore()
    for (const e of [startedEvent('victim'), completed('victim')]) await store.append(e)
    const tool = makeTraceTools(store, undefined, { selfOnly: true }).find(t => t.name === 'get_run_io')!
    const res = await tool.handler({ runId: 'victim' }, delivered('job-run-1')) as
      { question?: string; finalAnswer?: string; error?: string }
    expect(res.error).toBeDefined()
    expect(res.question).toBeUndefined()
    expect(JSON.stringify(res)).not.toContain('REPORT ANSWER')
  })

  it('with NO deliveredRunIds in ctx, selfOnly still ignores/refuses every runId', async () => {
    const store = new MemoryEventStore()
    for (const e of [startedEvent('victim'), llmReq('victim'), completed('victim')]) await store.append(e)
    const tools = makeTraceTools(store, undefined, { selfOnly: true })
    const exec = await tools.find(t => t.name === 'get_execution')!.handler({ runId: 'victim' }, {} as never) as { turns?: unknown[] }
    expect(exec.turns).toEqual([])
    const io = await tools.find(t => t.name === 'get_run_io')!.handler({ runId: 'victim' }, {} as never) as { error?: string }
    expect(io.error).toBeDefined()
  })
})

describe('#200 C: full (diagnoser) mode is unaffected — any runId still honoured', () => {
  it('get_run_io reads an arbitrary runId without a delivered allowlist', async () => {
    const store = new MemoryEventStore()
    for (const e of [startedEvent('target'), completed('target')]) await store.append(e)
    const tool = makeTraceTools(store).find(t => t.name === 'get_run_io')!
    const res = await tool.handler({ runId: 'target' }, {} as never) as { question: string; finalAnswer: string }
    expect(res.question).toBe('REPORT QUESTION')
    expect(res.finalAnswer).toBe('REPORT ANSWER')
  })
})
