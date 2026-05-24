import { buildTimelineTree } from '../trace/render/tree'
import type { Event } from '../trace/types'

const e = (over: Partial<Event> & { id: string, runId: string, type: Event['type'] }): Event => ({
  actor: 'runtime', timestamp: 0, payload: {}, ...over,
})

describe('buildTimelineTree', () => {
  it('returns one root per run, ordered by agent.run.started.timestamp', () => {
    const events: Event[] = [
      e({ id: 'b', runId: 'r2', type: 'agent.run.started', timestamp: 20 }),
      e({ id: 'a', runId: 'r1', type: 'agent.run.started', timestamp: 10 }),
    ]
    const tree = buildTimelineTree(events)
    expect(tree.map(n => n.runId)).toEqual(['r1', 'r2'])
  })

  it('pairs an llm.requested/responded into one entry via causedBy', () => {
    const events: Event[] = [
      e({ id: 's', runId: 'r1', type: 'agent.run.started', timestamp: 1 }),
      e({ id: 'q', runId: 'r1', type: 'llm.requested',  timestamp: 2,
          payload: { request: {}, requestHash: 'h' } }),
      e({ id: 'a', runId: 'r1', type: 'llm.responded',  timestamp: 3, causedBy: 'q',
          payload: { response: {}, requestHash: 'h' } }),
    ]
    const tree = buildTimelineTree(events)
    expect(tree).toHaveLength(1)
    const entries = tree[0]!.entries
    const llmEntries = entries.filter(en => en.kind === 'llm')
    expect(llmEntries).toHaveLength(1)
    expect(llmEntries[0]!.requestedId).toBe('q')
    expect(llmEntries[0]!.respondedId).toBe('a')
  })

  it('orphan llm.requested without a response stays as one entry (in-flight or error)', () => {
    const events: Event[] = [
      e({ id: 's', runId: 'r1', type: 'agent.run.started', timestamp: 1 }),
      e({ id: 'q', runId: 'r1', type: 'llm.requested',  timestamp: 2,
          payload: { request: {}, requestHash: 'h' } }),
    ]
    const tree = buildTimelineTree(events)
    const llmEntries = tree[0]!.entries.filter(en => en.kind === 'llm')
    expect(llmEntries).toHaveLength(1)
    expect(llmEntries[0]!.respondedId).toBeUndefined()
  })

  it('nests a child run under its parent via agent.run.started.parentId', () => {
    const events: Event[] = [
      e({ id: 'ps', runId: 'parent', type: 'agent.run.started', timestamp: 1,
          payload: { agentId: 'p', goal: 'g', input: 'i', contextId: 'parent' } }),
      e({ id: 'cs', runId: 'child', type: 'agent.run.started', timestamp: 2,
          payload: { agentId: 'c', goal: 'g', input: 'i', contextId: 'child', parentId: 'parent' } }),
    ]
    const tree = buildTimelineTree(events)
    expect(tree.map(n => n.runId)).toEqual(['parent'])
    expect(tree[0]!.children.map(n => n.runId)).toEqual(['child'])
  })
})
