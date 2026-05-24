import type { Event } from '../types.js'

export interface LlmEntry {
  kind:          'llm'
  requestedId:   string
  respondedId?:  string
  timestamp:     number
  requestHash?:  string
}

export interface ToolEntry {
  kind:          'tool'
  requestedId:   string
  respondedId?:  string
  timestamp:     number
  toolName:      string
}

export interface LifecycleEntry {
  kind:        'lifecycle'
  eventId:     string
  eventType:   'agent.run.started' | 'agent.run.completed'
  timestamp:   number
}

export type TimelineEntry = LlmEntry | ToolEntry | LifecycleEntry

export interface TimelineNode {
  runId:     string
  startedAt: number
  agentId?:  string
  status?:   string
  parentId?: string
  entries:   TimelineEntry[]
  children:  TimelineNode[]
}

/**
 * Build a tree of TimelineNodes from a flat event array spanning one or more
 * runs. Pairs `*.requested` with its `*.responded` (via causedBy). Nests child
 * runs under parents by `agent.run.started.payload.parentId`.
 *
 * Pure function — no I/O, no clock, no randomness.
 */
export function buildTimelineTree(events: Event[]): TimelineNode[] {
  // Group events by runId.
  const byRun = new Map<string, Event[]>()
  for (const evt of events) {
    const bucket = byRun.get(evt.runId)
    if (bucket) bucket.push(evt)
    else byRun.set(evt.runId, [evt])
  }

  // Build a node per run.
  const nodes = new Map<string, TimelineNode>()
  for (const [runId, evts] of byRun) {
    const sorted = [...evts].sort((a, b) => a.timestamp - b.timestamp)
    const started   = sorted.find(e => e.type === 'agent.run.started') as
      (Event & { payload: { agentId: string, parentId?: string } }) | undefined
    const completed = sorted.find(e => e.type === 'agent.run.completed') as
      (Event & { payload: { status: string } }) | undefined

    const entries: TimelineEntry[] = []
    const requestedById = new Map<string, Event>()
    for (const evt of sorted) requestedById.set(evt.id, evt)

    // Pair *.responded back to *.requested via causedBy.
    const consumed = new Set<string>()
    for (const evt of sorted) {
      if (evt.type === 'llm.responded' || evt.type === 'tool.responded') {
        if (evt.causedBy && requestedById.has(evt.causedBy)) consumed.add(evt.id)
      }
    }
    for (const evt of sorted) {
      if (consumed.has(evt.id)) continue
      if (evt.type === 'llm.requested') {
        const paired = sorted.find(o =>
          o.type === 'llm.responded' && o.causedBy === evt.id) as Event | undefined
        entries.push({
          kind: 'llm', requestedId: evt.id, respondedId: paired?.id,
          timestamp: evt.timestamp,
          requestHash: (evt.payload as { requestHash?: string }).requestHash,
        })
      } else if (evt.type === 'tool.requested') {
        const paired = sorted.find(o =>
          o.type === 'tool.responded' && o.causedBy === evt.id) as Event | undefined
        entries.push({
          kind: 'tool', requestedId: evt.id, respondedId: paired?.id,
          timestamp: evt.timestamp,
          toolName: (evt.payload as { toolName: string }).toolName,
        })
      } else if (evt.type === 'agent.run.started' || evt.type === 'agent.run.completed') {
        entries.push({ kind: 'lifecycle', eventId: evt.id, eventType: evt.type, timestamp: evt.timestamp })
      }
    }

    nodes.set(runId, {
      runId,
      startedAt: started?.timestamp ?? sorted[0]?.timestamp ?? 0,
      agentId:   started?.payload.agentId,
      status:    completed?.payload.status,
      parentId:  started?.payload.parentId,
      entries,
      children:  [],
    })
  }

  // Wire children under parents.
  const roots: TimelineNode[] = []
  for (const node of nodes.values()) {
    if (node.parentId && nodes.has(node.parentId)) {
      nodes.get(node.parentId)!.children.push(node)
    } else {
      roots.push(node)
    }
  }
  // Stable ordering by startedAt at each level.
  const sortRec = (ns: TimelineNode[]): void => {
    ns.sort((a, b) => a.startedAt - b.startedAt)
    for (const n of ns) sortRec(n.children)
  }
  sortRec(roots)
  return roots
}
