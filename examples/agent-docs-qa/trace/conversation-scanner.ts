import { promises as fs } from 'fs'
import path from 'path'
import type { Event } from '../../../src/trace/types.js'

export interface ConversationSummary {
  contextId: string
  agentId:   string
  startedAt: number       // earliest startedAt across runs
  status:    'active' | 'completed' | 'error' | 'interrupted'
  runIds:    string[]     // in startedAt order
  eventCount: number
}

interface RunMeta {
  runId:     string
  contextId: string
  agentId:   string
  startedAt: number
  status:    'active' | 'completed' | 'error' | 'interrupted'
  eventCount: number
}

async function listRunFiles(runsDir: string): Promise<string[]> {
  try {
    const entries = await fs.readdir(runsDir)
    return entries.filter(n => n.endsWith('.jsonl'))
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return []
    throw err
  }
}

async function readMeta(runsDir: string, file: string): Promise<RunMeta | null> {
  try {
    const content = await fs.readFile(path.join(runsDir, file), 'utf-8')
    const lines = content.split('\n').filter(l => l.length > 0)
    if (lines.length === 0) return null

    const first = JSON.parse(lines[0]!) as Event
    if (first.type !== 'agent.run.started') return null

    const startedPayload = first.payload as { agentId: string; contextId: string }
    let status: RunMeta['status'] = 'active'

    for (let i = lines.length - 1; i >= 0; i--) {
      const evt = JSON.parse(lines[i]!) as Event
      if (evt.type === 'agent.run.completed') {
        const p = evt.payload as { status: string }
        status = (p.status as RunMeta['status']) ?? 'completed'
        break
      }
    }

    return {
      runId:      first.runId,
      contextId:  startedPayload.contextId,
      agentId:    startedPayload.agentId,
      startedAt:  first.timestamp,
      status,
      eventCount: lines.length,
    }
  } catch {
    return null
  }
}

/**
 * Scan the runs directory and group runs into conversations by contextId.
 * Sorted by most-recent startedAt descending. Active conversations (no
 * completed event in the latest run) get status='active'.
 */
export async function scanConversations(runsDir: string): Promise<ConversationSummary[]> {
  const files = await listRunFiles(runsDir)
  const metas: RunMeta[] = []
  for (const f of files) {
    const m = await readMeta(runsDir, f)
    if (m) metas.push(m)
  }

  const grouped = new Map<string, RunMeta[]>()
  for (const m of metas) {
    const arr = grouped.get(m.contextId) ?? []
    arr.push(m)
    grouped.set(m.contextId, arr)
  }

  const conversations: ConversationSummary[] = []
  for (const [contextId, runs] of grouped) {
    runs.sort((a, b) => a.startedAt - b.startedAt)
    const latest = runs[runs.length - 1]!
    conversations.push({
      contextId,
      agentId:    latest.agentId,
      startedAt:  runs[0]!.startedAt,
      status:     latest.status,
      runIds:     runs.map(r => r.runId),
      eventCount: runs.reduce((sum, r) => sum + r.eventCount, 0),
    })
  }

  conversations.sort((a, b) => b.startedAt - a.startedAt)
  return conversations
}

/**
 * Read all events for a contextId across its constituent runs, sorted
 * by timestamp ascending (in-conversation chronological order).
 */
export async function readEventsForContext(runsDir: string, contextId: string): Promise<Event[]> {
  const conversations = await scanConversations(runsDir)
  const target = conversations.find(c => c.contextId === contextId)
  if (!target) return []

  const all: Event[] = []
  for (const runId of target.runIds) {
    const content = await fs.readFile(path.join(runsDir, `${runId}.jsonl`), 'utf-8')
    const events = content
      .split('\n')
      .filter(l => l.length > 0)
      .map(l => JSON.parse(l) as Event)
    all.push(...events)
  }
  all.sort((a, b) => a.timestamp - b.timestamp)
  return all
}
