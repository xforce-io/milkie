import { scanConversations, readEventsForContext } from '../trace/conversation-scanner'
import fs from 'fs'
import os from 'os'
import path from 'path'

function writeRun(
  baseDir: string,
  runId: string,
  contextId: string,
  startedAt: number,
  completedStatus?: string,
): void {
  const events: Array<Record<string, unknown>> = [
    {
      id: `${runId}-s`, runId, type: 'agent.run.started', actor: 'runtime', timestamp: startedAt,
      payload: { agentId: 'sanguo-researcher', goal: 'g', input: 'i', contextId },
    },
    {
      id: `${runId}-l1`, runId, type: 'llm.requested', actor: 'runtime', timestamp: startedAt + 1,
      payload: { request: {}, requestHash: 'h' },
    },
  ]
  if (completedStatus) {
    events.push({
      id: `${runId}-c`, runId, type: 'agent.run.completed', actor: 'runtime', timestamp: startedAt + 2,
      payload: { status: completedStatus },
    })
  }
  fs.writeFileSync(
    path.join(baseDir, `${runId}.jsonl`),
    events.map(e => JSON.stringify(e)).join('\n') + '\n',
  )
}

describe('conversation-scanner', () => {
  let tmpDir: string
  beforeEach(() => { tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'conv-scanner-')) })
  afterEach(() => { fs.rmSync(tmpDir, { recursive: true, force: true }) })

  it('scanConversations returns empty for empty dir', async () => {
    expect(await scanConversations(tmpDir)).toEqual([])
  })

  it('scanConversations returns empty for nonexistent dir', async () => {
    expect(await scanConversations(path.join(tmpDir, 'nope'))).toEqual([])
  })

  it('scanConversations groups multiple runs by contextId', async () => {
    writeRun(tmpDir, 'run1', 'ctxA', 100, 'completed')
    writeRun(tmpDir, 'run2', 'ctxA', 200, 'completed')
    writeRun(tmpDir, 'run3', 'ctxB', 150, 'completed')

    const convs = await scanConversations(tmpDir)
    expect(convs).toHaveLength(2)

    const ctxA = convs.find(c => c.contextId === 'ctxA')!
    expect(ctxA.runIds.sort()).toEqual(['run1', 'run2'])
    expect(ctxA.agentId).toBe('sanguo-researcher')

    const ctxB = convs.find(c => c.contextId === 'ctxB')!
    expect(ctxB.runIds).toEqual(['run3'])
  })

  it('scanConversations sorts by most-recent startedAt descending', async () => {
    writeRun(tmpDir, 'old',  'ctxOld',  100, 'completed')
    writeRun(tmpDir, 'newer', 'ctxNew', 500, 'completed')
    writeRun(tmpDir, 'mid',  'ctxMid',  300, 'completed')

    const convs = await scanConversations(tmpDir)
    expect(convs.map(c => c.contextId)).toEqual(['ctxNew', 'ctxMid', 'ctxOld'])
  })

  it('scanConversations marks "active" when latest run lacks completed event', async () => {
    writeRun(tmpDir, 'r1', 'ctxLive', 100)  // no completed
    const convs = await scanConversations(tmpDir)
    expect(convs[0]!.status).toBe('active')
  })

  it('scanConversations reports "completed" when latest run has completed event', async () => {
    writeRun(tmpDir, 'r1', 'ctxDone', 100, 'completed')
    const convs = await scanConversations(tmpDir)
    expect(convs[0]!.status).toBe('completed')
  })

  it('readEventsForContext returns events of all matching runIds in timestamp order', async () => {
    writeRun(tmpDir, 'run1', 'ctxA', 100, 'completed')
    writeRun(tmpDir, 'run2', 'ctxA', 200, 'completed')
    writeRun(tmpDir, 'run3', 'ctxB', 150, 'completed')

    const events = await readEventsForContext(tmpDir, 'ctxA')
    expect(events.filter(e => e.runId === 'run3')).toHaveLength(0)
    expect(events.length).toBe(6)  // 3 events × 2 runs

    const timestamps = events.map(e => e.timestamp)
    const sorted = [...timestamps].sort((a, b) => a - b)
    expect(timestamps).toEqual(sorted)
  })

  it('readEventsForContext returns empty for unknown contextId', async () => {
    writeRun(tmpDir, 'r1', 'ctxA', 100, 'completed')
    expect(await readEventsForContext(tmpDir, 'ctxNonexistent')).toEqual([])
  })
})
