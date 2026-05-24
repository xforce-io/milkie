import fs from 'fs'
import os from 'os'
import path from 'path'
import { findDescendantRuns } from '../trace/render/children'

function writeRun(baseDir: string, runId: string, parentId?: string): void {
  const startedEvent = {
    id: `${runId}-started`,
    runId,
    type: 'agent.run.started',
    actor: 'runtime',
    timestamp: 1,
    payload: { agentId: 'a', goal: 'g', input: 'i', contextId: runId, parentId },
  }
  fs.writeFileSync(
    path.join(baseDir, `${runId}.jsonl`),
    JSON.stringify(startedEvent) + '\n',
  )
}

describe('findDescendantRuns', () => {
  let tmpDir: string
  beforeEach(() => { tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-children-')) })
  afterEach(() => { fs.rmSync(tmpDir, { recursive: true, force: true }) })

  it('returns empty when no children exist', async () => {
    writeRun(tmpDir, 'root')
    expect(await findDescendantRuns(tmpDir, 'root')).toEqual([])
  })

  it('finds direct children', async () => {
    writeRun(tmpDir, 'root')
    writeRun(tmpDir, 'child-a', 'root')
    writeRun(tmpDir, 'child-b', 'root')
    const ids = await findDescendantRuns(tmpDir, 'root')
    expect(ids.sort()).toEqual(['child-a', 'child-b'])
  })

  it('finds transitive grandchildren', async () => {
    writeRun(tmpDir, 'root')
    writeRun(tmpDir, 'child', 'root')
    writeRun(tmpDir, 'grand', 'child')
    const ids = await findDescendantRuns(tmpDir, 'root')
    expect(ids.sort()).toEqual(['child', 'grand'])
  })

  it('returns empty when baseDir does not exist', async () => {
    expect(await findDescendantRuns(path.join(tmpDir, 'nope'), 'root')).toEqual([])
  })
})
