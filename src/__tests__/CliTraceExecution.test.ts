import { main } from '../cli/main'
import fs from 'fs'
import os from 'os'
import path from 'path'

describe('CLI: trace execution', () => {
  let tmpDir: string
  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-execution-'))
  })
  afterEach(() => { fs.rmSync(tmpDir, { recursive: true, force: true }) })

  it('trace execution <runId> emits the execution projection as JSON to stdout', async () => {
    fs.mkdirSync(path.join(tmpDir, '.milkie', 'runs'), { recursive: true })
    const runId = 'demo-run'
    const events = [
      { id: 's', runId, type: 'agent.run.started', actor: 'runtime', timestamp: 1,
        payload: { agentId: 'echo', goal: 'g', input: 'i', contextId: runId } },
      { id: 'ra', runId, type: 'region.added', actor: 'echo', timestamp: 2,
        payload: { id: 'header', target: 'system', section: 'sys', stability: 'immutable', reason: 'agent-set' } },
      { id: 'q', runId, type: 'llm.requested', actor: 'echo', timestamp: 3,
        payload: { request: { model: 'm', messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }] }, requestHash: 'h1' } },
      { id: 'a', runId, type: 'llm.responded', actor: 'echo', timestamp: 4, causedBy: 'q',
        payload: { response: { content: [{ type: 'text', text: 'hi' }], toolCalls: [], finishReason: 'end_turn' },
          requestHash: 'h1', cacheStats: { readTokens: 90, creationTokens: 0, totalInputTokens: 100, hitRate: 0.9 } } },
      { id: 'c', runId, type: 'agent.run.completed', actor: 'runtime', timestamp: 9,
        payload: { status: 'completed', lastTextOutput: 'hi' } },
    ]
    fs.writeFileSync(
      path.join(tmpDir, '.milkie', 'runs', `${runId}.jsonl`),
      events.map(e => JSON.stringify(e)).join('\n') + '\n',
    )

    const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(tmpDir)
    try {
      const result = await main(['trace', 'execution', runId])
      expect(result.exitCode).toBe(0)
      const proj = JSON.parse(result.stdout) as {
        steps: Array<{ kind: string; cacheHealth?: { tier: string }; regionGroups?: Array<{ stability: string }> }>
      }
      const llm = proj.steps.find(s => s.kind === 'llm')
      expect(llm).toBeDefined()
      expect(llm!.cacheHealth?.tier).toBe('hot')
      expect(llm!.regionGroups?.[0]?.stability).toBe('immutable')
    } finally {
      cwdSpy.mockRestore()
    }
  })
})
