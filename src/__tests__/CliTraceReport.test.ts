import { main } from '../cli/main'
import fs from 'fs'
import os from 'os'
import path from 'path'

describe('CLI: trace render-html', () => {
  let tmpDir: string
  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-render-'))
  })
  afterEach(() => { fs.rmSync(tmpDir, { recursive: true, force: true }) })

  it('reads JSONL from --input file and writes self-contained HTML to stdout', async () => {
    const events = [
      { id: 's', runId: 'r1', type: 'agent.run.started', actor: 'runtime', timestamp: 1,
        payload: { agentId: 'echo', goal: 'g', input: 'i', contextId: 'c' } },
      { id: 'c', runId: 'r1', type: 'agent.run.completed', actor: 'runtime', timestamp: 9,
        payload: { status: 'completed', lastTextOutput: 'hi' } },
    ]
    const input = path.join(tmpDir, 'events.jsonl')
    fs.writeFileSync(input, events.map(e => JSON.stringify(e)).join('\n') + '\n')

    const result = await main(['trace', 'render-html', '--input', input])
    expect(result.exitCode).toBe(0)
    expect(result.stdout.startsWith('<!doctype html>')).toBe(true)
    expect(result.stdout).toContain('echo')
    expect(result.stdout).toContain('r1')
  })

  it('exits non-zero with diagnostic when --input file is missing', async () => {
    const result = await main(['trace', 'render-html', '--input', path.join(tmpDir, 'nope.jsonl')])
    expect(result.exitCode).not.toBe(0)
    expect(result.stderr).toMatch(/ENOENT|not found|no such file/i)
  })

  it('handles empty JSONL gracefully (still emits valid HTML)', async () => {
    const input = path.join(tmpDir, 'empty.jsonl')
    fs.writeFileSync(input, '')
    const result = await main(['trace', 'render-html', '--input', input])
    expect(result.exitCode).toBe(0)
    expect(result.stdout).toContain('<!doctype html>')
  })
})
