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

  it('trace report <runId> renders the run from .milkie/runs/ as HTML', async () => {
    fs.mkdirSync(path.join(tmpDir, '.milkie', 'runs'), { recursive: true })
    const runId = 'demo-run'
    const events = [
      { id: 's', runId, type: 'agent.run.started', actor: 'runtime', timestamp: 1,
        payload: { agentId: 'echo', goal: 'g', input: 'i', contextId: runId } },
      { id: 'c', runId, type: 'agent.run.completed', actor: 'runtime', timestamp: 9,
        payload: { status: 'completed', lastTextOutput: 'hi' } },
    ]
    fs.writeFileSync(
      path.join(tmpDir, '.milkie', 'runs', `${runId}.jsonl`),
      events.map(e => JSON.stringify(e)).join('\n') + '\n',
    )

    const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(tmpDir)
    try {
      const result = await main(['trace', 'report', runId])
      expect(result.exitCode).toBe(0)
      expect(result.stdout.startsWith('<!doctype html>')).toBe(true)
      expect(result.stdout).toContain('echo')
      expect(result.stdout).toContain(runId)
    } finally {
      cwdSpy.mockRestore()
    }
  })

  it('trace report --data-dir <dir> reads <dir>/runs/ directly (alfred sidecar layout, no .milkie, cwd-independent)', async () => {
    // alfred's sidecar persists to `<data-dir>/runs/` (serve --data-dir), NOT under
    // a `.milkie/`. --data-dir lets trace read that layout without findMilkieDir.
    fs.mkdirSync(path.join(tmpDir, 'runs'), { recursive: true })
    const runId = 'sidecar-run'
    const events = [
      { id: 's', runId, type: 'agent.run.started', actor: 'runtime', timestamp: 1,
        payload: { agentId: 'echo', goal: 'g', input: 'i', contextId: runId } },
      { id: 'c', runId, type: 'agent.run.completed', actor: 'runtime', timestamp: 9,
        payload: { status: 'completed', lastTextOutput: 'hi' } },
    ]
    fs.writeFileSync(
      path.join(tmpDir, 'runs', `${runId}.jsonl`),
      events.map(e => JSON.stringify(e)).join('\n') + '\n',
    )

    // No .milkie/ anywhere; cwd points somewhere irrelevant (os.tmpdir, not tmpDir).
    const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(os.tmpdir())
    try {
      const result = await main(['trace', 'report', '--data-dir', tmpDir, runId])
      expect(result.exitCode).toBe(0)
      expect(result.stdout.startsWith('<!doctype html>')).toBe(true)
      expect(result.stdout).toContain(runId)
      expect(result.stdout).toContain('echo')
    } finally {
      cwdSpy.mockRestore()
    }
  })

  it('trace inspect --data-dir <dir> emits the run events from <dir>/runs/', async () => {
    fs.mkdirSync(path.join(tmpDir, 'runs'), { recursive: true })
    const runId = 'insp-run'
    fs.writeFileSync(
      path.join(tmpDir, 'runs', `${runId}.jsonl`),
      JSON.stringify({ id: 's', runId, type: 'agent.run.started', actor: 'runtime', timestamp: 1,
        payload: { agentId: 'echo', goal: 'g', input: 'i', contextId: runId } }) + '\n',
    )
    const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(os.tmpdir())
    try {
      const result = await main(['trace', 'inspect', '--data-dir', tmpDir, runId])
      expect(result.exitCode).toBe(0)
      expect(result.stdout).toContain('agent.run.started')
      expect(result.stdout).toContain(runId)
    } finally {
      cwdSpy.mockRestore()
    }
  })

  it('trace report includes descendant sub-agent runs in one HTML', async () => {
    fs.mkdirSync(path.join(tmpDir, '.milkie', 'runs'), { recursive: true })
    const parent = 'parent-run'
    const child  = 'child-run'
    fs.writeFileSync(
      path.join(tmpDir, '.milkie', 'runs', `${parent}.jsonl`),
      JSON.stringify({ id: 'p', runId: parent, type: 'agent.run.started', actor: 'runtime', timestamp: 1,
        payload: { agentId: 'p', goal: 'g', input: 'i', contextId: parent } }) + '\n',
    )
    fs.writeFileSync(
      path.join(tmpDir, '.milkie', 'runs', `${child}.jsonl`),
      JSON.stringify({ id: 'c', runId: child, type: 'agent.run.started', actor: 'runtime', timestamp: 2,
        payload: { agentId: 'c', goal: 'g', input: 'i', contextId: child, parentId: parent } }) + '\n',
    )

    const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(tmpDir)
    try {
      const result = await main(['trace', 'report', parent])
      expect(result.exitCode).toBe(0)
      const parentIdx = result.stdout.indexOf('data-run-id="parent-run"')
      const childIdx  = result.stdout.indexOf('data-run-id="child-run"')
      expect(parentIdx).toBeGreaterThan(-1)
      expect(childIdx).toBeGreaterThan(parentIdx)
    } finally {
      cwdSpy.mockRestore()
    }
  })
})
