import { main } from '../cli/main'
import fs from 'fs'
import path from 'path'
import os from 'os'

describe('CLI: agent list', () => {
  let tmpDir: string

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-cli-'))
    fs.mkdirSync(path.join(tmpDir, '.milkie'))
    fs.mkdirSync(path.join(tmpDir, 'agents'))
  })

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true })
  })

  function writeAgentFile(name: string, agentId: string): void {
    const content = `---
agentId: ${agentId}
fsm:
  states: []
model:
  provider: stub
  model: stub
  adapter: stub
---
test`
    fs.writeFileSync(path.join(tmpDir, 'agents', name), content)
  }

  function writeManifest(agents: { id: string, file: string }[]): void {
    fs.writeFileSync(
      path.join(tmpDir, '.milkie', 'agents.json'),
      JSON.stringify({ agents }),
    )
  }

  it('outputs JSONL for each agent loaded from manifest', async () => {
    writeAgentFile('router.md',   'router')
    writeAgentFile('verifier.md', 'verifier')
    writeManifest([
      { id: 'router',   file: '../agents/router.md' },
      { id: 'verifier', file: '../agents/verifier.md' },
    ])

    const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(tmpDir)
    try {
      const result = await main(['agent', 'list'])
      expect(result.exitCode).toBe(0)

      const lines = result.stdout.trim().split('\n').filter(Boolean)
      expect(lines).toHaveLength(2)
      const ids = lines.map((l: string) => (JSON.parse(l) as { id: string }).id).sort()
      expect(ids).toEqual(['router', 'verifier'])
    } finally {
      cwdSpy.mockRestore()
    }
  })

  it('--help prints usage and exits 0', async () => {
    const result = await main(['--help'])
    expect(result.exitCode).toBe(0)
    expect(result.stdout).toMatch(/Usage: milkie/)
    expect(result.stdout).toMatch(/agent/)
  })

  it('unknown command exits non-zero with diagnostic on stderr', async () => {
    const result = await main(['bogus'])
    expect(result.exitCode).not.toBe(0)
    expect(result.stderr).toMatch(/bogus|unknown/i)
  })

  it('outputs nothing and exits 0 when no manifest is found upward from cwd', async () => {
    const isolatedDir = fs.mkdtempSync(path.join(tmpDir, 'isolated-'))
    const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(isolatedDir)
    try {
      const result = await main(['agent', 'list'])
      expect(result.exitCode).toBe(0)
      expect(result.stdout).toBe('')
    } finally {
      cwdSpy.mockRestore()
    }
  })

  describe('run / resume / interrupt (need .milkie/ + state)', () => {
    function writeAgentFile(name: string, agentId: string): void {
      const content = `---
agentId: ${agentId}
fsm:
  states: []
model:
  provider: stub
  model: stub
  adapter: openai-compatible
---
sys`
      fs.writeFileSync(path.join(tmpDir, 'agents', name), content)
    }

    function writeManifest(agents: { id: string, file: string }[]): void {
      fs.writeFileSync(
        path.join(tmpDir, '.milkie', 'agents.json'),
        JSON.stringify({ agents }),
      )
    }

    it('agent run exits non-zero when the agentId is not in the manifest', async () => {
      writeAgentFile('router.md', 'router')
      writeManifest([{ id: 'router', file: '../agents/router.md' }])

      const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(tmpDir)
      try {
        const result = await main(['agent', 'run', 'unknown-agent', '--input', 'hi'])
        expect(result.exitCode).not.toBe(0)
        expect(result.stderr).toMatch(/unknown-agent|Agent not found/i)
      } finally {
        cwdSpy.mockRestore()
      }
    })

    it('agent interrupt writes an interrupt flag for the contextId', async () => {
      writeAgentFile('router.md', 'router')
      writeManifest([{ id: 'router', file: '../agents/router.md' }])

      const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(tmpDir)
      try {
        const result = await main(['agent', 'interrupt', 'ctx-abc'])
        expect(result.exitCode).toBe(0)
        expect(JSON.parse(result.stdout.trim())).toEqual({
          contextId: 'ctx-abc',
          status:    'interrupt-signaled',
        })
        // Verify the flag actually persisted to SQLite
        const { SQLiteStore } = await import('../store/SQLiteStore')
        const ss = new SQLiteStore({ path: path.join(tmpDir, '.milkie', 'state.sqlite') })
        await ss.init()
        const flag = await ss.get('context:ctx-abc:interrupt')
        expect(flag).toBe(true)
      } finally {
        cwdSpy.mockRestore()
      }
    })

    it('agent resume exits non-zero when no checkpoint exists for the contextId', async () => {
      writeAgentFile('router.md', 'router')
      writeManifest([{ id: 'router', file: '../agents/router.md' }])

      const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(tmpDir)
      try {
        const result = await main(['agent', 'resume', 'no-such-context'])
        expect(result.exitCode).not.toBe(0)
        expect(result.stderr).toMatch(/no checkpoint|no-such-context/i)
      } finally {
        cwdSpy.mockRestore()
      }
    })
  })
})
