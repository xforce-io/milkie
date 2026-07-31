import { main } from '../cli/main'
import type { MainResult } from '../cli/main'
import { SQLiteStore } from '../store/SQLiteStore'
import fs from 'fs'
import path from 'path'
import os from 'os'
import { spawnSync } from 'child_process'

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
    function writeTerminalErrorAgentFile(name: string, agentId: string): void {
      const content = `---
agentId: ${agentId}
fsm:
  states:
    - name: react
      type: llm
      max_iterations: 0
model:
  provider: stub
  model: stub
  adapter: openai-compatible
---
sys`
      fs.writeFileSync(path.join(tmpDir, 'agents', name), content)
    }

    async function seedResumableCheckpoint(contextId: string, runId: string): Promise<void> {
      const stateStore = new SQLiteStore({ path: path.join(tmpDir, '.milkie', 'state.sqlite') })
      await stateStore.init()
      await stateStore.set(`context:${contextId}:checkpoint-run:latest`, runId)

      const checkpoint = {
        checkpointId: 'checkpoint-terminal-error',
        sequence:     1,
        goal:         'resume error',
        currentTurn:  'resume error',
        fsm:          { currentState: 'paused', resumeState: 'react', stateData: null },
        context: {
          workingMemory: { data: {}, log: [] },
          regions:       { epoch: 0, regions: [] },
        },
        pendingEvents: [],
        children:      [],
        meta: {
          agentId:    'router',
          agentRunId: runId,
          timestamp:  1,
          traceId:    'trace-terminal-error',
        },
      }
      fs.mkdirSync(path.join(tmpDir, '.milkie', 'runs'), { recursive: true })
      fs.writeFileSync(
        path.join(tmpDir, '.milkie', 'runs', `${runId}.jsonl`),
        JSON.stringify({
          id:        'checkpoint-event',
          runId,
          type:      'agent.checkpoint',
          actor:     'router',
          timestamp: 1,
          payload:   { checkpoint },
        }) + '\n',
      )
    }

    function expectTerminalAgentError(result: MainResult, runId: string, contextId: string): void {
      expect(result.exitCode).toBe(1)
      expect(JSON.parse(result.stdout.trim())).toMatchObject({
        runId,
        contextId,
        status: 'error',
      })
      expect(JSON.parse(result.stderr.trim())).toMatchObject({
        error: {
          code:    'AGENT_RUN_ERROR',
          status:  'error',
          runId,
          contextId,
        },
      })
    }

    it('agent run emits terminal error JSON to stdout and AGENT_RUN_ERROR to stderr', async () => {
      writeTerminalErrorAgentFile('router.md', 'router')
      writeManifest([{ id: 'router', file: '../agents/router.md' }])

      const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(tmpDir)
      try {
        const result = await main(['agent', 'run', 'router', '--input', 'fail'])
        const { runId, contextId } = JSON.parse(result.stdout.trim()) as { runId: string, contextId: string }
        expectTerminalAgentError(result, runId, contextId)
      } finally {
        cwdSpy.mockRestore()
      }
    })

    it('CLI entry process exits 1 for a terminal agent error', () => {
      writeTerminalErrorAgentFile('router.md', 'router')
      writeManifest([{ id: 'router', file: '../agents/router.md' }])

      const child = spawnSync(
        process.execPath,
        [
          '--import',
          path.join(__dirname, '..', '..', 'node_modules', 'tsx', 'dist', 'loader.mjs'),
          path.join(__dirname, '..', 'cli', 'index.ts'),
          'agent',
          'run',
          'router',
          '--input',
          'fail',
        ],
        { cwd: tmpDir, encoding: 'utf8' },
      )

      expect(child.status).toBe(1)
      expect(JSON.parse(child.stdout.trim()).status).toBe('error')
      expect(JSON.parse(child.stderr.trim()).error.code).toBe('AGENT_RUN_ERROR')
    })

    it('agent resume emits terminal error JSON to stdout and AGENT_RUN_ERROR to stderr', async () => {
      const contextId = 'ctx-terminal-error'
      const runId = 'run-terminal-error'
      writeTerminalErrorAgentFile('router.md', 'router')
      writeManifest([{ id: 'router', file: '../agents/router.md' }])
      await seedResumableCheckpoint(contextId, runId)

      const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(tmpDir)
      try {
        const result = await main(['agent', 'resume', contextId])
        expectTerminalAgentError(result, runId, contextId)
      } finally {
        cwdSpy.mockRestore()
      }
    })


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
