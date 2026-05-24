import { main } from '../cli/main'
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { JsonlEventStore } from '../trace/JsonlEventStore'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import fs from 'fs'
import path from 'path'
import os from 'os'

class SequentialGateway implements IModelGateway {
  public callCount = 0
  constructor(private readonly responses: ModelResponse[]) {}
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    this.callCount++
    const r = this.responses.shift()
    if (!r) throw new Error('SequentialGateway exhausted')
    return r
  }
  async *stream(_req: ModelRequest): AsyncIterable<never> { yield* [] }
}

const text = (s: string): ModelResponse => ({
  content: [{ type: 'text', text: s }], toolCalls: [], finishReason: 'end_turn',
})

describe('CLI: trace replay', () => {
  let tmpDir: string

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-cli-trace-'))
    fs.mkdirSync(path.join(tmpDir, '.milkie'))
    fs.mkdirSync(path.join(tmpDir, '.milkie', 'runs'))
    fs.mkdirSync(path.join(tmpDir, 'agents'))
  })

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true })
  })

  function writeAgentMd(name: string, agentId: string): void {
    // adapter: openai-compatible so createGateway() succeeds during replay
    // setup (no actual API call: cache serves every llm.requested)
    const content = `---
agentId: ${agentId}
fsm:
  states:
    - name: react
      type: llm
      instructions: say hi
      tools: []
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

  it('replays a recorded run from .milkie/runs/, outputs JSON with status', async () => {
    writeAgentMd('router.md', 'router')
    writeManifest([{ id: 'router', file: '../agents/router.md' }])

    // 1. record a run via SDK with JsonlEventStore in tmpDir/.milkie/runs/
    const gateway = new SequentialGateway([text('hello')])
    const eventStore = new JsonlEventStore(path.join(tmpDir, '.milkie', 'runs'))
    const recordMilkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway,
      eventStore,
    })
    recordMilkie.loadAgentFile(path.join(tmpDir, 'agents', 'router.md'))
    const original = await recordMilkie.invoke({ agentId: 'router', goal: 'g', input: 'i' })
    expect(original.status).toBe('completed')

    // 2. CLI replay from the same .milkie/runs/ directory
    const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(tmpDir)
    try {
      const result = await main(['trace', 'replay', original.agentRunId])
      expect(result.exitCode).toBe(0)
      const out = JSON.parse(result.stdout.trim()) as { status: string, newRunId: string }
      expect(out.status).toBe('completed')
      expect(out.newRunId).toBe(original.agentRunId)
    } finally {
      cwdSpy.mockRestore()
    }
  })

  it('inspect outputs JSONL of every event in the run', async () => {
    const runId = 'demo-run-1'
    const events = [
      { id: 'e1', runId, type: 'agent.run.started', actor: 'runtime', timestamp: 1, payload: { agentId: 'r', goal: 'g', input: 'i', contextId: 'c' } },
      { id: 'e2', runId, type: 'llm.requested',    actor: 'runtime', timestamp: 2, payload: { request: {}, requestHash: 'h1' } },
    ]
    fs.writeFileSync(
      path.join(tmpDir, '.milkie', 'runs', `${runId}.jsonl`),
      events.map(e => JSON.stringify(e)).join('\n') + '\n',
    )

    const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(tmpDir)
    try {
      const result = await main(['trace', 'inspect', runId])
      expect(result.exitCode).toBe(0)
      const lines = result.stdout.trim().split('\n').filter((l: string) => l.length > 0)
      expect(lines).toHaveLength(2)
      expect(JSON.parse(lines[0]!)).toMatchObject({ id: 'e1', type: 'agent.run.started' })
      expect(JSON.parse(lines[1]!)).toMatchObject({ id: 'e2', type: 'llm.requested' })
    } finally {
      cwdSpy.mockRestore()
    }
  })

  it('inspect outputs empty stdout for a runId with no events', async () => {
    const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(tmpDir)
    try {
      const result = await main(['trace', 'inspect', 'nonexistent'])
      expect(result.exitCode).toBe(0)
      expect(result.stdout).toBe('')
    } finally {
      cwdSpy.mockRestore()
    }
  })

  it('exits non-zero with diagnostic when runId is not found', async () => {
    writeAgentMd('router.md', 'router')
    writeManifest([{ id: 'router', file: '../agents/router.md' }])

    const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(tmpDir)
    try {
      const result = await main(['trace', 'replay', 'nonexistent-run'])
      expect(result.exitCode).not.toBe(0)
      expect(result.stderr).toMatch(/nonexistent-run|no events|not found/i)
    } finally {
      cwdSpy.mockRestore()
    }
  })
})
