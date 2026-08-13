// #134: a run that uses the side-effectful run_command tool must replay from
// cache WITHOUT re-executing the subprocess. Proven directly: run_command appends
// one byte to a file at record time; replay must NOT append again (handler is
// served from CacheIndex, never re-run — see ReplayingIOPort / ExplodingInnerPort).
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { RecordingIOPort } from '../trace/RecordingIOPort'
import { ReplayingIOPort } from '../trace/ReplayingIOPort'
import { CacheIndex } from '../trace/CacheIndex'
import type { IIOPort } from '../runtime/IOPort'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { AgentConfig } from '../types/agent'
import type { ToolRequestedPayload } from '../trace/types'

function toolResp(id: string, name: string, input: unknown): ModelResponse {
  return { content: [{ type: 'tool_use', id, name, input }], toolCalls: [{ id, name, input }], finishReason: 'tool_use' }
}

class ScriptedGateway implements IModelGateway {
  private n = 0
  constructor(private readonly command: string) {}
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    this.n++
    if (this.n === 1) return toolResp('c1', 'run_command', { command: this.command })
    return { content: [{ type: 'text', text: 'done' }], toolCalls: [], finishReason: 'end_turn' }
  }
  async *stream(_r: ModelRequest): AsyncIterable<never> { yield* [] }
}

class ReplayInnerPort implements IIOPort {
  private clock = 0
  private uuidValue = 0

  async invokeLLM(_req: ModelRequest): Promise<ModelResponse> {
    return { content: [], toolCalls: [], finishReason: 'end_turn' }
  }

  async invokeTool(_name: string, _input: unknown, execute: (signal: AbortSignal) => Promise<unknown>): Promise<unknown> {
    return execute(new AbortController().signal)
  }

  now(): number { return this.clock++ }
  uuid(): string { return `replay-${this.uuidValue++}` }
}


const agent: AgentConfig = {
  agentId: 'shell-runner', version: '1.0.0',
  systemPrompt: 'run the command then answer',
  fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 10 }] },
  model: { provider: 'test', model: 'test', adapter: 'test' },
}

describe('determinism: run_command (#134) replays from cache without re-executing the subprocess', () => {
  it('side effect happens exactly once at record time; replay does not re-fork', async () => {
    const dir  = await mkdtemp(join(tmpdir(), 'milkie-exec-replay-'))
    const file = join(dir, 'side-effect.log')
    // append one byte + emit stdout; if replay re-ran the handler the file would grow.
    const command = `printf X >> '${file}'; echo ran-ok`

    try {
      const milkie = new Milkie({
        stateStore: new MemoryStore(),
        eventStore: new MemoryEventStore(),
        gateway:    new ScriptedGateway(command),
      })
      milkie.registerAgent(agent)

      const run = await milkie.invoke({ agentId: 'shell-runner', goal: 'g', input: 'go', contextId: 'exec-replay' })
      expect(run.status).toBe('completed')
      expect(await readFile(file, 'utf8')).toBe('X') // ran exactly once at record time

      // Replay the recorded run. The script still exists, but the point is the
      // handler must NOT run at all — so the file must stay at one byte.
      const replayed = await milkie.replay(run.agentRunId)
      expect(replayed.output).toBe(run.output)
      expect(await readFile(file, 'utf8')).toBe('X') // STILL one byte → handler not re-run

      // Replaying again must remain stable (idempotent, no accumulation).
      await milkie.replay(run.agentRunId)
      expect(await readFile(file, 'utf8')).toBe('X')
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })
})
describe('#219 invalid tool arguments retain distinct replay identity', () => {
  it('replays the rejected malformed call separately from a valid empty call without executing again', async () => {
    const store = new MemoryEventStore()
    const invalidArguments = {
      code:      'TOOL_ARGUMENTS_INVALID_JSON' as const,
      message:   'Tool arguments are not valid JSON',
      rawLength: 9,
    }
    const record = new RecordingIOPort(new ReplayInnerPort(), store, 'run-219')
    const rejectedHandler = jest.fn(async () => {
      throw Object.assign(new Error(invalidArguments.message), { code: invalidArguments.code })
    })
    const validHandler = jest.fn(async () => ({ ok: true }))

    await expect(record.invokeTool('noop', {}, rejectedHandler, { invalidArguments }))
      .rejects.toMatchObject({ code: 'TOOL_ARGUMENTS_INVALID_JSON' })
    await expect(record.invokeTool('noop', {}, validHandler)).resolves.toEqual({ ok: true })
    expect(rejectedHandler).toHaveBeenCalledTimes(1)
    expect(validHandler).toHaveBeenCalledTimes(1)

    const events = await store.readByRunId('run-219')
    const requests = events
      .filter(event => event.type === 'tool.requested')
      .map(event => event.payload as ToolRequestedPayload)
    expect(requests[0]).toMatchObject({
      invalidArguments: { code: 'TOOL_ARGUMENTS_INVALID_JSON' },
    })
    expect(requests[0]!.requestHash).not.toBe(requests[1]!.requestHash)

    const replayHandler = jest.fn(async () => { throw new Error('replay must not execute') })
    const replay = new ReplayingIOPort(CacheIndex.fromEvents(events), new ReplayInnerPort())
    await expect(replay.invokeTool('noop', {}, replayHandler, { invalidArguments }))
      .rejects.toMatchObject({ code: 'TOOL_ARGUMENTS_INVALID_JSON' })
    await expect(replay.invokeTool('noop', {}, replayHandler)).resolves.toEqual({ ok: true })
    expect(replayHandler).not.toHaveBeenCalled()
  })
})
