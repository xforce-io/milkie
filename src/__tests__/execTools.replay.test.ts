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
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { AgentConfig } from '../types/agent'

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
