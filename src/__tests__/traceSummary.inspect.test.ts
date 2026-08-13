import fs from 'fs'
import os from 'os'
import path from 'path'
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { JsonlEventStore } from '../trace/JsonlEventStore'
import { parseJsonlEvents, summarizeRun, TraceInspectError } from '../trace/summarizeRun'
import { main } from '../cli/main'
import type { AgentConfig } from '../types/agent'
import type { IModelGateway, ModelResponse } from '../types/model'

class TextGateway implements IModelGateway {
  async complete(): Promise<ModelResponse> {
    return { content: [{ type: 'text', text: 'hello' }], toolCalls: [], finishReason: 'stop' }
  }
  async *stream(): AsyncIterable<never> { yield* [] }
}

describe('#246 inspect fail-closed and summary', () => {
  it('summary matches invoke envelope and inspect ids equal raw JSONL', async () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-246-'))
    const runsDir = path.join(tmp, 'runs')
    fs.mkdirSync(runsDir)
    const eventStore = new JsonlEventStore(runsDir)
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore,
      gateway: new TextGateway(),
    })
    milkie.registerAgent({
      agentId: 'a',
      version: '1',
      systemPrompt: 's',
      fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 3 }] },
      model: { provider: 't', model: 't', adapter: 't' },
    } satisfies AgentConfig)

    const result = await milkie.invoke({ agentId: 'a', goal: 'g', input: 'i' })
    const raw = fs.readFileSync(path.join(runsDir, `${result.agentRunId}.jsonl`), 'utf-8')
    const events = parseJsonlEvents(raw)
    const rawIds = events.map(e => e.id)

    const summary = await milkie.getRunSummary(result.agentRunId)
    expect(summary.status).toBe(result.status)
    expect(summary.stopReason).toBe(result.stopReason)
    expect(summary.artifacts).toEqual(result.artifacts)
    expect(summary.turns).toBeGreaterThanOrEqual(1)
    expect(rawIds.sort()).toEqual(events.map(e => e.id).sort())
    expect(summarizeRun(events, result.agentRunId).runId).toBe(result.agentRunId)

    const inspect = await main(['trace', 'inspect', result.agentRunId, '--data-dir', tmp])
    expect(inspect.exitCode).toBe(0)
    const inspectIds = inspect.stdout.trim().split('\n').filter(Boolean).map(l => (JSON.parse(l) as { id: string }).id)
    expect(inspectIds.sort()).toEqual(rawIds.sort())

    const sumCli = await main(['trace', 'summary', result.agentRunId, '--data-dir', tmp])
    expect(sumCli.exitCode).toBe(0)
    expect(JSON.parse(sumCli.stdout).stopReason).toBe(result.stopReason)
  })

  it('truncated JSONL fail-closed: no stdout payload', async () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-246-bad-'))
    const runsDir = path.join(tmp, 'runs')
    fs.mkdirSync(runsDir)
    fs.writeFileSync(path.join(runsDir, 'bad.jsonl'), '{"id":"1","runId":"bad"}\n{not json\n')
    expect(() => parseJsonlEvents(fs.readFileSync(path.join(runsDir, 'bad.jsonl'), 'utf-8')))
      .toThrow(TraceInspectError)

    const inspect = await main(['trace', 'inspect', 'bad', '--data-dir', tmp])
    expect(inspect.exitCode).not.toBe(0)
    expect(inspect.stdout).toBe('')
    expect(JSON.parse(inspect.stderr).error.code).toBe('TRACE_INSPECT_INCOMPLETE')

    const summary = await main(['trace', 'summary', 'bad', '--data-dir', tmp])
    expect(summary.exitCode).not.toBe(0)
    expect(summary.stdout).toBe('')
  })
})
