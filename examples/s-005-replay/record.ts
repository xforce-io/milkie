/**
 * Record a sample run for the s-005 deterministic-replay example.
 *
 *   $ npx tsx examples/s-005-replay/record.ts
 *
 * Writes events to `.milkie/runs/<runId>.jsonl` and the chosen runId to
 * `.milkie/last-run.txt` so replay scripts can default to it. The agent
 * here uses a tiny in-process StubGateway — no API key required.
 *
 * Story: docs/stories/s-005-deterministic-replay.md
 */
import fs from 'fs'
import path from 'path'
import { Milkie } from '../../src/runtime/Milkie'
import { MemoryStore } from '../../src/store/MemoryStore'
import { JsonlEventStore } from '../../src/trace/JsonlEventStore'
import type { IModelGateway, ModelRequest, ModelResponse } from '../../src/types/model'

class StubGateway implements IModelGateway {
  constructor(private readonly responses: ModelResponse[]) {}
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    const r = this.responses.shift()
    if (!r) throw new Error('StubGateway exhausted')
    return r
  }
  async *stream(_req: ModelRequest): AsyncIterable<never> { yield* [] }
}

async function main(): Promise<void> {
  const exampleDir = __dirname
  const milkieDir  = path.join(exampleDir, '.milkie')
  const runsDir    = path.join(milkieDir, 'runs')
  fs.mkdirSync(runsDir, { recursive: true })

  // One-shot agent that produces a single response, then completes.
  const gateway = new StubGateway([
    {
      content: [{ type: 'text', text: 'hello, milkie!' }],
      toolCalls: [],
      finishReason: 'end_turn',
    },
  ])

  const milkie = new Milkie({
    stateStore: new MemoryStore(),
    gateway,
    eventStore: new JsonlEventStore(runsDir),
  })
  milkie.loadAgentFile(path.join(exampleDir, 'agents', 'echo.md'))

  const result = await milkie.invoke({
    agentId: 'echo',
    goal:    'demo',
    input:   'say hello',
  })

  fs.writeFileSync(path.join(milkieDir, 'last-run.txt'), result.agentRunId)

  console.log(JSON.stringify({
    runId:  result.agentRunId,
    status: result.status,
    output: result.output,
    eventFile: path.join(runsDir, `${result.agentRunId}.jsonl`),
  }, null, 2))
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
