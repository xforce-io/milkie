/**
 * Replay a recorded run via the SDK.
 *
 *   $ npx tsx examples/s-005-replay/replay-sdk.ts [runId]
 *
 * Defaults to the runId in `.milkie/last-run.txt` (written by record.ts).
 * Pass an explicit runId to replay a different recorded run.
 *
 * Story: docs/stories/s-005-deterministic-replay.md
 */
import fs from 'fs'
import path from 'path'
import { Milkie } from '../../src/runtime/Milkie'
import { JsonlEventStore } from '../../src/trace/JsonlEventStore'

async function main(): Promise<void> {
  const exampleDir = __dirname
  const milkieDir  = path.join(exampleDir, '.milkie')
  const runsDir    = path.join(milkieDir, 'runs')

  const runId = process.argv[2] ?? fs.readFileSync(path.join(milkieDir, 'last-run.txt'), 'utf-8').trim()

  const milkie = new Milkie({
    eventStore: new JsonlEventStore(runsDir),
  })
  await milkie.loadManifest(path.join(milkieDir, 'agents.json'))

  const result = await milkie.replay(runId)

  console.log(JSON.stringify({
    via:      'sdk',
    runId,
    status:   result.status,
    output:   result.output,
  }, null, 2))
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
