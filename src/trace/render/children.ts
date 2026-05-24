import { promises as fs } from 'fs'
import path from 'path'

/**
 * Scan a JsonlEventStore base directory for runs whose first event is an
 * `agent.run.started` with `parentId` in the descendant closure of `rootRunId`.
 * Returns descendant runIds (not including the root). Directory may not exist.
 */
export async function findDescendantRuns(baseDir: string, rootRunId: string): Promise<string[]> {
  let entries: string[]
  try {
    entries = await fs.readdir(baseDir)
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return []
    throw err
  }

  // Map every runId in the dir → its parentId (or undefined).
  const parentOf = new Map<string, string | undefined>()
  for (const entry of entries) {
    if (!entry.endsWith('.jsonl')) continue
    const runId = entry.slice(0, -'.jsonl'.length)
    try {
      const content = await fs.readFile(path.join(baseDir, entry), 'utf-8')
      const firstLine = content.split('\n').find(l => l.length > 0)
      if (!firstLine) continue
      const evt = JSON.parse(firstLine) as { type?: string, payload?: { parentId?: string } }
      if (evt.type !== 'agent.run.started') continue
      parentOf.set(runId, evt.payload?.parentId)
    } catch { /* skip unparseable files */ }
  }

  // BFS over parentOf, gathering anyone in the closure.
  const result: string[] = []
  const frontier = [rootRunId]
  while (frontier.length > 0) {
    const current = frontier.shift()!
    for (const [runId, parentId] of parentOf) {
      if (parentId === current && !result.includes(runId)) {
        result.push(runId)
        frontier.push(runId)
      }
    }
  }
  return result
}
