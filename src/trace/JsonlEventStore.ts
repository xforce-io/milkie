import { promises as fs } from 'fs'
import path from 'path'
import { randomUUID } from 'crypto'
import type { IEventStore } from './EventStore.js'
import type { Event } from './types.js'

/**
 * Per-run JSONL file event store. Each run gets a `${runId}.jsonl` file
 * in `baseDir`; events are appended as JSON lines.
 *
 * Directory is created lazily on first write.
 */
export class JsonlEventStore implements IEventStore {
  private dirEnsured = false

  constructor(private readonly baseDir: string) {}

  private fileFor(runId: string): string {
    return path.join(this.baseDir, `${runId}.jsonl`)
  }

  private async ensureDir(): Promise<void> {
    if (this.dirEnsured) return
    await fs.mkdir(this.baseDir, { recursive: true })
    this.dirEnsured = true
  }

  async append(event: Event): Promise<void> {
    await this.ensureDir()
    await fs.appendFile(this.fileFor(event.runId), JSON.stringify(event) + '\n', 'utf-8')
  }

  async readByRunId(runId: string): Promise<Event[]> {
    try {
      const content = await fs.readFile(this.fileFor(runId), 'utf-8')
      return content
        .trim()
        .split('\n')
        .filter(line => line.length > 0)
        .map(line => JSON.parse(line) as Event)
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') return []
      throw err
    }
  }

  async readRange(runId: string, fromIndex: number, count?: number): Promise<Event[]> {
    const all = await this.readByRunId(runId)
    return count !== undefined
      ? all.slice(fromIndex, fromIndex + count)
      : all.slice(fromIndex)
  }

  async reconcileAbandonedRuns(): Promise<number> {
    await this.ensureDir()
    const files = await fs.readdir(this.baseDir)
    let reconciled = 0

    for (const file of files.filter(name => name.endsWith('.jsonl'))) {
      const runId = file.slice(0, -'.jsonl'.length)
      let events: Event[]
      try {
        events = await this.readByRunId(runId)
      } catch {
        continue
      }
      const started = events.find(event => event.type === 'agent.run.started')
      const completed = events.some(event => event.type === 'agent.run.completed')
      const parentId = started && typeof started.payload === 'object' && started.payload !== null
        ? (started.payload as { parentId?: unknown }).parentId
        : undefined
      if (!started || completed || parentId) continue

      await this.append({
        id:        randomUUID(),
        runId,
        type:      'agent.run.completed',
        actor:     'runtime',
        timestamp: Date.now(),
        payload: {
          status: 'interrupted',
          error: {
            code:      'RUN_ABANDONED',
            message:   'Run was abandoned before the service restarted.',
            phase:     'recovery',
            retryable: true,
          },
        },
      })
      reconciled++
    }

    return reconciled
  }
}
