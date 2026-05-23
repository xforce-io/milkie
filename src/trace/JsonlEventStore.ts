import { promises as fs } from 'fs'
import path from 'path'
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
}
