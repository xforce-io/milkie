import { promises as fs } from 'fs'
import path from 'path'
import { randomUUID } from 'crypto'
import type { ICrashSafeEventStore } from './EventStore.js'
import type { Event } from './types.js'
import {
  fsyncDirectory,
  fsyncStoreHierarchy,
  mkdirDurable,
} from './durableFs.js'

/**
 * Per-run JSONL file event store. Each run gets a `${runId}.jsonl` file
 * in `baseDir`; events are appended as JSON lines.
 *
 * Directory is created lazily on first write with durable parent fsync
 * (#227 crash-safe evidence). confirmRunDurable always fsyncs base parent.
 */
export class JsonlEventStore implements ICrashSafeEventStore {
  readonly durability = 'crash-safe' as const
  private dirEnsured = false

  constructor(private readonly baseDir: string) {}

  private fileFor(runId: string): string {
    return path.join(this.baseDir, `${runId}.jsonl`)
  }

  private async ensureDir(): Promise<void> {
    if (this.dirEnsured) return
    await mkdirDurable(this.baseDir)
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

  /**
   * #227: fsync the run's JSONL file and the directory chain from baseDir
   * through the configured base parent so crash-safe finalization can treat
   * event evidence as durable across process takeover.
   */
  async confirmRunDurable(runId: string): Promise<void> {
    await this.ensureDir()
    const file = this.fileFor(runId)
    // Ensure the file is readable before confirming durability.
    let fh: fs.FileHandle | undefined
    try {
      fh = await fs.open(file, 'r')
      await fh.sync()
    } catch (err) {
      if (fh) {
        try { await fh.close() } catch { /* ignore */ }
      }
      throw err
    }
    await fh.close()

    // leaf == base for flat per-run files; always fsync base parent.
    await fsyncStoreHierarchy({
      leafDir: this.baseDir,
      baseDir: this.baseDir,
      syncDirectory: fsyncDirectory,
    })
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
