import { chmodSync, existsSync, mkdirSync, openSync, closeSync, readFileSync, writeFileSync, renameSync, unlinkSync, realpathSync, statSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { randomUUID } from 'node:crypto'
import Database from 'better-sqlite3'
import { ExecutionError, type ExecutionContext, type ExecutionRecord } from './types.js'
export function assertId(id: string): void {
  if (typeof id !== 'string' || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(id)) throw new ExecutionError('invalid_request')
}
export function workingDirectory(cwd: string): string {
  try { const path = realpathSync(cwd); if (statSync(path).isDirectory()) return path } catch { /* fixed error below */ }
  throw new ExecutionError('invalid_request')
}
export class ExecutionStore {
  readonly root: string
  /** Same-process reentry. The database lock is what other processes wait behind. */
  private readonly heldLocks = new Set<string>()
  constructor(root: string) {
    this.root = resolve(root)
    mkdirSync(this.root, { recursive: true, mode: 0o700 })
    for (const dir of ['contexts', 'runs', 'active', 'cancel', 'native', 'calls', 'locks']) mkdirSync(join(this.root, dir), { recursive: true, mode: 0o700 })
  }
  /** One context at a time. The process exit releases the database lock. */
  exclusive<T>(contextId: string, body: () => T): T {
    assertId(contextId)
    if (this.heldLocks.has(contextId)) return body()
    const db = this.acquireContextLock(contextId)
    this.heldLocks.add(contextId)
    try { return body() }
    finally {
      this.heldLocks.delete(contextId)
      this.releaseContextLock(db)
    }
  }
  private acquireContextLock(contextId: string): Database.Database {
    const file = join(this.root, 'locks', `${contextId}.sqlite`)
    let db: Database.Database | undefined
    try {
      db = new Database(file, { timeout: 0 })
      chmodSync(file, 0o600)
      db.pragma('busy_timeout = 0')
      db.exec('BEGIN IMMEDIATE')
      return db
    } catch (error) {
      try { db?.close() } catch { /* The connection was not opened. */ }
      throw new ExecutionError((error as { code?: string }).code === 'SQLITE_BUSY' ? 'context_busy' : 'storage_error')
    }
  }
  private releaseContextLock(db: Database.Database): void {
    try { db.exec('ROLLBACK') } catch { /* The connection is already closed. */ }
    db.close()
  }
  path(kind: string, id: string): string { assertId(id); return join(this.root, kind, `${id}.json`) }
  read<T>(kind: string, id: string): T | undefined {
    const file = this.path(kind, id)
    try {
      const data = JSON.parse(readFileSync(file, 'utf8'))
      if (data.version !== 1) throw new ExecutionError('storage_error')
      return data as T
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code === 'ENOENT') return undefined
      throw new ExecutionError('storage_error')
    }
  }
  write(kind: string, id: string, value: unknown): void {
    const target = this.path(kind, id), temp = `${target}.${randomUUID()}.tmp`
    try { writeFileSync(temp, JSON.stringify(value), { mode: 0o600, flag: 'wx' }); renameSync(temp, target) }
    catch { try { unlinkSync(temp) } catch { /* absent */ } throw new ExecutionError('storage_error') }
  }
  context(id: string): ExecutionContext {
    const c = this.read<ExecutionContext>('contexts', id)
    if (!c) throw new ExecutionError('context_not_found')
    return c
  }
  run(id: string): ExecutionRecord | undefined { return this.read<ExecutionRecord>('runs', id) }
  claim(contextId: string, runId: string): void {
    this.exclusive(contextId, () => {
      try { const fd = openSync(this.path('active', contextId), 'wx', 0o600); try { writeFileSync(fd, runId) } finally { closeSync(fd) } }
      catch (e) { throw new ExecutionError((e as NodeJS.ErrnoException).code === 'EEXIST' ? 'context_busy' : 'storage_error') }
    })
  }
  release(contextId: string, runId: string): void {
    this.exclusive(contextId, () => {
      const file = this.path('active', contextId)
      let current: string
      try { current = readFileSync(file, 'utf8') } catch { throw new ExecutionError('storage_error') }
      // Compare and delete under the context lock so a newer claim cannot be removed.
      if (current !== runId) throw new ExecutionError('storage_error')
      unlinkSync(file)
    })
  }
  requestCancel(runId: string): void { this.write('cancel', runId, { version: 1, runId }) }
  isCancelled(runId: string): boolean { return existsSync(this.path('cancel', runId)) }
}
