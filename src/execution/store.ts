import { existsSync, mkdirSync, openSync, closeSync, readFileSync, writeFileSync, renameSync, unlinkSync, realpathSync, statSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { randomUUID } from 'node:crypto'
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
  constructor(root: string) {
    this.root = resolve(root)
    mkdirSync(this.root, { recursive: true, mode: 0o700 })
    for (const dir of ['contexts', 'runs', 'active', 'cancel', 'native']) mkdirSync(join(this.root, dir), { recursive: true, mode: 0o700 })
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
    try { const fd = openSync(this.path('active', contextId), 'wx', 0o600); try { writeFileSync(fd, runId) } finally { closeSync(fd) } }
    catch (e) { throw new ExecutionError((e as NodeJS.ErrnoException).code === 'EEXIST' ? 'context_busy' : 'storage_error') }
  }
  release(contextId: string, runId: string): void {
    const file = this.path('active', contextId)
    // Only the owning execution can release its claim, after terminal persistence.
    if (readFileSync(file, 'utf8') !== runId) throw new ExecutionError('storage_error')
    unlinkSync(file)
  }
  requestCancel(runId: string): void { this.write('cancel', runId, { version: 1, runId }) }
  isCancelled(runId: string): boolean { return existsSync(this.path('cancel', runId)) }
}
