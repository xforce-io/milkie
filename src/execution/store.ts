import { existsSync, mkdirSync, openSync, closeSync, readFileSync, writeFileSync, renameSync, unlinkSync, realpathSync, statSync } from 'node:fs'
import { Worker } from 'node:worker_threads'
import { join, resolve } from 'node:path'
import { randomUUID } from 'node:crypto'
import { ExecutionError, type ExecutionContext, type ExecutionRecord } from './types.js'

const CONTEXT_LOCK = `import fcntl, sys
try:
    handle = open(sys.argv[1], 'a+')
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(1)
except OSError:
    sys.exit(2)
sys.stdout.write('locked\\n')
sys.stdout.flush()
sys.stdin.read()
`
/** Runs beside the blocked caller so the helper's pipes stay alive. States: 0 waiting, 1 held, 2 busy, 3 released, 4 failed. */
const LOCK_WORKER = `
const { workerData } = require('node:worker_threads')
const { spawn } = require('node:child_process')
const view = new Int32Array(workerData.shared)
const helper = spawn('python3', ['-c', workerData.script, workerData.file], { stdio: ['pipe', 'pipe', 'ignore'] })
if (helper.pid) Atomics.store(view, 2, helper.pid)
let text = ''
helper.stdout.on('data', (chunk) => { text += chunk })
helper.on('error', () => { if (Atomics.load(view, 0) === 0) { Atomics.store(view, 0, 4); Atomics.notify(view, 0) } })
const timer = setInterval(() => {
  if (Atomics.load(view, 0) === 0 && text.startsWith('locked')) { Atomics.store(view, 0, 1); Atomics.notify(view, 0) }
  if (Atomics.load(view, 1) === 1) { clearInterval(timer); helper.stdin.end() }
}, 5)
helper.on('exit', (code) => {
  clearInterval(timer)
  const held = Atomics.load(view, 0) === 1 || Atomics.load(view, 1) === 1
  Atomics.store(view, 0, held ? 3 : (code === 1 ? 2 : 4))
  Atomics.notify(view, 0)
  process.exit(0)
})
`
interface HeldContextLock { worker: Worker; view: Int32Array }
export function assertId(id: string): void {
  if (typeof id !== 'string' || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(id)) throw new ExecutionError('invalid_request')
}
export function workingDirectory(cwd: string): string {
  try { const path = realpathSync(cwd); if (statSync(path).isDirectory()) return path } catch { /* fixed error below */ }
  throw new ExecutionError('invalid_request')
}
export class ExecutionStore {
  readonly root: string
  /** Same-process reentry. The file lock is what other processes wait behind. */
  private readonly heldLocks = new Set<string>()
  constructor(root: string) {
    this.root = resolve(root)
    mkdirSync(this.root, { recursive: true, mode: 0o700 })
    for (const dir of ['contexts', 'runs', 'active', 'cancel', 'native', 'calls', 'locks']) mkdirSync(join(this.root, dir), { recursive: true, mode: 0o700 })
  }
  /** One context at a time. The kernel drops the lock when the holder process exits. */
  exclusive<T>(contextId: string, body: () => T): T {
    assertId(contextId)
    if (this.heldLocks.has(contextId)) return body()
    const helper = this.acquireContextLock(contextId)
    this.heldLocks.add(contextId)
    try { return body() }
    finally {
      this.heldLocks.delete(contextId)
      this.releaseContextLock(helper)
    }
  }
  private acquireContextLock(contextId: string): HeldContextLock {
    const file = join(this.root, 'locks', `${contextId}.lock`)
    try { closeSync(openSync(file, 'a', 0o600)) } catch { throw new ExecutionError('storage_error') }
    const view = new Int32Array(new SharedArrayBuffer(12))
    let worker: Worker
    try { worker = new Worker(LOCK_WORKER, { eval: true, workerData: { shared: view.buffer, file, script: CONTEXT_LOCK } }) }
    catch { throw new ExecutionError('storage_error') }
    worker.unref()
    Atomics.wait(view, 0, 0, 2000)
    if (Atomics.load(view, 0) !== 1) {
      this.stopLockHelper(view, worker)
      throw new ExecutionError(Atomics.load(view, 0) === 2 ? 'context_busy' : 'storage_error')
    }
    return { worker, view }
  }
  private releaseContextLock(held: HeldContextLock): void {
    Atomics.store(held.view, 1, 1)
    Atomics.notify(held.view, 1)
    Atomics.wait(held.view, 0, 1, 2000)
    if (Atomics.load(held.view, 0) !== 3) this.stopLockHelper(held.view, held.worker)
  }
  private stopLockHelper(view: Int32Array, worker: Worker): void {
    const pid = Atomics.load(view, 2)
    if (pid > 0) { try { process.kill(pid, 'SIGKILL') } catch { /* The helper is already gone. */ } }
    void worker.terminate()
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
