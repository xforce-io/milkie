/**
 * Task Outcome Finalization Store — SPI + Memory + File (#227 / s-017).
 *
 * Single fact source for immutable final results. create-if-absent is linearized
 * per backend concurrency domain (Memory: process instance; File: filesystem dir).
 */

import { promises as fs } from 'fs'
import path from 'path'
import { randomBytes } from 'crypto'
import { sha256Hex } from '../trace/hash.js'
import {
  fsyncDirectory,
  fsyncStoreHierarchy,
  mkdirDurable,
} from '../trace/durableFs.js'
import type {
  DurabilityClass,
  TaskOutcomeFinalization,
} from '../types/outcome.js'
import {
  TaskOutcomeFinalizationCorruptionError,
  TaskOutcomeFinalizationStoreError,
  TaskOutcomeFinalizationValidationError,
} from '../types/outcome.js'
import {
  canonicalizeFinalization,
  parseAndValidateFinalization,
  snapshotFinalization,
} from './finalizationHash.js'

export interface ITaskOutcomeFinalizationStore {
  readonly durability: DurabilityClass
  create(record: TaskOutcomeFinalization): Promise<
    | { readonly created: true; readonly record: TaskOutcomeFinalization }
    | { readonly created: false; readonly existing: TaskOutcomeFinalization }
  >
  get(runId: string): Promise<TaskOutcomeFinalization | null>
}

// ─── shared helpers ──────────────────────────────────────────────────────────

function assertCreateInput(record: TaskOutcomeFinalization): TaskOutcomeFinalization {
  // Validate + independent snapshot; never store caller object references.
  try {
    return parseAndValidateFinalization(record, record.runId)
  } catch (err) {
    if (err instanceof TaskOutcomeFinalizationValidationError) {
      throw new TaskOutcomeFinalizationStoreError(
        'invalid_record',
        'create_validate',
        err.message,
        err,
      )
    }
    throw err
  }
}

function corruption(runId: string, message: string, cause?: unknown): TaskOutcomeFinalizationCorruptionError {
  const e = new TaskOutcomeFinalizationCorruptionError(runId, message)
  if (cause !== undefined) {
    ;(e as Error & { cause?: unknown }).cause = cause
  }
  return e
}

// ─── Memory ──────────────────────────────────────────────────────────────────

/**
 * Process-local finalization store. Linearizes create via a per-instance mutex
 * chain so concurrent awaits still serialize has/set. durability = 'process'.
 * Not for persistent deployments; tests/explicit temp only.
 */
export class MemoryTaskOutcomeFinalizationStore implements ITaskOutcomeFinalizationStore {
  readonly durability: DurabilityClass = 'process'
  private readonly records = new Map<string, string>()
  /** Serialize create critical sections across concurrent callers. */
  private chain: Promise<void> = Promise.resolve()

  private async withLock<T>(fn: () => Promise<T> | T): Promise<T> {
    let release!: () => void
    const next = new Promise<void>(resolve => {
      release = resolve
    })
    const prev = this.chain
    this.chain = prev.then(() => next)
    await prev
    try {
      return await fn()
    } finally {
      release()
    }
  }

  async create(
    record: TaskOutcomeFinalization,
  ): Promise<
    | { readonly created: true; readonly record: TaskOutcomeFinalization }
    | { readonly created: false; readonly existing: TaskOutcomeFinalization }
  > {
    const canonical = assertCreateInput(record)
    const canonicalString = canonicalizeFinalization(canonical)

    return this.withLock(() => {
      const existingStr = this.records.get(canonical.runId)
      if (existingStr !== undefined) {
        let existing: TaskOutcomeFinalization
        try {
          existing = parseAndValidateFinalization(JSON.parse(existingStr), canonical.runId)
        } catch (err) {
          throw corruption(
            canonical.runId,
            'existing finalization record is corrupted',
            err,
          )
        }
        return { created: false as const, existing }
      }
      this.records.set(canonical.runId, canonicalString)
      return { created: true as const, record: snapshotFinalization(canonical) }
    })
  }

  async get(runId: string): Promise<TaskOutcomeFinalization | null> {
    const id = typeof runId === 'string' ? runId.trim() : ''
    if (!id) {
      throw new TaskOutcomeFinalizationStoreError(
        'invalid_record',
        'get_validate',
        'get requires a non-empty runId',
      )
    }
    return this.withLock(() => {
      const existingStr = this.records.get(id)
      if (existingStr === undefined) return null
      try {
        return parseAndValidateFinalization(JSON.parse(existingStr), id)
      } catch (err) {
        throw corruption(id, 'finalization record is corrupted', err)
      }
    })
  }
}

// ─── File ────────────────────────────────────────────────────────────────────

export interface FileTaskOutcomeFinalizationStoreOptions {
  /** Injected for fault-injection tests. Defaults to real fs.promises ops. */
  readonly fsOps?: FileFinalizationFsOps
}

export interface FileFinalizationFsOps {
  mkdir(path: string, opts?: { recursive?: boolean }): Promise<string | undefined>
  open(
    path: string,
    flags: string | number,
    mode?: number,
  ): Promise<{
    writeFile(data: string, encoding: BufferEncoding): Promise<void>
    sync(): Promise<void>
    close(): Promise<void>
  }>
  link(existingPath: string, newPath: string): Promise<void>
  readFile(path: string, encoding: BufferEncoding): Promise<string>
  rm(path: string, opts?: { force?: boolean }): Promise<void>
  /** fsync a directory path (open O_RDONLY + sync + close). */
  syncDirectory(dirPath: string): Promise<void>
  readdir(path: string): Promise<string[]>
  stat(path: string): Promise<{ mtimeMs: number }>
}

function defaultFsOps(): FileFinalizationFsOps {
  return {
    mkdir: (p, opts) => fs.mkdir(p, opts),
    open: async (p, flags, mode) => {
      const fh = await fs.open(p, flags, mode)
      return {
        writeFile: (data, encoding) => fh.writeFile(data, encoding),
        sync: () => fh.sync(),
        close: () => fh.close(),
      }
    },
    link: (a, b) => fs.link(a, b),
    readFile: (p, enc) => fs.readFile(p, enc),
    rm: (p, opts) => fs.rm(p, opts),
    syncDirectory: (dirPath) => fsyncDirectory(dirPath),
    readdir: (p) => fs.readdir(p),
    stat: (p) => fs.stat(p),
  }
}

/**
 * Crash-safe file finalization store.
 * Layout: `<base>/sha256/<first2>/<remaining>.json`
 * Linearization: `link(temp, target)`. Durable visibility: fsync the full
 * directory chain leaf→base and always the configured base parent before
 * returning created/existing/final. Directory fsync failure → commit_unknown.
 */
export class FileTaskOutcomeFinalizationStore implements ITaskOutcomeFinalizationStore {
  readonly durability: DurabilityClass = 'crash-safe'
  private readonly ops: FileFinalizationFsOps
  /** True once this instance has observed or created baseDir durably. */
  private baseReady = false

  constructor(
    private readonly baseDir: string,
    opts?: FileTaskOutcomeFinalizationStoreOptions,
  ) {
    this.ops = opts?.fsOps ?? defaultFsOps()
  }

  private targetPath(runId: string): string {
    const hex = sha256Hex(runId)
    return path.join(this.baseDir, 'sha256', hex.slice(0, 2), `${hex.slice(2)}.json`)
  }

  /**
   * Ensure baseDir exists. Creation durability is handled by mkdirDurable;
   * every later create/get acknowledgment always fsyncs the base parent so
   * takeover instances do not skip root-entry durability.
   */
  private async ensureBase(): Promise<void> {
    if (this.baseReady) return
    try {
      await mkdirDurable(this.baseDir, {
        mkdir: (p, o) => this.ops.mkdir(p, o),
        stat: (p) => this.ops.stat(p),
        syncDirectory: (p) => this.ops.syncDirectory(p),
      })
    } catch (err) {
      throw new TaskOutcomeFinalizationStoreError(
        'io',
        'ensure_base',
        'failed to create durable finalization base directory',
        err,
      )
    }
    this.baseReady = true
  }

  /**
   * Fsync leaf parent → base (inclusive) and always the parent of baseDir.
   * Used after link on winner/loser and on get before returning final/existing.
   */
  private async syncHierarchy(target: string, stage: string): Promise<void> {
    try {
      await fsyncStoreHierarchy({
        leafDir: path.dirname(target),
        baseDir: this.baseDir,
        syncDirectory: (p) => this.ops.syncDirectory(p),
      })
    } catch (err) {
      throw new TaskOutcomeFinalizationStoreError(
        'commit_unknown',
        stage,
        `failed to fsync finalization directory hierarchy at ${stage}`,
        err,
      )
    }
  }

  private async readValidateExisting(
    target: string,
    runId: string,
  ): Promise<TaskOutcomeFinalization> {
    let text: string
    try {
      text = await this.ops.readFile(target, 'utf-8')
    } catch (err) {
      throw new TaskOutcomeFinalizationStoreError(
        'io',
        'read_existing',
        'failed to read existing finalization record',
        err,
      )
    }
    let parsed: unknown
    try {
      parsed = JSON.parse(text)
    } catch (err) {
      throw corruption(runId, 'existing finalization JSON is malformed', err)
    }
    try {
      return parseAndValidateFinalization(parsed, runId)
    } catch (err) {
      if (err instanceof TaskOutcomeFinalizationValidationError) {
        throw corruption(runId, err.message, err)
      }
      throw err
    }
  }

  private async cleanupTemp(tmp: string): Promise<void> {
    try {
      await this.ops.rm(tmp, { force: true })
    } catch {
      // temp cleanup failure is non-fatal; never changes confirmed target.
    }
  }

  async create(
    record: TaskOutcomeFinalization,
  ): Promise<
    | { readonly created: true; readonly record: TaskOutcomeFinalization }
    | { readonly created: false; readonly existing: TaskOutcomeFinalization }
  > {
    const canonical = assertCreateInput(record)
    const canonicalString = canonicalizeFinalization(canonical)
    await this.ensureBase()

    const target = this.targetPath(canonical.runId)
    const parent = path.dirname(target)
    // Durable mkdir of sha256/<shard> under base (base already durable via ensureBase).
    try {
      await mkdirDurable(parent, {
        mkdir: (p, o) => this.ops.mkdir(p, o),
        stat: (p) => this.ops.stat(p),
        syncDirectory: (p) => this.ops.syncDirectory(p),
      })
    } catch (err) {
      throw new TaskOutcomeFinalizationStoreError(
        'io',
        'ensure_target_dirs',
        'failed to create durable finalization target directories',
        err,
      )
    }

    const tmp = path.join(
      parent,
      `.${path.basename(target)}.${process.pid}.${Date.now()}.${randomBytes(8).toString('hex')}.tmp`,
    )

    // 1–2. write + fsync temp
    let fh: Awaited<ReturnType<FileFinalizationFsOps['open']>> | undefined
    try {
      fh = await this.ops.open(tmp, 'wx', 0o644)
      await fh.writeFile(canonicalString, 'utf-8')
      await fh.sync()
      await fh.close()
      fh = undefined
    } catch (err) {
      if (fh) {
        try { await fh.close() } catch { /* ignore */ }
      }
      await this.cleanupTemp(tmp)
      throw new TaskOutcomeFinalizationStoreError(
        'io',
        'write_temp',
        'failed to write/fsync finalization temp file',
        err,
      )
    }

    // 3. atomic create-if-absent
    try {
      await this.ops.link(tmp, target)
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code
      if (code === 'EEXIST') {
        // loser path
        let existing: TaskOutcomeFinalization
        try {
          existing = await this.readValidateExisting(target, canonical.runId)
        } catch (readErr) {
          await this.cleanupTemp(tmp)
          throw readErr
        }
        try {
          await this.syncHierarchy(target, 'create_loser_dir_fsync')
        } catch (syncErr) {
          await this.cleanupTemp(tmp)
          throw syncErr
        }
        await this.cleanupTemp(tmp)
        return { created: false, existing }
      }
      await this.cleanupTemp(tmp)
      throw new TaskOutcomeFinalizationStoreError(
        'io',
        'link',
        'failed to link finalization temp to target',
        err,
      )
    }

    // 4. winner: full hierarchy fsync before acknowledging created
    try {
      await this.syncHierarchy(target, 'create_winner_dir_fsync')
    } catch (syncErr) {
      await this.cleanupTemp(tmp)
      throw syncErr
    }
    await this.cleanupTemp(tmp)
    return { created: true, record: snapshotFinalization(canonical) }
  }

  async get(runId: string): Promise<TaskOutcomeFinalization | null> {
    const id = typeof runId === 'string' ? runId.trim() : ''
    if (!id) {
      throw new TaskOutcomeFinalizationStoreError(
        'invalid_record',
        'get_validate',
        'get requires a non-empty runId',
      )
    }
    await this.ensureBase()
    const target = this.targetPath(id)

    let text: string
    try {
      text = await this.ops.readFile(target, 'utf-8')
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') return null
      throw new TaskOutcomeFinalizationStoreError(
        'io',
        'get_read',
        'failed to read finalization record',
        err,
      )
    }

    let parsed: unknown
    try {
      parsed = JSON.parse(text)
    } catch (err) {
      throw corruption(id, 'finalization JSON is malformed', err)
    }

    let record: TaskOutcomeFinalization
    try {
      record = parseAndValidateFinalization(parsed, id)
    } catch (err) {
      if (err instanceof TaskOutcomeFinalizationValidationError) {
        throw corruption(id, err.message, err)
      }
      throw err
    }

    // crash-safe: observed target must have full hierarchy durability confirmed
    await this.syncHierarchy(target, 'get_dir_fsync')
    return record
  }
}
