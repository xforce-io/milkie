import { promises as fs } from 'fs'
import path from 'path'
import { contentAddressForCanonicalBytes } from './hash.js'
import {
  fsyncDirectory,
  fsyncStoreHierarchy,
  mkdirDurable,
} from './durableFs.js'

export interface ITraceObjectStore {
  putCanonical(bytes: string): Promise<string>
  getCanonical(hash: string): Promise<string | undefined>
  has(hash: string): Promise<boolean>
}

/**
 * Optional crash-safe durability capability for object evidence used by
 * task outcome finalization (#227 / s-017). Memory stores do not implement this.
 */
export interface ICrashSafeTraceObjectStore extends ITraceObjectStore {
  readonly durability: 'crash-safe'
  /**
   * Confirm each object's inode and the directory chain from the leaf up to
   * the configured durable root are fsync'd. Hashes must already be readable.
   */
  confirmObjectsDurable(hashes: readonly `sha256:${string}`[]): Promise<void>
}

export function isCrashSafeTraceObjectStore(
  store: ITraceObjectStore,
): store is ICrashSafeTraceObjectStore {
  const candidate = store as ICrashSafeTraceObjectStore
  return (
    candidate !== null &&
    typeof candidate === 'object' &&
    candidate.durability === 'crash-safe' &&
    typeof candidate.confirmObjectsDurable === 'function'
  )
}

function assertSupportedHash(hash: string): string {
  const prefix = 'sha256:'
  if (!hash.startsWith(prefix)) {
    throw new Error(`Unsupported trace object hash: ${hash}`)
  }
  const hex = hash.slice(prefix.length)
  if (!/^[a-f0-9]{64}$/.test(hex)) {
    throw new Error(`Invalid sha256 trace object hash: ${hash}`)
  }
  return hex
}

function assertHashMatches(hash: string, bytes: string): void {
  const actual = contentAddressForCanonicalBytes(bytes)
  if (actual !== hash) {
    throw new Error(`Trace object hash mismatch: expected ${hash}, got ${actual}`)
  }
}

export class MemoryTraceObjectStore implements ITraceObjectStore {
  /** Process-lifetime only; does not implement crash-safe durability. */
  readonly durability = 'process' as const
  private readonly objects = new Map<string, string>()

  async putCanonical(bytes: string): Promise<string> {
    const hash = contentAddressForCanonicalBytes(bytes)
    const existing = this.objects.get(hash)
    if (existing !== undefined && existing !== bytes) {
      throw new Error(`Trace object hash collision or corruption for ${hash}`)
    }
    this.objects.set(hash, bytes)
    return hash
  }

  async getCanonical(hash: string): Promise<string | undefined> {
    assertSupportedHash(hash)
    const bytes = this.objects.get(hash)
    if (bytes !== undefined) assertHashMatches(hash, bytes)
    return bytes
  }

  async has(hash: string): Promise<boolean> {
    assertSupportedHash(hash)
    return this.objects.has(hash)
  }
}

export class FileTraceObjectStore implements ICrashSafeTraceObjectStore {
  readonly durability = 'crash-safe' as const
  private baseReady = false

  constructor(private readonly baseDir: string) {}

  private fileFor(hash: string): string {
    const hex = assertSupportedHash(hash)
    return path.join(this.baseDir, 'sha256', hex.slice(0, 2), hex.slice(2))
  }

  private async ensureBase(): Promise<void> {
    if (this.baseReady) return
    await mkdirDurable(this.baseDir)
    this.baseReady = true
  }

  async putCanonical(bytes: string): Promise<string> {
    const hash = contentAddressForCanonicalBytes(bytes)
    const file = this.fileFor(hash)
    if (await this.has(hash)) return hash

    await this.ensureBase()
    await mkdirDurable(path.dirname(file))
    const tmp = `${file}.${process.pid}.${Date.now()}.${Math.random().toString(16).slice(2)}.tmp`
    try {
      await fs.writeFile(tmp, bytes, { encoding: 'utf-8', flag: 'wx' })
      // Object bytes themselves are not crash-safe until confirmObjectsDurable;
      // still fsync the temp inode so a later confirm has durable file content.
      const tmpFh = await fs.open(tmp, 'r')
      try {
        await tmpFh.sync()
      } finally {
        await tmpFh.close()
      }
      await fs.link(tmp, file)
      await fs.rm(tmp, { force: true })
    } catch (err) {
      await fs.rm(tmp, { force: true }).catch(() => undefined)
      if ((err as NodeJS.ErrnoException).code !== 'EEXIST') throw err
      const existing = await fs.readFile(file, 'utf-8')
      if (existing !== bytes) {
        throw new Error(`Trace object hash collision or corruption for ${hash}`)
      }
    }
    return hash
  }

  async getCanonical(hash: string): Promise<string | undefined> {
    const file = this.fileFor(hash)
    try {
      const bytes = await fs.readFile(file, 'utf-8')
      assertHashMatches(hash, bytes)
      return bytes
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') return undefined
      throw err
    }
  }

  async has(hash: string): Promise<boolean> {
    return (await this.getCanonical(hash)) !== undefined
  }

  /**
   * #227: fsync each object inode and the directory chain from the leaf up to
   * `baseDir` and its parent so crash-safe finalization can treat object
   * evidence as durable across process takeover.
   */
  async confirmObjectsDurable(hashes: readonly `sha256:${string}`[]): Promise<void> {
    await this.ensureBase()

    // De-dupe while preserving order of first occurrence.
    const unique: string[] = []
    const seen = new Set<string>()
    for (const h of hashes) {
      if (seen.has(h)) continue
      seen.add(h)
      unique.push(h)
    }

    for (const hash of unique) {
      const file = this.fileFor(hash)
      // Must already be readable; getCanonical validates hash match.
      const bytes = await this.getCanonical(hash)
      if (bytes === undefined) {
        throw new Error(`confirmObjectsDurable: object not found for ${hash}`)
      }
      await fsyncDirectory(file)

      await fsyncStoreHierarchy({
        leafDir: path.dirname(file),
        baseDir: this.baseDir,
        syncDirectory: fsyncDirectory,
      })
    }
  }
}
