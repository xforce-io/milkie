import { promises as fs } from 'fs'
import path from 'path'
import { contentAddressForCanonicalBytes } from './hash.js'

export interface ITraceObjectStore {
  putCanonical(bytes: string): Promise<string>
  getCanonical(hash: string): Promise<string | undefined>
  has(hash: string): Promise<boolean>
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

export class FileTraceObjectStore implements ITraceObjectStore {
  constructor(private readonly baseDir: string) {}

  private fileFor(hash: string): string {
    const hex = assertSupportedHash(hash)
    return path.join(this.baseDir, 'sha256', hex.slice(0, 2), hex.slice(2))
  }

  async putCanonical(bytes: string): Promise<string> {
    const hash = contentAddressForCanonicalBytes(bytes)
    const file = this.fileFor(hash)
    if (await this.has(hash)) return hash

    await fs.mkdir(path.dirname(file), { recursive: true })
    const tmp = `${file}.${process.pid}.${Date.now()}.${Math.random().toString(16).slice(2)}.tmp`
    try {
      await fs.writeFile(tmp, bytes, { encoding: 'utf-8', flag: 'wx' })
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
}
