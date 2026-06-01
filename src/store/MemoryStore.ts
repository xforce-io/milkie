import type { IStateStore } from '../types/store.js'

interface Entry {
  value:   unknown
  expires: number | null
}

export class MemoryStore implements IStateStore {
  private store: Map<string, Entry> = new Map()

  async set(key: string, value: unknown, ttl?: number): Promise<void> {
    this.store.set(key, {
      value,
      expires: ttl ? Date.now() + ttl * 1000 : null,
    })
  }

  async get(key: string): Promise<unknown> {
    const entry = this.store.get(key)
    if (!entry) return undefined
    if (entry.expires && Date.now() > entry.expires) {
      this.store.delete(key)
      return undefined
    }
    return entry.value
  }

  async delete(key: string): Promise<void> {
    this.store.delete(key)
  }

  async exists(key: string): Promise<boolean> {
    const entry = this.store.get(key)
    if (!entry) return false
    if (entry.expires && Date.now() > entry.expires) {
      this.store.delete(key)
      return false
    }
    return true
  }

  async list(prefix: string): Promise<Array<{ key: string; value: unknown }>> {
    const out: Array<{ key: string; value: unknown }> = []
    const now = Date.now()
    for (const [key, entry] of this.store) {
      if (!key.startsWith(prefix)) continue
      if (entry.expires && now > entry.expires) {
        this.store.delete(key)
        continue
      }
      out.push({ key, value: entry.value })
    }
    return out
  }

  clear(): void {
    this.store.clear()
  }
}
