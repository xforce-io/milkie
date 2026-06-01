import type { IStateStore } from '../types/store.js'

export interface RedisStoreOptions {
  host?:     string
  port?:     number
  db?:       number
  password?: string
  url?:      string
  lazyConnect?:          boolean
  maxRetriesPerRequest?: number
  connectTimeout?:       number
  enableOfflineQueue?:   boolean
  retryStrategy?:        (times: number) => number | void | null
}

export class RedisStore implements IStateStore {
  private client: import('ioredis').Redis | null = null
  private readonly options: RedisStoreOptions

  constructor(options: RedisStoreOptions = {}) {
    this.options = options
  }

  async init(): Promise<void> {
    const { Redis } = await import('ioredis')
    if (this.options.url) {
      this.client = new Redis(this.options.url, {
        lazyConnect:          this.options.lazyConnect,
        maxRetriesPerRequest: this.options.maxRetriesPerRequest,
        connectTimeout:       this.options.connectTimeout ?? 500,
        enableOfflineQueue:   this.options.enableOfflineQueue,
        retryStrategy:        this.options.retryStrategy,
      })
    } else {
      this.client = new Redis({
        host:     this.options.host ?? 'localhost',
        port:     this.options.port ?? 6379,
        db:       this.options.db ?? 0,
        password: this.options.password,
        lazyConnect:          this.options.lazyConnect,
        maxRetriesPerRequest: this.options.maxRetriesPerRequest,
        connectTimeout:       this.options.connectTimeout ?? 500,
        enableOfflineQueue:   this.options.enableOfflineQueue,
        retryStrategy:        this.options.retryStrategy,
      })
    }
    this.client.on('error', () => undefined)
    if (this.options.lazyConnect) {
      await this.client.connect()
    }
  }

  async set(key: string, value: unknown, ttl?: number): Promise<void> {
    if (!this.client) throw new Error('RedisStore is not initialized')
    const serialized = JSON.stringify(value)
    if (ttl) {
      await this.client.setex(key, ttl, serialized)
    } else {
      await this.client.set(key, serialized)
    }
  }

  async get(key: string): Promise<unknown> {
    if (!this.client) throw new Error('RedisStore is not initialized')
    const raw = await this.client.get(key)
    if (raw === null) return undefined
    return JSON.parse(raw) as unknown
  }

  async delete(key: string): Promise<void> {
    if (!this.client) throw new Error('RedisStore is not initialized')
    await this.client.del(key)
  }

  async exists(key: string): Promise<boolean> {
    if (!this.client) throw new Error('RedisStore is not initialized')
    const count = await this.client.exists(key)
    return count > 0
  }

  async list(prefix: string): Promise<Array<{ key: string; value: unknown }>> {
    if (!this.client) throw new Error('RedisStore is not initialized')
    // SCAN (not KEYS) to avoid blocking on large keyspaces.
    const keys: string[] = []
    let cursor = '0'
    do {
      const [next, batch] = await this.client.scan(cursor, 'MATCH', `${prefix}*`, 'COUNT', 100)
      cursor = next
      keys.push(...batch)
    } while (cursor !== '0')
    if (keys.length === 0) return []
    const values = await this.client.mget(keys)
    const out: Array<{ key: string; value: unknown }> = []
    keys.forEach((key, i) => {
      const raw = values[i]
      if (raw !== null && raw !== undefined) out.push({ key, value: JSON.parse(raw) as unknown })
    })
    return out
  }

  async flushdb(): Promise<void> {
    if (!this.client) return
    await this.client.flushdb()
  }

  async disconnect(): Promise<void> {
    if (!this.client) return
    this.client.disconnect()
    this.client = null
  }
}
