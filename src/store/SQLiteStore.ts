import type { IStateStore } from '../types/store.js'

export interface SQLiteStoreOptions {
  path: string
}

export class SQLiteStore implements IStateStore {
  private db!: import('better-sqlite3').Database
  private readonly path: string

  constructor(options: SQLiteStoreOptions) {
    this.path = options.path
  }

  async init(): Promise<void> {
    const Database = (await import('better-sqlite3')).default
    this.db = new Database(this.path)
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS kv (
        key     TEXT PRIMARY KEY,
        value   TEXT NOT NULL,
        expires INTEGER
      )
    `)
  }

  async set(key: string, value: unknown, ttl?: number): Promise<void> {
    const expires = ttl ? Date.now() + ttl * 1000 : null
    const stmt = this.db.prepare(
      'INSERT OR REPLACE INTO kv (key, value, expires) VALUES (?, ?, ?)'
    )
    stmt.run(key, JSON.stringify(value), expires)
  }

  async get(key: string): Promise<unknown> {
    const stmt = this.db.prepare('SELECT value, expires FROM kv WHERE key = ?')
    const row = stmt.get(key) as { value: string; expires: number | null } | undefined
    if (!row) return undefined
    if (row.expires && Date.now() > row.expires) {
      await this.delete(key)
      return undefined
    }
    return JSON.parse(row.value) as unknown
  }

  async delete(key: string): Promise<void> {
    this.db.prepare('DELETE FROM kv WHERE key = ?').run(key)
  }

  async exists(key: string): Promise<boolean> {
    const result = await this.get(key)
    return result !== undefined
  }

  close(): void {
    this.db.close()
  }
}
