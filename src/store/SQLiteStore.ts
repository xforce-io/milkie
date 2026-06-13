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
    let Database: typeof import('better-sqlite3').default
    try {
      Database = (await import('better-sqlite3')).default
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      if (msg.includes('NODE_MODULE_VERSION')) {
        const built   = msg.match(/NODE_MODULE_VERSION (\d+)\./)?.[1] ?? '?'
        const runtime = msg.match(/requires NODE_MODULE_VERSION (\d+)/)?.[1] ?? process.versions.modules
        throw new Error(
          `better-sqlite3 ABI mismatch: native module was built for NODE_MODULE_VERSION ${built}, ` +
          `but this Node.js requires NODE_MODULE_VERSION ${runtime} (Node ${process.version}). ` +
          `Run \`npm rebuild better-sqlite3\` in the milkie project directory on Node ${process.version}.`
        )
      }
      throw err
    }
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

  async list(prefix: string): Promise<Array<{ key: string; value: unknown }>> {
    // Escape LIKE wildcards in the (normally wildcard-free) prefix for safety.
    const escaped = prefix.replace(/[\\%_]/g, c => '\\' + c)
    const stmt = this.db.prepare(
      "SELECT key, value FROM kv WHERE key LIKE ? ESCAPE '\\' AND (expires IS NULL OR expires > ?)"
    )
    const rows = stmt.all(escaped + '%', Date.now()) as Array<{ key: string; value: string }>
    return rows.map(r => ({ key: r.key, value: JSON.parse(r.value) as unknown }))
  }

  close(): void {
    this.db.close()
  }
}
