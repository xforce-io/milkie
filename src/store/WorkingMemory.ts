export interface WorkingMemoryEntry {
  type:    string
  content: unknown
  ts:      number
}

export class WorkingMemory {
  private data: Map<string, unknown> = new Map()
  private log:  WorkingMemoryEntry[] = []

  set(key: string, value: unknown): void {
    this.data.set(key, value)
  }

  get(key: string): unknown {
    return this.data.get(key)
  }

  has(key: string): boolean {
    return this.data.has(key)
  }

  delete(key: string): void {
    this.data.delete(key)
  }

  append(entry: Omit<WorkingMemoryEntry, 'ts'>): void {
    this.log.push({ ...entry, ts: Date.now() })
  }

  getLog(): WorkingMemoryEntry[] {
    return [...this.log]
  }

  toJSON(): unknown {
    return {
      data: Object.fromEntries(this.data.entries()),
      log:  this.log,
    }
  }

  static fromJSON(raw: unknown): WorkingMemory {
    const wm = new WorkingMemory()
    if (raw && typeof raw === 'object') {
      const r = raw as { data?: Record<string, unknown>; log?: WorkingMemoryEntry[] }
      if (r.data) {
        for (const [k, v] of Object.entries(r.data)) {
          wm.data.set(k, v)
        }
      }
      if (r.log) {
        wm.log = r.log
      }
    }
    return wm
  }
}
