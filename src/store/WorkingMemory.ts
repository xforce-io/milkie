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
    // Deep-clone (JSON round-trip) so the returned snapshot is frozen at capture
    // time: callers (prompt assembly, checkpoint, wm.mutated events) must not be
    // aliased to live state — later set()/append()/in-place mutation must not
    // change it. WM holds JSON-serialisable state, so the round-trip is lossless.
    return JSON.parse(JSON.stringify({
      data: Object.fromEntries(this.data.entries()),
      log:  this.log,
    }))
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
