// Region store for the context substrate.
// Spec: docs/superpowers/specs/2026-05-25-context-region-substrate-design.md §4.2

import type { Region, RegionInput, RegionSnapshot } from './Region'

export type Clock = () => number

export interface RegionChangeDelta {
  kind:    'added' | 'removed'
  id:      string
  /** Present on 'added' deltas only (so consumers can index into section without re-reading the region). */
  section?: string
  /** Present on 'added' deltas only. */
  target?:  'system' | 'message' | 'tool'
  /** Present on 'added' deltas only. */
  stability?: 'immutable' | 'session-stable' | 'turn-stable' | 'volatile'
}

export interface ContextRegionsOptions {
  /** Fired on every successful set/delete. Optional — substrate stays pure-ish without it. */
  onChange?: (delta: RegionChangeDelta) => void
}

export class ContextRegions {
  private readonly regions = new Map<string, Region>()
  private epoch = 0

  constructor(
    private readonly clock: Clock,
    private readonly options: ContextRegionsOptions = {},
  ) {}

  set(id: string, input: RegionInput): void {
    const existing = this.regions.get(id)
    const createdAt = existing?.createdAt ?? this.clock()
    const region: Region = { id, createdAt, ...input }
    this.regions.set(id, region)
    this.epoch++
    this.options.onChange?.({
      kind:      'added',
      id,
      section:   region.section,
      target:    region.target,
      stability: region.stability,
    })
  }

  delete(id: string): boolean {
    const existed = this.regions.delete(id)
    if (existed) {
      this.epoch++
      this.options.onChange?.({ kind: 'removed', id })
    }
    return existed
  }

  get(id: string): Region | undefined {
    return this.regions.get(id)
  }

  getEpoch(): number {
    return this.epoch
  }

  // Public iterator for the assemble layer. Leading underscore signals
  // "internal to substrate — not for general consumers". Iteration order
  // is unspecified: assemble must do its own sorting.
  _allRegions(): IterableIterator<Region> {
    return this.regions.values()
  }

  snapshot(): RegionSnapshot {
    return {
      epoch:   this.epoch,
      regions: [...this.regions.values()],
    }
  }

  restore(snap: RegionSnapshot): void {
    this.regions.clear()
    for (const r of snap.regions) {
      this.regions.set(r.id, r)
    }
    this.epoch = snap.epoch
  }
}
