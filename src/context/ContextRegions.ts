// Region store for the context substrate.
// Spec: docs/superpowers/specs/2026-05-25-context-region-substrate-design.md §4.2

import type { Region, RegionInput, RegionSnapshot } from './Region'

export type Clock = () => number

export class ContextRegions {
  private readonly regions = new Map<string, Region>()
  private epoch = 0

  constructor(private readonly clock: Clock) {}

  set(id: string, input: RegionInput): void {
    const existing = this.regions.get(id)
    const createdAt = existing?.createdAt ?? this.clock()
    const region: Region = { id, createdAt, ...input }
    this.regions.set(id, region)
    this.epoch++
  }

  delete(id: string): boolean {
    const existed = this.regions.delete(id)
    if (existed) this.epoch++
    return existed
  }

  get(id: string): Region | undefined {
    return this.regions.get(id)
  }

  getEpoch(): number {
    return this.epoch
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
