import fs from 'fs'

// ─── schema types ─────────────────────────────────────────────────────────────

interface LevelSchema {
  name: string
  label: string
  idColumn: string
  labelColumn: string
  parentLevel?: string
}

interface Schema {
  version: number
  levels: LevelSchema[]
  searchColumns: string[]
  metaColumns: string[]
}

// ─── public API types ─────────────────────────────────────────────────────────

export interface LookupInput {
  op: 'lookup'
  query: string
  context: {
    level: string
    pinned?: Record<string, string>
    sessionHint?: string
  }
}

export interface LookupOutput {
  candidates: Array<{ id: string; label: string; path: string[]; score: number }>
  options: string[]
  suggested: string | null
}

export interface CommitInput {
  op: 'commit'
  selected: string
  context: {
    level: string
    pinned?: Record<string, string>
  }
}

export interface ResolvedEntity {
  id: string
  label: string
  path: string[]
  meta: Record<string, unknown>
}

export type CommitOutput =
  | { status: 'complete'; resolved: ResolvedEntity }
  | { status: 'corrected'; resolved: ResolvedEntity; correctedLevels: Record<string, string> }
  | { status: 'invalid_selection'; message: string }
  | { status: 'missing'; message: string }
  | { status: 'ambiguous'; message: string }
  | { status: 'unknown'; message: string }

// ─── internal ────────────────────────────────────────────────────────────────

interface EntityRecord {
  id: string
  label: string
  path: string[]
  ancestors: Record<string, string>
  meta: Record<string, unknown>
  searchValues: Record<string, string>
}

// ─── EntityResolver ───────────────────────────────────────────────────────────

export class EntityResolver {
  private schema: Schema
  private levelOrder: string[]
  private index: Map<string, Map<string, EntityRecord>>

  constructor(opts: { schemaPath: string; dataPath: string }) {
    this.schema = JSON.parse(fs.readFileSync(opts.schemaPath, 'utf-8')) as Schema
    this.validateSchema()
    const rows = parseCSV(fs.readFileSync(opts.dataPath, 'utf-8'))
    this.validateColumns(rows)
    this.levelOrder = this.schema.levels.map(l => l.name)
    this.index = this.buildIndex(rows)
  }

  lookup(input: LookupInput): LookupOutput {
    const levelMap = this.index.get(input.context.level)
    if (!levelMap) throw new Error(`Unknown level: "${input.context.level}"`)

    const pinned = input.context.pinned ?? {}
    const query = input.query.toLowerCase()

    const scored: Array<EntityRecord & { score: number }> = []
    for (const entity of levelMap.values()) {
      if (!matchesPinned(entity, pinned)) continue
      const score = scoreEntity(entity, query, this.schema.searchColumns)
      if (!query || score > 0) scored.push({ ...entity, score })
    }

    scored.sort((a, b) => b.score - a.score || a.id.localeCompare(b.id))

    const candidates = scored.map(e => ({ id: e.id, label: e.label, path: e.path, score: e.score }))
    const options = candidates.map(c => c.id)

    let suggested: string | null = null
    if (candidates.length === 1) {
      suggested = candidates[0]!.id
    } else if (
      candidates.length > 1 &&
      candidates[0]!.score >= 0.9 &&
      candidates[0]!.score - candidates[1]!.score > 0.2
    ) {
      suggested = candidates[0]!.id
    }

    return { candidates, options, suggested }
  }

  commit(input: CommitInput): CommitOutput {
    const levelMap = this.index.get(input.context.level)
    if (!levelMap) throw new Error(`Unknown level: "${input.context.level}"`)

    const entity = levelMap.get(input.selected)
    if (!entity) {
      for (const [, otherMap] of this.index) {
        if (otherMap !== levelMap && otherMap.has(input.selected)) {
          return { status: 'unknown', message: `Entity "${input.selected}" exists but not at level "${input.context.level}"` }
        }
      }
      return { status: 'missing', message: `Entity "${input.selected}" not found` }
    }

    const pinned = input.context.pinned ?? {}
    for (const [ancestorLevel, ancestorId] of Object.entries(pinned)) {
      if (entity.ancestors[ancestorLevel] !== ancestorId) {
        const correction = this.findCorrection(entity, input.context.level, pinned)
        if (correction.type === 'found') {
          return {
            status: 'corrected',
            resolved: toResolved(correction.entity),
            correctedLevels: computeCorrectedLevels(entity, correction.entity),
          }
        }
        if (correction.type === 'ambiguous') {
          return { status: 'ambiguous', message: `Auto-correction is ambiguous for "${input.selected}" under the pinned context` }
        }
        return { status: 'invalid_selection', message: `Entity "${input.selected}" does not match the pinned context and no correction is possible` }
      }
    }

    return { status: 'complete', resolved: toResolved(entity) }
  }

  private validateSchema(): void {
    const names = new Set(this.schema.levels.map(l => l.name))
    for (const level of this.schema.levels) {
      if (level.parentLevel && !names.has(level.parentLevel)) {
        throw new Error(
          `Level "${level.name}" references unknown parentLevel "${level.parentLevel}"`,
        )
      }
    }
  }

  private validateColumns(rows: Record<string, string>[]): void {
    if (rows.length === 0) return
    const cols = new Set(Object.keys(rows[0]!))
    for (const level of this.schema.levels) {
      if (!cols.has(level.idColumn)) {
        throw new Error(`Missing required column "${level.idColumn}" for level "${level.name}"`)
      }
      if (!cols.has(level.labelColumn)) {
        throw new Error(`Missing required column "${level.labelColumn}" for level "${level.name}"`)
      }
    }
  }

  private buildIndex(rows: Record<string, string>[]): Map<string, Map<string, EntityRecord>> {
    const index = new Map<string, Map<string, EntityRecord>>()
    for (const level of this.schema.levels) {
      index.set(level.name, new Map())
    }

    for (const row of rows) {
      for (let i = 0; i < this.schema.levels.length; i++) {
        const levelSchema = this.schema.levels[i]!
        const id = row[levelSchema.idColumn]
        if (!id) continue

        const levelMap = index.get(levelSchema.name)!

        // Build ancestors: all levels before this one
        const ancestors: Record<string, string> = {}
        const ancestorLabels: string[] = []
        for (let j = 0; j < i; j++) {
          const anc = this.schema.levels[j]!
          ancestors[anc.name] = row[anc.idColumn] ?? ''
          ancestorLabels.push(row[anc.labelColumn] ?? '')
        }

        const label = row[levelSchema.labelColumn] ?? ''
        const path = [...ancestorLabels, label]

        // Collect searchable values
        const searchValues: Record<string, string> = {}
        for (const col of this.schema.searchColumns) {
          if (row[col] !== undefined) searchValues[col] = row[col]!
        }

        // Collect meta values
        const meta: Record<string, unknown> = {}
        for (const col of this.schema.metaColumns) {
          if (row[col] !== undefined) meta[col] = row[col]
        }

        const existing = levelMap.get(id)
        if (existing) {
          // Same id must have same ancestors (id uniqueness per branch)
          for (const [k, v] of Object.entries(ancestors)) {
            if (existing.ancestors[k] !== v) {
              throw new Error(
                `Duplicate id "${id}" at level "${levelSchema.name}" with conflicting ancestor "${k}": "${existing.ancestors[k]}" vs "${v}"`,
              )
            }
          }
          // Already indexed this entity (same id, same ancestors) — skip duplicate rows
          continue
        }

        levelMap.set(id, { id, label, path, ancestors, meta, searchValues })
      }
    }

    return index
  }

  private findCorrection(
    selected: EntityRecord,
    level: string,
    pinned: Record<string, string>,
  ): { type: 'found'; entity: EntityRecord } | { type: 'ambiguous' } | { type: 'none' } {
    const levelMap = this.index.get(level)!

    const candidates = [...levelMap.values()].filter(
      e => e.id !== selected.id && matchesPinned(e, pinned),
    )

    if (candidates.length === 0) return { type: 'none' }

    const sameLabel = candidates.filter(e => e.label === selected.label)
    if (sameLabel.length === 1) return { type: 'found', entity: sameLabel[0]! }
    if (sameLabel.length > 1) return { type: 'ambiguous' }

    return { type: 'none' }
  }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

function parseCSV(content: string): Record<string, string>[] {
  const lines = content.split('\n').map(l => l.trim()).filter(l => l.length > 0)
  if (lines.length < 2) return []
  const headers = lines[0]!.split(',').map(h => h.trim())
  return lines.slice(1).map(line => {
    const values = line.split(',').map(v => v.trim())
    const row: Record<string, string> = {}
    headers.forEach((h, i) => { row[h] = values[i] ?? '' })
    return row
  })
}

function matchesPinned(entity: EntityRecord, pinned: Record<string, string>): boolean {
  for (const [k, v] of Object.entries(pinned)) {
    if (entity.ancestors[k] !== v) return false
  }
  return true
}

function scoreEntity(
  entity: EntityRecord,
  query: string,
  searchColumns: string[],
): number {
  if (!query) return 0.5

  let best = 0
  for (const col of searchColumns) {
    const val = (entity.searchValues[col] ?? '').toLowerCase()
    if (!val) continue
    if (val === query) { best = Math.max(best, 1.0); break }
    if (val.includes(query)) best = Math.max(best, 0.7)
  }
  // Boost exact label match
  if (entity.label.toLowerCase() === query) best = Math.max(best, 1.0)
  else if (entity.label.toLowerCase().includes(query)) best = Math.max(best, 0.8)

  return best
}

function toResolved(entity: EntityRecord): ResolvedEntity {
  return { id: entity.id, label: entity.label, path: entity.path, meta: entity.meta }
}

function computeCorrectedLevels(selected: EntityRecord, correction: EntityRecord): Record<string, string> {
  const diff: Record<string, string> = {}
  for (const [k, v] of Object.entries(correction.ancestors)) {
    if (selected.ancestors[k] !== v) diff[k] = v
  }
  return diff
}
