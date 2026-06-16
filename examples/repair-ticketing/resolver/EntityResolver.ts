// Portable hierarchical entity resolver core (#167).
//
// This module is the SINGLE source of the usable contract. It is a *pure*,
// filesystem-free library: callers pass already-parsed `schema` and raw `csv`
// content to `loadHierarchicalDict`, then reuse the returned `dict` across any
// number of `lookupEntities` / `commitEntities` calls — no path/fs access. The
// CLI wrappers (bin/entity-resolver, scripts/resolver.ts) are the only layers
// that touch the filesystem; they read the files and feed this core.
//
// JSON-in / JSON-out contract (the three functions below are the API). Callers
// do NOT pass a target level: it is derived from `pinned`. With no level, `lookup`
// searches from the next unfilled level down to the leaf, so a clear full-path
// utterance with empty `pinned` still recalls a deep entity; `commit` resolves the
// selection at the deepest searched level that holds it, so a clear suggestion is
// directly commit-able (#167 item 1). An explicit `level` overrides both.
//   loadHierarchicalDict(schema, csv)                      -> HierarchicalDict
//   lookupEntities({ utterance, pinned, dict })            -> LookupOutput
//   commitEntities({ utterance, selected, pinned, dict })  -> CommitOutput

// ─── schema types ─────────────────────────────────────────────────────────────

export interface LevelSchema {
  name: string
  label: string
  idColumn: string
  labelColumn: string
  parentLevel?: string
  // #167: the CSV columns this level owns for fuzzy matching (e.g. department →
  // [dept_name, dept_alias]). Declared per level — NOT inferred from CSV header
  // order — so a parent never matches on a descendant's name/alias and behavior
  // is independent of physical column layout.
  searchColumns: string[]
}

export interface Schema {
  version: number
  levels: LevelSchema[]
  metaColumns: string[]
}

// ─── loaded dictionary (opaque, reusable, no fs) ───────────────────────────────

export interface HierarchicalDict {
  schema: Schema
  levelOrder: string[]
  index: Map<string, Map<string, EntityRecord>>
}

// ─── public API types ─────────────────────────────────────────────────────────

export interface LookupRequest {
  utterance: string
  // Optional: the portable core contract is `{ utterance, pinned, dict }`. When
  // omitted the target level is derived from `pinned` (the next unfilled level),
  // so in-process callers never need to reimplement level derivation. An explicit
  // level is accepted as an override.
  level?: string
  pinned?: Record<string, string>
  sessionHint?: string
}

export interface LookupOutput {
  candidates: Array<{ id: string; label: string; path: string[]; score: number }>
  options: string[]
  suggested: string | null
}

export interface CommitRequest {
  selected: string
  // Optional — derived from `pinned` when omitted (see `LookupRequest.level`).
  level?: string
  pinned?: Record<string, string>
  // Accepted for contract symmetry with lookup / the CLI wrapper; the commit
  // decision is driven by `selected`, so the utterance is not used here.
  utterance?: string
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

// ─── load ──────────────────────────────────────────────────────────────────────

// Build a reusable dictionary from already-loaded schema + CSV content. No fs:
// CLI/host layers are responsible for reading files and parsing the schema JSON.
export function loadHierarchicalDict(schema: Schema, csv: string): HierarchicalDict {
  validateSchema(schema)
  // #167: validate the schema↔CSV column mapping against the declared HEADER, not the
  // data rows — otherwise an empty or header-only CSV with missing/misspelled mapped
  // columns would load silently into an empty dictionary instead of failing fast.
  validateColumns(schema, new Set(parseHeader(csv)))
  const rows = parseCSV(csv)
  const levelOrder = schema.levels.map(l => l.name)
  const index = buildIndex(schema, rows)
  return { schema, levelOrder, index }
}

// ─── lookup ──────────────────────────────────────────────────────────────────

export function lookupEntities(input: LookupRequest & { dict: HierarchicalDict }): LookupOutput {
  const { dict } = input
  const pinned = input.pinned ?? {}
  const levels = targetLevels(dict, pinned, input.level)

  const query = input.utterance.toLowerCase()

  const scored: Array<EntityRecord & { score: number }> = []
  for (const level of levels) {
    const levelMap = dict.index.get(level)
    if (!levelMap) throw new Error(`Unknown level: "${level}"`)

    // A pin at the *target level itself* is the previously-confirmed value, not an
    // ancestor constraint (an entity's own level never appears in `ancestors`).
    // Filter only by strictly-higher (ancestor) pins, else the already-selected
    // entity would be filtered out whenever the target level is pinned (#167).
    const ancestorPins = ancestorOnly(pinned, level)

    for (const entity of levelMap.values()) {
      if (!matchesPinned(entity, ancestorPins)) continue
      const score = scoreEntity(entity, query)
      if (!query || score > 0) scored.push({ ...entity, score })
    }
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

// ─── commit ────────────────────────────────────────────────────────────────────

export function commitEntities(input: CommitRequest & { dict: HierarchicalDict }): CommitOutput {
  const { dict } = input
  const pinned = input.pinned ?? {}
  const level = input.level ?? commitLevel(dict, pinned, input.selected)
  const levelMap = dict.index.get(level)
  if (!levelMap) throw new Error(`Unknown level: "${level}"`)

  const entity = levelMap.get(input.selected)
  if (!entity) {
    // Per-level confirmation semantics (#167): the selection is judged against
    // the requested target level. An id that resolves at a *different* level is
    // an under-selection — the target level itself has not been chosen yet, so
    // report `missing`. An id that appears at no level at all is `unknown`.
    for (const [, otherMap] of dict.index) {
      if (otherMap !== levelMap && otherMap.has(input.selected)) {
        return { status: 'missing', message: `Entity "${input.selected}" resolves at a different level; target level "${level}" is not yet selected` }
      }
    }
    return { status: 'unknown', message: `Entity "${input.selected}" not found at any level` }
  }

  // A `pinned` entry at the *target level itself* is the previously-confirmed
  // value, not an ancestor constraint (#167 corrected-override case). Separate it
  // out: ancestor constraints are only the strictly-higher levels.
  const ancestorPins = ancestorOnly(pinned, level)
  const sameLevelPin = pinned[level]

  // 1. Ancestor-branch validation: if the selection's actual ancestor chain
  //    conflicts with the pinned ancestors, try to auto-correct to the sibling
  //    under the pinned branch.
  for (const [ancestorLevel, ancestorId] of Object.entries(ancestorPins)) {
    if (entity.ancestors[ancestorLevel] !== ancestorId) {
      const correction = findCorrection(dict, entity, level, ancestorPins)
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

  // 2. Same-level override (#167): ancestors are consistent, but `selected`
  //    differs from the id already confirmed at this level in `pinned`. Honor the
  //    new selection and report `corrected` with the updated level + path.
  if (sameLevelPin !== undefined && sameLevelPin !== input.selected) {
    return {
      status: 'corrected',
      resolved: toResolved(entity),
      correctedLevels: { [level]: input.selected },
    }
  }

  return { status: 'complete', resolved: toResolved(entity) }
}

// ─── EntityResolver — ergonomic in-process wrapper over a loaded dict ───────────
//
// Holds one loaded dictionary so an in-process host (e.g. a milkie adapter) can
// reuse it across calls. It is a thin shell over the functions above and, like
// the core, performs NO filesystem access.

export class EntityResolver {
  private readonly dict: HierarchicalDict

  constructor(dict: HierarchicalDict) {
    this.dict = dict
  }

  static load(schema: Schema, csv: string): EntityResolver {
    return new EntityResolver(loadHierarchicalDict(schema, csv))
  }

  lookup(input: LookupRequest): LookupOutput {
    return lookupEntities({ ...input, dict: this.dict })
  }

  commit(input: CommitRequest): CommitOutput {
    return commitEntities({ ...input, dict: this.dict })
  }
}

// ─── level derivation (shared by CLI wrappers) ─────────────────────────────────

// The CLI wrappers do not take an explicit level: per #167 the next level to
// resolve is derived from how many ancestors are already pinned — the first
// level (root → leaf) whose name is not present in `pinned`. When every level is
// pinned, the deepest level is returned (re-selection at the leaf).
export function nextLevel(dict: HierarchicalDict, pinned: Record<string, string>): string {
  for (const name of dict.levelOrder) {
    if (!(name in pinned)) return name
  }
  return dict.levelOrder[dict.levelOrder.length - 1]!
}

// Which levels a lookup should search. An explicit level is honored verbatim. When
// omitted (the portable/CLI contract — no target level is supplied), search from
// the next unfilled level down to the leaf rather than the single next-unfilled
// level. Otherwise an initial lookup with empty `pinned` is confined to the root
// level, so a clear full-path utterance (e.g. just an assignee name) can never be
// recalled or suggested from the CLI (#167 item 1). With a longer `pinned` prefix
// this naturally narrows to the remaining levels, collapsing to the single leaf
// once every ancestor is pinned.
function targetLevels(
  dict: HierarchicalDict,
  pinned: Record<string, string>,
  level?: string,
): string[] {
  if (level !== undefined) return [level]
  const start = dict.levelOrder.indexOf(nextLevel(dict, pinned))
  return dict.levelOrder.slice(start)
}

// Target level for a commit when none is supplied: the deepest searched level that
// actually holds `selected`, so the clear `lookup.suggested` value is directly
// commit-able under the same level-less CLI contract (#167 item 1). Falls back to
// the next unfilled level when `selected` is absent, letting the not-found branch
// report `missing`/`unknown` as before.
function commitLevel(
  dict: HierarchicalDict,
  pinned: Record<string, string>,
  selected: string,
): string {
  const levels = targetLevels(dict, pinned, undefined)
  for (let i = levels.length - 1; i >= 0; i--) {
    if (dict.index.get(levels[i]!)!.has(selected)) return levels[i]!
  }
  return nextLevel(dict, pinned)
}

// ─── schema / column validation ────────────────────────────────────────────────

function validateSchema(schema: Schema): void {
  const names = new Set(schema.levels.map(l => l.name))
  for (const level of schema.levels) {
    if (level.parentLevel && !names.has(level.parentLevel)) {
      throw new Error(
        `Level "${level.name}" references unknown parentLevel "${level.parentLevel}"`,
      )
    }
  }

  // Linear-path constraint (#162 acceptance): levels must form a single
  // root-to-leaf chain — no fork (two levels sharing a parent), gap, or cycle.
  const levels = schema.levels
  for (let i = 0; i < levels.length; i++) {
    const level = levels[i]!
    if (i === 0) {
      if (level.parentLevel) {
        throw new Error(`Root level "${level.name}" must not declare a parentLevel`)
      }
    } else {
      const expected = levels[i - 1]!.name
      if (level.parentLevel !== expected) {
        throw new Error(
          `Levels must form a single chain: level "${level.name}" must have parentLevel "${expected}" but got "${level.parentLevel ?? '(none)'}"`,
        )
      }
    }
  }
}

function validateColumns(schema: Schema, cols: Set<string>): void {
  // #167: validate against the CSV's declared header columns regardless of data-row
  // count. A CSV with no header at all cannot satisfy any column mapping → fail fast.
  if (cols.size === 0) {
    throw new Error('CSV has no header row — cannot validate the schema↔CSV column mapping')
  }
  for (const level of schema.levels) {
    if (!cols.has(level.idColumn)) {
      throw new Error(`Missing required column "${level.idColumn}" for level "${level.name}"`)
    }
    if (!cols.has(level.labelColumn)) {
      throw new Error(`Missing required column "${level.labelColumn}" for level "${level.name}"`)
    }
    // #167: each level's own searchColumns must map to real CSV columns —
    // otherwise a misspelled alias loads silently and degrades lookup.
    for (const col of level.searchColumns) {
      if (!cols.has(col)) {
        throw new Error(`Missing searchColumn "${col}" for level "${level.name}" but absent from CSV`)
      }
    }
  }
  // #167: metaColumns are part of the schema↔CSV column mapping and must
  // validate too — a misspelled meta column should fail fast, not degrade meta.
  for (const col of schema.metaColumns) {
    if (!cols.has(col)) {
      throw new Error(`Missing metaColumn "${col}" referenced by schema but absent from CSV`)
    }
  }
}

function buildIndex(schema: Schema, rows: Record<string, string>[]): Map<string, Map<string, EntityRecord>> {
  const index = new Map<string, Map<string, EntityRecord>>()
  for (const level of schema.levels) {
    index.set(level.name, new Map())
  }

  for (const row of rows) {
    for (let i = 0; i < schema.levels.length; i++) {
      const levelSchema = schema.levels[i]!
      const id = row[levelSchema.idColumn]
      if (!id) continue

      const levelMap = index.get(levelSchema.name)!

      // Validate parent references (#167 acceptance): a child id is only valid if
      // every ancestor level in the chain also carries a non-empty id in this row.
      // Otherwise the record would load under a non-existent parent (e.g. an
      // assignee with an empty dept_id), so fail at load time.
      for (let j = 0; j < i; j++) {
        const anc = schema.levels[j]!
        if (!row[anc.idColumn]) {
          throw new Error(
            `Invalid hierarchy: ${levelSchema.name} "${id}" has a missing parent reference for level "${anc.name}"`,
          )
        }
      }

      // Build ancestors: all levels before this one
      const ancestors: Record<string, string> = {}
      const ancestorLabels: string[] = []
      for (let j = 0; j < i; j++) {
        const anc = schema.levels[j]!
        ancestors[anc.name] = row[anc.idColumn]!
        ancestorLabels.push(row[anc.labelColumn] ?? '')
      }

      const label = row[levelSchema.labelColumn] ?? ''
      const path = [...ancestorLabels, label]

      // Collect searchable values — only the columns this level explicitly owns
      // (#167, level.searchColumns), so a parent never matches on a descendant's
      // name/alias and the result never depends on CSV column order.
      const searchValues: Record<string, string> = {}
      for (const col of levelSchema.searchColumns) {
        if (row[col] !== undefined) searchValues[col] = row[col]!
      }

      // Collect meta values
      const meta: Record<string, unknown> = {}
      for (const col of schema.metaColumns) {
        if (row[col] !== undefined) meta[col] = row[col]
      }

      const existing = levelMap.get(id)
      if (existing) {
        // #167 item 2: a repeated id must denote the *same* entity. The wide CSV
        // legitimately repeats parent ids across child rows (design §8.2), so an
        // exact repeat is fine — but a repeat whose data diverges is a real
        // conflict that must fail loudly. Silently keeping the first row would hide
        // the conflict and make the loaded record depend on row order.
        assertConsistentDuplicate(
          existing,
          { id, label, ancestors, searchValues, meta },
          levelSchema.name,
          i === schema.levels.length - 1,
        )
        // Same id, identical data — a redundant denormalized row; skip it.
        continue
      }

      levelMap.set(id, { id, label, path, ancestors, meta, searchValues })
    }
  }

  return index
}

// Reject a duplicate id whose data conflicts with the already-indexed record
// (#167 item 2). The level's own identifying fields — label and ancestor branch —
// must match for every level. At the *leaf* the row's aliases/metadata belong to
// the entity itself, so they must match too; at parent levels the wide table
// absorbs descendant columns that legitimately vary across child rows, so those
// are not part of the parent's identity and are not compared here.
function assertConsistentDuplicate(
  existing: EntityRecord,
  incoming: {
    id: string
    label: string
    ancestors: Record<string, string>
    searchValues: Record<string, string>
    meta: Record<string, unknown>
  },
  levelName: string,
  isLeaf: boolean,
): void {
  if (existing.label !== incoming.label) {
    throw new Error(
      `Duplicate id "${incoming.id}" at level "${levelName}" with conflicting label: "${existing.label}" vs "${incoming.label}"`,
    )
  }
  for (const [k, v] of Object.entries(incoming.ancestors)) {
    if (existing.ancestors[k] !== v) {
      throw new Error(
        `Duplicate id "${incoming.id}" at level "${levelName}" with conflicting ancestor "${k}": "${existing.ancestors[k]}" vs "${v}"`,
      )
    }
  }
  if (!isLeaf) return
  for (const [k, v] of Object.entries(incoming.searchValues)) {
    if (existing.searchValues[k] !== v) {
      throw new Error(
        `Duplicate id "${incoming.id}" at level "${levelName}" with conflicting value "${k}": "${existing.searchValues[k]}" vs "${v}"`,
      )
    }
  }
  for (const [k, v] of Object.entries(incoming.meta)) {
    if (existing.meta[k] !== v) {
      throw new Error(
        `Duplicate id "${incoming.id}" at level "${levelName}" with conflicting metadata "${k}": "${String(existing.meta[k])}" vs "${String(v)}"`,
      )
    }
  }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

// #167: the declared header columns, independent of data rows — used to validate the
// schema↔CSV column mapping even for an empty or header-only CSV.
function parseHeader(content: string): string[] {
  const lines = content.split('\n').map(l => l.trim()).filter(l => l.length > 0)
  return lines.length === 0 ? [] : parseCSVLine(lines[0]!)
}

function parseCSV(content: string): Record<string, string>[] {
  const lines = content.split('\n').map(l => l.trim()).filter(l => l.length > 0)
  if (lines.length < 2) return []
  const headers = parseCSVLine(lines[0]!)
  return lines.slice(1).map(line => {
    const values = parseCSVLine(line)
    const row: Record<string, string> = {}
    headers.forEach((h, i) => { row[h] = values[i] ?? '' })
    return row
  })
}

// RFC 4180 line parser: supports double-quoted fields containing the comma
// separator and "" as an escaped quote. Unquoted fields are trimmed; quoted
// fields are taken verbatim (minus the surrounding quotes).
function parseCSVLine(line: string): string[] {
  const fields: string[] = []
  let cur = ''
  let inQuotes = false
  let quoted = false
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]!
    if (inQuotes) {
      if (ch === '"') {
        if (line[i + 1] === '"') { cur += '"'; i++ }
        else inQuotes = false
      } else {
        cur += ch
      }
    } else if (ch === '"') {
      inQuotes = true
      quoted = true
    } else if (ch === ',') {
      fields.push(quoted ? cur : cur.trim())
      cur = ''
      quoted = false
    } else {
      cur += ch
    }
  }
  fields.push(quoted ? cur : cur.trim())
  return fields
}

// Strip the target level's own pin, leaving only strictly-higher (ancestor)
// constraints. An entity's own level is never present in `ancestors`, so a pin
// at the target level must not be used for topological filtering (#167).
function ancestorOnly(pinned: Record<string, string>, level: string): Record<string, string> {
  const ancestors: Record<string, string> = {}
  for (const [k, v] of Object.entries(pinned)) {
    if (k !== level) ancestors[k] = v
  }
  return ancestors
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
): number {
  if (!query) return 0.5

  let best = 0
  // #167: score against the entity's own search values (its level's columns only).
  for (const raw of Object.values(entity.searchValues)) {
    const val = raw.toLowerCase()
    if (!val) continue
    if (val === query) { best = Math.max(best, 1.0); break }
    if (val.includes(query)) best = Math.max(best, 0.7)
    // Reverse direction: a natural utterance (e.g. "请把这个工单派给王芳")
    // contains the field value, not the other way around.
    else if (query.includes(val)) best = Math.max(best, 0.9)
  }
  // Boost exact / contained label match
  const label = entity.label.toLowerCase()
  if (label === query) best = Math.max(best, 1.0)
  else if (label.includes(query)) best = Math.max(best, 0.8)
  else if (query.includes(label)) best = Math.max(best, 0.9)

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

function findCorrection(
  dict: HierarchicalDict,
  selected: EntityRecord,
  level: string,
  pinned: Record<string, string>,
): { type: 'found'; entity: EntityRecord } | { type: 'ambiguous' } | { type: 'none' } {
  const levelMap = dict.index.get(level)!

  const candidates = [...levelMap.values()].filter(
    e => e.id !== selected.id && matchesPinned(e, pinned),
  )

  if (candidates.length === 0) return { type: 'none' }

  const sameLabel = candidates.filter(e => e.label === selected.label)
  if (sameLabel.length === 1) return { type: 'found', entity: sameLabel[0]! }
  if (sameLabel.length > 1) return { type: 'ambiguous' }

  return { type: 'none' }
}
