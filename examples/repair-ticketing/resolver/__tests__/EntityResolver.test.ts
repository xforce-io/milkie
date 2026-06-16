import fs from 'fs'
import path from 'path'
import {
  loadHierarchicalDict,
  lookupEntities,
  commitEntities,
  nextLevel,
  EntityResolver,
  type Schema,
  type HierarchicalDict,
} from '../EntityResolver'

const SCHEMA_PATH = path.join(__dirname, '../schema.json')
const DATA_PATH = path.join(__dirname, '../data.csv')

// ─── helpers ─────────────────────────────────────────────────────────────────
//
// The core is filesystem-free (#167): tests read schema/CSV themselves and pass
// already-loaded content. These helpers exist only to keep the cases terse.

function baseSchema(): Schema {
  return JSON.parse(fs.readFileSync(SCHEMA_PATH, 'utf-8')) as Schema
}

function baseCsv(): string {
  return fs.readFileSync(DATA_PATH, 'utf-8')
}

function load(schema: Schema = baseSchema(), csv: string = baseCsv()): HierarchicalDict {
  return loadHierarchicalDict(schema, csv)
}

// ─── loading ─────────────────────────────────────────────────────────────────

describe('loadHierarchicalDict — loading', () => {
  it('loads valid schema + CSV (already-loaded, no fs in core) without throwing', () => {
    expect(() => load()).not.toThrow()
  })

  it('reuses one dictionary across multiple lookups without re-loading', () => {
    const dict = load()
    const a = lookupEntities({ utterance: '王芳', level: 'assignee', dict })
    const b = lookupEntities({ utterance: '张', level: 'assignee', dict })
    expect(a.candidates[0]!.id).toBe('E008')
    expect(b.candidates.length).toBeGreaterThan(1)
  })

  it('throws on missing idColumn in CSV', () => {
    const csv = 'site_name,bldg_id,bldg_name,dept_id,dept_name,dept_alias,emp_id,emp_name,emp_alias,emp_email,emp_phone,dept_head\n总部,B01,主楼,D03,IT网络部,网络组,E007,张伟,小张,zhangwei@corp.com,13800138007,李明\n'
    expect(() => load(baseSchema(), csv)).toThrow(/site_id/)
  })

  it('throws on missing labelColumn in CSV', () => {
    const csv = 'site_id,bldg_id,bldg_name,dept_id,dept_name,dept_alias,emp_id,emp_name,emp_alias,emp_email,emp_phone,dept_head\nS01,B01,主楼,D03,IT网络部,网络组,E007,张伟,小张,zhangwei@corp.com,13800138007,李明\n'
    expect(() => load(baseSchema(), csv)).toThrow(/site_name/)
  })

  it('throws when a searchColumn maps to a column absent from the CSV (#167 item 4)', () => {
    const schema = baseSchema()
    const assignee = schema.levels[schema.levels.length - 1]!
    assignee.searchColumns = [...assignee.searchColumns, 'emp_nikname'] // misspelled alias column on the assignee level
    expect(() => load(schema)).toThrow(/emp_nikname/)
  })

  it('throws when a metaColumn maps to a column absent from the CSV (#167 item 4)', () => {
    const schema = baseSchema()
    schema.metaColumns = [...schema.metaColumns, 'emp_titel'] // misspelled meta column
    expect(() => load(schema)).toThrow(/emp_titel/)
  })

  // #167: column-mapping validation must run against the declared HEADER, not the data
  // rows — otherwise an empty / header-only CSV with missing mapped columns loads silently.
  it('validates columns against the header even for a header-only CSV (#167)', () => {
    const headerMissingId = 'site_name,bldg_id,bldg_name,dept_id,dept_name,dept_alias,emp_id,emp_name,emp_alias,emp_email,emp_phone,dept_head\n'
    expect(() => load(baseSchema(), headerMissingId)).toThrow(/site_id/) // no data rows, still caught
  })

  it('throws on an empty CSV with no header (#167)', () => {
    expect(() => load(baseSchema(), '')).toThrow(/header/i)
  })

  it('loads a valid header with zero data rows without throwing (empty dictionary) (#167)', () => {
    const headerOnly = 'site_id,site_name,bldg_id,bldg_name,dept_id,dept_name,dept_alias,emp_id,emp_name,emp_alias,emp_email,emp_phone,dept_head\n'
    expect(() => load(baseSchema(), headerOnly)).not.toThrow()
    const dict = load(baseSchema(), headerOnly)
    expect(lookupEntities({ utterance: '王芳', level: 'assignee', dict }).candidates).toHaveLength(0)
  })

  it('throws when a level references a non-existent parentLevel', () => {
    const schema = baseSchema()
    schema.levels = [
      { name: 'site', label: '站点', idColumn: 'site_id', labelColumn: 'site_name', searchColumns: ['site_name'] },
      { name: 'assignee', label: '负责人', idColumn: 'emp_id', labelColumn: 'emp_name', parentLevel: 'ghost_level', searchColumns: ['emp_name'] },
    ]
    expect(() => load(schema)).toThrow(/ghost_level/)
  })

  it('accepts a valid single-chain (linear) schema', () => {
    const schema = baseSchema()
    schema.levels = [
      { name: 'site', label: '站点', idColumn: 'site_id', labelColumn: 'site_name', searchColumns: ['site_name'] },
      { name: 'building', label: '楼宇', idColumn: 'bldg_id', labelColumn: 'bldg_name', parentLevel: 'site', searchColumns: ['bldg_name'] },
    ]
    schema.metaColumns = [] // this 2-level schema has no employee level, so drop the employee-scoped meta columns
    expect(() => load(schema)).not.toThrow()
  })

  it('throws on a forked schema (two levels sharing the same parentLevel)', () => {
    const schema = baseSchema()
    schema.levels = [
      { name: 'site', label: '站点', idColumn: 'site_id', labelColumn: 'site_name', searchColumns: ['site_name'] },
      { name: 'building', label: '楼宇', idColumn: 'bldg_id', labelColumn: 'bldg_name', parentLevel: 'site', searchColumns: ['bldg_name'] },
      { name: 'department', label: '部门', idColumn: 'dept_id', labelColumn: 'dept_name', parentLevel: 'site', searchColumns: ['dept_name'] },
    ]
    expect(() => load(schema)).toThrow(/single chain/)
  })

  it('parses RFC 4180 quoted fields containing the comma separator', () => {
    const csv = [
      'site_id,site_name,bldg_id,bldg_name,dept_id,dept_name,dept_alias,emp_id,emp_name,emp_alias,emp_email,emp_phone,dept_head',
      'S01,总部,B01,主楼,D03,IT网络部,网络组,E007,张伟,小张,zhangwei@corp.com,13800138007,"李明, 主管"',
    ].join('\n') + '\n'
    const dict = load(baseSchema(), csv)
    const out = commitEntities({ selected: 'E007', level: 'assignee', dict })
    expect(out.status).toBe('complete')
    if (out.status !== 'complete') return
    expect(out.resolved.meta.dept_head).toBe('李明, 主管')
  })

  it('throws on duplicate entity id with conflicting ancestors', () => {
    const csv = [
      'site_id,site_name,bldg_id,bldg_name,dept_id,dept_name,dept_alias,emp_id,emp_name,emp_alias,emp_email,emp_phone,dept_head',
      'S01,总部,B01,主楼,D03,IT网络部,网络组,E007,张伟,小张,zhangwei@corp.com,13800138007,李明',
      'S01,总部,B01,主楼,D07,安全部,安保组,E007,张伟,小张,zhangwei@corp.com,13800138007,李明',
    ].join('\n') + '\n'
    expect(() => load(baseSchema(), csv)).toThrow(/E007/)
  })

  it('throws on a duplicate leaf id with conflicting label (#167 item 2)', () => {
    // Same emp_id E007, same ancestor branch, but a different emp_name. Silently
    // keeping the first row would hide the conflict and make the loaded label
    // depend on row order.
    const csv = [
      'site_id,site_name,bldg_id,bldg_name,dept_id,dept_name,dept_alias,emp_id,emp_name,emp_alias,emp_email,emp_phone,dept_head',
      'S01,总部,B01,主楼,D03,IT网络部,网络组,E007,张伟,小张,zhangwei@corp.com,13800138007,李明',
      'S01,总部,B01,主楼,D03,IT网络部,网络组,E007,李四,小李,zhangwei@corp.com,13800138007,李明',
    ].join('\n') + '\n'
    expect(() => load(baseSchema(), csv)).toThrow(/E007/)
  })

  it('throws on a duplicate leaf id with conflicting alias/metadata (#167 item 2)', () => {
    // Same emp_id E007, same ancestors, same name, but different alias + email.
    // The previous loader only checked ancestors, so this conflict was hidden.
    const csv = [
      'site_id,site_name,bldg_id,bldg_name,dept_id,dept_name,dept_alias,emp_id,emp_name,emp_alias,emp_email,emp_phone,dept_head',
      'S01,总部,B01,主楼,D03,IT网络部,网络组,E007,张伟,小张,zhangwei@corp.com,13800138007,李明',
      'S01,总部,B01,主楼,D03,IT网络部,网络组,E007,张伟,大张,other@corp.com,13800138007,李明',
    ].join('\n') + '\n'
    expect(() => load(baseSchema(), csv)).toThrow(/E007/)
  })

  it('throws on a duplicate parent id with conflicting label (#167 item 2)', () => {
    // Same dept_id D03 under the same building but a different dept_name.
    const csv = [
      'site_id,site_name,bldg_id,bldg_name,dept_id,dept_name,dept_alias,emp_id,emp_name,emp_alias,emp_email,emp_phone,dept_head',
      'S01,总部,B01,主楼,D03,IT网络部,网络组,E007,张伟,小张,zhangwei@corp.com,13800138007,李明',
      'S01,总部,B01,主楼,D03,IT安全部,网络组,E008,王芳,小王,wangfang@corp.com,13800138008,李明',
    ].join('\n') + '\n'
    expect(() => load(baseSchema(), csv)).toThrow(/D03/)
  })

  it('accepts an exact-duplicate row — repeated parent ids are inherent to the wide table (#167 item 2)', () => {
    // The wide CSV repeats parent ids across child rows; a fully-identical repeat
    // carries no conflicting data and must load without throwing.
    const csv = [
      'site_id,site_name,bldg_id,bldg_name,dept_id,dept_name,dept_alias,emp_id,emp_name,emp_alias,emp_email,emp_phone,dept_head',
      'S01,总部,B01,主楼,D03,IT网络部,网络组,E007,张伟,小张,zhangwei@corp.com,13800138007,李明',
      'S01,总部,B01,主楼,D03,IT网络部,网络组,E007,张伟,小张,zhangwei@corp.com,13800138007,李明',
    ].join('\n') + '\n'
    expect(() => load(baseSchema(), csv)).not.toThrow()
  })

  it('throws when a child row has a missing parent reference (#167 item 3)', () => {
    // emp_id is set but its parent department id (dept_id) is empty — the
    // assignee would otherwise load under a non-existent department.
    const csv = [
      'site_id,site_name,bldg_id,bldg_name,dept_id,dept_name,dept_alias,emp_id,emp_name,emp_alias,emp_email,emp_phone,dept_head',
      'S01,总部,B01,主楼,,,,E007,张伟,小张,zhangwei@corp.com,13800138007,李明',
    ].join('\n') + '\n'
    expect(() => load(baseSchema(), csv)).toThrow(/E007/)
  })

  it('throws when an intermediate parent reference is missing (#167 item 3)', () => {
    // bldg_id empty but the deeper department + assignee ids are present.
    const csv = [
      'site_id,site_name,bldg_id,bldg_name,dept_id,dept_name,dept_alias,emp_id,emp_name,emp_alias,emp_email,emp_phone,dept_head',
      'S01,总部,,,D03,IT网络部,网络组,E007,张伟,小张,zhangwei@corp.com,13800138007,李明',
    ].join('\n') + '\n'
    expect(() => load(baseSchema(), csv)).toThrow(/D03|E007/)
  })
})

// ─── lookup ──────────────────────────────────────────────────────────────────

describe('lookupEntities — lookup', () => {
  let dict: HierarchicalDict

  beforeAll(() => { dict = load() })

  it('returns empty candidates and null suggested when query matches nothing', () => {
    const out = lookupEntities({ utterance: '不存在的人', level: 'assignee', dict })
    expect(out.candidates).toHaveLength(0)
    expect(out.options).toHaveLength(0)
    expect(out.suggested).toBeNull()
  })

  it('returns a single candidate with non-null suggested when query uniquely matches', () => {
    const out = lookupEntities({ utterance: '王芳', level: 'assignee', dict })
    expect(out.candidates).toHaveLength(1)
    expect(out.candidates[0]!.id).toBe('E008')
    expect(out.suggested).toBe('E008')
    expect(out.options).toContain('E008')
  })

  it('matches a full natural-language utterance that contains the assignee name', () => {
    const out = lookupEntities({ utterance: '请把这个工单派给王芳处理', level: 'assignee', dict })
    expect(out.candidates.length).toBeGreaterThan(0)
    expect(out.candidates.map(c => c.id)).toContain('E008')
    expect(out.suggested).toBe('E008')
  })

  it('returns multiple candidates with null suggested when query matches ambiguously', () => {
    const out = lookupEntities({ utterance: '张', level: 'assignee', dict })
    expect(out.candidates.length).toBeGreaterThan(1)
    expect(out.suggested).toBeNull()
  })

  it('filters candidates by pinned ancestor', () => {
    const out = lookupEntities({
      utterance: '张',
      level: 'assignee',
      pinned: { building: 'B01', department: 'D03' },
      dict,
    })
    const ids = out.candidates.map(c => c.id)
    expect(ids).toContain('E007')   // 张伟, under D03
    expect(ids).toContain('E009')   // 张亮, under D03
    expect(ids).not.toContain('E012') // 张伟, under D07 (B02) — filtered out
  })

  it('includes full ancestor path in each candidate', () => {
    const out = lookupEntities({ utterance: '王芳', level: 'assignee', dict })
    expect(out.candidates[0]!.path).toEqual(['总部', '主楼', 'IT网络部', '王芳'])
  })

  it('scores are between 0 and 1 (exclusive lower bound)', () => {
    const out = lookupEntities({ utterance: '张', level: 'assignee', dict })
    for (const c of out.candidates) {
      expect(c.score).toBeGreaterThan(0)
      expect(c.score).toBeLessThanOrEqual(1)
    }
  })

  it('options matches candidate ids in score order', () => {
    const out = lookupEntities({ utterance: '张', level: 'assignee', dict })
    expect(out.options).toEqual(out.candidates.map(c => c.id))
  })

  it('derives the target level from pinned when level is omitted (#167 item 1)', () => {
    // Portable core contract: callers pass only { utterance, pinned, dict }; the
    // target level is derived from how many ancestors are pinned.
    const out = lookupEntities({
      utterance: '王芳',
      pinned: { site: 'S01', building: 'B01', department: 'D03' },
      dict,
    })
    expect(out.candidates.map(c => c.id)).toContain('E008')
  })

  it('does not filter out the target-level entity when that level is pinned (#167 item 2)', () => {
    // All levels pinned: the target level itself ("assignee") is present in
    // pinned, but it is not an ancestor constraint — the already-selected entity
    // must still be returned, not filtered out.
    const out = lookupEntities({
      utterance: '张伟',
      level: 'assignee',
      pinned: { site: 'S01', building: 'B01', department: 'D03', assignee: 'E007' },
      dict,
    })
    expect(out.candidates.map(c => c.id)).toContain('E007')
  })

  it('with all levels pinned and level omitted, still returns the pinned entity (#167 items 1+2)', () => {
    const out = lookupEntities({
      utterance: '张伟',
      pinned: { site: 'S01', building: 'B01', department: 'D03', assignee: 'E007' },
      dict,
    })
    expect(out.candidates.map(c => c.id)).toContain('E007')
  })

  it('finds a deep entity from a clear utterance with empty pinned and no level (#167 item 1, CLI contract)', () => {
    // The real portable/CLI contract: no target level, no pinned ancestors. A
    // clear assignee name must still be recalled and suggested — not confined to
    // the root level by the next-unfilled derivation.
    const out = lookupEntities({ utterance: '王芳', dict })
    expect(out.candidates.map(c => c.id)).toContain('E008')
    expect(out.suggested).toBe('E008')
  })

  it('does not match a parent level against a descendant-owned search column (#167)', () => {
    // "张伟" is an emp_name (assignee-owned). A department-level lookup must NOT
    // recall a department just because one of its child rows carried that employee
    // name — searchValues are scoped to the entity's own level, so a parent record
    // never absorbs descendant alias columns (and the result is row-order stable).
    const out = lookupEntities({ utterance: '张伟', level: 'department', dict })
    expect(out.candidates.map(c => c.id)).not.toContain('D03')
    expect(out.candidates).toHaveLength(0)
  })

  it('does not match a parent level against a descendant-owned alias column (#167)', () => {
    // "小张" is an emp_alias (assignee-owned). Same scoping requirement as above.
    const out = lookupEntities({ utterance: '小张', level: 'department', dict })
    expect(out.candidates).toHaveLength(0)
  })

  it('does not match the site level against a descendant-owned search column (#167)', () => {
    const out = lookupEntities({ utterance: '张伟', level: 'site', dict })
    expect(out.candidates).toHaveLength(0)
  })

  it('still recalls a parent by its own label and alias columns (#167)', () => {
    // The scoping must not regress legitimate parent recall: a department is still
    // found by its own dept_name and dept_alias.
    const byName = lookupEntities({ utterance: 'IT网络部', level: 'department', dict })
    expect(byName.candidates.map(c => c.id)).toContain('D03')
    const byAlias = lookupEntities({ utterance: '网络组', level: 'department', dict })
    expect(byAlias.candidates.map(c => c.id)).toContain('D03')
  })
})

// ─── commit ──────────────────────────────────────────────────────────────────

describe('commitEntities — commit', () => {
  let dict: HierarchicalDict

  beforeAll(() => { dict = load() })

  it('complete: resolves with status complete when selected is valid and ancestors match', () => {
    const out = commitEntities({
      selected: 'E007',
      level: 'assignee',
      pinned: { site: 'S01', building: 'B01', department: 'D03' },
      dict,
    })
    expect(out.status).toBe('complete')
    if (out.status !== 'complete') return
    expect(out.resolved.id).toBe('E007')
    expect(out.resolved.label).toBe('张伟')
    expect(out.resolved.path).toEqual(['总部', '主楼', 'IT网络部', '张伟'])
  })

  it('complete: resolved.meta includes metaColumns values', () => {
    const out = commitEntities({ selected: 'E007', level: 'assignee', dict })
    expect(out.status).toBe('complete')
    if (out.status !== 'complete') return
    expect(out.resolved.meta).toMatchObject({ emp_email: 'zhangwei@corp.com', dept_head: '李明' })
  })

  it('missing: under-selection — a department id committed at target level assignee returns missing', () => {
    const out = commitEntities({ selected: 'D03', level: 'assignee', dict })
    expect(out.status).toBe('missing')
  })

  it('missing: a site-level id committed at target level assignee returns missing', () => {
    const out = commitEntities({ selected: 'S01', level: 'assignee', dict })
    expect(out.status).toBe('missing')
  })

  it('unknown: returns status unknown when selected id is absent from every level', () => {
    const out = commitEntities({ selected: 'E999', level: 'assignee', dict })
    expect(out.status).toBe('unknown')
  })

  it('corrected: returns status corrected with correctedLevels when selected has wrong ancestor branch', () => {
    // E012 (张伟) is under B02/D07, but pinned says B01/D03 → correct to E007 (also 张伟, under B01/D03)
    const out = commitEntities({
      selected: 'E012',
      level: 'assignee',
      pinned: { building: 'B01', department: 'D03' },
      dict,
    })
    expect(out.status).toBe('corrected')
    if (out.status !== 'corrected') return
    expect(out.resolved.id).toBe('E007')
    expect(out.correctedLevels).toMatchObject({ building: 'B01', department: 'D03' })
  })

  it('corrected: selected overrides a different id already confirmed at the same level (#167 item 3)', () => {
    // pinned already confirmed assignee=E007; LLM now selects a different valid
    // assignee E012. Ancestors are not constrained (no ancestor pins), so the new
    // selection is honored and reported as a same-level correction — NOT
    // invalid_selection.
    const out = commitEntities({
      selected: 'E012',
      level: 'assignee',
      pinned: { assignee: 'E007' },
      dict,
    })
    expect(out.status).toBe('corrected')
    if (out.status !== 'corrected') return
    expect(out.resolved.id).toBe('E012')
    expect(out.resolved.path).toEqual(['总部', '东楼', '安全部', '张伟'])
    expect(out.correctedLevels).toMatchObject({ assignee: 'E012' })
  })

  it('derives the target level from pinned when level is omitted on commit (#167 item 1)', () => {
    const out = commitEntities({
      selected: 'E007',
      pinned: { site: 'S01', building: 'B01', department: 'D03' },
      dict,
    })
    expect(out.status).toBe('complete')
    if (out.status !== 'complete') return
    expect(out.resolved.id).toBe('E007')
  })

  it('complete: a same-level pin equal to selected is a no-op confirmation', () => {
    const out = commitEntities({
      selected: 'E007',
      level: 'assignee',
      pinned: { assignee: 'E007' },
      dict,
    })
    expect(out.status).toBe('complete')
  })

  it('invalid_selection: returns status invalid_selection when selected has wrong ancestor and no same-label candidate exists under pinned', () => {
    const out = commitEntities({
      selected: 'E020',
      level: 'assignee',
      pinned: { site: 'S01' },
      dict,
    })
    expect(out.status).toBe('invalid_selection')
  })

  it('ambiguous: returns status ambiguous when auto-correction is ambiguous (multiple same-label candidates under pinned)', () => {
    const csv = [
      'site_id,site_name,bldg_id,bldg_name,dept_id,dept_name,dept_alias,emp_id,emp_name,emp_alias,emp_email,emp_phone,dept_head',
      'S01,总部,B01,主楼,D03,IT网络部,网络组,E007,张伟,小张,zhangwei@corp.com,13800138007,李明',
      'S01,总部,B01,主楼,D03,IT网络部,网络组,E013,张伟,小张二,zhangwei3@corp.com,13800138013,李明',
      'S01,总部,B02,东楼,D07,安全部,安保组,E012,张伟,小张,zhangwei2@corp.com,13800138012,王磊',
    ].join('\n') + '\n'
    const localDict = load(baseSchema(), csv)
    const out = commitEntities({
      selected: 'E012',
      level: 'assignee',
      pinned: { building: 'B01', department: 'D03' },
      dict: localDict,
    })
    expect(out.status).toBe('ambiguous')
  })
})

// ─── level derivation (CLI helper) ─────────────────────────────────────────────

describe('nextLevel — derive target level from pinned (CLI contract)', () => {
  let dict: HierarchicalDict

  beforeAll(() => { dict = load() })

  it('returns the root level when nothing is pinned', () => {
    expect(nextLevel(dict, {})).toBe('site')
  })

  it('returns the first unpinned level given a prefix of ancestors', () => {
    expect(nextLevel(dict, { site: 'S01' })).toBe('building')
    expect(nextLevel(dict, { site: 'S01', building: 'B01', department: 'D03' })).toBe('assignee')
  })

  it('returns the deepest level when every level is already pinned', () => {
    expect(nextLevel(dict, { site: 'S01', building: 'B01', department: 'D03', assignee: 'E007' })).toBe('assignee')
  })
})

// ─── clear-case ──────────────────────────────────────────────────────────────

describe('clear-case (lookup.suggested → commit.selected)', () => {
  it('commit with lookup.suggested succeeds: complete, not corrected', () => {
    const dict = load()
    const lookupOut = lookupEntities({ utterance: '王芳', level: 'assignee', dict })
    expect(lookupOut.suggested).not.toBeNull()

    const commitOut = commitEntities({ selected: lookupOut.suggested!, level: 'assignee', dict })
    expect(commitOut.status).toBe('complete')
    if (commitOut.status !== 'complete') return
    expect(commitOut.resolved.id).toBe(lookupOut.suggested)
  })

  it('the clear lookup.suggested is directly commit-able via the CLI contract — no level, empty pinned (#167 item 1)', () => {
    const dict = load()
    // Exactly what the CLI passes through: { utterance, pinned } with no level.
    const lookupOut = lookupEntities({ utterance: '王芳', dict })
    expect(lookupOut.suggested).toBe('E008')

    const commitOut = commitEntities({ selected: lookupOut.suggested!, dict })
    expect(commitOut.status).toBe('complete')
    if (commitOut.status !== 'complete') return
    expect(commitOut.resolved.id).toBe('E008')
    expect(commitOut.resolved.path).toEqual(['总部', '主楼', 'IT网络部', '王芳'])
  })
})

// ─── EntityResolver wrapper ────────────────────────────────────────────────────

describe('EntityResolver — in-process wrapper over a loaded dict', () => {
  it('wraps a dict and exposes lookup/commit with no fs access', () => {
    const er = EntityResolver.load(baseSchema(), baseCsv())
    expect(er.lookup({ utterance: '王芳', level: 'assignee' }).suggested).toBe('E008')
    expect(er.commit({ selected: 'E007', level: 'assignee' }).status).toBe('complete')
  })
})

// ─── portability: no fs, no milkie runtime imports ─────────────────────────────

describe('EntityResolver.ts — portable core', () => {
  let src: string
  beforeAll(() => { src = fs.readFileSync(path.join(__dirname, '../EntityResolver.ts'), 'utf-8') })

  it('does not import the node fs module in the core (#167 item 1)', () => {
    expect(src).not.toMatch(/from ['"]fs['"]/)
    expect(src).not.toMatch(/require\(['"]fs['"]\)/)
  })

  it('does not import from the milkie runtime', () => {
    expect(src).not.toMatch(/from ['"].*milkie/)
    expect(src).not.toMatch(/require\(['"].*milkie/)
    expect(src).not.toMatch(/@milkie\//)
  })
})
