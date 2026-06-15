import fs from 'fs'
import os from 'os'
import path from 'path'
import { EntityResolver } from '../EntityResolver'

const SCHEMA = path.join(__dirname, '../schema.json')
const DATA = path.join(__dirname, '../data.csv')

// ─── helpers ─────────────────────────────────────────────────────────────────

function tmpSchema(overrides: object): string {
  const base = JSON.parse(fs.readFileSync(SCHEMA, 'utf-8'))
  const merged = { ...base, ...overrides }
  const p = path.join(os.tmpdir(), `schema-${Date.now()}-${Math.random().toString(36).slice(2)}.json`)
  fs.writeFileSync(p, JSON.stringify(merged))
  return p
}

function tmpData(csv: string): string {
  const p = path.join(os.tmpdir(), `data-${Date.now()}-${Math.random().toString(36).slice(2)}.csv`)
  fs.writeFileSync(p, csv)
  return p
}

// ─── loading ─────────────────────────────────────────────────────────────────

describe('EntityResolver — loading', () => {
  it('loads valid schema + CSV without throwing', () => {
    expect(() => new EntityResolver({ schemaPath: SCHEMA, dataPath: DATA })).not.toThrow()
  })

  it('throws on missing idColumn in CSV', () => {
    const csv = 'site_name,bldg_id,bldg_name,dept_id,dept_name,dept_alias,emp_id,emp_name,emp_alias,emp_email,emp_phone,dept_head\n总部,B01,主楼,D03,IT网络部,网络组,E007,张伟,小张,zhangwei@corp.com,13800138007,李明\n'
    const data = tmpData(csv)
    expect(() => new EntityResolver({ schemaPath: SCHEMA, dataPath: data })).toThrow(/site_id/)
  })

  it('throws on missing labelColumn in CSV', () => {
    const csv = 'site_id,bldg_id,bldg_name,dept_id,dept_name,dept_alias,emp_id,emp_name,emp_alias,emp_email,emp_phone,dept_head\nS01,B01,主楼,D03,IT网络部,网络组,E007,张伟,小张,zhangwei@corp.com,13800138007,李明\n'
    const data = tmpData(csv)
    expect(() => new EntityResolver({ schemaPath: SCHEMA, dataPath: data })).toThrow(/site_name/)
  })

  it('throws when a level references a non-existent parentLevel', () => {
    const schema = tmpSchema({
      levels: [
        { name: 'site', label: '站点', idColumn: 'site_id', labelColumn: 'site_name' },
        { name: 'assignee', label: '负责人', idColumn: 'emp_id', labelColumn: 'emp_name', parentLevel: 'ghost_level' },
      ],
    })
    expect(() => new EntityResolver({ schemaPath: schema, dataPath: DATA })).toThrow(/ghost_level/)
  })

  it('throws on duplicate entity id with conflicting ancestors', () => {
    // E007 appears twice: once under D03, once under D07 — same id, different dept
    const csv = [
      'site_id,site_name,bldg_id,bldg_name,dept_id,dept_name,dept_alias,emp_id,emp_name,emp_alias,emp_email,emp_phone,dept_head',
      'S01,总部,B01,主楼,D03,IT网络部,网络组,E007,张伟,小张,zhangwei@corp.com,13800138007,李明',
      'S01,总部,B01,主楼,D07,安全部,安保组,E007,张伟,小张,zhangwei@corp.com,13800138007,李明',
    ].join('\n') + '\n'
    const data = tmpData(csv)
    expect(() => new EntityResolver({ schemaPath: SCHEMA, dataPath: data })).toThrow(/E007/)
  })
})

// ─── lookup ──────────────────────────────────────────────────────────────────

describe('EntityResolver — lookup', () => {
  let er: EntityResolver

  beforeAll(() => {
    er = new EntityResolver({ schemaPath: SCHEMA, dataPath: DATA })
  })

  it('returns empty candidates and null suggested when query matches nothing', () => {
    const out = er.lookup({ op: 'lookup', query: '不存在的人', context: { level: 'assignee' } })
    expect(out.candidates).toHaveLength(0)
    expect(out.options).toHaveLength(0)
    expect(out.suggested).toBeNull()
  })

  it('returns a single candidate with non-null suggested when query uniquely matches', () => {
    const out = er.lookup({ op: 'lookup', query: '王芳', context: { level: 'assignee' } })
    expect(out.candidates).toHaveLength(1)
    expect(out.candidates[0]!.id).toBe('E008')
    expect(out.suggested).toBe('E008')
    expect(out.options).toContain('E008')
  })

  it('returns multiple candidates with null suggested when query matches ambiguously', () => {
    // "张" matches 张伟 (E007), 张亮 (E009), 张伟 (E012)
    const out = er.lookup({ op: 'lookup', query: '张', context: { level: 'assignee' } })
    expect(out.candidates.length).toBeGreaterThan(1)
    expect(out.suggested).toBeNull()
  })

  it('filters candidates by pinned ancestor', () => {
    const out = er.lookup({
      op: 'lookup',
      query: '张',
      context: { level: 'assignee', pinned: { building: 'B01', department: 'D03' } },
    })
    const ids = out.candidates.map(c => c.id)
    expect(ids).toContain('E007')   // 张伟, under D03
    expect(ids).toContain('E009')   // 张亮, under D03
    expect(ids).not.toContain('E012') // 张伟, under D07 (B02) — filtered out
  })

  it('includes full ancestor path in each candidate', () => {
    const out = er.lookup({ op: 'lookup', query: '王芳', context: { level: 'assignee' } })
    expect(out.candidates[0]!.path).toEqual(['总部', '主楼', 'IT网络部', '王芳'])
  })

  it('scores are between 0 and 1 (exclusive lower bound)', () => {
    const out = er.lookup({ op: 'lookup', query: '张', context: { level: 'assignee' } })
    for (const c of out.candidates) {
      expect(c.score).toBeGreaterThan(0)
      expect(c.score).toBeLessThanOrEqual(1)
    }
  })

  it('options matches candidate ids in score order', () => {
    const out = er.lookup({ op: 'lookup', query: '张', context: { level: 'assignee' } })
    expect(out.options).toEqual(out.candidates.map(c => c.id))
  })
})

// ─── commit ──────────────────────────────────────────────────────────────────

describe('EntityResolver — commit', () => {
  let er: EntityResolver

  beforeAll(() => {
    er = new EntityResolver({ schemaPath: SCHEMA, dataPath: DATA })
  })

  it('complete: resolves with status complete when selected is valid and ancestors match', () => {
    const out = er.commit({
      op: 'commit',
      selected: 'E007',
      context: { level: 'assignee', pinned: { site: 'S01', building: 'B01', department: 'D03' } },
    })
    expect(out.status).toBe('complete')
    if (out.status !== 'complete') return
    expect(out.resolved.id).toBe('E007')
    expect(out.resolved.label).toBe('张伟')
    expect(out.resolved.path).toEqual(['总部', '主楼', 'IT网络部', '张伟'])
  })

  it('complete: resolved.meta includes metaColumns values', () => {
    const out = er.commit({
      op: 'commit',
      selected: 'E007',
      context: { level: 'assignee' },
    })
    expect(out.status).toBe('complete')
    if (out.status !== 'complete') return
    expect(out.resolved.meta).toMatchObject({ emp_email: 'zhangwei@corp.com', dept_head: '李明' })
  })

  it('missing: returns status missing when selected id is absent from all levels', () => {
    const out = er.commit({ op: 'commit', selected: 'E999', context: { level: 'assignee' } })
    expect(out.status).toBe('missing')
  })

  it('unknown: returns status unknown when a site-level id is committed at assignee level', () => {
    const out = er.commit({ op: 'commit', selected: 'S01', context: { level: 'assignee' } })
    expect(out.status).toBe('unknown')
  })

  it('corrected: returns status corrected with correctedLevels when selected has wrong ancestor branch', () => {
    // E012 (张伟) is under B02/D07, but pinned says B01/D03 → should correct to E007 (also 张伟, under B01/D03)
    const out = er.commit({
      op: 'commit',
      selected: 'E012',
      context: { level: 'assignee', pinned: { building: 'B01', department: 'D03' } },
    })
    expect(out.status).toBe('corrected')
    if (out.status !== 'corrected') return
    expect(out.resolved.id).toBe('E007')
    expect(out.correctedLevels).toMatchObject({ building: 'B01', department: 'D03' })
  })

  it('invalid_selection: returns status invalid_selection when selected has wrong ancestor and no same-label candidate exists under pinned', () => {
    // E020 (赵明) is under S02; no '赵明' exists under S01 → correction impossible
    const out = er.commit({
      op: 'commit',
      selected: 'E020',
      context: { level: 'assignee', pinned: { site: 'S01' } },
    })
    expect(out.status).toBe('invalid_selection')
  })

  it('ambiguous: returns status ambiguous when auto-correction is ambiguous (multiple same-label candidates under pinned)', () => {
    // Fixture with two "张伟" under D03: one is the correction target, so it's ambiguous
    const csv = [
      'site_id,site_name,bldg_id,bldg_name,dept_id,dept_name,dept_alias,emp_id,emp_name,emp_alias,emp_email,emp_phone,dept_head',
      'S01,总部,B01,主楼,D03,IT网络部,网络组,E007,张伟,小张,zhangwei@corp.com,13800138007,李明',
      'S01,总部,B01,主楼,D03,IT网络部,网络组,E013,张伟,小张二,zhangwei3@corp.com,13800138013,李明',
      'S01,总部,B02,东楼,D07,安全部,安保组,E012,张伟,小张,zhangwei2@corp.com,13800138012,王磊',
    ].join('\n') + '\n'
    const data = tmpData(csv)
    const resolver = new EntityResolver({ schemaPath: SCHEMA, dataPath: data })
    const out = resolver.commit({
      op: 'commit',
      selected: 'E012',
      context: { level: 'assignee', pinned: { building: 'B01', department: 'D03' } },
    })
    expect(out.status).toBe('ambiguous')
  })
})

// ─── clear-case ──────────────────────────────────────────────────────────────

describe('EntityResolver — clear-case (lookup.suggested → commit.selected)', () => {
  it('commit with lookup.suggested succeeds: no validationError, no corrected', () => {
    const er = new EntityResolver({ schemaPath: SCHEMA, dataPath: DATA })
    const lookupOut = er.lookup({ op: 'lookup', query: '王芳', context: { level: 'assignee' } })
    expect(lookupOut.suggested).not.toBeNull()

    const commitOut = er.commit({
      op: 'commit',
      selected: lookupOut.suggested!,
      context: { level: 'assignee' },
    })
    expect(commitOut.status).toBe('complete')
    if (commitOut.status !== 'complete') return
    expect(commitOut.resolved.id).toBe(lookupOut.suggested)
  })
})

// ─── no-milkie import ────────────────────────────────────────────────────────

describe('EntityResolver — no milkie runtime imports', () => {
  it('EntityResolver.ts does not import from milkie runtime', () => {
    const src = fs.readFileSync(path.join(__dirname, '../EntityResolver.ts'), 'utf-8')
    expect(src).not.toMatch(/from ['"].*milkie/)
    expect(src).not.toMatch(/require\(['"].*milkie/)
    expect(src).not.toMatch(/@milkie\//)
  })
})
