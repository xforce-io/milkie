import fs from 'fs'
import path from 'path'

import { loadHierarchicalDict, type Schema, type HierarchicalDict } from '../EntityResolver'
import { recall, type RecallResult } from '../recall'

const schema = JSON.parse(fs.readFileSync(path.join(__dirname, '../schema.json'), 'utf-8')) as Schema
const csv = fs.readFileSync(path.join(__dirname, '../data.csv'), 'utf-8')
const dict: HierarchicalDict = loadHierarchicalDict(schema, csv)

const at = (r: RecallResult, level: string) => r.byLevel.find(l => l.level === level)
const ids = (r: RecallResult, level: string) => (at(r, level)?.candidates ?? []).map(c => c.id)

describe('fusion recall (#180) — step 1, deterministic, LLM-free', () => {
  it('exact name → decisive sole site', () => {
    const r = recall(dict, '总部')
    expect(at(r, 'site')!.decisive).toBe('S01')
    expect(at(r, 'site')!.candidates[0]!.via).toContain('exact')
  })

  it('alias → recalls the aliased department (硬件部 = IT硬件组 D02)', () => {
    const r = recall(dict, '硬件部', { site: 'S01', building: 'B01' })
    expect(ids(r, 'department')).toContain('D02')
    expect(at(r, 'department')!.candidates.find(c => c.id === 'D02')!.matchedSurface).toBe('硬件部')
  })

  it('typo → edit-distance recalls 总部 from 总布', () => {
    const r = recall(dict, '总布')
    const site = at(r, 'site')!
    expect(site.candidates.map(c => c.id)).toContain('S01')
    expect(site.candidates.find(c => c.id === 'S01')!.via).toContain('edit')
  })

  it('ambiguous → multiple candidates, NOT decisive (张 → 张伟 + 张亮)', () => {
    const r = recall(dict, '张', { site: 'S01', building: 'B01', department: 'D03' })
    const a = at(r, 'assignee')!
    expect(a.candidates.map(c => c.id)).toEqual(expect.arrayContaining(['E007', 'E009']))
    expect(a.candidates.length).toBeGreaterThanOrEqual(2)
    expect(a.decisive).toBeNull()   // → must go to step-2 LLM, not auto-commit
  })

  it('unknown → empty candidates (→ clarify/unknown, never hard-commit)', () => {
    const r = recall(dict, '火星基地', { site: 'S01', building: 'B01', department: 'D03' })
    expect(at(r, 'assignee')!.candidates).toHaveLength(0)
    expect(at(r, 'assignee')!.decisive).toBeNull()
  })

  it('pinned filter → branch-confined; clear name is decisive', () => {
    const r = recall(dict, '王芳', { site: 'S01', building: 'B01', department: 'D03' })
    expect(at(r, 'assignee')!.decisive).toBe('E008')
  })

  it('level-less → an embedded name lands on its own level only', () => {
    // "网络部" appears in dept_name "IT网络部"; no pinned context.
    const r = recall(dict, '网络部')
    expect(ids(r, 'department')).toContain('D03')
    expect(at(r, 'site')!.candidates).toHaveLength(0)       // does not bleed into other levels
    expect(at(r, 'building')!.candidates).toHaveLength(0)
  })
})
