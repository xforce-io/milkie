import type { ObjectType, RelationType, CoreObjectType, CoreRelationType } from '../trace/types'

// #113 P4: the lineage vocabulary is extensible. The TYPE ANNOTATIONS below are
// the real assertions — `const x: ObjectType = 'code:function'` only compiles
// under the widened union; a closed union would make `tsc --noEmit` fail. The
// runtime expects are sanity checks.
describe('lineage vocabulary is extensible (#113 P4)', () => {
  it('core object/relation types remain a controlled vocabulary', () => {
    const coreObj: CoreObjectType = 'passage'
    const coreRel: CoreRelationType = 'cites'
    expect(coreObj).toBe('passage')
    expect(coreRel).toBe('cites')
  })

  it('namespaced app types are assignable alongside core ones', () => {
    const appObj: ObjectType = 'code:function'        // closed union → TS error
    const appRel: RelationType = 'app:tested_by'      // closed union → TS error
    const objs: ObjectType[]   = ['passage', 'claim', 'code:function', 'db:row']
    const rels: RelationType[] = ['cites', 'derives_from', 'app:tested_by']
    expect(appObj.startsWith('code:')).toBe(true)
    expect(appRel.includes(':')).toBe(true)
    expect(objs).toContain('db:row')
    expect(rels).toContain('app:tested_by')
  })
})
