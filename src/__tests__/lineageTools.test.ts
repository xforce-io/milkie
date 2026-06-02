import { lineageTools } from '../tools/lineage'
import type { ToolContext } from '../types/tool'

// A mock ToolContext that records what the handler declared, with a known set of
// resolvable objectIds (simulating ids minted earlier this run by read/grep).
function mockCtx(known: Set<string>) {
  const created: Array<{ objectId: string; type: string }> = []
  const relations: Array<{ type: string; fromObjectId: string; toObjectId: string }> = []
  const promoted: string[] = []
  const ctx = {
    resolveObject: (id: string) => (known.has(id) ? { type: 'passage' as const } : undefined),
    promoteObject: (id: string) => { promoted.push(id) },
    createObject:  (spec: { type: string }) => { const objectId = `obj:made:${created.length}`; created.push({ objectId, type: spec.type }); return { objectId } },
    createRelation: (spec: { type: string; fromObjectId: string; toObjectId: string }) => { relations.push(spec); return { relationId: `rel:${relations.length}` } },
  } as unknown as ToolContext
  return { ctx, created, relations, promoted }
}

const cite            = lineageTools.find(t => t.name === 'cite')!
const declareRelation = lineageTools.find(t => t.name === 'declare_relation')!

describe('built-in lineage tools (#113 P3)', () => {
  it('registers cite and declare_relation', () => {
    expect(cite).toBeTruthy()
    expect(declareRelation).toBeTruthy()
  })

  it('cite: resolvable id → mints a claim + cites relation, ok:true, promotes the cited id', async () => {
    const { ctx, created, relations, promoted } = mockCtx(new Set(['obj:p']))
    const out = await cite.handler({ claim: '某陈述', objectId: 'obj:p' }, ctx) as { ok: boolean }
    expect(out.ok).toBe(true)
    expect(created).toHaveLength(1)
    expect(created[0]!.type).toBe('claim')
    expect(relations).toEqual([{ type: 'cites', fromObjectId: created[0]!.objectId, toObjectId: 'obj:p' }])
    expect(promoted).toContain('obj:p')
  })

  it('cite: unresolvable id → ok:false and declares nothing', async () => {
    const { ctx, created, relations } = mockCtx(new Set())
    const out = await cite.handler({ claim: 'x', objectId: 'obj:fake' }, ctx) as { ok: boolean; error?: string }
    expect(out.ok).toBe(false)
    expect(out.error).toMatch(/不存在|objectId/)
    expect(created).toHaveLength(0)
    expect(relations).toHaveLength(0)
  })

  it('declare_relation: both ids resolvable → typed edge, ok:true', async () => {
    const { ctx, relations } = mockCtx(new Set(['obj:a', 'obj:b']))
    const out = await declareRelation.handler({ type: 'equivalent_to', fromObjectId: 'obj:a', toObjectId: 'obj:b' }, ctx) as { ok: boolean }
    expect(out.ok).toBe(true)
    expect(relations).toEqual([{ type: 'equivalent_to', fromObjectId: 'obj:a', toObjectId: 'obj:b' }])
  })

  it('declare_relation: an unknown endpoint → ok:false and no edge', async () => {
    const { ctx, relations } = mockCtx(new Set(['obj:a']))
    const out = await declareRelation.handler({ type: 'derives_from', fromObjectId: 'obj:a', toObjectId: 'obj:x' }, ctx) as { ok: boolean }
    expect(out.ok).toBe(false)
    expect(relations).toHaveLength(0)
  })

  it('lineage tools degrade gracefully when the runtime did not wire lineage (no ctx methods)', async () => {
    const bare = {} as ToolContext  // no resolveObject/createObject/etc.
    // No resolveObject → no fail-fast; createObject/createRelation absent → no throw.
    const out = await cite.handler({ claim: 'x', objectId: 'obj:p' }, bare) as { ok: boolean }
    expect(out.ok).toBe(true)
  })
})
