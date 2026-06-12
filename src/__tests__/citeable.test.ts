import { citeable } from '../tools/lineage'
import { sha256Hex } from '../trace/hash'
import type { ToolContext } from '../types/tool'

// A minimal ToolContext that records registerObject calls (lazy lineage sink).
function mockCtx() {
  const calls: Array<{ type: string; hash?: string; meta?: Record<string, unknown> }> = []
  const ctx = {
    registerObject: (spec: { type: string; hash?: string; meta?: Record<string, unknown> }) => {
      calls.push(spec)
      return { objectId: `obj:reg:${calls.length}` }
    },
  } as unknown as ToolContext
  return { ctx, calls }
}

describe('citeable helper (#155)', () => {
  it('mints via registerObject and puts objectId first in the result', () => {
    const { ctx, calls } = mockCtx()
    const out = citeable(ctx, 'hello world', { text: 'hello world', source: 'doc1' })

    expect(calls).toHaveLength(1)
    expect(calls[0]!.type).toBe('passage')                 // default type
    expect(calls[0]!.hash).toBe(sha256Hex('hello world'))  // content-bound hash
    expect((out as Record<string, unknown>).objectId).toBe('obj:reg:1')
    expect(Object.keys(out)[0]).toBe('objectId')          // objectId first → survives truncation
    expect((out as Record<string, unknown>).source).toBe('doc1')  // original fields kept
  })

  it('honors a custom type and meta', () => {
    const { ctx, calls } = mockCtx()
    citeable(ctx, 'out', { stdout: 'out' }, { type: 'shell:stdout', meta: { command: 'ls', exitCode: 0 } })

    expect(calls[0]!.type).toBe('shell:stdout')
    expect(calls[0]!.meta).toEqual({ command: 'ls', exitCode: 0 })
  })

  it('without a lineage sink returns the result unchanged (no objectId key)', () => {
    const ctx = {} as unknown as ToolContext   // registerObject absent → lineage off
    const out = citeable(ctx, 'x', { text: 'x' })

    expect('objectId' in out).toBe(false)
    expect(out).toEqual({ text: 'x' })
  })
})
