import { hashModelRequest, hashToolCall, canonicalize } from '../trace/hash'
import type { ModelRequest } from '../types/model'

describe('canonicalize', () => {
  it('sorts object keys recursively', () => {
    const a = canonicalize({ b: 1, a: { d: 2, c: 3 } })
    const b = canonicalize({ a: { c: 3, d: 2 }, b: 1 })
    expect(a).toBe(b)
  })

  it('preserves array order', () => {
    expect(canonicalize([1, 2, 3])).toBe(JSON.stringify([1, 2, 3]))
  })

  it('treats undefined as missing key', () => {
    expect(canonicalize({ a: 1, b: undefined })).toBe(canonicalize({ a: 1 }))
  })

  it('distinguishes null from missing', () => {
    expect(canonicalize({ a: null })).not.toBe(canonicalize({}))
  })

  it('throws on unsupported types (Date, Map, etc.)', () => {
    expect(() => canonicalize(new Date(0))).toThrow(/unsupported type/i)
    expect(() => canonicalize(new Map())).toThrow(/unsupported type/i)
  })
})

const reqA = (): ModelRequest => ({
  model:    'm1',
  messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
  system:   'sys',
  tools:    [],
})

describe('hashModelRequest', () => {
  it('returns stable 64-hex SHA-256', () => {
    const h = hashModelRequest(reqA())
    expect(h).toMatch(/^[0-9a-f]{64}$/)
  })

  it('is stable under key reordering', () => {
    const h1 = hashModelRequest(reqA())
    const reordered: ModelRequest = {
      tools:    [],
      system:   'sys',
      messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
      model:    'm1',
    }
    expect(hashModelRequest(reordered)).toBe(h1)
  })

  it('changes when any field changes', () => {
    const h1 = hashModelRequest(reqA())
    const mutated: ModelRequest = { ...reqA(), model: 'm2' }
    expect(hashModelRequest(mutated)).not.toBe(h1)
  })
})

describe('hashToolCall', () => {
  it('returns stable 64-hex', () => {
    expect(hashToolCall('grep', { pattern: 'x' })).toMatch(/^[0-9a-f]{64}$/)
  })

  it('changes when name or input differs', () => {
    const a = hashToolCall('grep', { pattern: 'x' })
    const b = hashToolCall('grep', { pattern: 'y' })
    const c = hashToolCall('rg',   { pattern: 'x' })
    expect(a).not.toBe(b)
    expect(a).not.toBe(c)
  })

  it('is stable under input key reordering', () => {
    const a = hashToolCall('t', { x: 1, y: 2 })
    const b = hashToolCall('t', { y: 2, x: 1 })
    expect(a).toBe(b)
  })

  it('produces a stable known digest (regression guard)', () => {
    // Pinned: any change to the canonicalize algorithm or SHA-256 plumbing
    // will break this assertion. Update only with intentional algorithm change.
    expect(hashToolCall('grep', { pattern: 'x' })).toBe('5850a24803a595aeaafe553129f5830550e196fa19cc3673e339865132e94434')
  })
})
