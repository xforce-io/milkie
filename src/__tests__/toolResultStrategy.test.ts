import { applyShape, serializeOutput } from '../runtime/toolResultStrategy'
import type { Shape } from '../types/tool'

describe('serializeOutput', () => {
  test('string passes through', () => {
    expect(serializeOutput('hello')).toBe('hello')
  })
  test('null → "null"', () => {
    expect(serializeOutput(null)).toBe('null')
  })
  test('undefined → ""', () => {
    expect(serializeOutput(undefined)).toBe('')
  })
  test('object → JSON', () => {
    expect(serializeOutput({ a: 1 })).toBe('{"a":1}')
  })
})

describe('applyShape — verbatim', () => {
  test('returns serialized output as-is', () => {
    expect(applyShape({ a: 1 }, 'verbatim')).toBe('{"a":1}')
  })

  test('long content not truncated', () => {
    const long = 'x'.repeat(10000)
    expect(applyShape(long, 'verbatim')).toBe(long)
  })
})

describe('applyShape — truncate', () => {
  test('content shorter than maxChars passes through', () => {
    expect(applyShape('short', { kind: 'truncate', maxChars: 100 })).toBe('short')
  })

  test('content longer than maxChars cut to maxChars', () => {
    const long = 'x'.repeat(1000)
    const out = applyShape(long, { kind: 'truncate', maxChars: 100 })
    expect(out.length).toBeLessThanOrEqual(150)   // 100 chars + truncation marker
    expect(out.startsWith('xxxx')).toBe(true)
  })

  test('tailHint=true appends explanatory marker about truncated bytes', () => {
    const long = 'x'.repeat(1000)
    const out = applyShape(long, { kind: 'truncate', maxChars: 100, tailHint: true })
    expect(out).toMatch(/\[\.\.\.truncated.*900.*chars\.\.\.\]/)
  })

  test('tailHint=false (or omitted) appends bare ellipsis marker', () => {
    const long = 'x'.repeat(1000)
    const out = applyShape(long, { kind: 'truncate', maxChars: 100 })
    expect(out.endsWith('...')).toBe(true)
  })

  test('object serialized then truncated', () => {
    const obj = { data: 'y'.repeat(1000) }
    const out = applyShape(obj, { kind: 'truncate', maxChars: 50 })
    expect(out.length).toBeLessThanOrEqual(100)
    expect(out.startsWith('{"data":"yyy')).toBe(true)
  })
})

describe('applyShape — tail', () => {
  test('content shorter than maxChars passes through', () => {
    expect(applyShape('short', { kind: 'tail', maxChars: 100 })).toBe('short')
  })

  test('content longer than maxChars cut to LAST maxChars', () => {
    const content = 'A'.repeat(500) + 'B'.repeat(500)
    const out = applyShape(content, { kind: 'tail', maxChars: 100 })
    expect(out.endsWith('BBBB')).toBe(true)
    expect(out.length).toBeLessThanOrEqual(150)
  })

  test('prepends ellipsis marker showing how many bytes dropped from head', () => {
    const long = 'z'.repeat(1000)
    const out = applyShape(long, { kind: 'tail', maxChars: 100 })
    expect(out.startsWith('[...')).toBe(true)
  })
})
