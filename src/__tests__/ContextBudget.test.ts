import {
  applyContextBudget,
  ContextBudgetError,
  estimateBudgetTokens,
  type ContextBudgetItem,
} from '../context/budget'
import type { Message } from '../types/common'

function textMessage(role: Message['role'], text: string): Message {
  return { role, content: [{ type: 'text', text }] }
}

function item(
  id: string,
  region: ContextBudgetItem['region'],
  target: ContextBudgetItem['target'],
  content: ContextBudgetItem['content'],
  order: number,
): ContextBudgetItem {
  return { id, region, target, content, order }
}

describe('applyContextBudget', () => {
  it('drops oldest complete history sources before exceeding the global cap', () => {
    const items = [
      item('header', 'control', 'system', 'system', 0),
      item('current', 'currentTurn', 'message', [textMessage('user', 'current')], 1),
      item('history:old', 'history', 'message', [textMessage('user', 'old user'), textMessage('assistant', 'old answer')], 2),
      item('history:new', 'history', 'message', [textMessage('user', 'new user'), textMessage('assistant', 'new answer')], 3),
    ]

    const projected = applyContextBudget(items, {
      maxInputTokens: 400,
      regionCaps: { history: 180 },
    }, 7)

    expect(projected.report.totalEstimated).toBeLessThanOrEqual(400)
    expect(projected.messages.flatMap(message => message.content)
      .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
      .map(part => part.text)).toContain('new user')
    expect(projected.messages.flatMap(message => message.content)
      .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
      .map(part => part.text)).not.toContain('old user')
    expect(projected.report.notices).toEqual(expect.arrayContaining([
      expect.objectContaining({ sourceId: 'history:old', region: 'history' }),
    ]))
  })

  it('fails closed before a model call when a required region exceeds its cap', () => {
    const items = [
      item('header', 'control', 'system', 'system instructions that are too large', 0),
      item('current', 'currentTurn', 'message', [textMessage('user', 'current')], 1),
    ]

    expect(() => applyContextBudget(items, {
      maxInputTokens: 256,
      regionCaps: { control: 8 },
    }, 1)).toThrow(ContextBudgetError)
    try {
      applyContextBudget(items, { maxInputTokens: 256, regionCaps: { control: 8 } }, 1)
    } catch (error) {
      expect(error).toMatchObject({
        code: 'CONTEXT_BUDGET_REQUIRED_REGION_EXCEEDED',
        region: 'control',
      })
    }
  })

  it('keeps tool protocol identifiers while bounding tool-result text', () => {
    const items = [
      item('header', 'control', 'system', 'system', 0),
      item('current', 'currentTurn', 'message', [textMessage('user', 'go')], 1),
      item('scratch:assistant', 'scratchpad', 'message', [{
        role: 'assistant',
        content: [{ type: 'tool_use', id: 'call-1', name: 'lookup', input: { q: 'q' } }],
      }], 2),
      item('scratch:result', 'scratchpad', 'message', [{
        role: 'tool',
        content: [{ type: 'tool_result', tool_use_id: 'call-1', content: 'x'.repeat(4000), is_error: true }],
      }], 3),
    ]

    const projected = applyContextBudget(items, {
      maxInputTokens: 700,
      regionCaps: { scratchpad: 400 },
    }, 1)
    const toolResult = projected.messages
      .flatMap(message => message.content)
      .find((part): part is Extract<typeof part, { type: 'tool_result' }> => part.type === 'tool_result')!

    expect(toolResult.tool_use_id).toBe('call-1')
    expect(toolResult.is_error).toBe(true)
    expect(estimateBudgetTokens(projected.messages)).toBeLessThanOrEqual(700)
    expect(projected.report.notices).toEqual(expect.arrayContaining([
      expect.objectContaining({ sourceId: 'scratch:result', region: 'scratchpad', evidenceRef: 'scratch:result' }),
    ]))
  })

  it('fails closed when no total budget remains for an active tool protocol', () => {
    const required = [
      item('header', 'control', 'system', 'system', 0),
      item('current', 'currentTurn', 'message', [textMessage('user', 'go')], 1),
    ]
    const exactLimit = applyContextBudget(required, { maxInputTokens: 1000 }, 1).report.totalEstimated
    const withProtocol = [...required, item('scratch:assistant', 'scratchpad', 'message', [{
      role: 'assistant',
      content: [{ type: 'tool_use', id: 'call-1', name: 'lookup', input: {} }],
    }], 2)]

    expect(() => applyContextBudget(withProtocol, {
      maxInputTokens: exactLimit,
      regionCaps: { scratchpad: exactLimit },
    }, 1)).toThrow(expect.objectContaining({
      code: 'CONTEXT_BUDGET_REQUIRED_REGION_EXCEEDED', region: 'scratchpad',
    }))
  })

  it('drops oldest working-memory log entries before using a hash-only fallback', () => {
    const raw = {
      data: { current: 'keep' },
      log: [{ entry: 'old'.repeat(90) }, { entry: 'new'.repeat(20) }],
    }
    const rendered = '\n--- Working Memory ---\n' + JSON.stringify(raw, null, 2)
    const items = [
      item('header', 'control', 'system', 'system', 0),
      item('current', 'currentTurn', 'message', [textMessage('user', 'go')], 1),
      { ...item('wm', 'workingMemory', 'system', rendered, 2), rawContent: raw },
    ]

    const projected = applyContextBudget(items, {
      maxInputTokens: 1000,
      regionCaps: { workingMemory: 250 },
    }, 1)

    expect(projected.system).toContain('new'.repeat(20))
    expect(projected.system).not.toContain('old'.repeat(20))
  })

  it('omits optional working memory when its positive cap cannot encode a string', () => {
    const raw = { data: { current: 'keep' }, log: [{ entry: 'old' }] }
    const items = [
      item('header', 'control', 'system', 'system', 0),
      item('current', 'currentTurn', 'message', [textMessage('user', 'go')], 1),
      {
        ...item('wm', 'workingMemory', 'system', '\n--- Working Memory ---\n' + JSON.stringify(raw), 2),
        rawContent: raw,
      },
    ]

    const projected = applyContextBudget(items, {
      maxInputTokens: 1000,
      regionCaps: { workingMemory: 1 },
    }, 1)

    expect(projected.system).not.toContain('Working Memory')
    expect(projected.report.regions).toContainEqual(expect.objectContaining({
      region: 'workingMemory', projectedEstimated: 0,
    }))
    expect(projected.report.notices).toContainEqual(expect.objectContaining({
      sourceId: 'wm', projectedBudgetTokens: 0, reason: 'region_cap',
    }))
  })

  it('normalizes implicit region caps to a low valid total', () => {
    const projected = applyContextBudget([
      item('header', 'control', 'system', 'system', 0),
      item('current', 'currentTurn', 'message', [textMessage('user', 'go')], 1),
    ], { maxInputTokens: 4096 }, 1)

    expect(projected.report.regionCaps).toEqual(expect.objectContaining({
      control: 4096,
      currentTurn: 4096,
      scratchpad: 4096,
      workingMemory: 4096,
      sessionContext: 2048,
    }))
  })

  it('charges fixed request fields in the reported total', () => {
    const items = [
      item('header', 'control', 'system', 'system', 0),
      item('current', 'currentTurn', 'message', [textMessage('user', 'go')], 1),
    ]
    const projected = applyContextBudget(items, { maxInputTokens: 1000 }, 1, {
      model: 'specific-model', cacheBreakpoint: 'system-end',
    })
    expect(projected.report.totalEstimated).toBe(estimateBudgetTokens({
      model: 'specific-model',
      system: projected.system,
      messages: projected.messages,
      cacheBreakpoint: 'system-end',
    }))
  })

  it('rejects invalid region caps deterministically', () => {
    expect(() => applyContextBudget([], {
      maxInputTokens: 100,
      regionCaps: { history: 0 },
    }, 1)).toThrow(expect.objectContaining({ code: 'CONTEXT_BUDGET_INVALID_CONFIG' }))
  })
})
