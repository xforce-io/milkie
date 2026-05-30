import { summarizeEvent } from '../trace/diagnostics/summarizeEvent'
import type { Event } from '../trace/types'

const ev = (type: string, payload: unknown): Event =>
  ({ id: 'x', runId: 'r1', actor: 'a', type: type as Event['type'], timestamp: 0, payload })

describe('summarizeEvent', () => {
  it('labels tool events with the tool name', () => {
    expect(summarizeEvent(ev('tool.responded', { toolName: 'classify_intent', output: {} })))
      .toBe('tool.responded(classify_intent)')
    expect(summarizeEvent(ev('tool.requested', { toolName: 'search', input: {} })))
      .toBe('tool.requested(search)')
  })

  it('falls back to the bare type for events without a tool name', () => {
    expect(summarizeEvent(ev('llm.responded', {}))).toBe('llm.responded')
    expect(summarizeEvent(ev('agent.run.started', { agentId: 'x' }))).toBe('agent.run.started')
  })
})
