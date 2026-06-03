// #128: project one run's events into a canonical Message[] transcript.
// user ← agent.run.started.input; assistant ← llm.responded.response.content;
// tool ← tool.responded (paired by toolCallId). Events are walked in append
// order, which is the correct transcript order (assistant tool_use → tool_result
// → assistant final). Uses only discrete I/O events, so a run contributes only
// its own turn — no restored-prefix duplication when runs are concatenated.
import { runEventsToMessages } from '../trace/diagnostics/sessionHistory'
import type { Event } from '../trace/types'
import type { Message } from '../types/common'

let seq = 0
function ev<P>(type: Event['type'], payload: P): Event<P> {
  seq += 1
  return { id: `e${seq}`, runId: 'rA', type, actor: 'agent', timestamp: seq, payload }
}

describe('#128 runEventsToMessages — one run → Message[]', () => {
  it('projects a tool-using turn into user / assistant(tool_use) / tool_result / assistant(final)', () => {
    const events: Event[] = [
      ev('agent.run.started', { agentId: 'a', goal: 'g', input: 'hi', contextId: 'X' }),
      ev('llm.requested', { request: { model: 'm', messages: [] }, requestHash: 'h1' }),
      ev('llm.responded', {
        response: {
          content: [
            { type: 'text', text: 'let me check' },
            { type: 'tool_use', id: 't1', name: 'search', input: { q: 'x' } },
          ],
          toolCalls: [{ id: 't1', name: 'search', input: { q: 'x' } }],
          finishReason: 'tool_use',
        },
        requestHash: 'h1',
      }),
      ev('tool.requested', { toolName: 'search', input: { q: 'x' }, toolCallId: 't1', requestHash: 'h2' }),
      ev('tool.responded', { toolName: 'search', toolCallId: 't1', status: 'ok', output: { r: 'found' }, requestHash: 'h2' }),
      ev('llm.requested', { request: { model: 'm', messages: [] }, requestHash: 'h3' }),
      ev('llm.responded', {
        response: { content: [{ type: 'text', text: 'done' }], toolCalls: [], finishReason: 'end_turn' },
        requestHash: 'h3',
      }),
      ev('agent.run.completed', { status: 'completed', lastTextOutput: 'done' }),
    ]

    const messages = runEventsToMessages(events)

    const expected: Message[] = [
      { role: 'user', content: [{ type: 'text', text: 'hi' }] },
      { role: 'assistant', content: [
        { type: 'text', text: 'let me check' },
        { type: 'tool_use', id: 't1', name: 'search', input: { q: 'x' } },
      ] },
      { role: 'tool', content: [{ type: 'tool_result', tool_use_id: 't1', content: '{"r":"found"}' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'done' }] },
    ]
    expect(messages).toEqual(expected)
  })

  it('marks an errored tool result with is_error and uses the error message', () => {
    const events: Event[] = [
      ev('agent.run.started', { agentId: 'a', goal: 'g', input: 'go', contextId: 'X' }),
      ev('llm.responded', {
        response: { content: [{ type: 'tool_use', id: 't9', name: 'boom', input: {} }], toolCalls: [{ id: 't9', name: 'boom', input: {} }], finishReason: 'tool_use' },
        requestHash: 'h1',
      }),
      ev('tool.responded', { toolName: 'boom', toolCallId: 't9', status: 'error', error: { message: 'kaboom' }, requestHash: 'h2' }),
    ]

    const messages = runEventsToMessages(events)

    expect(messages[messages.length - 1]).toEqual({
      role: 'tool',
      content: [{ type: 'tool_result', tool_use_id: 't9', content: 'kaboom', is_error: true }],
    })
  })

  it('passes a string tool output through without JSON-encoding', () => {
    const events: Event[] = [
      ev('agent.run.started', { agentId: 'a', goal: 'g', input: 'go', contextId: 'X' }),
      ev('tool.responded', { toolName: 't', toolCallId: 'c1', status: 'ok', output: 'plain text', requestHash: 'h' }),
    ]
    const messages = runEventsToMessages(events)
    expect(messages[messages.length - 1]).toEqual({
      role: 'tool',
      content: [{ type: 'tool_result', tool_use_id: 'c1', content: 'plain text' }],
    })
  })
})
