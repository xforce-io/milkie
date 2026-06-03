import type { Event, LlmRespondedPayload, ToolRespondedPayload, AgentRunStartedPayload } from '../types.js'
import type { Message, MessageContent } from '../../types/common.js'

/**
 * #128: project ONE run's events into a canonical `Message[]` transcript.
 *
 * Walks the run's events in append order (which is the correct transcript order:
 * assistant `tool_use` → `tool_result` → assistant final) and emits, from the
 * discrete I/O events only:
 *   - `agent.run.started.input`        → a user message
 *   - `llm.responded.response.content` → an assistant message (text + tool_use)
 *   - `tool.responded`                 → a tool message (tool_result, paired by toolCallId)
 *
 * Using responded events (not the `llm.requested.messages` snapshot) means a run
 * contributes only its own turn — the restored prior-turn prefix is never
 * re-emitted as events — so concatenating per-run projections does not duplicate.
 */
export function runEventsToMessages(events: Event[]): Message[] {
  const messages: Message[] = []

  for (const e of events) {
    switch (e.type) {
      case 'agent.run.started': {
        const { input } = e.payload as AgentRunStartedPayload
        messages.push({ role: 'user', content: [{ type: 'text', text: input }] })
        break
      }
      case 'llm.responded': {
        const { response } = e.payload as LlmRespondedPayload
        const content = response.content ?? []
        if (content.length > 0) messages.push({ role: 'assistant', content })
        break
      }
      case 'tool.responded': {
        messages.push({ role: 'tool', content: [toolResult(e.payload as ToolRespondedPayload)] })
        break
      }
      default:
        break
    }
  }

  return messages
}

function toolResult(p: ToolRespondedPayload): MessageContent {
  const isError = p.status === 'error' || (p.status === undefined && p.error !== undefined)
  const text = isError
    ? (p.error?.message ?? 'error')
    : (typeof p.output === 'string' ? p.output : JSON.stringify(p.output))
  return {
    type:        'tool_result',
    tool_use_id: p.toolCallId ?? '',
    content:     text,
    ...(isError ? { is_error: true } : {}),
  }
}
