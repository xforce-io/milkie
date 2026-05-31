import type { Event } from '../types.js'

/**
 * Human-readable one-line label for an event. Shared by the HTML "Why?" block
 * and the decision viewer's causal chain, so both speak the same language.
 * tool.requested labels carry a short, truncated input summary so repeated
 * calls to the same tool (e.g. grep "孔明" vs grep "诸葛亮") are distinguishable.
 */
function shortInput(input: unknown): string | undefined {
  if (input === undefined || input === null) return undefined
  const json = JSON.stringify(input)
  if (json === undefined || json === '{}' || json === 'null' || json === '""') return undefined
  return json.length > 24 ? json.slice(0, 24) + '…' : json
}

export function summarizeEvent(event: Event): string {
  const p = event.payload as { toolName?: unknown; input?: unknown }
  if ((event.type === 'tool.requested' || event.type === 'tool.responded')
      && typeof p?.toolName === 'string') {
    const arg = event.type === 'tool.requested' ? shortInput(p.input) : undefined
    return arg ? `${event.type}(${p.toolName} · ${arg})` : `${event.type}(${p.toolName})`
  }
  return event.type
}
