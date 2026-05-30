import type { Event } from '../types.js'

/**
 * Human-readable one-line label for an event. Shared by the HTML "Why?" block
 * and (future) the CLI explainer (#36), so both speak the same language.
 */
export function summarizeEvent(event: Event): string {
  const p = event.payload as { toolName?: unknown }
  if ((event.type === 'tool.requested' || event.type === 'tool.responded')
      && typeof p?.toolName === 'string') {
    return `${event.type}(${p.toolName})`
  }
  return event.type
}
