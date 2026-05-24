import type { Event } from '../types.js'
import { buildTimelineTree, type TimelineEntry, type TimelineNode } from './tree.js'
import { STYLES, SCRIPT } from './template.js'

function esc(s: string): string {
  return s
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;')
}

// Status values allowed to appear in the badge CSS class. Anything else gets
// no extra class (badge defaults to the green completed style). Text content
// is always passed through esc(), so untrusted status strings are still
// visible as escaped text — just not as an attribute injection vector.
const BADGE_STATUS_CLASSES = new Set(['interrupted', 'error', 'in-flight'])

function summaryFor(entry: TimelineEntry): string {
  if (entry.kind === 'llm')       return 'LLM call' + (entry.respondedId ? '' : ' (no response)')
  if (entry.kind === 'tool')      return 'tool: ' + esc(entry.toolName) + (entry.respondedId ? '' : ' (no response)')
  return entry.eventType === 'agent.run.started' ? 'run started' : 'run completed'
}

function payloadFor(entry: TimelineEntry, eventById: Map<string, Event>): string {
  const sections: string[] = []
  if (entry.kind === 'lifecycle') {
    const evt = eventById.get(entry.eventId)
    if (evt) sections.push(JSON.stringify(evt.payload, null, 2))
  } else {
    const req = eventById.get(entry.requestedId)
    if (req) sections.push('request:\n' + JSON.stringify(req.payload, null, 2))
    const resp = entry.respondedId ? eventById.get(entry.respondedId) : undefined
    if (resp) sections.push('response:\n' + JSON.stringify(resp.payload, null, 2))
  }
  return sections.length > 0 ? esc(sections.join('\n\n')) : ''
}

function renderEntry(entry: TimelineEntry, eventById: Map<string, Event>): string {
  const icon = entry.kind === 'llm' ? '◆' : entry.kind === 'tool' ? '▣' : '●'
  return `<div class="entry ${entry.kind}" data-kind="${entry.kind}">`
       + `<div class="entry-head">`
       + `<span class="icon">${icon}</span>`
       + `<span class="summary">${summaryFor(entry)}</span>`
       + `<span class="ts">${entry.timestamp}</span>`
       + `</div>`
       + `<pre class="payload">${payloadFor(entry, eventById)}</pre>`
       + `</div>`
}

function renderNode(node: TimelineNode, eventById: Map<string, Event>): string {
  const status = node.status ?? 'in-flight'
  const badgeClass = BADGE_STATUS_CLASSES.has(status) ? ' ' + status : ''
  return `<section class="run" data-run-id="${esc(node.runId)}">`
       + `<div class="run-head">`
       + `<strong>${esc(node.agentId ?? '(unknown)')}</strong>`
       + `<span class="run-id">${esc(node.runId)}</span>`
       + `<span class="badge${badgeClass}">${esc(status)}</span>`
       + `</div>`
       + node.entries.map(en => renderEntry(en, eventById)).join('')
       + (node.children.length > 0
           ? `<div class="child-run">${node.children.map(n => renderNode(n, eventById)).join('')}</div>`
           : '')
       + `</section>`
}

/**
 * Pure projection: flat events → self-contained HTML report.
 *
 * The output is a single HTML document with inline CSS, vanilla JS for
 * fold/unfold + type filter, and the raw events embedded as a JSON script
 * tag so the file is its own re-renderable archive.
 *
 * No I/O, no clock, no randomness. The renderer cannot reach into the event
 * store — this is the architectural firewall behind "UI is a pure projection
 * over CLI / SDK output" (ARCHITECTURE.md `## User-facing surfaces`).
 */
export function renderHtml(events: Event[]): string {
  const tree = buildTimelineTree(events)
  const eventById = new Map<string, Event>()
  for (const evt of events) eventById.set(evt.id, evt)
  const dataJson = JSON.stringify(events)
    // close-tag-safe inlining: prevent the JSON from ending the script element.
    .replace(/<\/script/gi, '<\\/script')
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>milkie trace report</title>
<style>${STYLES}</style>
</head>
<body>
<h1>milkie trace report</h1>
<div class="filters">
  <span class="chip" data-kind="llm">LLM</span>
  <span class="chip" data-kind="tool">tool</span>
  <span class="chip" data-kind="lifecycle">lifecycle</span>
</div>
${tree.map(n => renderNode(n, eventById)).join('')}
<script type="application/json" id="trace-data">${dataJson}</script>
<script>${SCRIPT}</script>
</body>
</html>`
}
