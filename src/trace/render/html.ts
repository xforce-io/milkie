import type { Event } from '../types.js'
import { buildTimelineTree, type TimelineEntry, type TimelineNode } from './tree.js'
import { STYLES, SCRIPT } from './template.js'
import { explainTransition, type TransitionExplanation } from '../diagnostics/explainTransition.js'
import { contextRefsAt, regionReuseCounts, type RegionContentRef } from '../RegionContextView.js'

interface RegionCtx {
  events:        Event[]
  reuseCounts:   Map<string, number>
  regionContent: Map<string, string>
}

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
  if (entry.kind === 'region')    return esc(entry.summary)
  if (entry.kind === 'fsm')       return esc(entry.summary)
  return entry.eventType === 'agent.run.started' ? 'run started' : 'run completed'
}

function payloadFor(entry: TimelineEntry, eventById: Map<string, Event>): string {
  const sections: string[] = []
  if (entry.kind === 'lifecycle') {
    const evt = eventById.get(entry.eventId)
    if (evt) sections.push(JSON.stringify(evt.payload, null, 2))
  } else if (entry.kind === 'region') {
    sections.push(JSON.stringify(entry.payload, null, 2))
  } else if (entry.kind === 'fsm') {
    sections.push(JSON.stringify(entry.payload, null, 2))
  } else {
    const req = eventById.get(entry.requestedId)
    if (req) sections.push('request:\n' + JSON.stringify(req.payload, null, 2))
    const resp = entry.respondedId ? eventById.get(entry.respondedId) : undefined
    if (resp) sections.push('response:\n' + JSON.stringify(resp.payload, null, 2))
  }
  return sections.length > 0 ? esc(sections.join('\n\n')) : ''
}

function entryEventIds(entry: TimelineEntry): string[] {
  if (entry.kind === 'llm' || entry.kind === 'tool') {
    return entry.respondedId ? [entry.requestedId, entry.respondedId] : [entry.requestedId]
  }
  return [entry.eventId]
}

function renderWhy(exp: TransitionExplanation): string {
  const trigger = exp.trigger.causedByEventId
    ? `<div class="why-trigger">触发: <a href="#ev-${esc(exp.trigger.causedByEventId)}">${esc(exp.trigger.causedBySummary ?? exp.trigger.causedByEventId)}</a></div>`
    : ''
  const guards = exp.guards.length
    ? `<div class="why-guards">guard: ${esc(exp.guards.map(g => `${g.guardId}=${String(g.result)}`).join(', '))}</div>`
    : ''
  const chain = `<div class="why-chain">`
    + exp.causalChain.map(c => `<a href="#ev-${esc(c.eventId)}">${esc(c.summary)}</a>`).join(' → ')
    + `</div>`
  return `<div class="why">`
       + `<div class="why-summary">${esc(exp.summary)}</div>`
       + trigger + guards + chain
       + `</div>`
}

function renderAssembled(requestedId: string, ctx: RegionCtx): string {
  const refs: RegionContentRef[] = [...contextRefsAt(ctx.events, requestedId, 'at').values()]
  if (refs.length === 0) return ''
  const rows = refs.map(ref => {
    const hash    = ref.contentHash
    const reuse   = hash ? (ctx.reuseCounts.get(hash) ?? 1) : 1
    const reuseB  = reuse > 1 ? ` <span class="reuse">复用 ×${reuse}</span>` : ''
    const avail   = !!(hash && ctx.regionContent.has(hash))
    const note    = hash && !avail ? ` <span class="ar-note">(内容不可用)</span>` : ''
    const meta    = `<span class="ar-id">${esc(ref.id)}</span> · ${esc(ref.section)} · ${esc(ref.target)} · ${esc(ref.reason)}`
    const preview = avail ? `<pre class="region-preview"></pre>` : ''
    const dataH   = hash ? ` data-hash="${esc(hash)}"` : ''
    return `<div class="ar-row stab-${esc(ref.stability)}"${dataH}><div class="ar-meta">${meta}${reuseB}${note}</div>${preview}</div>`
  }).join('')
  return `<div class="assembled"><div class="ar-head">Assembled by ${refs.length} regions</div>${rows}</div>`
}

function renderEntry(
  entry: TimelineEntry,
  eventById: Map<string, Event>,
  explanations: Map<string, TransitionExplanation>,
  regionCtx: RegionCtx,
): string {
  const icon = entry.kind === 'llm' ? '◆'
             : entry.kind === 'tool' ? '▣'
             : entry.kind === 'region'
               ? (entry.eventType === 'context.boundary.applied' ? '⌖'
                 : entry.eventType === 'region.added' ? '＋' : '－')
             : entry.kind === 'fsm' ? '⇒'
             : '●'
  const ids = entryEventIds(entry)
  const primaryId = ids[0]!
  const extraAnchors = ids.slice(1).map(id => `<span class="anchor" id="ev-${esc(id)}"></span>`).join('')
  const why = entry.kind === 'fsm' && explanations.has(entry.eventId)
    ? renderWhy(explanations.get(entry.eventId)!)
    : ''
  const assembled = entry.kind === 'llm' ? renderAssembled(entry.requestedId, regionCtx) : ''
  return `<div class="entry ${entry.kind}" data-kind="${entry.kind}" id="ev-${esc(primaryId)}">`
       + extraAnchors
       + `<div class="entry-head">`
       + `<span class="icon">${icon}</span>`
       + `<span class="summary">${summaryFor(entry)}</span>`
       + `<span class="ts">${entry.timestamp}</span>`
       + `</div>`
       + why
       + assembled
       + `<pre class="payload">${payloadFor(entry, eventById)}</pre>`
       + `</div>`
}

function renderNode(
  node: TimelineNode,
  eventById: Map<string, Event>,
  explanations: Map<string, TransitionExplanation>,
  regionCtx: RegionCtx,
): string {
  const status = node.status ?? 'in-flight'
  const badgeClass = BADGE_STATUS_CLASSES.has(status) ? ' ' + status : ''
  return `<section class="run" data-run-id="${esc(node.runId)}">`
       + `<div class="run-head">`
       + `<strong>${esc(node.agentId ?? '(unknown)')}</strong>`
       + `<span class="run-id">${esc(node.runId)}</span>`
       + `<span class="badge${badgeClass}">${esc(status)}</span>`
       + `</div>`
       + node.entries.map(en => renderEntry(en, eventById, explanations, regionCtx)).join('')
       + (node.children.length > 0
           ? `<div class="child-run">${node.children.map(n => renderNode(n, eventById, explanations, regionCtx)).join('')}</div>`
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
export function renderHtml(events: Event[], opts: { regionContent?: Map<string, string> } = {}): string {
  const tree = buildTimelineTree(events)
  const eventById = new Map<string, Event>()
  for (const evt of events) eventById.set(evt.id, evt)
  const explanations = new Map<string, TransitionExplanation>()
  for (const evt of events) {
    if (evt.type === 'fsm.transition') explanations.set(evt.id, explainTransition(events, evt.id))
  }

  const regionContent = opts.regionContent ?? new Map<string, string>()
  const regionCtx: RegionCtx = { events, reuseCounts: regionReuseCounts(events), regionContent }

  const registryJson = JSON.stringify(Object.fromEntries(regionContent))
    .replace(/<\/script/gi, '<\\/script')
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
  <span class="chip" data-kind="region">region</span>
  <span class="chip" data-kind="fsm">fsm</span>
</div>
${tree.map(n => renderNode(n, eventById, explanations, regionCtx)).join('')}
<script type="application/json" id="region-content">${registryJson}</script>
<script type="application/json" id="trace-data">${dataJson}</script>
<script>${SCRIPT}</script>
</body>
</html>`
}
