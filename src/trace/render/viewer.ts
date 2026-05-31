import type { Event } from '../types.js'
import { buildDecisionSpine, type DecisionNode } from '../diagnostics/buildDecisionSpine.js'
import { explainTransition } from '../diagnostics/explainTransition.js'
import { explainLlmCall } from '../diagnostics/explainLlmCall.js'
import { explainToolCall } from '../diagnostics/explainToolCall.js'
import { renderTimelineSections } from './html.js'
import { VIEWER_STYLES, VIEWER_SCRIPT } from './viewer-template.js'

function esc(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;')
}

interface PanelRecord {
  title:            string
  bodyHtml:         string
  causeDecisionId?: string
  chain:            Array<{ eventId: string; type: string; summary: string }>
  rawJson:          string
}

// One panel "explanation" record per node, consumed verbatim by the client JS.
function panelRecord(events: Event[], node: DecisionNode, eventById: Map<string, Event>): PanelRecord {
  const base = {
    causeDecisionId: node.causeDecisionId,
    chain: [] as Array<{ eventId: string; type: string; summary: string }>,
    rawJson: JSON.stringify(eventById.get(node.eventId)?.payload ?? {}, null, 2),
  }
  if (node.kind === 'transition') {
    const x = explainTransition(events, node.eventId)
    const guards = x.guards.map(g => `${esc(g.guardId)} 判定 ${esc(String(g.result))}`).join('、')
    return { ...base, chain: x.causalChain, title: `为什么 ${esc(x.from)} → ${esc(x.to)}?`,
      bodyHtml: `触发: ${esc(x.trigger.causedBySummary ?? x.trigger.name)}<br>guard: ${guards || '(无)'}` }
  }
  if (node.kind === 'llm') {
    const x = explainLlmCall(events, node.eventId)
    return { ...base, chain: x.causalChain, title: `为什么这次 LLM 调用?`,
      bodyHtml: `state: ${esc(x.fsmState ?? '(未知)')}<br>触发: ${esc(x.trigger.causedBySummary ?? '(无上游)')}<br>prompt 由 ${x.regionCount} 个 region 拼成` }
  }
  if (node.kind === 'tool') {
    const x = explainToolCall(events, node.eventId)
    return { ...base, chain: x.causalChain, title: `工具 ${esc(x.toolName)}`,
      bodyHtml: `调用方: ${esc(x.trigger.causedBySummary ?? '(无上游)')}<br>入参: ${esc(JSON.stringify(x.input))}<br>出参: ${esc(JSON.stringify(x.output ?? null))}` }
  }
  // output
  const evt = eventById.get(node.eventId)
  const out = (evt?.payload as { lastTextOutput?: string; status?: string } | undefined)
  return { ...base, chain: [], title: '为什么是这个结果?',
    bodyHtml: `输出: ${esc(out?.lastTextOutput ?? out?.status ?? '')}<br>由上游决策产生(点 ← 谁导致的 下钻)` }
}

/**
 * Causal drill-down trace viewer: decision spine + Why panel + a raw timeline
 * tab. Self-contained static HTML; all explanations precomputed and embedded,
 * client JS only looks them up. Pure projection (region content hydration is
 * the caller's job, passed via opts, same as renderHtml).
 */
export function renderViewer(events: Event[], opts: { regionContent?: Map<string, string> } = {}): string {
  const spine = buildDecisionSpine(events)
  const eventById = new Map<string, Event>()
  for (const e of events) eventById.set(e.id, e)

  const explanations: Record<string, PanelRecord> = {}
  for (const node of spine.nodes) explanations[node.eventId] = panelRecord(events, node, eventById)

  const spineHtml = spine.nodes.map(n => {
    if (n.kind === 'output') {
      return `<div class="spine-output"><div class="node k-output" data-id="${esc(n.eventId)}">${esc(n.label)}<span class="why-entry">❓ 为什么是这个结果</span></div></div>`
    }
    return `<div class="node k-${n.kind}" data-id="${esc(n.eventId)}">${esc(n.label)}</div>`
  }).join('')

  const spineJson = JSON.stringify(spine).replace(/<\/script/gi, '<\\/script')
  const expsJson = JSON.stringify(explanations).replace(/<\/script/gi, '<\\/script')

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>milkie trace viewer</title>
<style>${VIEWER_STYLES}</style>
</head>
<body>
<h1>milkie trace viewer</h1>
<div class="tabs">
  <span class="tab active" data-tab="decision">决策视图</span>
  <span class="tab" data-tab="raw">原始时间线</span>
</div>
<div id="pane-decision" class="pane active">
  <div class="spine">${spineHtml || '<p style="color:#888;font-size:12px">无决策事件</p>'}</div>
  <div id="why-panel"><p class="ph">从底部输出 ❓ 或任一决策点开始</p></div>
</div>
<div id="pane-raw" class="pane">${renderTimelineSections(events, opts)}</div>
<script type="application/json" id="spine-data">${spineJson}</script>
<script type="application/json" id="explanations-data">${expsJson}</script>
<script>${VIEWER_SCRIPT}</script>
</body>
</html>`
}
