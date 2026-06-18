// Live e2e quality eval for HER slot-filling + repair-ticketing (#174).
//
// Drives the REAL repair-ticketing agent (reused verbatim from #162) against a
// real LLM — no stub gateway — over the tagged cases in cases.jsonl, then scores
// every case with the deterministic, judge-free scorer in ./scoring.ts and writes
// an aggregated report.
//
// Run: `npm run eval:repair` (gated under test:e2e:live — requires live model
// credentials VOLCENGINE_TOKEN + VOLCENGINE_API_BASE). Without credentials the
// script prints a SKIP notice and exits 0.
//
// Why state is read from live return values, NOT checkpoints: emit_ticket →
// completed is a terminal turn and #172 does not persist a checkpoint for it, so
// the assembled ticket only exists on the live `milkie.invoke` output. The
// accumulated slot ids come from `wm.mutated` events (the same source the example
// server uses to render slot chips), never a checkpoint.

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

import { Milkie } from '../../../src/runtime/Milkie.js'
import { MemoryStore } from '../../../src/store/MemoryStore.js'
import { MemoryEventStore } from '../../../src/trace/MemoryEventStore.js'
import { createGateway } from '../../../src/gateway/GatewayFactory.js'
import { OpenAICompatibleAdapter } from '../../../src/gateway/OpenAICompatibleAdapter.js'
import type { IModelGateway } from '../../../src/types/model.js'
import type { Event } from '../../../src/trace/types.js'

import { EntityResolver, type Schema } from '../resolver/EntityResolver.js'
import { buildRepairTicketingTools, repairTicketingAgentConfig } from '../src/agent.js'

// Eval model override (#174/#180): the example ships with doubao-seed-2.0-lite,
// too weak to exercise the flow. When DEEPSEEK_API_KEY is set, run the SAME
// agent/flow against DeepSeek; the committed example config is untouched.
const useDeepseek = !!process.env['DEEPSEEK_API_KEY']
const evalAgentConfig = useDeepseek
  ? {
      ...repairTicketingAgentConfig,
      model: {
        provider: 'deepseek',
        model:    process.env['EVAL_MODEL'] ?? 'deepseek-chat',
        adapter:  'openai-compatible' as const,
        baseUrl:  process.env['DEEPSEEK_API_BASE'] ?? 'https://api.deepseek.com',
      },
    }
  : repairTicketingAgentConfig

function buildEvalGateway(): IModelGateway {
  if (useDeepseek) {
    return new OpenAICompatibleAdapter({
      baseUrl: process.env['DEEPSEEK_API_BASE'] ?? 'https://api.deepseek.com',
      apiKey:  process.env['DEEPSEEK_API_KEY'],
    })
  }
  return createGateway(repairTicketingAgentConfig.model)
}

import {
  scoreCase, aggregate, LEVELS,
  type EvalCase, type Observed, type IdsByLevel, type CaseResult, type Metrics,
} from './scoring.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
const exampleDir = join(__dirname, '..')
const resolverDir = join(exampleDir, 'resolver')

// ─────────────────────────────── Fixtures ────────────────────────────────────

const schema = JSON.parse(readFileSync(join(resolverDir, 'schema.json'), 'utf8')) as Schema
const csv    = readFileSync(join(resolverDir, 'data.csv'), 'utf8')
const resolver = EntityResolver.load(schema, csv)

/** Known ids per level, parsed straight from the CSV by the schema's idColumns —
 *  the ground-truth dictionary the scorer uses to attribute wrong slot values. */
function buildIdsByLevel(): IdsByLevel {
  const lines = csv.trim().split(/\r?\n/)
  const header = lines[0]!.split(',')
  const ids = { site: new Set<string>(), building: new Set<string>(), department: new Set<string>(), assignee: new Set<string>() }
  const colOf = (name: string): number => header.indexOf(name)
  const cols = {
    site:       colOf(schema.levels.find(l => l.name === 'site')!.idColumn),
    building:   colOf(schema.levels.find(l => l.name === 'building')!.idColumn),
    department: colOf(schema.levels.find(l => l.name === 'department')!.idColumn),
    assignee:   colOf(schema.levels.find(l => l.name === 'assignee')!.idColumn),
  }
  for (const line of lines.slice(1)) {
    const f = line.split(',')
    for (const level of LEVELS) {
      const v = f[cols[level]]
      if (v) ids[level].add(v)
    }
  }
  return ids
}

function loadCases(): EvalCase[] {
  const text = readFileSync(join(__dirname, 'cases.jsonl'), 'utf8')
  const all = text.trim().split(/\r?\n/).filter(Boolean).map(l => JSON.parse(l) as EvalCase)
  // CASE_FILTER (optional): comma-separated keys; keep a case if any key matches its
  // exact id, its id prefix, or one of its tags. Lets you run a subset (e.g. only
  // `oneshot`) for a fast, low-cost, repeatable spot-check without touching the file.
  const filter = process.env['CASE_FILTER']?.trim()
  if (!filter) return all
  const keys = filter.split(',').map(s => s.trim()).filter(Boolean)
  return all.filter(c => keys.some(k => c.id === k || c.id.startsWith(k) || c.tag.includes(k)))
}

// ─────────────────────────── Live state extraction ───────────────────────────

/** A `wm.mutated` payload carries the full WorkingMemory as `{ data, log }`; the
 *  cumulative field map for a run is its last snapshot. Mirrors the example
 *  server's reader — same event source, never a checkpoint. */
function latestWorkingMemory(events: Event[], prev: Record<string, unknown>): Record<string, unknown> {
  let data = prev
  for (const e of events) {
    if (e.type === 'wm.mutated') {
      const snap = (e.payload as { snapshot?: { data?: Record<string, unknown> } }).snapshot
      if (snap?.data) data = snap.data
    }
  }
  return data
}

function isTicket(v: unknown): v is Record<string, unknown> {
  return !!v && typeof v === 'object' && typeof (v as Record<string, unknown>)['ticketId'] === 'string'
}

/**
 * Read the assembled ticket. #175 made assemble_ticket a TOOL whose result the
 * model RELAYS as formatted prose (a markdown table), so the run's text output is
 * no longer raw ticket JSON — but assemble_ticket also stores the ticket object in
 * WM (`ctx.workingMemory.set('ticket', …)`). Read it from WM first (authoritative,
 * model-paraphrase-proof); fall back to JSON in the text output for older shapes.
 */
function readTicket(workingMemory: Record<string, unknown>, output: string): Record<string, unknown> | null {
  if (isTicket(workingMemory['ticket'])) return workingMemory['ticket'] as Record<string, unknown>
  try {
    const parsed = JSON.parse(output) as Record<string, unknown>
    return isTicket(parsed) ? parsed : null
  } catch {
    return null
  }
}

/** Feed every turn of one case through a fresh Milkie and capture the observed end
 *  state (final WM, terminal-turn ticket, turn count). */
async function runCase(c: EvalCase, gateway: IModelGateway): Promise<Observed> {
  const eventStore = new MemoryEventStore()
  const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore, gateway })
  for (const tool of buildRepairTicketingTools(resolver)) milkie.registerTool(tool)
  milkie.registerAgent(evalAgentConfig)

  const contextId = `eval-${c.id}`
  let workingMemory: Record<string, unknown> = {}
  let lastOutput = ''
  let status: Observed['status'] = 'completed'

  try {
    for (const turn of c.turns) {
      const result = await milkie.invoke({
        agentId:   repairTicketingAgentConfig.agentId,
        goal:      '登记一张报修工单',
        input:     turn,
        contextId,
      })
      lastOutput = result.output
      status = result.status
      workingMemory = latestWorkingMemory(await eventStore.readByRunId(result.agentRunId), workingMemory)
    }
  } catch (err) {
    return {
      status: 'error', workingMemory, ticket: readTicket(workingMemory, lastOutput),
      turnCount: c.turns.length, error: (err as Error).message,
    }
  }

  return {
    status, workingMemory, ticket: readTicket(workingMemory, lastOutput),
    turnCount: c.turns.length,
    // Surface the invoke's error signal (e.g. a 401 / model error returned as the
    // run output) so the report's runtime_error rows are diagnosable, not silent.
    ...(status === 'error' ? { error: lastOutput } : {}),
  }
}

// ─────────────────────────────── Reporting ───────────────────────────────────

const pct = (n: number): string => `${(n * 100).toFixed(1)}%`

function renderReport(m: Metrics, results: CaseResult[], model: string, startedAt: string): string {
  const lines: string[] = []
  lines.push(`# repair-ticketing eval report`)
  lines.push('')
  lines.push(`- model: \`${model}\``)
  lines.push(`- started: ${startedAt}`)
  lines.push(`- cases: ${m.totalCases}`)
  lines.push(`- overall pass rate: **${pct(m.passRate)}**`)
  lines.push(`- average turn count: ${m.avgTurnCount.toFixed(2)}`)
  lines.push('')
  lines.push(`## Metrics`)
  lines.push('')
  lines.push(`| metric | value |`)
  lines.push(`| --- | --- |`)
  lines.push(`| slot full-match (${m.slotFill.cases} cases) | ${pct(m.slotFill.fullMatchRate)} |`)
  for (const level of LEVELS) lines.push(`| · ${level} per-slot | ${pct(m.slotFill.perLevelRate[level])} |`)
  lines.push(`| ticket field exact-match (${m.ticket.cases} cases) | ${pct(m.ticket.exactMatchRate)} |`)
  lines.push(`| clarification accuracy (${m.clarification.cases} cases) | ${pct(m.clarification.accuracy)} |`)
  lines.push(`| description clean — soft, not a pass-gate (${m.descriptionClean.cases} oneshot cases) | ${pct(m.descriptionClean.cleanRate)} |`)
  lines.push('')
  lines.push(`## Pass rate by tag`)
  lines.push('')
  lines.push(`| tag | passed/cases | rate |`)
  lines.push(`| --- | --- | --- |`)
  for (const tag of Object.keys(m.byTag).sort()) {
    const e = m.byTag[tag]!
    lines.push(`| ${tag} | ${e.passed}/${e.cases} | ${pct(e.passRate)} |`)
  }
  lines.push('')
  lines.push(`## Failure attribution`)
  lines.push('')
  const dist = Object.entries(m.failureDistribution).sort((a, b) => b[1] - a[1])
  if (dist.length === 0) {
    lines.push(`_no failures_`)
  } else {
    lines.push(`| kind | count |`)
    lines.push(`| --- | --- |`)
    for (const [k, v] of dist) lines.push(`| ${k} | ${v} |`)
  }
  lines.push('')
  lines.push(`## Failing cases`)
  lines.push('')
  const failing = results.filter(r => !r.passed)
  if (failing.length === 0) {
    lines.push(`_all cases passed_`)
  } else {
    lines.push(`| id | tag | failures |`)
    lines.push(`| --- | --- | --- |`)
    for (const r of failing) lines.push(`| ${r.id} | ${r.tag.join(',')} | ${r.failures.join(', ') || '—'} |`)
  }
  lines.push('')
  return lines.join('\n')
}

// ─────────────────────────────────── Main ────────────────────────────────────

async function main(): Promise<void> {
  const hasVolc = process.env['VOLCENGINE_TOKEN'] && process.env['VOLCENGINE_API_BASE']
  if (!useDeepseek && !hasVolc) {
    console.log(
      'SKIPPED: live model credentials not set. ' +
      'Set VOLCENGINE_TOKEN + VOLCENGINE_API_BASE (or DEEPSEEK_API_KEY) to run the repair-ticketing eval.',
    )
    return
  }

  const cases = loadCases()
  const ids   = buildIdsByLevel()
  const gateway = buildEvalGateway()
  const model = evalAgentConfig.model.model
  const startedAt = new Date().toISOString()

  console.log(`Running ${cases.length} cases against ${model} …`)
  const results: CaseResult[] = []
  for (const c of cases) {
    const obs = await runCase(c, gateway)
    const r   = scoreCase(c, obs, ids)
    results.push(r)
    console.log(`  ${r.passed ? 'PASS' : 'FAIL'}  ${c.id.padEnd(28)} ${r.failures.join(',')}`)
    if (obs.error) console.log(`        ↳ error: ${obs.error}`)
  }

  const metrics = aggregate(cases, results)
  const report  = renderReport(metrics, results, model, startedAt)

  const reportsDir = join(__dirname, 'reports')
  mkdirSync(reportsDir, { recursive: true })
  const stamp = startedAt.replace(/[:.]/g, '-')
  writeFileSync(join(reportsDir, `eval-${stamp}.md`), report)
  writeFileSync(join(reportsDir, `eval-${stamp}.json`), JSON.stringify({ startedAt, model, metrics, results }, null, 2))

  console.log('\n' + report)
  console.log(`\nReports written to ${reportsDir}`)
}

main().catch(err => { console.error(err); process.exit(1) })
