// Deterministic scoring for the repair-ticketing e2e eval (#174).
//
// This module is intentionally PURE — no milkie, no I/O, no model. The harness
// (run-eval.ts) drives the live agent and hands the *observed* end state here;
// scoring is exact-match against discrete ground-truth ids, so there is no LLM
// judge and the verdict is fully reproducible. Keeping it pure also lets the
// unit test exercise every scoring branch without live credentials.

/** Ordered hierarchy levels — mirrors REQUIRED_SLOTS in the example's agent. */
export const LEVELS = ['site', 'building', 'department', 'assignee'] as const
export type Level = (typeof LEVELS)[number]

/** A single eval case as stored in cases.jsonl. */
export interface EvalCase {
  id:    string
  tag:   string[]
  turns: string[]
  expect: {
    /** Ground-truth ids. A "full" fill case carries all four levels; a
     *  clarify/reject case carries only the known-good prefix (the failing level
     *  must stay empty). */
    slots?:   Partial<Record<Level, string>>
    /** Ground-truth id-level ticket fields (present on completing cases). */
    ticket?:  Partial<Record<Level, string>>
    /** For ambiguous/unknown cases: the agent must NOT fill the failing slot and
     *  must NOT emit a ticket. */
    outcome?: 'clarify' | 'reject'
  }
}

/** What the harness observed after feeding every turn of a case. */
export interface Observed {
  status:        'completed' | 'interrupted' | 'error'
  /** Final accumulated working memory across all turns (level → id, plus
   *  `description`/`ticket`). Read from `wm.mutated` events, never a checkpoint. */
  workingMemory: Record<string, unknown>
  /** Ticket parsed from the LIVE terminal-turn invoke output, or null. */
  ticket:        Record<string, unknown> | null
  turnCount:     number
  /** Set when the run threw. */
  error?:        string
}

/** Failure attribution buckets (#174 acceptance: wrong level / hallucinated id /
 *  premature emit / missing slot, plus a few finer-grained ones). */
export type FailureKind =
  | 'wrong_level'
  | 'hallucinated_id'
  | 'premature_emit'
  | 'missing_slot'
  | 'missing_ticket'
  | 'wrong_entity'
  | 'erroneous_fill'
  | 'wrong_ticket_field'
  | 'runtime_error'
  /** oneshot only (#185): the ticket emitted, but the stored description was not
   *  the clean fault substring — empty, or the entire one-shot turn verbatim (the
   *  level原话 leaked in). Never raised for multi-turn cases. */
  | 'description_contaminated'

export interface CaseResult {
  id:            string
  tag:           string[]
  passed:        boolean
  turnCount:     number
  slotFullMatch?:  boolean
  slotPerLevel?:   Record<Level, boolean>
  ticketMatch?:    boolean
  clarificationOk?: boolean
  correctionOk?:   boolean
  /** oneshot only (#185): true when the emitted ticket's description is the clean
   *  fault substring (non-empty and not the verbatim one-shot turn). Undefined for
   *  every non-oneshot case — the multi-turn "don't score description" rule stands. */
  descriptionCleanOk?: boolean
  failures:      FailureKind[]
}

/** Known ids per level — built from the resolver CSV by the harness; used to
 *  attribute a wrong slot value to hallucinated_id vs wrong_level vs wrong_entity. */
export type IdsByLevel = Record<Level, Set<string>>

const isFullSlotCase = (c: EvalCase): boolean =>
  !!c.expect.slots && LEVELS.every(l => c.expect.slots![l] != null)

const wmValue = (wm: Record<string, unknown>, level: Level): string => {
  const v = wm[level]
  return v == null ? '' : String(v)
}

/** Classify a single wrong (non-empty, non-matching) slot value. */
function classifyWrongSlot(value: string, level: Level, ids: IdsByLevel): FailureKind {
  if (ids[level].has(value)) return 'wrong_entity'
  for (const other of LEVELS) {
    if (other !== level && ids[other].has(value)) return 'wrong_level'
  }
  return 'hallucinated_id'
}

/** Score one case against the observed end state. */
export function scoreCase(c: EvalCase, obs: Observed, ids: IdsByLevel): CaseResult {
  const failures = new Set<FailureKind>()
  const isCorrection = c.tag.includes('correction')

  if (obs.status === 'error') {
    failures.add('runtime_error')
    return {
      id: c.id, tag: c.tag, passed: false, turnCount: obs.turnCount,
      failures: [...failures],
    }
  }

  // ── Clarify / reject cases ──────────────────────────────────────────────────
  if (c.expect.outcome) {
    // "Emitted a ticket" === a ticket object actually exists. NOT `status ===
    // 'completed'`: under #175's single autonomous llm state EVERY normal turn
    // completes (the run finishes each turn), so a correct clarify/reject turn is
    // also 'completed' — using status here false-flagged every clarify/reject case
    // as premature_emit. The ticket is read authoritatively from WM (run-eval), so
    // its presence is the right signal.
    const emittedTicket = obs.ticket != null
    if (emittedTicket) failures.add('premature_emit')

    const knownPrefix = c.expect.slots ?? {}
    let prefixOk = true
    let erroneous = false
    for (const level of LEVELS) {
      const expected = knownPrefix[level]
      const actual   = wmValue(obs.workingMemory, level)
      if (expected != null) {
        if (actual !== expected) prefixOk = false
      } else if (actual !== '') {
        // A level the agent should have left empty got filled → it guessed.
        erroneous = true
        failures.add(ids[level].has(actual) ? 'erroneous_fill' : 'hallucinated_id')
      }
    }

    const clarificationOk = !emittedTicket && !erroneous && prefixOk
    return {
      id: c.id, tag: c.tag, passed: clarificationOk, turnCount: obs.turnCount,
      clarificationOk, failures: [...failures],
    }
  }

  // ── Full fill / completion cases ────────────────────────────────────────────
  const slotPerLevel = {} as Record<Level, boolean>
  if (isFullSlotCase(c)) {
    for (const level of LEVELS) {
      const expected = c.expect.slots![level]!
      const actual   = wmValue(obs.workingMemory, level)
      const ok = actual === expected
      slotPerLevel[level] = ok
      if (!ok) {
        if (actual === '') failures.add('missing_slot')
        else failures.add(classifyWrongSlot(actual, level, ids))
      }
    }
  }
  const slotFullMatch = isFullSlotCase(c) && LEVELS.every(l => slotPerLevel[l])

  // Ticket field correctness — only meaningful when a ticket is expected.
  let ticketMatch: boolean | undefined
  if (c.expect.ticket) {
    if (obs.ticket == null) {
      ticketMatch = false
      // No ticket emitted though one was expected. If slots are incomplete the gap
      // is already attributed via missing_slot above; if every slot matched, the
      // flow stalled after slot-filling and never emitted — attribute that to
      // missing_ticket so the failed case never lands with an empty `failures`
      // column (#174).
      if (!slotFullMatch) {
        if (!failures.has('missing_slot')) failures.add('missing_slot')
      } else {
        failures.add('missing_ticket')
      }
    } else {
      ticketMatch = true
      for (const level of LEVELS) {
        const expected = c.expect.ticket[level]
        if (expected != null && String(obs.ticket[level] ?? '') !== expected) {
          ticketMatch = false
          failures.add('wrong_ticket_field')
        }
      }
      // ticketId is a deterministic function of the four ids.
      const expectedId =
        `TKT-${c.expect.ticket.site}-${c.expect.ticket.building}` +
        `-${c.expect.ticket.department}-${c.expect.ticket.assignee}`
      if (LEVELS.every(l => c.expect.ticket![l] != null) &&
          String(obs.ticket['ticketId'] ?? '') !== expectedId) {
        ticketMatch = false
        failures.add('wrong_ticket_field')
      }
      // NOTE: ticket.description is intentionally NOT scored (#174). It is free
      // text — paraphrasing or reformatting must not fail an otherwise-correct
      // ticket. Only deterministic id-level fields are exact-matched.
    }
  }

  const idsPassed = slotFullMatch && (c.expect.ticket ? ticketMatch === true : true)

  // oneshot weak assertion (#185): a turns=1 case packs every level AND the fault
  // into one utterance. With commit_description's optional param the model should
  // store the clean fault substring, not the whole turn (level原话). We check this
  // only for oneshot-tagged cases — the multi-turn "don't score description" rule
  // (above) is untouched.
  let descriptionCleanOk: boolean | undefined
  if (c.tag.includes('oneshot')) {
    const desc = String(obs.workingMemory['description'] ?? '').trim()
    const turn = (c.turns[0] ?? '').trim()
    descriptionCleanOk = desc !== '' && desc !== turn
    // Only attribute contamination when the run otherwise succeeded; else the
    // primary failure already explains the row and this would just be noise.
    if (idsPassed && !descriptionCleanOk) failures.add('description_contaminated')
  }

  const passed = idsPassed && (descriptionCleanOk === undefined || descriptionCleanOk)
  return {
    id: c.id, tag: c.tag, passed, turnCount: obs.turnCount,
    slotFullMatch, slotPerLevel,
    ...(ticketMatch !== undefined ? { ticketMatch } : {}),
    ...(isCorrection ? { correctionOk: slotFullMatch } : {}),
    ...(descriptionCleanOk !== undefined ? { descriptionCleanOk } : {}),
    failures: [...failures],
  }
}

// ─────────────────────────────── Aggregation ─────────────────────────────────

export interface Metrics {
  totalCases:   number
  passRate:     number
  slotFill:     { cases: number; fullMatchRate: number; perLevelRate: Record<Level, number> }
  ticket:       { cases: number; exactMatchRate: number }
  clarification:{ cases: number; accuracy: number }
  correction:   { cases: number; successRate: number }
  avgTurnCount: number
  failureDistribution: Record<string, number>
  byTag:        Record<string, { cases: number; passed: number; passRate: number }>
}

const rate = (n: number, d: number): number => (d === 0 ? 0 : n / d)

export function aggregate(cases: EvalCase[], results: CaseResult[]): Metrics {
  const slotCases   = results.filter(r => r.slotFullMatch !== undefined)
  const ticketCases = results.filter(r => r.ticketMatch !== undefined)
  const clarifyCases = results.filter(r => r.clarificationOk !== undefined)
  const correctionCases = results.filter(r => r.correctionOk !== undefined)

  const perLevelRate = {} as Record<Level, number>
  for (const level of LEVELS) {
    const ok = slotCases.filter(r => r.slotPerLevel?.[level]).length
    perLevelRate[level] = rate(ok, slotCases.length)
  }

  const failureDistribution: Record<string, number> = {}
  for (const r of results) {
    for (const f of r.failures) failureDistribution[f] = (failureDistribution[f] ?? 0) + 1
  }

  const byTag: Record<string, { cases: number; passed: number; passRate: number }> = {}
  for (const r of results) {
    for (const t of r.tag) {
      const e = byTag[t] ?? { cases: 0, passed: 0, passRate: 0 }
      e.cases  += 1
      e.passed += r.passed ? 1 : 0
      byTag[t] = e
    }
  }
  for (const t of Object.keys(byTag)) byTag[t]!.passRate = rate(byTag[t]!.passed, byTag[t]!.cases)

  return {
    totalCases: results.length,
    passRate:   rate(results.filter(r => r.passed).length, results.length),
    slotFill: {
      cases:         slotCases.length,
      fullMatchRate: rate(slotCases.filter(r => r.slotFullMatch).length, slotCases.length),
      perLevelRate,
    },
    ticket: {
      cases:          ticketCases.length,
      exactMatchRate: rate(ticketCases.filter(r => r.ticketMatch).length, ticketCases.length),
    },
    clarification: {
      cases:    clarifyCases.length,
      accuracy: rate(clarifyCases.filter(r => r.clarificationOk).length, clarifyCases.length),
    },
    correction: {
      cases:       correctionCases.length,
      successRate: rate(correctionCases.filter(r => r.correctionOk).length, correctionCases.length),
    },
    avgTurnCount: rate(results.reduce((s, r) => s + r.turnCount, 0), results.length),
    failureDistribution,
    byTag,
  }
}
