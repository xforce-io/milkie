// Unit tests for the deterministic eval scorer (#174). These run in plain CI —
// no live model, no credentials — and pin every scoring/attribution branch so the
// live harness can be trusted to grade correctly.

import {
  scoreCase, aggregate, LEVELS,
  type EvalCase, type Observed, type IdsByLevel,
} from '../scoring.js'

const IDS: IdsByLevel = {
  site:       new Set(['S01', 'S02']),
  building:   new Set(['B01', 'B02', 'B03']),
  department: new Set(['D02', 'D03', 'D07', 'D10']),
  assignee:   new Set(['E005', 'E006', 'E007', 'E008', 'E009', 'E012', 'E020']),
}

const DESC = '三楼会议室的投影仪无法开机'

const fullCase = (overrides: Partial<EvalCase> = {}): EvalCase => ({
  id:    'canonical-01',
  tag:   ['canonical'],
  turns: ['总部', '主楼', 'IT网络部', '王芳', DESC],
  expect: {
    slots:  { site: 'S01', building: 'B01', department: 'D03', assignee: 'E008' },
    ticket: { site: 'S01', building: 'B01', department: 'D03', assignee: 'E008' },
  },
  ...overrides,
})

const completedObs = (overrides: Partial<Observed> = {}): Observed => ({
  status:        'completed',
  workingMemory: { site: 'S01', building: 'B01', department: 'D03', assignee: 'E008', description: DESC },
  ticket: {
    ticketId: 'TKT-S01-B01-D03-E008',
    site: 'S01', building: 'B01', department: 'D03', assignee: 'E008', description: DESC,
  },
  turnCount:       5,
  ...overrides,
})

describe('scoreCase — full fill / completion', () => {
  it('passes a fully correct completion', () => {
    const r = scoreCase(fullCase(), completedObs(), IDS)
    expect(r.passed).toBe(true)
    expect(r.slotFullMatch).toBe(true)
    expect(r.ticketMatch).toBe(true)
    expect(r.failures).toEqual([])
    expect(LEVELS.every(l => r.slotPerLevel![l])).toBe(true)
  })

  it('attributes a same-level wrong id to wrong_entity', () => {
    const obs = completedObs({
      workingMemory: { site: 'S01', building: 'B01', department: 'D03', assignee: 'E009', description: DESC },
      ticket: { ticketId: 'TKT-S01-B01-D03-E009', site: 'S01', building: 'B01', department: 'D03', assignee: 'E009', description: DESC },
    })
    const r = scoreCase(fullCase(), obs, IDS)
    expect(r.passed).toBe(false)
    expect(r.slotPerLevel!.assignee).toBe(false)
    expect(r.failures).toContain('wrong_entity')
  })

  it('attributes an id from a different level to wrong_level', () => {
    const obs = completedObs({
      workingMemory: { site: 'S01', building: 'B01', department: 'D03', assignee: 'D07', description: DESC },
      ticket: null,
    })
    const r = scoreCase(fullCase(), obs, IDS)
    expect(r.failures).toContain('wrong_level')
  })

  it('attributes an unknown id to hallucinated_id', () => {
    const obs = completedObs({
      workingMemory: { site: 'S01', building: 'B01', department: 'D03', assignee: 'E999', description: DESC },
      ticket: null,
    })
    const r = scoreCase(fullCase(), obs, IDS)
    expect(r.failures).toContain('hallucinated_id')
  })

  it('attributes an empty expected slot to missing_slot', () => {
    const obs = completedObs({
      status: 'interrupted',
      workingMemory: { site: 'S01', building: 'B01', department: 'D03', description: DESC },
      ticket: null,
    })
    const r = scoreCase(fullCase(), obs, IDS)
    expect(r.failures).toContain('missing_slot')
    expect(r.ticketMatch).toBe(false)
  })

  it('attributes a complete-slots-but-no-ticket completion to missing_ticket (#174)', () => {
    // All four slots match ground truth, yet the flow never emitted a ticket.
    // This is a real end-to-end failure and MUST carry an attribution — the
    // failed row may never land with an empty `failures` column.
    const obs = completedObs({ ticket: null })
    const r = scoreCase(fullCase(), obs, IDS)
    expect(r.passed).toBe(false)
    expect(r.slotFullMatch).toBe(true)
    expect(r.ticketMatch).toBe(false)
    expect(r.failures.length).toBeGreaterThan(0)
    expect(r.failures).toContain('missing_ticket')
    // The slots genuinely matched, so this is NOT a slot-fill failure.
    expect(r.failures).not.toContain('missing_slot')
  })

  it('does NOT score the free-text description (#174): a paraphrase still passes', () => {
    const obs = completedObs({
      ticket: {
        ticketId: 'TKT-S01-B01-D03-E008',
        site: 'S01', building: 'B01', department: 'D03', assignee: 'E008',
        description: '投影仪故障',  // paraphrase ≠ the fed turn — must NOT fail
      },
    })
    const r = scoreCase(fullCase(), obs, IDS)
    expect(r.passed).toBe(true)
    expect(r.ticketMatch).toBe(true)
    expect(r.failures).toEqual([])
  })

  it('marks correctionOk for correction-tagged cases by full slot match', () => {
    const r = scoreCase(fullCase({ id: 'correction-01', tag: ['correction'] }), completedObs(), IDS)
    expect(r.correctionOk).toBe(true)
  })

  it('flags a runtime error', () => {
    const obs = completedObs({ status: 'error', error: 'boom', ticket: null })
    const r = scoreCase(fullCase(), obs, IDS)
    expect(r.passed).toBe(false)
    expect(r.failures).toEqual(['runtime_error'])
  })
})

describe('scoreCase — clarify / reject', () => {
  const ambiguous = (): EvalCase => ({
    id: 'ambiguous-01', tag: ['ambiguous'],
    turns: ['总部', '主楼', 'IT网络部', '张'],
    expect: { outcome: 'clarify', slots: { site: 'S01', building: 'B01', department: 'D03' } },
  })

  it('passes when the agent leaves the failing slot empty and emits no ticket', () => {
    const obs: Observed = {
      status: 'interrupted',
      workingMemory: { site: 'S01', building: 'B01', department: 'D03' },
      ticket: null, turnCount: 4,
    }
    const r = scoreCase(ambiguous(), obs, IDS)
    expect(r.passed).toBe(true)
    expect(r.clarificationOk).toBe(true)
    expect(r.failures).toEqual([])
  })

  it('fails with premature_emit when a ticket is produced anyway', () => {
    const obs: Observed = {
      status: 'completed',
      workingMemory: { site: 'S01', building: 'B01', department: 'D03', assignee: 'E007' },
      ticket: { ticketId: 'TKT-S01-B01-D03-E007' }, turnCount: 5,
    }
    const r = scoreCase(ambiguous(), obs, IDS)
    expect(r.passed).toBe(false)
    expect(r.failures).toContain('premature_emit')
  })

  it('fails with erroneous_fill when it guesses the ambiguous slot without emitting', () => {
    const obs: Observed = {
      status: 'interrupted',
      workingMemory: { site: 'S01', building: 'B01', department: 'D03', assignee: 'E007' },
      ticket: null, turnCount: 4,
    }
    const r = scoreCase(ambiguous(), obs, IDS)
    expect(r.clarificationOk).toBe(false)
    expect(r.failures).toContain('erroneous_fill')
  })

  it('passes a reject case that fills nothing', () => {
    const reject: EvalCase = {
      id: 'unknown-01', tag: ['unknown'],
      turns: ['火星基地'], expect: { outcome: 'reject', slots: {} },
    }
    const obs: Observed = { status: 'interrupted', workingMemory: {}, ticket: null, turnCount: 1 }
    const r = scoreCase(reject, obs, IDS)
    expect(r.passed).toBe(true)
  })
})

describe('aggregate', () => {
  it('rolls up rates, per-level slot accuracy, failure distribution, and by-tag', () => {
    const cases = [
      fullCase({ id: 'c1' }),
      fullCase({ id: 'c2' }),
      { id: 'a1', tag: ['ambiguous'], turns: ['x'], expect: { outcome: 'clarify' as const, slots: { site: 'S01' } } },
    ]
    const results = [
      scoreCase(cases[0]!, completedObs(), IDS),                                  // pass
      scoreCase(cases[1]!, completedObs({                                         // assignee wrong
        workingMemory: { site: 'S01', building: 'B01', department: 'D03', assignee: 'E009', description: DESC },
        ticket: null,
      }), IDS),
      scoreCase(cases[2]!, { status: 'interrupted', workingMemory: { site: 'S01' }, ticket: null, turnCount: 1 }, IDS), // clarify pass
    ]
    const m = aggregate(cases, results)

    expect(m.totalCases).toBe(3)
    expect(m.slotFill.cases).toBe(2)
    expect(m.slotFill.fullMatchRate).toBeCloseTo(0.5)
    expect(m.slotFill.perLevelRate.site).toBeCloseTo(1)
    expect(m.slotFill.perLevelRate.assignee).toBeCloseTo(0.5)
    expect(m.clarification.cases).toBe(1)
    expect(m.clarification.accuracy).toBeCloseTo(1)
    expect(m.failureDistribution['wrong_entity']).toBe(1)
    expect(m.byTag['canonical']!.cases).toBe(2)
    expect(m.byTag['canonical']!.passRate).toBeCloseTo(0.5)
    expect(m.byTag['ambiguous']!.passRate).toBeCloseTo(1)
  })
})
