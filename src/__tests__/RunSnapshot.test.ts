import { extractRunSnapshot } from '../trace/RunSnapshot'
import { ReplayError } from '../trace/ReplayError'
import type { Event } from '../trace/types'

const startedEvent = (over: Partial<Event> = {}): Event => ({
  id: 'e1', runId: 'r1', type: 'agent.run.started', actor: 'runtime', timestamp: 1,
  payload: { agentId: 'a1', goal: 'g', input: 'i', contextId: 'c1' },
  ...over,
})

describe('extractRunSnapshot', () => {
  it('extracts identity from agent.run.started', () => {
    const snap = extractRunSnapshot([startedEvent()])
    expect(snap).toEqual({
      agentId: 'a1', goal: 'g', input: 'i', contextId: 'c1', parentId: undefined,
      terminalStatus: undefined,
    })
  })

  it('reads terminalStatus when agent.run.completed present', () => {
    const completed: Event = {
      id: 'e2', runId: 'r1', type: 'agent.run.completed', actor: 'runtime', timestamp: 2,
      payload: { status: 'completed', lastTextOutput: 'done' },
    }
    const snap = extractRunSnapshot([startedEvent(), completed])
    expect(snap.terminalStatus).toBe('completed')
    expect(snap.lastTextOutput).toBe('done')
  })

  it('reads structured terminalError for RUN_CANCELLED completed payload', () => {
    const completed: Event = {
      id: 'e2', runId: 'r1', type: 'agent.run.completed', actor: 'runtime', timestamp: 2,
      payload: {
        status: 'error',
        error: {
          code: 'RUN_CANCELLED',
          message: 'Run cancelled by caller',
          phase: 'agent_loop',
          retryable: true,
        },
      },
    }
    const snap = extractRunSnapshot([startedEvent(), completed])
    expect(snap.terminalStatus).toBe('error')
    expect(snap.terminalError).toMatchObject({ code: 'RUN_CANCELLED' })
  })

  it('throws ReplayError on empty events', () => {
    expect(() => extractRunSnapshot([])).toThrow(ReplayError)
    expect(() => extractRunSnapshot([])).toThrow(/no events/)
  })

  it('throws ReplayError when no agent.run.started present', () => {
    const onlyResp: Event = {
      id: 'e1', runId: 'r1', type: 'llm.responded', actor: 'runtime', timestamp: 1,
      payload: { response: { content: [], toolCalls: [], finishReason: 'end_turn' }, requestHash: 'h' },
    }
    expect(() => extractRunSnapshot([onlyResp])).toThrow(/no lifecycle start/)
  })
})
