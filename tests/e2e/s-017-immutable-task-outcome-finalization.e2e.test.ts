/**
 * s-017: Immutable task outcome finalization with evidence (#227).
 *
 * Hermetic crash-safe path:
 *   JsonlEventStore + FileTraceObjectStore + File finalization store
 *   + real Recording Tool path producing object.created with content hash.
 *
 * Asserts S1 (finalize once, durable reopen, observation isolation) and
 * S2 (concurrent/repeat finalize never overwrites; conflict diagnosable).
 */

import { promises as fs } from 'fs'
import os from 'os'
import path from 'path'
import { Milkie } from '../../src/runtime/Milkie'
import { MemoryStore } from '../../src/store/MemoryStore'
import { JsonlEventStore } from '../../src/trace/JsonlEventStore'
import { FileTraceObjectStore } from '../../src/trace/TraceObjectStore'
import { contentAddressForCanonicalBytes } from '../../src/trace/hash'
import { FileTaskOutcomeFinalizationStore } from '../../src/outcome/TaskOutcomeFinalizationStore'
import type { IModelGateway, ModelRequest, ModelResponse } from '../../src/types/model'
import type { AgentConfig } from '../../src/types/agent'
import type { ToolDefinition } from '../../src/types/tool'
import type { ObjectCreatedPayload } from '../../src/trace/types'
import type { FinalizeTaskOutcomeInput } from '../../src/types/outcome'
import { TaskOutcomeRunNotFoundError } from '../../src/types/outcome'

class SequentialGateway implements IModelGateway {
  public callCount = 0
  private idx = 0
  constructor(private readonly responses: ModelResponse[]) {}
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    this.callCount++
    const r = this.responses[Math.min(this.idx, this.responses.length - 1)]!
    this.idx++
    return r
  }
  async *stream(_req: ModelRequest): AsyncIterable<never> {
    yield* []
  }
}

const textOnly = (s: string): ModelResponse => ({
  content: [{ type: 'text', text: s }],
  toolCalls: [],
  finishReason: 'end_turn',
})

const toolCall = (name: string, input: Record<string, unknown>): ModelResponse => ({
  content: [{ type: 'tool_use', id: 'tc-1', name, input }],
  toolCalls: [{ id: 'tc-1', name, input }],
  finishReason: 'tool_use',
})

async function withTempDirs<T>(fn: (dirs: {
  events: string
  objects: string
  finals: string
}) => Promise<T>): Promise<T> {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'milkie-s017-'))
  const dirs = {
    events: path.join(root, 'events'),
    objects: path.join(root, 'objects'),
    finals: path.join(root, 'finals'),
  }
  await Promise.all([
    fs.mkdir(dirs.events, { recursive: true }),
    fs.mkdir(dirs.objects, { recursive: true }),
    fs.mkdir(dirs.finals, { recursive: true }),
  ])
  try {
    return await fn(dirs)
  } finally {
    await fs.rm(root, { recursive: true, force: true })
  }
}

function makePublishEvidenceTool(objectStore: FileTraceObjectStore): ToolDefinition {
  return {
    name: 'publish_evidence',
    description: 'Write canonical evidence bytes and mint a hashed lineage object',
    inputSchema: {
      type: 'object',
      properties: {
        text: { type: 'string' },
      },
      required: ['text'],
    },
    handler: async (input, ctx) => {
      const text = String(
        input && typeof input === 'object' && 'text' in input
          ? (input as { text?: unknown }).text ?? ''
          : '',
      )
      const hash = await objectStore.putCanonical(text)
      const obj = ctx.createObject?.({
        type: 'passage',
        hash,
        meta: { kind: 'evidence', text },
      })
      return {
        ok: true,
        objectId: obj?.objectId,
        hash,
      }
    },
  }
}

const agentId = 's017-finalization-agent'

function makeAgent(tools: string[]): AgentConfig {
  return {
    agentId,
    version: '0.0.0',
    systemPrompt: 'publish evidence then answer',
    fsm: {
      states: [
        { name: 'work', type: 'llm', terminal: true, tools },
      ],
    },
    model: { provider: 'stub', model: 'stub', adapter: 'stub' },
  } as AgentConfig
}

describe('s-017 immutable task outcome finalization', () => {
  it('S1: crash-safe finalize once with event+object evidence; reopen; observation isolation', async () => {
    await withTempDirs(async (dirs) => {
      const eventStore = new JsonlEventStore(dirs.events)
      const objectStore = new FileTraceObjectStore(dirs.objects)
      const finalStore = new FileTaskOutcomeFinalizationStore(dirs.finals)
      const evidenceText = 'canonical evidence body for s-017'
      const expectedHash = contentAddressForCanonicalBytes(evidenceText) as `sha256:${string}`

      const gateway = new SequentialGateway([
        toolCall('publish_evidence', { text: evidenceText }),
        textOnly('done'),
      ])

      const milkie = new Milkie({
        stateStore: new MemoryStore(),
        eventStore,
        traceObjectStore: objectStore,
        outcomeFinalizationStore: finalStore,
        gateway,
        tools: [makePublishEvidenceTool(objectStore)],
      })
      milkie.registerAgent(makeAgent(['publish_evidence']))

      const invokeResult = await milkie.invoke({
        agentId,
        goal: 'publish evidence and finish',
        input: 'go',
      })
      expect(invokeResult.status).toBe('completed')
      const runId = invokeResult.agentRunId
      const callsAfterInvoke = gateway.callCount

      const events = await eventStore.readByRunId(runId)
      const completed = events.filter(e => e.type === 'agent.run.completed')
      expect(completed).toHaveLength(1)
      const completedEventId = completed[0]!.id

      const objEvents = events.filter(e => e.type === 'object.created')
      expect(objEvents.length).toBeGreaterThanOrEqual(1)
      const objPayload = objEvents[0]!.payload as ObjectCreatedPayload
      expect(objPayload.hash).toBe(expectedHash)
      expect(objPayload.objectId).toBeTruthy()

      const bytes = await objectStore.getCanonical(expectedHash)
      expect(bytes).toBe(evidenceText)
      expect(contentAddressForCanonicalBytes(bytes!)).toBe(expectedHash)

      const input: FinalizeTaskOutcomeInput = {
        runId,
        expectedState: 'unfinalized',
        finalizationId: 's017-finalize-1',
        value: 'success',
        verifierClaim: { type: 'eval', id: 's017-suite' },
        evidence: [
          { kind: 'event', eventId: completedEventId },
          { kind: 'object', objectId: objPayload.objectId, hash: expectedHash },
        ],
        note: 's017 s1',
        scores: [{ name: 'quality', value: 1 }],
      }

      const finalized = await milkie.finalizeTaskOutcome(input)
      expect(finalized.status).toBe('finalized')
      if (finalized.status !== 'finalized') return

      expect(finalized.final.state).toBe('finalized')
      expect(finalized.final.value).toBe('success')
      expect(finalized.final.verifierClaim).toEqual({ type: 'eval', id: 's017-suite' })
      expect(finalized.final.evidence).toEqual([
        { kind: 'event', eventId: completedEventId },
        { kind: 'object', objectId: objPayload.objectId, hash: expectedHash },
      ].sort((a, b) => {
        // store returns normalized sorted order
        const key = (r: { kind: string; eventId?: string; objectId?: string; hash?: string }) =>
          r.kind === 'event' ? `event\0${r.eventId}` : `object\0${r.objectId}\0${r.hash}`
        return key(a) < key(b) ? -1 : key(a) > key(b) ? 1 : 0
      }))
      expect(finalized.final.intentHash).toMatch(/^sha256:[a-f0-9]{64}$/)
      expect(finalized.final.recordHash).toMatch(/^sha256:[a-f0-9]{64}$/)

      const queried = await milkie.getFinalTaskOutcome(runId)
      expect(queried!.recordHash).toBe(finalized.final.recordHash)

      // Rebuild all durable stores and re-verify.
      const eventStore2 = new JsonlEventStore(dirs.events)
      const objectStore2 = new FileTraceObjectStore(dirs.objects)
      const finalStore2 = new FileTaskOutcomeFinalizationStore(dirs.finals)
      const milkie2 = new Milkie({
        stateStore: new MemoryStore(),
        eventStore: eventStore2,
        traceObjectStore: objectStore2,
        outcomeFinalizationStore: finalStore2,
        gateway: new SequentialGateway([]),
      })

      const events2 = await eventStore2.readByRunId(runId)
      expect(events2.some(e => e.type === 'agent.run.completed')).toBe(true)
      const bytes2 = await objectStore2.getCanonical(expectedHash)
      expect(bytes2).toBe(evidenceText)
      expect(contentAddressForCanonicalBytes(bytes2!)).toBe(expectedHash)

      const final2 = await milkie2.getFinalTaskOutcome(runId)
      expect(final2!.recordHash).toBe(finalized.final.recordHash)
      expect(final2!.value).toBe('success')
      expect(final2!.verifierClaim).toEqual({ type: 'eval', id: 's017-suite' })

      // Observation LWW is independent of final.
      await milkie2.recordTaskOutcome({
        runId,
        value: 'failure',
        source: 'human',
        note: 'post-final observation',
      })
      const obs = await milkie2.getTaskOutcome(runId)
      expect(obs!.value).toBe('failure')
      const finalStill = await milkie2.getFinalTaskOutcome(runId)
      expect(finalStill!.value).toBe('success')
      expect(finalStill!.recordHash).toBe(finalized.final.recordHash)

      // No extra LLM on finalize/query paths.
      expect(gateway.callCount).toBe(callsAfterInvoke)
    })
  })

  it('S2: concurrent different finalizationIds → one winner; idempotent/conflict matrix', async () => {
    await withTempDirs(async (dirs) => {
      const eventStore = new JsonlEventStore(dirs.events)
      const objectStore = new FileTraceObjectStore(dirs.objects)
      const finalStore = new FileTaskOutcomeFinalizationStore(dirs.finals)
      const evidenceText = 's017 concurrent evidence'
      const expectedHash = contentAddressForCanonicalBytes(evidenceText) as `sha256:${string}`

      const gateway = new SequentialGateway([
        toolCall('publish_evidence', { text: evidenceText }),
        textOnly('done'),
      ])
      const milkie = new Milkie({
        stateStore: new MemoryStore(),
        eventStore,
        traceObjectStore: objectStore,
        outcomeFinalizationStore: finalStore,
        gateway,
        tools: [makePublishEvidenceTool(objectStore)],
      })
      milkie.registerAgent(makeAgent(['publish_evidence']))

      const invokeResult = await milkie.invoke({
        agentId,
        goal: 'publish evidence and finish',
        input: 'go',
      })
      const runId = invokeResult.agentRunId
      const events = await eventStore.readByRunId(runId)
      const completedEventId = events.find(e => e.type === 'agent.run.completed')!.id
      const objPayload = events.find(e => e.type === 'object.created')!.payload as ObjectCreatedPayload

      const mkInput = (
        finalizationId: string,
        value: 'success' | 'failure' | 'partial' | 'unknown',
      ): FinalizeTaskOutcomeInput => ({
        runId,
        expectedState: 'unfinalized',
        finalizationId,
        value,
        verifierClaim: { type: 'service', id: 'verifier-svc' },
        evidence: [
          { kind: 'event', eventId: completedEventId },
          { kind: 'object', objectId: objPayload.objectId, hash: expectedHash },
        ],
      })

      // Barrier-style concurrent different finalizationIds
      const attempts = await Promise.all([
        milkie.finalizeTaskOutcome(mkInput('fid-a', 'success')),
        milkie.finalizeTaskOutcome(mkInput('fid-b', 'failure')),
        milkie.finalizeTaskOutcome(mkInput('fid-c', 'partial')),
      ])

      const winners = attempts.filter(a => a.status === 'finalized')
      const conflicts = attempts.filter(a => a.status === 'conflict')
      expect(winners).toHaveLength(1)
      expect(conflicts).toHaveLength(2)

      const winnerHash = winners[0]!.status === 'finalized'
        ? winners[0]!.final.recordHash
        : ''
      expect(winnerHash).toMatch(/^sha256:/)

      for (const c of conflicts) {
        expect(c.status).toBe('conflict')
        if (c.status !== 'conflict') continue
        expect(c.conflict.kind).toBe('already_finalized')
        expect(c.existing.recordHash).toBe(winnerHash)
      }

      const winnerInput = mkInput(
        winners[0]!.status === 'finalized' ? winners[0]!.final.finalizationId : 'fid-a',
        winners[0]!.status === 'finalized' ? winners[0]!.final.value : 'success',
      )

      // same key + same intent → idempotent
      const idemp = await milkie.finalizeTaskOutcome(winnerInput)
      expect(idemp.status).toBe('idempotent')
      if (idemp.status === 'idempotent') {
        expect(idemp.final.recordHash).toBe(winnerHash)
      }

      // same key + different intent → idempotency_key_reused
      const reused = await milkie.finalizeTaskOutcome({
        ...winnerInput,
        value: winnerInput.value === 'success' ? 'failure' : 'success',
      })
      expect(reused.status).toBe('conflict')
      if (reused.status === 'conflict') {
        expect(reused.conflict.kind).toBe('idempotency_key_reused')
        expect(reused.existing.recordHash).toBe(winnerHash)
      }

      // same value but different finalizationId still already_finalized
      const sameValueDiffKey = await milkie.finalizeTaskOutcome(
        mkInput('brand-new-key', winnerInput.value),
      )
      expect(sameValueDiffKey.status).toBe('conflict')
      if (sameValueDiffKey.status === 'conflict') {
        expect(sameValueDiffKey.conflict.kind).toBe('already_finalized')
      }

      // Exactly one durable record
      const final = await milkie.getFinalTaskOutcome(runId)
      expect(final!.recordHash).toBe(winnerHash)

      const reopened = new FileTaskOutcomeFinalizationStore(dirs.finals)
      const disk = await reopened.get(runId)
      expect(disk!.recordHash).toBe(winnerHash)
    })
  })

  it('getFinal returns null for unfinalized known run; unknown run throws', async () => {
    await withTempDirs(async (dirs) => {
      const eventStore = new JsonlEventStore(dirs.events)
      const finalStore = new FileTaskOutcomeFinalizationStore(dirs.finals)
      const gateway = new SequentialGateway([textOnly('hi')])
      const milkie = new Milkie({
        stateStore: new MemoryStore(),
        eventStore,
        outcomeFinalizationStore: finalStore,
        gateway,
      })
      milkie.registerAgent(makeAgent([]))

      const result = await milkie.invoke({
        agentId,
        goal: 'say hi',
        input: 'hi',
      })
      await expect(milkie.getFinalTaskOutcome(result.agentRunId)).resolves.toBeNull()

      await expect(milkie.getFinalTaskOutcome('run-does-not-exist')).rejects.toBeInstanceOf(
        TaskOutcomeRunNotFoundError,
      )
    })
  })
})
