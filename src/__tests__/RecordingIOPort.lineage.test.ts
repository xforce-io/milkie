import { RecordingIOPort } from '../trace/RecordingIOPort'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import type { IIOPort } from '../runtime/IOPort'
import type { ModelRequest, ModelResponse } from '../types/model'
import type { LineageBuffer, ObjectCreatedPayload, ObjectType, PendingObject, RelationCreatedPayload, ToolRespondedPayload } from '../trace/types'

class ExecInnerPort implements IIOPort {
  private nextClock = 1000
  private nextUuid  = 1
  async invokeLLM(_req: ModelRequest): Promise<ModelResponse> {
    return { content: [], toolCalls: [], finishReason: 'end_turn' }
  }
  async invokeTool(_n: string, _i: unknown, execute: () => Promise<unknown>): Promise<unknown> {
    return execute()
  }
  now():  number { return this.nextClock++ }
  uuid(): string { return `uuid-${this.nextUuid++}` }
}

describe('RecordingIOPort — lineage flush (#37/#38)', () => {
  it('emits object.created after tool.responded with producerEventId resolving to it', async () => {
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(new ExecInnerPort(), store, 'r1')
    // The handler "declares" a passage object mid-execute (what ctx.createObject does).
    const lineage: LineageBuffer = { objects: [], relations: [] }
    await port.invokeTool('read_file', { relPath: 'ch.txt' }, async () => {
      lineage.objects.push({ objectId: 'obj:abc', type: 'passage', meta: { file: 'ch.txt', lineStart: 4, lineEnd: 5 } })
      return { content: '诗两行', objectId: 'obj:abc' }
    }, { lineage })

    const events = await store.readByRunId('r1')
    const respIdx = events.findIndex(e => e.type === 'tool.responded')
    const objIdx  = events.findIndex(e => e.type === 'object.created')
    expect(respIdx).toBeGreaterThanOrEqual(0)
    // object.created comes AFTER tool.responded.
    expect(objIdx).toBeGreaterThan(respIdx)

    const resp = events[respIdx]!.payload as ToolRespondedPayload
    const obj  = events[objIdx]!.payload as ObjectCreatedPayload
    // producerEventId points at the tool.responded event id.
    expect(obj.producerEventId).toBe(events[respIdx]!.id)
    expect(obj).toMatchObject({ objectId: 'obj:abc', type: 'passage', meta: { file: 'ch.txt', lineStart: 4, lineEnd: 5 } })
    // tool.responded lists the produced object via artifactRefs.
    expect(resp.artifactRefs).toEqual(['obj:abc'])
  })

  it('emits relation.created with from/to and causedByEventId at the tool.responded', async () => {
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(new ExecInnerPort(), store, 'r1')
    const lineage: LineageBuffer = { objects: [], relations: [] }
    await port.invokeTool('cite', { objectId: 'obj:p' }, async () => {
      lineage.relations.push({ relationId: 'rel:1', type: 'cites', fromObjectId: 'obj:claim', toObjectId: 'obj:p' })
      return { ok: true }
    }, { lineage })

    const events = await store.readByRunId('r1')
    const resp = events.find(e => e.type === 'tool.responded')!
    const rel  = events.find(e => e.type === 'relation.created')!
    const p = rel.payload as RelationCreatedPayload
    expect(p).toMatchObject({ type: 'cites', fromObjectId: 'obj:claim', toObjectId: 'obj:p' })
    expect(p.causedByEventId).toBe(resp.id)
  })

  it('records nothing when no lineage buffer is passed (back-compat)', async () => {
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(new ExecInnerPort(), store, 'r1')
    await port.invokeTool('t', {}, async () => 'x')
    const events = await store.readByRunId('r1')
    expect(events.some(e => e.type === 'object.created')).toBe(false)
    const resp = events.find(e => e.type === 'tool.responded')!.payload as ToolRespondedPayload
    expect(resp.artifactRefs).toBeUndefined()
  })

  it('does not emit object.created on the error branch (no output, no objects)', async () => {
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(new ExecInnerPort(), store, 'r1')
    const lineage: LineageBuffer = { objects: [], relations: [] }
    await expect(
      port.invokeTool('read_file', {}, async () => {
        // even if something got pushed before throwing, the error branch skips flush
        lineage.objects.push({ objectId: 'obj:x', type: 'passage' })
        throw new Error('boom')
      }, { lineage })
    ).rejects.toThrow('boom')
    const events = await store.readByRunId('r1')
    expect(events.some(e => e.type === 'object.created')).toBe(false)
  })

  it('lazy-promote: producerEventId anchors to the retrieval turn (turn A), not the cite turn (turn B)', async () => {
    // Simulates AgentRuntime's mintedObjects map and the backfillProducerEventId callback.
    type MintedEntry = { type: ObjectType; meta?: Record<string, unknown>; producerEventId?: string; promoted: boolean }
    const mintedObjects = new Map<string, MintedEntry>()

    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(new ExecInnerPort(), store, 'r1')

    // Turn A: tool that calls registerObject (lazy, no push to lineage.objects yet).
    const lineageA: LineageBuffer = {
      objects: [],
      relations: [],
      registeredObjectIds: [],
      backfillProducerEventId: (respEventId) => {
        for (const id of lineageA.registeredObjectIds ?? []) {
          const o = mintedObjects.get(id)
          if (o && !o.producerEventId) o.producerEventId = respEventId
        }
      },
    }
    await port.invokeTool('run_command', { cmd: 'echo hi' }, async () => {
      // Simulate ctx.registerObject
      mintedObjects.set('obj:stdout', { type: 'passage', promoted: false })
      lineageA.registeredObjectIds!.push('obj:stdout')
      return { output: 'hi' }
    }, { lineage: lineageA })

    // Turn B: tool that calls promoteObject (cite).
    const lineageB: LineageBuffer = { objects: [], relations: [] }
    await port.invokeTool('cite', { objectId: 'obj:stdout' }, async () => {
      // Simulate ctx.promoteObject
      const o = mintedObjects.get('obj:stdout')
      if (o && !o.promoted) {
        const pending: PendingObject = { objectId: 'obj:stdout', type: o.type }
        if (o.producerEventId) pending.producerEventId = o.producerEventId
        lineageB.objects.push(pending)
        o.promoted = true
      }
      return { ok: true }
    }, { lineage: lineageB })

    const events = await store.readByRunId('r1')
    const turnAResp = events.find(
      e => e.type === 'tool.responded' && (e.payload as ToolRespondedPayload).toolName === 'run_command',
    )!
    const turnBResp = events.find(
      e => e.type === 'tool.responded' && (e.payload as ToolRespondedPayload).toolName === 'cite',
    )!
    const objCreated = events.find(e => e.type === 'object.created')!

    expect(turnAResp).toBeDefined()
    expect(turnBResp).toBeDefined()
    expect(objCreated).toBeDefined()
    // object.created must use turn A's tool.responded id, not turn B's.
    expect((objCreated.payload as ObjectCreatedPayload).producerEventId).toBe(turnAResp.id)
    expect((objCreated.payload as ObjectCreatedPayload).producerEventId).not.toBe(turnBResp.id)
  })
})
