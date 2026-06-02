import { startServer, stopServer } from '../server'
import type { Server } from 'http'
import type { IModelGateway, ModelRequest, ModelResponse } from '../../../src/types/model'
import type { ObjectCreatedPayload, RelationCreatedPayload, ToolRespondedPayload } from '../../../src/trace/types'
import { JsonlEventStore } from '../../../src/trace/JsonlEventStore'
import { canonicalize, contentAddressForCanonicalBytes } from '../../../src/trace/hash'
import fs from 'fs'
import os from 'os'
import path from 'path'

class StubGateway implements IModelGateway {
  constructor(private readonly responses: ModelResponse[]) {}
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    const r = this.responses.shift()
    if (!r) throw new Error('StubGateway exhausted')
    return r
  }
  async *stream(_req: ModelRequest): AsyncIterable<never> { yield* [] }
}
const text = (s: string): ModelResponse => ({ content: [{ type: 'text', text: s }], toolCalls: [], finishReason: 'end_turn' })
const toolCall = (name: string, args: unknown, id = 'tc-' + name): ModelResponse => ({ content: [], finishReason: 'tool_use', toolCalls: [{ id, name, input: args }] })

async function postJson(url: string, body: unknown): Promise<{ status: number; body: string }> {
  const res = await fetch(url, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })
  return { status: res.status, body: await res.text() }
}

// Mirrors buildToolContext's content-address of a passage object.
function passageId(file: string, lineStart: number, lineEnd: number): string {
  return 'obj:' + contentAddressForCanonicalBytes(canonicalize({ type: 'passage', meta: { file, lineStart, lineEnd }, hash: null }))
}

describe('lineage emit end-to-end (#37/#38/新) — real runtime path', () => {
  let server: Server
  let baseUrl: string
  let exampleDir: string
  let runsDir: string

  const start = async (responses: ModelResponse[]) => {
    exampleDir = fs.mkdtempSync(path.join(os.tmpdir(), 'adq-lineage-'))
    runsDir = path.join(exampleDir, '.milkie', 'runs')
    fs.mkdirSync(runsDir, { recursive: true })
    server = await startServer({
      port: 0, exampleDir,
      gateway: new StubGateway(responses),
      agentFile: path.join(__dirname, '..', 'agents', 'sanguo-researcher.md'),
      corpusRoot: path.join(__dirname, '..', 'corpus'),
    })
    const addr = server.address()
    if (!addr || typeof addr === 'string') throw new Error('no address')
    baseUrl = `http://localhost:${addr.port}`
  }

  afterEach(async () => {
    await stopServer(server)
    fs.rmSync(exampleDir, { recursive: true, force: true })
  })

  it('read_file mints a passage object.created; cite mints a claim + cites relation', async () => {
    const file = 'chapter-49-赤壁借东风.txt'
    const objId = passageId(file, 4, 5)
    await start([
      toolCall('read_file', { relPath: file, lineStart: 4, lineEnd: 5 }, 'tc-1'),
      toolCall('cite', { claim: '诸葛亮定计破曹', objectId: objId }, 'tc-2'),
      text('赤壁之战的结局……'),
    ])

    const chat = await postJson(`${baseUrl}/chat`, { input: '赤壁之战的结局' })
    expect(chat.status).toBe(200)
    const runId = JSON.parse(chat.body).runId as string

    const events = await new JsonlEventStore(runsDir).readByRunId(runId)

    // #37: read_file produced a passage object, anchored to its tool.responded.
    const passage = events.find(e => e.type === 'object.created' && (e.payload as ObjectCreatedPayload).type === 'passage')
    expect(passage).toBeTruthy()
    const pp = passage!.payload as ObjectCreatedPayload
    expect(pp.objectId).toBe(objId)
    expect(pp.meta).toMatchObject({ file, lineStart: 4, lineEnd: 5 })
    const producer = events.find(e => e.id === pp.producerEventId)
    expect(producer?.type).toBe('tool.responded')
    expect((producer!.payload as ToolRespondedPayload).artifactRefs).toContain(objId)

    // 新/#38: cite produced a claim object + a cites relation claim → passage.
    const claim = events.find(e => e.type === 'object.created' && (e.payload as ObjectCreatedPayload).type === 'claim')
    expect(claim).toBeTruthy()
    const rel = events.find(e => e.type === 'relation.created')
    expect(rel).toBeTruthy()
    const rp = rel!.payload as RelationCreatedPayload
    expect(rp.type).toBe('cites')
    expect(rp.toObjectId).toBe(objId)
    expect(rp.fromObjectId).toBe((claim!.payload as ObjectCreatedPayload).objectId)

    // P1: a real cited objectId passes the registry check → cite reports ok:true.
    const okCite = events.find(e => e.type === 'tool.responded' && (e.payload as ToolRespondedPayload).toolName === 'cite')!
    expect((okCite.payload as ToolRespondedPayload).output).toMatchObject({ ok: true })
  })

  it('P1 fail-fast: cite with a fabricated objectId → ok:false, NO relation/claim recorded', async () => {
    await start([
      toolCall('cite', { claim: '编造的引用', objectId: 'obj:fabricated-not-real' }, 'tc-1'),
      text('answer'),
    ])
    const chat = await postJson(`${baseUrl}/chat`, { input: 'x' })
    const runId = JSON.parse(chat.body).runId as string
    const events = await new JsonlEventStore(runsDir).readByRunId(runId)

    // Fail-fast bailed before declaring anything: no cites edge, no claim object.
    expect(events.some(e => e.type === 'relation.created')).toBe(false)
    expect(events.some(e => e.type === 'object.created')).toBe(false)
    // The structured error rides back on cite's tool.responded so the model can self-correct.
    const citeResp = events.find(e => e.type === 'tool.responded' && (e.payload as ToolRespondedPayload).toolName === 'cite')!
    const out = (citeResp.payload as ToolRespondedPayload).output as { ok: boolean; error?: string }
    expect(out.ok).toBe(false)
    expect(out.error).toMatch(/不存在|objectId/)
  })

  it('P1: the registry is run-level — an id minted in one call resolves in a later cite', async () => {
    const file = 'chapter-01-桃园三结义.txt'
    const idA = passageId(file, 1, 5)
    const idB = passageId(file, 10, 15)
    await start([
      toolCall('read_file', { relPath: file, lineStart: 1, lineEnd: 5 }, 'tc-1'),
      toolCall('read_file', { relPath: file, lineStart: 10, lineEnd: 15 }, 'tc-2'),
      toolCall('cite', { claim: 'A 段', objectId: idA }, 'tc-3'),  // minted 2 calls earlier
      toolCall('cite', { claim: 'B 段', objectId: idB }, 'tc-4'),
      text('done'),
    ])
    const chat = await postJson(`${baseUrl}/chat`, { input: 'x' })
    const runId = JSON.parse(chat.body).runId as string
    const events = await new JsonlEventStore(runsDir).readByRunId(runId)

    const cites = events.filter(e => e.type === 'relation.created')
    expect(cites.length).toBe(2)
    expect(cites.map(e => (e.payload as RelationCreatedPayload).toObjectId).sort()).toEqual([idA, idB].sort())
    const citeOks = events
      .filter(e => e.type === 'tool.responded' && (e.payload as ToolRespondedPayload).toolName === 'cite')
      .map(e => (e.payload as ToolRespondedPayload).output as { ok: boolean })
    expect(citeOks.length).toBe(2)
    expect(citeOks.every(o => o.ok)).toBe(true)
  })

  it('P1: a run mixing cite ok + fail-fast replays deterministically (registry is record-only)', async () => {
    const file = 'chapter-49-赤壁借东风.txt'
    const okId = passageId(file, 4, 5)
    await start([
      toolCall('read_file', { relPath: file, lineStart: 4, lineEnd: 5 }, 'tc-1'),
      toolCall('cite', { claim: '真引用', objectId: okId }, 'tc-2'),       // ok
      toolCall('cite', { claim: '假引用', objectId: 'obj:fake' }, 'tc-3'), // fail-fast
      text('赤壁之战的结局……'),
    ])
    const chat = await postJson(`${baseUrl}/chat`, { input: '赤壁' })
    const runId = JSON.parse(chat.body).runId as string

    // Replay re-runs from cache (zero live calls); handlers don't run, so the
    // mintedObjects registry stays empty on replay — yet the cached cite outputs
    // (ok:true / ok:false) are served verbatim, so the run reproduces identically.
    const replay = await postJson(`${baseUrl}/run/${runId}/replay`, {})
    expect(replay.status).toBe(200)
    const body = JSON.parse(replay.body)
    expect(body.status).toBe('deterministic')
    expect(body.matchesOriginal).toBe(true)
  })
})
