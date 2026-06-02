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
  })

  it('a fabricated objectId still records a cites edge, but it dangles (no matching object.created)', async () => {
    await start([
      toolCall('cite', { claim: '编造的引用', objectId: 'obj:fabricated-not-real' }, 'tc-1'),
      text('answer'),
    ])
    const chat = await postJson(`${baseUrl}/chat`, { input: 'x' })
    const runId = JSON.parse(chat.body).runId as string
    const events = await new JsonlEventStore(runsDir).readByRunId(runId)

    const rel = events.find(e => e.type === 'relation.created')!
    const toId = (rel.payload as RelationCreatedPayload).toObjectId
    expect(toId).toBe('obj:fabricated-not-real')
    // Unforgeable: no object.created mints that id, so the edge resolves to nothing.
    const resolved = events.some(e => e.type === 'object.created' && (e.payload as ObjectCreatedPayload).objectId === toId)
    expect(resolved).toBe(false)
  })
})
