import { RecordingIOPort } from '../trace/RecordingIOPort'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import type { IIOPort } from '../runtime/IOPort'
import type { ModelRequest, ModelResponse } from '../types/model'
import type { ToolRequestedPayload, ToolRespondedPayload } from '../trace/types'

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

async function payloads(store: MemoryEventStore) {
  const events = await store.readByRunId('r1')
  const req = events.find(e => e.type === 'tool.requested')!.payload as ToolRequestedPayload
  const res = events.find(e => e.type === 'tool.responded')!.payload as ToolRespondedPayload
  return { req, res }
}

describe('RecordingIOPort — readable tool payload (#81)', () => {
  it('stamps toolCallId onto both tool.requested and tool.responded so they pair', async () => {
    const store = new MemoryEventStore()
    const port: IIOPort = new RecordingIOPort(new ExecInnerPort(), store, 'r1')

    await port.invokeTool('search', { q: 'milkie' }, async () => ({ hits: 3 }), {
      toolCallId: 'call-abc',
    })

    const { req, res } = await payloads(store)
    expect(req.toolCallId).toBe('call-abc')
    expect(res.toolCallId).toBe('call-abc')
    // pairing: same toolCallId links the two events for an external consumer
    expect(res.toolCallId).toBe(req.toolCallId)
  })

  it('marks status="ok" on the success branch and keeps name/input/output readable', async () => {
    const store = new MemoryEventStore()
    const port: IIOPort = new RecordingIOPort(new ExecInnerPort(), store, 'r1')

    await port.invokeTool('search', { q: 'milkie' }, async () => ({ hits: 3 }), {
      toolCallId: 'call-ok',
    })

    const { req, res } = await payloads(store)
    expect(req.toolName).toBe('search')
    expect(req.input).toEqual({ q: 'milkie' })
    expect(res.status).toBe('ok')
    expect(res.output).toEqual({ hits: 3 })
    expect(res.error).toBeUndefined()
  })

  it('marks status="error" and keeps toolCallId on the failure branch', async () => {
    const store = new MemoryEventStore()
    const port: IIOPort = new RecordingIOPort(new ExecInnerPort(), store, 'r1')

    await expect(
      port.invokeTool('search', { q: 'x' }, async () => { throw new Error('boom') }, {
        toolCallId: 'call-err',
      }),
    ).rejects.toThrow('boom')

    const { req, res } = await payloads(store)
    expect(req.toolCallId).toBe('call-err')
    expect(res.toolCallId).toBe('call-err')
    expect(res.status).toBe('error')
    expect(res.error?.message).toBe('boom')
    expect(res.output).toBeUndefined()
  })

  it('omits toolCallId when none is supplied (back-compat), still sets status', async () => {
    const store = new MemoryEventStore()
    const port: IIOPort = new RecordingIOPort(new ExecInnerPort(), store, 'r1')

    await port.invokeTool('t', {}, async () => 'hi')

    const { req, res } = await payloads(store)
    expect(req.toolCallId).toBeUndefined()
    expect(res.toolCallId).toBeUndefined()
    expect(res.status).toBe('ok')
  })
})
