import { RecordingIOPort } from '../trace/RecordingIOPort'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { MemoryTraceObjectStore } from '../trace/TraceObjectStore'
import { canonicalize, contentAddressForCanonicalBytes } from '../trace/hash'
import type { IIOPort } from '../runtime/IOPort'
import type { ModelRequest, ModelResponse } from '../types/model'
import type { ToolRespondedPayload } from '../trace/types'

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

async function respondedPayload(store: MemoryEventStore): Promise<ToolRespondedPayload> {
  const events = await store.readByRunId('r1')
  return events.find(e => e.type === 'tool.responded')!.payload as ToolRespondedPayload
}

describe('RecordingIOPort — tool output metadata (#25)', () => {
  it('adds outputHash and outputBytes derived from canonical output', async () => {
    const inner = new ExecInnerPort()
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(inner, store, 'r1')

    const output = { lines: ['a', 'b'], n: 2 }
    await port.invokeTool('read_file', { path: 'x' }, async () => output)

    const p = await respondedPayload(store)
    const canonical = canonicalize(output)
    expect(p.outputHash).toBe(contentAddressForCanonicalBytes(canonical))
    expect(p.outputBytes).toBe(Buffer.byteLength(canonical, 'utf8'))
    expect(p.output).toEqual(output)
  })

  it('produces identical outputHash for identical output across runs', async () => {
    const run = async () => {
      const store = new MemoryEventStore()
      await new RecordingIOPort(new ExecInnerPort(), store, 'r1')
        .invokeTool('t', { a: 1 }, async () => ({ z: 1, a: [2, 3] }))
      return (await respondedPayload(store)).outputHash
    }
    expect(await run()).toBe(await run())
  })

  it('still records hash/bytes when no traceObjectStore is provided', async () => {
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(new ExecInnerPort(), store, 'r1')
    await port.invokeTool('t', {}, async () => 'hello')
    const p = await respondedPayload(store)
    expect(typeof p.outputHash).toBe('string')
    expect(p.outputBytes).toBe(Buffer.byteLength(canonicalize('hello'), 'utf8'))
  })

  it('dedupes large output by hash in the trace object store', async () => {
    const objectStore = new MemoryTraceObjectStore()
    let putCount = 0
    const counting = {
      putCanonical: async (b: string) => { putCount++; return objectStore.putCanonical(b) },
      getCanonical: (h: string) => objectStore.getCanonical(h),
      has:          (h: string) => objectStore.has(h),
    }
    const store = new MemoryEventStore()
    const big = 'x'.repeat(100_000)
    const out = { chapter: big }

    for (let i = 0; i < 3; i++) {
      await new RecordingIOPort(new ExecInnerPort(), store, `r${i}`, undefined, counting)
        .invokeTool('read_file', { path: 'ch1', n: i }, async () => out)
    }

    const hash = contentAddressForCanonicalBytes(canonicalize(out))
    expect(await objectStore.has(hash)).toBe(true)
    expect(putCount).toBe(3)
    expect(await objectStore.getCanonical(hash)).toBe(canonicalize(out))
  })
})
