// src/__tests__/traceToolsServeRegistration.test.ts
// #196: self-explain read-trace tools must be registered wherever an eventStore is
// present — not only via the opt-in loadStandardAgents path. This covers the serve
// --agent path, which #189's tests never exercised.
//
// #196 follow-up (P1): the generic registration must not let an agent read an ARBITRARY
// runId — on a shared serve instance that would leak another session's I/O. #200 C
// refines this: the runId axis is present but GATED to ctx.deliveredRunIds (runs
// delivered to this session via a projection); a non-delivered runId is ignored
// (get_execution/get_lineage) or refused (get_run_io). Reading any runId remains the
// diagnoser's privilege via loadStandardAgents().
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import type { ToolDefinition } from '../types/tool'

const toolNames = (m: Milkie): string[] =>
  (m as unknown as { extraTools: Array<{ name: string }> }).extraTools.map(t => t.name)

describe('#196 read-trace tools registered whenever an eventStore is present', () => {
  it('a Milkie built with an eventStore exposes self-view tools without loadStandardAgents', () => {
    const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore: new MemoryEventStore() })
    // No loadStandardAgents() call — mirrors the serve --agent path.
    expect(toolNames(milkie)).toEqual(expect.arrayContaining(['get_execution', 'get_lineage']))
  })

  it('the generic registration exposes get_run_io but gated to delivered runIds (#200 C)', async () => {
    const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore: new MemoryEventStore() })
    expect(toolNames(milkie)).toContain('get_run_io')
    // Without a delivered runId in ctx, get_run_io refuses — no cross-session read.
    const tool = (milkie as unknown as { extraTools: ToolDefinition[] }).extraTools
      .find(t => t.name === 'get_run_io')!
    const res = await tool.handler({ runId: 'someone-elses-run' }, {} as never) as { error?: string }
    expect(res.error).toBeDefined()
  })

  it('the generic get_execution exposes runId but gates it to delivered runs (#200 C)', () => {
    const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore: new MemoryEventStore() })
    const tool = (milkie as unknown as { extraTools: ToolDefinition[] }).extraTools
      .find(t => t.name === 'get_execution')!
    // runId axis is present (delivered runs are reachable); the gating is behavioural,
    // covered by traceTools.deliveredDeref.test.ts (non-delivered runId → self window).
    expect((tool.inputSchema as { properties: Record<string, unknown> }).properties.runId).toBeDefined()
  })

  it('a Milkie built without an eventStore registers no read-trace tools', () => {
    const milkie = new Milkie({ stateStore: new MemoryStore() })
    expect(toolNames(milkie)).not.toContain('get_execution')
  })
})

describe('constructor does not mutate the caller-supplied tools array (P1)', () => {
  it('leaves the passed array untouched', () => {
    const tools: ToolDefinition[] = []
    new Milkie({ stateStore: new MemoryStore(), eventStore: new MemoryEventStore(), tools })
    expect(tools).toHaveLength(0)
  })

  it('reusing one array across instances does not leak trace tools into a store-less instance', () => {
    const tools: ToolDefinition[] = []
    new Milkie({ stateStore: new MemoryStore(), eventStore: new MemoryEventStore(), tools })
    const b = new Milkie({ stateStore: new MemoryStore(), tools }) // no eventStore
    expect(toolNames(b)).not.toContain('get_execution')
  })
})
