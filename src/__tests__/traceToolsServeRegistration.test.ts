// src/__tests__/traceToolsServeRegistration.test.ts
// #196: self-explain read-trace tools must be registered wherever an eventStore is
// present — not only via the opt-in loadStandardAgents path. This covers the serve
// --agent path, which #189's tests never exercised.
//
// #196 follow-up (P1): the generic registration must be SELF-ONLY — get_execution /
// get_lineage without a runId axis, and NO get_run_io. Reading an arbitrary runId is
// the diagnoser's privilege (loadStandardAgents). On a shared serve instance the full
// version would let any session read another's I/O.
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

  it('the generic registration does NOT expose get_run_io (diagnoser-only privilege)', () => {
    const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore: new MemoryEventStore() })
    expect(toolNames(milkie)).not.toContain('get_run_io')
  })

  it('the generic get_execution has no runId axis (cannot read a foreign run)', () => {
    const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore: new MemoryEventStore() })
    const tool = (milkie as unknown as { extraTools: ToolDefinition[] }).extraTools
      .find(t => t.name === 'get_execution')!
    expect((tool.inputSchema as { properties: Record<string, unknown> }).properties.runId).toBeUndefined()
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
