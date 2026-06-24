// src/__tests__/traceToolsServeRegistration.test.ts
// #196: read-trace tools (get_execution/get_lineage/get_run_io) must be registered
// wherever an eventStore is present — not only via the dead-code loadStandardAgents
// path. This covers the serve --agent path, which #189's tests never exercised.
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'

const toolNames = (m: Milkie): string[] =>
  (m as unknown as { extraTools: Array<{ name: string }> }).extraTools.map(t => t.name)

describe('#196 read-trace tools registered whenever an eventStore is present', () => {
  it('a Milkie built with an eventStore exposes read-trace tools without loadStandardAgents', () => {
    const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore: new MemoryEventStore() })
    // No loadStandardAgents() call — mirrors the serve --agent path.
    expect(toolNames(milkie)).toEqual(
      expect.arrayContaining(['get_execution', 'get_lineage', 'get_run_io']))
  })

  it('a Milkie built without an eventStore registers no read-trace tools', () => {
    const milkie = new Milkie({ stateStore: new MemoryStore() })
    expect(toolNames(milkie)).not.toContain('get_execution')
  })
})
