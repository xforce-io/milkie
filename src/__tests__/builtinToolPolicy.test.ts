// #235: built-in tool allowlist — schema + dispatch share one registry boundary.
import { AgentRuntime } from '../runtime/AgentRuntime'
import { DefaultIOPort } from '../runtime/IOPort'
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { InMemoryRecorder } from '../trajectory/InMemoryRecorder'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import type { AgentConfig } from '../types/agent'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { ToolDefinition } from '../types/tool'
import {
  BUILTIN_TOOL_NAMES,
  resolveEffectiveBuiltinTools,
  validateBuiltinToolPolicy,
} from '../tools/builtinTools'
import type { AgentRunStartedPayload } from '../trace/types'
import fs from 'fs'
import os from 'os'
import path from 'path'

function makeConfig(overrides: Partial<AgentConfig> = {}): AgentConfig {
  return {
    agentId: 'policy-agent',
    version: '1.0.0',
    systemPrompt: 'sys',
    fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 5 }] },
    model: { provider: 'test', model: 'test', adapter: 'test' },
    ...overrides,
  }
}

class CapturingGateway implements IModelGateway {
  requests: ModelRequest[] = []
  private seq: ModelResponse[]
  private i = 0
  constructor(seq: ModelResponse[]) { this.seq = seq }
  async complete(req: ModelRequest): Promise<ModelResponse> {
    this.requests.push(req)
    const r = this.seq[this.i++]
    if (!r) throw new Error('no more responses')
    return r
  }
  async *stream(): AsyncIterable<never> { yield* [] }
}

function text(t: string): ModelResponse {
  return { content: [{ type: 'text', text: t }], toolCalls: [], finishReason: 'end_turn' }
}
function toolCall(id: string, name: string, input: unknown = {}): ModelResponse {
  return {
    content: [{ type: 'tool_use', id, name, input }],
    toolCalls: [{ id, name, input }],
    finishReason: 'tool_use',
  }
}

describe('#235 builtin tool policy', () => {
  describe('resolveEffectiveBuiltinTools', () => {
    it('omitted policy → all current built-ins (compat)', () => {
      expect(resolveEffectiveBuiltinTools({})).toEqual([...BUILTIN_TOOL_NAMES])
    })

    it('empty allow → zero built-ins', () => {
      expect(resolveEffectiveBuiltinTools({ builtinTools: { allow: [] } })).toEqual([])
    })

    it('rejects unknown and duplicate names before start', () => {
      expect(() => validateBuiltinToolPolicy({ allow: ['run_command', 'nope'] as string[] }))
        .toThrow(/Unknown built-in tool name/)
      expect(() => validateBuiltinToolPolicy({ allow: ['think', 'think'] }))
        .toThrow(/Duplicate built-in tool name/)
    })

    it('child intersects parent; omitted child inherits parent', () => {
      const parent = resolveEffectiveBuiltinTools({ builtinTools: { allow: ['think', 'run_command'] } })
      expect(resolveEffectiveBuiltinTools({}, parent)).toEqual(['think', 'run_command'])
      expect(resolveEffectiveBuiltinTools({ builtinTools: { allow: ['think', 'cite'] } }, parent))
        .toEqual(['think'])
    })
  })

  it('S1: restricted allowlist hides unauthorized built-ins from schema and dispatch; custom tools remain', async () => {
    const markerDir = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-bt-s1-'))
    const markerFile = path.join(markerDir, 'run-command-should-not-exist')
    let customRan = false
    const custom: ToolDefinition = {
      name: 'custom_echo',
      description: 'custom',
      inputSchema: { type: 'object', properties: { x: { type: 'string' } }, required: ['x'] },
      handler: async (input) => { customRan = true; return input },
    }

    try {
      const gw = new CapturingGateway([
        toolCall('c1', 'custom_echo', { x: 'hi' }),
        // Unauthorized built-in: if dispatch ever reached the real handler it would create markerFile.
        toolCall('c2', 'run_command', {
          command: `printf pwned > '${markerFile.replace(/'/g, `'\\''`)}'`,
        }),
        text('done'),
      ])
      const milkie = new Milkie({
        stateStore: new MemoryStore(),
        eventStore: new MemoryEventStore(),
        gateway: gw,
        tools: [custom],
      })
      milkie.registerAgent(makeConfig({
        builtinTools: { allow: ['think'] },
        fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 5, tools: ['think', 'custom_echo', 'run_command'] }] },
      }))

      const result = await milkie.invoke({ agentId: 'policy-agent', goal: 'g', input: 'i' })
      expect(result.status).toBe('completed')
      expect(customRan).toBe(true)

      // Host side-effect proof: unauthorized run_command must not create the marker.
      expect(fs.existsSync(markerFile)).toBe(false)

      // First LLM request schema must include custom_echo + think, never run_command.
      const names = (gw.requests[0]?.tools ?? []).map(t => t.name).sort()
      expect(names).toContain('custom_echo')
      expect(names).toContain('think')
      expect(names).not.toContain('run_command')

      // Second request after tool results still must not list run_command.
      for (const req of gw.requests) {
        expect((req.tools ?? []).map(t => t.name)).not.toContain('run_command')
      }
    } finally {
      fs.rmSync(markerDir, { recursive: true, force: true })
    }
  })

  it('S2: omitted builtinTools keeps current default built-ins including run_command', async () => {
    const gw = new CapturingGateway([text('ok')])
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: new MemoryEventStore(),
      gateway: gw,
    })
    milkie.registerAgent(makeConfig())
    await milkie.invoke({ agentId: 'policy-agent', goal: 'g', input: 'i' })
    const names = (gw.requests[0]?.tools ?? []).map(t => t.name)
    expect(names).toEqual(expect.arrayContaining(['run_command', 'think', 'skill_list', 'cite']))
  })

  it('trace run-start records effective built-in name summary', async () => {
    const eventStore = new MemoryEventStore()
    const gw = new CapturingGateway([text('ok')])
    const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore, gateway: gw })
    milkie.registerAgent(makeConfig({ builtinTools: { allow: ['think', 'create_plan'] } }))
    const result = await milkie.invoke({ agentId: 'policy-agent', goal: 'g', input: 'i' })
    const started = (await eventStore.readByRunId(result.agentRunId))
      .find(e => e.type === 'agent.run.started')
    const payload = started!.payload as AgentRunStartedPayload
    expect(payload.builtinTools).toEqual(['create_plan', 'think'])
  })

  it('unknown allow name fails at AgentRuntime construction (before start)', () => {
    expect(() => new AgentRuntime({
      config: makeConfig({ builtinTools: { allow: ['not_a_tool'] } }),
      goal: 'g', input: 'i',
      stateStore: new MemoryStore(),
      recorder: new InMemoryRecorder(undefined, 'policy-agent'),
      ioPort: new DefaultIOPort(new CapturingGateway([])),
    })).toThrow(/Unknown built-in tool name/)
  })

  it('child cannot widen parent allowlist', async () => {
    const parent = ['think'] as const
    expect(resolveEffectiveBuiltinTools(
      { builtinTools: { allow: ['think', 'run_command'] } },
      parent,
    )).toEqual(['think'])
  })

  it('manifest/frontmatter loads builtinTools.allow', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-bt-'))
    const file = path.join(dir, 'a.md')
    fs.writeFileSync(file, `---
agentId: front-bt
fsm:
  states: [{ name: react, type: llm }]
builtinTools:
  allow: [think, skill_list]
---
sys`)
    try {
      const cfg = new Milkie({ stateStore: new MemoryStore() }).loadAgentFile(file)
      expect(cfg.builtinTools).toEqual({ allow: ['think', 'skill_list'] })
    } finally {
      fs.rmSync(dir, { recursive: true, force: true })
    }
  })
})
