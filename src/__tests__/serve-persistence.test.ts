// #130: serve can run on a persistent backend (SQLite state + Jsonl events) so a
// restarted sidecar recovers a context from its event-sourced checkpoint — no
// alfred-side import/replay. buildServeStores picks the backend; default stays
// in-memory (backward compatible with #86/#124/#126/#128).
import { buildServeStores } from '../cli/serve'
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { AgentConfig } from '../types/agent'
import type { ToolDefinition } from '../types/tool'
import fs from 'fs'
import path from 'path'
import os from 'os'

const recorderAgent: AgentConfig = {
  agentId: 'recorder', version: '1.0.0', systemPrompt: 'answer',
  fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 50 }] },
  model: { provider: 'test', model: 'test', adapter: 'test' },
}

function factWriter(): ToolDefinition {
  return {
    name: 'record_fact', description: 'record a fact',
    inputSchema: { type: 'object', properties: { key: { type: 'string' }, value: { type: 'string' } }, required: ['key', 'value'] },
    handler: async (input: unknown, ctx) => {
      const { key, value } = input as { key: string; value: string }
      ctx.workingMemory.set(key, value)
      return { recorded: key }
    },
  }
}

/** Each turn: first call records fact{n} (tool_use), second call completes. */
function multiTurnRecorder(): IModelGateway {
  let calls = 0, facts = 0
  return {
    async complete(_req: ModelRequest): Promise<ModelResponse> {
      calls++
      if (calls % 2 === 1) {
        facts++
        const input = { key: `fact${facts}`, value: `v${facts}` }
        return { content: [{ type: 'tool_use', id: `c${facts}`, name: 'record_fact', input }], toolCalls: [{ id: `c${facts}`, name: 'record_fact', input }], finishReason: 'tool_use' }
      }
      return { content: [{ type: 'text', text: 'done' }], toolCalls: [], finishReason: 'end_turn' }
    },
    async *stream(_r: ModelRequest): AsyncIterable<never> { yield* [] },
  }
}

function closeIfPossible(store: unknown): void {
  if (store && typeof (store as { close?: unknown }).close === 'function') (store as { close(): void }).close()
}

let tmpDir: string
beforeEach(() => { tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-persist-')) })
afterEach(() => { fs.rmSync(tmpDir, { recursive: true, force: true }) })

describe('#130 buildServeStores — backend selection', () => {
  it('defaults to in-memory stores (no data-dir, no files written)', async () => {
    const { stateStore } = await buildServeStores({})
    expect(stateStore).toBeInstanceOf(MemoryStore)
    await stateStore.set('k', 'v')
    expect(await stateStore.get('k')).toBe('v')
  })

  it('state-store=sqlite persists state to disk across fresh store instances (restart)', async () => {
    const first = await buildServeStores({ stateStore: 'sqlite', dataDir: tmpDir })
    await first.stateStore.set('context:C:checkpoint-run:latest', 'run-1')
    closeIfPossible(first.stateStore)
    expect(fs.existsSync(path.join(tmpDir, 'state.sqlite'))).toBe(true)

    // A brand-new store instance over the same data-dir reads the persisted value.
    const second = await buildServeStores({ stateStore: 'sqlite', dataDir: tmpDir })
    expect(await second.stateStore.get('context:C:checkpoint-run:latest')).toBe('run-1')
    closeIfPossible(second.stateStore)
  })

  it('state-store=sqlite persists events to disk (one jsonl file per run)', async () => {
    const { stateStore, eventStore } = await buildServeStores({ stateStore: 'sqlite', dataDir: tmpDir })
    await eventStore.append({ id: 'e1', runId: 'run-1', type: 'agent.run.started', actor: 'a', timestamp: 1, payload: { agentId: 'a', goal: 'g', input: 'hi', contextId: 'C' } })
    closeIfPossible(stateStore)

    const re = await buildServeStores({ stateStore: 'sqlite', dataDir: tmpDir })
    const events = await re.eventStore.readByRunId('run-1')
    expect(events).toHaveLength(1)
    expect(events[0]!.runId).toBe('run-1')
    closeIfPossible(re.stateStore)
  })

  it('state-store=sqlite without data-dir is rejected', async () => {
    await expect(buildServeStores({ stateStore: 'sqlite' })).rejects.toThrow(/data-dir/)
  })
})

describe('#130 restart recovery — same contextId continues from checkpoint after a fresh instance', () => {
  it('a new Milkie over the same data-dir sees the prior turns\' context', async () => {
    const contextId = 'C'

    // Instance 1: persistent stores, two turns that record facts into working memory.
    const s1 = await buildServeStores({ stateStore: 'sqlite', dataDir: tmpDir })
    const m1 = new Milkie({ stateStore: s1.stateStore, eventStore: s1.eventStore, gateway: multiTurnRecorder(), tools: [factWriter()] })
    m1.registerAgent(recorderAgent)
    await m1.invoke({ agentId: 'recorder', goal: 'g', input: 'turn1', contextId })  // → fact1
    await m1.invoke({ agentId: 'recorder', goal: 'g', input: 'turn2', contextId })  // → fact2
    closeIfPossible(s1.stateStore)   // simulate shutdown: release the SQLite handle

    // Instance 2: brand-new stores over the SAME data-dir (a "restart"). A capturing
    // gateway records the prompt the recovered turn assembles.
    let captured: ModelRequest | undefined
    const capturing: IModelGateway = {
      async complete(req: ModelRequest): Promise<ModelResponse> {
        captured ??= req
        return { content: [{ type: 'text', text: 'ok' }], toolCalls: [], finishReason: 'end_turn' }
      },
      async *stream(_r: ModelRequest): AsyncIterable<never> { yield* [] },
    }
    const s2 = await buildServeStores({ stateStore: 'sqlite', dataDir: tmpDir })
    const m2 = new Milkie({ stateStore: s2.stateStore, eventStore: s2.eventStore, gateway: capturing, tools: [factWriter()] })
    m2.registerAgent(recorderAgent)
    await m2.invoke({ agentId: 'recorder', goal: 'g', input: 'turn3', contextId })
    closeIfPossible(s2.stateStore)

    // The recovered turn's prompt carries the prior turns' working memory — restored
    // from the on-disk checkpoint event + pointer, no import/replay.
    const prompt = JSON.stringify(captured)
    expect(prompt).toContain('fact1')
    expect(prompt).toContain('fact2')
  })

  it('default (memory) does NOT recover across instances — confirms persistence is what enables recovery', async () => {
    const contextId = 'C'
    const s1 = await buildServeStores({})
    const m1 = new Milkie({ stateStore: s1.stateStore, eventStore: s1.eventStore, gateway: multiTurnRecorder(), tools: [factWriter()] })
    m1.registerAgent(recorderAgent)
    await m1.invoke({ agentId: 'recorder', goal: 'g', input: 'turn1', contextId })

    let captured: ModelRequest | undefined
    const capturing: IModelGateway = {
      async complete(req: ModelRequest): Promise<ModelResponse> { captured ??= req; return { content: [{ type: 'text', text: 'ok' }], toolCalls: [], finishReason: 'end_turn' } },
      async *stream(_r: ModelRequest): AsyncIterable<never> { yield* [] },
    }
    const s2 = await buildServeStores({})   // fresh in-memory stores: nothing carried over
    const m2 = new Milkie({ stateStore: s2.stateStore, eventStore: s2.eventStore, gateway: capturing, tools: [factWriter()] })
    m2.registerAgent(recorderAgent)
    await m2.invoke({ agentId: 'recorder', goal: 'g', input: 'turn2', contextId })

    expect(JSON.stringify(captured)).not.toContain('fact1')
  })
})
