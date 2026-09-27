/** 离线确定性契约探针；产品契约失败 exit 1，探针/环境异常 exit 2。 */
import assert from 'node:assert/strict'
import type { AddressInfo } from 'node:net'
import { Milkie } from '../../../../src/runtime/Milkie'
import { MemoryStore } from '../../../../src/store/MemoryStore'
import { MemoryEventStore } from '../../../../src/trace/MemoryEventStore'
import { BroadcastingEventStore } from '../../../../src/trace/BroadcastingEventStore'
import { createServeServer } from '../../../../src/cli/serve'
import type { AgentConfig } from '../../../../src/types/agent'
import type { ChildAgentRecord } from '../../../../src/types/store'
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent } from '../../../../src/types/model'
import type { ToolDefinition } from '../../../../src/types/tool'

const config = (agentId = 'probe'): AgentConfig => ({
  agentId, version: '1', systemPrompt: agentId,
  fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 1 }] },
  model: { provider: 'stub', model: 'stub', adapter: 'stub' },
})
const note: ToolDefinition = {
  name: 'note', description: 'register a fixture artifact', inputSchema: { type: 'object' },
  handler: async (_, ctx) => {
    ctx.recordArtifact?.({ name: 'note', type: 'file', path: 'fixture-note.txt' })
    return 'noted' // 仅登记定位信息；不伪称验证了真实文件写入。
  },
}
class Gateway implements IModelGateway {
  calls = 0
  mode: 'loop' | 'parent' | 'text' | 'error' = 'loop'
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    this.calls++
    if (this.mode === 'error') throw new Error('fixture model failure')
    if (this.mode === 'text') return { content: [{ type: 'text', text: 'done' }], toolCalls: [], finishReason: 'stop' }
    const name = this.mode === 'parent' && this.calls === 1 ? 'child' : 'note'
    return { content: [], toolCalls: [{ id: `t${this.calls}`, name, input: name === 'child' ? { goal: 'g', input: 'i' } : {} }], finishReason: 'tool_use' }
  }
  async *stream(req: ModelRequest): AsyncIterable<ModelEvent> {
    const response = await this.complete(req)
    for (const call of response.toolCalls ?? []) {
      yield { type: 'tool_call_start', data: { toolCallId: call.id, name: call.name } }
      yield { type: 'tool_call_done', data: { toolCallId: call.id, input: call.input } }
    }
  }
}
function setup(eventStore: MemoryEventStore | BroadcastingEventStore | undefined = new MemoryEventStore()) {
  const gateway = new Gateway(), stateStore = new MemoryStore()
  const milkie = new Milkie({ gateway, stateStore, eventStore, tools: [note] })
  milkie.registerAgent(config())
  return { milkie, gateway, stateStore, eventStore: eventStore! }
}
let failures = 0
function record(id: string, pass: boolean, actual: unknown): void {
  if (!pass) failures++
  console.log(JSON.stringify({ id, result: pass ? 'pass' : 'fail', actual }))
}
async function main(): Promise<void> {
  {
    const { milkie, eventStore } = setup()
    const first = await milkie.invoke({ agentId: 'probe', goal: 'g', input: 'i', contextId: 'resume-probe' })
    assert.equal(first.stopReason, 'budget_exhausted')
    assert.ok(first.checkpointId)
    const initialEvents = await eventStore.readByRunId(first.agentRunId)
    assert.ok(initialEvents.some(e => e.type === 'agent.checkpoint'))
    let error: string | undefined
    try { await milkie.resume(first.checkpointId, 'probe', 'g', 'i') }
    catch (err) { error = String(err) }
    record('259-returned-id', !error, { checkpointId: first.checkpointId, error })
    // 独立实例避免修复 #259 后成功恢复改变 #260 探针的基线。
    const isolated = setup()
    const r = await isolated.milkie.invoke({ agentId: 'probe', goal: 'g', input: 'i', contextId: 'boundary-probe' })
    const results = []
    for (const mode of ['loop', 'text', 'error'] as const) {
      isolated.gateway.mode = mode
      results.push(await isolated.milkie.resume('context:boundary-probe:checkpoint:latest', 'probe', 'g', 'i'))
    }
    assert.deepEqual(results.map(x => x.stopReason), ['budget_exhausted', 'model_stop', 'runtime_error'])
    const runIds = [...new Set([r.agentRunId, ...results.map(x => x.agentRunId)])]
    const events = (await Promise.all(runIds.map(id => isolated.eventStore.readByRunId(id)))).flat()
    const starts = events.filter(e => e.type === 'agent.run.started').length
    const terminals = events.filter(e => e.type === 'agent.run.completed')
    record('260-resume-boundaries', starts === 4 && terminals.length === 4,
      { starts, terminals: terminals.map(e => e.payload), resumeReasons: results.map(x => x.stopReason) })
    // 此检查使用既有 started/completed 事件；若未来采用显式 segment 事件，应先评审并更新探针契约。
  }
  {
    const milkie = new Milkie({ gateway: new Gateway(), stateStore: new MemoryStore(), tools: [note] })
    milkie.registerAgent(config())
    const result = await milkie.invoke({ agentId: 'probe', goal: 'g', input: 'i' })
    record('259-without-event-store', !result.checkpointId, { stopReason: result.stopReason, checkpointId: result.checkpointId })
  }
  {
    const { milkie, gateway, stateStore, eventStore } = setup()
    gateway.mode = 'parent'
    milkie.registerAgent({ ...config(), subAgents: { child: '1' } })
    milkie.registerAgent(config('child'))
    const parent = await milkie.invoke({ agentId: 'probe', goal: 'g', input: 'i', contextId: 'parent-probe' })
    const children = await stateStore.get('context:parent-probe:children') as unknown as ChildAgentRecord[]
    assert.equal(children.length, 1)
    const childRunId = children[0]!.runId
    assert.ok(childRunId)
    const childEvents = await eventStore.readByRunId(childRunId)
    const terminal = childEvents.find(e => e.type === 'agent.run.completed')?.payload as Record<string, unknown>
    assert.ok(terminal)
    const parentEvents = await eventStore.readByRunId(parent.agentRunId)
    const response = parentEvents.find(e => e.type === 'tool.responded')?.payload
    assert.ok(response)
    const rawOutput = (response as Record<string, unknown>).output
    let decoded: unknown = rawOutput
    if (typeof rawOutput === 'string') {
      try { decoded = JSON.parse(rawOutput) } catch { decoded = null }
    }
    const envelope = decoded && typeof decoded === 'object' ? decoded as Record<string, unknown> : undefined
    const pass = terminal.stopReason === 'budget_exhausted' && terminal.partial === true
      && Array.isArray(terminal.artifacts) && terminal.artifacts.length > 0
      && Boolean(terminal.checkpointId) && envelope?.stopReason === 'budget_exhausted'
      && envelope.partial === true && Array.isArray(envelope.artifacts) && envelope.artifacts.length > 0
    record('261-child-result', pass, { terminal, child: children[0], response })
  }
  {
    const broadcaster = new BroadcastingEventStore(new MemoryEventStore())
    const { milkie } = setup(broadcaster)
    const server = createServeServer({ milkie, broadcaster, agentId: 'probe' })
    await new Promise<void>((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve) })
    try {
      const port = (server.address() as AddressInfo).port
      const response = await fetch(`http://127.0.0.1:${port}/chat`, {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ contextId: 'http-probe', input: 'i' }), signal: AbortSignal.timeout(10000),
      })
      assert.equal(response.status, 200)
      const frames = (await response.text()).split('\n\n').filter(s => s.startsWith('event: agent.run.completed\n'))
      assert.equal(frames.length, 1)
      const terminal = JSON.parse(frames[0]!.split('\ndata: ')[1]!)
      const persisted = (await broadcaster.readByRunId(terminal.runId)).find(e => e.type === 'agent.run.completed')?.payload as Record<string, unknown>
      assert.equal(persisted.stopReason, 'budget_exhausted')
      const keys = ['stopReason', 'stopCode', 'partial', 'checkpointId', 'artifacts']
      record('261-http-result', keys.every(key => JSON.stringify(terminal[key]) === JSON.stringify(persisted[key])), { terminal, persisted })
    } finally {
      server.closeAllConnections()
      await new Promise<void>((resolve, reject) => server.close(err => err ? reject(err) : resolve()))
    }
  }
  process.exitCode = failures ? 1 : 0
}
main().catch(err => { console.error(err); process.exitCode = 2 })
