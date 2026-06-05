import { AgentRuntime } from '../runtime/AgentRuntime'
import { DefaultIOPort } from '../runtime/IOPort'
import { MemoryStore } from '../store/MemoryStore'
import { InMemoryRecorder } from '../trajectory/InMemoryRecorder'
import type { AgentConfig } from '../types/agent'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { ToolDefinition } from '../types/tool'

// #139 提议2（最小版）：把工具注册顺序的覆盖语义钉成回归测试。
//
// AgentRuntime.registerTools 的顺序是 systemTools → cognitive → lineage → extraTools
// → subAgents，配合 ToolRegistry 的 Map last-wins。这两条特性决定了：
//   1. 宿主 extraTools 可覆盖同名 system 工具（这是进程内 host 替换 skill_list 的口子）。
//   2. subAgent 在 extraTools 之后注册 → 同名时 subAgent 反覆盖 extraTool（footgun）。
//
// 它们目前是隐式行为、无测试保障。这里不写文档契约那套仪式，只用一条 characterization
// 测试锁住现状：日后若有人重构 registerTools 改了顺序，本测试会立刻 surface。

function makeConfig(overrides: Partial<AgentConfig> = {}): AgentConfig {
  return {
    agentId:      'override-test',
    version:      '1.0.0',
    systemPrompt: 'test',
    fsm:          { states: [{ name: 'react', type: 'llm' }] },   // 无 tools 限制 → 暴露全部已注册工具
    model:        { provider: 'test', model: 'test-model', adapter: 'test' },
    ...overrides,
  }
}

function fakeTool(name: string, description: string): ToolDefinition {
  return { name, description, inputSchema: { type: 'object', properties: {}, required: [] }, handler: async () => ({}) }
}

// 捕获暴露给模型的工具 schema（name/description 来自实际注册的 winner），随即终止本轮。
class CapturingGateway implements IModelGateway {
  public lastTools: Array<{ name: string; description: string }> = []
  async complete(req: ModelRequest): Promise<ModelResponse> {
    this.lastTools = (req.tools ?? []).map(t => ({ name: t.name, description: t.description }))
    return { content: [{ type: 'text', text: 'done' }], toolCalls: [], finishReason: 'end_turn' }
  }
  async *stream(): AsyncIterable<never> { yield* [] }
}

function runWith(opts: { config?: AgentConfig; extraTools?: ToolDefinition[] }): Promise<CapturingGateway> {
  const gw = new CapturingGateway()
  const runtime = new AgentRuntime({
    config:     opts.config ?? makeConfig(),
    goal:       'g',
    input:      'i',
    stateStore: new MemoryStore(),
    recorder:   new InMemoryRecorder(undefined, 'override-test'),
    ioPort:     new DefaultIOPort(gw),
    ...(opts.extraTools ? { extraTools: opts.extraTools } : {}),
  })
  return runtime.run('i').then(() => gw)
}

describe('tool 注册覆盖契约 (#139 提议2)', () => {
  it('extraTools 覆盖同名 system 工具（last-wins）— skill_list 可被宿主替换', async () => {
    const gw = await runWith({ extraTools: [fakeTool('skill_list', 'OVERRIDDEN_SKILL_LIST')] })
    const sl = gw.lastTools.find(t => t.name === 'skill_list')
    expect(sl).toBeDefined()
    expect(sl!.description).toBe('OVERRIDDEN_SKILL_LIST')   // 默认 system stub 被 extraTool 覆盖
  })

  it('subAgent 在 extraTools 之后注册 → 同名 extraTool 被 subAgent 反覆盖（已知 footgun，锁住现状）', async () => {
    const gw = await runWith({
      config:     makeConfig({ subAgents: { worker: '1.0.0' } }),
      extraTools: [fakeTool('worker', 'OVERRIDDEN_WORKER')],
    })
    const w = gw.lastTools.find(t => t.name === 'worker')
    expect(w).toBeDefined()
    // subAgent 后注册 → 它赢；若哪天改成 extraTool 赢，这条会失败、强制重新审视顺序契约。
    expect(w!.description).toBe('Invoke the worker sub-agent.')
  })
})
