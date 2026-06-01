# #89 内置/标准 agent 层(v1)— 实现 spec

**Issue:** #89(设计定稿见 issue 正文);本 doc 补**实现落点 + 测试策略**。
**日期:** 2026-06-01 · **分支:** `feat/89-standard-agent-layer`(基于含 #90 的 main)

## 目标

确立"内置/标准 agent(模板)"语义层:框架自带、任意应用可 opt-in 加载、model 由应用提供。第一版只为**把 #88 的 diagnoser 从 example 上移**所需的最小机制。

## 实现落点

### 1. `AgentConfig.model` 改可选 — `src/types/agent.ts`

`model: ModelConfig` → `model?: ModelConfig`。内置模板不带 model。

### 2. `parseConfig` 允许缺 model — `src/runtime/Milkie.ts:527`

当前缺 model.provider/model/adapter 直接抛错。改为:**有 model 块则照常校验三字段;完全无 model 块则 `model` 留 undefined**(模板态)。fsm.states 仍必填。

### 3. `resolveGateway(config)` helper + `defaultModel` opt — `src/runtime/Milkie.ts`

`new Milkie({ ..., defaultModel?: ModelConfig })` 新增 opt(存 `this.defaultModel`)。

抽 helper 统一现有 4 处 `this.gatewayOverride ?? createGateway(config.model)`(invoke:184 / resume:284 / replay:391 / makeChildPort:79):

```ts
private resolveGateway(config: AgentConfig): IModelGateway {
  if (this.gatewayOverride) return this.gatewayOverride
  const model = config.model ?? this.defaultModel
  if (!model) throw new Error(
    `Agent "${config.agentId}" has no model and Milkie has no gateway/defaultModel; ` +
    `built-in agents need a gateway or defaultModel at construction.`)
  return createGateway(model)
}
```
4 处替换为 `this.resolveGateway(config)`。(makeChildPort 处 config 变量名为 childConfig,照改。)

### 4. `loadStandardAgents()` — `src/runtime/Milkie.ts`

```ts
loadStandardAgents(): string[] {           // returns loaded agentIds
  const dir = path.join(__dirname, '..', '..', 'agents')   // 包根/agents,dev(src)与 prod(dist)路径一致
  if (!fs.existsSync(dir)) return []
  const loaded: string[] = []
  for (const f of fs.readdirSync(dir)) {
    if (f.endsWith('.md')) { loaded.push(this.loadAgentFile(path.join(dir, f)).agentId) }
  }
  return loaded
}
```
**显式 opt-in**(应用主动调,不在构造自动跑)。同 id 覆盖沿用 `Map.set`(应用先 loadStandardAgents 再 loadAgentFile 自己的同名即覆盖)。

### 5. 读-Trace 工具内置进框架 — 移动 + 随 `loadStandardAgents()` 注册

- 把 `examples/agent-docs-qa/tools/trace-tools.ts` 的 `makeTraceTools(eventStore, objectStore)` 移到 `src/tools/trace.ts`(导出 `makeTraceTools`)。
- **有状态**(需 eventStore/objectStore),不能进静态 `systemTools`。它是标准 agent(diagnoser)的依赖,所以**在 `loadStandardAgents()` 内注册**(跟着 opt-in 一起),不在 Milkie 构造时注册——避免给不用 diagnoser 的实例平白塞工具:
  ```ts
  // loadStandardAgents() 内,加载 md 之前/之后:
  if (this.eventStore) {
    for (const t of makeTraceTools(this.eventStore, this.traceObjectStore)) this.registerTool(t)
  }
  ```
- `makeTraceTools` 须容忍 `objectStore` 为 undefined(`get_run_io` 不用它;`get_execution` 的 hydrate 在 objectStore 缺省时跳过内容、仍返回投影)。implementer 确认其签名/实现接受 `ITraceObjectStore | undefined`。
- `ToolRegistry.getForState(agent.tools)` 已按 agent 声明的 tools 过滤,所以 trace 工具只对声明了它们的 agent(diagnoser)可见,不污染领域 agent。

### 6. 顶级 `agents/` 目录 + diagnoser 上移

- 新建顶级 `agents/diagnoser.md` = 原 `examples/agent-docs-qa/agents/diagnoser.md` **删掉整个 `model:` 块**(其余 frontmatter/body 不变)。
- 删除 `examples/agent-docs-qa/agents/diagnoser.md` 与 `examples/agent-docs-qa/tools/trace-tools.ts`(及其 import 处)。
- `package.json` 加 `"files": [...]` 含 `agents/` 与 `dist/`(当前无 files 字段;补一个把 dist + agents 纳入发布)。

### 7. example 改造 — `examples/agent-docs-qa/server.ts`

- 删 `import { makeTraceTools } from './tools/trace-tools.js'` → 改 `from '../../src/tools/trace.js'`(或干脆不在 server 手动注册:trace 工具已由 Milkie 构造自动注册)。
- 删手动加载 diagnoser.md 那段,改为 `milkie.loadStandardAgents()`。
- example 用真实 doubao,diagnoser 无 model → 走 defaultModel/gateway 回退;server 构造 Milkie 时传 `gateway`(已有)或 `defaultModel`,使 diagnoser 可跑。

## 测试策略

- **核心单测(TDD)**:
  - `parseConfig`:有 model→校验;无 model 块→`model` undefined,不抛错;model 块缺字段→仍抛错。
  - `resolveGateway`:gatewayOverride 优先;无 override 用 config.model;config 无 model 用 defaultModel;都无→抛清晰错误。
  - `loadStandardAgents()`:加载顶级 `agents/` 的 md,返回 agentId(含 diagnoser);同 id 后续 loadAgentFile 覆盖。
  - Milkie 构造注册内置 trace 工具(有 eventStore 时)。
- **集成测**:用 stub gateway 构造 Milkie + `loadStandardAgents()` + `defaultModel`(或 gateway),invoke diagnoser(input=某 runId)→ 复用 #88 的非-hollow 断言(get_execution 的 tool.responded 带真实投影 + 输出 JSON 契约)。证内置 diagnoser 端到端可跑。
- **回归**:`src/__tests__` + `examples/agent-docs-qa/__tests__` 全绿;example server 测试(diagnoser 经 loadStandardAgents 跑通,无本地副本)。
- **不回归 #90**:trace 工具移动后,原 example diagnoser 测试等价通过(改为消费内置)。

## 文件清单

| 文件 | 改动 |
|---|---|
| `src/types/agent.ts` | `model?` 可选 |
| `src/runtime/Milkie.ts` | parseConfig 容许缺 model;`defaultModel` opt;`resolveGateway` helper(4 处替换);`loadStandardAgents()`;构造注册内置 trace 工具 |
| `src/tools/trace.ts` | 新增(从 example 移入 `makeTraceTools`) |
| `agents/diagnoser.md` | 新增(顶级;无 model) |
| `package.json` | 加 `files`(dist + agents) |
| `examples/agent-docs-qa/server.ts` | 改用内置 trace 工具 + `loadStandardAgents()` |
| `examples/agent-docs-qa/agents/diagnoser.md`、`tools/trace-tools.ts` | 删除(上移) |
| `src/__tests__/`、example `__tests__/` | 新增/调整测试 |

## 非目标

标准 skill 库(verifier 上移)、版本化、派生 API、跨多 model 变体并存。
