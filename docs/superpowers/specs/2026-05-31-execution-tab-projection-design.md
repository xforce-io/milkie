# #70 把 agent-docs-qa 的 Execution(原 Steps)tab 改成投影驱动 — 设计 spec

**Issue:** #70(diagnosable;Parent #20;follow-up of #68 / PR #69)
**日期:** 2026-05-31
**定位:** 把 `examples/agent-docs-qa` 参考应用 audit panel 里「执行·成本」(原 Steps,`data-tab="steps"`)tab 的**前端归因逻辑**搬到核心**纯投影**,前端只渲染——守住 ARCHITECTURE.md invariant 12/13「UI = projection over CLI/SDK,前端不自带归因逻辑」。

经 brainstorm 定稿:**路 A(核心出结构化 JSON,前端纯渲染)** + **本 MVP 含 CLI 入口**(守 invariant 13)。

---

## 1. 目标

`renderStepsTab`(`public/index.html:882-1009`)目前在浏览器里手工解析事件重写归因:walk `region.added/removed` 维护 `activeRegions`、`llm.requested` 时快照、按 `requestHash` 配对 `tool/llm.responded` 取 `cacheStats`、`classifyCacheTier` 分级、`renderRegionExpander` 按 stability 分组、current-turn 内容预览。本 issue 把这套归因搬进核心纯投影函数,前端改为消费投影 JSON,**行为/外观零变化**。

## 2. 背景与现状

#68(PR #69)给 panel 加了「决策·为什么」tab,**整块复用核心 `renderViewer`**(iframe),前端零归因;但刻意把「执行·成本」tab 的投影化重构踢成独立 follow-up(本 issue),避免 PR 膨胀。

现状关键事实:
- 前端 `renderStepsTab` 是同步函数,从本地已 catch-up 的 `eventsByRunId` 渲染,**不经任何 endpoint**。
- 核心已有可复用投影:`contextRefsAt(events, eventId, 'at')`(某时刻 region composition,`src/trace/RegionContextView.ts:46`)、`regionReuseCounts(events)`(同文件:106)。但**没有** cache tier 分级、没有"组装 Execution step 列表"的投影。
- CLI `trace report <runId>` 出的是 HTML(renderViewer);**没有**出 JSON 投影的子命令。

## 3. Non-goals

- 不改「决策·为什么」tab(#68)、Sources、Provenance。
- 不改 Execution tab 的视觉/交互/文案(纯搬迁,行为保真:cache 分级阈值、stability 顺序、badge 文案、展开预览全部不变)。
- MVP 不做:Execution tab 的 live 自动刷新、descendant 子 run 合并。
- 不引入新依赖、不改事件 schema。

## 4. 架构与组件

```
src/trace/diagnostics/buildExecutionProjection.ts   // 新增:纯投影函数 + 类型
src/__tests__/buildExecutionProjection.test.ts      // 新增:核心单测(TDD 主体)
src/cli/main.ts                                      // 新增 `trace execution <runId>` → JSON
examples/agent-docs-qa/server.ts                    // 新增 GET /run/:runId/execution → JSON
examples/agent-docs-qa/public/index.html            // renderStepsTab 改 fetch + 纯渲染
examples/agent-docs-qa/__tests__/server.test.ts     // endpoint 测试
```

### 4.1 核心 — `buildExecutionProjection`(纯函数,只读 events,无 IO)

与 `buildDecisionSpine`/`explainLlmCall` 并列,风格一致(只读 events、返回可序列化 plain object)。

```ts
export interface CacheHealth {
  tier: 'hot' | 'warm' | 'cold'
  readTokens:       number
  creationTokens:   number
  totalInputTokens: number
  hitRate:          number
}

export interface RegionGroup {
  stability: 'immutable' | 'session-stable' | 'turn-stable' | 'volatile'
  regions:   RegionContentRef[]   // from RegionContextView; preview filled from regionContent
}

export interface ExecutionStep {
  kind:  'llm' | 'tool'
  label: string
  // llm-only:
  messageCount?: number
  cacheHealth?:  CacheHealth | null
  regionGroups?: RegionGroup[]    // 已按 STABILITY_ORDER 排好,空组省略
  prompt?:       { system?: unknown; messages: unknown[]; tools: unknown[] } | null
  response?:     unknown
  // tool-only:
  tool?: { name: string; input?: unknown; output?: unknown; error?: unknown; status: 'ok' | 'pending' | 'error' }
}

export interface ExecutionProjection { steps: ExecutionStep[] }

export function buildExecutionProjection(
  events: Event[],
  opts?: { regionContent?: Map<string, string> },
): ExecutionProjection
```

实现要点(把前端逻辑逐项搬入,**复用核心已有**):
- 遍历 events,遇 `llm.requested` 产出一个 llm step,遇 `tool.requested` 产出一个 tool step(顺序 = 事件顺序)。
- **region composition**:用 `contextRefsAt(events, llmRequestedEventId, 'at')` 取该时刻 region 集合(替代前端手 walk `activeRegions`)。`opts.regionContent`(按 contentHash)命中则给 `RegionContentRef.content` 灌内容。注:前端原来只硬编码预览 `current-turn`(代码注释自承"not a general mechanism");这里用 #26 内容寻址**正确泛化**为"任何有落盘内容的 region 都可预览"(与 Why tab / renderViewer 一致),是预期内的小改进而非行为漂移;无内容则降级"(内容不可用)",不报错。
- **cache health**:按 `requestHash` 找配对 `llm.responded.cacheStats`;tier 阈值搬自前端 `classifyCacheTier`(`hit≥0.7→hot`;`hit≥0.3 || creationTokens>0→warm`;否则 `cold`),无 cacheStats → `cacheHealth: null`。
- **region 分组**:常量 `STABILITY_ORDER = ['immutable','session-stable','turn-stable','volatile']` 搬入核心,按序分组,空组省略;返回 `regionGroups`(前端不再自己分组)。
- **复用计数**:`regionReuseCounts(events)`,把计数附到对应 region(前端"复用 ×N"标注的数据源)。
- **tool 配对**:按 `requestHash` 找 `tool.responded`,有 output → `status:'ok'`,有 error → `'error'`,无 → `'pending'`。

### 4.2 CLI — `trace execution <runId>`(invariant 13)

`src/cli/main.ts` 新增子命令,照搬 `trace report`(:179-204)的 regionContent hydrate 模式,但输出 JSON:

1. `findMilkieDir` → `JsonlEventStore(runsDir)` → `events = readByRunId(runId)`(含 descendant?MVP 单 runId,与 server 一致;descendant 留扩展)。
2. hydrate `regionContent`:遍历 `regionReuseCounts(events).keys()`,`FileTraceObjectStore.getCanonical(h)` 命中填入 map。
3. `stdout.push(JSON.stringify(buildExecutionProjection(events, { regionContent })) + '\n')`。

### 4.3 Server — `GET /run/:runId/execution` → JSON

平行 #68 `handleViewer`(`server.ts`),但出 JSON:
1. `events = await eventStore.readByRunId(runId)`;空 → `sendJson(res, 404, { error: 'run not found' })`。
2. hydrate `regionContent`(同 `handleViewer`)。
3. `sendJson(res, 200, buildExecutionProjection(events, { regionContent }))`。

### 4.4 前端 — `renderStepsTab` 改纯渲染(`public/index.html`)

- 切到「执行·成本」tab 时 `fetch('/run/${runId}/execution')` 取 `ExecutionProjection` JSON(lazy,与 #68 viewer tab 一致);加 loading / error 态。
- 把 `steps[]` 字段塞进**现有 DOM/CSS**(`ap-step`、cacheBadge、region expander 外壳保留,视觉不变)。
- **删除**前端归因:`activeRegions` walk、`requestHash` 配对、`classifyCacheTier`、`STABILITY_ORDER` 分组、preview 推导(若这些 helper 仅 Execution tab 使用则一并删除)。
- cacheBadge 文案逻辑(`{hit}% read`/`wrote {n} cache`/`no cache`)属"渲染呈现",可留前端;但 **tier 分级**(hot/warm/cold 的判定)来自投影 `cacheHealth.tier`,前端不再自己判。

## 5. 数据流 / 边界

- 投影在 server/CLI 请求期算好,前端不重算。
- 未知 runId → endpoint 404 / CLI 报错;前端显错误态。
- region 无内容 hash / objectStore 未命中 → 预览降级"(内容不可用)"(沿用 #26/#64),不报错。
- 无 llm/tool 事件的 run → `steps: []`,前端显"No recorded steps"(沿用现状)。
- 旧 run(objectStore 无内容)→ 预览降级,不报错。

## 6. 测试

- **核心单测(TDD 主体)** `buildExecutionProjection.test.ts`(纯函数,易测,先红后绿):
  - cache tier 三档:hit≥0.7→hot;hit 0.3~0.7 或 creationTokens>0→warm;否则 cold;无 cacheStats→null。
  - region 按 stability 分组顺序 = STABILITY_ORDER,空组省略。
  - 复用计数附着正确(同 hash region 复用 ×N)。
  - tool step 配对:ok / error / pending 三态。
  - regionContent 命中 → preview 灌入;未命中 → 降级无 preview。
  - step 顺序 = 事件顺序;llm step 带 messageCount/prompt/response。
- **CLI** `trace execution <runId>` → stdout 合法 JSON、结构匹配(用既有 e2e 风格或集成测)。
- **server** `server.test.ts`:`GET /run/:id/execution` → 200 + JSON 结构(steps 非空、含 cacheHealth/regionGroups);未知 runId → 404;现有测试全绿。
- **回归断言**:`public/index.html` grep 不到 `activeRegions` / `classifyCacheTier` / `STABILITY_ORDER` 解析逻辑(证明归因已搬走)。
- **browser 端到端(dev-browser)**:起 server → 发一次 chat → 开 audit panel → 切「执行·成本」tab → 断言:steps 列表渲染、cache badge tier class(hot/warm/cold)、region expander 按 stability 分组展开、内容预览可见 → 与重构前外观一致。

## 7. 改动文件清单

| 文件 | 改动 |
|---|---|
| `src/trace/diagnostics/buildExecutionProjection.ts` | 新增纯投影函数 + 类型 |
| `src/__tests__/buildExecutionProjection.test.ts` | 新增核心单测 |
| `src/cli/main.ts` | 新增 `trace execution <runId>` → JSON |
| `examples/agent-docs-qa/server.ts` | 新增 `GET /run/:runId/execution` handler |
| `examples/agent-docs-qa/public/index.html` | `renderStepsTab` 改 fetch + 纯渲染,删前端归因 |
| `examples/agent-docs-qa/__tests__/server.test.ts` | endpoint 测试 |
| `roadmap.md` | 标注 #70 落地 |

## 8. Follow-up(不进本 PR)

- Execution tab live 自动刷新、descendant 子 run 合并。
- `trace execution` 的 descendant 合并(与 `trace report` 对齐)。

## 9. 验收对照(#70)

- [ ] Execution tab 改投影驱动:server/CLI 出投影 JSON,前端只渲染 → §4.1-4.4
- [ ] 保留现有 UX:cache health 分级(hot/warm/cold + token)、region 按 stability 分组 → §4.1 投影 + §4.4 渲染 + §6 测试
- [ ] 前端不自带归因逻辑(grep 回归断言)→ §6
- [ ] 守 invariant 13:新能力随 CLI 入口交付 → §4.2

## 10. 分支

`feat/70-execution-tab-projection`,基于 main(含 #68/#26)。
