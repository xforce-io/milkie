# #26 region composition + 内容预览 UI — 设计 spec

**Issue:** #26（observable P1;依赖 #23 region 内容可寻址,已落地;Parent #20）
**日期:** 2026-05-31
**定位:** 把"这次 LLM 调用的 prompt 由哪些 region 拼成、各 region 内容是什么"做成**核心库能力 + 核心静态报告的可见效果**。复用 #23 的 `RegionContextView`,renderHtml 保持纯函数。

---

## 1. 目标与范围

排查 prompt 出问题时,点 `llm.requested` 能看到:**Assembled by** —— 参与 region 的 `{id, target, section, stability, reason}` + 内容预览(折叠/全文)、同 hash 去重标注复用次数、stability 视觉区分。

**范围(已与用户确认):**
- **下沉到核心库**:做成核心可复用能力,核心静态报告(`renderHtml` / `milkie trace report`)直接展示——这是可见效果的落点。
- 示例 app(`examples/agent-docs-qa`)的 audit panel **本次不动**(它 Phase 4a 已有 metadata 版 Assembled-by;复用核心能力是 follow-up)。

## 2. Non-goals

- **不改示例 app**(留 follow-up)。
- **不新增事件类型 / 不改事件 schema**:数据全来自已有 `region.added/removed` 事件 + #23 的 objectStore。
- **renderHtml 保持纯函数**:读 objectStore 的 I/O 由 CLI 调用方做,内容作为数据传入。架构防火墙不破(ARCHITECTURE.md「UI is a pure projection」)。

## 3. 数据齐备性(#23 已就绪)

| 要素 | 来源 | 状态 |
|---|---|---|
| 某 llm.requested 时刻的活跃 region 集 | `contextRefsAt(events, eventId, 'at')`(`RegionContextView.ts:46`) | ✅ 现成,纯 |
| region 的 {id,target,section,stability,reason,contentHash} | `RegionContentRef` | ✅ 现成 |
| 按 hash 取内容 | `hydrateRegionContext` / `objectStore.getCanonical(hash)` | ✅ 现成(#23) |
| CLI 的 objectStore 通路 | `FileTraceObjectStore(objects/)`(`cli/main.ts:34,204`) | ✅ 现成 |

新增的只有:**报告级同 hash 去重 + 复用次数**,和 **renderHtml 接内容 + 渲染 Assembled-by 块**。

## 4. 架构

```
renderHtml(events, opts?)                       // 仍是纯函数
  ├─ 对每个 llm.requested:contextRefsAt(events, reqId, 'at')  → 活跃 region 集(纯)
  ├─ regionReuseCounts(events)                  // 新增纯函数:报告级 contentHash → 引用次数
  └─ opts.regionContent?: Map<contentHash,string> // 由 CLI hydrate 传入(可选)
CLI `trace report`:
  收集所有被引用的 contentHash → objectStore.getCanonical 批量取 → Map<hash,content> → 传给 renderHtml
```

### 4.1 新增纯函数(`src/trace/RegionContextView.ts` —— region-context 逻辑的归属,最可复用)

- `regionReuseCounts(events: Event[]): Map<string, number>`
  报告级聚合:遍历每个 `llm.requested` 的活跃 region 集(`contextRefsAt`),对每个 region 的 `contentHash` 计数——值 = 该内容被多少个 (llm.requested × region) 引用。用于"复用 ×N"。纯函数。

### 4.2 renderHtml 扩展

- 签名:`renderHtml(events: Event[], opts?: { regionContent?: Map<string, string> }): string`
  - `regionContent` 键 = `contentHash`,值 = 内容字符串。**省略时**(如 `render-html` 从纯 JSONL,无 objectStore):Assembled-by 列表照常出(metadata),内容预览显示"(内容不可用)"。优雅降级。
  - 向后兼容:现有 `renderHtml(events)` 调用不变。

### 4.3 CLI `trace report` 改造(`src/cli/main.ts`)

- 构造 `FileTraceObjectStore(objects/)`(路径已知);
- 收集所有 llm.requested 活跃 region 的 `contentHash`,去重后 `getCanonical` 批量取,组装 `Map<hash, content>`;
- `renderHtml(events, { regionContent })`。
- `render-html`(从 --input JSONL,无 objectStore)继续调 `renderHtml(events)`,自动降级为 metadata-only。

## 5. 静态报告渲染(`src/trace/render/`)

给 `llm.requested` 条目(`LlmEntry`)加 **"Assembled by N regions" 展开块**(思路同 #33 的 "Why?" 块):

- **region 来源清单**:每行 `{id} · {section} · {target} · {reason}`,带 **stability 配色 class**(`immutable` / `session-stable` / `turn-stable` / `volatile` 四档,template.ts 各一色)。
- **内容预览**:每行一个可展开 `<pre>`,默认折叠、点击切全文。
- **同 hash 去重**:内容**按 contentHash 在报告内只内联一次**(嵌入一个 `<script type="application/json" id="region-content">` hash→content 注册表或一组隐藏块);每行展开时从注册表取该 hash 内容显示;每行标注 **复用 ×N**(N = `regionReuseCounts` 值)。避免长 prompt 把 HTML 撑爆,满足">20 region 仍可读"。
- **降级**:`regionContent` 缺该 hash → 该行预览显示"(内容不可用)",清单仍在。

## 6. 错误处理 / 边界

- llm.requested 无活跃 region → 不渲 Assembled-by 块(保持现状条目不变)。
- region 无 `contentHash`(纯 metadata region)→ 该行只显示 metadata,无预览。
- `getCanonical` 返回 undefined(hash 不在 store)→ 该 hash 不进 `regionContent`,渲染降级为"(内容不可用)"。
- 长 prompt(>20 region):折叠默认 + 去重控制体积,保证可读(验收项)。

## 7. 测试

- **`regionReuseCounts`**:构造多个 llm.requested 共用同 contentHash 的事件流 → 断言计数正确;单次引用 = 1。
- **renderHtml(带 regionContent)**:llm.requested 出 "Assembled by" 块、含某 region 的 id/section/stability class、内容可在注册表中找到、复用 ×N 标注;**省略 regionContent** → 清单在但预览为"(内容不可用)";向后兼容(无 region 的事件流照常渲染)。
- **去重**:两个 llm.requested 引用同 hash → 内容注册表只内联一次、两行各标复用次数。
- **CLI `trace report`**(可加轻量集成测试或手验):hydrate 后报告含内容;`render-html` 无 objectStore 时降级不报错。

## 8. 改动文件清单

| 文件 | 改动 |
|---|---|
| `src/trace/RegionContextView.ts` | +`regionReuseCounts(events)`(报告级 hash 引用计数) |
| `src/trace/render/html.ts` | `renderHtml` 加可选 `opts.regionContent`;llm 条目渲 "Assembled by" 块 + 内容注册表 |
| `src/trace/render/template.ts` | stability 配色 class + region 预览/复用样式 + 折叠 JS |
| `src/cli/main.ts` | `trace report` 构造 objectStore、hydrate 内容、传入 renderHtml |
| `src/__tests__/regionReuseCounts.test.ts` | 新建 |
| `src/__tests__/render-html.test.ts` | Assembled-by + 去重 + 降级断言 |

## 9. 验收对照(#26)

- [x] 任一 llm.requested 可展开查看完整 region 来源清单 → Assembled-by 块(metadata,不需 objectStore)
- [x] 点 region 行可看到该时刻内容(不依赖运行时副本)→ CLI hydrate 内容 → renderHtml 注册表;纯 event-log + objectStore,无运行时副本
- [x] 长 prompt(>20 region)仍可读 → 折叠默认 + 同 hash 去重
- 附:stability 视觉区分、复用次数标注 → §5
