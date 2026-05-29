# tool.responded 产出 metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 `tool.responded` 事件加上 `outputHash` / `outputBytes`（+ 为 #37 预留 `artifactRefs?`），并 best-effort 把工具产出写透到 `traceObjectStore`（按 hash 去重），保留 `output` 内联使 replay 零改动。

**Architecture:** 在 `RecordingIOPort.invokeTool` 成功分支，对 `output` 做一次 `canonicalize`，由它派生 `outputHash`（`contentAddressForCanonicalBytes`）与 `outputBytes`（UTF-8 字节数）填进 payload；若构造时注入了 `traceObjectStore` 则 `putCanonical` 写透（自带 `has` 去重）。全部 best-effort：canonicalize/put 失败则降级，不影响 tool 结果。`output` 仍内联 → `CacheIndex`/`ReplayingIOPort` 不动。

**Tech Stack:** TypeScript (ESM, `.js` import 后缀)、jest（项目用 jest；`npx jest <file> -t "<name>"`）。`hash.ts`（`canonicalize` / `contentAddressForCanonicalBytes`）、`ITraceObjectStore`（Memory/File，内容寻址 + 去重）。

**设计依据：** `docs/superpowers/specs/2026-05-29-tool-responded-metadata-design.md`

---

## File Structure

- `src/trace/types.ts` — `ToolRespondedPayload` 加 `outputHash?` / `outputBytes?` / `artifactRefs?`。
- `src/trace/RecordingIOPort.ts` — 构造函数加可选 `objectStore`；`invokeTool` 成功分支算 hash/bytes + 写透；新增私有 `outputMetadata()`。
- `src/runtime/Milkie.ts` — `wrapIOPort` 把 `traceObjectStore` 传进 `RecordingIOPort`。
- `src/__tests__/RecordingIOPort.metadata.test.ts` — 新建，单元测试（确定性 hash / bytes / 去重 / 无 store / best-effort / error 分支）。

---

## Task 1: 数据模型 + RecordingIOPort 计算 hash/bytes（核心）

**Files:**
- Modify: `src/trace/types.ts:66-78`（`ToolRespondedPayload`）
- Modify: `src/trace/RecordingIOPort.ts`（imports、constructor `:57-62`、`invokeTool` 成功分支 `:172-183`、新增私有方法）
- Modify: `src/runtime/Milkie.ts:63-68`（`wrapIOPort`）
- Test: `src/__tests__/RecordingIOPort.metadata.test.ts`（新建）

- [ ] **Step 1: 写失败测试** — 新建 `src/__tests__/RecordingIOPort.metadata.test.ts`：

```ts
import { RecordingIOPort } from '../trace/RecordingIOPort'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { MemoryTraceObjectStore } from '../trace/TraceObjectStore'
import { canonicalize, contentAddressForCanonicalBytes } from '../trace/hash'
import type { IIOPort } from '../runtime/IOPort'
import type { ModelRequest, ModelResponse } from '../types/model'
import type { ToolRespondedPayload } from '../trace/types'

// inner port that runs the execute thunk (like DefaultIOPort) so tests control output
class ExecInnerPort implements IIOPort {
  private nextClock = 1000
  private nextUuid  = 1
  async invokeLLM(_req: ModelRequest): Promise<ModelResponse> {
    return { content: [], toolCalls: [], finishReason: 'end_turn' }
  }
  async invokeTool(_n: string, _i: unknown, execute: () => Promise<unknown>): Promise<unknown> {
    return execute()
  }
  now():  number { return this.nextClock++ }
  uuid(): string { return `uuid-${this.nextUuid++}` }
}

async function respondedPayload(store: MemoryEventStore): Promise<ToolRespondedPayload> {
  const events = await store.readByRunId('r1')
  return events.find(e => e.type === 'tool.responded')!.payload as ToolRespondedPayload
}

describe('RecordingIOPort — tool output metadata (#25)', () => {
  it('adds outputHash and outputBytes derived from canonical output', async () => {
    const inner = new ExecInnerPort()
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(inner, store, 'r1')

    const output = { lines: ['a', 'b'], n: 2 }
    await port.invokeTool('read_file', { path: 'x' }, async () => output)

    const p = await respondedPayload(store)
    const canonical = canonicalize(output)
    expect(p.outputHash).toBe(contentAddressForCanonicalBytes(canonical))
    expect(p.outputBytes).toBe(Buffer.byteLength(canonical, 'utf8'))
    expect(p.output).toEqual(output)   // 仍内联
  })

  it('produces identical outputHash for identical output across runs', async () => {
    const run = async () => {
      const store = new MemoryEventStore()
      await new RecordingIOPort(new ExecInnerPort(), store, 'r1')
        .invokeTool('t', { a: 1 }, async () => ({ z: 1, a: [2, 3] }))
      return (await respondedPayload(store)).outputHash
    }
    expect(await run()).toBe(await run())
  })

  it('still records hash/bytes when no traceObjectStore is provided', async () => {
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(new ExecInnerPort(), store, 'r1')   // no objectStore
    await port.invokeTool('t', {}, async () => 'hello')
    const p = await respondedPayload(store)
    expect(typeof p.outputHash).toBe('string')
    expect(p.outputBytes).toBe(Buffer.byteLength(canonicalize('hello'), 'utf8'))
  })
})
```

- [ ] **Step 2: 跑测试确认失败**

Run: `npx jest src/__tests__/RecordingIOPort.metadata.test.ts -t "tool output metadata"`
Expected: FAIL（`outputHash`/`outputBytes` 还不存在，为 undefined）

- [ ] **Step 3a: 扩展 `ToolRespondedPayload`** — `src/trace/types.ts`，在 `requestHash` 后加：

```ts
export interface ToolRespondedPayload {
  toolName: string
  output?: unknown
  error?: {
    message:    string
    retryable?: boolean
    code?:      string
    name?:      string
  }
  /** Mirrors the requested-event hash. */
  requestHash: string
  /** #25: 成功时 hashCanonical(output) → "sha256:..."；error 分支不填。 */
  outputHash?:   string
  /** #25: canonicalize(output) 的 UTF-8 字节数（= 对象库中那份大小）。 */
  outputBytes?:  number
  /** #25: 为 #37 (object.created) 预留；本期不填。 */
  artifactRefs?: string[]
}
```

- [ ] **Step 3b: `RecordingIOPort` 加 objectStore + 计算逻辑** — `src/trace/RecordingIOPort.ts`：

imports 顶部加：
```ts
import { hashModelRequest, hashToolCall, canonicalize, contentAddressForCanonicalBytes } from './hash.js'
import type { ITraceObjectStore } from './TraceObjectStore.js'
```
（把 `canonicalize, contentAddressForCanonicalBytes` 合并进既有的 `from './hash.js'` import；`hashModelRequest`/`hashToolCall` 本来就在。）

constructor（现 `:57-62`）加第 5 个可选参数：
```ts
  constructor(
    private readonly inner: IIOPort,
    private readonly store: IEventStore,
    private readonly runId: string,
    private readonly actor: string = 'runtime',
    private readonly objectStore?: ITraceObjectStore,
  ) {}
```

新增私有方法（放在 `invokeTool` 附近）：
```ts
  /**
   * Best-effort 产出物元数据。canonicalize 对非普通对象会抛 TypeError —— 此时降级返回 {}，
   * 不让 trace 记录影响 tool 结果。写透对象库同样吞错。
   */
  private async outputMetadata(output: unknown): Promise<{ outputHash?: string; outputBytes?: number }> {
    let canonical: string
    try {
      canonical = canonicalize(output)
    } catch {
      return {}
    }
    if (this.objectStore) {
      try { await this.objectStore.putCanonical(canonical) } catch { /* best-effort */ }
    }
    return {
      outputHash:  contentAddressForCanonicalBytes(canonical),
      outputBytes: Buffer.byteLength(canonical, 'utf8'),
    }
  }
```

`invokeTool` 成功分支（现 `:172-183`）改为先算 meta 再 append：
```ts
      const output = await this.inner.invokeTool(toolName, input, execute)
      const meta = await this.outputMetadata(output)
      await this.store.append({
        id:        this.inner.uuid(),
        runId:     this.runId,
        type:      'tool.responded',
        actor:     this.actor,
        causedBy:  reqEventId,
        timestamp: this.inner.now(),
        payload:   { toolName, output, requestHash, ...meta } satisfies ToolRespondedPayload,
      })
      return output
```
（error 分支不动 —— 不加 hash/bytes。）

- [ ] **Step 3c: `Milkie.wrapIOPort` 传 traceObjectStore** — `src/runtime/Milkie.ts:63-68`：

```ts
  private wrapIOPort(gateway: IModelGateway, runId: string): IIOPort {
    const base = new DefaultIOPort(gateway)
    return this.eventStore
      ? new RecordingIOPort(base, this.eventStore, runId, undefined, this.traceObjectStore ?? undefined)
      : base
  }
```
（第 4 个参数 `undefined` 让 `actor` 取默认 `'runtime'`，第 5 个传对象库。）

- [ ] **Step 4: 跑测试确认通过**

Run: `npx jest src/__tests__/RecordingIOPort.metadata.test.ts`
Expected: PASS（3 个用例）

- [ ] **Step 5: Commit**

```bash
git add src/trace/types.ts src/trace/RecordingIOPort.ts src/runtime/Milkie.ts src/__tests__/RecordingIOPort.metadata.test.ts
git commit -m "feat(trace): tool.responded carries outputHash/outputBytes + write-through (#25)"
```

---

## Task 2: 大 output 在对象库按 hash 去重

**Files:**
- Test: `src/__tests__/RecordingIOPort.metadata.test.ts`（追加）

- [ ] **Step 1: 写测试** — 追加到同一 describe。`MemoryTraceObjectStore` 无公开 size，用 `has(hash)` + 包一层计数 put 来验证去重（同一 canonical 第二次 put 命中 `has` 不重复存）：

```ts
  it('dedupes large output by hash in the trace object store', async () => {
    const objectStore = new MemoryTraceObjectStore()
    let putCount = 0
    const counting = {
      putCanonical: async (b: string) => { putCount++; return objectStore.putCanonical(b) },
      getCanonical: (h: string) => objectStore.getCanonical(h),
      has:          (h: string) => objectStore.has(h),
    }
    const store = new MemoryEventStore()
    const big = 'x'.repeat(100_000)
    const out = { chapter: big }

    // 同一 output 读三次（三次独立 tool call）
    for (let i = 0; i < 3; i++) {
      await new RecordingIOPort(new ExecInnerPort(), store, `r${i}`, undefined, counting)
        .invokeTool('read_file', { path: 'ch1', n: i }, async () => out)
    }

    const hash = contentAddressForCanonicalBytes(canonicalize(out))
    expect(await objectStore.has(hash)).toBe(true)
    // putCanonical 被调 3 次，但对象库按 hash 去重 → 只存一份；getCanonical 返回那一份
    expect(putCount).toBe(3)
    expect(await objectStore.getCanonical(hash)).toBe(canonicalize(out))
  })
```
（说明：`MemoryTraceObjectStore.putCanonical` 内部对相同 hash 是覆盖写同值、天然去重，不会膨胀；本测试验证“同一产出物始终归到同一 hash、可经 hash 取回一份”。`putCount===3` 仅证明每次 tool call 都尝试写透。）

- [ ] **Step 2: 跑测试**

Run: `npx jest src/__tests__/RecordingIOPort.metadata.test.ts -t "dedupes large output"`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/__tests__/RecordingIOPort.metadata.test.ts
git commit -m "test(trace): tool output deduped by hash in object store (#25)"
```

---

## Task 3: best-effort 降级 + error 分支不带元数据

**Files:**
- Test: `src/__tests__/RecordingIOPort.metadata.test.ts`（追加）

- [ ] **Step 1: 写测试** — 追加两个用例：

```ts
  it('degrades gracefully when output is not canonicalizable (no throw, no hash fields)', async () => {
    class Weird { constructor(public v = 1) {} }   // 非普通对象 → canonicalize 抛 TypeError
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(new ExecInnerPort(), store, 'r1', undefined, new MemoryTraceObjectStore())

    const out = new Weird()
    const ret = await port.invokeTool('t', {}, async () => out)

    expect(ret).toBe(out)                       // tool 结果照常返回
    const p = await respondedPayload(store)
    expect(p.outputHash).toBeUndefined()        // 降级：无 hash/bytes
    expect(p.outputBytes).toBeUndefined()
    expect(p.output).toBe(out)                  // output 仍内联
  })

  it('error branch carries no output metadata', async () => {
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(new ExecInnerPort(), store, 'r1', undefined, new MemoryTraceObjectStore())

    await expect(
      port.invokeTool('t', {}, async () => { throw new Error('boom') })
    ).rejects.toThrow('boom')

    const p = await respondedPayload(store)
    expect(p.error?.message).toBe('boom')
    expect(p.outputHash).toBeUndefined()
    expect(p.outputBytes).toBeUndefined()
  })
```

- [ ] **Step 2: 跑测试**

Run: `npx jest src/__tests__/RecordingIOPort.metadata.test.ts -t "degrades gracefully|error branch"`
Expected: PASS（注意：若 `Weird` 实例经 `inner.invokeTool` 原样返回，`canonicalize` 在 `outputMetadata` 内抛 → 被 catch → 返回 `{}`，append 照常）

- [ ] **Step 3: Commit**

```bash
git add src/__tests__/RecordingIOPort.metadata.test.ts
git commit -m "test(trace): best-effort degrade + error branch has no output metadata (#25)"
```

---

## Task 4: 全量回归 + 类型检查

- [ ] **Step 1: 全量单测**

Run: `npx jest src/__tests__ --runInBand`
Expected: 全绿。重点确认 `Replay.test.ts` / `ReplayingIOPort*.test.ts` / `CacheIndex*.test.ts` 不回归（`output` 仍内联，replay 路径未变）；`Trace.test.ts` / `RecordingIOPort.nondet.test.ts` 不回归（新增构造参数可选、旧 3 参调用仍合法）。

- [ ] **Step 2: 类型检查 / 构建**

Run: `npm run build`（= `tsc`）
Expected: EXIT 0，无类型错误。

- [ ] **Step 3: 若失败** 按 superpowers:systematic-debugging 定位；修复后回 Step 1。不得用 `--no-verify`。

---

## Self-Review 备忘（计划作者已核对）

- **Spec 覆盖**：数据模型(T1) / 确定性 hash(T1) / outputBytes(T1) / 无 store 不抛(T1) / 去重(T2) / best-effort 降级 + canonicalize 失败(T3) / error 分支(T3) / replay 不回归(T4)。全部验收项有对应 task。
- **deferral**（spec 已列）：jsonl 卸载（方案 B）、`artifactRefs` emit（#37）—— 本计划不含。
- **类型一致**：`ToolRespondedPayload` 三新字段（T1 定义）在 T1/T3 测试中按 `outputHash?: string` / `outputBytes?: number` 使用一致；`RecordingIOPort` 第 5 参 `objectStore?: ITraceObjectStore` 在 T1 wiring 与 T2/T3 测试构造中签名一致（`new RecordingIOPort(inner, store, runId, undefined, objectStore)`）。
- **合并协调点**（spec §5）：PR #49 的 `buildMakeChildPort` 后合时需给子 port 也传 `traceObjectStore` —— 不在本计划，提醒后合方处理。
- **已知取舍**：`outputBytes` 用 canonical 字节数（非 raw output 大小），与对象库存储的那份一致。
