# tool.responded 产出 metadata（#25）

**Issue:** #25 [observable P0] tool.responded 产出 metadata
**Parent:** #20（Trace substrate gap — 6-capability surface）
**Date:** 2026-05-29
**Blocks:** #37（object.created + tool 关联）

## 背景

今天 `ToolRespondedPayload`（`src/trace/types.ts:66-78`）只有 `output / error / requestHash`，
工具产出没有稳定身份。两个下游需要它：

- **lineage（#37 `object.created`）** 要一个稳定的产出物身份（hash）能被引用——否则所有
  lineage 关系没有可挂的 id；
- **replay 内容寻址** 可用 outputHash 作 cache key。

附带：同一份大 output（如 1MB 章节被 `read_file` 读三次）今天在对象库层面没有去重锚点。

## Scope 决策：方案 A（加元数据 + 写透对象库，保留内联）

工具输出与 region 内容有一个**关键区别**：region 内容 replay 不消费（仅用于 UI 重建），所以
#23 能在 `region.added` 里只存 `contentHash`、内容进 `traceObjectStore`；而
`tool.responded.output` **被 replay 消费**——`CacheIndex.fromEvents` 读它建 tool cache、
`ReplayingIOPort` 把它吐回 runtime。且 `traceObjectStore` 是**可选**的（多数 run/测试不带）。
因此不能照搬 region 的“只存 hash、丢内联”，否则任何无对象库的 run 会失去 replay 能力。

**采用方案 A**：
- `tool.responded` 始终携带 `outputHash` / `outputBytes`（仅成功、output 存在时）；
- `output` **仍内联**——replay / `CacheIndex` / `ReplayingIOPort` 零改动，不依赖对象库；
- best-effort 写透 `traceObjectStore`（`putCanonical(canonicalize(output))`），其自带 `has(hash)`
  去重 → 3×1MB 在对象库里 1 份，lineage 可按 hash 取字节。

**显式不做（方案 B，YAGNI）**：不把大 output 移出 jsonl（不做阈值卸载）。#37 只需稳定 hash 身份，
不需要 jsonl 瘦身；卸载会引入阈值 magic number + replay hydrate + “卸载过的 run 必须带对象库”的
耦合。jsonl 内联副本是方案 A 的已知取舍，若日后 jsonl 体积成真问题，再单开 issue 做卸载。

## 设计

### 1. 数据模型（`src/trace/types.ts`）

`ToolRespondedPayload` 增三个可选字段：

```ts
export interface ToolRespondedPayload {
  toolName: string
  output?: unknown
  error?: { message: string; retryable?: boolean; code?: string; name?: string }
  requestHash: string
  /** 成功时：hashCanonical(output) → "sha256:..."（与 region contentHash 同款内容地址）。 */
  outputHash?:   string
  /** 成功时：Buffer.byteLength(canonicalize(output), 'utf8')，即对象库中那份 canonical 字节数。 */
  outputBytes?:  number
  /** 为 #37（object.created）预留的产出物引用接口；本 issue 不填。 */
  artifactRefs?: string[]
}
```

全部可选 → 既有事件与 replay 不受影响。`error` 分支（无 output）不填这三个字段。

### 2. 计算 + 写透位置（`src/trace/RecordingIOPort.ts`）

放在 `RecordingIOPort.invokeTool` 的成功分支——它本就构造 `tool.responded`、写事件那一刻手里就有
`output` 值。（备选“放 AgentRuntime 像 region 那样”被否决：AgentRuntime 拿不到 tool 的 output
值，要回传，绕。）

- 给 `RecordingIOPort` 构造函数加一个可选 `traceObjectStore?: ITraceObjectStore`；
- 成功分支：
  - `const canonical = canonicalize(output)`；
  - `outputHash = contentAddressForCanonicalBytes(canonical)`；`outputBytes = Buffer.byteLength(canonical, 'utf8')`；
  - 这两个字段填进 `tool.responded` payload；
  - 若 `traceObjectStore` 存在：`await traceObjectStore.putCanonical(canonical)`（其内部 `has(hash)`
    去重）；
- `Milkie.wrapIOPort`（`src/runtime/Milkie.ts:63-68`）把 `this.traceObjectStore ?? undefined`
  传进 `new RecordingIOPort(...)`。

### 3. 语义 & 边界

- **hash/bytes 总是算**（只要 output 存在），与 `traceObjectStore` 是否存在无关——它们是事件自带
  元数据（#37 id + replay cache-key 候选），不依赖对象库。
- **写透仅当 `traceObjectStore` 存在**，且 **best-effort**：`canonicalize` / `putCanonical` 抛错
  时降级——略过 put，必要时连 hash/bytes 一起省（与 region 的 `persistRegionContent` /
  `buildRegionAddedPayload` 容错同姿势）。**绝不**改变 tool 调用结果或 agent 行为：tool 的
  `output` 照常返回、`tool.responded` 照常写（只是可能缺这几个元数据字段）。
- **replay 零改动**：`output` 仍内联；`CacheIndex.fromEvents` / `ReplayingIOPort` 不动；新字段
  对 replay 是“读不到也不需要”。`RecordingIOPort` 只在 record 路径运行，故 hash 计算不进 replay、
  无 nondeterminism。
- **canonicalize 限制**：`canonicalize` 对非普通对象（自定义类实例）抛 `TypeError`
  （`hash.ts:18-19`）。此时按 best-effort 降级（该 output 不出 hash/bytes、不入对象库），tool
  仍正常返回。

### 4. 错误处理

- 工具报错（`error` 分支）：不加 hash/bytes/artifactRefs（无 output）。现有 error 记录路径不变。
- 写透失败 / canonicalize 失败：见 §3 best-effort，不影响业务。

## 测试（`src/__tests__/`，扩展 `Trace.test.ts` / `RecordingIOPort.nondet.test.ts` 既有 harness）

1. 纯函数工具同 input 两次 run → 相同 `outputHash`（验收 #1）。
2. `outputBytes` === `Buffer.byteLength(canonicalize(output), 'utf8')`（验收 #2）。
3. 带 `traceObjectStore`：大 output 进对象库；同一 output 第二次 `putCanonical` 命中 `has` 去重
   （对象库只存一份）（验收 #3）。
4. 无 `traceObjectStore`：`outputHash` / `outputBytes` 仍在 payload，不抛错，tool 正常返回。
5. error 分支：`tool.responded` 不含 outputHash/outputBytes。
6. canonicalize 不支持的 output（如类实例）→ best-effort 降级，tool 正常返回、不抛。
7. 现有 replay 测试全过（验收 #4：`Replay.test.ts` 不回归）。

## 验收（本 issue 范围内）

- [ ] `ToolRespondedPayload` 增 `outputHash` / `outputBytes` / `artifactRefs?`（可选）
- [ ] 同一 tool input 两次 run 产生相同 `outputHash`（纯函数工具）
- [ ] `outputBytes` 正确反映 canonical stringify 后字节数
- [ ] 带 `traceObjectStore` 时大 output 按 hash 去重（对象库一份）
- [ ] 无 `traceObjectStore` 时不抛错、hash/bytes 仍在 payload
- [ ] error 分支不带这些字段；canonicalize 失败时 best-effort 降级
- [ ] replay 测试全过（CacheIndex / ReplayingIOPort 零改动）

## 显式 deferral（不在本 issue）

- 大 output 移出 jsonl（方案 B 的阈值卸载 + replay hydrate）→ 若 jsonl 体积成真问题再单开 issue；
- `artifactRefs` 的实际 emit（`object.created`）→ #37；
- relation.created / lineage 查询 → #38 / #41。

## 合并协调点

`feat/47-sub-agent-first-class-replay`（PR #49，本 spec 写时未合）新增了 `Milkie.buildMakeChildPort`，
其中也 `new RecordingIOPort(...)` 给子 agent 铸 port。本 issue 给 `RecordingIOPort` 构造函数加了
`traceObjectStore` 参数——**后合的那个 PR** 应顺手给子 port 也传 `this.traceObjectStore`，否则子
agent 的 tool output 不进对象库（功能正确，只是缺子 output 的去重）。

## Related

- ARCH.md §Implementation Status
- 依据：#23 region 内容可寻址（`AgentRuntime.buildRegionAddedPayload` / `persistRegionContent`、
  `ITraceObjectStore`、`hash.ts`）——本 issue 复用同一内容寻址与 best-effort 容错套路
- Blocks: #37（object.created + tool 关联）
