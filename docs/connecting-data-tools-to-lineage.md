# 数据工具接入 lineage 指引

面向**工具作者**:你写了一个会从外部召回/读取资料的工具(检索、读文件、查库、抓网页……),如何让它召回的资料能被 agent 用 `cite` 溯源。

配套阅读:[`lineage-taxonomy.md`](./lineage-taxonomy.md)(object/relation 的受控词汇表)、[`lineage-lifecycle.md`](./lineage-lifecycle.md)(从取证到落账的完整时序与 lazy-promote)。

## 何时该接

判断标准只有一条:**这个工具是否把"外部的、不可凭空重现的资料"带进了 agent 的视野,且这份资料可能成为答案里某个论断的依据?**

| 工具类型 | 例子 | 接 lineage? |
|----------|------|-------------|
| 数据召回 / 读取 | recall、read_file、grep、query_db、fetch_url | **要** —— 这是溯源的源头 |
| 副作用型 | mkdir、write_file、git_commit | 不要 —— 没有"被引用的事实" |
| 纯计算 / 控制 | calculator、子 agent 调度、create_plan | 不要 |

`run_command` 比较特殊:它既能跑副作用命令又能 cat 文件,所以用 lazy 兜底——只要有 stdout 就 register,不被引用就不落账。

## 怎么接:一行 `citeable`

用 `src/tools/lineage.ts` 导出的 `citeable` helper。它把样板和三个易错点一次封死:**lazy 注册**、**objectId 放结果最前**、**没接 lineage sink 时不污染结果**。

```ts
import { citeable } from '../tools/lineage.js'

handler: async (input, ctx) => {
  const results = await myRetriever.search(input.query)   // 你现有的召回逻辑
  // 批量:每条结果各 mint 一个可 cite 的 object
  const hits = results.map((r) =>
    citeable(ctx, r.text, { text: r.text, source: r.docId }, { meta: { source: r.docId, score: r.score } }),
  )
  return { hits }
}
```

`citeable(ctx, content, result, opts?)`:

- `content` —— 用来算 hash 的**原文**(内容寻址 → 同样内容跨 run 去重)。
- `result` —— 要返回给 agent 的结果体;`objectId` 会被 spread 到它**最前面**。
- `opts.type` —— object 类型,默认 `'passage'`(见下)。
- `opts.meta` —— 定位信息(doc id、行号、url、score 等),会进 `object.created` payload。

接入后**零额外接线**:agent 在 state 的 `tools` allowlist 里只要有 `cite`,就能引用这些 objectId,RecordingIOPort 在 `tool.responded` 之后自动落账。记得把工具名和 `cite` 一起加进用到它的 state 的 `tools`。

## objectType 怎么选

- 一般召回的一段文本 → 直接用核心类型 `'passage'`,跨 run 查询能按核心类型聚合。
- 想区分来源便于后续分析 → 用 `namespace:kind`(如 `'shell:stdout'`、`'rag:chunk'`),约定见 [`lineage-taxonomy.md`](./lineage-taxonomy.md)。

## 工具 description 的话术模板

description 要告诉 agent"每条结果带 objectId,用 cite 引用,别在散文里手写来源"。照抄这个骨架:

> Returns `{ hits: [{ objectId, text, source }] }`. Each hit carries an `objectId` —
> to source a claim from a retrieved passage, pass that objectId to the `cite` tool.
> Never write provenance like "(doc:...)" in prose; cite the objectId instead.

## 为什么 objectId 必须在最前

result-truncation 策略可能截掉结果尾部。objectId 在尾部 → agent 看不见 → 没法 cite。`citeable` 已经替你保证它在首位,但如果你手写返回结构,务必把 objectId 放第一个键。

## 跨语言:外部语言只做数据源

召回服务是 Python/Go/Java 写的没关系。让 milkie 这边的 handler 当**瘦适配层**:通过 subprocess / HTTP / RPC 调外部服务,外部只返回**原始数据**(text + 定位信息),`citeable`、hash、objectId 全在 **TS 侧** mint。这样跨语言**零对齐成本**——内容寻址只在一处发生。`run_command`(`src/tools/exec.ts`)就是这个模式的现成证明:它 spawn 任意语言的子进程,只对 stdout 做物化。

只有当外部系统要**独立持久化溯源、跨服务共享同一个 objectId** 时,才需要在那门语言里复刻 objectId 的内容寻址算法(canonical JSON + sha256,见 `src/trace/hash.ts`)。那是另一个量级的工程,不要提前做。
