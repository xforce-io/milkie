# 续接宿主工具并核对未回复调用

| 项 | 值 |
|---|---|
| Issue | [#267](https://github.com/xforce-io/milkie/issues/267) |
| 状态 | Draft |
| 版本 | v1 |
| 产品基线 | [product.md](product.md) v1。确认来源是 Issue #267 正文 |
| 关联 | L1 S1.A1–S2.A3 |

## 1. 设计依据与技术目标

落实 L1 v1 的 S1.A1–S2.A3。续接时工具表按本轮重建，未登记名称在入口拒绝。未送达调用保持 `pending`，未核对完之前 `start` 返回 `context_busy`。核对只记录宿主给出的文本，并在没有剩余 `pending` 时释放该上下文的占用。

不做 CLI 会话补写，不做业务去重。

## 2. 现状与改动范围

现在的路径：

- `ExecutionClient.start` 用 `ExecutionStore.claim` 占用上下文。执行状态不是 `unknown` 时，`runExecution` 的 `finish` 才 `release`。宿主消失因此一直 `context_busy`（`src/execution/worker.ts` 的 `finish`，`src/execution/ExecutionClient.ts` 的 `start`）。
- 工具帧在 `openToolBridge` 里解析。合法调用先 `store.write('calls', …, pending)`，再 `onHost`。未登记名称直接 `rejected`（`src/execution/hostTools.ts`）。串行时整段处理都排在前一次处理函数之后，所以后一帧可能在宿主消失前还没有记录。
- 续接已经由 `hasExecuted` 选择 Grok `--resume` 或既有 Pi 会话文件（`src/execution/adapters.ts` 的 `cliCommand`、`assertNativeSession`）。

改动留在 `ExecutionClient`、`openToolBridge` 和调用记录状态。不改 CLI 命令拼装，不改 #266 的清单核对。

## 3. 总体架构与关键路径

接入进程可以退出。新的 `ExecutionClient` 指向同一 `dataDir`。`start` 先看该上下文还有没有 `pending` 调用，有则 `context_busy`，并且不占用新的执行。没有时沿用原来的 claim 和续接。

`reconcile` 把一条 `pending` 记为 `reconciled`。该上下文再无 `pending`，所属执行的 `stopped` 已是 true，且占用文件里的 runId 就是这条调用所属的执行时，才释放占用。未确认本地资源停止时占用继续留着。`start` 在取得占用之后、启动 worker 之前再查一次 `pending`；若这时已经有未核对调用，释放本轮占用并返回 `context_busy`。

工具帧一解析完就写记录，然后才进入串行队列。拒绝和参数失败仍立即回复，不进队列。

CLI 目录和授权表分开。`visibleHostTools` 把本上下文先前运行写过的工具规格并进本轮目录，授权 Map 仍只有本轮 `tools`。Grok 的 tools 文件和 Pi 的扩展、`--tools` 使用这个目录；`openToolBridge` 只接受本轮名称。同名以本轮规格为准。目录超过 32 个名称时拒绝启动。

## 4. 数据与状态契约

`calls/<callId>.json` 版本仍是 1。状态增加 `reconciled`。`pending` 没有 `output`。`reconciled` 有宿主写入的 `output`。

不新增第二种调用标识。占用文件仍是 `active/<contextId>.json`，内容是 runId。只释放与这条调用的 runId 相同的占用。

## 5. 接口与协作契约

`reconcile(callId, output)`：

- `output` 必须是非空字符串，最长 65536。否则 `invalid_request`。
- 没有这条调用，或状态不是 `pending`：`invalid_request`。
- 所属执行仍是 `starting` 或 `running`：`context_busy`。
- 成功后返回更新过的调用记录。

`start` 在 claim 之前检查 `pending`。错误文本仍只有错误码。`pendingToolCalls(contextId)` 返回该上下文的 `pending` 记录。上下文不存在是 `context_not_found`。

## 6. 运行与保障机制

核对不取消一条仍在运行的执行，也不向 CLI 发送工具结果。宿主消失后的 `unknown` 记录保持不变。处理函数看到的记录在它被调用前已经是 `pending`。

## 7. 迁移、发布与回滚

旧调用记录没有 `reconciled`。已有 `pending` 在升级后仍阻塞 `start`，直到宿主核对。没有数据迁移。回滚到不含 `reconcile` 的版本时，这些上下文会停在占用上，需要按 #266 的规则处理，不能靠新接口释放。

## 8. 测试与验证

功能文件：[19-agent-cli-host-tool-resume.md](../../../.agents/skills/verify-milkie/features/19-agent-cli-host-tool-resume.md)。

| 验收 | 机制 |
|---|---|
| S1.A1、S2.A1 | `npm run test:execution`。夹具证明拒绝、阻塞、核对和副作用次数。不证明真实模型会调用撤销工具 |
| S1.A2、S1.A3、S2.A2、S2.A3 | `MILKIE_LIVE_RESUME=1 ./node_modules/.bin/tsx tests/e2e/agent-cli-resume.live.ts`。真实 Grok 1.0.41 与 Pi 0.85.1 |

## 9. 技术风险与开放问题

已撤销名称仍留在 CLI 目录里，真实 CLI 才能把调用送到入口。模型仍不调用时，S1 的真实验收失败，不能改成只看模型没调用。核对后的下一轮若再次调用仍在授权表里的工具，由测试里的宿主根据已核对记录不再写副作用；milkie 不拦截第二次合法登记的调用。
