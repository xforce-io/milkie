# 子执行与并行协作

## 用户入口

- AgentConfig.subAgents；模型以子 Agent ID 调用工具；SDK AgentFactory.spawn

## 源码依据

- [src/runtime/AgentRuntime.ts](../../../../src/runtime/AgentRuntime.ts)
- [src/runtime/AgentFactory.ts](../../../../src/runtime/AgentFactory.ts)
- [src/types/store.ts](../../../../src/types/store.ts)
- [src/trace/types.ts](../../../../src/trace/types.ts)

## 驾驶路径与判定

| 路径 | 操作 | 可判定结果 |
|---|---|---|
| 正常 | 父 Agent 调用两个子 Agent，各自执行工具并返回 | 父子 runId 与事件归属可关联，父执行收到结果；并行安全条件可核对。 |
| 停止信息 | 子 Agent max_iterations=1，工具先登记产物，再停止；核对父工具响应、子结束事件和 children 记录 | 应保留停止原因、partial、artifacts、可用 checkpointId；当前 #261 失败。 |
| 错误/控制 | 配置不存在的子 Agent；传播父取消或较早 deadline | 错误可观察；子执行受父控制；不能仅凭工具 ok 推断子任务完整成功。 |

## 验证方法

下列命令从仓库根运行；是自动化覆盖入口，不代表上表所有路径已通过。具体执行记录见[首版核验](../references/initial-audit.md)。

```sh
./node_modules/.bin/jest --runInBand --runTestsByPath src/__tests__/AgentRuntime.test.ts src/__tests__/CausedByGraph.test.ts src/__tests__/runtimeDeadlineCancellation.test.ts src/__tests__/Replay.test.ts
```

- [src/__tests__/AgentRuntime.test.ts](../../../../src/__tests__/AgentRuntime.test.ts)
- [src/__tests__/CausedByGraph.test.ts](../../../../src/__tests__/CausedByGraph.test.ts)
- [src/__tests__/runtimeDeadlineCancellation.test.ts](../../../../src/__tests__/runtimeDeadlineCancellation.test.ts)
- [src/__tests__/Replay.test.ts](../../../../src/__tests__/Replay.test.ts)

## 已知缺口

#261：当前仅 return result.output，结束事件漏字段，预算停止子记录被映射为 success 且缺停止语义。

## 对应 Stories

- [docs/stories/s-007-inter-agent-parallel-code-review.md](../../../../docs/stories/s-007-inter-agent-parallel-code-review.md)
