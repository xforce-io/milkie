# 执行、停止原因与交付物

## 用户入口

- SDK：`invoke`；AgentConfig.deliverables/onBudgetFinalize；AgentInvokeRequest.control/deliverables
- CLI：`agent run <agentId>`；HTTP：`POST /chat` 的执行结果另见 13

## 源码依据

- [src/runtime/Milkie.ts](../../../../src/runtime/Milkie.ts)
- [src/runtime/AgentRuntime.ts](../../../../src/runtime/AgentRuntime.ts)
- [src/fsm/FSMEngine.ts](../../../../src/fsm/FSMEngine.ts)
- [src/runtime/RunLifecycle.ts](../../../../src/runtime/RunLifecycle.ts)
- [src/runtime/RunControl.ts](../../../../src/runtime/RunControl.ts)
- [src/runtime/deliverables.ts](../../../../src/runtime/deliverables.ts)
- [src/cli/main.ts](../../../../src/cli/main.ts)

## 驾驶路径与判定

| 路径 | 操作 | 可判定结果 |
|---|---|---|
| 单态循环 | 用单个 llm 状态、工具和 WorkingMemory 完成槽位收集，再给出最终文本 | 工具轮继续循环，最终文本结束；不依赖 on/ctx.emit 业务跳转。 |
| 正常 | 确定性模型先调用工具登记产物，再返回最终文本；核对 SDK、CLI 和持久化结束记录 | model_stop 可读；已登记 artifacts 可定位；invoke 的交付清单覆盖 Agent 默认。 |
| 预算/取消 | 分别触发 max_iterations、deadline、AbortSignal；令 finalize 钩子失败 | 停止新增调度；区分 budget_exhausted/deadline/cancelled；钩子失败不改写预算原因。 |
| 不完整/错误 | 缺必选交付物；另让模型连接失败 | 缺必选项标记 partial；runtime_error 与预算耗尽可区分；不自动判定任务成功。 |

## 验证方法

下列命令从仓库根运行；是自动化覆盖入口，不代表上表所有路径已通过。具体执行记录见[首版核验](../references/initial-audit.md)。

```sh
./node_modules/.bin/jest --runInBand --runTestsByPath src/__tests__/stopReason.envelope.test.ts src/__tests__/deliverables.contract.test.ts src/__tests__/runtimeDeadlineCancellation.test.ts src/__tests__/AgentRuntime.control.test.ts src/__tests__/CliAgent.test.ts src/__tests__/FSMEngine.test.ts src/__tests__/RunLifecycle.test.ts tests/e2e/s-011-multi-state-fsm-intent-routing-and-slot-filling.e2e.test.ts
```

- [src/__tests__/stopReason.envelope.test.ts](../../../../src/__tests__/stopReason.envelope.test.ts)
- [src/__tests__/deliverables.contract.test.ts](../../../../src/__tests__/deliverables.contract.test.ts)
- [src/__tests__/runtimeDeadlineCancellation.test.ts](../../../../src/__tests__/runtimeDeadlineCancellation.test.ts)
- [src/__tests__/AgentRuntime.control.test.ts](../../../../src/__tests__/AgentRuntime.control.test.ts)
- [src/__tests__/CliAgent.test.ts](../../../../src/__tests__/CliAgent.test.ts)

## 已知缺口

核心已经移除多态业务 FSM；`fsm.states` 类型残留不代表 `on`/`ctx.emit` 业务拓扑仍可用。s-011 当前测试验证单态工具槽位收集。

checkpointId 仅在事件存储保存快照后返回；子执行和 HTTP 使用共享结果投影。completed 表示执行受控停止，不能独立证明任务完成。修复验收见[本次验收入口](../references/issues-259-261.md)。

## 对应 Stories

- [docs/stories/s-001-react-with-intra-agent-parallel-tools.md](../../../../docs/stories/s-001-react-with-intra-agent-parallel-tools.md)
- [docs/stories/s-009-multi-turn-with-tool-error-recovery.md](../../../../docs/stories/s-009-multi-turn-with-tool-error-recovery.md)

- [s-011 迁移后的确定性验证](../../../../tests/e2e/s-011-multi-state-fsm-intent-routing-and-slot-filling.e2e.test.ts)
