# 上下文预算、工具结果与 Skill 生命周期

## 用户入口

- AgentConfig.contextBudget；ToolDefinition.resultStrategy；AgentConfig.skills/skillInstructions
- 模型工具：`skill_list`、`skill_request`；SDK ContextRegions、assemble、WorkingMemory

## 源码依据

- [src/context/budget.ts](../../../../src/context/budget.ts)
- [src/context/assemble.ts](../../../../src/context/assemble.ts)
- [src/context/lifecycleEngine.ts](../../../../src/context/lifecycleEngine.ts)
- [src/runtime/toolResultStrategy.ts](../../../../src/runtime/toolResultStrategy.ts)
- [src/tools/system.ts](../../../../src/tools/system.ts)
- [src/runtime/AgentRuntime.ts](../../../../src/runtime/AgentRuntime.ts)

## 驾驶路径与判定

| 路径 | 操作 | 可判定结果 |
|---|---|---|
| 预算 | 分别累积大工具输出、长历史、WorkingMemory 和外部投递；检查每次模型请求和计量 | 配置的总量与分区上限受控；原始证据保留；模型投影不破坏工具调用配对。 |
| 必需区域错误 | 给 control/currentTurn 太小预算或非法配置 | 模型请求前明确失败，不静默删除当前用户原话。 |
| Skill | 列举清单，请求 turn 与 session Skill，跨上下文 epoch 和下一轮核对 | 只在声明边界生效；turn 结束释放，session 跨轮保留；未知 Skill 不伪装成功加载。 |
| 默认工具结果 | 未声明 resultStrategy 的工具返回大文本，再严格回放 | 默认模型投影有界，原始结果仍可追溯；回放保持同一投影。 |

## 验证方法

下列命令从仓库根运行；是自动化覆盖入口，不代表上表所有路径已通过。具体执行记录见[首版核验](../references/initial-audit.md)。

```sh
./node_modules/.bin/jest --runInBand --runTestsByPath src/__tests__/ContextBudget.test.ts src/__tests__/AgentRuntime.toolResultStrategy.test.ts src/__tests__/ContextRegions.test.ts src/__tests__/lifecycleEngine.test.ts src/__tests__/skillListManifest.test.ts src/__tests__/Replay.test.ts
```

- [src/__tests__/ContextBudget.test.ts](../../../../src/__tests__/ContextBudget.test.ts)
- [src/__tests__/AgentRuntime.toolResultStrategy.test.ts](../../../../src/__tests__/AgentRuntime.toolResultStrategy.test.ts)
- [src/__tests__/ContextRegions.test.ts](../../../../src/__tests__/ContextRegions.test.ts)
- [src/__tests__/lifecycleEngine.test.ts](../../../../src/__tests__/lifecycleEngine.test.ts)
- [src/__tests__/skillListManifest.test.ts](../../../../src/__tests__/skillListManifest.test.ts)
- [src/__tests__/Replay.test.ts](../../../../src/__tests__/Replay.test.ts)

## 已知缺口

7e2687f 已有上下文预算；估算器按 UTF-8 字节保守计量，不等于供应商账单 token。s-010 的 Skill 生命周期已有实现，自动 A/B 搜索见 15。

## 对应 Stories

- [docs/stories/s-010-skill-versioned-load-and-ab-experiment.md](../../../../docs/stories/s-010-skill-versioned-load-and-ab-experiment.md)
