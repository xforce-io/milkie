# 工具调用、计划与命令执行

## 用户入口

- SDK：`registerTool`；ToolDefinition.handler/inputSchema/resultStrategy/parallelSafe；AgentConfig.builtinTools
- 模型工具：`think`、`create_plan`、`update_step`、`run_command`

## 源码依据

- [src/tools/ToolRegistry.ts](../../../../src/tools/ToolRegistry.ts)
- [src/tools/builtinTools.ts](../../../../src/tools/builtinTools.ts)
- [src/tools/cognitive.ts](../../../../src/tools/cognitive.ts)
- [src/tools/exec.ts](../../../../src/tools/exec.ts)
- [src/runtime/toolProtocol.ts](../../../../src/runtime/toolProtocol.ts)
- [src/runtime/AgentRuntime.ts](../../../../src/runtime/AgentRuntime.ts)

## 驾驶路径与判定

| 路径 | 操作 | 可判定结果 |
|---|---|---|
| 正常 | 模型调用自定义工具，创建计划并更新一步；在临时目录执行无副作用命令 | 工具响应可观察；计划可查询；命令 stdout/stderr/退出信息可读取；并行只对允许的工具生效。 |
| 协议错误 | 发送畸形 JSON、字段错误和 handler 业务异常，随后模型改参重试 | 不可修复或校验失败的参数不执行 handler；控制工具允许有界修复后通过校验再执行；协议错误与执行错误可区分；单次工具错误不默认终止整次执行。 |
| 边界 | builtinTools.allow=[]，以及未知/重复名称；子 Agent 申请更大集合 | 空集合不暴露内建工具；非法声明拒绝；子 Agent 不扩大父级权限；自定义工具另行注册。 |

## 验证方法

下列命令从仓库根运行；是自动化覆盖入口，不代表上表所有路径已通过。具体执行记录见[首版核验](../references/initial-audit.md)。

```sh
./node_modules/.bin/jest --runInBand --runTestsByPath src/__tests__/controlToolProtocol.test.ts src/__tests__/builtinToolPolicy.test.ts src/__tests__/toolOverrideContract.test.ts src/__tests__/execTools.test.ts src/__tests__/AgentRuntime.test.ts
```

- [src/__tests__/controlToolProtocol.test.ts](../../../../src/__tests__/controlToolProtocol.test.ts)
- [src/__tests__/builtinToolPolicy.test.ts](../../../../src/__tests__/builtinToolPolicy.test.ts)
- [src/__tests__/toolOverrideContract.test.ts](../../../../src/__tests__/toolOverrideContract.test.ts)
- [src/__tests__/execTools.test.ts](../../../../src/__tests__/execTools.test.ts)
- [src/__tests__/AgentRuntime.test.ts](../../../../src/__tests__/AgentRuntime.test.ts)

## 已知缺口

run_command 会执行真实子进程，驾驶仅在测试临时目录使用无害命令。不要将测试工具权限等同于操作系统沙箱。

## 对应 Stories

- [docs/stories/s-001-react-with-intra-agent-parallel-tools.md](../../../../docs/stories/s-001-react-with-intra-agent-parallel-tools.md)
- [docs/stories/s-009-multi-turn-with-tool-error-recovery.md](../../../../docs/stories/s-009-multi-turn-with-tool-error-recovery.md)
