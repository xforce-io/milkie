# Agent 注册与配置

## 用户入口

- SDK：`loadManifest`、`loadAgentFile`、`registerAgent`、`getAgent`、`listAgents`、`loadStandardAgents`
- CLI：`agent list`；清单 `.milkie/agents.json`；Markdown Agent 文件

## 源码依据

- [src/runtime/Milkie.ts](../../../../src/runtime/Milkie.ts)
- [src/types/agent.ts](../../../../src/types/agent.ts)
- [src/cli/main.ts](../../../../src/cli/main.ts)

## 驾驶路径与判定

| 路径 | 操作 | 可判定结果 |
|---|---|---|
| 正常 | 加载临时清单和两个 Agent，再列举与读取 | SDK 返回对应 ID；CLI 逐行输出 JSON，source 为 manifest。 |
| 空 | 在没有上级清单的临时目录运行 agent list | 退出 0，stdout 为空。 |
| 错误 | 给出不存在文件、非法模型配置或未知 Agent ID | 记录具体拒绝点；不能把静默回退认作正确加载。 |

## 验证方法

下列命令从仓库根运行；是自动化覆盖入口，不代表上表所有路径已通过。具体执行记录见[首版核验](../references/initial-audit.md)。

```sh
./node_modules/.bin/jest --runInBand --runTestsByPath src/__tests__/LoadManifest.test.ts src/__tests__/parseConfig.models.test.ts src/__tests__/standardAgentLayer.test.ts src/__tests__/CliAgent.test.ts
```

- [src/__tests__/LoadManifest.test.ts](../../../../src/__tests__/LoadManifest.test.ts)
- [src/__tests__/parseConfig.models.test.ts](../../../../src/__tests__/parseConfig.models.test.ts)
- [src/__tests__/standardAgentLayer.test.ts](../../../../src/__tests__/standardAgentLayer.test.ts)
- [src/__tests__/CliAgent.test.ts](../../../../src/__tests__/CliAgent.test.ts)

## 已知缺口

`registerAgent`/`loadAgentFile` 的同 ID 注册覆盖已有值，不是重复注册拒绝。Markdown parseConfig 只解析源码列出的字段；此基线未将 contextBudget、deliverables 从 frontmatter 转成配置，不能把 SDK 配置面自动外推到文件加载。
