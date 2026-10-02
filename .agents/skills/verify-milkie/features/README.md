# Milkie 功能地图

首版基线：`7e2687f276134b26998fec5c7957de1bb98a2c92`；建立日期：2026-09-27。

这是当前用户能力与验证方法的目录。入口手册见 [SKILL.md](../SKILL.md)，术语以 [docs/glossary.md](../../../../docs/glossary.md) 为准。Stories 保留场景意图；本地图不改写它们，也不继承其 active 标记作为通过证据。

首版覆盖主 SDK、全部已注册 CLI/HTTP 入口，并按能力收录配置、模型工具、存储和报告。低层导出类按所属能力核对，不声称逐个内部方法均已驾驶。源码核对和实际验证结果分开，详见[首版核验](../references/initial-audit.md)。

## 功能目录

| 文件 | 用户能力 | 当前注意事项 |
|---|---|---|
| [01-registration](01-registration.md) | Agent 注册与配置 | 已有入口；通过范围以核验记录为准 |
| [02-model-connection](02-model-connection.md) | 模型连接与单次调用 | 已有入口；通过范围以核验记录为准 |
| [03-run-control](03-run-control.md) | 执行、停止原因与交付物 | 已有入口；通过范围以核验记录为准 |
| [04-interrupt-resume](04-interrupt-resume.md) | 中断、恢复与执行边界 | #259/#260 修复候选；按新 runId 验证 |
| [05-tools](05-tools.md) | 工具调用、计划与命令执行 | 已有入口；通过范围以核验记录为准 |
| [06-subagents](06-subagents.md) | 子执行与并行协作 | #261 修复候选；结构化结果 |
| [07-context-skills](07-context-skills.md) | 上下文预算、工具结果与 Skill 生命周期 | 已有入口；通过范围以核验记录为准 |
| [08-session-context](08-session-context.md) | 多轮会话、变量与外部投递 | 已有入口；通过范围以核验记录为准 |
| [09-session-portability](09-session-portability.md) | 会话导入与导出 | 已有入口；通过范围以核验记录为准 |
| [10-trace-inspection](10-trace-inspection.md) | 执行记录、汇总与 HTML 报告 | HTML 交互待浏览器验收 |
| [11-replay](11-replay.md) | 确定性回放与失败重建 | 已有入口；通过范围以核验记录为准 |
| [12-lineage-outcome](12-lineage-outcome.md) | 证据查询与任务结果 | 已有入口；通过范围以核验记录为准 |
| [13-http-service](13-http-service.md) | HTTP 服务与流式执行 | #261 修复候选；结构化结果 |
| [14-persistence](14-persistence.md) | 存储、重启与发布物 | Redis/消费者安装待验证 |
| [15-planned-boundaries](15-planned-boundaries.md) | 规划能力与旧场景边界 | 规划或部分实现 |

| [16-agent-cli-execution](16-agent-cli-execution.md) | 外部 CLI 执行与原生会话续接 | #263 候选；按 S1–S5 验证真实 CLI |
| [17-agent-cli-dedicated-storage](17-agent-cli-dedicated-storage.md) | CLI 专用配置目录与会话目录 | #265；缺失时启动前失败 |
| [18-agent-cli-host-tools](18-agent-cli-host-tools.md) | CLI 宿主工具、能力关停与调用记录 | #266；真实 CLI 验收 |
| [19-agent-cli-host-tool-resume](19-agent-cli-host-tool-resume.md) | 续接时重装宿主工具并核对未回复调用 | #267；真实 CLI 验收 |

## 主入口逐项归属

此表用于源码漂移检查；每个入口仅指定一个主归属，跨能力路径通过功能文件说明。

| 类型 | 入口 | 功能文件 |
|---|---|---|
| SDK | `ExecutionClient.capabilities` | [16-agent-cli-execution](16-agent-cli-execution.md) |
| SDK | `ExecutionClient.createContext` | [16-agent-cli-execution](16-agent-cli-execution.md) |
| SDK | `ExecutionClient.getContext` | [16-agent-cli-execution](16-agent-cli-execution.md) |
| SDK | `ExecutionClient.start` | [16-agent-cli-execution](16-agent-cli-execution.md) |
| SDK | `ExecutionClient.query` | [16-agent-cli-execution](16-agent-cli-execution.md) |
| SDK | `ExecutionClient.wait` | [16-agent-cli-execution](16-agent-cli-execution.md) |
| SDK | `ExecutionClient.cancel` | [16-agent-cli-execution](16-agent-cli-execution.md) |
| SDK | `ExecutionClient.toolCall` | [18-agent-cli-host-tools](18-agent-cli-host-tools.md) |
| SDK | `ExecutionClient.reconcile` | [19-agent-cli-host-tool-resume](19-agent-cli-host-tool-resume.md) |
| SDK | `loadManifest` | [01-registration](01-registration.md) |
| SDK | `loadAgentFile` | [01-registration](01-registration.md) |
| SDK | `registerAgent` | [01-registration](01-registration.md) |
| SDK | `getAgent` | [01-registration](01-registration.md) |
| SDK | `listAgents` | [01-registration](01-registration.md) |
| SDK | `loadStandardAgents` | [01-registration](01-registration.md) |
| SDK | `registerTool` | [05-tools](05-tools.md) |
| SDK | `complete` | [02-model-connection](02-model-connection.md) |
| SDK | `invoke` | [03-run-control](03-run-control.md) |
| SDK | `resume` | [04-interrupt-resume](04-interrupt-resume.md) |
| SDK | `interrupt` | [04-interrupt-resume](04-interrupt-resume.md) |
| SDK | `getContextState` | [04-interrupt-resume](04-interrupt-resume.md) |
| SDK | `replay` | [11-replay](11-replay.md) |
| SDK | `exportSession` | [09-session-portability](09-session-portability.md) |
| SDK | `importSession` | [09-session-portability](09-session-portability.md) |
| SDK | `getSessionHistory` | [08-session-context](08-session-context.md) |
| SDK | `getContextVar` | [08-session-context](08-session-context.md) |
| SDK | `setContextVar` | [08-session-context](08-session-context.md) |
| SDK | `deleteContextVar` | [08-session-context](08-session-context.md) |
| SDK | `listContextVars` | [08-session-context](08-session-context.md) |
| SDK | `attachProjection` | [08-session-context](08-session-context.md) |
| SDK | `listContextProjections` | [08-session-context](08-session-context.md) |
| SDK | `recordTaskOutcome` | [12-lineage-outcome](12-lineage-outcome.md) |
| SDK | `getTaskOutcome` | [12-lineage-outcome](12-lineage-outcome.md) |
| SDK | `finalizeTaskOutcome` | [12-lineage-outcome](12-lineage-outcome.md) |
| SDK | `getFinalTaskOutcome` | [12-lineage-outcome](12-lineage-outcome.md) |
| SDK | `getRunSummary` | [10-trace-inspection](10-trace-inspection.md) |
| CLI | `agent list` | [01-registration](01-registration.md) |
| CLI | `agent run <agentId>` | [03-run-control](03-run-control.md) |
| CLI | `agent resume <contextId>` | [04-interrupt-resume](04-interrupt-resume.md) |
| CLI | `agent interrupt <contextId>` | [04-interrupt-resume](04-interrupt-resume.md) |
| CLI | `trace inspect <runId>` | [10-trace-inspection](10-trace-inspection.md) |
| CLI | `trace summary <runId>` | [10-trace-inspection](10-trace-inspection.md) |
| CLI | `trace render-html` | [10-trace-inspection](10-trace-inspection.md) |
| CLI | `trace report <runId>` | [10-trace-inspection](10-trace-inspection.md) |
| CLI | `trace execution <runId>` | [10-trace-inspection](10-trace-inspection.md) |
| CLI | `trace replay <runId>` | [11-replay](11-replay.md) |
| CLI | `serve` | [13-http-service](13-http-service.md) |
| HTTP | `GET /health` | [13-http-service](13-http-service.md) |
| HTTP | `POST /chat` | [13-http-service](13-http-service.md) |
| HTTP | `POST /interrupt` | [04-interrupt-resume](04-interrupt-resume.md) |
| HTTP | `POST /resume` | [04-interrupt-resume](04-interrupt-resume.md) |
| HTTP | `POST /context/set` | [08-session-context](08-session-context.md) |
| HTTP | `POST /context/get` | [08-session-context](08-session-context.md) |
| HTTP | `POST /context/list` | [08-session-context](08-session-context.md) |
| HTTP | `POST /context/state` | [04-interrupt-resume](04-interrupt-resume.md) |
| HTTP | `POST /projection/attach` | [08-session-context](08-session-context.md) |
| HTTP | `POST /projection/list` | [08-session-context](08-session-context.md) |
| HTTP | `POST /llm` | [02-model-connection](02-model-connection.md) |
| HTTP | `POST /session/history` | [08-session-context](08-session-context.md) |
| HTTP | `POST /session/export` | [09-session-portability](09-session-portability.md) |
| HTTP | `POST /session/import` | [09-session-portability](09-session-portability.md) |

## Stories 迁移对照

| Story | 功能文件 | 判断边界 |
|---|---|---|
| [s-001](../../../../docs/stories/s-001-react-with-intra-agent-parallel-tools.md) | [03-run-control](03-run-control.md)、[05-tools](05-tools.md) | 场景验收仍以原 Story 为准；本地图提供入口及验证方法 |
| [s-002](../../../../docs/stories/s-002-inspect-a-completed-run.md) | [10-trace-inspection](10-trace-inspection.md) | 场景验收仍以原 Story 为准；本地图提供入口及验证方法 |
| [s-003](../../../../docs/stories/s-003-explain-a-decision-with-context.md) | [10-trace-inspection](10-trace-inspection.md) | 场景验收仍以原 Story 为准；本地图提供入口及验证方法 |
| [s-004](../../../../docs/stories/s-004-lineage-from-artifact-to-source.md) | [12-lineage-outcome](12-lineage-outcome.md) | 证据登记/关系查询已有实现，完整场景待对照 |
| [s-005](../../../../docs/stories/s-005-deterministic-replay.md) | [11-replay](11-replay.md) | 非确定性日志已有实现，旧 readiness 已滞后 |
| [s-006](../../../../docs/stories/s-006-fork-at-event-for-what-if.md) | [15-planned-boundaries](15-planned-boundaries.md) | 场景验收仍以原 Story 为准；本地图提供入口及验证方法 |
| [s-007](../../../../docs/stories/s-007-inter-agent-parallel-code-review.md) | [06-subagents](06-subagents.md) | 场景验收仍以原 Story 为准；本地图提供入口及验证方法 |
| [s-008](../../../../docs/stories/s-008-long-task-interrupt-and-resume.md) | [04-interrupt-resume](04-interrupt-resume.md) | 修复精确快照与独立恢复边界；按当前验收重跑 |
| [s-009](../../../../docs/stories/s-009-multi-turn-with-tool-error-recovery.md) | [03-run-control](03-run-control.md)、[05-tools](05-tools.md)、[08-session-context](08-session-context.md) | 场景验收仍以原 Story 为准；本地图提供入口及验证方法 |
| [s-010](../../../../docs/stories/s-010-skill-versioned-load-and-ab-experiment.md) | [07-context-skills](07-context-skills.md)、[15-planned-boundaries](15-planned-boundaries.md) | Skill 已实现；自动 A/B 不据此宣称实现 |
| [s-011](../../../../docs/stories/s-011-multi-state-fsm-intent-routing-and-slot-filling.md) | [03-run-control](03-run-control.md)、[15-planned-boundaries](15-planned-boundaries.md) | 原多态业务 FSM 已移出核心；现测试是单态槽位收集 |
| [s-012](../../../../docs/stories/s-012-batch-replay-suite-and-classify-divergences.md) | [15-planned-boundaries](15-planned-boundaries.md) | 批量回放 Story；不要与同编号 deadline 测试混淆 |
| [s-013](../../../../docs/stories/s-013-variant-search-with-bounded-cost.md) | [15-planned-boundaries](15-planned-boundaries.md) | 变体搜索 Story；不要与同编号失败回放测试混淆 |
| [s-014](../../../../docs/stories/s-014-reverse-reference-lineage-query.md) | [12-lineage-outcome](12-lineage-outcome.md) | 显式关系反查已有代码，旧 blocked 不能照搬 |
| [s-015](../../../../docs/stories/s-015-subagent-reads-parent-trace-runtime.md) | [12-lineage-outcome](12-lineage-outcome.md)、[15-planned-boundaries](15-planned-boundaries.md) | selfOnly 工具不证明完整父执行进行中读取 |
| [s-016](../../../../docs/stories/s-016-record-and-query-task-outcome.md) | [12-lineage-outcome](12-lineage-outcome.md) | 场景验收仍以原 Story 为准；本地图提供入口及验证方法 |
| [s-017](../../../../docs/stories/s-017-immutable-task-outcome-finalization.md) | [12-lineage-outcome](12-lineage-outcome.md) | 场景验收仍以原 Story 为准；本地图提供入口及验证方法 |

## 维护规则

新增/删除公开入口时，同时更新入口表和对应功能文件。每次功能验证记录入口、路径、结果、代码基线和证据位置；已知失败保留为 fail，不通过放宽文案改成 pass。旧 Story 状态有漂移时，在此记出处，另行修正文档事实源。

运行 `python3 .agents/skills/verify-milkie/scripts/check_map.py` 可检查索引、相对文件引用及主 SDK/CLI/HTTP 入口遗漏；它不验证产品行为，也不覆盖动态注册扩展或全部低层 SDK 导出。

#259–#261 的设计与验收对应关系见[本次验收入口](../references/issues-259-261.md)。首版核验保留历史结果，不改写成新版本证明。
