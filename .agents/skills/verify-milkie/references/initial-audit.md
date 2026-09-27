# 首版核验记录

- 日期：2026-09-27。
- 产品基线：`7e2687f276134b26998fec5c7957de1bb98a2c92`。
- 范围：建立仓库内验证手册和功能地图；核对当前用户入口、旧 Stories 与已知缺陷；不修复产品代码，不修改线上 Issue。
- 本地分支从 `495d516` 快进到上述已有远端提交，随后编写本手册；没有创建分支、提交或推送。

## 地图完整性

| 对象 | 首版结果 | 证据与边界 |
|---|---|---|
| 功能文件 | 15 份，均被索引引用 | `features/README.md` 与 check_map.py 核对；其中 15 为规划/历史能力边界 |
| 主 SDK | 27 个公开 Milkie 方法均有主归属 | 对照 src/runtime/Milkie.ts；低层类导出按能力分组，不逐方法声称验证 |
| CLI | 11 个业务命令均有主归属 | 对照 src/cli/main.ts；不将 Commander 自动 help 算作业务能力 |
| HTTP | 14 个路由均有主归属 | 对照 src/cli/serve.ts，逐项区分 JSON/SSE、参数和状态码 |
| Stories | s-001 至 s-017 全部映射 | 按完整文件名与行为映射，不以编号猜测测试对应关系 |
| 手册发现 | keel-verify lookup 返回 found | 识别 verify-milkie、SKILL.md、features/README.md 和 15 份功能文件 |

辅助脚本已实际运行：check_map.py 正向检查通过，临时移除 complete 入口的负向检查能检出遗漏；probe-known-gaps.ts 通过严格 TypeScript 类型检查，运行得到下文五个已知失败。skill-creator 的 quick_validate.py 验证通过（本机 /usr/bin/python3 提供 PyYAML）。

入口对照检查只证明主入口没有遗漏；不是端到端产品行为全通过。`check_map.py` 按当前源码声明形态解析，重构注册方式时须一起维护解析器。

## 已执行验证

初次使用 Node v23.11.0；机器中 better-sqlite3 为 ABI 127，而 Node 23 要求 ABI 131。首轮 66 个套件：62 通过、4 失败，626 项测试中 612 通过、14 失败。失败集中在 CLI/HTTP/SQLite 存储路径。

改用本机已安装的 Node v22.22.3（仅改变本次进程 PATH，未重装依赖），内存 SQLite `select 1` 成功。重跑上述 4 个套件，并加测 FSMEngine、RunLifecycle 和迁移后的 s-011，共 7 个套件、100 项测试全部通过。按同一套件取最后一次结果汇总，**69 个不同套件、665 项测试全部通过**，无 skip。该数字仅对应所选自动化测试，不覆盖下文已知缺陷探针或所有手工路径。

复现选择范围：各功能文件“验证方法”中的全部去重测试，外加：

- [确定性基础 E2E](../../../../tests/e2e/deterministic.test.ts)
- [真实进程 HTTP/SSE E2E](../../../../tests/e2e/serve.e2e.test.ts)
- [执行记录 E2E](../../../../tests/e2e/s-002-inspect-a-completed-run.e2e.test.ts)
- [决策上下文 E2E](../../../../tests/e2e/s-003-explain-a-decision-with-context.e2e.test.ts)
- [IOPort 取消与 deadline E2E](../../../../tests/e2e/s-012-ioport-deadline-cancellation.e2e.test.ts)
- [模型失败回放 E2E](../../../../tests/e2e/s-013-llm-failure-replay.e2e.test.ts)
- [任务结果记录 E2E](../../../../tests/e2e/s-016-record-and-query-task-outcome.e2e.test.ts)
- [不可变结果确认 E2E](../../../../tests/e2e/s-017-immutable-task-outcome-finalization.e2e.test.ts)

| 功能 | 已核验证据 | 不能外推的范围 |
|---|---|---|
| 01 注册 | LoadManifest、parseConfig.models、standardAgentLayer、CliAgent | 未证明任意 frontmatter 字段都能加载 |
| 02 模型 | modelConnectionContract 本地 HTTP、complete、图像消息、错误与 serve 测试 | 未调用真实供应商 |
| 03 执行 | stopReason、交付物、取消/deadline、CLI、生命周期、单态槽位收集 | 受控停止不等于任务完成 |
| 04 恢复 | checkpoint、serve 中断恢复、SQLite 重建测试通过；缺陷探针失败见下表 | 返回 UUID 与连续恢复边界未满足契约 |
| 05 工具 | 控制工具协议、权限、覆盖、exec、AgentRuntime | 不构成任意命令安全或操作系统隔离证明 |
| 06 子执行 | AgentRuntime、因果图、控制传播、Replay | 结构化结果仍丢失 |
| 07 上下文 | ContextBudget、投影、区域生命周期、Skill 清单、Replay | token 估算不等于供应商账单；Skill A/B 非此范围 |
| 08 会话 | 历史、变量、投递、HTTP | HTTP 无变量 delete/TTL 入参 |
| 09 导入导出 | portable-session、serve | 原始包数据安全性不通过含真实用户数据的测试验证 |
| 10 记录 | summary、CLI inspect/execution/report、HTML 生成、区域引用 | 未用浏览器验收 HTML 交互 |
| 11 回放 | Replay、非确定性记录、失败重建、工作状态事件 | 未实现 fork/suite 的结论不因回放测试通过改变 |
| 12 证据与结果 | lineage、selfOnly/投递访问限制、TaskOutcome、Finalization | s-015 完整父执行进行中读取仍未证明 |
| 13 服务 | serve、CLI wiring、BroadcastingEventStore、独立进程 HTTP/SSE | SSE 字段完整性仍失败；fixture 不覆盖生产模型 |
| 14 存储 | Memory、SQLite 变量/重建、事件存储、ABI 提示、发布防覆盖 | Redis 与临时消费者安装未执行 |
| 15 规划边界 | 当前 CLI/Milkie 入口与 #175 源码、旧 Stories 对照 | 无运行 pass 结论 |

## 已知缺陷的独立证据

命令（仓库根）：

```sh
LOG_LEVEL=silent ./node_modules/.bin/tsx .agents/skills/verify-milkie/scripts/probe-known-gaps.ts
```

此基线退出码为 **1**，以下五项均为产品契约 `fail`，不是测试环境错误：

| 探针 | 期望 | 当前实际 |
|---|---|---|
| 259-returned-id | 返回 checkpointId 可直接用于 resume | 快照事件存在，但返回 UUID 无法解析，Checkpoint not found |
| 259-without-event-store | 无恢复存储时不冒充有可用快照 | 仍返回 checkpointId |
| 260-resume-boundaries | 初次执行加三次恢复均有可识别起止 | 仅 1 个 started 和 1 个 completed；恢复返回预算耗尽、正常停止、运行错误，记录却仍是旧终态 |
| 261-child-result | 父执行与子结束记录保留结构化停止信息 | 父工具响应只有文字，子记录 success，结束事件缺字段 |
| 261-http-result | HTTP 结束帧与持久化终态停止语义一致 | SSE 缺 stopReason、stopCode、partial、checkpointId、artifacts |

探针只覆盖上述具体断点；#259 的精确旧快照定位/持久化跨实例、#261 的取消/deadline 全矩阵尚未证明。#260 探针目前用已有 started/completed 事件计数；未来若采用合法的分段事件方案，须按新契约更新它，不能据此强制创建新 runId。

相关 Issue：[259](https://github.com/xforce-io/milkie/issues/259)、[260](https://github.com/xforce-io/milkie/issues/260)、[261](https://github.com/xforce-io/milkie/issues/261)。

## 旧文档漂移

- s-005 的 readiness 仍写等待非确定性日志，但当前有 RecordingIOPort、CacheIndex、Replay.nondet 实现与通过证据。
- s-011 原叙事是多态业务 FSM；#175 已移出核心，当前同名 E2E 是单态工具槽位收集。地图把旧能力和现能力明确分开。
- s-012/s-013 的 Story 与同编号测试不同义：批量回放/变体搜索不能借取消/失败回放测试宣称通过。
- s-004/s-014 旧 blocked 没反映已有显式对象关系和查询能力；s-015 的 selfOnly 工具基础不等于完整原场景。
- 7e2687f 已实现上下文预算；工具结果默认模型投影有界。旧 #257 背景只能作为历史依据。

## 未执行范围

完整浏览器交互、真实模型服务、Redis、发布物消费者安装及所有功能文件的全部手工变体均未执行。首版完成的是可维护的功能地图及上述有限验证，**不是全图回归通过**。

## 原始证据

本机证据保留于忽略目录 `test-output/feature-map-initial/`，不提交测试产物：

- `command.json`、`jest.log`、`jest-results.json`、`jest-exit-code.txt`：首轮命令及输出。
- `node22-command.json`、`jest-node22.log`、`jest-node22-results.json`、`jest-node22-exit-code.txt`：ABI 兼容后的复核。
- `combined-summary.json`：按套件去重汇总，原始逐项结果以上述两个 JSON 为准。
- `known-gaps.jsonl`：五项契约探针的实际结果。
- `map-check.txt`、`skill-validation.txt`、`handbook-lookup.json`：手册结构与发现证据。

换机器后可按功能文件重跑；这里的运行结论始终绑定本次日期和产品基线，不自动升级为新版本的通过记录。
