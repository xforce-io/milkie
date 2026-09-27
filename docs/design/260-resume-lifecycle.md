# 恢复执行具有完整边界

状态：Draft；Issue：[260](https://github.com/xforce-io/milkie/issues/260)；分支：`feat/260-resume-lifecycle`。

## 1 背景

resume 当前复用旧 runId 且不记录起止，恢复后的活动追加在旧终态之后。

## 2 名词解释

沿用 [名词表](../glossary.md)。`resumedFromCheckpointId` 为开始事件中本次恢复所采用的确定快照 ID。

## 3 目标与非目标

目标：每次已开始的恢复拥有唯一边界、来源和与返回一致的终态。非目标：改动任务结果模型、自动补写历史缺失边界、改变 #259 的标识契约。

## 4 能力

恢复执行生成新 runId，contextId 保持不变；来源为 previousRunId 与 resumedFromCheckpointId。连续恢复和错误路径均可逐次对账。

### 4.1 UI/UX

无页面。SDK/CLI/HTTP 返回本次新 runId；查询旧 run 不混入后续活动。启动前非法参数、找不到快照不制造虚假执行记录。

## 5 思路与折衷

采用每次恢复新 run，复用现有单 run 单终态契约与会话前驱链。放弃同 run 分段，需要更改所有 first-start/terminal 消费方且易混淆旧记录。代价：依赖 resume 返回旧 runId 的消费者须改为读实际返回值。

## 6 架构

Milkie 解析快照→建立新执行→恢复状态→记录 started→运行→记录 completed。失败发生在执行内仍记录对应错误终态；启动前失败不开始执行。历史、导出及回放从来源关联读取前次状态。

## 7 模块

Milkie 恢复流程管理边界；RecordingIOPort 保存起止；RunSnapshot 投影恢复来源，replay 从确切前驱快照恢复后再消费新 run 的记录。

## 8 API/CLI

AgentRunStartedPayload 增加可选 resumedFromCheckpointId；previousRunId 沿用现有会话链。AgentResult.agentRunId 变为新 runId；接口参数和已有 context 别名不变。

## 9 边界

至少两次连续恢复、预算停止、正常停止与运行错误均独立。旧终态不被覆盖；来源快照丢失时回放明确失败，不从空状态或 latest 猜测。存储写入失败按现有异常机制暴露，不承诺存储故障下记录成功。

## 10 迁移/兼容/回滚

新增可选事件字段，旧记录保持可读。原本缺边界的历史恢复不补写。回滚会恢复旧混合记录行为，不改变已有新记录。

## 11 测试计划

- E2E：原快照恢复→新 run 活动→新终态→核对来源与原 run 不变。S1 → `.agents/skills/verify-milkie/features/04-interrupt-resume.md`。
- E2E：连续恢复预算/正常/错误；SDK、CLI context 恢复、HTTP resume 和状态查询均验证。S2 → 同一功能文件。
- Integration：恢复后的历史、会话导入导出、严格回放、广播归属；Unit：来源字段投影。

## 12 开放问题

无阻塞问题。前驱链表示恢复来源；从旧快照恢复形成来源分支，不改写旧链。

## 13 关联

#259；#244；[259 设计](259-checkpoint-resume.md)。PR：尚未创建。
