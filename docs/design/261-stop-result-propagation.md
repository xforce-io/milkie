# 子执行与服务出口完整保留停止结果

状态：Draft；Issue：[261](https://github.com/xforce-io/milkie/issues/261)；分支：`feat/261-stop-result-propagation`。

## 1 背景

顶层 SDK/记录已有 #244 字段，子执行工具返回及服务结束帧漏掉这些字段。

## 2 名词解释

沿用 [名词表](../glossary.md) 的 AgentResult、stopReason、partial、artifacts。

## 3 目标与非目标

目标：父执行、子执行记录、执行事件及 HTTP 结束帧均保留停止原因和部分结果。非目标：把预算耗尽改为任务失败、自动判定任务完成、内联产物内容、改变 summary API。

## 4 能力

S1：子 Agent 工具返回完整 AgentResult；子结束事件、agent.returned 和 children 记录具有同一停止信息。S2：服务结束帧投影顶层 SDK 结果，保留输出别名和 runId。

### 4.1 UI/UX

无页面。模型看到可解析的子结果对象，output 保留原文字；服务消费者保留原 status/output/runId 读取方式。缺少执行结果的启动前错误仍为 error 帧，不编造 checkpoint/artifacts。

## 5 思路与折衷

用一个共享结果到结束事件的投影，避免各出口独立列字段导致遗漏。放弃把 stopReason 合并进 status，保留 completed/success 的执行层语义并附加独立字段。子工具 output 从字符串变为对象是有意契约调整；原文字位于 result.output。

## 6 架构

AgentRuntime 产生 AgentResult→子工具返回结果，同时投影到子结束事件、父 agent.returned 和 children 记录。Milkie invoke/resume 与 serve 复用投影。预算/正常/取消/中断/deadline/运行错误均从同一结果读取，不二次判断停止原因。

## 7 模块

共享结果投影位于 runtime；AgentRuntime 管子出口，Milkie 管持久化终态，serve 管 SSE。类型允许旧事件缺字段，新增事件完整填写。

## 8 API/CLI

子工具输出为 AgentResult。AgentReturnedPayload 和 ChildAgentRecord 新增可选 stopReason、stopCode、partial、artifacts、checkpointId、error 等字段；HTTP 结束帧保留既有 status/output/runId，并增加相同停止字段与 contextId。工具传输 ok 不等于子任务完整完成。

## 9 边界

只转发存在的 checkpointId；错误信息不丢；产物仍是引用。启动前异常或存储异常不强行补造完整 AgentResult。旧 consumers 读取文本需改为 result.output；默认模型投影仍服从上下文预算，事件保存完整原始结果。

## 10 迁移/兼容/回滚

旧事件保持可读，新字段可选；子工具返回形状变化需通知集成方。回滚后出口再次丢字段，不能继续宣称结果完整。

## 11 测试计划

- E2E：真实父子执行覆盖预算、取消、运行错误、正常停止，并补 deadline/interrupted；比较工具返回、子结束事件、父返回事件和 children。S1 → `.agents/skills/verify-milkie/features/06-subagents.md`。
- E2E：真实本地 HTTP /chat 和 /resume，比较 SDK 返回、持久化终态与唯一 SSE 结束帧；覆盖同类停止结果。S2 → `.agents/skills/verify-milkie/features/13-http-service.md`。
- Integration：子执行严格回放、上下文模型投影、运行错误原有 error 帧兼容；Unit：共享字段投影。

## 12 开放问题

无阻塞问题。外部存储故障不属于保证“已成功持久化”的场景。

## 13 关联

#244；#259；#260；[260 设计](260-resume-lifecycle.md)。PR：尚未创建。
