# Codex 全局警告兼容

版本 v0.1，修复契约；[Issue #281](https://github.com/xforce-io/milkie/issues/281)。L1 reuse：沿用[执行契约 v0.4](277-codex-sidecar-execution.md)的精确会话与工具转发约定，修正合法通知误中断，不新增产品取舍；用户端到端任务及指定 Sol 的范围覆盖该接入修复。L2 write：明确原生身份边界。采用项目十三节结构；§11对应本票细化验收。

## 1 背景

Inquiry3dbe328与Milkie98b5ec1真实Sol/high探针已认证、完成一次工具送达，随后warning(threadId:null)触发session_mismatch。Codex0.166.0生成的WarningNotification Schema允许目标为string/null或省略。`codex.ts::message`却将所有显式threadId与当前会话比较。确定性原入口回归亦复现：全局警告失败，省略和匹配目标两项通过。证据位置见[调查](../plan/281-codex-global-warning.md)。

## 2 名词解释

已有术语及全局警告见[名词表](../glossary.md)。

## 3 目标与非目标

目标：合法全局警告不改变执行归属；真实跨会话、错误轮次及无身份工具继续拒绝。非目标：改变模型、认证、工具权限、执行预算、重试策略或公共观测格式。

## 4 能力

宿主原有启动→工具请求→处理/回传→读取终态路径保留。成功为原生执行完成且停止确认；身份错误继续失败，全局警告不作为成功依据。

### 4.1 UI/UX

N/A：没有页面；沿用现有SDK/Sidecar结果。空输出、失败和停止未知语义不变。

## 5 思路与折衷

仅在原生warning通知（没有RPC请求id）、message为字符串且threadId为null时承认无具体会话目标。省略目标沿用既有行为；具体目标继续精确比较。放弃对所有通知泛用非空比较，避免将无身份工具或执行事件也放行；不记录警告正文，防止泄露原生内容。

## 6 架构

分层：原生app-server帧→Codex协议边界→既有工具分发/观测→执行记录→SDK/Sidecar。主路径：全局警告经限定解释后继续正常处理后续完成事件。失败路径：具体会话错配、工具缺身份或轮次错配在宿主效果前拒绝；此前已授权工具效果不回滚、不重放。原有turn/start回执、保存会话及进程停止校验均不变。

## 7 模块

N/A：仅改现有Codex帧边界；测试夹具增加相应协议输入。

## 8 API/CLI

N/A：不新增公共接口、字段或错误码。允许的原生输入为warning(message:string,threadId:null)，不把带id的RPC请求作为此通知。

## 9 边界

例外只作用threadId比较；turnId检查、宿主工具名称/Schema/去重/授权不变。带其它会话目标的warning仍拒绝；原生item/tool/call的null threadId及错误turnId仍拒绝。全局警告不能补造输出或更改终态。

## 10 迁移/兼容/回滚

无存储迁移。旧调用保持；回滚会恢复该合法警告的误拒绝，不重放已有工具，不将历史失败改成成功。

## 11 测试计划

本票S1/S2主要功能文件：[21-execution-sidecar](../../.agents/skills/verify-milkie/features/21-execution-sidecar.md)。没有页面，验证实际SDK/子进程协议。

| ID / 归属 | 路径 | 可判定结果 | 层级 |
|---|---|---|---|
| S1.A1 / S1 | SDK启动→一次宿主工具→global/omitted/targeted warning→读取完成 | 三种正常目标形态均succeeded、stopped；宿主效果1次，警告正文不写执行记录 | Integration |
| S1.A2 / S1 | 固定原生CLI接本机合成提供者→工具→全局警告→SDK终态 | 原生协议完整完成，或以精确注入证明该帧兼容；明确零真实模型，不冒充研究质量 | Integration |
| S2.A1 / S2 | foreign warning / null tool thread / wrong tool turn | 各session_mismatch且停止确认；无身份或错误轮次工具效果0，警告前已授权效果仅1次 | Integration |
| S2.A2 / S2 | 现有执行及Sidecar套件 | 续接、工具转发、取消、观测和协议边界无回归 | Integration / Unit |

真实Sol完整研究属于Inquiry #7，须锁定本候选依赖后另登记；本票不代替文章验收。

## 12 开放问题

没有未决产品问题；原生warning具体文本未采集，不作为修复条件。真实研究仍需独立证明。

## 13 关联

- https://github.com/xforce-io/inquiry/issues/7
- https://github.com/xforce-io/milkie/issues/277
- https://github.com/xforce-io/milkie/issues/278
