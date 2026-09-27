# 返回的 checkpointId 可直接恢复

状态：Draft；Issue：[259](https://github.com/xforce-io/milkie/issues/259)；分支：`feat/259-checkpoint-resume`。

## 1 背景

#244 已约定 checkpointId 可用于 resume，但当前 UUID 无解析路由，且无事件存储时仍返回该字段。

## 2 名词解释

沿用 [名词表](../glossary.md) 的 checkpointId、resume、AgentResult。

## 3 目标与非目标

目标：返回标识精确定位已保存快照，持久化事件存储重建后仍可用。非目标：恢复执行边界、全局扫描历史 UUID、外部副作用去重。

## 4 能力

S1：调用方把 checkpointId 原样传给 resume，恢复指定状态。S2：仅事件存储确实保存快照时返回标识；缺失快照通过稳定错误码判定。

### 4.1 UI/UX

无页面。SDK 不要求解析 ID 或自行拼接；成功继续执行，缺失返回 CHECKPOINT_NOT_FOUND，未保存不返回 checkpointId。

## 5 思路与折衷

采用携带 run 定位信息和快照唯一部分的版本化不透明标识，事件日志仍是唯一快照事实源。放弃另存快照副本和新增持久化 UUID 索引，避免双写与导入重建问题。代价：新 ID 不再是裸 UUID，消费者必须视其为不透明字符串。

## 6 架构

运行时生成标识并保存事件；SDK 根据标识读取目标 run，再按完整 ID 选取快照。主路径为保存成功→返回 ID→精确读取→恢复。失败路径为未配置事件存储→不返回 ID；目标不存在→稳定拒绝，不能回退到 latest。

## 7 模块

AgentRuntime 负责生成与公布标识；Milkie 负责解析；checkpointFromEvents 支持精确选择并保留默认 latest 行为。

## 8 API/CLI

新 ID 形态为 `checkpoint:v1:<URI 编码 runId>:<UUID>`，格式由框架维护，调用方不得解析。resume 未找到时抛出的 Error 增加 `code=CHECKPOINT_NOT_FOUND`。现有 context/run 别名及 stateStore 中手工保存的旧快照保持可读。

## 9 边界

同 run 多快照、同 context 后续快照不能改变旧 ID 指向。损坏或未知精确标识不得回退最新快照。没有事件存储时，字段缺失本身表示本结果不提供可恢复标识；不新增任务状态。

## 10 迁移/兼容/回滚

不改快照 schema。旧裸 UUID 只有已有 stateStore 对应记录时可读，不新增全局扫描。会话导出/导入保留事件内 ID，因此无需额外索引迁移。回滚后新 ID 不可解析，原 context latest 路径仍存在。

## 11 测试计划

- E2E：SDK invoke 返回 ID→原样 resume→读取恢复状态；同 run 多快照与后续 context 新快照仍精确。S1 → `.agents/skills/verify-milkie/features/04-interrupt-resume.md`。
- E2E：SQLite+JSONL 重建实例、只重建 JSONL 读取实例、会话导入后原 ID 恢复；无 eventStore 不返回 ID且恢复失败有稳定码。S2 → 同一功能文件。
- Integration：保留 CLI/HTTP context 别名恢复；Unit：精确选择不退化到 latest。

## 12 开放问题

无阻塞问题；不要求历史不可寻址 UUID 自动升级。

## 13 关联

#244；#260；[既有设计](244-budget-stop-reason.md)。PR：开发模式尚未创建。
