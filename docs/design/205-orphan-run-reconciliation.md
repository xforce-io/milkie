# #205 悬空运行终态协调

最后更新：2026-07-12

## 问题

进程退出或执行被遗弃后，事件日志可能只有 `agent.run.started` 而没有 `agent.run.completed`，导致 trace 永久显示 in-flight。

## 设计

事件日志保持单一事实源。serve 启动时扫描 run 文件，对包含 started 且不存在 completion 的顶层 run 追加一个幂等终态：

- `status`: `interrupted`
- `error.code`: `RUN_ABANDONED`
- `error.phase`: `recovery`
- `error.retryable`: `true`

只追加事件，不改写或删除历史；第二次启动不重复追加。客户端主动断开仍只取消订阅，不取消健康运行，只有服务重启后仍非终态的旧 run 才会被协调。

## 测试计划

- 单元测试：完整、悬空、畸形日志分类和幂等追加。
- 集成测试：serve 使用包含悬空 run 的 data dir 初始化后写入终态。
- 功能测试：有效 completed/interrupted run 不变，悬空 run 获得结构化终态。
- 端到端测试：工具请求后杀死 sidecar，以同一 data dir 重启，旧 run 恰好新增一个终态且新请求成功。

## 非目标

不自动恢复或重试旧 run；上层调度器根据结构化结果决定后续动作。
