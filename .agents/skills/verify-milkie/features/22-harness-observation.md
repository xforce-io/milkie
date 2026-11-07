# Harness 统一执行观测

主票 [#278](https://github.com/xforce-io/milkie/issues/278)，依赖 [21-execution-sidecar](21-execution-sidecar.md)。细化验收以 [设计 v0.4](../../../../docs/design/278-harness-observation-tool-callbacks.md) §11 为准；没有观测浏览器页面。

## 入口

- SDK：`ExecutionClient.capabilities().observation`；`start` 的显式 `constraints.observation`；`ExecutionClient.observations(runId, afterSequence, limit)`。
- stdio：`capabilities`、`execution.start`、`observation.read`；通知 `execution.observation` 和 `execution.observation.status`。
- 未提供观测选项不启用来源缓冲；指定时必须给 traceparent、maxEvents、maxBytes。来源缓冲不是 TaskLoop 最终 trace 库，原有执行记录仍决定执行状态。

## 驾驶路径与判定

1. **S1.A1**：真实 Codex 与第二真实 Harness 同入口启动，核对声明与实际事件：监督层生命周期、原生活动和报告过的 token 用量。没有原生模型请求明细/用量时必须保留 unavailable 或无事件；不能补零或提取隐藏推理。记录固定版本、实际模型及真实服务输入；每轮状态由证据决定。
2. **S1.A2**：SDK 和 stdio 多页读取、按游标继续、从旧游标复读。稳定 id/来源序号不能变化；超出容量、尾帧/损坏、未启用、未结束分别可判定。缓冲容量耗尽必须仍有终止或缺口状态，不能影响业务结果或标完整。
3. **S1.A3**：两个上下文分别带不同 traceparent 并发，核对 run/context/trace 不串线；错误 traceparent/容量启动前拒绝，旧无观测路径回归。
4. **S2.A1**：真实宿主工具的成功、execution_failed、invalid_input/rejected、取消和未送达均可关联。取消不保证宿主副作用已撤销；pending 不重放。原始真实结果单独保留，不由确定性检查代替。
5. **S2.A2**：并行调用分别核对 callId/nativeCallId/traceparent；请求、宿主响应和送达不是同一事实。复算参数/成功结果的 UTF-8 字节数与 SHA-256；事件内没有完整正文、原始异常或凭据。有原生 id 的重复请求不得重复执行，缺少原生 id 时明确能力限制。

## 自动化与证明边界

```sh
npm run build
./node_modules/.bin/jest --runInBand --runTestsByPath src/__tests__/executionObservation.test.ts tests/e2e/execution-sidecar.test.ts
npm test
python3 .agents/skills/verify-milkie/scripts/check_map.py
```

新套件覆盖来源容量/分页/损坏、上下文、共享工具分发及真实子进程协议；它使用确定性 CLI 夹具，不替代两种真实 Harness。原生 Codex 的离线协议探针仍按21执行，不调用真实模型，不能据此写 S1/S2 pass。

## 公开保留清理（#278 v0.3 / TaskLoop #8）

通过 SDK `purgeObservations(runId)` 或 Sidecar `observation.purge` 清理完成且原生停止、无 pending 工具的 Run。活动/停止未知/pending 必须 context_busy，原文件保留；成功及重复成功均核对仅观测文件消失，执行/认证/工具/原生会话不变。模拟删除中断后观察 cleanup_pending，重试后 expired，不能假 complete。对应确定性命令为 executionObservation.test.ts、execution-sidecar.test.ts；真实 Harness 和 TaskLoop 驱动另验。
