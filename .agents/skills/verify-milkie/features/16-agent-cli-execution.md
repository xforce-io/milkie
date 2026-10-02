# 外部 CLI 执行与原生会话续接

## 用户入口

SDK `ExecutionClient`；无新增 HTTP、CLI 或页面。设计为 `docs/design/263-agent-cli-execution.md` v1，S1–S5 的所有子项归本功能。

## 源码依据

`src/execution/ExecutionClient.ts`、`worker.ts`、`store.ts`、`adapters.ts`、`types.ts`；连接扩展见 `src/connection/parse.ts` 与 `types.ts`。

## 驾驶路径与判定

| 路径 | 操作 | 可判定结果 |
|---|---|---|
| S1.A1 | 真实 API 与 Grok/Pi 经同一 SDK 执行查询，另选 claude-code | 三成功、一明确不支持执行；API 无行为回归 |
| S2.A1 | 每 CLI 在独立宿主进程连续三轮；仅首轮给随机标记 | 同原生 ID、不同 runId；首轮进程已退，后续仅新输入仍记得标记 |
| S3.A1 | 同目录双上下文交替续接，再新建 | 标记不串；显式新建不同原生 ID |
| S3.A2 | 两宿主同时 start 同一 contextId | 一个活动，另一个 busy；无自动重放 |
| S4.A1 | 会话缺失、认证失效、宿主强杀及状态未知 | 不静默新建、不猜最近会话；未确认停止不续接；修复后显式继续或新建 |
| S5.A1 | 实际 cwd/只读工具写入拒绝/超时/取消/未知约束 | 工作目录一致、文件未变且工具拒绝、超时停止；取消十秒内确认所属子进程停止；未知约束启动前拒绝 |

## 验证方法

- `npm run test:execution`：确定性适配协议与进程管理，不证明真实供应商支持。
- `MILKIE_LIVE_EXECUTION=1 ./node_modules/.bin/tsx tests/e2e/agent-execution.live.ts`：真实 S2/S3.A1、只读与 cwd 的部分路径；必须已构建、安装并登录 CLI。
- `MILKIE_LIVE_EXECUTION=1 ./node_modules/.bin/tsx tests/e2e/agent-execution-controls.live.ts` 覆盖 S3.A2/S4/S5 的认证、会话缺失、双宿主竞争、宿主强杀、取消与执行时限；真实 API 另通过 SDK 配置测试连接执行，不输出密钥。
- 既有连接回归：`./node_modules/.bin/jest --runInBand --runTestsByPath src/__tests__/modelConnectionContract.test.ts`。

## 已知缺口

真实验证正在进行，结果以冻结候选的验证记录为准；确定性子进程不能替代真实 CLI。
