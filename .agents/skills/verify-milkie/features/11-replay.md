# 确定性回放与失败重建

## 用户入口

- SDK：`replay`；CLI：`trace replay <runId>`

## 源码依据

- [src/runtime/Milkie.ts](../../../../src/runtime/Milkie.ts)
- [src/trace/ReplayingIOPort.ts](../../../../src/trace/ReplayingIOPort.ts)
- [src/trace/CacheIndex.ts](../../../../src/trace/CacheIndex.ts)
- [src/trace/RunSnapshot.ts](../../../../src/trace/RunSnapshot.ts)
- [src/trace/LlmOutcome.ts](../../../../src/trace/LlmOutcome.ts)

## 驾驶路径与判定

| 路径 | 操作 | 可判定结果 |
|---|---|---|
| 成功 | 先记录模型、工具和工作状态变更，再用禁止 live 调用的网关回放 | 结果一致；模型和工具外部副作用调用数为 0；时钟、UUID 和上下文输入可重建。 |
| 失败终态 | 记录模型错误及工具取消/deadline，再回放 | 错误类型、稳定码与停止语义可区分；不再次连接故障服务。 |
| 分歧/缺损 | 改变配置或删改请求/事件，触发多消费和少消费 | 明确回放分歧或完整性错误，不悄悄回退 live。 |

## 验证方法

下列命令从仓库根运行；是自动化覆盖入口，不代表上表所有路径已通过。具体执行记录见[首版核验](../references/initial-audit.md)。

```sh
./node_modules/.bin/jest --runInBand --runTestsByPath src/__tests__/Replay.test.ts src/__tests__/Replay.nondet.test.ts src/__tests__/ReplayingIOPort.test.ts src/__tests__/CacheIndex.test.ts src/__tests__/RecordingIOPort.llmFailure.test.ts src/__tests__/determinism-wm-eventing.test.ts
```

- [src/__tests__/Replay.test.ts](../../../../src/__tests__/Replay.test.ts)
- [src/__tests__/Replay.nondet.test.ts](../../../../src/__tests__/Replay.nondet.test.ts)
- [src/__tests__/ReplayingIOPort.test.ts](../../../../src/__tests__/ReplayingIOPort.test.ts)
- [src/__tests__/CacheIndex.test.ts](../../../../src/__tests__/CacheIndex.test.ts)
- [src/__tests__/RecordingIOPort.llmFailure.test.ts](../../../../src/__tests__/RecordingIOPort.llmFailure.test.ts)
- [src/__tests__/determinism-wm-eventing.test.ts](../../../../src/__tests__/determinism-wm-eventing.test.ts)

## 已知缺口

CLI replay 当前输出 newRunId/status/output，不能假设等同完整 AgentResult。s-005 旧 readiness 中“等待非确定性日志”已过时；当前存在对应实现与测试。

## 对应 Stories

- [docs/stories/s-005-deterministic-replay.md](../../../../docs/stories/s-005-deterministic-replay.md)
