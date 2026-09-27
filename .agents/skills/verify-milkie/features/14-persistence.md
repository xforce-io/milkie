# 存储、重启与发布物

## 用户入口

- SDK：MemoryStore、SQLiteStore、RedisStore；MemoryEventStore、JsonlEventStore；MemoryTraceObjectStore、FileTraceObjectStore；TrajectoryStore/recorders
- CLI：serve --state-store sqlite --data-dir；包 @freemanxu/milkie 的 milkie binary

## 源码依据

- [src/store/MemoryStore.ts](../../../../src/store/MemoryStore.ts)
- [src/store/SQLiteStore.ts](../../../../src/store/SQLiteStore.ts)
- [src/store/RedisStore.ts](../../../../src/store/RedisStore.ts)
- [src/trace/JsonlEventStore.ts](../../../../src/trace/JsonlEventStore.ts)
- [src/trace/TraceObjectStore.ts](../../../../src/trace/TraceObjectStore.ts)
- [src/trajectory/TrajectoryStore.ts](../../../../src/trajectory/TrajectoryStore.ts)
- [src/cli/serve.ts](../../../../src/cli/serve.ts)
- [package.json](../../../../package.json)

## 驾驶路径与判定

| 路径 | 操作 | 可判定结果 |
|---|---|---|
| 本地持久化 | 在临时目录建立 SQLite+JSONL 存储，执行并保存状态；关闭后用同目录重建实例 | 历史、变量和 context 路由仍可读取；Memory 配置仅保证当前进程。 |
| 环境错误 | 缺少 sqlite data-dir、SQLite ABI 不匹配、缺失对象文件 | 错误明确，不能静默降级到内存并声称已持久化。 |
| 发布物 | 构建并本地 pack，在新临时消费者目录安装包，运行 milkie --help 和 agent list | 包包含 dist/cli/index.js 和可执行入口；无需消费者构建源码；不得误装无 scope 的 milkie。 |
| Redis | 启动明确的测试 Redis，执行跨连接存取、TTL 和会话恢复 | 状态可共享且隔离；未启动 Redis 时标未验证，不把 Memory 结果外推。 |

## 验证方法

下列命令从仓库根运行；是自动化覆盖入口，不代表上表所有路径已通过。具体执行记录见[首版核验](../references/initial-audit.md)。

```sh
./node_modules/.bin/jest --runInBand --runTestsByPath src/__tests__/MemoryStore.test.ts src/__tests__/SQLiteStoreContextVars.test.ts src/__tests__/SQLiteStore.abiMismatch.test.ts src/__tests__/serve-persistence.test.ts src/__tests__/MemoryEventStore.test.ts src/__tests__/assertUnpublished.test.ts
```

- [src/__tests__/MemoryStore.test.ts](../../../../src/__tests__/MemoryStore.test.ts)
- [src/__tests__/SQLiteStoreContextVars.test.ts](../../../../src/__tests__/SQLiteStoreContextVars.test.ts)
- [src/__tests__/SQLiteStore.abiMismatch.test.ts](../../../../src/__tests__/SQLiteStore.abiMismatch.test.ts)
- [src/__tests__/serve-persistence.test.ts](../../../../src/__tests__/serve-persistence.test.ts)
- [src/__tests__/MemoryEventStore.test.ts](../../../../src/__tests__/MemoryEventStore.test.ts)
- [src/__tests__/assertUnpublished.test.ts](../../../../src/__tests__/assertUnpublished.test.ts)

## 已知缺口

真实 Redis 与消费者安装首版未执行；npm pack/prepack 会构建，安装可能访问 registry，按本次验证范围执行。恢复 UUID 仍受 #259 影响。
