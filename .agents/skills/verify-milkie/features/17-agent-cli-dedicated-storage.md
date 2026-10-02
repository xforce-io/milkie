# CLI 专用配置与会话存储

## 用户入口

SDK `ExecutionClient.createContext(cwd, { configDir, sessionDir })`。方法登记在 [16-agent-cli-execution](16-agent-cli-execution.md)。无新增 HTTP、CLI 或页面。设计为 `docs/design/265-agent-cli-dedicated-storage/product.md` v1，S1 与 S2 的子项归本功能。

## 源码依据

`src/execution/ExecutionClient.ts`、`adapters.ts`、`worker.ts`、`types.ts`。

## 驾驶路径与判定

| 路径 | 操作 | 可判定结果 |
|---|---|---|
| S1.A1 | Grok：临时配置目录只放登录材料，空会话目录，执行后换进程续接 | 两轮成功、同一原生会话；会话只在指定目录；不读宿主 `~/.grok`，leader socket 在配置目录内 |
| S1.A2 | Pi：同上 | 两轮成功、同一原生会话；配置目录与 `--session-dir` 均为指定位置 |
| S1.A3 | Linux 容器内不挂载宿主 HOME，完成一条执行、重启续接，并取消一次运行中的 CLI | 续接成功；取消后所属 CLI 已停止 |
| S2.A1 | Grok 配置目录缺失或没有登录材料 | 启动前 `config_missing`，不启动 CLI |
| S2.A2 | Grok 删掉精确会话后续接 | 启动前 `session_missing`，不新建会话 |
| S2.A3 | Pi 配置缺失 | 启动前 `config_missing` |
| S2.A4 | Pi 会话缺失 | 启动前 `session_missing` |
| S2.A5 | Linux 容器内一次配置或会话缺失 | 对应错误码；诊断无凭据或完整会话 |

## 验证方法

- `npm run test:execution`：启动前拒绝与子进程环境，不证明真实 CLI。
- `MILKIE_LIVE_STORAGE=1 ./node_modules/.bin/tsx tests/e2e/agent-cli-storage.live.ts`：真实 Grok 与 Pi 的 S1.A1、S1.A2、S2.A1–S2.A4。
- Linux 容器探针覆盖 S1.A3 与 S2.A5，不挂载宿主 HOME。

## 已知缺口

容器路径与真实 CLI 路径的通过记录以冻结候选的验证记录为准。
