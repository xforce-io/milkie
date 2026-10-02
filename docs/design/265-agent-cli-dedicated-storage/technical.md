# 专用配置与会话存储技术设计

| 项 | 值 |
|---|---|
| Issue | [#265](https://github.com/xforce-io/milkie/issues/265) |
| L1 | [product.md](product.md) v1 |
| 版本 | v1 |
| 验收 | S1.A1–S1.A3、S2.A1–S2.A5 |

## 1. 设计依据与技术目标

落实 L1 v1：CLI 执行上下文只使用宿主给出的配置目录和会话目录；缺失时在拉起 CLI 之前失败。不实现宿主工具，不管理容器。

## 2. 现状与改动范围

#263 的未发布草稿在 `src/execution/`。当前机制：

- `ExecutionClient.createContext` 只接收工作目录。Pi 的会话文件放在 milkie `dataDir/native`。Grok 的会话路径由 `adapters.nativeFile` 从 `HOME/.grok/sessions` 推导。
- 子进程环境是 `ExecutionClientOptions.env`，缺省为宿主 `process.env`。没有设置 `GROK_HOME`、`PI_CODING_AGENT_DIR` 或 `--leader-socket`。
- `assertNativeSession` 在续接前检查会话文件；Pi 还核对首行会话头。失败码是 `session_missing`。

本票改 `types`、`adapters`、`ExecutionClient` 和 `worker` 的 CLI 环境。API 传输的 `createContext(cwd)` 保持不变。存储版本仍是 1；旧的无存储字段上下文不能继续当作 CLI 上下文使用，启动时 `config_missing`，不回退到 HOME。

## 3. 总体架构与关键路径

创建时解析两个目录并写入执行上下文。启动和监督进程在 `spawn` 之前再次检查。Grok 的数据根是配置目录；它的 `sessions` 入口必须指向会话目录。Pi 用配置目录环境变量和 `--session-dir`，会话文件仍是会话目录下的精确路径。

失败路径不拉起 CLI。两个上下文若共用一个 Grok 配置目录但会话目录不同，创建第二个时拒绝，避免改写已经绑定的会话入口。

## 4. 数据与状态契约

`ExecutionContext` 增加 `configDir` 与 `sessionDir`，保存 `realpath` 后的绝对路径。不保存 `auth.json` 内容、环境变量或会话正文。

登录材料的判定是配置目录内的普通文件 `auth.json`，大小非 0，且其实路径仍在配置目录内。不读取文件内容。

Grok 会话文件位置为 `sessionDir/encodeURIComponent(cwd)/nativeSessionId/chat_history.jsonl`。Pi 会话文件为 `sessionDir/<contextId>.jsonl`。续接沿用已有的非空与 Pi 会话头核对。

配置目录中的 `sessions`：不存在则创建指向会话目录的符号链接；已是指向同一实路径的符号链接或目录则复用；指向别处则 `invalid_request`。不移动已有数据。

## 5. 接口与协作契约

```ts
interface CliStorage { configDir: string; sessionDir: string }
createContext(cwd: string, storage?: CliStorage): ExecutionContext
```

CLI 运行时缺少 `storage`、配置目录不可用或没有登录材料：`config_missing`。会话目录不可用：`session_missing`。两个目录实路径相同：`invalid_request`。API 传输传入 `storage`：`invalid_request`。错误文本保持 `Execution request failed: <code>.`，不附带路径、凭据或会话正文。

Grok 子进程：`GROK_HOME` 为配置目录，`GROK_LEADER_SOCKET` 与参数 `--leader-socket` 为配置目录下的 `leader.sock`。不使用 `~/.grok/leader.sock`。当前 headless `grok` 没有 `--no-leader`；隔离靠专用 socket 路径，不连接默认 leader。

Pi 子进程：`PI_CODING_AGENT_DIR` 为配置目录，参数同时包含 `--session-dir <会话目录>` 和 `--session <精确文件>`。不传 `--continue` 或无目标的 `--resume`。

子进程环境从调用方环境复制后，删除名为 `GROK_AUTH` 的变量，以及以 `_API_KEY`、`_AUTH_TOKEN`、`_OAUTH_TOKEN`、`_ACCESS_TOKEN`、`_REFRESH_TOKEN` 结尾的变量。然后写入上面的专用变量。PATH 与 HOME 保留；CLI 配置不从 HOME 推导。

## 6. 运行与保障机制

检查放在 `createContext`、`start` 和监督进程拉起 CLI 之前。任一检查失败都不 `spawn`。取消与时限仍由 #263 的监督进程负责；容器验收只额外确认该停止信号在 Linux 容器内仍然结束所属 CLI。监督进程清点进程时设置 `PS_PERSONALITY=bsd`：Debian 的 `ps` 否则拒绝 `-x`，macOS 的 `ps` 忽略该变量。Pi 若先发出助手错误再自动重试，以最后一条助手消息判定成败。

诊断沿用现有规则：运行记录不保存 CLI 的 stdout/stderr。`auth_failed` 仍只表示 CLI 在登录材料存在时报告登录失效，与目录缺失区分开。

## 7. 迁移、发布与回滚

#263 SDK 尚未进入默认分支。已按 HOME 推导会话的草稿上下文缺少新字段，启动得到 `config_missing`，不会改去读宿主 HOME。回滚是停用新 SDK，不删除宿主目录里的原生会话。

## 8. 测试与验证

功能地图：`.agents/skills/verify-milkie/features/17-agent-cli-dedicated-storage.md`。S1 与 S2 的主文件都是它。#263 的 `16-agent-cli-execution.md` 只保留执行生命周期，存储路径改由本文件证明。

| 验收 | 机制 |
|---|---|
| S1.A1 S1.A2 | 真实 CLI 活探针：临时配置目录只放入登录材料，会话目录单独挂接；断言会话文件、环境变量和 leader socket |
| S1.A3 S2.A5 | Linux 容器活探针：不挂载宿主 HOME；成功续接一条，缺失失败一条，并取消一次运行中的 CLI |
| S2.A1–S2.A4 | SDK 单测在 `spawn` 前抛出对应错误；确定性子进程不能代替真实 CLI 的隔离证明 |
| 回归 | 既有执行单测改为显式传入两个目录，确认不再写 `HOME/.grok` |

## 9. 技术风险与开放问题

Grok 1.0.41 会在空的 `GROK_HOME` 里初始化说明文件，但仍会因缺少登录而失败。实现在启动前拒绝没有 `auth.json` 的目录，因此不会依赖这次初始化。若后续版本把登录存到其它文件名，预检查会误拒绝；那时改 L2 的登录材料判定，不改 L1 的失败语义。

容器内需要 Linux 构建的 CLI。安装方式只影响 S1.A3 与 S2.A5 能否取证，不能用宿主 macOS 结果代替。
