# 本地执行 Sidecar

主票 [Milkie #277](https://github.com/xforce-io/milkie/issues/277)，S1/S2 及细化验收以 [设计 v0.4](../../../../docs/design/277-codex-sidecar-execution.md) §11 为准。本功能文件是驾驶方法，不代表整票通过。

## 入口与前置

- 构建：`npm run build`。
- CLI：`node dist/cli/index.js execution-sidecar --help`。
- 启动：`node dist/cli/index.js execution-sidecar --data-dir /absolute/test-data --connection-file /absolute/connection.json`。
- 连接文件是宿主准备的 ConnectionInput，不把凭据放 argv；专用配置目录与专用会话目录仍由宿主显式准备。禁止复制真实登录材料作为本票测试准备。支持与就绪是两个结果；capabilities.availability=unchecked 不能作为真实模型可用证据。

## 驾驶路径与判定

1. **S1.A1**：先发送不兼容 initialize 或未握手的 execution.start。必须收到明确错误，不能产生执行记录。再次用 protocolVersion=1 握手返回实际能力；重复握手拒绝。execution.query 不存在的 UUID 返回 null。
2. **S1.A2**：用同一 CLI/协议入口，分别运行固定版本 Codex 与第二真实 Harness；读取模型实际结果，执行一个宿主工具回调，核对 runId/contextId/callId。记录 CLI 版本、模型、专用存储、终态及送达结果。确定性 fixture 无法证明本项。真实矩阵固定为 Codex0.166.0/gpt-6-astra/high 与 Pi1.2.0/openai-codex/gpt-5.6-sol/high；每轮实际结果单独留证，不把前置配置视为通过。
3. **S1.A3**：错误 JSON、超限帧、并发重复 ID、错误/晚到工具响应、stdout 失效分别验证。错误输出只含固定代码；不能回显配置、原始错误或提示词。断开后不继续接受命令，不错配旧回调，不重放。
4. **S2.A1**：两个真实上下文并发执行；查询分别关联，取消其中一个、核对原生进程确已停止，另一个继续完成。stop RPC 返回本身不能当停止证据。确定性版本只覆盖协议机制，真实版本仍 not_run。
5. **S2.A2**：无宿主工具执行时分别 EOF、SIGTERM、SIGKILL Sidecar，核对原生进程退出和执行记录。强杀失联保持 unknown，stopped 只有确认后才能 true；在宿主工具未回传时断开，应保留 pending 效果且禁止自动重试。对不支持约束验证启动前拒绝。
6. **S2.A3**：回归 [16](16-agent-cli-execution.md)、[17](17-agent-cli-dedicated-storage.md)、[18](18-agent-cli-host-tools.md)、[19](19-agent-cli-host-tool-resume.md)、[20](20-cli-iteration-tool-bounds.md)。保留已存在的心跳波动，不把定向复验通过改写为历史全套通过。

自动化命令：

```sh
npm run build
./node_modules/.bin/jest --runInBand --runTestsByPath tests/e2e/execution-sidecar.test.ts
./node_modules/.bin/jest --runInBand --runTestsByPath src/__tests__/codexExecution.test.ts
npm run test:execution
python3 .agents/skills/verify-milkie/scripts/check_map.py
```

Sidecar 套件使用已有 tests/fixtures/agent-cli.cjs；只证明跨进程协议、隔离、工具关联、宿主存活契约，不证明原生 Codex/Pi/Grok 行为。运行时应保留命令、候选 SHA、退出码、原始日志与未覆盖验收。整票 S1/S2 不可由本套件单独记 pass。


## 原生协议探针（无真实模型）

将宿主安装的0.166.0加入 PATH，运行 `MILKIE_LIVE_CODEX_PROTOCOL=1 node tests/e2e/codex-native-protocol.live.cjs`。探针使用本次临时专用目录、明确的假凭据和本地 HTTP 401 提供方，断言 auth_failed、原生进程停止及真实 JSONL 的 id/cwd；不访问真实供应商、不复制登录材料。输出在 test-output/verify-milkie 下，不能作为 S1.A2/S2.A1 真实模型成功证据。它补充了 thread/start 路径先出现、文件在 turn/start 后才创建的原生差异；确定性套件同样覆盖延迟创建。
