# #263 开发与验证状态

> 这是 #265 落地前的历史快照。执行套件现为 27 项；存储验收以 `test-output/verify-milkie/265/completion.md` 为准。不要把下面的 blocked 行当成当前树的存储结论。

- 分支：`feat/263-agent-cli-execution`
- 基线：`7865ffcc14a8359a055e5e6e0998b56ab2160379`
- 状态：未提交的实现候选；尚未完成验收、独立评审或 PR。
- 设计：`docs/design/263-agent-cli-execution.md` v1。产品依据：本会话批准调整 #263，并要求开发测试后 PR；不包含合入/部署。
- 环境：macOS，Node v23.11.0；Grok 1.0.41，Pi 0.85.1。会话中途改为 restricted 网络和文件权限，`.git`、`.agents` 只读。

## 已实现的候选

统一 SDK、持久上下文/执行记录、跨宿主排他占用、监督进程、精确 Grok/Pi 参数和事件解析、取消/超时、未知状态阻止重放、配置兼容、Pi 连接契约 fixtures。SDK 用法见 `docs/agent-cli-execution.md`。

## 已执行证据

证据目录：`test-output/verify-milkie/263-probe/`（忽略目录，不提交原始运行产物）。

| 命令/检查 | 结果 | 证明范围 |
|---|---|---|
| `npm run build` | pass | TypeScript 编译 |
| 修改前默认 `npm test` 的两段原有套件 | pass，121 Unit + 8 E2E | 既有默认套件回归；见 npm-test.log |
| `npm run test:execution` | pass，21 项 | SDK API 生命周期、真实本地子进程协议、跨进程占用、宿主强杀后查询/取消、会话关联、进程组停止、错误脱敏；使用假 CLI 和假 API gateway，不能证明真实供应商行为；见 execution-test.log |
| 连接完整套件 | 18 pass，2 环境失败 | 两个真实本地 HTTP listener 被 EPERM 拒绝；不是全套通过 |
| 连接 fixtures/解析专项 | pass，4 项 | 全部版本 fixtures（含 Pi）与 CLI 字段行为；另外 16 项被命令过滤，不能算通过；见 connection-fixtures.log |
| npm 发布物 dry-run | pass | 包含 worker.js、ExecutionClient.js 和类型声明；未发布 |
| Grok/Pi 真实 SDK 首轮探针 | fail | 均 process_failed；后续直接 CLI 诊断均 exit 1、permissionDenied=true、argumentError=false；见 real-diagnostics.json |
| GitHub Issue 查询 | blocked | 代理连接 `127.0.0.1:9567` 被沙箱 operation not permitted 拒绝 |
| `git diff --check` | pass | 已跟踪差异的空白检查，不代表验收 |

## 验收汇总

| Issue / L1 验收 | 状态 | 尚缺的权威证据 |
|---|---|---|
| S1 / S1.A1 | blocked | 真实 API 与两种 CLI 的 SDK 成功路径；当前只有确定性生命周期证明 |
| S2 / S2.A1 | blocked | 两种 CLI 各真实三轮，原生会话不变且宿主重启；已有 opt-in live 探针，尚未通过 |
| S3 / S3.A1、S3.A2 | blocked | 真实双上下文隔离、新建及双宿主冲突；确定性协议和原子占用已验证 |
| S4 / S4.A1 | blocked | 每种真实 CLI 会话缺失、登录失效、宿主异常退出后的核对；受控子进程已覆盖相关机制 |
| S5 / S5.A1 | not_run / 实现缺口 | 真实 cwd、只读工具、时限、取消与未知约束；当前仅进程组停止核对，脱离组的任务子进程归属尚未完成，不能宣称全资源已停止 |

## 下一步

1. 恢复真实 CLI 所需的用户存储写权限、网络、本地端口和进程信息访问；恢复 `.git` / `.agents` 写权限，保留本工作区。
2. 对真实 CLI 查证工具权限与子进程所有权，补齐 S5 的资源核对缺口；若技术调查影响产品范围，按已确认 Issue 提出差异，不能直接降低验收。
3. 将 `263-functional-map.pending.md` 的条目应用到 `.agents/skills/verify-milkie/features/16-agent-cli-execution.md` 及 README；现有替代文件不算地图已完成。
4. 完成所有真实验收与必要回归，提交冻结候选，按 keel-verify 重新取证。
5. 按 keel-review 进行独立评审，通过后按 keel-release 创建 PR。当前没有审查 PASS 或发布完成证据，不能跳过门禁。

已读环节：keel、keel-how、keel-design、keel-dev、keel-verify、keel-review、keel-release；并读取项目 verify-milkie 与本地 code-review 合同。读取后两个环节不代表已执行或通过。

## 本轮续跑

- 前一轮为 progress：实现候选与确定性测试已落盘；本轮再次确认 GitHub 和 ps 仍被 operation not permitted 拒绝。
- 复现并修复 API 在事件循环阻塞后超过时限仍返回成功；回归用例先失败，再在返回前校验时限后通过。
- 获取上下文占用后重读会话关联，避免前轮在竞争窗口完成后使用旧关联；返回的上下文配置不再共享客户端内部对象。
- 停止信号被拒绝时在限定时间内持久记录 unknown，保留占用，不无限等待进程退出；确定性信号拒绝用例约 7.7 秒完成。
- 最新 `npm run test:execution` 编译通过，21/21 通过；原有 129 项回归结果不变。本轮未重跑未受影响的旧套件。
- 同一外部权限阻塞持续存在。真实 S1–S5、脱离进程组子进程核对、功能地图、冻结提交、独立评审和 PR 仍未完成。
