---
name: verify-milkie
description: 验证 Milkie 的 SDK、CLI、HTTP 服务及执行记录行为；在 Milkie 功能验收、缺陷复现或功能地图维护时使用，按需读取功能文件并记录证据。
---

# Milkie 验证手册

从仓库根执行命令。先读 [功能地图](features/README.md)，再读本次涉及的功能文件；不要把整张地图的测试每次都跑一遍。名词以 [名词表](../../../docs/glossary.md) 为准。没有应用页面；HTML 执行报告有浏览器交互入口。

## Launch

1. 记录 `git remote get-url origin`、`git rev-parse HEAD` 和 `git status --short`。本手册首版基线为 `7e2687f`；不同版本须重新核对入口。
2. 需要 Node.js 20+、npm、Python 3。已有依赖时使用 `./node_modules/.bin/jest` 和 `./node_modules/.bin/tsx`，避免 npx 隐式下载。缺依赖则按仓库 lockfile 用 `npm ci`；不要为验证升级依赖。
3. 证据放在忽略目录 `test-output/verify-milkie/<本次唯一标识>/`。维护全图也可使用 keel 指定的 `.grok/verify-runs/regression/`，先确认其不会被误提交。测试数据用 `mktemp -d` 创建的目录，禁止借用用户 `.milkie/` 数据。
4. SDK 路径优先用功能文件链接的确定性 fixture；它们经过真实运行时、注入假模型网关，不调用付费模型。CLI 参数/输出、SDK 调用和 HTTP 路由是不同入口，分别记录。
5. HTTP 进程验证使用已有 fixture：

   ```sh
   PORT=0 STEPS=6 STEP_MS=120 ./node_modules/.bin/tsx tests/e2e/fixtures/serve-stub-entry.ts
   ```

   保持 stdin 打开，等待 stdout 的 `MILKIE_SERVE_READY <port>`；使用该端口，不能假设固定端口。这是单个测试会话的有状态模型 fixture，多组场景应重启进程。它不覆盖真实 `serve --agent` 加载路径。
6. 真实 CLI 的入口为 `./node_modules/.bin/tsx src/cli/index.ts`；先 `--help`。真实模型调用需要明确的测试连接，不从生产环境随意选 Agent 或模型。SQLite 模式必须使用本次临时 `--data-dir`。HTML 报告写到证据目录，再按环境可用的浏览器手册打开。

## Doctor

```sh
node --version
npm --version
python3 .agents/skills/verify-milkie/scripts/check_map.py
./node_modules/.bin/tsx src/cli/index.ts --help
```

需要 SQLite 的路径先检查本机 ABI：

```sh
node -e 'const D=require("better-sqlite3"); const d=new D(":memory:"); console.log(d.prepare("select 1 as ok").get()); d.close()'
```

若 Node 与已有 SQLite 二进制 ABI 不符，先检查本机已安装的兼容 Node，并只对本次命令调整 PATH；不要把应用环境错误写成产品缺陷，也不要无必要重装依赖。

HTTP 启动后 `curl -fsS http://127.0.0.1:<port>/health` 应返回 `{"ok":true}`。缺依赖、ABI 错误、端口未就绪是环境阻塞，保留报错；不可将未执行写成通过。Redis 和真实供应商测试单列，不从其他后端结果外推。

## Drive

- 每条验收对应一个功能文件，依照其“驾驶路径与判定”操作。涉及多个入口时逐项执行；一次 SDK 成功不能代表 HTTP/CLI 成功。
- 功能文件的 Jest 命令用于复用自动化证据。读取相关断言确认实际覆盖；测试名、文件存在或绿色退出本身不能证明全部验收。
- 确定性的进程/HTTP 主路径可直接运行：

  ```sh
  ./node_modules/.bin/jest --runInBand --runTestsByPath tests/e2e/serve.e2e.test.ts
  ```

- 需要复查 #259–#261 时运行 [恢复与结果出口探针](scripts/probe-known-gaps.ts)：

  ```sh
  ./node_modules/.bin/tsx .agents/skills/verify-milkie/scripts/probe-known-gaps.ts
  ```

  它通过 SDK 与真实本地 HTTP 请求核对期望契约，输出逐项 JSON；存在缺陷时退出 1，环境/脚本异常退出 2。不得反转断言，把“缺陷仍存在”记为产品 pass。
- 手工 HTTP 请求示例：`POST /chat` 的 JSON 为 `{"contextId":"verify-one","input":"continue"}`；读取 SSE 直至结束，保留活动和终态。中断恢复需要另一连接在执行中 POST `/interrupt`，等待结束后 POST `/resume`，两者均带相同 contextId。
- 首版仅建立地图、核验源码并执行有限确定性测试；完整结果见 [首版核验](references/initial-audit.md)。未执行的浏览器、远端模型、Redis 等路径继续标未验证。

## Evidence

每次至少保留：代码 SHA、脏工作区摘要、命令与退出码、功能文件/入口/路径、预期与实际、原始输出位置。HTTP 结果保留状态码和 SSE；执行结果保留 runId/contextId 及对应事件，使用无敏感信息的测试数据。

结果表使用 `pass | fail | skip`：pass 仅限实际断言范围；fail 包括已知缺陷；skip 必须写未执行的原因和缺失证据。源码核对另记“已核对”，不混为运行 pass。产品失败不妨碍地图记录完成，但不能宣称全图回归通过。修复后用同一预期契约重新验证。

## Cleanup

只停止本次启动且记录过的 PID；关闭临时服务、数据库和 Redis 连接。只删除本次创建的临时测试数据，保留证据目录。不要清理用户数据，不提交测试产物，不自动提交、推送或修改线上 Issue。
