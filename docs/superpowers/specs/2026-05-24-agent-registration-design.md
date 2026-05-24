---
title: Agent registration — manifest convention for CLI / SDK
date: 2026-05-24
status: draft
phase: cross-cutting
subsystems:
  - agent-runtime
unblocks:
  - CLI surface spec §9 OQ #1 (agent registration mechanism)
  - `milkie agent run / list` CLI implementation (P0)
  - examples/ scaffold (each example needs at least one registered agent)
related:
  - docs/superpowers/specs/2026-05-24-cli-surface-design.md
  - ARCHITECTURE.md
---

# Agent registration — manifest convention

This spec closes the gap surfaced in
`docs/superpowers/specs/2026-05-24-cli-surface-design.md` §9 OQ #1:
the CLI assumes a registry of agents exists but does not define how the
registry gets populated. This document specifies that mechanism.

## 1. 目标与边界

**目标.** 让 CLI 短进程能在每次启动时知道"哪些 agent 是已注册的"，
不破坏现有 SDK 的 in-process registry。

**已经在 substrate 里的（本 spec 不动）：**

- `Milkie.registerAgent(config)` — 程序化注册（`src/runtime/Milkie.ts:77`）
- `Milkie.loadAgentFile(path)` — 从带 frontmatter 的 markdown 加载并注册（`src/runtime/Milkie.ts:69`）
- `Milkie.invoke({ agentId, ... })` — 执行；agent 未注册时 throw（`src/runtime/Milkie.ts:85`）
- Registry 是 `Milkie` 实例上的 `Map<agentId, AgentConfig>`

**本 spec 添加的：** 一份项目级 manifest，CLI 进程启动时读它、循环调
`loadAgentFile()` 把声明的 agents 注册进 in-process registry。SDK
consumers 可选用同一个 manifest（避免重复在代码里列 agents），也可继
续手写 `registerAgent()`。

**显式不在范围内：**

- ❌ 修改既有 substrate（`registerAgent` / `loadAgentFile` / `invoke`）
- ❌ Hot reload（manifest 改了，已启动的 CLI 进程不感知；下次启动生效）
- ❌ 多 manifest cascade（project + user + system 暂不支持，单 manifest）
- ❌ 跨进程持久 registry（CLI 每次进程独立，状态从 manifest 重建）
- ❌ Plugin discovery / npm package convention（未来 design）
- ❌ Daemon-based registry server（CLI 设计 spec §1 已排除）
- ❌ `milkie init` 命令初始化 `.milkie/` 目录骨架（独立的 ergonomics 议题，未来加）

## 2. `.milkie/` —— milkie 的项目级文件系统 footprint

milkie 在项目目录下使用一个 `.milkie/` 目录作为所有项目级文件的 root：

```
<project>/
├── package.json
├── tsconfig.json
├── .milkie/
│   ├── agents.json       # 本 spec 定义：agent manifest
│   ├── runs/             # （未来 phase）持久化 run 记录
│   ├── suites/           # （未来 phase）saved suite definitions
│   ├── cache/            # （未来 phase）cache projections
│   └── ...
```

**约定：**

- `.milkie/agents.json` 是 manifest（**gitted**，是用户配置）
- `.milkie/runs/` / `.milkie/cache/` 等运行时产物视情况 git 或 ignore（由
  各自后续 spec 定）
- 本 spec 只定义 `.milkie/agents.json`；目录里其他子项由后续 spec 在
  需要时加入

**为什么是 `.milkie/` 而不是项目根 `milkie.config.json`：**

- milkie 预期会有更多文件系统需求（runs / suites / cache / lineage
  index）；单一 dot-directory 比散在项目根的多个 `milkie.*` 文件更
  scalable
- 跟 `.git` / `.vscode` / `.turbo` / `.next` 等工具约定一致
- 项目根保持干净

## 3. `.milkie/agents.json` schema

```json
{
  "agents": [
    {
      "id": "router",
      "file": "../agents/router.md"
    },
    {
      "id": "verifier",
      "file": "../agents/verifier.md"
    }
  ]
}
```

**字段：**

| 字段 | 类型 | 说明 |
|---|---|---|
| `agents` | array | 必填，agent 声明列表 |
| `agents[].id` | string | 必填，agent identifier；CLI 通过此 id 调用 |
| `agents[].file` | string | 必填，agent .md 文件路径（相对 manifest 自身位置，即 `.milkie/`） |

**路径相对性：** `agents[].file` 相对 `.milkie/agents.json` 自身位置
解析（不是 cwd 也不是项目根）——这样 manifest 可移植，子项目可以放
自己的 `.milkie/`。

**校验：**

| 错误 | Exit code |
|---|---|
| `agents[].id` 在 manifest 内重复 | 2 (Invalid args) |
| `agents[].file` 不存在 | 3 (Not found) |
| `agents[].file` frontmatter `agentId` 与 manifest `id` 不匹配 | 2 (Invalid args) |
| Manifest JSON 解析失败 | 2 (Invalid args) |

**`agents` 数组为空：** 允许（合法状态，没有 agent 注册）。

## 4. CLI 启动时的行为

每次 `milkie <domain> <verb>` 启动：

1. **Manifest 查找.** 从 cwd 向上查找 `.milkie/agents.json` 直到找到或
   到达 fs root；找到的第一份 manifest 即生效（不递归 / 不合并）
2. **找到 → eager load.** 调 `Milkie.loadManifest(path)` 内部循环调
   `loadAgentFile()`，把所有声明的 agent 注册进 in-process registry
3. **没找到 → silent ok.** CLI 仍然启动，registry 为空；调用
   `agent run <id>` 时报 exit code 3 (Not found)，错误信息提示用户检查
   `.milkie/agents.json`

**Eager 理由：** 简单、`agent list` 直接可用、CLI 启动延迟感知不到
（10 个 agent 文件预加载 < 50ms）。Lazy 留到将来 agent 数量爆炸时再考虑。

**Cwd 向上查找理由：** 跟 git 一致；让用户在子目录里也能直接跑 CLI。

## 5. SDK consumer 怎么用 manifest（可选）

新增一个便利方法：

```typescript
const milkie = new Milkie({ ... })
await milkie.loadManifest()  // 默认从 cwd 向上找 .milkie/agents.json
// 等价于：for each agent in manifest, milkie.loadAgentFile(resolvedPath)

// 也可指定 manifest 路径：
await milkie.loadManifest('./custom/agents.json')
```

SDK consumer **不必**用 manifest——继续手写 `registerAgent()` /
`loadAgentFile()` 完全合法。manifest 是 **CLI 的必须 + SDK 的便利**，
不是 SDK 的强制路径。

## 6. CLI verb 行为细化（更新到 CLI surface spec §4）

### 6.1 Terminology fix

CLI surface spec §4 写的是 `<agentName>`；改为 `<agentId>`，对齐既有
代码的 `agentId` 命名。本 spec sign-off 后单独提一笔 CLI spec 修订。

### 6.2 `milkie agent list`

输出当前已 load 的 agents（manifest load 完毕后 = manifest 声明 + 任
何额外的 in-process register）：

```jsonl
{"id":"router","source":"manifest","file":"/abs/path/to/router.md"}
{"id":"verifier","source":"manifest","file":"/abs/path/to/verifier.md"}
```

| 字段 | 说明 |
|---|---|
| `id` | agent identifier |
| `source` | `manifest` \| `programmatic`（未来扩展：`plugin` 等） |
| `file` | 绝对路径；仅 `source: manifest` 时存在 |

**找不到 manifest 时.** 命令仍正常退出 0，stdout 为空，stderr 提示
"no manifest found"。

### 6.3 不提供 `register` / `unregister` CLI

Registry 在 CLI 短进程里 = manifest 派生；要"注册" agent 就编辑
`.milkie/agents.json`。SDK 仍可在 in-process 里程序化 register（不持久）。
显式不在 CLI 提供 mutate registry 的命令，避免歧义（mutate 一个短进程
的 in-memory 状态对 CLI caller 没意义）。

## 7. 与既有代码的关系

- `Milkie.registerAgent()` / `loadAgentFile()` / `invoke()` 保持原状
- **新增** `Milkie.loadManifest(path?: string): Promise<{ loaded: string[], skipped: { id: string, reason: string }[] }>`
- 内部实现：解析 JSON → 路径相对 manifest 自身解析 → 循环调
  `loadAgentFile()` → 把 frontmatter `agentId` 与 manifest `id` 对账
- CLI entry 在执行任何 verb 前先调 `milkie.loadManifest()`（fail-soft：
  manifest 找不到不抛错，registry 为空）
- 现有 agent file 的 markdown + frontmatter 格式不变；frontmatter 已
  有 `agentId` 字段，跟 manifest `id` 直接对账

## 8. Open questions

- **JSON Schema 提供与否.** 是否随包发一份 JSON Schema 供 IDE
  autocomplete / 校验？倾向：v1 不强制，作为后续 nice-to-have
- **`.milkie/` 的 `.gitignore` 默认.** 如 `cache/` / `runs/` 默认 ignore，
  `agents.json` 默认 git 跟踪。可能在 `milkie init` 命令落地时一并定，
  不在本 spec
- **Agent file 路径相对性.** 已定：**相对 manifest 自身位置**。如果将
  来发现"相对项目根"更符合直觉，可加 `pathBase: "manifest" | "project"`
  字段，向后兼容

## 9. 下一步

按 sign-off 后顺序：

1. **本 spec sign-off** → 把 CLI surface spec §4 的 `<agentName>` 全
   改为 `<agentId>`，关掉 OQ #1，提一笔 docs commit
2. **`Milkie.loadManifest()` 实现** → src/runtime/Milkie.ts 加方法 +
   单元测试
3. **CLI scaffold（薄 wrapper）** → P0 verbs（`agent run/resume/interrupt`
   + `trace inspect/replay`）的 CLI 入口实现；启动时自动 `loadManifest()`
4. **examples/ 第一例（s-005 replay）** → 包含一份最小
   `.milkie/agents.json` 作为 pedagogy demonstration；用户照搬即可在
   自己项目里复刻 manifest 结构

第 3 / 4 步可以并行启动，但都依赖第 2 步。
