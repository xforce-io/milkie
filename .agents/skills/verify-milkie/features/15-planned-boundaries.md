# 规划能力与旧场景边界

## 用户入口

- 没有已确认的 fork/diff/suite/自动变体搜索公共 CLI 或 Milkie 方法；不要发明调用命令

## 源码依据

- [src/cli/main.ts](../../../../src/cli/main.ts)
- [src/runtime/Milkie.ts](../../../../src/runtime/Milkie.ts)
- [docs/stories/INDEX.md](../../../../docs/stories/INDEX.md)
- [roadmap.md](../../../../roadmap.md)

## 驾驶路径与判定

| 路径 | 操作 | 可判定结果 |
|---|---|---|
| 已移除 | 对照 s-011 原叙事、#175 设计与当前 FSMEngine | 多态业务跳转已移出核心，当前单态槽位收集见 03；不能把原 Story active 当成旧能力仍存在。 |
| 规划核对 | 将每个旧 Story 的目标与当前入口对照 | s-006 fork、s-012 batch replay/classification、s-013 bounded variant search 仍不作为已实现入口。 |
| 部分实现 | 分别核对 s-010 Skill 与 A/B；s-015 read-trace 与父执行进行中读取 | 已有基础能力映射 07/12，未证明部分保留缺口，不将整个 Story 直接标 pass。 |

## 验证方法

核对公开方法、CLI 命令和对应 Story；无实现入口的规划项记为未实现，不执行虚构命令。

## 已知缺口

测试目录的 s-012-ioport-deadline-cancellation 与 Story s-012-batch-replay-suite-and-classify-divergences 不是同一场景；s-013-llm-failure-replay 也不是 Story s-013-variant-search-with-bounded-cost。必须按完整文件名和行为匹配，不能只看编号。

## 对应 Stories

- [docs/stories/s-006-fork-at-event-for-what-if.md](../../../../docs/stories/s-006-fork-at-event-for-what-if.md)
- [docs/stories/s-010-skill-versioned-load-and-ab-experiment.md](../../../../docs/stories/s-010-skill-versioned-load-and-ab-experiment.md)
- [docs/stories/s-012-batch-replay-suite-and-classify-divergences.md](../../../../docs/stories/s-012-batch-replay-suite-and-classify-divergences.md)
- [docs/stories/s-013-variant-search-with-bounded-cost.md](../../../../docs/stories/s-013-variant-search-with-bounded-cost.md)
- [docs/stories/s-015-subagent-reads-parent-trace-runtime.md](../../../../docs/stories/s-015-subagent-reads-parent-trace-runtime.md)

- [s-011 原场景](../../../../docs/stories/s-011-multi-state-fsm-intent-routing-and-slot-filling.md)
- [#175 设计](../../../../docs/design/175-decore-multistate-fsm.md)
