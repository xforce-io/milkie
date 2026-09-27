# 执行记录、汇总与 HTML 报告

## 用户入口

- SDK：`getRunSummary`；`summarizeRun`、`parseJsonlEvents`、`contextAt`、`contextBefore`、`getRegionAt`、`getRegionBefore`
- CLI：`trace inspect <runId>`、`trace summary <runId>`、`trace execution <runId>`、`trace report <runId>`、`trace render-html`；inspect 的 --include-children；--data-dir

## 源码依据

- [src/cli/main.ts](../../../../src/cli/main.ts)
- [src/trace/summarizeRun.ts](../../../../src/trace/summarizeRun.ts)
- [src/trace/RegionContextView.ts](../../../../src/trace/RegionContextView.ts)
- [src/trace/diagnostics/buildExecutionProjection.ts](../../../../src/trace/diagnostics/buildExecutionProjection.ts)
- [src/trace/render/viewer.ts](../../../../src/trace/render/viewer.ts)
- [src/trace/render/html.ts](../../../../src/trace/render/html.ts)

## 驾驶路径与判定

| 路径 | 操作 | 可判定结果 |
|---|---|---|
| 记录与汇总 | 用临时数据目录记录含模型/工具/产物的执行，分别运行 inspect、summary、execution | inspect 事件 ID 与原始 JSONL 对账；summary 的计数、状态、停止原因与原始记录一致；execution 可关联步骤。 |
| 报告 | 分别由 JSONL 调 render-html、由 runId 调 report，浏览器打开输出文件 | 离线 HTML 可读；report 包含后代执行；折叠、步骤选择及上下文显示需在浏览器实际检查。 |
| 坏数据/空 | 截断 JSONL、破坏中间一行、使用不存在 runId；分别驱动各命令 | inspect/summary 应明确非零失败；检查 stdout 是否部分输出；其它命令按实际记录，不能假设共享同一错误契约。 |

## 验证方法

下列命令从仓库根运行；是自动化覆盖入口，不代表上表所有路径已通过。具体执行记录见[首版核验](../references/initial-audit.md)。

```sh
./node_modules/.bin/jest --runInBand --runTestsByPath src/__tests__/traceSummary.inspect.test.ts src/__tests__/CliTrace.test.ts src/__tests__/CliTraceExecution.test.ts src/__tests__/CliTraceReport.test.ts src/__tests__/render-html.test.ts src/__tests__/render-viewer.test.ts src/__tests__/regionReuseCounts.test.ts
```

- [src/__tests__/traceSummary.inspect.test.ts](../../../../src/__tests__/traceSummary.inspect.test.ts)
- [src/__tests__/CliTrace.test.ts](../../../../src/__tests__/CliTrace.test.ts)
- [src/__tests__/CliTraceExecution.test.ts](../../../../src/__tests__/CliTraceExecution.test.ts)
- [src/__tests__/CliTraceReport.test.ts](../../../../src/__tests__/CliTraceReport.test.ts)
- [src/__tests__/render-html.test.ts](../../../../src/__tests__/render-html.test.ts)
- [src/__tests__/render-viewer.test.ts](../../../../src/__tests__/render-viewer.test.ts)
- [src/__tests__/regionReuseCounts.test.ts](../../../../src/__tests__/regionReuseCounts.test.ts)

## 已知缺口

#260/#261 修复后，新恢复 run 独立记录终态，子执行保留停止字段；旧记录不自动回填。INDEX 中旧的“只有全量事件、无 summary”描述不适用于此基线。HTML 视觉/交互首版未做浏览器验收。

## 对应 Stories

- [docs/stories/s-002-inspect-a-completed-run.md](../../../../docs/stories/s-002-inspect-a-completed-run.md)
- [docs/stories/s-003-explain-a-decision-with-context.md](../../../../docs/stories/s-003-explain-a-decision-with-context.md)
