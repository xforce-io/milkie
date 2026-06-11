import pino from 'pino'

/**
 * #79 服务日志（service log）：运维遥测层，与 event log（业务真相）、trajectory
 * （span 诊断）三层并存，用 runId/contextId 关联。边界纪律、字段 schema 见
 * docs/design/79-service-logging.md。
 *
 * 类型上只 re-export pino.Logger —— 不在 pino 外造抽象层，调用方不直接 import pino。
 */
export type ServiceLogger = pino.Logger

export interface ServiceLoggerOptions {
  level?:       string                    // 默认 LOG_LEVEL ?? 'info'
  format?:      'json' | 'pretty'         // 默认 LOG_FORMAT ?? (stderr TTY ? pretty : json)
  destination?: pino.DestinationStream    // 默认 stderr；测试注入内存 sink
}

/** format 决策独立成纯函数，TTY/env 分支可单测。 */
export function resolveFormat(env: { LOG_FORMAT?: string }, isTTY: boolean): 'json' | 'pretty' {
  if (env.LOG_FORMAT === 'pretty') return 'pretty'
  if (env.LOG_FORMAT === 'json') return 'json'
  return isTTY ? 'pretty' : 'json'
}

export function createServiceLogger(opts: ServiceLoggerOptions = {}): ServiceLogger {
  const level  = opts.level ?? process.env['LOG_LEVEL'] ?? 'info'
  const format = opts.format ?? resolveFormat(
    { LOG_FORMAT: process.env['LOG_FORMAT'] },
    process.stderr.isTTY === true,
  )
  // 日志走 stderr：stdout 已被程序输出占用（CLI 结果、MILKIE_SERVE_READY 端口标记）。
  let destination: pino.DestinationStream | undefined = opts.destination
  if (!destination && format === 'pretty') {
    try {
      // pino-pretty 是 devDependency；生产未安装时静默回退 json（即 undefined → 下方 stderr）
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const pretty = require('pino-pretty') as (o: object) => pino.DestinationStream
      destination = pretty({ destination: 2 })
    } catch { /* fall back to json on stderr */ }
  }
  destination ??= pino.destination(2)
  return pino({
    level,
    base: undefined,                                            // 去掉 pid/hostname
    timestamp: () => `,"ts":"${new Date().toISOString()}"`,     // 字段名用 ts（设计 §4）
    formatters: { level: label => ({ level: label }) },         // 字符串标签而非数字
    serializers: { err: pino.stdSerializers.err },
  }, destination)
}

let defaultLogger: ServiceLogger | undefined

/** 惰性默认单例（读环境变量）。模块级埋点（如 tools/system.ts）从这里取。 */
export function getLogger(): ServiceLogger {
  defaultLogger ??= createServiceLogger()
  return defaultLogger
}

/** 测试注入用：换掉单例；传 undefined 恢复惰性默认。 */
export function setLogger(logger: ServiceLogger | undefined): void {
  defaultLogger = logger
}
