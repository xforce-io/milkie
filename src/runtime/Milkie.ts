import { v4 as uuid } from 'uuid'
import { checkpointFromEvents } from '../trace/diagnostics/checkpointFromEvents.js'
import { runEventsToMessages } from '../trace/diagnostics/sessionHistory.js'
import matter from 'gray-matter'
import fs from 'fs'
import path from 'path'
import type { AgentConfig, FSMDefinition, ModelConfig } from '../types/agent.js'
import type {
  AgentInvokeRequest,
  AgentResult,
  AttachProjectionRequest,
  ContextProjection,
  JSONValue,
  Message,
} from '../types/common.js'
import type { ChildAgentRecord, IStateStore, AgentCheckpoint } from '../types/store.js'
import type { ToolDefinition } from '../types/tool.js'
import type { IModelGateway, ModelEvent, ModelRequest, ModelResponse } from '../types/model.js'
import { aggregateStream } from '../gateway/StreamAggregator.js'
import type { ResolvedManifest } from '../types/trajectory.js'
import { MemoryStore } from '../store/MemoryStore.js'
import { InMemoryRecorder } from '../trajectory/InMemoryRecorder.js'
import { TrajectoryStore } from '../trajectory/TrajectoryStore.js'
import { createGateway } from '../gateway/GatewayFactory.js'
import { AgentRuntime } from './AgentRuntime.js'
import { readCheckpointLifecycle } from './checkpointSchema.js'
import { DefaultIOPort, type IIOPort } from './IOPort.js'
import type { IEventStore } from '../trace/EventStore.js'
import { RecordingIOPort } from '../trace/RecordingIOPort.js'
import { CausalCursor } from '../trace/CausalCursor.js'
import type { ITraceObjectStore } from '../trace/TraceObjectStore.js'
import { CacheIndex } from '../trace/CacheIndex.js'
import { ReplayingIOPort } from '../trace/ReplayingIOPort.js'
import { ReplayError } from '../trace/ReplayError.js'
import { ReplayDivergenceError } from '../trace/ReplayDivergenceError.js'
import { extractRunSnapshot } from '../trace/RunSnapshot.js'
import type { Event, AgentRunStartedPayload } from '../trace/types.js'
import { makeTraceTools } from '../tools/trace.js'
import {
  type PortableSession,
  PORTABLE_SESSION_SCHEMA_VERSION,
  collectRunTree,
} from './PortableSession.js'
import { getLogger, type ServiceLogger } from '../logging/logger.js'

/**
 * #137: read-only run-state of a context, projected from its latest checkpoint.
 * `paused` ⇔ the FSM stopped in the reserved `paused` state (entered only via
 * interrupt); `resumable` ⇔ a paused checkpoint exists to /resume from.
 * `interruptSignaled` is the transient `context:{id}:interrupt` flag — set the
 * moment interrupt is requested, before the turn has yielded to `paused`.
 */
export interface ContextRunState {
  contextId:         string
  exists:            boolean
  paused:            boolean
  resumable:         boolean
  currentState:      string | null
  interruptSignaled: boolean
}

export interface MilkieOptions {
  /**
   * #99: required — no silent in-memory fallback. Choose the backend explicitly:
   * MemoryStore (tests / single-process), SQLiteStore (same-host persistence),
   * RedisStore (cross-process production). MemoryStore does NOT satisfy #83's
   * cross-process context vars.
   */
  stateStore:       IStateStore
  gateway?:         IModelGateway   // override all agents; if omitted, each agent uses its own adapter
  defaultModel?:    ModelConfig     // fallback model for agents that declare no model block
  tools?:           ToolDefinition[]
  trajectoryStore?: TrajectoryStore
  /**
   * Optional Agent Trace event store. When provided, every agent run is
   * recorded as an append-only event stream (LLM and tool I/O paired
   * requested/responded events). The stored events are the run's source
   * of truth for replay/fork/diff/lineage operations.
   *
   * When omitted, runs execute with no event log; Trajectory remains the
   * only observability surface (Phase 2 behavior).
   */
  eventStore?:      IEventStore
  traceObjectStore?: ITraceObjectStore
  /** #79：服务日志 logger；缺省用进程级 getLogger()。测试注入内存 sink。 */
  logger?:          ServiceLogger
}

export class Milkie {
  private readonly stateStore:      IStateStore
  private readonly gatewayOverride: IModelGateway | null
  private readonly defaultModel:    ModelConfig | null
  private readonly extraTools:      ToolDefinition[]
  private readonly trajectoryStore: TrajectoryStore | null
  private readonly eventStore:      IEventStore | null
  private readonly traceObjectStore: ITraceObjectStore | null
  private readonly log:             ServiceLogger

  private readonly agents:   Map<string, AgentConfig> = new Map()

  constructor(opts: MilkieOptions) {
    if (!opts?.stateStore) {
      throw new Error(
        'Milkie requires an explicit stateStore. Pass MemoryStore for tests/single-process, ' +
        'SQLiteStore for same-host persistence, or RedisStore for cross-process production.',
      )
    }
    this.stateStore      = opts.stateStore
    this.gatewayOverride = opts.gateway          ?? null
    this.defaultModel    = opts.defaultModel     ?? null
    this.extraTools      = opts.tools            ?? []
    this.trajectoryStore = opts.trajectoryStore  ?? null
    this.eventStore      = opts.eventStore       ?? null
    this.traceObjectStore = opts.traceObjectStore ?? null
    this.log             = opts.logger           ?? getLogger()

    // #196: read-trace tools (get_execution/get_lineage/get_run_io) are generic
    // self-explain capabilities — register them wherever an eventStore exists,
    // symmetric with the write-side lineageTools that AgentRuntime registers
    // unconditionally. Previously these were only wired via loadStandardAgents(),
    // which the serve --agent path never calls, leaving serve agents unable to
    // self-explain. Dedup is handled downstream (registry registers by name).
    if (this.eventStore) {
      this.extraTools.push(...makeTraceTools(this.eventStore, this.traceObjectStore ?? undefined))
    }
  }

  private resolveModel(config: AgentConfig, tier?: string): ModelConfig | undefined {
    // #126: a given+matched tier wins; otherwise fall back to the default model
    // (no throw — serve configured with only a default stays usable for any tier).
    if (tier && config.models?.[tier]) return config.models[tier]
    if (tier) this.log.warn({ mod: 'runtime', agentId: config.agentId, tier }, 'tier not found; falling back to default model')
    return config.model ?? this.defaultModel ?? undefined
  }

  private resolveGateway(config: AgentConfig, tier?: string): IModelGateway {
    if (this.gatewayOverride) return this.gatewayOverride
    const model = this.resolveModel(config, tier)
    if (!model) {
      throw new Error(
        `Agent "${config.agentId}" has no model and Milkie has no gateway or defaultModel; ` +
        `built-in agents need a gateway or defaultModel at construction.`)
    }
    return createGateway(model, this.log.child({ mod: 'gateway' }))
  }

  private wrapIOPort(gateway: IModelGateway, runId: string, cursor?: CausalCursor): IIOPort {
    const base = new DefaultIOPort(gateway)
    return this.eventStore
      ? new RecordingIOPort(base, this.eventStore, runId, undefined, this.traceObjectStore ?? undefined, cursor)
      : base
  }

  private buildMakeChildPort(): import('./AgentRuntime.js').MakeChildPort | undefined {
    if (!this.eventStore) return undefined
    const eventStore = this.eventStore
    const objectStore = this.traceObjectStore ?? undefined
    return async (childRunId, childConfig, start) => {
      const gw     = this.resolveGateway(childConfig)
      const cursor = new CausalCursor()
      const port   = new RecordingIOPort(new DefaultIOPort(gw), eventStore, childRunId, undefined, objectStore, cursor)
      await port.attach(start)
      return { port, finish: (c) => port.detach(c), cursor }
    }
  }

  registerTool(tool: ToolDefinition): void {
    this.extraTools.push(tool)
  }

  /**
   * Opt-in load of milkie's built-in/standard agents (package-root `agents/`).
   * Also registers the read-Trace tools those agents depend on (when an
   * eventStore is present). Same-id agents loaded afterwards override these.
   */
  loadStandardAgents(): string[] {
    if (this.eventStore) {
      for (const t of makeTraceTools(this.eventStore, this.traceObjectStore ?? undefined)) {
        this.registerTool(t)
      }
    }
    const dir = path.join(__dirname, '..', '..', 'agents')   // src/runtime & dist/runtime both → package-root/agents
    if (!fs.existsSync(dir)) return []
    const loaded: string[] = []
    for (const f of fs.readdirSync(dir)) {
      if (f.endsWith('.md')) loaded.push(this.loadAgentFile(path.join(dir, f)).agentId)
    }
    return loaded
  }

  loadAgentFile(filePath: string): AgentConfig {
    const raw   = fs.readFileSync(filePath, 'utf-8')
    const { data, content } = matter(raw)
    const config = this.parseConfig(data, content.trim())
    this.agents.set(config.agentId, config)
    return config
  }

  registerAgent(config: AgentConfig): void {
    this.agents.set(config.agentId, config)
  }

  getAgent(agentId: string): AgentConfig | undefined {
    return this.agents.get(agentId)
  }

  listAgents(): string[] {
    return Array.from(this.agents.keys())
  }

  /**
   * #124: one-shot LLM completion that bypasses the FSM — no agent run, no event
   * log. Resolves the agent's own gateway/model (the only place model config
   * lives) and calls it directly, mirroring DefaultIOPort.invokeLLM: streaming +
   * aggregation when `onEvent` is provided, a single non-streaming call when not.
   * Exposed over HTTP as serve's `POST /llm` (alfred's `call_llm` equivalent).
   */
  async complete(
    agentId: string,
    request: { system?: string; messages: Message[]; tier?: string; temperature?: number },
    onEvent?: (e: ModelEvent) => void,
  ): Promise<ModelResponse> {
    const config = this.agents.get(agentId)
    if (!config) {
      throw new Error(`Agent not found: "${agentId}". Call registerAgent() or loadAgentFile() first.`)
    }
    const gateway = this.resolveGateway(config, request.tier)
    const req: ModelRequest = {
      model:       this.resolveModel(config, request.tier)?.model ?? '',
      system:      request.system,
      messages:    request.messages,
      temperature: request.temperature,
    }
    return onEvent
      ? aggregateStream(gateway.stream(req), onEvent)
      : gateway.complete(req)
  }

  /**
   * Load every agent declared in a manifest file (`.milkie/agents.json`).
   *
   * Manifest schema:
   *   { "agents": [ { "id": "...", "file": "<path relative to manifest>" } ] }
   *
   * Each entry's `file` is resolved relative to the manifest's own location;
   * each is loaded via `loadAgentFile()`. Returns the ids successfully loaded
   * and the ids skipped with reasons (duplicates, missing files,
   * frontmatter `agentId` mismatch).
   *
   * See `docs/superpowers/specs/2026-05-24-agent-registration-design.md`.
   */
  async loadManifest(manifestPath?: string): Promise<{
    loaded:  string[],
    skipped: { id: string, reason: string }[],
  }> {
    const resolvedPath = manifestPath ?? this.findManifestUpward(process.cwd())
    if (!resolvedPath) {
      return { loaded: [], skipped: [] }
    }
    const raw = fs.readFileSync(resolvedPath, 'utf-8')
    const manifest = JSON.parse(raw) as { agents: Array<{ id: string, file: string }> }
    const manifestDir = path.dirname(resolvedPath)
    const loaded:  string[] = []
    const skipped: { id: string, reason: string }[] = []
    const seen = new Set<string>()
    for (const entry of manifest.agents) {
      if (seen.has(entry.id)) {
        skipped.push({ id: entry.id, reason: 'duplicate id in manifest' })
        continue
      }
      seen.add(entry.id)
      const agentPath = path.resolve(manifestDir, entry.file)
      let config: AgentConfig
      try {
        config = this.loadAgentFile(agentPath)
      } catch (err) {
        const reason = err instanceof Error ? err.message : String(err)
        skipped.push({ id: entry.id, reason })
        continue
      }
      if (config.agentId !== entry.id) {
        // loadAgentFile registered under the frontmatter id; revert it
        this.agents.delete(config.agentId)
        skipped.push({
          id: entry.id,
          reason: `agentId mismatch: manifest "${entry.id}" vs frontmatter "${config.agentId}"`,
        })
        continue
      }
      loaded.push(entry.id)
    }
    return { loaded, skipped }
  }

  private findManifestUpward(startDir: string): string | undefined {
    let dir = startDir
    while (true) {
      const candidate = path.join(dir, '.milkie', 'agents.json')
      if (fs.existsSync(candidate)) return candidate
      const parent = path.dirname(dir)
      if (parent === dir) return undefined
      dir = parent
    }
  }

  async invoke(request: AgentInvokeRequest): Promise<AgentResult> {
    const config = this.agents.get(request.agentId)
    if (!config) {
      throw new Error(`Agent not found: "${request.agentId}". Call registerAgent() or loadAgentFile() first.`)
    }

    const gateway  = this.resolveGateway(config)
    const contextId = request.contextId ?? uuid()

    // Check for an existing context checkpoint (multi-turn continuation).
    // #73: the event log is the source of truth for resume state — read the
    // context's latest checkpointed run via the routing pointer and project the
    // checkpoint from its events; fall back to the legacy stateStore blob.
    // #128: the run this turn continues from (= the session's previous run). Used
    // both to restore the checkpoint and to chain runs for getSessionHistory.
    let restoredCheckpoint: AgentCheckpoint | null = null
    let previousRunId: string | undefined
    if (request.contextId) {
      previousRunId = await this.stateStore.get(`context:${request.contextId}:checkpoint-run:latest`) as string | undefined
      if (previousRunId && this.eventStore) {
        restoredCheckpoint = checkpointFromEvents(await this.eventStore.readByRunId(previousRunId))
      }
      if (!restoredCheckpoint) {
        restoredCheckpoint = (await this.stateStore.get(`context:${request.contextId}:checkpoint:latest`) as AgentCheckpoint | null) ?? null
      }
    }

    const agentRunId = uuid()
    const runtimeConfig = { ...config, model: this.resolveModel(config) }
    const childRecorderFactory = this.trajectoryStore
      ? (childConfig: AgentConfig, childContextId: string, childTraceId: string) =>
        this.trajectoryStore!.makeRecorder({
          agentRunId,
          contextId: childContextId,
          agentId: childConfig.agentId,
          traceId: childTraceId,
          resolvedManifest: this.buildResolvedManifest(childConfig),
        })
      : undefined
    const recorder = this.trajectoryStore
      ? this.trajectoryStore.makeRecorder({
        agentRunId,
        contextId,
        agentId: config.agentId,
        resolvedManifest: this.buildResolvedManifest(config),
      })
      : new InMemoryRecorder(undefined, config.agentId)

    const causalCursor = new CausalCursor()
    const ioPort = this.wrapIOPort(gateway, agentRunId, causalCursor)
    const makeChildPort = this.buildMakeChildPort()

    // #83: snapshot this context's persistent vars at invoke entry (next-invoke visibility).
    const sessionVariables = await this.listContextVars(contextId)
    // #146: snapshot delivered projections at invoke entry; later attaches are
    // visible on the next invoke, matching #83's isolation semantics.
    const externalProjections = await this.listContextProjections(contextId)

    const runtime = new AgentRuntime({
      config: runtimeConfig,
      goal:            request.goal,
      input:           request.input,
      variables:       request.variables,  // #82: per-turn variables → turn-context region
      sessionVariables,                    // #83: persistent session vars → session-context region
      externalProjections,                 // #146: external delivered reports → external-context region
      contextId,
      agentRunId,
      stateStore:      this.stateStore,
      recorder,
      ioPort,
      eventStore:      this.eventStore ?? undefined,
      traceObjectStore: this.traceObjectStore ?? undefined,
      extraTools:      this.extraTools,
      subAgentConfigs: this.agents,
      childRecorderFactory,
      makeChildPort,
      causalCursor,
      ...(previousRunId ? { previousRunId } : {}),
      ...(request.onModelEvent ? { onModelEvent: request.onModelEvent } : {}),
    })

    if (restoredCheckpoint) {
      await runtime.loadCheckpoint(restoredCheckpoint)
    }

    const rec = ioPort instanceof RecordingIOPort ? ioPort : null
    await rec?.attach({
      agentId:   config.agentId,
      goal:      request.goal,
      input:     request.input,
      contextId,
      ...(previousRunId ? { previousRunId } : {}),
    })

    // #79：每 invoke 一条服务日志 wide event（边界汇总，设计 §5）。
    // turns/token 不在 AgentResult 上，token 已由 LoggingGateway 按调用记录。
    const invokeLog = this.log.child({ mod: 'runtime', runId: agentRunId, contextId })
    const invokeStartedAt = Date.now()
    try {
      const result = await runtime.run(request.input)
      await rec?.detach({ status: result.status, lastTextOutput: result.output })
      invokeLog.info({ agentId: config.agentId, durationMs: Date.now() - invokeStartedAt, status: result.status }, 'invoke completed')
      return result
    } catch (err) {
      await rec?.detach({ status: 'error', error: err instanceof Error ? err.message : String(err) })
      invokeLog.error({ agentId: config.agentId, durationMs: Date.now() - invokeStartedAt, err }, 'invoke failed')
      throw err
    }
  }

  /**
   * Resume execution from a saved checkpoint.
   */
  async resume(checkpointId: string, agentId: string, goal: string, input: string, opts?: { onModelEvent?: (e: ModelEvent) => void }): Promise<AgentResult> {
    const config = this.agents.get(agentId)
    if (!config) {
      throw new Error(`Agent not found: "${agentId}". Call registerAgent() or loadAgentFile() first.`)
    }

    // #73: resolve the resume state from the event log (source of truth). The
    // checkpointId key identifies a run (directly, or via the context routing
    // pointer); project the checkpoint from that run's agent.checkpoint event.
    // Fall back to a stateStore blob under the key (legacy / manually-seeded).
    const checkpoint = await this.resolveCheckpoint(checkpointId)
    if (!checkpoint) {
      throw new Error(`Checkpoint not found: "${checkpointId}"`)
    }

    const gateway = this.resolveGateway(config)
    const contextId = checkpoint.meta.contextId ?? uuid()
    const agentRunId = checkpoint.meta.agentRunId
    const runtimeConfig = { ...config, model: this.resolveModel(config) }
    const childRecorderFactory = this.trajectoryStore
      ? (childConfig: AgentConfig, childContextId: string, childTraceId: string) =>
        this.trajectoryStore!.makeRecorder({
          agentRunId,
          contextId: childContextId,
          agentId: childConfig.agentId,
          traceId: childTraceId,
          resolvedManifest: this.buildResolvedManifest(childConfig),
        })
      : undefined

    const recorder = this.trajectoryStore
      ? this.trajectoryStore.makeRecorder({
        agentRunId,
        contextId,
        agentId: config.agentId,
        traceId: checkpoint.meta.traceId || undefined,
        resolvedManifest: this.buildResolvedManifest(config),
      })
      : new InMemoryRecorder(checkpoint.meta.traceId || undefined, config.agentId)

    const makeChildPort = this.buildMakeChildPort()
    const causalCursor = new CausalCursor()

    const runtime = new AgentRuntime({
      config: runtimeConfig,
      goal,
      input,
      agentRunId,
      contextId,
      stateStore: this.stateStore,
      recorder,
      ioPort:          this.wrapIOPort(gateway, agentRunId, causalCursor),
      eventStore:      this.eventStore ?? undefined,
      traceObjectStore: this.traceObjectStore ?? undefined,
      extraTools:      this.extraTools,
      subAgentConfigs: this.agents,
      childRecorderFactory,
      makeChildPort,
      causalCursor,
      ...(opts?.onModelEvent ? { onModelEvent: opts.onModelEvent } : {}),
    })

    await runtime.loadCheckpoint(checkpoint)

    return runtime.run(input)
  }

  /**
   * #73: resolve a checkpoint from the event log (single source of truth) for a
   * checkpointId key. The key identifies a run either directly (a
   * `…:run:<runId>:checkpoint…` key) or via the `context:<id>:checkpoint:latest`
   * convention (resolved through the context→runId routing pointer). Falls back
   * to a stateStore blob stored under the key (legacy or test-seeded).
   */
  private async resolveCheckpoint(checkpointId: string): Promise<AgentCheckpoint | null> {
    if (this.eventStore) {
      let runId: string | undefined
      const ctxMatch = checkpointId.match(/^context:(.+):checkpoint(?::latest)?$/)
      const runMatch = checkpointId.match(/:run:([^:]+):checkpoint/)
      if (ctxMatch) {
        runId = (await this.stateStore.get(`context:${ctxMatch[1]}:checkpoint-run:latest`) as string | undefined) ?? undefined
      } else if (runMatch) {
        runId = runMatch[1]
      }
      if (runId) {
        const cp = checkpointFromEvents(await this.eventStore.readByRunId(runId))
        if (cp) return cp
      }
    }
    return (await this.stateStore.get(checkpointId) as AgentCheckpoint | null) ?? null
  }

  /**
   * Re-run a recorded agent run from its event log. Every IIOPort method
   * (LLM, tool, port.now, port.uuid) is served from the event-derived
   * CacheIndex — no live calls. Replay is byte-identical for any value
   * the agent observes through the port: paired req/resp matched by
   * canonical request hash for LLM/tool, position-FIFO for clock/uuid.
   *
   * Strict P-wide divergence:
   *  - over-consume on any queue → ReplayDivergenceError immediately
   *  - under-consume on any queue at replay tail → ReplayDivergenceError
   *
   * Constraints:
   *  - Requires this Milkie has an eventStore configured
   *  - Requires this Milkie has the original agentId registered
   *  - Throws ReplayError on structural failures (missing run, missing
   *    lifecycle event, unknown agentId)
   *  - Throws ReplayDivergenceError on any LLM/tool/clock/uuid divergence
   */
  async replay(runId: string): Promise<AgentResult> {
    if (!this.eventStore) {
      throw new ReplayError('Milkie has no eventStore; cannot replay')
    }

    const events = await this.eventStore.readByRunId(runId)
    const snapshot = extractRunSnapshot(events)

    const config = this.agents.get(snapshot.agentId)
    if (!config) {
      throw new ReplayError(`agentId "${snapshot.agentId}" not registered on this Milkie instance`)
    }

    const cache  = CacheIndex.fromEvents(events)
    const inner  = new DefaultIOPort(this.resolveGateway(config))
    const ioPort = new ReplayingIOPort(cache, inner)

    const recorder = new InMemoryRecorder(undefined, config.agentId)

    // Capture any ReplayDivergenceError thrown by the ioPort so we can
    // re-throw it after AgentRuntime.run() returns (run() swallows all
    // errors into status:'error' results).
    let divergenceError: ReplayDivergenceError | undefined

    const proxyPort: IIOPort = {
      async invokeLLM(req) {
        try {
          return await ioPort.invokeLLM(req)
        } catch (err) {
          if (err instanceof ReplayDivergenceError) divergenceError = err
          throw err
        }
      },
      async invokeTool(name, input, execute, opts) {
        try {
          return await ioPort.invokeTool(name, input, execute, opts)
        } catch (err) {
          if (err instanceof ReplayDivergenceError) divergenceError = err
          throw err
        }
      },
      now: () => {
        try { return ioPort.now() }
        catch (err) {
          if (err instanceof ReplayDivergenceError) divergenceError = err
          throw err
        }
      },
      uuid: () => {
        try { return ioPort.uuid() }
        catch (err) {
          if (err instanceof ReplayDivergenceError) divergenceError = err
          throw err
        }
      },
    }

    const replayRuntimeConfig = { ...config, model: this.resolveModel(config) }
    const runtime = new AgentRuntime({
      config: replayRuntimeConfig,
      goal:            snapshot.goal,
      input:           snapshot.input,
      contextId:       snapshot.contextId,
      agentRunId:      runId,
      parentId:        snapshot.parentId,
      stateStore:      new MemoryStore(),  // ephemeral
      recorder,
      ioPort:          proxyPort,            // NOT wrapped — replay writes no events
      extraTools:      this.extraTools,
      subAgentConfigs: this.agents,
      childRecorderFactory: undefined,
      // SPIKE(#73): recorded WM snapshots (one per tool call) → replay restores
      // tool-written working memory the handler (not re-run) would have produced.
      replayWmSnapshots: events
        .filter(e => e.type === 'wm.mutated')
        .map(e => (e.payload as { snapshot: unknown }).snapshot),
    })

    const result = await runtime.run(snapshot.input)
    if (divergenceError) throw divergenceError

    // P-wide strict under-consume check: any recorded event the replay
    // failed to consume signals divergence (the run took a different path
    // than recording, or recording captured events the runtime no longer
    // emits). Check all four queues.
    const remaining = cache.remaining()
    for (const kind of ['clock', 'uuid', 'llm', 'tool'] as const) {
      const n = remaining[kind]
      if (n > 0) {
        throw new ReplayDivergenceError(
          kind, '',
          `${n} ${kind} event(s) unconsumed after replay completed`,
          []
        )
      }
    }
    return result
  }

  // ---- #84: portable session export/import ----
  // The event log is the source of truth (#73): export bundles the run-tree's
  // events (latest checkpointed run + sub-agent descendants) plus the context's
  // persistent vars (#83) under a versioned manifest. Import re-injects them into
  // this instance so `invoke({ contextId })` continues with prior history.

  /**
   * Export a context's session as a portable, serialisable snapshot.
   * Requires an eventStore (events are the payload). Throws when the context has
   * no checkpointed run to export.
   */
  async exportSession(contextId: string): Promise<PortableSession> {
    if (!this.eventStore) {
      throw new Error('Milkie has no eventStore; cannot export a portable session')
    }
    const latestRunId = await this.stateStore.get(`context:${contextId}:checkpoint-run:latest`) as string | undefined
    if (!latestRunId) {
      throw new Error(`No session to export for contextId "${contextId}"`)
    }

    const events = await collectRunTree(this.eventStore, latestRunId)

    // agentId: prefer the latest run's checkpoint meta, fall back to its
    // agent.run.started event (a run-tree always has at least the root's lifecycle).
    const latestRunEvents = events.filter(e => e.runId === latestRunId)
    const cp = checkpointFromEvents(latestRunEvents)
    const started = latestRunEvents.find(e => e.type === 'agent.run.started')
    const agentId = cp?.meta.agentId
      ?? (started?.payload as AgentRunStartedPayload | undefined)?.agentId
      ?? ''

    return {
      manifest: {
        schemaVersion: PORTABLE_SESSION_SCHEMA_VERSION,
        contextId,
        agentId,
        latestRunId,
        exportedAt: Date.now(),
      },
      events,
      variables: await this.listContextVars(contextId),
    }
  }

  /**
   * Import a portable session into this instance: append its run-tree events
   * into the eventStore, install the context→run routing pointer, and restore
   * its persistent vars. Afterwards `invoke({ contextId })` continues the
   * conversation with the prior history. Returns the contextId now installed.
   */
  async importSession(session: PortableSession): Promise<{ contextId: string }> {
    if (!this.eventStore) {
      throw new Error('Milkie has no eventStore; cannot import a portable session')
    }
    if (session.manifest.schemaVersion !== PORTABLE_SESSION_SCHEMA_VERSION) {
      throw new Error(
        `Unsupported portable session schemaVersion ${session.manifest.schemaVersion}; ` +
        `this build imports v${PORTABLE_SESSION_SCHEMA_VERSION}`)
    }
    const { contextId, latestRunId } = session.manifest

    for (const event of session.events) {
      await this.eventStore.append(event)
    }
    // Routing pointer: invoke() reads this to project the resume checkpoint from
    // the event log (#73).
    await this.stateStore.set(`context:${contextId}:checkpoint-run:latest`, latestRunId)

    for (const [name, value] of Object.entries(session.variables)) {
      await this.setContextVar(contextId, name, value)
    }
    return { contextId }
  }

  /**
   * #128: the full per-message transcript of a whole session — every run/turn
   * under `contextId`, tool chains intact, in turn order. Unlike exportSession's
   * forward state snapshot, this walks each run's event log (the only place the
   * inter-turn-dropped tool chain survives) and concatenates the per-run
   * projections. Throws when the context has no runs.
   *
   * The session's runs are enumerated from the event log itself: start at the
   * latest run (`checkpoint-run:latest`) and walk backwards via each run's
   * `agent.run.started.previousRunId`. This needs no separate, non-atomic
   * by-context index, so it survives export/import (the latest pointer is the only
   * thing import must set, which it already does) and has no concurrent-append
   * lost-update. The walk stops where events are absent (e.g. an imported session
   * carries only the latest run-tree), degrading gracefully rather than failing.
   */
  async getSessionHistory(contextId: string): Promise<Message[]> {
    if (!this.eventStore) {
      throw new Error('Milkie has no eventStore; cannot read session history')
    }
    const latest = await this.stateStore.get(`context:${contextId}:checkpoint-run:latest`) as string | undefined
    if (!latest) {
      throw new Error(`No session for contextId "${contextId}"`)
    }

    // Walk the run chain newest→oldest, collecting each run's events, then reverse
    // to chronological order. Guard against cycles and missing runs.
    const chain: Event[][] = []
    const seen = new Set<string>()
    let runId: string | undefined = latest
    while (runId && !seen.has(runId)) {
      seen.add(runId)
      const events = await this.eventStore.readByRunId(runId)
      if (events.length === 0) break
      chain.push(events)
      const started = events.find(e => e.type === 'agent.run.started')
      runId = (started?.payload as AgentRunStartedPayload | undefined)?.previousRunId
    }

    const messages: Message[] = []
    for (const events of chain.reverse()) {
      messages.push(...runEventsToMessages(events))
    }
    return messages
  }

  async interrupt(contextId: string): Promise<void> {
    await this.stateStore.set(`context:${contextId}:interrupt`, true)
    const children = await this.stateStore.get(`context:${contextId}:children`) as ChildAgentRecord[] | undefined
    for (const child of children ?? []) {
      if (child.status === 'running' && child.contextId) {
        await this.stateStore.set(`context:${child.contextId}:interrupt`, true)
      }
    }
  }

  /**
   * #137: read-only run-state query for a context. Projects the latest
   * checkpoint (event-sourced, via resolveCheckpoint) and reports whether the
   * context is interrupt-paused and resumable — letting an external provider
   * decide resume-vs-stop without driving a turn. A completed or never-run
   * context is not paused (`paused` ⇔ FSM stopped in the reserved `paused` state).
   */
  async getContextState(contextId: string): Promise<ContextRunState> {
    const interruptSignaled = (await this.stateStore.get(`context:${contextId}:interrupt`)) === true
    const checkpoint = await this.resolveCheckpoint(`context:${contextId}:checkpoint:latest`)
    if (!checkpoint) {
      return { contextId, exists: false, paused: false, resumable: false, currentState: null, interruptSignaled }
    }
    // #175 §8/D7: read lifecycle via the v1/v2 seam, not checkpoint.fsm. Keep the
    // external `currentState: 'paused'` string for a suspended context (the
    // documented resumable signal) so this surface is stable across v1/v2.
    const lc = readCheckpointLifecycle(checkpoint)
    const currentState = lc.suspended ? 'paused' : lc.status
    return { contextId, exists: true, paused: lc.suspended, resumable: lc.suspended, currentState, interruptSignaled }
  }

  // ---- #83: session context variables (key contract: context:{id}:var:{name}) ----
  // Storage handles JSON (de)serialization; this layer only namespaces keys.
  // The `:var:` segment keeps these isolated from checkpoint/interrupt/children keys.

  private varKey(contextId: string, name: string): string {
    return `context:${contextId}:var:${name}`
  }

  async getContextVar(contextId: string, name: string): Promise<JSONValue | undefined> {
    return (await this.stateStore.get(this.varKey(contextId, name))) as JSONValue | undefined
  }

  async setContextVar(contextId: string, name: string, value: JSONValue, ttl?: number): Promise<void> {
    await this.stateStore.set(this.varKey(contextId, name), value, ttl)
  }

  async deleteContextVar(contextId: string, name: string): Promise<void> {
    await this.stateStore.delete(this.varKey(contextId, name))
  }

  async listContextVars(contextId: string): Promise<Record<string, JSONValue>> {
    const prefix = `context:${contextId}:var:`
    const entries = await this.stateStore.list(prefix)
    const out: Record<string, JSONValue> = {}
    for (const { key, value } of entries) {
      out[key.slice(prefix.length)] = value as JSONValue
    }
    return out
  }

  // ---- #146: delivered external context projections ----

  private projectionPrefix(contextId: string): string {
    return `context:${contextId}:projection:`
  }

  private projectionKey(contextId: string, sourceRunId: string): string {
    return `${this.projectionPrefix(contextId)}${sourceRunId}`
  }

  async attachProjection(contextId: string, request: AttachProjectionRequest): Promise<ContextProjection> {
    if (!request.sourceRunId) throw new Error('sourceRunId is required')
    if (!request.displayText) throw new Error('displayText is required')
    if (request.bound?.maxCount !== undefined && request.bound.maxCount < 1) {
      throw new Error('bound.maxCount must be at least 1')
    }

    const projection: ContextProjection = {
      sourceRunId: request.sourceRunId,
      ...(request.sourceContextId ? { sourceContextId: request.sourceContextId } : {}),
      displayText: request.displayText,
      ...(request.summary ? { summary: request.summary } : {}),
      deliveredAt: request.deliveredAt ?? Date.now(),
      attachedAt:  Date.now(),
    }

    await this.stateStore.set(
      this.projectionKey(contextId, request.sourceRunId),
      projection,
      request.bound?.ttl,
    )

    await this.trimContextProjections(contextId, request.bound?.maxCount ?? 5)
    return projection
  }

  async listContextProjections(contextId: string): Promise<ContextProjection[]> {
    const entries = await this.stateStore.list(this.projectionPrefix(contextId))
    return entries
      .map(e => e.value as ContextProjection)
      .sort(byProjectionTime)
  }

  private async trimContextProjections(contextId: string, maxCount: number): Promise<void> {
    if (maxCount < 1) return
    const entries = await this.stateStore.list(this.projectionPrefix(contextId))
    const sorted = entries
      .map(e => ({ key: e.key, value: e.value as ContextProjection }))
      .sort((a, b) => byProjectionTime(a.value, b.value))
    const excess = sorted.length - maxCount
    if (excess <= 0) return
    for (const item of sorted.slice(0, excess)) {
      await this.stateStore.delete(item.key)
    }
  }

  private buildResolvedManifest(config: AgentConfig): ResolvedManifest {
    const tools: Record<string, { version: string }> = {}
    for (const tool of this.extraTools) {
      tools[tool.name] = { version: 'local' }
    }

    const subAgents: Record<string, { version: string }> = {}
    for (const [agentId, pinnedVersion] of Object.entries(config.subAgents ?? {})) {
      const resolved = this.agents.get(agentId)
      subAgents[agentId] = { version: resolved?.version ?? pinnedVersion }
    }

    return {
      agentId:      config.agentId,
      agentVersion: config.version,
      model: config.model ? {
        provider: config.model.provider,
        model:    config.model.model,
        adapter:  config.model.adapter,
        baseUrl:  config.model.baseUrl,
      } : undefined,
      tools,
      toolboxes: Object.fromEntries(
        Object.entries(config.toolboxes ?? {}).map(([name, version]) => [name, { version }]),
      ),
      skills: Object.fromEntries(
        Object.entries(config.skills ?? {}).map(([name, version]) => [name, { version }]),
      ),
      subAgents,
    }
  }

  private parseConfig(data: Record<string, unknown>, systemPrompt: string): AgentConfig {
    const fsm = data['fsm'] as { states: unknown[] } | undefined
    if (!fsm || !Array.isArray(fsm.states)) {
      throw new Error('Agent config must have fsm.states')
    }

    const model = data['model'] as Record<string, string> | undefined
    if (model && (!model.provider || !model.model || !model.adapter)) {
      throw new Error('Agent config model must have provider, model, adapter')
    }

    // #126: open named model tiers. Each tier is validated like the default model.
    const rawModels = data['models'] as Record<string, Record<string, string>> | undefined
    const models = rawModels
      ? Object.fromEntries(Object.entries(rawModels).map(([tier, m]) => {
          if (!m || !m.provider || !m.model || !m.adapter) {
            throw new Error(`Agent config models.${tier} must have provider, model, adapter`)
          }
          return [tier, {
            provider: m['provider']!,
            model:    m['model']!,
            adapter:  m['adapter']!,
            baseUrl:  m['baseUrl'] as string | undefined,
          }]
        }))
      : undefined

    const agentId =
      (data['agentId'] as string | undefined) ??
      (data['id']      as string | undefined)
    if (!agentId) throw new Error('Agent config must have agentId or id')

    return {
      agentId,
      version:      (data['version'] as string | undefined) ?? '0.0.0',
      systemPrompt,
      fsm:          fsm as FSMDefinition,
      model: model ? {
        provider: model['provider']!,
        model:    model['model']!,
        adapter:  model['adapter']!,
        baseUrl:  model['baseUrl'] as string | undefined,
      } : undefined,
      models,
      toolboxes:  data['toolboxes']  as Record<string, string> | undefined,
      skills:     data['skills']     as Record<string, string> | undefined,
      skillInstructions: data['skillInstructions'] as Record<string, string> | undefined,
      subAgents:  (data['sub_agents'] ?? data['subAgents']) as Record<string, string> | undefined,
      dispatch:   data['dispatch']   as 'local' | 'queue' | undefined,
    }
  }
}

function byProjectionTime(a: ContextProjection, b: ContextProjection): number {
  if (a.deliveredAt !== b.deliveredAt) return a.deliveredAt - b.deliveredAt
  if (a.attachedAt !== b.attachedAt) return a.attachedAt - b.attachedAt
  return a.sourceRunId.localeCompare(b.sourceRunId)
}
