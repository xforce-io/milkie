import { v4 as uuid } from 'uuid'
import { checkpointFromEvents } from '../trace/diagnostics/checkpointFromEvents.js'
import matter from 'gray-matter'
import fs from 'fs'
import path from 'path'
import type { AgentConfig, FSMDefinition } from '../types/agent.js'
import type { AgentInvokeRequest, AgentResult } from '../types/common.js'
import type { ChildAgentRecord, IStateStore, AgentCheckpoint } from '../types/store.js'
import type { ToolDefinition } from '../types/tool.js'
import type { IModelGateway } from '../types/model.js'
import type { ResolvedManifest } from '../types/trajectory.js'
import { MemoryStore } from '../store/MemoryStore.js'
import { InMemoryRecorder } from '../trajectory/InMemoryRecorder.js'
import { TrajectoryStore } from '../trajectory/TrajectoryStore.js'
import { createGateway } from '../gateway/GatewayFactory.js'
import { AgentRuntime } from './AgentRuntime.js'
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
import type { ToolEmittedPayload } from '../trace/types.js'

export interface MilkieOptions {
  stateStore?:      IStateStore
  gateway?:         IModelGateway   // override all agents; if omitted, each agent uses its own adapter
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
}

export class Milkie {
  private readonly stateStore:      IStateStore
  private readonly gatewayOverride: IModelGateway | null
  private readonly extraTools:      ToolDefinition[]
  private readonly trajectoryStore: TrajectoryStore | null
  private readonly eventStore:      IEventStore | null
  private readonly traceObjectStore: ITraceObjectStore | null

  private readonly agents:   Map<string, AgentConfig> = new Map()

  constructor(opts: MilkieOptions = {}) {
    this.stateStore      = opts.stateStore      ?? new MemoryStore()
    this.gatewayOverride = opts.gateway          ?? null
    this.extraTools      = opts.tools            ?? []
    this.trajectoryStore = opts.trajectoryStore  ?? null
    this.eventStore      = opts.eventStore       ?? null
    this.traceObjectStore = opts.traceObjectStore ?? null
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
    const gatewayOverride = this.gatewayOverride
    const objectStore = this.traceObjectStore ?? undefined
    return async (childRunId, childConfig, start) => {
      const gw     = gatewayOverride ?? createGateway(childConfig.model!)
      const cursor = new CausalCursor()
      const port   = new RecordingIOPort(new DefaultIOPort(gw), eventStore, childRunId, undefined, objectStore, cursor)
      await port.attach(start)
      return { port, finish: (c) => port.detach(c), cursor }
    }
  }

  registerTool(tool: ToolDefinition): void {
    this.extraTools.push(tool)
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

    const gateway  = this.gatewayOverride ?? createGateway(config.model!)
    const contextId = request.contextId ?? uuid()

    // Check for an existing context checkpoint (multi-turn continuation).
    // #73: the event log is the source of truth for resume state — read the
    // context's latest checkpointed run via the routing pointer and project the
    // checkpoint from its events; fall back to the legacy stateStore blob.
    let restoredCheckpoint: AgentCheckpoint | null = null
    if (request.contextId) {
      const runPtr = await this.stateStore.get(`context:${request.contextId}:checkpoint-run:latest`) as string | undefined
      if (runPtr && this.eventStore) {
        restoredCheckpoint = checkpointFromEvents(await this.eventStore.readByRunId(runPtr))
      }
      if (!restoredCheckpoint) {
        restoredCheckpoint = (await this.stateStore.get(`context:${request.contextId}:checkpoint:latest`) as AgentCheckpoint | null) ?? null
      }
    }

    const agentRunId = uuid()
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

    const runtime = new AgentRuntime({
      config,
      goal:            request.goal,
      input:           request.input,
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
    })

    try {
      const result = await runtime.run(request.input)
      await rec?.detach({ status: result.status, lastTextOutput: result.output })
      return result
    } catch (err) {
      await rec?.detach({ status: 'error', error: err instanceof Error ? err.message : String(err) })
      throw err
    }
  }

  /**
   * Resume execution from a saved checkpoint.
   */
  async resume(checkpointId: string, agentId: string, goal: string, input: string): Promise<AgentResult> {
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

    const gateway = this.gatewayOverride ?? createGateway(config.model!)
    const contextId = checkpoint.meta.contextId ?? uuid()
    const agentRunId = checkpoint.meta.agentRunId
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
      config,
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
    const inner  = new DefaultIOPort(this.gatewayOverride ?? createGateway(config.model!))
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
      async invokeTool(name, input, execute) {
        try {
          return await ioPort.invokeTool(name, input, execute)
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

    const runtime = new AgentRuntime({
      config,
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
      // #60: recorded emit-driven FSM events keyed by (toolCallId, occurrence) →
      // replay re-emits them so emit-driven transitions reproduce without re-running
      // tool handlers. The occurrence suffix keeps a tool_call id reused across
      // responses from collapsing in the map (provider ids are only unique per response).
      replayEmits: new Map(events
        .filter(e => e.type === 'tool.emitted')
        .map(e => {
          const p = e.payload as ToolEmittedPayload
          return [`${p.toolCallId}#${p.occurrence ?? 0}`, p.event] as const
        })),
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

  async interrupt(contextId: string): Promise<void> {
    await this.stateStore.set(`context:${contextId}:interrupt`, true)
    const children = await this.stateStore.get(`context:${contextId}:children`) as ChildAgentRecord[] | undefined
    for (const child of children ?? []) {
      if (child.status === 'running' && child.contextId) {
        await this.stateStore.set(`context:${child.contextId}:interrupt`, true)
      }
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
      model: {
        provider: config.model!.provider,
        model:    config.model!.model,
        adapter:  config.model!.adapter,
        baseUrl:  config.model!.baseUrl,
      },
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
      toolboxes:  data['toolboxes']  as Record<string, string> | undefined,
      skills:     data['skills']     as Record<string, string> | undefined,
      skillInstructions: data['skillInstructions'] as Record<string, string> | undefined,
      subAgents:  (data['sub_agents'] ?? data['subAgents']) as Record<string, string> | undefined,
      stateStore: data['state_store'] as 'memory' | 'sqlite' | 'redis' | undefined,
      dispatch:   data['dispatch']   as 'local' | 'queue' | undefined,
    }
  }
}
