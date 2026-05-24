import { v4 as uuid } from 'uuid'
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
import { CacheIndex } from '../trace/CacheIndex.js'
import { ReplayingIOPort } from '../trace/ReplayingIOPort.js'
import { ReplayError } from '../trace/ReplayError.js'
import { ReplayDivergenceError } from '../trace/ReplayDivergenceError.js'
import { extractRunSnapshot } from '../trace/RunSnapshot.js'

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
}

export class Milkie {
  private readonly stateStore:      IStateStore
  private readonly gatewayOverride: IModelGateway | null
  private readonly extraTools:      ToolDefinition[]
  private readonly trajectoryStore: TrajectoryStore | null
  private readonly eventStore:      IEventStore | null

  private readonly agents:   Map<string, AgentConfig> = new Map()

  constructor(opts: MilkieOptions = {}) {
    this.stateStore      = opts.stateStore      ?? new MemoryStore()
    this.gatewayOverride = opts.gateway          ?? null
    this.extraTools      = opts.tools            ?? []
    this.trajectoryStore = opts.trajectoryStore  ?? null
    this.eventStore      = opts.eventStore       ?? null
  }

  private wrapIOPort(gateway: IModelGateway, runId: string): IIOPort {
    const base = new DefaultIOPort(gateway)
    return this.eventStore
      ? new RecordingIOPort(base, this.eventStore, runId)
      : base
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

    const gateway  = this.gatewayOverride ?? createGateway(config.model)
    const contextId = request.contextId ?? uuid()

    // Check for an existing context checkpoint (multi-turn continuation)
    let restoredCheckpoint: AgentCheckpoint | null = null
    if (request.contextId) {
      const cpKey = `context:${request.contextId}:checkpoint:latest`
      restoredCheckpoint = (await this.stateStore.get(cpKey) as AgentCheckpoint | null) ?? null
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

    const ioPort = this.wrapIOPort(gateway, agentRunId)

    const runtime = new AgentRuntime({
      config,
      goal:            request.goal,
      input:           request.input,
      contextId,
      agentRunId,
      stateStore:      this.stateStore,
      recorder,
      ioPort,
      extraTools:      this.extraTools,
      subAgentConfigs: this.agents,
      childRecorderFactory,
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

    // Load checkpoint by its direct store key
    const checkpoint = await this.stateStore.get(checkpointId) as AgentCheckpoint | null
    if (!checkpoint) {
      throw new Error(`Checkpoint not found: "${checkpointId}"`)
    }

    const gateway = this.gatewayOverride ?? createGateway(config.model)
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

    const runtime = new AgentRuntime({
      config,
      goal,
      input,
      agentRunId,
      contextId,
      stateStore: this.stateStore,
      recorder,
      ioPort:          this.wrapIOPort(gateway, agentRunId),
      extraTools:      this.extraTools,
      subAgentConfigs: this.agents,
      childRecorderFactory,
    })

    await runtime.loadCheckpoint(checkpoint)

    return runtime.run(input)
  }

  /**
   * Re-run a recorded agent run from its event log; all LLM/tool I/O
   * is served from the event-derived CacheIndex — no live calls. Result
   * is structurally equivalent to the original run (status, output);
   * timestamps and UUIDs are not guaranteed identical (byte-identical
   * replay is Phase 4).
   *
   * Phase 3 constraints:
   *  - Requires this Milkie has an eventStore configured
   *  - Requires this Milkie has the original agentId registered
   *  - Throws ReplayError on structural failures (missing run, missing
   *    lifecycle event, unknown agentId)
   *  - Throws ReplayDivergenceError when the replayed agent issues an
   *    LLM/tool call whose hash is not in the recorded cache
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
    const inner  = new DefaultIOPort(this.gatewayOverride ?? createGateway(config.model))
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
      now:  () => ioPort.now(),
      uuid: () => ioPort.uuid(),
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
    })

    const result = await runtime.run(snapshot.input)
    if (divergenceError) throw divergenceError
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
        provider: config.model.provider,
        model:    config.model.model,
        adapter:  config.model.adapter,
        baseUrl:  config.model.baseUrl,
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
    if (!model?.provider || !model?.model || !model?.adapter) {
      throw new Error('Agent config must have model.provider, model.model, model.adapter')
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
      model: {
        provider: model['provider']!,
        model:    model['model']!,
        adapter:  model['adapter']!,
        baseUrl:  model['baseUrl'] as string | undefined,
      },
      toolboxes:  data['toolboxes']  as Record<string, string> | undefined,
      skills:     data['skills']     as Record<string, string> | undefined,
      skillInstructions: data['skillInstructions'] as Record<string, string> | undefined,
      subAgents:  (data['sub_agents'] ?? data['subAgents']) as Record<string, string> | undefined,
      stateStore: data['state_store'] as 'memory' | 'sqlite' | 'redis' | undefined,
      dispatch:   data['dispatch']   as 'local' | 'queue' | undefined,
    }
  }
}
