import { v4 as uuidv4 } from 'uuid'
import { createHash } from 'crypto'
import type { AgentConfig, FSMState } from '../types/agent.js'
import type { AgentResult } from '../types/common.js'
import { MaxIterationsError } from '../types/common.js'
import type { IStateStore, AgentCheckpoint, ChildAgentRecord } from '../types/store.js'
import type { ITrajectoryRecorder, Span } from '../types/trajectory.js'
import type { ToolDefinition, ToolContext, ToolResult, ToolResultStrategy } from '../types/tool.js'
import { applyShape, serializeOutput } from './toolResultStrategy.js'
import type { MessageContent } from '../types/common.js'
import { FSMEngine } from '../fsm/FSMEngine.js'
import { ContextRegions } from '../context/ContextRegions.js'
import { assemble, type AssembleScope } from '../context/assemble.js'
import {
  makeHeaderRegion,
  makeSkillRegion,
  makeCurrentTurnRegion,
  makeScratchpadAssistantRegion,
  makeScratchpadToolResultRegion,
  makeStateInstructionsRegion,
  makeWmRegion,
  makeToolSchemaRegion,
  runInterTurnEngine,
  rehydrateSnapshot,
} from '../context/lifecycleEngine.js'
import type { ToolSchema, ModelRequest } from '../types/model.js'
import { ToolRegistry } from '../tools/ToolRegistry.js'
import { WorkingMemory } from '../store/WorkingMemory.js'
import { CheckpointManager } from '../store/CheckpointManager.js'
import { AgentFactory, type AgentSpawnOptions } from './AgentFactory.js'
import type { IIOPort } from './IOPort.js'
import { cognitiveTools } from '../tools/cognitive.js'
import { systemTools } from '../tools/system.js'
import type { InMemoryRecorder } from '../trajectory/InMemoryRecorder.js'
import type { ITraceObjectStore } from '../trace/TraceObjectStore.js'
import { canonicalize, contentAddressForCanonicalBytes } from '../trace/hash.js'
import type { Region } from '../context/Region.js'
import type { SkillLifecyclePayload } from '../trace/types.js'

export interface AgentRuntimeOptions {
  config:            AgentConfig
  goal:              string
  input:             string
  contextId?:        string
  agentRunId?:       string  // if provided, use this (allows caller to correlate with recorder meta)
  parentId?:         string
  stateStore:        IStateStore
  recorder:          ITrajectoryRecorder
  ioPort:            IIOPort
  /**
   * Optional event store for emitting region.added / region.removed /
   * context.boundary.applied events into the live trace stream (so HTML
   * report / web UI can see them). Pass only in recording paths
   * (Milkie.invoke / continueTurn). Replay must NOT pass this — otherwise
   * region events get re-written on top of the existing run JSONL.
   */
  eventStore?:       import('../trace/EventStore.js').IEventStore
  traceObjectStore?: ITraceObjectStore
  extraTools?:       ToolDefinition[]
  subAgentConfigs?:  Map<string, AgentConfig>  // agentId → config, for spawning
  childRecorderFactory?: (config: AgentConfig, contextId: string, traceId: string) => ITrajectoryRecorder
}

type SkillLoadRequest = {
  name:         string
  instructions: string
  scope:        'turn' | 'session'
  lifecycle:    SkillLifecyclePayload
}

export class AgentRuntime {
  private readonly config:           AgentConfig
  private readonly goal:             string
  readonly contextId:        string
  private readonly agentRunId:       string
  private readonly parentId?:        string
  private readonly stateStore:       IStateStore
  private readonly recorder:         ITrajectoryRecorder
  private readonly ioPort:           IIOPort
  private readonly eventStore?:      import('../trace/EventStore.js').IEventStore
  private readonly traceObjectStore?: ITraceObjectStore
  private readonly subAgentConfigs?: Map<string, AgentConfig>
  private readonly childRecorderFactory?: (config: AgentConfig, contextId: string, traceId: string) => ITrajectoryRecorder
  private readonly extraTools:      ToolDefinition[]

  private readonly fsm:         FSMEngine
  private readonly regions:     ContextRegions
  private readonly registry:    ToolRegistry
  private readonly memory:      WorkingMemory
  private readonly checkpoints: CheckpointManager
  private readonly factory:     AgentFactory

  private eventQueue:    Array<{ type: string; payload: unknown }> = []
  private pendingEvents: Array<{ type: string; payload: unknown }> = []
  private pendingSkillLoads: Array<SkillLoadRequest> = []
  private loadedSkills: Map<string, SkillLifecyclePayload> = new Map()
  private pendingTraceWrites: Promise<unknown>[] = []
  private traceWriteChain: Promise<unknown> = Promise.resolve()
  private initialRegionsEmitted = false
  private needsResumeCrystallization = false
  private childRecords: Map<string, ChildAgentRecord> = new Map()
  private rootSpan!:     Span
  private turnNumber:    number = 0
  private lastTextOutput: string = ''

  constructor(opts: AgentRuntimeOptions) {
    this.ioPort          = opts.ioPort
    this.config          = opts.config
    this.goal            = opts.goal
    this.contextId       = opts.contextId ?? this.ioPort.uuid()
    this.agentRunId      = opts.agentRunId ?? this.ioPort.uuid()
    this.parentId        = opts.parentId
    this.stateStore      = opts.stateStore
    this.recorder        = opts.recorder
    this.eventStore      = opts.eventStore
    this.traceObjectStore = opts.traceObjectStore
    this.subAgentConfigs = opts.subAgentConfigs
    this.childRecorderFactory = opts.childRecorderFactory
    this.extraTools      = opts.extraTools ?? []

    this.fsm     = new FSMEngine(opts.config.fsm)
    this.memory  = new WorkingMemory()
    this.regions = new ContextRegions(
      () => this.ioPort.now(),
      { onChange: (delta) => this.emitRegionDelta(delta, 'agent-set') },
    )
    this.regions.set('header', makeHeaderRegion(opts.config.systemPrompt))
    this.registry    = new ToolRegistry()
    this.checkpoints = new CheckpointManager(
      this.stateStore,
      this.config.agentId,
      this.agentRunId,
    )

    this.factory = new AgentFactory(async (spawnOpts: AgentSpawnOptions) => {
      const child = new AgentRuntime({
        ...spawnOpts,
        parentId: this.agentRunId,
        childRecorderFactory: this.childRecorderFactory,
      })
      return child.run(spawnOpts.input)
    })

    this.setupFSMCallbacks()
    this.registerTools(opts.extraTools ?? [])
  }

  private setupFSMCallbacks(): void {
    this.fsm.onTransitionCallback((from, to, event) => {
      const span = this.recorder.startSpan('fsm.transition', {
        fromState: from,
        toState:   to,
        event:     event.name,
      })
      this.recorder.endSpan(span, 'ok')

      // Also write to the event log so trace HTML / web UI / inspect can see
      // these. Recorder-only (span events) doesn't reach the live event stream.
      // Bypass IOPort for the same reason region.added does: these events are
      // informational and not consumed by replay's nondet cache, so we mustn't
      // burn an ioPort.uuid()/now() that replay's skip-write branch wouldn't
      // match. Date.now() / uuidv4() direct keep record/replay paths symmetric.
      if (this.eventStore) {
        this.enqueueTraceWrite(async () => {
          await this.eventStore!.append({
            id:        uuidv4(),
            runId:     this.agentRunId,
            type:      'fsm.transition',
            actor:     this.config.agentId,
            timestamp: Date.now(),
            payload: {
              from,
              to,
              trigger: {
                // emitEvent / framework emit sites stamp domain explicitly; the
                // only path that leaves it unset is direct construction in tests.
                domain:  event.domain ?? 'business',
                name:    event.name,
                ...(event.payload !== undefined ? { payload: event.payload } : {}),
              },
            },
          })
        })
      }
    })
  }

  private registerTools(extraTools: ToolDefinition[]): void {
    for (const tool of systemTools)    this.registry.register(tool)
    for (const tool of cognitiveTools) this.registry.register(tool)
    for (const tool of extraTools)     this.registry.register(tool)
    for (const [agentId] of Object.entries(this.config.subAgents ?? {})) {
      this.registry.register(this.makeSubAgentTool(agentId))
    }
  }

  private makeSubAgentTool(agentId: string): ToolDefinition {
    return {
      name:         agentId,
      description:  `Invoke the ${agentId} sub-agent.`,
      parallelSafe: true,
      inputSchema: {
        type: 'object',
        properties: {
          goal:  { type: 'string' },
          input: { type: 'string' },
        },
        required: ['goal', 'input'],
      },
      handler: async (rawInput: unknown, ctx: ToolContext) => {
        const { goal, input: subInput } = rawInput as { goal: string; input: string }
        const subConfig = this.subAgentConfigs?.get(agentId)
        if (!subConfig) {
          throw new Error(`Sub-agent config not found: ${agentId}. Pass subAgentConfigs to AgentRuntime.`)
        }

        const childTraceId = this.ioPort.uuid()
        const childContextId = this.ioPort.uuid()
        const childRecorder = this.childRecorderFactory?.(subConfig, childContextId, childTraceId) ?? this.recorder
        const taskId = this.ioPort.uuid()
        const spawnSpan = this.recorder.startSpan('agent.spawn', {
          childAgentId: agentId,
          taskId,
          turn:         this.turnNumber,
          childTraceId,
          childContextId,
        })
        await this.recordChild({
          taskId,
          agentId,
          contextId: childContextId,
          status: 'running',
        })
        this.emitAgentSpawned(childContextId, agentId, goal)

        try {
          const result = await ctx.agentFactory.spawn({
            config:      subConfig,
            goal,
            input:       subInput,
            contextId:   childContextId,
            agentRunId:   this.agentRunId,
            stateStore:  this.stateStore,
            recorder:    childRecorder,
            ioPort:      this.ioPort,
            extraTools:  this.extraTools,
          })
          const childCheckpoint = result.status === 'interrupted'
            ? await this.stateStore.get(`context:${childContextId}:checkpoint:latest`) as AgentCheckpoint | undefined
            : undefined
          await this.recordChild({
            taskId,
            agentId,
            contextId:     childContextId,
            checkpointId:  childCheckpoint?.checkpointId,
            status:        result.status === 'interrupted' ? 'interrupted' : result.status === 'completed' ? 'success' : 'error',
          })
          this.recorder.recordEvent(spawnSpan, 'agent.spawn.complete', { resultStatus: result.status })
          this.emitAgentReturned(childContextId, result.status)
          spawnSpan.attributes['resultStatus']  = result.status
          spawnSpan.attributes['childTraceId']  = childTraceId
          spawnSpan.attributes['childContextId'] = childContextId
          if (childCheckpoint?.checkpointId) {
            spawnSpan.attributes['checkpointId'] = childCheckpoint.checkpointId
          }
          this.recorder.endSpan(spawnSpan, 'ok')
          return result.output
        } catch (err) {
          await this.recordChild({
            taskId,
            agentId,
            contextId: childContextId,
            status: 'error',
          })
          spawnSpan.attributes['resultStatus'] = 'error'
          this.recorder.endSpan(spawnSpan, 'error')
          throw err
        }
      },
    }
  }

  interrupt(): void {
    this.eventQueue.push({ type: 'interrupt', payload: null })
  }

  private async recordChild(record: ChildAgentRecord): Promise<void> {
    this.childRecords.set(record.taskId, record)
    await this.stateStore.set(
      `context:${this.contextId}:children`,
      Array.from(this.childRecords.values()),
    )
  }

  private buildToolContext(emitFn: (event: string, payload?: unknown) => void): ToolContext {
    return {
      workingMemory: this.memory,
      agentFactory:  this.factory,
      stateStore:    this.stateStore,
      emit:          emitFn,
      requestSkill:  (name: string, scope?: 'turn' | 'session') => this.requestSkill(name, scope),
    }
  }

  private requestSkill(name: string, scope: 'turn' | 'session' = 'turn'): { requested: string; status: string; version?: string; scope?: 'turn' | 'session' } {
    const normalized = name.trim().replace(/\s+skill$/i, '')
    const version = this.config.skills?.[normalized]
    const instructions = this.config.skillInstructions?.[normalized]
    if (!version || !instructions) {
      return { requested: name, status: 'unavailable' }
    }
    this.pendingSkillLoads.push({
      name: normalized,
      instructions,
      scope,
      lifecycle: this.buildSkillLifecyclePayload(normalized, version, instructions),
    })
    return { requested: normalized, status: 'pending_next_epoch', version, scope }
  }

  private applyPendingSkills(): void {
    for (const { name, instructions, scope, lifecycle } of this.pendingSkillLoads) {
      const id = `skill:${name}`
      const existing = this.regions.get(id)
      const previous = this.loadedSkills.get(id) ?? this.lifecycleFromExistingSkill(id, existing)
      // Same version already loaded: preserve createdAt and keep repeated
      // skill_request idempotent.
      if (existing && previous?.version === lifecycle.version) continue
      if (existing && previous) {
        this.emitSkillLifecycle('skill.unloaded', previous)
      }
      this.loadedSkills.set(id, lifecycle)
      this.regions.set(id, makeSkillRegion(name, instructions, scope))
      this.emitSkillLifecycle('skill.loaded', lifecycle)
    }
    this.pendingSkillLoads = []
  }

  private buildSkillLifecyclePayload(name: string, version: string, instructions: string): SkillLifecyclePayload {
    return {
      skillId: `skill:${name}`,
      version,
      source:  'agent-config.skillInstructions',
      sha:     createHash('sha256').update(instructions).digest('hex'),
    }
  }

  private lifecycleFromExistingSkill(id: string, existing: ReturnType<ContextRegions['get']>): SkillLifecyclePayload | undefined {
    if (!existing || !id.startsWith('skill:')) return undefined
    const name = id.slice('skill:'.length)
    const version = this.config.skills?.[name]
    const instructions = (existing.content as { instructions?: unknown }).instructions
    if (!version || typeof instructions !== 'string') return undefined
    return this.buildSkillLifecyclePayload(name, version, instructions)
  }

  private setCurrentTurn(input: string): void {
    this.regions.set('current-turn', makeCurrentTurnRegion(input))
  }

  private getCurrentTurn(): string | null {
    const r = this.regions.get('current-turn')
    if (!r) return null
    return r.content as string
  }

  private appendScratchpadAssistant(content: MessageContent[]): void {
    const id = `scratch:${this.ioPort.uuid()}`
    const hasToolUse = content.some(c => c.type === 'tool_use')
    this.regions.set(id, makeScratchpadAssistantRegion(content, hasToolUse))
  }

  private appendScratchpadToolResults(content: MessageContent[]): void {
    const id = `scratch:${this.ioPort.uuid()}`
    this.regions.set(id, makeScratchpadToolResultRegion(content))
  }

  private toolStrategyFor(toolName: string): ToolResultStrategy | undefined {
    return this.registry.get(toolName)?.resultStrategy
  }

  /**
   * Apply the tool's declared resultStrategy (if any) to a single ToolResult,
   * returning the string content that should go into the tool_result message.
   * Default strategy is 'verbatim' (backwards compatible — pre-PR-E behavior).
   *
   * The raw output is recorded as-is on the tool.responded event (in
   * executeSingleTool / RecordingIOPort); this method only shapes what goes
   * into the LLM's scratchpad message.
   */
  private shapeToolResultForLlm(r: ToolResult, llmSpan: Span): string {
    const strategy = this.toolStrategyFor(r.toolName)
    const shape = r.isError ? (strategy?.onError ?? 'verbatim') : (strategy?.shape ?? 'verbatim')
    const raw = r.isError ? r.error : r.output
    const rawString = serializeOutput(raw)
    const shaped = applyShape(raw, shape)

    // Only emit tool.shaped when a non-verbatim shape actually changed bytes.
    if (shape !== 'verbatim' && shaped.length !== rawString.length) {
      this.recorder.recordEvent(llmSpan, 'tool.shaped', {
        toolName:    r.toolName,
        toolCallId:  r.toolCallId,
        shapeKind:   typeof shape === 'object' ? shape.kind : shape,
        rawBytes:    rawString.length,
        storedBytes: shaped.length,
        onErrorPath: r.isError,
      })
    }
    return shaped
  }

  private emitRegionDelta(delta: import('../context/ContextRegions.js').RegionChangeDelta, reason: string): void {
    if (!this.rootSpan) return   // pre-run mutations not yet attributable to a root span
    const region = delta.kind === 'added' ? this.regions.get(delta.id) : undefined
    const addedPayload = region ? this.buildRegionAddedPayload(region, reason) : undefined
    const recorderPayload = delta.kind === 'added'
      ? { ...(addedPayload ?? {
        id:        delta.id,
        section:   delta.section,
        target:    delta.target,
        stability: delta.stability,
        reason,
      }) }
      : { id: delta.id, reason }
    this.recorder.recordEvent(this.rootSpan, delta.kind === 'added' ? 'region.added' : 'region.removed', recorderPayload)
    // Also write to the event log so trace HTML / web UI / inspect can see
    // these. Recorder-only (span events) doesn't reach the live event stream.
    if (this.eventStore) {
      // Bypass IOPort for event metadata: these region/boundary events are
      // informational (not consumed by replay's nondet cache), so we mustn't
      // burn an ioPort.uuid()/now() in record mode that replay's skip-write
      // branch wouldn't match. Date.now() / uuid() direct keep record and
      // replay paths symmetric on IOPort consumption.
      this.enqueueTraceWrite(async () => {
        if (addedPayload) await this.persistRegionContent(region!)
        await this.eventStore!.append({
          id:        uuidv4(),
          runId:     this.agentRunId,
          type:      delta.kind === 'added' ? 'region.added' : 'region.removed',
          actor:     this.config.agentId,
          timestamp: Date.now(),
          payload:   delta.kind === 'added'
            ? (addedPayload ?? { id: delta.id, target: delta.target ?? 'system', section: delta.section ?? 'unknown', stability: delta.stability ?? 'volatile', reason })
            : { id: delta.id, reason },
        })
      })
    }

    if (delta.kind === 'removed' && delta.id.startsWith('skill:')) {
      const lifecycle = this.loadedSkills.get(delta.id)
      if (lifecycle) {
        this.emitSkillLifecycle('skill.unloaded', lifecycle)
        this.loadedSkills.delete(delta.id)
      }
    }
  }

  private emitSkillLifecycle(type: 'skill.loaded' | 'skill.unloaded', payload: SkillLifecyclePayload): void {
    if (!this.rootSpan) return
    this.recorder.recordEvent(this.rootSpan, type, { ...payload })
    if (this.eventStore) {
      void this.eventStore.append({
        id:        uuidv4(),
        runId:     this.agentRunId,
        type,
        actor:     this.config.agentId,
        timestamp: Date.now(),
        payload,
      })
    }
  }

  private emitAgentSpawned(childRunId: string, agentId: string, goal: string): void {
    if (!this.eventStore) return
    // Bypass IOPort (Date.now/uuidv4 direct): informational event, not consumed
    // by replay's nondet cache. The recorder already carries the agent.spawn
    // span for the trajectory view, so this writes to the event log only.
    this.enqueueTraceWrite(async () => {
      await this.eventStore!.append({
        id:        uuidv4(),
        runId:     this.agentRunId,
        type:      'agent.spawned',
        actor:     this.config.agentId,
        timestamp: Date.now(),
        payload:   { parentRunId: this.agentRunId, childRunId, agentId, goal },
      })
    })
  }

  private emitAgentReturned(childRunId: string, status: 'completed' | 'interrupted' | 'error'): void {
    if (!this.eventStore) return
    this.enqueueTraceWrite(async () => {
      await this.eventStore!.append({
        id:        uuidv4(),
        runId:     this.agentRunId,
        type:      'agent.returned',
        actor:     this.config.agentId,
        timestamp: Date.now(),
        payload:   { childRunId, status },
      })
    })
  }

  private buildRegionAddedPayload(region: Region, reason: string): import('../trace/types.js').RegionAddedPayload {
    let contentHash: string | undefined
    let renderedHash: string | undefined
    try {
      if (this.traceObjectStore) {
        contentHash = this.hashCanonical(region.content)
        const rendered = this.renderRegion(region)
        if (rendered !== undefined) renderedHash = this.hashCanonical(rendered)
      }
    } catch {
      // Region lifecycle events are observability records. Unsupported content
      // shapes should degrade the hash fields, not make ContextRegions.set()
      // fail and change agent behavior.
    }
    return {
      id:        region.id,
      target:    region.target,
      section:   region.section,
      stability: region.stability,
      reason,
      ...(contentHash ? { contentHash } : {}),
      ...(renderedHash ? { renderedHash } : {}),
    }
  }

  private renderRegion(region: Region): unknown {
    return region.format(region.content)
  }

  private hashCanonical(value: unknown): string {
    return contentAddressForCanonicalBytes(canonicalize(value))
  }

  private async persistRegionContent(region: Region): Promise<void> {
    if (!this.traceObjectStore) return
    try {
      await this.traceObjectStore.putCanonical(canonicalize(region.content))
      const rendered = this.renderRegion(region)
      if (rendered !== undefined) {
        await this.traceObjectStore.putCanonical(canonicalize(rendered))
      }
    } catch {
      // Keep trace object storage best-effort. The event remains useful with
      // metadata only if canonicalization/storage rejects a region shape.
    }
  }

  private enqueueTraceWrite(write: () => Promise<void>): void {
    const next = this.traceWriteChain.then(write)
    this.traceWriteChain = next.catch(() => undefined)
    this.pendingTraceWrites.push(next)
  }

  private async flushTraceWrites(): Promise<void> {
    const writes = this.pendingTraceWrites
    this.pendingTraceWrites = []
    const settled = await Promise.allSettled(writes)
    const rejected = settled.find((r): r is PromiseRejectedResult => r.status === 'rejected')
    if (rejected) throw rejected.reason
  }

  private async tryFlushTraceWrites(): Promise<void> {
    try {
      await this.flushTraceWrites()
    } catch {
      // Trace persistence is best-effort and must not change the business
      // outcome of a run. Durable deployments should surface store failures at
      // the store/hosting layer.
    }
  }

  private emitBoundaryApplied(summary: import('../context/lifecycleEngine.js').CrystallizationSummary): void {
    if (!this.rootSpan) return
    this.recorder.recordEvent(this.rootSpan, 'context.boundary.applied', {
      boundary: 'turn-end',
      epoch:    this.regions.getEpoch(),
      crystallization: {
        kept:         summary.kept.length,
        dropped:      summary.dropped.length,
        promoted:     summary.promoted.length,
        archivedPair: summary.archivedPair,
      },
    })
    if (this.eventStore) {
      // Bypass IOPort for event metadata: these region/boundary events are
      // informational (not consumed by replay's nondet cache), so we mustn't
      // burn an ioPort.uuid()/now() in record mode that replay's skip-write
      // branch wouldn't match. Date.now() / uuid() direct keep record and
      // replay paths symmetric on IOPort consumption.
      this.enqueueTraceWrite(async () => {
        await this.eventStore!.append({
          id:        uuidv4(),
          runId:     this.agentRunId,
          type:      'context.boundary.applied',
          actor:     this.config.agentId,
          timestamp: Date.now(),
          payload: {
            boundary: 'turn-end',
            epoch:    this.regions.getEpoch(),
            crystallization: {
              kept:         summary.kept.length,
              dropped:      summary.dropped.length,
              promoted:     summary.promoted.length,
              archivedPair: summary.archivedPair,
            },
          },
        })
      })
    }
  }

  private emitInitialRegionAdds(): void {
    if (this.initialRegionsEmitted) return
    this.initialRegionsEmitted = true
    for (const region of this.regions._allRegions()) {
      this.emitRegionDelta({
        kind:      'added',
        id:        region.id,
        section:   region.section,
        target:    region.target,
        stability: region.stability,
      }, 'runtime-initialized')
    }
  }

  // Idempotent — safe to call multiple times per turn. Second call is a no-op
  // because runInterTurnEngine deletes the current-turn region (turn-local) on
  // the first run, so the fallback lookup returns undefined and archiving is
  // skipped. Called both from executeFSM before the wait-for-user checkpoint
  // (so the checkpoint sees crystallized state) and from run() after FSM
  // completes (so terminal-state runs also crystallize).
  private crystallizeTurn(userInput?: string): void {
    const input = userInput ?? (this.regions.get('current-turn')?.content as string | undefined)
    if (input === undefined) return
    runInterTurnEngine(this.regions, {
      boundary:   'turn-end',
      userInput:  input,
      now:        this.ioPort.now(),
      onBoundary: (summary) => this.emitBoundaryApplied(summary),
    })
  }

  private refreshTransientRegions(state: FSMState, schemas: ToolSchema[]): void {
    // State instructions — re-set every step so changes in state propagate.
    // The state-scoped intraTurn filter (assemble's isActive) means only the
    // current state's region is visible during assembly, but we still need
    // to put it there.
    if (state.instructions) {
      this.regions.set(
        `state-instr:${state.name}`,
        makeStateInstructionsRegion(state.name, state.instructions),
      )
    }

    // Working memory snapshot — written every step (deterministic key order in factory).
    const wmJson = this.memory.toJSON() as { data: Record<string, unknown>; log: unknown[] }
    const wmRegion = makeWmRegion(wmJson.data, wmJson.log)
    if (wmRegion) this.regions.set('wm', wmRegion)
    else this.regions.delete('wm')

    // Tool schemas — re-set per step in case state.tools restricted the set.
    // Clear existing tool regions first (only current schema set is valid).
    for (const r of [...this.regions._allRegions()].filter(r => r.target === 'tool')) {
      this.regions.delete(r.id)
    }
    for (const s of schemas) {
      this.regions.set(`tool:${s.name}`, makeToolSchemaRegion(s))
    }
  }

  private async checkEvents(): Promise<void> {
    // Poll stateStore for external interrupt signal (set by Milkie.interrupt)
    const extInterrupt = await this.stateStore.get(`context:${this.contextId}:interrupt`)
    if (extInterrupt) {
      await this.stateStore.delete(`context:${this.contextId}:interrupt`)
      this.eventQueue.unshift({ type: 'interrupt', payload: null })
    }

    const event = this.eventQueue.shift()
    if (!event) return

    if (event.type === 'interrupt') {
      const resumeState = this.fsm.currentState.name
      this.fsm.emitEvent('interrupt')
      this.fsm.processPendingEvent()
      const checkpoint = await this.saveCheckpoint(resumeState)
      await this.tryFlushTraceWrites()
      // Save to context:latest so Milkie.resume can find it
      await this.checkpoints.saveForContext(this.contextId, this.turnNumber, checkpoint)
      await this.stateStore.set(`context:${this.contextId}:checkpoint:latest`, checkpoint)
      const err = new Error('Agent interrupted')
      err.name = 'InterruptSignal'
      throw err
    }
    this.pendingEvents.push(event)
  }

  private async saveCheckpoint(resumeState?: string, currentTurn?: string): Promise<AgentCheckpoint> {
    return this.checkpoints.save({
      sequence:    this.turnNumber,
      goal:        this.goal,
      currentTurn: currentTurn ?? this.getCurrentTurn() ?? undefined,
      fsm:         this.fsm.snapshot(resumeState),
      context: {
        workingMemory: this.memory.toJSON(),
        regions:       this.regions.snapshot(),
      },
      pendingEvents: this.pendingEvents.map(e => ({ type: e.type, payload: e.payload })),
      children:      Array.from(this.childRecords.values()),
      meta: {
        agentId:       this.config.agentId,
        agentRunId:    this.agentRunId,
        parentAgentId: this.parentId,
        timestamp:     this.ioPort.now(),
        traceId:       (this.recorder as Partial<InMemoryRecorder>).traceId ?? '',
        contextId:      this.contextId,
      },
    })
  }

  // Load prior state for multi-turn continuation
  async loadCheckpoint(checkpoint: AgentCheckpoint): Promise<void> {
    this.turnNumber = checkpoint.sequence
    // Re-attach format functions dropped by JSON serialization.
    this.regions.restore(rehydrateSnapshot(checkpoint.context.regions))
    this.loadedSkills = new Map()
    for (const r of this.regions._allRegions()) {
      if (r.id.startsWith('skill:')) {
        const lifecycle = this.lifecycleFromExistingSkill(r.id, r)
        if (lifecycle) this.loadedSkills.set(r.id, lifecycle)
      }
    }
    const restoredMemory = WorkingMemory.fromJSON(checkpoint.context.workingMemory)
    Object.assign(this.memory, restoredMemory)
    this.fsm.restore(checkpoint.fsm)
    if (checkpoint.fsm.currentState === 'paused' && checkpoint.fsm.resumeState) {
      this.fsm.transitionTo(checkpoint.fsm.resumeState, { name: 'RESUME', domain: 'lifecycle' })
    }
    // Interrupt checkpoints did NOT crystallize (interrupt path saves mid-turn),
    // so scratchpad + current-turn survive in the snapshot. Defer
    // crystallization until run() has a rootSpan, otherwise region.added /
    // region.removed events would be silently dropped and event-sourced context
    // reconstruction would miss the resume boundary.
    this.needsResumeCrystallization = true
  }

  async run(input: string): Promise<AgentResult> {
    this.rootSpan = this.recorder.startSpan('agent.run', {
      agentId:   this.config.agentId,
      goal:      this.goal,
      contextId: this.contextId,
    })
    this.emitInitialRegionAdds()
    if (this.needsResumeCrystallization) {
      this.crystallizeTurn()
      this.needsResumeCrystallization = false
    }

    const turnInput = `Goal: ${this.goal}\n\n${input}`
    this.setCurrentTurn(turnInput)
    this.turnNumber++

    try {
      await this.executeFSM()
      // Turn completed successfully — crystallize.
      this.crystallizeTurn(turnInput)
      this.recorder.endSpan(this.rootSpan, 'ok')
      return {
        agentRunId: this.agentRunId,
        contextId:  this.contextId,
        output:     this.lastTextOutput,
        status:     'completed',
      }
    } catch (err: unknown) {
      const isInterrupt = err instanceof Error && err.name === 'InterruptSignal'
      if (isInterrupt) {
        this.recorder.endSpan(this.rootSpan, 'ok')
        return {
          agentRunId: this.agentRunId,
          contextId:  this.contextId,
          output:     this.lastTextOutput,
          status:     'interrupted',
        }
      }
      this.recorder.endSpan(this.rootSpan, 'error')
      return {
        agentRunId: this.agentRunId,
        contextId:  this.contextId,
        output:     err instanceof Error ? err.message : String(err),
        status:     'error',
      }
    } finally {
      await this.tryFlushTraceWrites()
      await this.recorder.flush()
    }
  }

  // Continue an existing session with new user input
  async continueTurn(input: string): Promise<AgentResult> {
    this.setCurrentTurn(input)
    this.turnNumber++
    return this.run(input)
  }

  // Public so Milkie can drive multi-turn execution directly
  async executeFSM(): Promise<void> {
    while (true) {
      const state = this.fsm.currentState

      if (state.name === 'paused' || state.name === 'failed') break
      if (state.type !== 'llm' && this.fsm.isTerminal()) break

      if (state.type === 'llm') {
        const shouldContinue = await this.runLLMState(state)
        if (!shouldContinue) {
          if (!state.terminal) {
            // Waiting for user — crystallize FIRST so the checkpoint includes
            // the (user, finalAssistant) history pair (otherwise the next
            // invoke sees an empty history).
            this.crystallizeTurn()
            const checkpoint = await this.saveCheckpoint()
            await this.tryFlushTraceWrites()
            await this.checkpoints.saveForContext(this.contextId, this.turnNumber, checkpoint)
            await this.stateStore.set(`context:${this.contextId}:checkpoint:latest`, checkpoint)
          }
          break
        }
      } else if (state.type === 'action') {
        await this.runActionState(state)
      } else {
        throw new Error(`Unknown state type: ${(state as FSMState).type}`)
      }
    }
  }

  // Returns true if FSM should continue (transitioned to new state), false if waiting for user
  private async runLLMState(state: FSMState): Promise<boolean> {
    const maxIter = state.max_iterations ?? 20
    let iterations = 0

    while (true) {
      await this.checkEvents()

      if (iterations >= maxIter) {
        throw new MaxIterationsError(state.name, maxIter)
      }
      iterations++

      const tools   = this.registry.getForState(state.tools)
      const schemas = this.registry.toSchemas(tools)

      this.refreshTransientRegions(state, schemas)

      const scope: AssembleScope = {
        currentState:  state.name,
        currentTurnId: `turn-${this.turnNumber}`,
        currentEpoch:  this.regions.getEpoch(),
      }
      const assembled = assemble(this.regions, scope)
      const request: ModelRequest = {
        model:    this.config.model.model,
        system:   assembled.system,
        messages: assembled.messages,
        ...(assembled.tools ? { tools: assembled.tools } : {}),
        ...(assembled.cacheBreakpoint ? { cacheBreakpoint: assembled.cacheBreakpoint } : {}),
      }
      await this.tryFlushTraceWrites()

      const llmSpan = this.recorder.startSpan('llm.call', {
        provider:     this.config.model.provider,
        model:        this.config.model.model,
        turn:         this.turnNumber,
        state:        state.name,
        loadedSkills: [...this.regions._allRegions()]
          .filter(r => r.section === 'persistent-skills' || r.section === 'session-skills')
          .map(r => (r.content as { name: string }).name),
        contextEpoch: this.regions.getEpoch(),
      })

      const response = await this.ioPort.invokeLLM(request)

      this.recorder.recordEvent(llmSpan, 'usage', {
        inputTokens:         response.usage?.inputTokens,
        outputTokens:        response.usage?.outputTokens,
        cacheReadTokens:     response.usage?.cacheReadTokens,
        cacheCreationTokens: response.usage?.cacheCreationTokens,
        cacheHitRate:        response.usage?.cacheReadTokens !== undefined && (response.usage?.inputTokens ?? 0) > 0
                               ? response.usage.cacheReadTokens / response.usage.inputTokens
                               : undefined,
      })
      this.recorder.endSpan(llmSpan, 'ok')

      // Append assistant turn to context
      this.appendScratchpadAssistant(response.content)

      // Capture text output
      for (const c of response.content) {
        if (c.type === 'text' && c.text) {
          this.lastTextOutput = c.text
        }
      }

      await this.checkEvents()

      // --- Decide next step ---

      if (response.toolCalls.length > 0) {
        const results = await this.executeTools(response.toolCalls, state.tools)
        await this.checkEvents()

        // Append tool results to context. shapeToolResultForLlm may emit
        // tool.shaped events on llmSpan; InMemoryRecorder.recordEvent only
        // pushes to span.events (doesn't check endTime), so emitting after
        // endSpan is fine — keeps llmSpan close to LLM-call boundary so
        // sub-agent spawning (inside executeTools) sees a closed parent span.
        const toolResultContent: MessageContent[] = results.map(r => ({
          type:        'tool_result' as const,
          tool_use_id: r.toolCallId,
          content:     this.shapeToolResultForLlm(r, llmSpan),
          is_error:    r.isError,
        }))
        this.appendScratchpadToolResults(toolResultContent)
        this.applyPendingSkills()

        // Did any tool emit a FSM event?
        const next = this.fsm.processPendingEvent()
        if (next) {
          if (next.name === 'paused' || next.name === 'failed') return false
          // For terminal LLM states, return true so outer loop runs them once
          if (next.terminal && next.type === 'llm') return true
          if (next.terminal) return false  // terminal action / reserved state
          return true  // transitioned to new state, outer loop continues
        }
        // No event → continue LLM loop
        continue
      }

      // Pure text output → trigger DONE
      const next = this.fsm.processDone()
      if (next) {
        if (next.terminal && next.type === 'llm') return true  // run terminal LLM once
        if (next.terminal) return false
        return true
      }

      // No on.DONE → waiting for user
      return false
    }
  }

  private async runActionState(state: FSMState): Promise<void> {
    await this.checkEvents()

    if (!state.handler) {
      // No handler, auto-DONE
      this.fsm.processDone()
      return
    }

    const tool = this.registry.get(state.handler)
    if (!tool) {
      throw new Error(`Action handler "${state.handler}" not found in tool registry`)
    }

    const ctx = this.buildToolContext((event, payload) => {
      this.fsm.emitEvent(event, payload)
    })

    const span = this.recorder.startSpan('tool.call', {
      toolName: state.handler,
      turn:     this.turnNumber,
      state:    state.name,
    })

    const actionInput = {
      goal:  this.goal,
      input: `Context: ${JSON.stringify(this.memory.toJSON())}\nCurrent turn: ${this.getCurrentTurn() ?? ''}`,
    }

    try {
      const output = await tool.handler(actionInput, ctx)
      span.attributes['output'] = output
      this.recorder.endSpan(span, 'ok')
      if (output && typeof output === 'string') this.lastTextOutput = output
    } catch (err) {
      this.recorder.endSpan(span, 'error')
      this.fsm.emitEvent('error', { message: err instanceof Error ? err.message : String(err) })
    }

    await this.checkEvents()
    const next = this.fsm.processPendingEvent()
    if (!next) this.fsm.processDone()
  }

  private async executeTools(
    calls: Array<{ id: string; name: string; input: unknown }>,
    allowedTools?: string[],
  ): Promise<ToolResult[]> {
    const tools    = this.registry.getForState(allowedTools)
    const toolMap  = new Map(tools.map(t => [t.name, t]))
    const batchId  = this.ioPort.uuid()

    const parallel: typeof calls = []
    const serial:   typeof calls = []

    for (const call of calls) {
      if (toolMap.get(call.name)?.parallelSafe) {
        parallel.push(call)
      } else {
        serial.push(call)
      }
    }

    const results: ToolResult[] = []

    if (parallel.length > 0) {
      const settled = await Promise.allSettled(
        parallel.map(c => this.executeSingleTool(c, batchId))
      )
      for (const r of settled) {
        results.push(
          r.status === 'fulfilled'
            ? r.value
            : { toolCallId: '', toolName: '', output: null, error: String(r.reason), isError: true, duration: 0 }
        )
      }
    }

    for (const call of serial) {
      results.push(await this.executeSingleTool(call, null))
    }

    return results
  }

  private async executeSingleTool(
    call: { id: string; name: string; input: unknown },
    batchId: string | null,
  ): Promise<ToolResult> {
    const ctx = this.buildToolContext((event, payload) => {
      this.fsm.emitEvent(event, payload)
    })

    const maxRetries = 3

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      const start = this.ioPort.now()
      const span  = this.recorder.startSpan('tool.call', {
        toolName:        call.name,
        toolCallId:      call.id,
        input:           call.input,
        turn:            this.turnNumber,
        attempt,
        parallelBatchId: batchId ?? undefined,
      })

      try {
        const output   = await this.ioPort.invokeTool(
          call.name,
          call.input,
          () => this.registry.execute(call.name, call.input, ctx),
        )
        span.attributes['output'] = output
        const duration = this.ioPort.now() - start
        this.recorder.recordEvent(span, 'tool.result', { output })
        this.recorder.endSpan(span, 'ok')
        return { toolCallId: call.id, toolName: call.name, output, isError: false, duration }
      } catch (err) {
        const retryable = (err as { retryable?: boolean }).retryable === true
        const isLastAttempt = attempt === maxRetries - 1
        const duration      = this.ioPort.now() - start

        if (!retryable || isLastAttempt) {
          const error = err instanceof Error ? err.message : String(err)
          this.recorder.endSpan(span, 'error')
          return { toolCallId: call.id, toolName: call.name, output: null, error, isError: true, duration }
        }

        this.recorder.endSpan(span, 'error')

        const retryFromState = this.fsm.currentState.name
        this.fsm.transitionTo('error_handling', {
          name:   'error',
          domain: 'runtime-control',
          payload: {
            attempt:  attempt + 1,
            toolName: call.name,
            message:  err instanceof Error ? err.message : String(err),
          },
        })
        await new Promise<void>(r => setTimeout(r, 500))
        this.fsm.transitionTo(retryFromState, {
          name:   'RETRY',
          domain: 'runtime-control',
          payload: {
            attempt:  attempt + 1,
            toolName: call.name,
          },
        })
      }
    }

    // Should never reach here — TypeScript requires explicit return
    return { toolCallId: call.id, toolName: call.name, output: null, error: 'Unexpected retry exhaustion', isError: true, duration: 0 }
  }
}
