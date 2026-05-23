import type { AgentConfig, FSMState } from '../types/agent.js'
import type { AgentResult } from '../types/common.js'
import { MaxIterationsError } from '../types/common.js'
import type { IStateStore, AgentCheckpoint, ChildAgentRecord } from '../types/store.js'
import type { ITrajectoryRecorder, Span } from '../types/trajectory.js'
import type { ToolDefinition, ToolContext, ToolResult } from '../types/tool.js'
import type { MessageContent } from '../types/common.js'
import { FSMEngine } from '../fsm/FSMEngine.js'
import { ContextLayer } from '../context/ContextLayer.js'
import { ToolRegistry } from '../tools/ToolRegistry.js'
import { WorkingMemory } from '../store/WorkingMemory.js'
import { CheckpointManager } from '../store/CheckpointManager.js'
import { AgentFactory, type AgentSpawnOptions } from './AgentFactory.js'
import type { IIOPort } from './IOPort.js'
import { cognitiveTools } from '../tools/cognitive.js'
import { systemTools } from '../tools/system.js'
import type { InMemoryRecorder } from '../trajectory/InMemoryRecorder.js'

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
  extraTools?:       ToolDefinition[]
  subAgentConfigs?:  Map<string, AgentConfig>  // agentId → config, for spawning
  childRecorderFactory?: (config: AgentConfig, contextId: string, traceId: string) => ITrajectoryRecorder
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
  private readonly subAgentConfigs?: Map<string, AgentConfig>
  private readonly childRecorderFactory?: (config: AgentConfig, contextId: string, traceId: string) => ITrajectoryRecorder
  private readonly extraTools:      ToolDefinition[]

  private readonly fsm:         FSMEngine
  private readonly context:     ContextLayer
  private readonly registry:    ToolRegistry
  private readonly memory:      WorkingMemory
  private readonly checkpoints: CheckpointManager
  private readonly factory:     AgentFactory

  private eventQueue:    Array<{ type: string; payload: unknown }> = []
  private pendingEvents: Array<{ type: string; payload: unknown }> = []
  private pendingSkills: Set<string> = new Set()
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
    this.subAgentConfigs = opts.subAgentConfigs
    this.childRecorderFactory = opts.childRecorderFactory
    this.extraTools      = opts.extraTools ?? []

    this.fsm     = new FSMEngine(opts.config.fsm)
    this.memory  = new WorkingMemory()
    this.context = new ContextLayer({
      systemPrompt: opts.config.systemPrompt,
      model:        opts.config.model.model,
    })
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
      requestSkill:  (name: string) => this.requestSkill(name),
    }
  }

  private requestSkill(name: string): { requested: string; status: string; version?: string } {
    const normalized = name.trim().replace(/\s+skill$/i, '')
    const version = this.config.skills?.[normalized]
    const instructions = this.config.skillInstructions?.[normalized]
    if (!version || !instructions) {
      return { requested: name, status: 'unavailable' }
    }
    this.pendingSkills.add(normalized)
    return { requested: normalized, status: 'pending_next_epoch', version }
  }

  private applyPendingSkills(): void {
    for (const name of this.pendingSkills) {
      const instructions = this.config.skillInstructions?.[name]
      if (instructions && !this.context.getLoadedInstructions().includes(name)) {
        this.context.loadInstructions(name, instructions)
      }
    }
    this.pendingSkills.clear()
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
    const ctxSnapshot = this.context.snapshot()
    return this.checkpoints.save({
      sequence:    this.turnNumber,
      goal:        this.goal,
      currentTurn: currentTurn ?? this.context.currentTurn ?? undefined,
      fsm:         this.fsm.snapshot(resumeState),
      context: {
        history:              ctxSnapshot.history,
        workingMemory:        this.memory.toJSON(),
        instructionsSnapshot: ctxSnapshot.instructionsSnapshot,
        instructions:         ctxSnapshot.instructions,
        contextEpoch:         ctxSnapshot.contextEpoch,
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
    this.context.restore({
      history:              checkpoint.context.history,
      instructionsSnapshot: checkpoint.context.instructionsSnapshot,
      instructions:         checkpoint.context.instructions,
      contextEpoch:         checkpoint.context.contextEpoch,
    })
    const restoredMemory = WorkingMemory.fromJSON(checkpoint.context.workingMemory)
    Object.assign(this.memory, restoredMemory)
    this.fsm.restore(checkpoint.fsm)
    if (checkpoint.fsm.currentState === 'paused' && checkpoint.fsm.resumeState) {
      this.fsm.transitionTo(checkpoint.fsm.resumeState, { name: 'RESUME' })
    }
  }

  async run(input: string): Promise<AgentResult> {
    this.rootSpan = this.recorder.startSpan('agent.run', {
      agentId:   this.config.agentId,
      goal:      this.goal,
      contextId: this.contextId,
    })

    this.context.setCurrentTurn(`Goal: ${this.goal}\n\n${input}`)
    this.turnNumber++

    try {
      await this.executeFSM()

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
      await this.recorder.flush()
    }
  }

  // Continue an existing session with new user input
  async continueTurn(input: string): Promise<AgentResult> {
    this.context.setCurrentTurn(input)
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
            // Waiting for user — save context checkpoint for multi-turn continuation
            const checkpoint = await this.saveCheckpoint()
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

      const tools    = this.registry.getForState(state.tools)
      const schemas  = this.registry.toSchemas(tools)
      const request  = this.context.buildRequest(schemas, this.memory, state.instructions)

      const llmSpan = this.recorder.startSpan('llm.call', {
        provider:     this.config.model.provider,
        model:        this.config.model.model,
        turn:         this.turnNumber,
        state:        state.name,
        loadedSkills: this.context.getLoadedInstructions(),
        contextEpoch: this.context.getContextEpoch(),
      })

      const response = await this.ioPort.invokeLLM(request)

      this.recorder.recordEvent(llmSpan, 'usage', {
        inputTokens:  response.usage?.inputTokens,
        outputTokens: response.usage?.outputTokens,
      })
      this.recorder.endSpan(llmSpan, 'ok')

      // Append assistant turn to context
      this.context.appendHistory({ role: 'assistant', content: response.content })

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

        // Append tool results to context
        const toolResultContent: MessageContent[] = results.map(r => ({
          type:        'tool_result' as const,
          tool_use_id: r.toolCallId,
          content:     r.error ?? JSON.stringify(r.output),
          is_error:    r.isError,
        }))
        this.context.appendHistory({ role: 'tool', content: toolResultContent })
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
      input: `Context: ${JSON.stringify(this.memory.toJSON())}\nCurrent turn: ${this.context.currentTurn ?? ''}`,
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
          name: 'error',
          payload: {
            attempt:  attempt + 1,
            toolName: call.name,
            message:  err instanceof Error ? err.message : String(err),
          },
        })
        await new Promise<void>(r => setTimeout(r, 500))
        this.fsm.transitionTo(retryFromState, {
          name: 'RETRY',
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
