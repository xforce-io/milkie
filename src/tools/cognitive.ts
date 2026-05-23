import { v4 as uuid } from 'uuid'
import type { ToolDefinition } from '../types/tool.js'

export interface PlanStep {
  id:     number
  desc:   string
  status: 'pending' | 'done' | 'failed'
}

export interface Plan {
  id:    string
  steps: PlanStep[]
}

export const cognitiveTools: ToolDefinition[] = [
  {
    name:        'think',
    description: 'Think step-by-step before acting. Use freely — has no side effects.',
    inputSchema: {
      type:       'object',
      properties: { thoughts: { type: 'string', description: 'Your reasoning' } },
      required:   ['thoughts'],
    },
    parallelSafe: true,
    handler: async (input: unknown, ctx) => {
      const { thoughts } = input as { thoughts: string }
      ctx.workingMemory.append({ type: 'thought', content: thoughts })
      return { recorded: true }
    },
  },

  {
    name:        'create_plan',
    description: 'Create a checklist of steps. Call once at the start of a multi-step task.',
    inputSchema: {
      type:       'object',
      properties: {
        steps: {
          type:  'array',
          items: { type: 'string' },
          description: 'Ordered list of steps to complete',
        },
      },
      required: ['steps'],
    },
    handler: async (input: unknown, ctx) => {
      const { steps } = input as { steps: string[] }
      const plan: Plan = {
        id:    uuid(),
        steps: steps.map((s, i) => ({ id: i, desc: s, status: 'pending' })),
      }
      ctx.workingMemory.set('plan', plan)
      return plan
    },
  },

  {
    name:        'update_step',
    description: 'Mark a step as done or failed. If failed, revise the plan with create_plan.',
    inputSchema: {
      type:       'object',
      properties: {
        stepId: { type: 'number', description: 'Step index (0-based)' },
        status: { type: 'string', enum: ['done', 'failed'] },
      },
      required: ['stepId', 'status'],
    },
    handler: async (input: unknown, ctx) => {
      const { stepId, status } = input as { stepId: number; status: 'done' | 'failed' }
      const plan = ctx.workingMemory.get('plan') as Plan | undefined
      if (!plan) throw new Error('No active plan. Call create_plan first.')
      const step = plan.steps[stepId]
      if (!step) throw new Error(`Step ${stepId} not found`)
      step.status = status
      ctx.workingMemory.set('plan', plan)
      return plan.steps
    },
  },
]
