import type { ToolDefinition } from '../types/tool.js'

// skill_list and skill_request are system-injected tools.
// In v1 they are stubs; the Skill Registry (Section 6) is wired in v2.
export const systemTools: ToolDefinition[] = [
  {
    name:        'skill_list',
    description: 'List all available skills with their names and brief descriptions.',
    inputSchema: { type: 'object', properties: {}, required: [] },
    handler: async (_input, _ctx) => {
      // v1 stub — no remote registry
      return { skills: [] }
    },
  },

  {
    name:        'skill_request',
    description: "Request a skill to be loaded in the next context epoch. Choose scope=turn for one-shot usage (auto-released at turn end) or scope=session for cross-turn persistence. Default: turn.",
    inputSchema: {
      type:       'object',
      properties: {
        name:  { type: 'string', description: 'Skill name to load' },
        scope: {
          type: 'string',
          enum: ['turn', 'session'],
          description: "Lifetime: 'turn' = available only for the current conversational turn (auto-released at turn end); 'session' = persists across turns. Default: 'turn'.",
        },
      },
      required:   ['name'],
    },
    handler: async (input: unknown, ctx) => {
      const { name, scope } = input as { name: string; scope?: 'turn' | 'session' }
      return ctx.requestSkill?.(name, scope) ?? { requested: name, status: 'unavailable' }
    },
  },
]
