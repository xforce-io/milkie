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
    description: 'Request a skill to be loaded in the next context epoch.',
    inputSchema: {
      type:       'object',
      properties: { name: { type: 'string', description: 'Skill name to load' } },
      required:   ['name'],
    },
    handler: async (input: unknown, ctx) => {
      const { name } = input as { name: string }
      return ctx.requestSkill?.(name) ?? { requested: name, status: 'unavailable' }
    },
  },
]
