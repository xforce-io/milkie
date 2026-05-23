import type { ToolDefinition, ToolContext } from '../types/tool.js'
import type { ToolSchema } from '../types/model.js'

export class ToolRegistry {
  private tools: Map<string, ToolDefinition> = new Map()

  register(tool: ToolDefinition): void {
    this.tools.set(tool.name, tool)
  }

  get(name: string): ToolDefinition | undefined {
    return this.tools.get(name)
  }

  has(name: string): boolean {
    return this.tools.has(name)
  }

  // Returns tools available for a given state (filtered by names list)
  getForState(names?: string[]): ToolDefinition[] {
    if (!names) return Array.from(this.tools.values())
    return names
      .map(n => this.tools.get(n))
      .filter((t): t is ToolDefinition => t !== undefined)
  }

  toSchemas(tools: ToolDefinition[]): ToolSchema[] {
    return tools.map(t => ({
      name:        t.name,
      description: t.description,
      inputSchema: t.inputSchema,
    }))
  }

  async execute(
    name: string,
    input: unknown,
    ctx: ToolContext,
  ): Promise<unknown> {
    const tool = this.tools.get(name)
    if (!tool) throw new Error(`Tool not found: ${name}`)
    return tool.handler(input, ctx)
  }

  list(): ToolDefinition[] {
    return Array.from(this.tools.values())
  }
}
