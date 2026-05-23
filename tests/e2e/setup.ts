import { Milkie } from '../../src/runtime/Milkie.js'
import { TrajectoryStore } from '../../src/trajectory/TrajectoryStore.js'
import { MemoryStore } from '../../src/store/MemoryStore.js'

export function createMilkie(opts: {
  tools?: import('../../src/types/tool.js').ToolDefinition[]
} = {}): { milkie: Milkie; trajectoryStore: TrajectoryStore } {
  const trajectoryStore = new TrajectoryStore({ jsonlDir: './test-output/trajectories' })
  const stateStore = new MemoryStore()

  const milkie = new Milkie({
    stateStore,
    trajectoryStore,
    tools: opts.tools ?? [],
  })

  return { milkie, trajectoryStore }
}
