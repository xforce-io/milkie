import type { ModelEvent, ModelResponse, ModelUsage } from '../types/model.js'
import type { MessageContent } from '../types/common.js'
import type { ToolCall } from '../types/tool.js'

interface ToolCallAccumulator {
  id:      string
  name:    string
  argsBuf: string
}

/**
 * 消费 AsyncIterable<ModelEvent>，将每个事件透传给 onEvent（若提供），
 * 同时将分片聚合成一个完整的 ModelResponse。
 *
 * - message_delta   → 累加 text
 * - tool_call_start → 注册工具调用（同 id 重复忽略），保留到达顺序
 * - tool_call_delta → 追加 argsBuf
 * - tool_call_done  → 若携带 input 则以它为权威替换 argsBuf
 * - usage           → 记录 usage
 * - error           → 设置 finishReason='error'（若未设）
 */
export async function aggregateStream(
  stream: AsyncIterable<ModelEvent>,
  onEvent?: (e: ModelEvent) => void,
): Promise<ModelResponse> {
  let textBuf = ''
  let usage: ModelUsage | undefined
  let finishReason: string | undefined

  // 保序：order 数组记到达顺序，map 存 accumulator
  const toolCallOrder: string[] = []
  const toolCallMap = new Map<string, ToolCallAccumulator>()

  for await (const event of stream) {
    // 先透传给 onEvent
    onEvent?.(event)

    switch (event.type) {
      case 'message_delta':
        textBuf += event.data.text
        break

      case 'tool_call_start': {
        const { toolCallId, name } = event.data
        if (!toolCallMap.has(toolCallId)) {
          toolCallOrder.push(toolCallId)
          toolCallMap.set(toolCallId, { id: toolCallId, name, argsBuf: '' })
        }
        break
      }

      case 'tool_call_delta': {
        const acc = toolCallMap.get(event.data.toolCallId)
        if (acc) {
          const delta = event.data.delta
          acc.argsBuf += typeof delta === 'string' ? delta : JSON.stringify(delta)
        }
        break
      }

      case 'tool_call_done': {
        const acc = toolCallMap.get(event.data.toolCallId)
        if (acc && event.data.input !== undefined) {
          // input 字段有权威值，直接序列化为 argsBuf
          acc.argsBuf = JSON.stringify(event.data.input)
        }
        break
      }

      case 'usage':
        usage = event.data
        break

      case 'error':
        if (!finishReason) {
          finishReason = 'error'
        }
        break
    }
  }

  // 构造 content
  const content: MessageContent[] = []
  const toolCalls: ToolCall[] = []

  // 文本：非空才 push
  if (textBuf) {
    content.push({ type: 'text', text: textBuf })
  }

  // 工具调用：按到达顺序
  for (const id of toolCallOrder) {
    const acc = toolCallMap.get(id)!
    let input: unknown
    try {
      input = acc.argsBuf ? JSON.parse(acc.argsBuf) : {}
    } catch {
      input = {}
    }
    content.push({ type: 'tool_use', id: acc.id, name: acc.name, input })
    toolCalls.push({ id: acc.id, name: acc.name, input })
  }

  return {
    content,
    toolCalls,
    ...(usage !== undefined ? { usage } : {}),
    ...(finishReason !== undefined ? { finishReason } : {}),
  }
}
