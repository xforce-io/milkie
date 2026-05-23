export const searchFixtures: Record<string, string> = {
  'Product A': '## Product A\n核心功能：实时协作编辑，定价 $20/mo，支持 API，市场份额 35%，用户好评率 4.5/5',
  'Product B': '## Product B\n核心功能：离线优先存储，定价 $15/mo，支持插件生态，市场份额 28%，用户好评率 4.2/5',
  'Product C': '## Product C\n核心功能：AI 辅助写作，定价 $25/mo，支持团队空间，市场份额 22%，用户好评率 4.7/5',
  'Product A features pricing': '## Product A\n核心功能：实时协作编辑，定价 $20/mo，支持 API，市场份额 35%，用户好评率 4.5/5',
  'Product B features pricing': '## Product B\n核心功能：离线优先存储，定价 $15/mo，支持插件生态，市场份额 28%，用户好评率 4.2/5',
  'Product C features pricing': '## Product C\n核心功能：AI 辅助写作，定价 $25/mo，支持团队空间，市场份额 22%，用户好评率 4.7/5',
  'TypeScript 5.0': 'TypeScript 5.0 主要特性：装饰器正式稳定、const 类型参数、多配置文件扩展、枚举优化、速度提升5倍',
  'TypeScript 5.0 features': 'TypeScript 5.0 主要特性：装饰器正式稳定、const 类型参数、多配置文件扩展、枚举优化、速度提升5倍',
}

export function lookupSearch(query: string): string {
  for (const [key, value] of Object.entries(searchFixtures)) {
    if (query.toLowerCase().includes(key.toLowerCase())) return value
  }
  return `搜索结果：关于 "${query}" 的信息暂时不可用，请使用现有知识回答。`
}
