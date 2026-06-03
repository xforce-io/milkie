// 类型声明,配合 UMD 的 citations.js(见 #132)。让 .ts 测试 import 时有类型,
// 不触发 TS7016。运行时实现仍是 citations.js。
export declare function linkifyCitations(html: string, count: number): string
