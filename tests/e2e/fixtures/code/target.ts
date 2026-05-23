// target.ts — 供代码审查 fixture 使用
// 含有已知问题：SQL 注入、N+1 查询、命名不一致

import * as db from './db'

// 安全问题：SQL 字符串拼接（SQL 注入风险）
export async function getUserByName(userName: string) {
  // SECURITY ISSUE: SQL injection vulnerability
  const query = "SELECT * FROM users WHERE name = '" + userName + "'"
  return db.raw(query)
}

// 性能问题：N+1 查询
export async function getOrdersWithItems(userId: string) {
  // PERFORMANCE ISSUE: N+1 query pattern
  const orders = await db.query('SELECT * FROM orders WHERE user_id = ?', [userId])
  const result = []
  for (const order of orders) {
    // Each iteration executes a new query — N+1 problem
    const items = await db.query('SELECT * FROM order_items WHERE order_id = ?', [order.id])
    result.push({ ...order, items })
  }
  return result
}

// 风格问题：命名不一致（驼峰、下划线混用）
export function ProcessPayment(amount: number, user_id: string, PaymentMethod: string) {
  // STYLE ISSUE: inconsistent naming conventions
  const payment_result = {
    UserId: user_id,
    Amount: amount,
    method: PaymentMethod,
    status_code: 'PENDING',
  }
  return payment_result
}
