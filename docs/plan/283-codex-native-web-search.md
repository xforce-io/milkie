# #283 实施与证据

2026-11-14，GitHub/keel，来自用户“那是否让milkie调整下”。查重完整读取3页/138张Issue，无同需求。Issue #283，从origin/main@c01de337建立feat/283-codex-native-web-search；未复用已合入的#281分支。没有新增模型调用。

## how依据

src/execution/codex.ts::LOCKS固定关闭web_search；message对webSearch的item事件拒绝为policy_mismatch；ExecutionClient.start拒绝未知约束，WorkerRequest/Sidecar尚无授权字段；Observation只支持安全摘要。固定Codex0.166.0通过app-server generate-ts导出的WebSearchItem包含query/action/results，results允许为null。原生命令help确认--search；features list显示standalone_web_search仍为开发中默认false，本次不自动开启该实验特性。协议生成、只读Issue检索存于旧#281工作树target/verify-native-search-investigation，新工作树保留协议副本于test-output/verify-milkie/283-investigation。CLI目录固定，未读取或复制认证。

## 状态

L1 reuse既有SDK/Sidecar生命周期及本轮明确搜索授权方向；L2 write v0.1。需要同时校验授权、事件与结果读取，不以改一行配置作为完成。零模型协议/确定性验证与真实联网严格分列；S1/S2/S3尚未验收。新接口仍须消费者适配与完整独立审查，不开PR、不合入。


## 开发验证（冻结前）

原执行套件4套/158项通过；额外含64条容量到界的Codex聚焦33项通过，build及功能地图通过。真实Codex0.166.0接本机合成Responses提供者已接纳webSearch事件并通过公开SDK保留query/action、results=null，执行succeeded/OK且stopped，零真实模型。原生本机提供者请求未声明web工具（自定义提供者的限制），不能据此证明真实OpenAI账号的搜索可用。证据test-output/verify-milkie/283-investigation，含首次空输出夹具失败（首版仅给web_search_call，缺少同Response的assistant message）及修正后的完整协议记录；Sidecar初次测试误用了默认Pi夹具，改为显式Codex入口后通过，不改弱断言。新候选冻结后需重跑适用检查和实际消费者。

真实联网验证尚未运行；将按新任务必要验证计入既有46/12600秒及研究43/10800秒上限，不重新申请已批准额度。若使用1入口做原生能力验证，所有固定必需剩余项仍可覆盖（当前25+探针1+必需20=46）；一次条件性裁决将不再有预留，发生真实分歧且需超限时必须停止并取得新的具体扩额，不能删除裁决要求或自动第4轮代码审查。
