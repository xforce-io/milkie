# Codex原生搜索（#283 S1–S3）

事实源：[Issue #283](https://github.com/xforce-io/milkie/issues/283)与[设计v0.1](../../../../docs/design/283-codex-native-web-search.md)。入口为SDK ExecutionClient.start/webSearches和Sidecar execution.start/web_search.read，无新增页面。

- S1：创建Codex上下文，分别以默认/disabled/cached/live执行；默认或显式disabled收到webSearch应拒绝。缓存/实时必须显式授权，恢复同Context保持首次模式；非法模式、非Codex授权及不兼容恢复在新进程前拒绝。授权搜索时Shell/MCP/文件等仍不可用。
- S2：SDK和Sidecar分别读取两个Run的记录，核对runId/contextId/nativeItemId及原生回执。results=null保持未知；取消后未完成条目仍started；畸形/错误会话/2MiB或64条容量到界应固定错误并真实确认停止，不把丢失回执标为完整。重开客户端仍可读同一归属；启用模式的记录文件缺失报storage_error。Trace中不得有查询或结果正文，原生搜索不能冒充宿主工具调用或上游请求计量。
- S3：冻结候选运行原执行套件及原生协议探针；额外指定MILKIE_NATIVE_SCENARIO=web-search运行既有codex-global-warning.native.cjs，固定0.166.0接本机合成提供者，确认实际webSearch事件可记录、nullable结果保持null、无查询正文进入Trace和原生停止。该夹具零模型，不证明真实联网。真实原生搜索须另登记并计入Inquiry已批准累计额度；结果缺失/服务拒绝保持blocked，不以模拟取代。关联21与22的生命周期及摘要保全。

命令：npm run build；npm run test:execution；python3 .agents/skills/verify-milkie/scripts/check_map.py。原生探针必须显式MILKIE_NATIVE_CODEX与独立MILKIE_EVIDENCE_DIR，禁用真实登录复制，假凭据只连本机服务。真实模型未运行不能将S3汇总pass；SDK成功不能替代Sidecar入口。
