# Milkie #283：显式 Codex 原生搜索

版本v0.1；产品依据：本会话用户“那是否让 milkie 调整下，避开这个问题？”，承接Inquiry #7及停用Brave指示。L1复用既有执行/查询/停止路径，新增显式搜索授权的方向由该原话确认；L2 write，本文件为唯一技术契约，不等于交付人审批准。

## 1 背景

[Issue #283](https://github.com/xforce-io/milkie/issues/283)。当前Codex适配器强制web_search=disabled并拒绝webSearch事件；网页搜索抓取限流并不代表用户Codex搜索不可用。固定0.166.0导出的WebSearchItem具有id/query/action/results，results可为null。仅移除配置锁不足以交付。

## 2 名词解释

沿用[名词表](../glossary.md)；本设计新增“原生搜索记录”，以该表为准。

## 3 目标与非目标

目标：宿主显式选择搜索模式、核对当前Run原生搜索回执，保持默认禁用及原有隔离。非目标：放开其它原生工具、代理修改全局配置、保证供应商可用、在Milkie验证文章、将模型给出的URL冒充搜索结果。Inquiry仍负责实际正文留存和引用核验。

## 4 能力

### 4.1 UI/UX

N/A：没有新增页面或CLI交互，沿用SDK/Sidecar创建上下文→执行→查询/取消→恢复。空结果保持可判定，错误以固定代码返回，不输出原始错误或凭据。

S1：开发者通过ExecutionConstraints.webSearch选择disabled/cached/live，省略为disabled；执行后可查询实际模式，恢复时必须使用首次执行的同一模式，错误运行时和非法模式启动前拒绝。S2：宿主通过公开webSearches(runId)或web_search.read读取归属当前Run的原生条目；未知Run返回null/undefined，结果缺失为null，不能补成搜索成功。停止中的条目保留started而非伪报completed。S3：固定原生版本实跑、确定性夹具、既有回归分别记录，不把原生协议夹具当真实联网结果。

## 5 思路与折衷

选择仅Codex显式按Run授权且同Context不可变，放弃由全局配置继承、自动开启或笼统放开原生工具。选择独立、私有、归属明确的搜索记录读取，而非将正文塞进Trace或伪装宿主工具回调。搜索结果仍为外部非可信数据；CLI没有返回时保持未知，不能解析作者文本补造回执。原生搜索可能经过多次请求，当前CLI不提供严格的请求前数量限额；本版本只约束时间和记录容量，原生条目数与宿主工具调用数分开，能力说明不得承诺统一工具预算。

## 6 架构

SDK/Sidecar校验显式约束→独立Worker→固定Codex app-server（只替换web_search模式锁）→身份核对后的webSearch事件→每Run私有搜索记录；公共查询从执行记录核对模式/归属后读取。失败路径：未授权事件policy_mismatch；错误会话session_mismatch；畸形事件protocol_error；容量超限native_search_limit_exceeded；均保留此前记录并沿原停止机制收尾。原始结果不进入观测正文，活动只沿已有agent.activity/tool分类。

## 7 模块

execution/types及ExecutionClient负责约束/恢复/公共读契约；codex负责模式注入与原生事件身份；webSearch负责有界归属记录；store负责原有私有原子持久化；Sidecar只转发公开SDK，不新建状态权威。

## 8 API/CLI

ExecutionConstraints.webSearch可为disabled/cached/live，默认disabled；仅agent-cli/codex可选择非disabled。ExecutionCapabilities.webSearch给出适配器支持模式，不表示安装/登录/联网就绪。ExecutionContext.nativeWebSearch记录首次实际建立的策略，旧Context缺失按disabled处理，不在恢复时升级。ExecutionRecord.webSearch记录本Run选择。

ExecutionClient.webSearches(runId)和Sidecar web_search.read({runId})返回version/runId/contextId/mode/items/complete。items含nativeItemId、phase(started/completed)、query、action及results；action按固定0.166协议的search/openPage/findInPage/other校验，results仅接受原生数组或null，不解释供应商扩展字段。complete只表示执行成功、确认停止且全部已见条目均完成，不表示取得来源；items为空或results=null不能作为来源发现成功。调用方须显式核对原生结果是否满足自己的领域合同。

## 9 边界

保持精确thread/turn身份和原生工具白名单，只新增已授权webSearch；Shell/文件/浏览器/MCP/Agent仍关闭。每Run最多64个搜索条目、全部已序列化搜索记录至多2MiB，超限立即失败并停止，不静默截断为完整结果。取消/超时沿原执行约束；已开始的原生搜索可能先于事件检查发生，不将记录容量称为上游请求前置限额。结果存储继承私有目录权限，不放进日志或Trace；公开读取只读取指定Run的记录，不读取原生内部会话文件。

## 10 迁移/兼容/回滚

新约束和公共读方法为可选增量，默认禁用保留旧行为。旧Context和Run未记录模式视为disabled；启用新模式须新Context。新增私有记录目录，不改旧产物。回滚时停宿主并保留本次数据，旧版本继续默认禁用，不将新启用的Context交给旧版本恢复。SDK和Sidecar协议版本1不改；已有消费者忽略新增capability字段，仅新消费者主动读取搜索方法。

## 11 测试计划

S1：Unit/Integration覆盖3模式、默认锁、非法模式、非Codex拒绝、恢复不兼容前置拒绝及其它原生能力仍拒绝。S2：Integration通过SDK及Sidecar覆盖2个Run归属、跨会话拒绝、null结果、停止started和2MiB容量失败，观测中无查询/正文。S3：固定原生协议夹具与真实联网分列；真实模型必须登记并计入Inquiry原有累计额度，不为本票隐式增加入口。主要功能地图为23-codex-native-web-search.md，关联21-execution-sidecar.md及22-harness-observation.md；真实搜索未通过时S3保持not_run/blocked，不能合入。

## 12 开放问题

固定CLI的真实提供者是否返回非null结构化results、现有TaskLoop消费者如何显式授权及消费、原生搜索能否满足Inquiry的来源发现/计量合同，均须实证；本能力候选不能自动解决这些问题，不能提前宣称Inquiry解阻。第三轮独立代码审查及真实效果验证仍受已有总预算与轮次约束。

## 13 关联

[Milkie #283](https://github.com/xforce-io/milkie/issues/283)、[Inquiry #7](https://github.com/xforce-io/inquiry/issues/7)、[原生警告修复 #281](https://github.com/xforce-io/milkie/issues/281)。
