# #281 调查与验证

## Overview / 怎么跑

`src/execution/worker.ts`为Codex建立`CodexExecution`；`codex.ts::run`先锁定能力、创建确切thread，再启动turn。`message`在分发所有帧前检查threadId/turnId，宿主工具经`createToolDispatcher`处理，worker最终负责停止与发布结果。此次失败来自顶层threadId比较，不是登录或保存文件检查。

## 问题证据

- 原入口：Inquiry公开Goals API单Planner；Inquiry3dbe328、Milkie98b5ec1、Codex0.166.0、gpt-6.1-sol/high、当前有效登录。两轮实际失败均保留，后轮透传观察只写方法/身份，不改字节。
- 后轮末帧：一次item/tool/call及结果送达→thread/tokenUsage/updated→account/rateLimits/updated→warning(threadId:null)；随后session_mismatch，未成稿。原日志在Inquiry工作树`work/verify/7-sol-protocol-3dbe328/native-identities.ndjson`及`trace.json`，生成Schema在前轮`protocol-schema/v2/WarningNotification.json`。真实模型累计入口8，不清零。
- 固定CLI生成的Schema说明warning的threadId为可选string/null。`src/execution/codex.ts::message`把null当成具体目标相比较，足以解释且确定性复现实际错误；先前“并发其他会话”假设没有证据，不作为根因。
- 修复前：当前84580c1源代码只加入协议夹具/断言，运行`jest ...codexExecution.test.ts -t 'warning after tool delivery'`：global失败、省略/指定当前目标2通过，退出1；日志`test-output/281-codex-global-warning/red-warning.log`。合成工具效果已发生，随后误判失败，符合真实触发链。

## 验收状态

设计v0.1的S1/S2待冻结候选复验；确定性复现不是修复生效证明。L1 reuse，L2 write；源代码的两处身份校验之外不改行为。无新增凭据、模型额度或自动重试。

开发验证：定向Codex套件通过；固定真实CLI接本地合成提供者及同形态警告注入，旧dist失败session_mismatch，新dist成功输出OK，均2次本地请求/1次宿主效果/确认停止，真实模型0次。summary分别在native-red/与native-green/。第一次准备漏建假auth.json，启动前config_missing，已保留native-setup-failure.log；未当作产品复现。Unit140项、确定性E2E10项、地图检查39 SDK/12 CLI/14 HTTP入口且0错误；完整执行回归仍在运行。最终验收须冻结后另记，不把开发日志改SHA。
