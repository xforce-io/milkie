// 测试进程默认静默服务日志（#79）；显式注入 logger 的用例不受影响。
process.env['LOG_LEVEL'] ??= 'silent'
