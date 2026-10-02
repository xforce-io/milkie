// Reproduce an OS signal-denial boundary in the supervisor only.
const originalKill = process.kill.bind(process)
process.kill = (pid, signal) => {
  if (signal !== 0) throw Object.assign(new Error('fixture denied signal'), {code:'EPERM'})
  return originalKill(pid, signal)
}
