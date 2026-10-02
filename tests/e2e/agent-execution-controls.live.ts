/** Opt-in real fault/control acceptance. Uses only disposable workspaces and synthetic prompts. */
import { spawn } from 'node:child_process'
import { mkdtempSync, mkdirSync, writeFileSync, existsSync, readFileSync, renameSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import assert from 'node:assert/strict'
import { ExecutionClient } from '../../dist/execution/ExecutionClient'
import { prepareDedicatedStorage } from './dedicated-storage'
import { ProcessTracker } from '../../dist/execution/processes'
import { nativeFile } from '../../dist/execution/adapters'

const evidence=(value:unknown)=>console.log(JSON.stringify(value))
async function main() {
  assert.equal(process.env.MILKIE_LIVE_EXECUTION,'1','Real CLI probe requires explicit opt-in')
  const root=mkdtempSync(join(tmpdir(),'milkie-263-controls-'))
  evidence({type:'environment',root})
  for(const runtime of (process.env.MILKIE_LIVE_RUNTIME ? [process.env.MILKIE_LIVE_RUNTIME] : ['grok-cli','pi'])) {
    const cwd=join(root,runtime);mkdirSync(cwd)
    const connection={contractVersion:1,fields:{transport:'agent-cli',runtime}},dataDir=join(cwd,'data')
    const storage=prepareDedicatedStorage(join(cwd,'storage'),runtime)
    const client=new ExecutionClient({dataDir,connection}),seed=client.createContext(cwd,storage)
    const first=await client.wait(client.start(seed.contextId,'Reply only READY. Do not use tools.'))
    evidence({runtime,test:'ready',result:first});assert.equal(first.status,'succeeded')
    const native=nativeFile(client.getContext(seed.contextId))
    renameSync(native,native+'.test-backup')
    try { assert.throws(()=>client.start(seed.contextId,'Continue.'),/session_missing/);assert.equal(existsSync(native),false) }
    finally { renameSync(native+'.test-backup',native) }
    evidence({runtime,test:'missing-session',result:'pass'})
    const authPath=join(seed.configDir!,'auth.json')
    const savedAuth=readFileSync(authPath)
    writeFileSync(authPath,'{"invalid":true}\n')
    let auth
    try { auth=await client.wait(client.start(seed.contextId,'Reply READY.',{timeoutMs:20000})) }
    finally { writeFileSync(authPath,savedAuth) }
    evidence({runtime,test:'invalid-login',result:auth});assert.equal(auth.status,'failed');assert.equal(auth.code,'auth_failed')
    const recovered=await client.wait(client.start(seed.contextId,'Reply only RECOVERED. Do not use tools.'))
    assert.equal(recovered.status,'succeeded');assert.equal(recovered.nativeSessionId,first.nativeSessionId)
    evidence({runtime,test:'explicit-continue-after-auth-repair',result:recovered})
    assert.throws(()=>client.start(seed.contextId,'unused',{costLimit:1} as any),/unsupported_constraint/)
    evidence({runtime,test:'unsupported-constraint',result:'pass'})

    // This program is created by the probe; the real CLI must invoke it through its native tool.
    const task=join(cwd,'task.cjs')
    writeFileSync(task,`const fs=require('fs'),cp=require('child_process');const name=process.argv[2];const child=cp.spawn(process.execPath,['-e','setInterval(()=>{},1000)'],{detached:true,stdio:'ignore'});fs.writeFileSync(name+'.pid',String(child.pid));fs.writeFileSync(name+'.parent',String(process.pid));setInterval(()=>{},1000);`)
    const prompt=(name:string)=>`Use your terminal/bash tool to execute this exact command in the current directory: node ${JSON.stringify(task)} ${name}. Run it now and wait for completion; it intentionally waits. Do not run it in the background and do not alter the program.`
    const waitTask=async(name:string,id:string)=>{
      for(let i=0;i<1200;i++) {
        if(existsSync(join(cwd,name+'.pid'))) return Number(readFileSync(join(cwd,name+'.pid'),'utf8'))
        const r=client.query(id)
        if(r && !['starting','running'].includes(r.status)) throw new Error(`Task never started: ${runtime} ${JSON.stringify(r)}`)
        await delay(50)
      }
      throw new Error('Task tool did not run before probe deadline')
    }
    const scope=client.createContext(cwd,storage)
    const host=spawn(process.execPath,[resolve('tests/fixtures/execution-host.cjs')],{stdio:['pipe','pipe','inherit']})
    const hostExit=new Promise<void>(res=>host.once('exit',()=>res()))
    let buffer=''
    const started=new Promise<{runId:string}>(res=>host.stdout.on('data',b=>{buffer+=b;const end=buffer.indexOf('\n');if(end>=0)res(JSON.parse(buffer.slice(0,end)))}))
    host.stdin.end(JSON.stringify({connection,dataDir,cwd,contextId:scope.contextId,input:prompt('cancel'),constraints:{toolPolicy:'standard',timeoutMs:120000}}))
    const {runId}=await started
    try {
      const pid=await waitTask('cancel',runId)
      // A second real host process attempts the conflicting request.
      const contender=spawn(process.execPath,[resolve('tests/fixtures/execution-host.cjs')],{stdio:['pipe','pipe','inherit']})
      let conflicting='';contender.stdout.on('data',b=>conflicting+=b)
      const contenderExit=new Promise(res=>contender.once('exit',res))
      contender.stdin.end(JSON.stringify({connection,dataDir,cwd,contextId:scope.contextId,input:'Do not run; this must be rejected.'}))
      await contenderExit;assert.ok(conflicting.includes('context_busy'))
      host.kill('SIGKILL');await hostExit
      assert.equal(client.query(runId)?.status,'running')
      const before=Date.now(),cancelled=await new ExecutionClient({dataDir,connection}).cancel(runId)
      evidence({runtime,test:'two-host-busy-host-kill-cancel',elapsedMs:Date.now()-before,detachedTaskPid:pid,result:cancelled})
      assert.ok(Date.now()-before<10000);assert.equal(cancelled.status,'cancelled');assert.equal(cancelled.stopped,true)
      assert.ok(cancelled.resources?.some(p=>p.pid===pid));assert.throws(()=>process.kill(pid,0))
    } finally {
      if(host.exitCode===null && host.signalCode===null)host.kill('SIGKILL')
      await new ProcessTracker(runId).stop()
    }
    const readonlyPath=join(cwd,'after-standard-readonly.txt')
    const readonlyResult=await client.wait(client.start(scope.contextId,'Attempt to write EXACT into after-standard-readonly.txt using an actual tool; report the permission result.',{toolPolicy:'read-only'}))
    evidence({runtime,test:'read-only-after-standard',result:readonlyResult})
    assert.equal(existsSync(readonlyPath),false)
    assert.ok(readonlyResult.status==='succeeded' || (readonlyResult.status==='failed' && readonlyResult.code==='native_cancelled'))
    const deadlineContext=client.createContext(cwd,storage),timedRun=client.start(deadlineContext.contextId,prompt('deadline'),{toolPolicy:'standard',timeoutMs:45000})
    try {
      const pid=await waitTask('deadline',timedRun),result=await client.wait(timedRun,60000)
      evidence({runtime,test:'deadline',detachedTaskPid:pid,result})
      assert.equal(result.status,'timed_out');assert.equal(result.stopped,true);assert.ok(result.resources?.some(p=>p.pid===pid));assert.throws(()=>process.kill(pid,0))
    } finally { await new ProcessTracker(timedRun).stop() }
    evidence({runtime,result:'pass',scope:['S3.A2','S4.A1','S5.cancel','S5.deadline','S5.unsupported']})
  }
}
main().catch(error=>{evidence({result:'fail',message:error instanceof Error?error.message:'probe failed'});process.exitCode=1})
