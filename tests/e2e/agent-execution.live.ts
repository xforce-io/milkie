/** Opt-in real SDK probe: MILKIE_LIVE_EXECUTION=1 tsx tests/e2e/agent-execution.live.ts
 * Emits synthetic-only evidence. This probe covers S2/S3 and part of S5, not all acceptance.
 */
import { spawn } from 'node:child_process'
import { mkdtempSync, existsSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { randomUUID } from 'node:crypto'
import assert from 'node:assert/strict'
import { ExecutionClient } from '../../dist/execution/ExecutionClient'
import { prepareDedicatedStorage } from './dedicated-storage'

async function main() {
  if (process.env.MILKIE_LIVE_EXECUTION !== '1') throw new Error('Explicit MILKIE_LIVE_EXECUTION=1 is required; real CLI requests use the current login.')
  const root=mkdtempSync(join(tmpdir(),'milkie-263-live-'))
  console.log(JSON.stringify({type:'environment',root,node:process.version}))
  for (const runtime of (process.env.MILKIE_LIVE_RUNTIME ? [process.env.MILKIE_LIVE_RUNTIME] : ['grok-cli','pi'])) {
    const connection={contractVersion:1,fields:{transport:'agent-cli',runtime}}
    const dataDir=join(root,`${runtime}-data`), client=new ExecutionClient({dataDir,connection})
    const storage=prepareDedicatedStorage(join(root,`${runtime}-storage`), runtime)
    const turn=async (input:string,contextId?:string,allowNativeRefusal=false) => {
      const child=spawn(process.execPath,[resolve('tests/fixtures/execution-host.cjs')],{stdio:['pipe','pipe','inherit']})
      let output='';child.stdout.on('data',b=>output+=b)
      const exit=new Promise<number|null>((res,rej)=>{child.on('error',rej);child.on('exit',res)})
      child.stdin.end(JSON.stringify({connection,dataDir,cwd:root,contextId,storage,input,constraints:{toolPolicy:'read-only',timeoutMs:120000}}))
      const code=await exit
      const events=output.trim().split('\n').filter(Boolean).map(line=>JSON.parse(line))
      console.log(JSON.stringify({runtime,hostExited:true,exitCode:code,events}))
      assert.equal(code,0)
      const result=events.find(e=>e.type==='result')?.result
      if (allowNativeRefusal && result?.status==='failed') assert.equal(result.code,'native_cancelled')
      else assert.equal(result?.status,'succeeded')
      assert.equal(result?.stopped,true)
      return result
    }
    const marker=`synthetic-${randomUUID()}`
    const first=await turn(`Remember exactly this synthetic marker: ${marker}. Reply ACK only. Do not use tools.`)
    const otherMarker=`synthetic-${randomUUID()}`
    const other=await turn(`Remember exactly this synthetic marker: ${otherMarker}. Reply ACK only. Do not use tools.`)
    for (let round=0;round<2;round++) {
      const next=await turn('Repeat only the synthetic marker previously given in this conversation. Do not use tools or read files.',first.contextId)
      assert.ok(next.output.includes(marker));assert.ok(!next.output.includes(otherMarker));assert.equal(next.nativeSessionId,first.nativeSessionId);assert.notEqual(next.runId,first.runId)
    }
    const nextOther=await turn('Repeat only the synthetic marker previously given in this conversation. Do not use tools or read files.',other.contextId)
    assert.ok(nextOther.output.includes(otherMarker));assert.ok(!nextOther.output.includes(marker))
    const fresh=await turn('Reply only FRESH. Do not use tools.')
    assert.notEqual(fresh.nativeSessionId,first.nativeSessionId);assert.notEqual(fresh.nativeSessionId,other.nativeSessionId)
    const filename=`readonly-${runtime}.txt`
    await turn(`Attempt to write EXACT to ${filename} in the current working directory using an actual tool. Report whether writing was permitted.`,fresh.contextId,true)
    assert.equal(existsSync(join(root,filename)),false)
    const sentinel=`cwd-${randomUUID()}`;writeFileSync(join(root,`${runtime}-sentinel.txt`),sentinel)
    const cwd=await turn(`Read ${runtime}-sentinel.txt in the current working directory using a tool and return its exact content.`,fresh.contextId)
    assert.ok(cwd.output.includes(sentinel))
    assert.equal(client.capabilities().availability,'unchecked')
    console.log(JSON.stringify({runtime,result:'pass',scope:['S2.A1','S3.A1','S5.read-only','S5.cwd']}))
  }
}
main().catch(error=>{console.error(JSON.stringify({result:'fail',reason:error instanceof Error ? error.message : 'probe failed'}));process.exitCode=1})
