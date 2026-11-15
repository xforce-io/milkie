// Opt-in real Codex0.166.0 + SDK, synthetic loopback provider and injected valid warning.
// Zero real model calls; no user login material. This is not article-quality evidence.
const fs = require('node:fs'), path = require('node:path'), os = require('node:os'), http = require('node:http'), assert = require('node:assert/strict')
const { ExecutionClient } = require(path.resolve('dist/execution/ExecutionClient.js'))
const nativeSearch = process.env.MILKIE_NATIVE_SCENARIO === 'web-search'
const binary = process.env.MILKIE_NATIVE_CODEX
assert(binary && path.isAbsolute(binary), 'Set MILKIE_NATIVE_CODEX to fixed Codex0.166.0')
const evidence = path.resolve(process.env.MILKIE_EVIDENCE_DIR || `test-output/verify-milkie/codex-warning-${Date.now()}`)
fs.mkdirSync(evidence, { recursive: true })
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-codex-warning-'))
for (const d of ['config', 'sessions', 'work', 'bin']) fs.mkdirSync(path.join(root, d))
let requests = 0, effects = 0, safeToRemove = true
const advertisedToolTypes = []
const server = http.createServer((req, res) => {
  let body = ''
  req.on('data', chunk => { body += chunk })
  req.on('end', () => {
  const request = JSON.parse(body)
  advertisedToolTypes.push((request.tools || []).map(t => t.type))
  requests++
  const item = requests === 1
    ? nativeSearch ? { id: 'search_fixture', type: 'web_search_call', status: 'completed', action: { type: 'search', query: 'synthetic-native-query' } } : { id: 'tool_fixture', type: 'custom_tool_call', status: 'completed', call_id: 'call_fixture', name: 'exec', input: 'const r = await tools.alpha({n:1}); text(r);' }
    : { id: 'msg_fixture', type: 'message', status: 'completed', role: 'assistant', content: [{ type: 'output_text', text: 'OK', annotations: [] }] }
  const message = { id:'msg_search_fixture',type:'message',status:'completed',role:'assistant',content:[{type:'output_text',text:'OK',annotations:[]}] }
  const output = nativeSearch ? [item,message] : [item]
  const response = { id: `resp_fixture_${requests}`, object: 'response', created_at: Math.floor(Date.now() / 1000), status: 'completed', output, usage: { input_tokens: 10, output_tokens: 1, total_tokens: 11 } }
  res.writeHead(200, { 'content-type': 'text/event-stream' })
  for (const event of [
    { type: 'response.created', response: { ...response, status: 'in_progress', output: [] } },
    { type: 'response.output_item.added', output_index: 0, item: requests === 1 ? { ...item, status: 'in_progress', input: '' } : { ...item, status: 'in_progress', content: [] } },
    ...(requests === 1 && nativeSearch ? [] : requests === 1
      ? [{ type: 'response.custom_tool_call_input.delta', output_index: 0, item_id: item.id, delta: item.input }]
      : [{ type: 'response.content_part.added', output_index: 0, item_id: item.id, content_index: 0, part: { type: 'output_text', text: '', annotations: [] } }, { type: 'response.output_text.delta', output_index: 0, item_id: item.id, content_index: 0, delta: 'OK' }]),
    { type: 'response.output_item.done', output_index: 0, item },
    ...(nativeSearch ? [{ type:'response.output_item.added', output_index:1, item:{...message,status:'in_progress',content:[]} }, {type:'response.content_part.added',output_index:1,item_id:message.id,content_index:0,part:{type:'output_text',text:'',annotations:[]}},{type:'response.output_text.delta',output_index:1,item_id:message.id,content_index:0,delta:'OK'},{type:'response.output_item.done',output_index:1,item:message}] : []),
    { type: 'response.completed', response },
  ]) res.write(`data: ${JSON.stringify(event)}\n\n`)
  res.end()
  })
})
;(async () => {
  await new Promise(r => server.listen(0, '127.0.0.1', r))
  const port = server.address().port
  fs.writeFileSync(path.join(root, 'config/auth.json'), JSON.stringify({ OPENAI_API_KEY: 'fixture-not-real' }), { mode: 0o600 })
  fs.writeFileSync(path.join(root, 'config/config.toml'), `model_reasoning_effort = "high"\nmodel_provider = "fixture"\n[model_providers.fixture]\nname = "fixture"\nbase_url = "http://127.0.0.1:${port}/v1"\nwire_api = "responses"\nrequires_openai_auth = false\nrequest_max_retries = 0\nstream_max_retries = 0\n`)
  const marker = path.join(root, 'warning-injected')
  fs.writeFileSync(path.join(root, 'bin/codex'), `#!${process.execPath}
const {spawn}=require('node:child_process'),fs=require('node:fs');
const args=process.argv.slice(2),child=spawn(${JSON.stringify(binary)},args,{stdio:['pipe','pipe','inherit']});
process.stdin.pipe(child.stdin);child.stdin.on('error',()=>{});let buffer='',injected=false;
child.stdout.on('data',chunk=>{if(!args.includes('app-server')){process.stdout.write(chunk);return}buffer+=chunk.toString('utf8');let end;while((end=buffer.indexOf('\\n'))>=0){const line=buffer.slice(0,end);buffer=buffer.slice(end+1);const m=JSON.parse(line);if(!injected&&m.method==='item/completed'&&m.params?.item?.type==='dynamicToolCall'){process.stdout.write(JSON.stringify({method:'warning',params:{threadId:null,message:'Synthetic global warning DO-NOT-LEAK-WARNING'}})+'\\n');fs.writeFileSync(${JSON.stringify(marker)},'once');injected=true}process.stdout.write(line+'\\n')}});
child.on('exit',(code,signal)=>{if(signal)process.kill(process.pid,signal);else process.exit(code??1)});for(const s of ['SIGINT','SIGTERM'])process.on(s,()=>child.kill(s));
`, { mode: 0o700 })
  const client = new ExecutionClient({ dataDir: path.join(root, 'data'), connection: { contractVersion: 1, fields: { transport: 'agent-cli', runtime: 'codex', model: 'gpt-6.1-sol' } }, env: { ...process.env, PATH: `${path.join(root, 'bin')}:${process.env.PATH}`, NO_PROXY: '127.0.0.1,localhost', no_proxy: '127.0.0.1,localhost' }, lifetime: 'host' })
  const c = client.createContext(path.join(root, 'work'), { configDir: path.join(root, 'config'), sessionDir: path.join(root, 'sessions') })
  const id = client.start(c.contextId, 'Local synthetic tool then OK.', { timeoutMs: 15000, ...(nativeSearch ? {webSearch:'live',observation:{traceparent:'00-'+'1'.repeat(32)+'-'+'2'.repeat(16)+'-01',maxEvents:100,maxBytes:65536}} : {}), tools: [{ name: 'alpha', description: 'Synthetic local tool', inputSchema: { type: 'object', properties: { n: { type: 'integer' } }, required: ['n'], additionalProperties: false } }] }, () => { effects++; return { ok: true, output: 'accepted-once' } })
  safeToRemove = false
  let result
  const deadline = Date.now() + 20000
  while (Date.now() < deadline) { result = client.query(id); if (result?.finishedAt) break; await new Promise(r => setTimeout(r, 25)) }
  if (!result?.finishedAt) { result = await client.cancel(id); throw new Error('Native warning fixture timed out') }
  safeToRemove = result.stopped === true
  const searches = nativeSearch ? client.webSearches(id) : undefined
  if (searches) fs.writeFileSync(path.join(evidence, 'searches.json'), JSON.stringify(searches,null,2))
  const summary = { nativeSearch, advertisedToolTypes, runtime: 'Codex0.166.0 + actual SDK', provider: 'loopback synthetic', realModelEntries: 0, injectedWarning: fs.existsSync(marker), requests, effects, status: result.status, code: result.code, output: result.output, stopped: result.stopped }
  fs.writeFileSync(path.join(evidence, 'summary.json'), JSON.stringify(summary, null, 2))
  console.log(JSON.stringify(summary))
  if (nativeSearch) {
    assert.equal(effects,0);assert.equal(searches.items.length,1);assert.equal(searches.items[0].query,'synthetic-native-query');assert.equal(searches.items[0].phase,'completed');assert.equal(searches.items[0].results,null);assert.equal(searches.complete,true);assert(!JSON.stringify(client.observations(id)).includes('synthetic-native-query'))
  } else { assert.equal(summary.injectedWarning, true); assert.equal(effects, 1); }
  assert.equal(result.status, 'succeeded'); assert.equal(result.output, 'OK'); assert.equal(result.stopped, true); assert.equal(requests, nativeSearch ? 1 : 2)
  assert(!JSON.stringify(result).includes('DO-NOT-LEAK-WARNING'))
})().catch(e => { console.error(e.message); process.exitCode = 1 }).finally(() => { server.close(); if (safeToRemove) fs.rmSync(root, { recursive: true, force: true }); else console.error(`Preserved unresolved fixture execution: ${root}`) })
