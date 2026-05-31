export const VIEWER_STYLES = `
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f7f7f8; color: #1c1c1e; }
  h1 { font-size: 16px; margin: 0; padding: 12px 16px; border-bottom: 1px solid #e5e5e7; background: #fff; }
  .tabs { display: flex; gap: 8px; padding: 8px 16px; background: #fff; border-bottom: 1px solid #e5e5e7; }
  .tab { font-size: 12px; padding: 4px 12px; border-radius: 999px; border: 1px solid #d1d1d6; cursor: pointer; background: #fff; user-select: none; }
  .tab.active { background: #1c1c1e; color: #fff; border-color: #1c1c1e; }
  .pane { display: none; }
  .pane.active { display: block; }
  #pane-decision { display: flex; height: calc(100vh - 90px); }
  #pane-decision.active { display: flex; }
  .spine { width: 42%; border-right: 1px solid #e5e5e7; overflow: auto; padding: 8px; background: #fafafa; }
  .node { font-family: ui-monospace, SFMono-Regular, monospace; font-size: 12px; padding: 5px 8px; margin: 3px 0; border-left: 3px solid transparent; border-radius: 3px; cursor: pointer; background: #fff; }
  .node.k-llm { border-left-color: #5b3ec9; } .node.k-tool { border-left-color: #2563eb; }
  .node.k-transition { border-left-color: #8a5a00; } .node.k-output { border-left-color: #1c1c1e; font-weight: 600; }
  .node.selected { outline: 2px solid #f0a; }
  .node.cause { background: #eef6ff; }
  .spine-output .why-entry { color: #c026a6; font-size: 11px; margin-left: 6px; }
  #why-panel { width: 58%; overflow: auto; padding: 14px; background: #fff; font-size: 13px; line-height: 1.6; }
  #why-panel .ph { color: #888; }
  #why-panel h3 { font-size: 14px; margin: 0 0 8px; }
  .why-block { background: #f7f7f8; padding: 10px; border-radius: 6px; margin-bottom: 10px; }
  .nav-link { display: block; padding: 6px 10px; border-radius: 5px; margin: 4px 0; cursor: pointer; }
  .nav-cause { background: #eef6ff; color: #2563eb; }
  .xlink { color: #36a; text-decoration: none; cursor: pointer; }
  .xlink:hover { text-decoration: underline; }
  .xdim { color: #aaa; }
  .rdetail { margin: 2px 0; font-family: ui-monospace, monospace; font-size: 11px; }
  .rdetail summary { cursor: pointer; }
  .rpre { white-space: pre-wrap; background: #fafafa; padding: 6px; border-radius: 4px; margin-top: 3px; }
  .ar-na { color: #a13; }
  .rawpre { font-family: ui-monospace, monospace; font-size: 11px; white-space: pre-wrap; background: #fafafa; padding: 8px; border-radius: 4px; max-height: 240px; overflow: auto; }
  #pane-raw { padding: 12px 16px; }
`

export const VIEWER_SCRIPT = `
(function () {
  var exps  = JSON.parse(document.getElementById('explanations-data').textContent || '{}');
  function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
  function nodeEl(id){ return document.querySelector('.node[data-id="'+CSS.escape(id)+'"]'); }
  function clearMarks(){ document.querySelectorAll('.node.selected,.node.cause').forEach(function(n){ n.classList.remove('selected','cause'); }); }
  function chainHtml(chain){ return (chain||[]).map(function(c){
    if(exps[c.eventId]){ return '<a class="xlink" data-go="'+esc(c.eventId)+'">'+esc(c.summary)+'</a>'; }
    return '<span class="xdim">'+esc(c.summary)+'</span>';
  }).join(' → '); }
  function navHtml(causeId){
    var out='';
    if(causeId){ out += '<a class="nav-link nav-cause" data-go="'+esc(causeId)+'">← 谁导致的:'+esc((exps[causeId]&&exps[causeId].title)||causeId)+'</a>'; }
    return out;
  }
  function panelHtml(id){
    var x = exps[id]; if(!x){ return '<p class="ph">选一个决策节点看 why</p>'; }
    var h = '<h3>'+esc(x.title)+'</h3>';
    h += '<div class="why-block">'+x.bodyHtml+'</div>';
    h += '<div class="why-block">'+navHtml(x.causeDecisionId)+'<div style="color:#888;font-size:11px;margin-top:6px">因果链: '+chainHtml(x.chain)+'</div></div>';
    h += '<details><summary style="cursor:pointer;color:#888;font-size:11px">原始 payload</summary><pre class="rawpre">'+esc(x.rawJson)+'</pre></details>';
    return h;
  }
  function selectNode(id){
    clearMarks();
    var n = nodeEl(id); if(n){ n.classList.add('selected'); }
    var cid = exps[id] && exps[id].causeDecisionId;
    if(cid){ var cn = nodeEl(cid); if(cn){ cn.classList.add('cause'); } }
    document.getElementById('why-panel').innerHTML = panelHtml(id);
  }
  document.addEventListener('click', function(ev){
    var go = ev.target.closest('[data-go]'); if(go){ selectNode(go.getAttribute('data-go')); return; }
    var node = ev.target.closest('.node[data-id]'); if(node){ selectNode(node.getAttribute('data-id')); return; }
    var tab = ev.target.closest('.tab[data-tab]');
    if(tab){
      document.querySelectorAll('.tab').forEach(function(t){ t.classList.remove('active'); });
      tab.classList.add('active');
      document.querySelectorAll('.pane').forEach(function(p){ p.classList.remove('active'); });
      document.getElementById('pane-'+tab.getAttribute('data-tab')).classList.add('active');
    }
  });
})();
`
