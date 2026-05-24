export const STYLES = `
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 24px; background: #f7f7f8; color: #1c1c1e; }
  h1 { font-size: 18px; margin: 0 0 16px 0; font-weight: 600; }
  .run { background: white; border: 1px solid #e5e5e7; border-radius: 8px;
         margin-bottom: 16px; padding: 16px; }
  .run-head { display: flex; gap: 12px; align-items: baseline; margin-bottom: 12px; }
  .run-id { font-family: ui-monospace, SFMono-Regular, monospace; font-size: 12px;
            color: #6e6e73; }
  .badge { font-size: 11px; padding: 2px 6px; border-radius: 4px;
           background: #e8f5e8; color: #2d6a2d; }
  .badge.error { background: #fde8e8; color: #a13; }
  .badge.interrupted { background: #fff4e0; color: #8a5a00; }
  .entry { display: flex; gap: 8px; padding: 6px 0; border-bottom: 1px solid #f0f0f2;
           cursor: pointer; }
  .entry:last-child { border-bottom: none; }
  .entry .icon { width: 16px; text-align: center; }
  .entry.llm .icon { color: #5b3ec9; }
  .entry.tool .icon { color: #2563eb; }
  .entry.lifecycle .icon { color: #6e6e73; }
  .entry .summary { flex: 1; font-size: 13px; }
  .entry .ts { font-family: ui-monospace, monospace; font-size: 11px; color: #6e6e73; }
  .child-run { margin-left: 24px; margin-top: 8px; border-left: 2px solid #e5e5e7;
               padding-left: 12px; }
  .filters { margin-bottom: 16px; display: flex; gap: 8px; flex-wrap: wrap; }
  .chip { font-size: 12px; padding: 4px 10px; border-radius: 999px; cursor: pointer;
          background: white; border: 1px solid #d1d1d6; user-select: none; }
  .chip.active { background: #1c1c1e; color: white; border-color: #1c1c1e; }
  .payload { display: none; margin-top: 6px; background: #fafafa; padding: 8px;
             font-family: ui-monospace, monospace; font-size: 11px; white-space: pre-wrap;
             border-radius: 4px; max-height: 320px; overflow: auto; }
  .entry.open .payload { display: block; }
`

export const SCRIPT = `
  (function () {
    document.addEventListener('click', function (ev) {
      var entry = ev.target.closest('.entry');
      if (entry) entry.classList.toggle('open');
      var chip = ev.target.closest('.chip');
      if (chip) {
        chip.classList.toggle('active');
        var kinds = Array.from(document.querySelectorAll('.chip.active'))
          .map(function (c) { return c.dataset.kind; });
        document.querySelectorAll('.entry').forEach(function (e) {
          var k = e.dataset.kind;
          e.style.display = (kinds.length === 0 || kinds.indexOf(k) >= 0) ? '' : 'none';
        });
      }
    });
  })();
`
