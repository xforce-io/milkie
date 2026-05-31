/**
 * Minimal, dependency-free markdown → HTML for the trace viewer's final-output
 * panel. Escapes HTML FIRST, then renders a small subset (headers, bold, inline
 * code, lists, paragraphs). All emitted tags are ours — no raw HTML passes
 * through, so the result is safe to inject via innerHTML.
 */
function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function inline(s: string): string {
  // s is already HTML-escaped. Bold first, then inline code.
  return s
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
}

export function renderMarkdown(text: string): string {
  const lines = escapeHtml(text).split(/\r?\n/)
  const out: string[] = []
  let para: string[] = []
  let list: { type: 'ul' | 'ol'; items: string[] } | null = null

  const flushPara = () => {
    if (para.length) { out.push(`<p>${para.map(inline).join('<br>')}</p>`); para = [] }
  }
  const flushList = () => {
    if (list) {
      out.push(`<${list.type}>${list.items.map(i => `<li>${inline(i)}</li>`).join('')}</${list.type}>`)
      list = null
    }
  }
  const flushAll = () => { flushPara(); flushList() }

  for (const line of lines) {
    const h  = line.match(/^(#{1,3})\s+(.*)$/)
    const ul = line.match(/^[-*]\s+(.*)$/)
    const ol = line.match(/^\d+\.\s+(.*)$/)
    if (h) {
      flushAll()
      const level = h[1]!.length + 2  // # -> h3, ## -> h4, ### -> h5
      out.push(`<h${level}>${inline(h[2]!)}</h${level}>`)
    } else if (ul) {
      flushPara()
      if (!list || list.type !== 'ul') { flushList(); list = { type: 'ul', items: [] } }
      list.items.push(ul[1]!)
    } else if (ol) {
      flushPara()
      if (!list || list.type !== 'ol') { flushList(); list = { type: 'ol', items: [] } }
      list.items.push(ol[1]!)
    } else if (line.trim() === '') {
      flushAll()
    } else {
      flushList()
      para.push(line)
    }
  }
  flushAll()
  return out.join('')
}
