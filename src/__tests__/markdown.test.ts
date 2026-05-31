import { renderMarkdown } from '../trace/render/markdown'

describe('renderMarkdown', () => {
  it('renders headers # ## ### as h3 h4 h5', () => {
    expect(renderMarkdown('# A')).toBe('<h3>A</h3>')
    expect(renderMarkdown('## B')).toBe('<h4>B</h4>')
    expect(renderMarkdown('### C')).toBe('<h5>C</h5>')
  })

  it('renders bold and inline code', () => {
    expect(renderMarkdown('**x**')).toBe('<p><strong>x</strong></p>')
    expect(renderMarkdown('`y`')).toBe('<p><code>y</code></p>')
  })

  it('renders unordered and ordered lists', () => {
    expect(renderMarkdown('- a\n- b')).toBe('<ul><li>a</li><li>b</li></ul>')
    expect(renderMarkdown('1. a\n2. b')).toBe('<ol><li>a</li><li>b</li></ol>')
  })

  it('groups consecutive non-empty lines into a paragraph with <br>', () => {
    expect(renderMarkdown('l1\nl2')).toBe('<p>l1<br>l2</p>')
  })

  it('escapes HTML before applying markdown (no injection)', () => {
    expect(renderMarkdown('<script>alert(1)</script>')).toBe('<p>&lt;script&gt;alert(1)&lt;/script&gt;</p>')
  })

  it('escapes HTML inside bold, inline code, and headers (no XSS through markup)', () => {
    expect(renderMarkdown('**<img src=x onerror=alert(1)>**'))
      .toBe('<p><strong>&lt;img src=x onerror=alert(1)&gt;</strong></p>')
    expect(renderMarkdown('`<script>`')).toBe('<p><code>&lt;script&gt;</code></p>')
    expect(renderMarkdown('# <b>Header</b>')).toBe('<h3>&lt;b&gt;Header&lt;/b&gt;</h3>')
  })

  it('returns empty string for empty input', () => {
    expect(renderMarkdown('')).toBe('')
  })
})
