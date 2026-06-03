import { linkifyCitations } from '../public/citations.js'

// linkifyCitations(html, count): 把正文 html 里的 [n](1<=n<=count)转成可点击角标
// <sup class="cite-ref" data-cite-index="n">[n]</sup>。锚点机制见 issue #132:
// 序号 n 对齐 lineage 里第 n 个 cite,零字符串匹配。

const sup = (n: number) => `<sup class="cite-ref" data-cite-index="${n}">[${n}]</sup>`

describe('linkifyCitations', () => {
  it('把范围内的 [n] 逐个转成角标', () => {
    expect(linkifyCitations('曹军惨败[1] 曹操北逃[2]', 2))
      .toBe(`曹军惨败${sup(1)} 曹操北逃${sup(2)}`)
  })

  it('越界的 [n](n>count)原样保留', () => {
    expect(linkifyCitations('张三[9]说', 2)).toBe('张三[9]说')
  })

  it('count=0 时不转任何 [n]', () => {
    expect(linkifyCitations('a[1]b', 0)).toBe('a[1]b')
  })

  it('没有 marker 时原样返回', () => {
    expect(linkifyCitations('纯文本,无引用', 3)).toBe('纯文本,无引用')
  })

  it('与已渲染的粗体 html 共存,只动 [n]', () => {
    expect(linkifyCitations('<strong>曹操</strong>[1]', 1))
      .toBe(`<strong>曹操</strong>${sup(1)}`)
  })

  it('支持多位数序号', () => {
    expect(linkifyCitations('末尾[10]', 10)).toBe(`末尾${sup(10)}`)
    expect(linkifyCitations('末尾[10]', 9)).toBe('末尾[10]')
  })

  it('[0] 视为越界(序号从 1 起),原样保留', () => {
    expect(linkifyCitations('零[0]', 3)).toBe('零[0]')
  })
})
