// #132: 正文逐句内联引用角标(序号锚点,零字符串匹配)。
//
// agent 在正文每条陈述末尾写 [n](n = 它第几次 cite);lineage 的 cites 关系
// 按出现顺序天然给出第 n 个 claim。序号是 agent 与前端共享的锚点 —— 无需任何
// 文本比对。本模块把已渲染的正文 html 里、范围内的 [n] 转成可点击角标;越界/
// 无号/count=0 原样保留(容错,且避免误伤正文里真实的 [数字])。
//
// UMD:浏览器经 <script src> 挂到 window.linkifyCitations;jest 经 CommonJS
// require 拿到 module.exports(配套 citations.d.ts 提供类型)。无需任何 jest
// transform —— 纯 CJS 文件被 node 直接加载。
;(function (root, factory) {
  var api = factory()
  if (typeof module !== 'undefined' && module.exports) module.exports = api
  if (root) root.linkifyCitations = api.linkifyCitations
})(typeof window !== 'undefined' ? window : null, function () {
  function linkifyCitations(html, count) {
    if (!count || count < 1) return html
    return html.replace(/\[(\d+)\]/g, function (marker, digits) {
      var n = Number(digits)
      if (n >= 1 && n <= count) {
        return '<sup class="cite-ref" data-cite-index="' + n + '">[' + n + ']</sup>'
      }
      return marker
    })
  }
  return { linkifyCitations: linkifyCitations }
})
