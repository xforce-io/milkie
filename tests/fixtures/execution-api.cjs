// API gateway fixture loaded only by the child process in the integration suite.
const assembly = require('../../dist/connection/assemble.js')
assembly.assembleApiGateway = () => ({ gateway: {
  complete: async (request, options) => {
    const text = request.messages[0].content[0].text
    if (text === 'fixture:busy') { const until=Date.now()+450; while (Date.now()<until) {} }
    if (text === 'fixture:sleep') await new Promise((resolve,reject) => {
      options.signal.addEventListener('abort', () => reject(new Error('aborted')), {once:true})
    })
    return { content: [{type:'text',text:`api:${text}`}], toolCalls:[] }
  }
} })
