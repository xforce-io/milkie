import { assertUnpublishedVersion } from '../release/assertUnpublished.js'

type Response = { status: number }

describe('assertUnpublishedVersion', () => {
  const packageName = '@xforce/milkie'
  const version = '0.1.0'

  it('permits a version only when the registry returns not found', async () => {
    const requested: string[] = []

    await expect(assertUnpublishedVersion({
      packageName,
      version,
      registryUrl: 'https://registry.example/',
      request: async url => {
        requested.push(url)
        return { status: 404 } as Response
      },
    })).resolves.toBeUndefined()

    expect(requested).toEqual(['https://registry.example/%40xforce%2Fmilkie/0.1.0'])
  })

  it('rejects an already published version before publish', async () => {
    await expect(assertUnpublishedVersion({
      packageName,
      version,
      registryUrl: 'https://registry.example',
      request: async () => ({ status: 200 } as Response),
    })).rejects.toThrow('@xforce/milkie@0.1.0 is already published')
  })

  it('fails closed when the registry cannot prove the version is absent', async () => {
    await expect(assertUnpublishedVersion({
      packageName,
      version,
      registryUrl: 'https://registry.example',
      request: async () => ({ status: 401 } as Response),
    })).rejects.toThrow('registry returned HTTP 401')
  })
})
