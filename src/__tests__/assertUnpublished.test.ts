import { assertUnpublishedVersion } from '../release/assertUnpublished.js'

type Response = { status: number }

describe('assertUnpublishedVersion', () => {
  const packageName = '@freemanxu/milkie'
  const version = '0.1.1'

  it('permits a version only when the registry returns not found', async () => {
    const requested: unknown[][] = []

    await expect(assertUnpublishedVersion({
      packageName,
      version,
      registryUrl: 'https://registry.example/',
      request: ((...args: unknown[]) => {
        requested.push(args)
        return Promise.resolve({ status: 404 } as Response)
      }) as never,
    })).resolves.toBeUndefined()

    expect(requested).toEqual([[
      'https://registry.example/%40freemanxu%2Fmilkie/0.1.1',
      { redirect: 'error' },
    ]])
  })

  it('rejects an already published version before publish', async () => {
    await expect(assertUnpublishedVersion({
      packageName,
      version,
      registryUrl: 'https://registry.example',
      request: async () => ({ status: 200 } as Response),
    })).rejects.toThrow('@freemanxu/milkie@0.1.1 is already published')
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
