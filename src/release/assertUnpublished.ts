export interface RegistryResponse {
  readonly status: number
}

export type RegistryRequest = (url: string) => Promise<RegistryResponse>

export interface AssertUnpublishedVersionOptions {
  readonly packageName: string
  readonly version: string
  readonly registryUrl: string
  readonly request: RegistryRequest
}

export async function assertUnpublishedVersion({
  packageName,
  version,
  registryUrl,
  request,
}: AssertUnpublishedVersionOptions): Promise<void> {
  const baseUrl = registryUrl.replace(/\/$/, '')
  const packagePath = encodeURIComponent(packageName)
  const versionPath = encodeURIComponent(version)
  const response = await request(`${baseUrl}/${packagePath}/${versionPath}`)

  if (response.status === 404) return
  if (response.status === 200) {
    throw new Error(`${packageName}@${version} is already published`)
  }
  throw new Error(
    `cannot verify ${packageName}@${version} is unpublished: registry returned HTTP ${response.status}`,
  )
}

async function main(): Promise<void> {
  const version = process.env['PACKAGE_VERSION']
  if (!version) throw new Error('PACKAGE_VERSION must be set')

  await assertUnpublishedVersion({
    packageName: process.env['PACKAGE_NAME'] ?? '@xforce/milkie',
    version,
    registryUrl: process.env['NPM_REGISTRY_URL'] ?? 'https://registry.npmjs.org',
    request: async url => {
      const response = await fetch(url)
      return { status: response.status }
    },
  })
}

if (require.main === module) {
  void main().catch(error => {
    const message = error instanceof Error ? error.message : String(error)
    process.stderr.write(`${message}\n`)
    process.exitCode = 1
  })
}
