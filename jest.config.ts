import type { Config } from 'jest'

const config: Config = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  rootDir: '.',
  testMatch: ['<rootDir>/src/**/*.test.ts', '<rootDir>/tests/**/*.test.ts'],
  moduleNameMapper: {
    '^@milkie/(.*)$':   '<rootDir>/src/$1',
    '^(\\.{1,2}/.+)\\.js$': '$1',   // strip .js for ts-jest CJS resolution
  },
  collectCoverageFrom: ['src/**/*.ts', '!src/**/*.d.ts'],
  transform: {
    '^.+\\.tsx?$': ['ts-jest', {
      tsconfig: {
        strict:          true,
        esModuleInterop: true,
        skipLibCheck:    true,
      },
    }],
  },
  testTimeout: 60000,
}

export default config
