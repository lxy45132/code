import path from 'node:path'
import { fileURLToPath } from 'node:url'

const dirname = path.dirname(fileURLToPath(import.meta.url))

export function fixture(name: string): string {
  return path.join(dirname, 'fixtures', name)
}
