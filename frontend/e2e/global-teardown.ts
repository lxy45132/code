import { execSync } from 'node:child_process'
import { readFileSync, unlinkSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const dirname = path.dirname(fileURLToPath(import.meta.url))

export default async function globalTeardown() {
  const pidFile = path.join(dirname, '.worker.pid')
  try {
    const pid = readFileSync(pidFile, 'utf8').trim()
    // Windows：结束进程树（shell → uv → python worker）
    execSync(`taskkill /pid ${pid} /T /F`, { stdio: 'ignore' })
  } catch {
    // worker 已退出
  }
  try {
    unlinkSync(pidFile)
  } catch {
    // ignore
  }
}
