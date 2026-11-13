import { spawn } from 'node:child_process'
import { writeFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const dirname = path.dirname(fileURLToPath(import.meta.url))

// 启动真实 solo Celery worker（FakeEngine 注入）。任务先入 broker 排队，
// worker 就绪后消费，无需等待启动完成。
export default async function globalSetup() {
  const worker = spawn('uv', ['run', '--no-sync', 'python', '-m', 'scripts.e2e_worker'], {
    cwd: path.resolve(dirname, '../..'),
    shell: true,
    stdio: 'ignore',
    env: { ...process.env, E2E_ITEM_DELAY: '0.8' },
  })
  writeFileSync(path.join(dirname, '.worker.pid'), String(worker.pid))
}
