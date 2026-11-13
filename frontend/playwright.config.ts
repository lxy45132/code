import { defineConfig } from '@playwright/test'

// e2e 对真实链路实跑：FastAPI（托管 dist）+ 真实 Redis + solo Celery worker。
// 前置：Redis 运行中；npm run build 已产出 dist（test:e2e 脚本自动构建）。
export default defineConfig({
  testDir: './e2e',
  timeout: 90_000,
  expect: { timeout: 20_000 },
  workers: 1,
  retries: 0,
  use: {
    baseURL: 'http://127.0.0.1:8000',
    // 复用系统已装的 Edge（Chromium 内核），免下载专用浏览器
    channel: 'msedge',
    trace: 'retain-on-failure',
  },
  globalSetup: './e2e/global-setup.ts',
  globalTeardown: './e2e/global-teardown.ts',
  webServer: {
    command: 'uv run --no-sync python -m scripts.e2e_server',
    cwd: '..',
    url: 'http://127.0.0.1:8000/api/v1/health',
    reuseExistingServer: true,
    timeout: 60_000,
  },
})
