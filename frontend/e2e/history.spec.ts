import { expect, test } from '@playwright/test'
import { readFileSync } from 'node:fs'
import { fixture } from './helpers'

test('历史页：列表展示 → 删除记录 → CSV 导出链接', async ({ page, request }) => {
  // 通过 API 播种一条带备注的记录
  const response = await request.post('/api/v1/inpaint', {
    multipart: {
      image: {
        name: 'image_a.png',
        mimeType: 'image/png',
        buffer: readFileSync(fixture('image_a.png')),
      },
      mask: {
        name: 'mask_a.png',
        mimeType: 'image/png',
        buffer: readFileSync(fixture('mask_a.png')),
      },
      note: 'e2e-history',
    },
  })
  expect(response.ok()).toBeTruthy()

  await page.goto('/history')
  // e2e 库跨运行累积，同备注可能多条：取第一条
  const row = page.locator('tr', { hasText: 'e2e-history' }).first()
  await expect(row).toBeVisible()
  // 用唯一 job 的输出链接标识该行（e2e 库跨运行累积同备注记录）
  const outputHref = await row.locator('a[href*="/outputs/"]').first().getAttribute('href')
  expect(outputHref).toBeTruthy()

  // 删除记录（popconfirm 确认；el-link 无 href 不是 link role，按文本定位）
  await row.getByText('删除', { exact: true }).click()
  await page.getByRole('button', { name: '删除', exact: true }).click()
  await expect(page.locator(`a[href="${outputHref}"]`)).toHaveCount(0)

  await expect(page.getByRole('link', { name: '导出 CSV' })).toHaveAttribute(
    'href',
    /history\/export$/,
  )
})
