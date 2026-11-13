import { expect, test } from '@playwright/test'
import { fixture } from './helpers'

test('批量全流程：配对上传 → SSE 进度 → report 下载', async ({ page }) => {
  await page.goto('/batch')
  const uploads = page.locator('.el-upload input[type=file]')
  await uploads.nth(0).setInputFiles([fixture('image_a.png'), fixture('image_b.png')])
  await uploads.nth(1).setInputFiles([fixture('mask_a.png'), fixture('mask_b.png')])

  await page.getByRole('button', { name: '提交批量' }).click()
  await expect(page.getByText('全部成功')).toBeVisible({ timeout: 60_000 })
  await expect(page.getByRole('link', { name: '下载 report.csv' })).toBeVisible()
  await expect(page.locator('.el-tag', { hasText: '成功' }).first()).toBeVisible()
})

test('批量取消：worker 完成当前项后停止，未执行项落 cancelled', async ({ page }) => {
  await page.goto('/batch')
  const uploads = page.locator('.el-upload input[type=file]')
  await uploads
    .nth(0)
    .setInputFiles([
      fixture('image_a.png'),
      fixture('image_b.png'),
      fixture('image_a.png'),
      fixture('image_b.png'),
    ])
  await uploads
    .nth(1)
    .setInputFiles([
      fixture('mask_a.png'),
      fixture('mask_b.png'),
      fixture('mask_a.png'),
      fixture('mask_b.png'),
    ])

  await page.getByRole('button', { name: '提交批量' }).click()
  const cancelButton = page.getByRole('button', { name: '取消' })
  await expect(cancelButton).toBeVisible()
  await cancelButton.click()

  // 4 项 × 0.8s 延迟：取消后剩余项应落 cancelled，终态为已取消
  await expect(page.getByText('已取消', { exact: false }).first()).toBeVisible({
    timeout: 60_000,
  })
  await expect(page.getByRole('link', { name: '下载 report.csv' })).toBeVisible()
})
