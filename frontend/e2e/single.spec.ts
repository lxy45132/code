import { expect, test } from '@playwright/test'
import { fixture } from './helpers'

test('单图修复全流程：上传 → 涂掩码 → 修复 → 结果对比', async ({ page }) => {
  await page.goto('/inpaint')
  await page.locator('.el-upload input[type=file]').setInputFiles(fixture('image_a.png'))

  const canvas = page.locator('canvas.overlay')
  await expect(canvas).toBeVisible()
  const box = await canvas.boundingBox()
  expect(box).not.toBeNull()
  if (!box) return

  // 在图像中心涂出一个洞
  await page.mouse.move(box.x + box.width * 0.4, box.y + box.height * 0.4)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.6, { steps: 8 })
  await page.mouse.up()

  await page.getByRole('button', { name: '开始修复' }).click()
  await expect(page.getByText('修复结果')).toBeVisible({ timeout: 30_000 })
  await expect(page.getByText('模型版本')).toBeVisible()
  await expect(page.getByRole('link', { name: '下载结果 PNG' })).toBeVisible()
})
