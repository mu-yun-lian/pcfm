import { test, expect } from '@playwright/test'

test('PCFM 主流程: 页面加载 → 助手对话 → 用户消息与助手回复渲染', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('#app')).toBeVisible()
  await page.getByRole('button', { name: '创建第一个人物' }).click()
  await expect(page.locator('.chat-workspace')).toBeVisible()
  await page.locator('.composer textarea').fill('你好')
  await page.getByRole('button', { name: '发送' }).click()
  await expect(page.locator('.message-row.user').first()).toBeVisible()
  await expect(page.locator('.message-row.assistant').first()).toBeVisible({ timeout: 30_000 })
})
