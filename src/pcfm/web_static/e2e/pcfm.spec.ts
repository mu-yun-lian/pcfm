import { test, expect } from '@playwright/test'

const name = 'E2E测试人物' + Date.now().toString().slice(-6)

test('PCFM 完整主流程: 创建人物 → 加资料 → 对话 → 现实对照 → 归档', async ({ page }) => {
  // 1. 创建人物
  await page.goto('/')
  await expect(page.locator('#app')).toBeVisible()
  await page.getByRole('button', { name: '创建第一个人物' }).click()
  await page.getByLabel('人物名称（必填）').fill(name)
  await page.getByRole('button', { name: '保存' }).click()
  await expect(page.locator('.chat-workspace')).toBeVisible()

  // 2. 加资料(粘贴文本)
  await page.getByLabel('更多操作', { exact: true }).click()
  await page.getByRole('button', { name: '人物资料' }).click()
  const pasteForm = page.locator('form.source-form').first()
  await pasteForm.getByLabel('标题', { exact: true }).fill('访谈')
  await pasteForm.getByLabel('说话人', { exact: true }).fill(name)
  await pasteForm.getByLabel('原始内容', { exact: true }).fill('Q: 你如何看待产品发布？\nA: 证据足够时才发布。')
  await pasteForm.getByRole('button', { name: '保存为待审核资料' }).click()
  await expect(page.locator('.source-item').first()).toBeVisible({ timeout: 15000 })
  await page.locator('dialog[open] .close-button').first().click()

  // 3. 对话(诚实标注: 助手消息渲染)
  await page.locator('.composer textarea').fill('你如何看待产品发布？')
  await page.getByRole('button', { name: '发送' }).click()
  await expect(page.locator('.message-row.user').first()).toBeVisible()
  await expect(page.locator('.message-row.assistant').first()).toBeVisible({ timeout: 30_000 })

  // 4. 现实对照(按钮打开抽屉或返回候选)
  await page.getByRole('button', { name: '现实回答' }).first().click()

  // 5. 归档(清理)
  const card = page.locator('.person-card', { hasText: name }).first()
  await card.locator('.person-more').click()
  await page.getByRole('button', { name: '移入归档' }).click()
  await page.getByRole('button', { name: '确认移入归档' }).click()
  await expect(page.locator('.toast')).toContainText('已移入归档', { timeout: 15000 })
  await expect(page.locator('.people-list .person-card', { hasText: name })).toHaveCount(0)
})
