import { describe, it, expect } from 'vitest'
import { statusLabel, humanStatus, shortTime } from '../lib/labels'

describe('statusLabel (answer_status -> Chinese)', () => {
  it('maps known answer statuses', () => {
    expect(statusLabel('answered')).toBe('已回答')
    expect(statusLabel('refused')).toBe('已拒绝强行预测')
    expect(statusLabel('needs_model')).toBe('需要选择对话模型')
    expect(statusLabel('general_assisted')).toBe('通用知识回答（非人物预测）')
    expect(statusLabel('direct_answer')).toBe('历史直接依据')
  })

  it('falls back for unknown / empty values', () => {
    expect(statusLabel('bogus')).toBe('未知状态')
    expect(statusLabel('')).toBe('未记录')
    expect(statusLabel(undefined)).toBe('未记录')
    expect(statusLabel(null)).toBe('未记录')
  })
})

describe('humanStatus', () => {
  it('maps known human-facing statuses', () => {
    expect(humanStatus('confirmed')).toBe('已确认')
    expect(humanStatus('pending')).toBe('待审核')
    expect(humanStatus('rejected')).toBe('已拒绝')
    expect(humanStatus('accepted_exploratory')).toBe('探索性版本已建立')
  })

  it('falls back for unknown / empty values', () => {
    expect(humanStatus('nope')).toBe('未知状态')
    expect(humanStatus('')).toBe('未记录')
  })
})

describe('shortTime', () => {
  it('returns empty string for missing input', () => {
    expect(shortTime('')).toBe('')
    expect(shortTime(null)).toBe('')
    expect(shortTime(undefined)).toBe('')
  })
})
