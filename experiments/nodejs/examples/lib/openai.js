'use strict'

const { requireEnv } = require('./env')

function parseJsonObject (content) {
  const trimmed = content.trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '')
  try {
    return JSON.parse(trimmed)
  } catch {
    const start = trimmed.indexOf('{')
    const end = trimmed.lastIndexOf('}')
    if (start !== -1 && end !== -1 && end > start) {
      return JSON.parse(trimmed.slice(start, end + 1))
    }
    throw new Error(`OpenAI response was not JSON: ${content}`)
  }
}

async function callOpenAIChat (llmobs, options) {
  const apiKey = requireEnv('OPENAI_API_KEY')
  const model = options.model || process.env.OPENAI_MODEL || 'gpt-4o-mini'
  const temperature = options.temperature ?? 0

  return llmobs.trace({
    kind: 'llm',
    name: options.name || 'openai.chat.completions',
    modelName: model,
    modelProvider: 'openai',
  }, async (span) => {
    let body
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        temperature,
        messages: options.messages,
        response_format: { type: 'json_object' },
      }),
    })

    const text = await response.text()
    try {
      body = text ? JSON.parse(text) : {}
    } catch {
      body = { raw: text }
    }

    if (!response.ok) {
      throw new Error(`OpenAI chat completion failed: HTTP ${response.status} ${text}`)
    }

    const content = body?.choices?.[0]?.message?.content ?? ''
    const metrics = {}
    if (typeof body?.usage?.prompt_tokens === 'number') metrics.inputTokens = body.usage.prompt_tokens
    if (typeof body?.usage?.completion_tokens === 'number') metrics.outputTokens = body.usage.completion_tokens
    if (typeof body?.usage?.total_tokens === 'number') metrics.totalTokens = body.usage.total_tokens
    llmobs.annotate(span, {
      inputData: options.messages,
      outputData: { role: 'assistant', content },
      metadata: { temperature },
      metrics,
    })
    return content
  })
}

async function callOpenAIJson (llmobs, options) {
  return parseJsonObject(await callOpenAIChat(llmobs, options))
}

module.exports = { callOpenAIChat, callOpenAIJson, parseJsonObject }
