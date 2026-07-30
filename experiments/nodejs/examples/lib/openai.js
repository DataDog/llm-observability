'use strict'

const OpenAI = require('openai')

const { requireEnv } = require('./env')

let openaiClient

function client () {
  if (openaiClient) return openaiClient
  const OpenAIClient = OpenAI.default || OpenAI
  openaiClient = new OpenAIClient({ apiKey: requireEnv('OPENAI_API_KEY') })
  return openaiClient
}

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
  const model = options.model || process.env.OPENAI_MODEL || 'gpt-5-mini'
  const temperature = options.temperature ?? 0

  return llmobs.trace({
    kind: 'llm',
    name: options.name || 'openai.chat.completions',
    modelName: model,
    modelProvider: 'openai',
  }, async (span) => {
    const response = await client().chat.completions.create({
      model,
      temperature,
      messages: options.messages,
      response_format: { type: 'json_object' },
    })

    const content = response.choices?.[0]?.message?.content ?? ''
    const metrics = {}
    if (typeof response.usage?.prompt_tokens === 'number') metrics.inputTokens = response.usage.prompt_tokens
    if (typeof response.usage?.completion_tokens === 'number') metrics.outputTokens = response.usage.completion_tokens
    if (typeof response.usage?.total_tokens === 'number') metrics.totalTokens = response.usage.total_tokens
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
