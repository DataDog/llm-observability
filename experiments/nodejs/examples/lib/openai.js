'use strict'

const { requireEnv } = require('./env')

const JSON_RESPONSE_FORMAT = {
  type: 'json_schema',
  json_schema: {
    name: 'json_response',
    schema: {
      type: 'object',
      additionalProperties: true,
    },
  },
}

let openaiClient

function client () {
  if (openaiClient) return openaiClient
  const OpenAI = require('openai')
  const OpenAIClient = OpenAI.default || OpenAI
  openaiClient = new OpenAIClient({ apiKey: requireEnv('OPENAI_API_KEY') })
  return openaiClient
}

async function callOpenAIJson (llmobs, options) {
  const model = options.model || process.env.OPENAI_MODEL || 'gpt-5-mini'
  const temperature = options.temperature ?? 0

  return llmobs.trace({
    kind: 'llm',
    name: options.name || 'openai.chat.completions',
    modelName: model,
    modelProvider: 'openai',
  }, async (span) => {
    const response = await client().chat.completions.parse({
      model,
      temperature,
      messages: options.messages,
      response_format: JSON_RESPONSE_FORMAT,
    })

    const message = response.choices?.[0]?.message
    const content = message?.content ?? ''
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

    if (message?.parsed === null || typeof message?.parsed !== 'object') {
      throw new Error(`OpenAI response was not parsed JSON: ${content}`)
    }
    return message.parsed
  })
}

module.exports = { callOpenAIJson }
