'use strict'

require('./observability')
const OpenAI = require('openai')
const { jsonSchemaFormat } = require('./agents/responses-agent')
const { annotate, submitEvaluation, traceSpan } = require('./observability')

const client = new OpenAI()

const booleanJudgeSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    value: { type: 'boolean' },
    reasoning: { type: 'string' },
  },
  required: ['value', 'reasoning'],
}

const scoreJudgeSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    value: { type: 'number' },
    reasoning: { type: 'string' },
  },
  required: ['value', 'reasoning'],
}

function parseJsonOutput (response) {
  const text = response.output_text || ''
  return JSON.parse(text.trim())
}

function completenessEvaluation (briefing, requestedTickers) {
  const resultTickers = new Set((briefing.analyses || []).map(analysis => analysis.ticker.toUpperCase()))
  const requested = new Set(requestedTickers.map(ticker => ticker.toUpperCase()))
  const missing = [...requested].filter(ticker => !resultTickers.has(ticker)).sort()
  const passed = missing.length === 0

  return {
    value: passed,
    assessment: passed ? 'pass' : 'fail',
    reasoning: passed
      ? 'All requested tickers present in output.'
      : `Missing tickers: ${missing.join(', ')}`,
  }
}

async function runBooleanJudge ({ name, prompt, outputData }) {
  return traceSpan({ kind: 'task', name }, async span => {
    annotate(span, { inputData: { output_data: outputData } })
    const response = await client.responses.create({
      model: process.env.OPENAI_EVAL_MODEL || 'gpt-4o-mini',
      input: prompt.replace('{{output_data}}', JSON.stringify(outputData, null, 2)),
      text: { format: jsonSchemaFormat(name, booleanJudgeSchema) },
    })
    const result = parseJsonOutput(response)
    annotate(span, { outputData: result })
    return {
      value: Boolean(result.value),
      assessment: result.value ? 'pass' : 'fail',
      reasoning: result.reasoning || '',
    }
  })
}

async function runScoreJudge ({ name, prompt, outputData, minThreshold }) {
  return traceSpan({ kind: 'task', name }, async span => {
    annotate(span, { inputData: { output_data: outputData } })
    const response = await client.responses.create({
      model: process.env.OPENAI_EVAL_MODEL || 'gpt-4o-mini',
      input: prompt.replace('{{output_data}}', JSON.stringify(outputData, null, 2)),
      text: { format: jsonSchemaFormat(name, scoreJudgeSchema) },
    })
    const result = parseJsonOutput(response)
    const value = Math.max(1, Math.min(5, Number(result.value)))
    annotate(span, { outputData: { value, reasoning: result.reasoning || '' } })
    return {
      value,
      assessment: value >= minThreshold ? 'pass' : 'fail',
      reasoning: result.reasoning || '',
    }
  })
}

const sentimentPrompt = `\
Review these stock analyses. For each stock, check whether the sentiment label \
(bullish/bearish/neutral) is consistent with the summary and key factors.

Rules:
- 'neutral' is valid when there are mixed positive and negative signals
- Only flag clear contradictions (e.g., 'bullish' but summary describes major losses)
- When in doubt, consider it consistent

Return JSON with {"value": boolean, "reasoning": string}.

Analyses:
{{output_data}}`

const groundingPrompt = `\
Rate the factual grounding of these stock analyses.

1 = Entirely vague, no specific facts cited
2 = Mostly vague with occasional specifics
3 = Mix of vague and specific claims
4 = Mostly grounded with concrete numbers, dates, and events
5 = Thoroughly grounded with specific revenue figures, dates, named events throughout

Return JSON with {"value": number, "reasoning": string}.

Analyses:
{{output_data}}`

async function runEvaluations (briefing, requestedTickers, spanContext) {
  const outputData = JSON.parse(JSON.stringify(briefing))

  const completeness = completenessEvaluation(briefing, requestedTickers)
  submitEvaluation(spanContext, {
    label: 'completeness',
    metricType: 'boolean',
    value: completeness.value,
    assessment: completeness.assessment,
    reasoning: completeness.reasoning,
  })

  const sentiment = await runBooleanJudge({
    name: 'sentiment_consistency',
    prompt: sentimentPrompt,
    outputData,
  })
  submitEvaluation(spanContext, {
    label: 'sentiment_consistency',
    metricType: 'boolean',
    value: sentiment.value,
    assessment: sentiment.assessment,
    reasoning: sentiment.reasoning,
  })

  const grounding = await runScoreJudge({
    name: 'factual_grounding',
    prompt: groundingPrompt,
    outputData,
    minThreshold: 3,
  })
  submitEvaluation(spanContext, {
    label: 'factual_grounding',
    metricType: 'score',
    value: grounding.value,
    assessment: grounding.assessment,
    reasoning: grounding.reasoning,
  })
}

module.exports = {
  completenessEvaluation,
  runEvaluations,
}
