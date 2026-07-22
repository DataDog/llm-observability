'use strict'

const stockAnalysisSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    ticker: { type: 'string' },
    company_name: { type: 'string' },
    current_price: { type: 'string' },
    price_change: { type: 'string' },
    recent_news: { type: 'array', items: { type: 'string' } },
    sentiment: { type: 'string', enum: ['bullish', 'bearish', 'neutral'] },
    public_sentiment_summary: { type: 'string' },
    key_factors: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
  required: [
    'ticker',
    'company_name',
    'current_price',
    'price_change',
    'recent_news',
    'sentiment',
    'public_sentiment_summary',
    'key_factors',
    'summary',
  ],
}

const researchBatchResultSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    analyses: { type: 'array', items: stockAnalysisSchema },
  },
  required: ['analyses'],
}

const portfolioBriefingSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    analyses: { type: 'array', items: stockAnalysisSchema },
    market_overview: { type: 'string' },
    highlights: { type: 'array', items: { type: 'string' } },
    generated_at: { type: 'string' },
  },
  required: ['analyses', 'market_overview', 'highlights', 'generated_at'],
}

function assertString (value, path) {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`${path} must be a non-empty string`)
  }
}

function assertStringArray (value, path) {
  if (!Array.isArray(value) || value.some(item => typeof item !== 'string')) {
    throw new Error(`${path} must be an array of strings`)
  }
}

function validateStockAnalysis (analysis, index = 0) {
  const path = `analyses[${index}]`
  if (!analysis || typeof analysis !== 'object' || Array.isArray(analysis)) {
    throw new Error(`${path} must be an object`)
  }
  assertString(analysis.ticker, `${path}.ticker`)
  assertString(analysis.company_name, `${path}.company_name`)
  assertString(analysis.current_price, `${path}.current_price`)
  assertString(analysis.price_change, `${path}.price_change`)
  assertStringArray(analysis.recent_news, `${path}.recent_news`)
  if (!['bullish', 'bearish', 'neutral'].includes(analysis.sentiment)) {
    throw new Error(`${path}.sentiment must be bullish, bearish, or neutral`)
  }
  assertString(analysis.public_sentiment_summary, `${path}.public_sentiment_summary`)
  assertStringArray(analysis.key_factors, `${path}.key_factors`)
  assertString(analysis.summary, `${path}.summary`)
  return analysis
}

function validateResearchBatchResult (result) {
  if (!result || typeof result !== 'object' || !Array.isArray(result.analyses)) {
    throw new Error('ResearchBatchResult must contain an analyses array')
  }
  result.analyses.forEach(validateStockAnalysis)
  return result
}

function validatePortfolioBriefing (briefing) {
  if (!briefing || typeof briefing !== 'object' || !Array.isArray(briefing.analyses)) {
    throw new Error('PortfolioBriefing must contain an analyses array')
  }
  briefing.analyses.forEach(validateStockAnalysis)
  assertString(briefing.market_overview, 'market_overview')
  assertStringArray(briefing.highlights, 'highlights')
  assertString(briefing.generated_at, 'generated_at')
  return briefing
}

module.exports = {
  stockAnalysisSchema,
  researchBatchResultSchema,
  portfolioBriefingSchema,
  validateResearchBatchResult,
  validatePortfolioBriefing,
}
