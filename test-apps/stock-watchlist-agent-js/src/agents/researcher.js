'use strict'

const { search } = require('./searcher')
const { runResponsesAgent } = require('./responses-agent')
const { researchBatchResultSchema, validateResearchBatchResult } = require('../models')
const { annotate, traceSpan } = require('../observability')

const stringArg = (name, description) => ({
  type: 'object',
  additionalProperties: false,
  properties: {
    [name]: { type: 'string', description },
  },
  required: [name],
})

const researcherTools = [
  {
    name: 'get_stock_quote',
    description: 'Get the current stock price, daily price change, and key trading data for a ticker symbol.',
    parameters: stringArg('ticker', 'Stock ticker symbol, e.g. AAPL'),
    execute: ({ ticker }) => search(`${ticker} stock price today current quote market data`),
  },
  {
    name: 'search_company_news',
    description: [
      'Search for recent company news and developments.',
      "You craft the search query — be specific about what you're looking for",
      "(e.g. 'NVIDIA Q4 2026 earnings results' or 'Apple AI features Siri delay').",
    ].join(' '),
    parameters: stringArg('query', 'Specific company-news search query'),
    execute: ({ query }) => search(query),
  },
  {
    name: 'search_public_sentiment',
    description: [
      'Search for public investor sentiment and discussions about a stock or company.',
      'Look for Reddit, social media, and forum discussions to gauge retail investor mood.',
    ].join(' '),
    parameters: stringArg('query', 'Specific public sentiment search query'),
    execute: ({ query }) => search(`${query} investor sentiment Reddit discussion forum opinions`),
  },
  {
    name: 'get_company_profile',
    description: 'Get a company overview including sector, market cap, business description, recent performance, and key financial metrics.',
    parameters: stringArg('ticker', 'Stock ticker symbol, e.g. NVDA'),
    execute: ({ ticker }) => search(`${ticker} company profile overview market cap sector fundamentals key metrics 2026`),
  },
]

const RESEARCHER_PROMPT = `\
You are a stock research analyst conducting thorough research on your assigned batch of stocks.

<scope>
You research stock tickers using available tools, conducting multi-step analysis \
for each stock in your assigned batch. You must produce a StockAnalysis for every \
ticker you are assigned.
</scope>

<approach>
Think step-by-step before taking action:
1. Plan your research approach for the assigned tickers
2. Get price quotes to establish current market positions
3. Search for recent news with specific, targeted queries — craft them carefully
4. Check public investor sentiment on Reddit, forums, social media
5. Get company profiles for fundamental context
6. After each round, reflect: did something interesting surface? Do you need to dig deeper?
7. When satisfied, produce a structured StockAnalysis for each ticker
</approach>

<tools>
get_stock_quote — Current price and daily change for a ticker.
search_company_news — Search recent news. You craft the query — be specific.
search_public_sentiment — Gauge retail investor mood from Reddit/forums.
get_company_profile — Company overview, sector, market cap, fundamentals.
</tools>

<examples>
Assigned: ["AAPL", "NVDA"]

Thought: Two tickers. I'll start by getting current prices for both, then research news for each.
Action: [calls get_stock_quote("AAPL")]
Action: [calls get_stock_quote("NVDA")]

Result: AAPL at $262.01 (-1.66%), NVDA at $197.58 (-0.65%)

Thought: Both are slightly down. Let me search for what's driving each.
Action: [calls search_company_news("Apple Q2 2026 earnings preview iPhone")]
Action: [calls search_company_news("NVIDIA AI data center revenue 2026")]

Result: Apple earnings coming April 30, NVIDIA reporting record AI revenue.

Thought: Interesting — Apple has earnings soon. Let me check sentiment and get NVIDIA's profile.
Action: [calls search_public_sentiment("AAPL Apple earnings April 2026")]
Action: [calls get_company_profile("NVDA")]

Result: [sentiment and profile data]

Thought: I have enough to produce analyses for both stocks.
[Returns ResearchBatchResult with analyses for AAPL and NVDA]
</examples>

<output>
Be specific — cite numbers, dates, and concrete facts. Avoid vague generalities.
Every StockAnalysis must have all fields populated with real data from your research.
Return JSON only matching the ResearchBatchResult schema.
</output>`

async function researchStocks (tickers) {
  return traceSpan({ kind: 'agent', name: 'stock_researcher' }, async span => {
    const prompt = [
      `Research these stocks: ${tickers.join(', ')}.`,
      'Get current prices, search for recent news and developments,',
      'check public investor sentiment, and gather company fundamentals.',
      'Dig deeper if something interesting surfaces.',
      'Produce a StockAnalysis for each ticker.',
    ].join(' ')

    annotate(span, { inputData: prompt, metadata: { tickers } })
    const result = await runResponsesAgent({
      name: 'stock_researcher',
      systemPrompt: RESEARCHER_PROMPT,
      userPrompt: prompt,
      outputSchema: researchBatchResultSchema,
      outputSchemaName: 'ResearchBatchResult',
      tools: researcherTools,
    })
    const validated = validateResearchBatchResult(result)
    annotate(span, { outputData: validated })
    return validated
  })
}

module.exports = {
  researchStocks,
  RESEARCHER_PROMPT,
}
