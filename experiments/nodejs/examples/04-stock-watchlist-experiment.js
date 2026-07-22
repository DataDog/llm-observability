'use strict'

const { assert, assertUrl, flushAndWait, initTracer, uniqueName } = require('./lib/env')
const { callOpenAIJson } = require('./lib/openai')

function uniqueValues (items) {
  return Array.from(new Set(items))
}

async function callStockJson (llmobs, name, config, messages) {
  return callOpenAIJson(llmobs, {
    name,
    model: config.model,
    temperature: config.temperature,
    messages,
  })
}

async function fetchStockQuote (llmobs, ticker, config) {
  return callStockJson(llmobs, 'openai.stock_watchlist.quote', config, [
    {
      role: 'system',
      content: 'You are a market data assistant. Respond only as JSON with shape {"price":"string","change":"string"}.',
    },
    {
      role: 'user',
      content: `Provide a concise recent price snapshot for ${ticker}. Use approximate public knowledge if needed.`,
    },
  ])
}

async function fetchStockNews (llmobs, ticker, config) {
  return callStockJson(llmobs, 'openai.stock_watchlist.news', config, [
    {
      role: 'system',
      content: [
        'You are an equity news analyst.',
        'Respond only as JSON with shape {"headline":"string","theme":"string"}.',
      ].join(' '),
    },
    {
      role: 'user',
      content: `Summarize one recent business theme or news driver for ${ticker}.`,
    },
  ])
}

async function fetchStockSentiment (llmobs, ticker, config) {
  return callStockJson(llmobs, 'openai.stock_watchlist.sentiment', config, [
    {
      role: 'system',
      content: [
        'You are an investor sentiment classifier.',
        'Respond only as JSON with shape {"sentiment":"bullish|neutral|bearish","reason":"string"}.',
      ].join(' '),
    },
    {
      role: 'user',
      content: `Classify current investor sentiment for ${ticker}.`,
    },
  ])
}

async function synthesizeStock (llmobs, ticker, quote, news, sentiment, config) {
  return callStockJson(llmobs, 'openai.stock_watchlist.ticker_synthesis', config, [
    {
      role: 'system',
      content: [
        'You are a portfolio analyst.',
        'Respond only as JSON with shape',
        '{"ticker":"string","recommendation":"watch|buy|hold|avoid",',
        '"rationale":"string","riskLevel":"low|medium|high"}.',
      ].join(' '),
    },
    {
      role: 'user',
      content: JSON.stringify({ ticker, quote, news, sentiment }),
    },
  ])
}

async function synthesizePortfolio (llmobs, analyses, config) {
  return callStockJson(llmobs, 'openai.stock_watchlist.portfolio_synthesis', config, [
    {
      role: 'system',
      content: [
        'You are a senior portfolio analyst.',
        'Respond only as JSON with shape',
        '{"summary":"string","topWatch":"string","riskThemes":["string"]}.',
      ].join(' '),
    },
    {
      role: 'user',
      content: JSON.stringify({ analyses }),
    },
  ])
}

function createStockWatchlistTask (llmobs) {
  return async function stock_watchlist_agent (inputData, config, metadata) {
    return llmobs.trace({
      kind: 'workflow',
      name: 'stock_watchlist_workflow',
      tags: { example: 'stock-watchlist', portfolio: metadata.portfolio },
    }, async (workflowSpan) => {
      const tickers = uniqueValues(inputData.tickers.map(ticker => String(ticker).toUpperCase()))
      llmobs.annotate(workflowSpan, {
        inputData,
        metadata: { portfolio: metadata.portfolio, ticker_count: tickers.length },
      })

      const analyses = await Promise.all(tickers.map(ticker => llmobs.trace({
        kind: 'agent',
        name: 'stock_researcher',
        tags: { example: 'stock-watchlist', ticker },
      }, async (researchSpan) => {
        llmobs.annotate(researchSpan, { inputData: { ticker }, metadata: { ticker } })

        const [quote, news, sentiment] = await Promise.all([
          fetchStockQuote(llmobs, ticker, config),
          fetchStockNews(llmobs, ticker, config),
          fetchStockSentiment(llmobs, ticker, config),
        ])
        const analysis = await synthesizeStock(llmobs, ticker, quote, news, sentiment, config)
        const normalized = {
          ticker,
          recommendation: String(analysis.recommendation || 'watch').toLowerCase(),
          rationale: String(analysis.rationale || '').trim(),
          riskLevel: String(analysis.riskLevel || 'medium').toLowerCase(),
          quote,
          news,
          sentiment,
        }

        llmobs.annotate(researchSpan, { outputData: normalized })
        return normalized
      })))

      const portfolio = await synthesizePortfolio(llmobs, analyses, config)
      const output = {
        analyses,
        summary: String(portfolio.summary || '').trim(),
        topWatch: String(portfolio.topWatch || analyses[0]?.ticker || '').toUpperCase(),
        riskThemes: Array.isArray(portfolio.riskThemes) ? portfolio.riskThemes : [],
      }

      llmobs.annotate(workflowSpan, { outputData: output })
      return output
    })
  }
}

function covers_all_tickers (inputData, outputData) {
  const expected = uniqueValues(inputData.tickers.map(ticker => String(ticker).toUpperCase())).sort()
  const actual = outputData.analyses.map(analysis => analysis.ticker).sort()
  return expected.length === actual.length && expected.every((ticker, index) => ticker === actual[index])
}

function has_recommendations (_inputData, outputData) {
  const allowed = new Set(['watch', 'buy', 'hold', 'avoid'])
  return outputData.analyses.every(analysis => allowed.has(analysis.recommendation))
}

function has_multiple_provider_calls (_inputData, outputData) {
  // The task calls quote/news/sentiment/synthesis per ticker, plus portfolio synthesis.
  // This checks the output reflects the multi-call research shape.
  return outputData.analyses.every(analysis => analysis.quote && analysis.news && analysis.sentiment)
}

function coverage_summary (_inputs, _outputs, _expectedOutputs, evaluatorResults) {
  const values = evaluatorResults.covers_all_tickers || []
  return values.filter(Boolean).length / values.length
}

async function main () {
  const tracer = initTracer()
  const dataset = tracer.llmobs.createDataset(uniqueName('nodejs-stock-watchlist'), {
    description: 'Node.js stock watchlist experiment dataset with multiple OpenAI calls per row',
    records: [
      {
        inputData: { tickers: ['AAPL', 'MSFT'] },
        expectedOutput: { tickers: ['AAPL', 'MSFT'] },
        metadata: { portfolio: 'large-cap-ai' },
      },
    ],
  })

  const result = await tracer.llmobs.experiment({
    name: uniqueName('nodejs-stock-watchlist-exp'),
    dataset,
    task: createStockWatchlistTask(tracer.llmobs),
    evaluators: [covers_all_tickers, has_recommendations, has_multiple_provider_calls],
    summaryEvaluators: [coverage_summary],
    config: {
      model: process.env.OPENAI_MODEL || 'gpt-4o-mini',
      temperature: 0,
      provider: 'openai',
    },
    tags: { sdk: 'nodejs', example: 'stock-watchlist', provider: 'openai' },
  }).run({ maxRetries: 1, retryDelay: () => 0, raiseErrors: true })

  assert.equal(result.rows.length, 1)
  assert.equal(result.rows[0].evaluations.covers_all_tickers, true)
  assert.equal(result.rows[0].evaluations.has_recommendations, true)
  assert.equal(result.rows[0].evaluations.has_multiple_provider_calls, true)
  assert.equal(result.summaryEvaluations.coverage_summary.value, 1)
  assertUrl(result.url, 'result.url')

  await flushAndWait(tracer)

  console.log('Stock watchlist experiment validation passed')
  console.log(`Dataset URL   : ${dataset.url()}`)
  console.log(`Experiment URL: ${result.url}`)
  console.log(`Experiment ID : ${result.experimentId}`)
  console.log('Each row should include multiple nested OpenAI LLM spans:')
  console.log([
    'experiment row → stock_watchlist_workflow → stock_researcher →',
    'quote/news/sentiment/ticker_synthesis + portfolio_synthesis',
  ].join(' '))
  for (const row of result.rows) {
    console.log(`Row ${row.index} span=${row.spanId} trace=${row.traceId}`)
  }
}

main().catch((err) => {
  console.error(err)
  process.exitCode = 1
})
