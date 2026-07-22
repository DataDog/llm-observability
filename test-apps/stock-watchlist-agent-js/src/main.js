#!/usr/bin/env node
'use strict'

const { flush, isLLMObsEnabled, mlApp, site, env } = require('./observability')
const { analyzePortfolio } = require('./agents/orchestrator')
const { runEvaluations } = require('./evals')

function printBriefing (briefing) {
  console.log('\n' + '='.repeat(60))
  console.log('  STOCK WATCHLIST BRIEFING')
  console.log('='.repeat(60))
  console.log(`\nGenerated: ${briefing.generated_at}`)
  console.log(`\n${'─'.repeat(60)}`)
  console.log('MARKET OVERVIEW')
  console.log('─'.repeat(60))
  console.log(briefing.market_overview)

  for (const analysis of briefing.analyses) {
    console.log(`\n${'─'.repeat(60)}`)
    console.log(`  ${analysis.ticker} (${analysis.company_name})`)
    console.log(`  ${analysis.current_price}  (${analysis.price_change})`)
    console.log(`  Sentiment: ${analysis.sentiment.toUpperCase()}`)
    console.log('─'.repeat(60))
    console.log(`\n${analysis.summary}\n`)
    console.log('Key Factors:')
    for (const factor of analysis.key_factors) {
      console.log(`  * ${factor}`)
    }
    console.log('\nRecent News:')
    for (const news of analysis.recent_news) {
      console.log(`  - ${news}`)
    }
    console.log('\nPublic Sentiment:')
    console.log(`  ${analysis.public_sentiment_summary}`)
  }

  console.log(`\n${'─'.repeat(60)}`)
  console.log('HIGHLIGHTS')
  console.log('─'.repeat(60))
  for (const highlight of briefing.highlights) {
    console.log(`  >> ${highlight}`)
  }

  console.log('\n' + '='.repeat(60) + '\n')
}

function parseArgs (argv) {
  const args = argv.slice(2)
  if (args.includes('-h') || args.includes('--help')) {
    console.log('Usage: node src/main.js <TICKER> [TICKER ...]')
    console.log('Example: node src/main.js AAPL GOOGL NVDA')
    process.exit(0)
  }
  return args.map(ticker => ticker.toUpperCase())
}

async function main (tickers) {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY is required')
  }

  console.log(`Analyzing ${tickers.length} ticker(s): ${tickers.join(', ')}`)
  if (isLLMObsEnabled()) {
    console.log(`LLMObs enabled: ml_app=${mlApp}, site=${site}, env=${env}`)
  } else {
    console.log('LLMObs disabled: set DD_API_KEY (or DD_LLMOBS_ENABLED=true) to submit traces')
  }
  console.log('Running parallel analysis with web search...\n')

  const { briefing, spanContext } = await analyzePortfolio(tickers)
  if (isLLMObsEnabled() && spanContext) {
    console.log(`LLMObs trace context: trace_id=${spanContext.traceId}, span_id=${spanContext.spanId}`)
  }
  printBriefing(briefing)

  if (isLLMObsEnabled() && spanContext) {
    console.log('Running evaluations...')
    await runEvaluations(briefing, tickers, spanContext)
    console.log('Evaluations submitted to LLM Observability.\n')
  }
}

async function cli () {
  const tickers = parseArgs(process.argv)
  if (tickers.length === 0) {
    console.error('Error: provide at least one ticker symbol')
    console.error('Usage: node src/main.js <TICKER> [TICKER ...]')
    process.exitCode = 1
    return
  }

  try {
    await main(tickers)
  } finally {
    flush()
  }
}

if (require.main === module) {
  cli().catch(err => {
    console.error(err.stack || err.message)
    process.exitCode = 1
  })
}

module.exports = {
  main,
  printBriefing,
}
