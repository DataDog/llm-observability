#!/usr/bin/env node
'use strict'

const { flush, isLLMObsEnabled, mlApp, site, env } = require('./observability')
const { analyzePortfolio } = require('./agents/orchestrator')
const { looksLikeImage } = require('./agents/vision')
const { runEvaluations } = require('./evals')

const USAGE = [
  'Usage: node src/main.js <TICKER|IMAGE> [TICKER|IMAGE ...]',
  '',
  'Inputs may be ticker symbols or local image files (mixed freely).',
  'Images are translated to ticker symbols by an LLM before research begins.',
  'Use --image <path> to force an argument to be treated as an image.',
  '',
  'Examples:',
  '  node src/main.js AAPL GOOGL NVDA',
  '  node src/main.js logos/apple.png logos/google.png NVDA',
  '  node src/main.js AAPL --image logos/nvidia.png',
].join('\n')

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

function printIdentifications (identifications) {
  console.log('\nImage inputs resolved to tickers:')
  for (const item of identifications) {
    const label = item.ticker === 'UNKNOWN' ? 'UNKNOWN (skipped)' : item.ticker
    console.log(`  ${item.source} -> ${label} (${item.company_name}, confidence: ${item.confidence})`)
    console.log(`    ${item.evidence}`)
  }
}

function parseArgs (argv) {
  const args = argv.slice(2)
  if (args.includes('-h') || args.includes('--help')) {
    console.log(USAGE)
    process.exit(0)
  }

  const tickers = []
  const images = []
  for (let i = 0; i < args.length; i++) {
    const arg = args[i]
    if (arg === '--image') {
      const value = args[++i]
      if (!value) {
        throw new Error('--image requires a file path')
      }
      images.push(value)
    } else if (looksLikeImage(arg)) {
      images.push(arg)
    } else {
      tickers.push(arg.toUpperCase())
    }
  }
  return { tickers, images }
}

async function main (inputTickers, images = []) {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY is required')
  }

  const described = [
    ...inputTickers,
    ...images.map(image => `${image} (image)`),
  ]
  console.log(`Analyzing ${described.length} input(s): ${described.join(', ')}`)
  if (isLLMObsEnabled()) {
    console.log(`LLMObs enabled: ml_app=${mlApp}, site=${site}, env=${env}`)
  } else {
    console.log('LLMObs disabled: set DD_API_KEY (or DD_LLMOBS_ENABLED=true) to submit traces')
  }
  if (images.length > 0) {
    console.log(`Translating ${images.length} image input(s) to ticker symbols...`)
  }
  console.log('Running parallel analysis with web search...\n')

  const { briefing, spanContext, tickers } = await analyzePortfolio(inputTickers, {
    images,
    onImagesResolved: identifications => {
      printIdentifications(identifications)
      console.log('')
    },
  })
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
  const { tickers, images } = parseArgs(process.argv)
  if (tickers.length === 0 && images.length === 0) {
    console.error('Error: provide at least one ticker symbol or image')
    console.error(USAGE)
    process.exitCode = 1
    return
  }

  try {
    await main(tickers, images)
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
  parseArgs,
  printBriefing,
}
