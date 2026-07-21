'use strict'

const { researchStocks } = require('./researcher')
const { runResponsesAgent } = require('./responses-agent')
const { portfolioBriefingSchema, validatePortfolioBriefing } = require('../models')
const { annotate, exportSpan, traceSpan } = require('../observability')

const delegateResearchTool = {
  name: 'delegate_research',
  description: [
    'Delegate a batch of stock tickers to a specialized research agent.',
    'The agent conducts multi-step research (price, news, sentiment, fundamentals)',
    'for every ticker in the batch and returns structured StockAnalysis results.',
    'Group related stocks together (same sector, similar themes) for focused research.',
    'You may call this multiple times with different batches — they run in parallel.',
  ].join(' '),
  parameters: {
    type: 'object',
    additionalProperties: false,
    properties: {
      tickers: {
        type: 'array',
        items: { type: 'string' },
        description: 'Batch of stock ticker symbols to research together',
      },
    },
    required: ['tickers'],
  },
  execute: async ({ tickers }) => {
    const normalized = tickers.map(ticker => String(ticker).toUpperCase())
    const result = await researchStocks(normalized)
    return JSON.stringify(result)
  },
}

const ORCHESTRATOR_PROMPT = `\
<scope>
You are a senior portfolio analyst who plans research strategy, delegates work to \
specialized research agents, and synthesizes results into a portfolio briefing.
</scope>

<approach>
Think step-by-step before taking action:
1. Review the tickers provided and plan how to batch them efficiently
2. Group related stocks together (same sector, similar themes) for focused research
3. Delegate each batch to a research agent via delegate_research
4. Review the results returned from each batch and identify cross-cutting themes
5. Synthesize everything into a comprehensive PortfolioBriefing
</approach>

<tools>
delegate_research — Delegate a batch of stock tickers to a research agent. \
The agent conducts multi-step research (price, news, sentiment, fundamentals) \
for every ticker in the batch. Group related stocks together for efficiency. \
You may call this multiple times with different batches — they run in parallel.
</tools>

<examples>
User: "Analyze AAPL, GOOGL, NVDA, MSFT, TSLA, AMZN"

Thought: 6 tickers across tech. I should batch by similarity:
- AAPL, MSFT, GOOGL, AMZN: Big tech / cloud platforms
- NVDA, TSLA: Semiconductors + EVs — smaller batch, different growth drivers

Action: [calls delegate_research(tickers=["AAPL", "MSFT", "GOOGL", "AMZN"])]
Action: [calls delegate_research(tickers=["NVDA", "TSLA"])]

Result: Research results for both batches returned.

Thought: I now have analyses for all 6 stocks. Key cross-cutting themes:
- Big tech investing heavily in AI infrastructure
- Semiconductor demand remains strong driven by AI
- Mixed sentiment on EV market but Tesla innovating

[Synthesizes into PortfolioBriefing]

---

User: "Analyze AAPL, NVDA"

Thought: Just 2 tickers, both tech but different subsectors. A single batch is most efficient.

Action: [calls delegate_research(tickers=["AAPL", "NVDA"])]

Result: Research for both stocks returned.

Thought: Apple focused on consumer tech, NVIDIA on AI infrastructure. \
Different growth drivers but both benefit from AI tailwinds. Now I'll synthesize.

[Produces PortfolioBriefing]
</examples>

<output>
Be concise and actionable. Cite specific numbers and dates from the research.
Focus on cross-cutting themes and what matters most to an investor reviewing their watchlist.
Return JSON only matching the PortfolioBriefing schema. Set generated_at to the current time in the user prompt.
</output>`

function formatUtcNow () {
  const date = new Date()
  const yyyy = date.getUTCFullYear()
  const mm = String(date.getUTCMonth() + 1).padStart(2, '0')
  const dd = String(date.getUTCDate()).padStart(2, '0')
  const hh = String(date.getUTCHours()).padStart(2, '0')
  const min = String(date.getUTCMinutes()).padStart(2, '0')
  return `${yyyy}-${mm}-${dd} ${hh}:${min} UTC`
}

async function analyzePortfolio (tickers) {
  return traceSpan({ kind: 'agent', name: 'analyze_portfolio' }, async span => {
    const now = formatUtcNow()
    const prompt = `Analyze these stock tickers: ${tickers.join(', ')}. Current time: ${now}`
    const spanContext = exportSpan(span)

    annotate(span, { inputData: tickers, metadata: { generated_at: now } })
    const briefing = await traceSpan({ kind: 'agent', name: 'orchestrator' }, async orchestratorSpan => {
      annotate(orchestratorSpan, { inputData: prompt, metadata: { generated_at: now } })
      const result = await runResponsesAgent({
        name: 'orchestrator',
        systemPrompt: ORCHESTRATOR_PROMPT,
        userPrompt: prompt,
        outputSchema: portfolioBriefingSchema,
        outputSchemaName: 'PortfolioBriefing',
        tools: [delegateResearchTool],
      })
      const validated = validatePortfolioBriefing(result)
      annotate(orchestratorSpan, { outputData: validated })
      return validated
    })
    annotate(span, { outputData: briefing })
    return { briefing, spanContext }
  })
}

module.exports = {
  analyzePortfolio,
  ORCHESTRATOR_PROMPT,
}
