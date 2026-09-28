import { defineTool } from "eve/tools";
import { z } from "zod";
import { analyzePortfolio } from "../lib/stock-watchlist/orchestrator";
import { runEvaluations } from "../lib/stock-watchlist/evals";
import { flush, isLLMObsEnabled, mlApp, site, env } from "../lib/stock-watchlist/observability";
import { PortfolioBriefingZodSchema } from "../lib/stock-watchlist/models";

const evaluationResultSchema = z.object({
  value: z.union([z.boolean(), z.number()]),
  assessment: z.enum(["pass", "fail"]),
  reasoning: z.string(),
});

const outputSchema = z.object({
  tickers: z.array(z.string()),
  briefing: PortfolioBriefingZodSchema,
  llmobs: z.object({
    enabled: z.boolean(),
    mlApp: z.string(),
    site: z.string(),
    env: z.string(),
    traceContext: z.unknown().nullable(),
  }),
  evaluations: z.record(z.string(), evaluationResultSchema).nullable(),
});

type AnalyzeWatchlistOutput = z.infer<typeof outputSchema>;

function normalizeTickers(tickers: string[]): string[] {
  return [...new Set(tickers.map((ticker) => ticker.trim().toUpperCase()).filter(Boolean))];
}

function formatBriefing(output: AnalyzeWatchlistOutput): string {
  const { briefing } = output;
  const lines = [
    `Generated: ${briefing.generated_at}`,
    "",
    "Market overview:",
    briefing.market_overview,
    "",
    "Highlights:",
    ...briefing.highlights.map((highlight) => `- ${highlight}`),
    "",
    "Ticker analyses:",
  ];

  for (const analysis of briefing.analyses) {
    lines.push(
      `- ${analysis.ticker} (${analysis.company_name}) — ${analysis.current_price} (${analysis.price_change}), sentiment: ${analysis.sentiment}`,
      `  Summary: ${analysis.summary}`,
      `  Key factors: ${analysis.key_factors.join("; ")}`,
      `  Recent news: ${analysis.recent_news.join("; ")}`,
      `  Public sentiment: ${analysis.public_sentiment_summary}`,
    );
  }

  if (output.evaluations) {
    lines.push("", "Evaluations:");
    for (const [name, result] of Object.entries(output.evaluations)) {
      lines.push(`- ${name}: ${result.assessment} (${result.value}) — ${result.reasoning}`);
    }
  }

  return lines.join("\n");
}

export default defineTool({
  description:
    "Analyze a stock watchlist by running the adapted stock-watchlist multi-agent workflow with web research, portfolio synthesis, and optional LLMObs evaluations.",
  inputSchema: z.object({
    tickers: z
      .array(z.string().regex(/^[A-Za-z][A-Za-z0-9.\-]{0,9}$/))
      .min(1)
      .max(12)
      .describe("Ticker symbols to analyze, for example ['AAPL', 'GOOGL', 'NVDA']."),
    runEvaluations: z
      .boolean()
      .default(true)
      .describe("When LLMObs is enabled, run and submit completeness, sentiment, and grounding evaluations."),
  }),
  outputSchema,
  async execute({ tickers, runEvaluations: shouldRunEvaluations }) {
    const normalized = normalizeTickers(tickers);
    if (normalized.length === 0) {
      throw new Error("At least one ticker symbol is required.");
    }

    try {
      const { briefing, spanContext } = await analyzePortfolio(normalized);
      const evaluations = shouldRunEvaluations && isLLMObsEnabled() && spanContext
        ? await runEvaluations(briefing, normalized, spanContext)
        : null;

      return {
        tickers: normalized,
        briefing,
        llmobs: {
          enabled: isLLMObsEnabled(),
          mlApp,
          site,
          env,
          traceContext: spanContext,
        },
        evaluations,
      };
    } finally {
      flush();
    }
  },
  toModelOutput(output) {
    return { type: "text", value: formatBriefing(output) };
  },
});
