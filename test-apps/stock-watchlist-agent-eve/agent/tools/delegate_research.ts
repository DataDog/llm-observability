import { defineTool } from "eve/tools";
import { z } from "zod";
import { researchStocks } from "../lib/stock-watchlist/researcher";
import { ResearchBatchResultZodSchema } from "../lib/stock-watchlist/models";

function normalizeTickers(tickers: string[]): string[] {
  return [...new Set(tickers.map((ticker) => ticker.trim().toUpperCase()).filter(Boolean))];
}

export default defineTool({
  description:
    "Delegate a multi-stock batch of at least two tickers to the specialized stock researcher. Use only for batch research that is not a full watchlist or portfolio briefing. Do not use for a single ticker or when the user requests focused quote, profile, news, or sentiment tools; call those focused tools directly.",
  inputSchema: z.object({
    tickers: z
      .array(z.string().regex(/^[A-Za-z][A-Za-z0-9.\-]{0,9}$/))
      .min(2)
      .max(12)
      .describe("Two or more stock ticker symbols to research together."),
  }),
  outputSchema: ResearchBatchResultZodSchema,
  async execute({ tickers }) {
    const normalized = normalizeTickers(tickers);
    return researchStocks(normalized);
  },
});
