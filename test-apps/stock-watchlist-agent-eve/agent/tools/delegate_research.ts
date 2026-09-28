import { defineTool } from "eve/tools";
import { z } from "zod";
import { researchStocks } from "../lib/stock-watchlist/researcher";
import { ResearchBatchResultZodSchema } from "../lib/stock-watchlist/models";

function normalizeTickers(tickers: string[]): string[] {
  return [...new Set(tickers.map((ticker) => ticker.trim().toUpperCase()).filter(Boolean))];
}

export default defineTool({
  description:
    "Delegate a batch of stock tickers to the specialized stock researcher. It runs quote, news, public sentiment, and company-profile research and returns structured StockAnalysis results.",
  inputSchema: z.object({
    tickers: z
      .array(z.string().regex(/^[A-Za-z][A-Za-z0-9.\-]{0,9}$/))
      .min(1)
      .max(12)
      .describe("Batch of stock ticker symbols to research together."),
  }),
  outputSchema: ResearchBatchResultZodSchema,
  async execute({ tickers }) {
    const normalized = normalizeTickers(tickers);
    return researchStocks(normalized);
  },
});
