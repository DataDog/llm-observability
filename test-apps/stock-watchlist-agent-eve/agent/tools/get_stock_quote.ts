import { defineTool } from "eve/tools";
import { z } from "zod";
import { search } from "../lib/stock-watchlist/searcher";

export default defineTool({
  description: "Get the current stock price, daily price change, and key trading data for a ticker symbol using web research.",
  inputSchema: z.object({
    ticker: z.string().regex(/^[A-Za-z][A-Za-z0-9.\-]{0,9}$/).describe("Stock ticker symbol, e.g. AAPL."),
  }),
  outputSchema: z.object({
    ticker: z.string(),
    summary: z.string(),
  }),
  async execute({ ticker }) {
    const normalized = ticker.trim().toUpperCase();
    const summary = await search(`${normalized} stock price today current quote market data`);
    return { ticker: normalized, summary };
  },
});
