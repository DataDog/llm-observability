import { defineTool } from "eve/tools";
import { z } from "zod";
import { search } from "../lib/stock-watchlist/searcher";

export default defineTool({
  description: "Get a company overview including sector, market cap, business description, recent performance, and key financial metrics.",
  inputSchema: z.object({
    ticker: z.string().regex(/^[A-Za-z][A-Za-z0-9.\-]{0,9}$/).describe("Stock ticker symbol, e.g. NVDA."),
  }),
  outputSchema: z.object({
    ticker: z.string(),
    summary: z.string(),
  }),
  async execute({ ticker }) {
    const normalized = ticker.trim().toUpperCase();
    const summary = await search(`${normalized} company profile overview market cap sector fundamentals key metrics 2026`);
    return { ticker: normalized, summary };
  },
});
