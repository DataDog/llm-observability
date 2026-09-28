import { defineTool } from "eve/tools";
import { z } from "zod";
import { search } from "../lib/stock-watchlist/searcher";

export default defineTool({
  description: "Search for public investor sentiment and discussions about a stock or company from Reddit, forums, social media, and investor commentary.",
  inputSchema: z.object({
    query: z.string().min(1).describe("Specific public-sentiment search query."),
  }),
  outputSchema: z.object({
    query: z.string(),
    summary: z.string(),
  }),
  async execute({ query }) {
    const expandedQuery = `${query} investor sentiment Reddit discussion forum opinions`;
    const summary = await search(expandedQuery);
    return { query: expandedQuery, summary };
  },
});
