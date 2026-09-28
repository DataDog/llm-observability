import { defineTool } from "eve/tools";
import { z } from "zod";
import { search } from "../lib/stock-watchlist/searcher";

export default defineTool({
  description: "Search for recent company news and developments. Use a specific query, such as earnings, product launches, guidance, or regulatory events.",
  inputSchema: z.object({
    query: z.string().min(1).describe("Specific company-news search query."),
  }),
  outputSchema: z.object({
    query: z.string(),
    summary: z.string(),
  }),
  async execute({ query }) {
    const summary = await search(query);
    return { query, summary };
  },
});
