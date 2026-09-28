import { defineEval } from "eve/evals";

const stockTools = [
  "analyze_watchlist",
  "delegate_research",
  "get_stock_quote",
  "search_company_news",
  "search_public_sentiment",
  "get_company_profile",
] as const;

export default defineEval({
  description: "A casual greeting should not trigger stock research tools.",
  tags: ["smoke", "behavior"],
  async test(t) {
    await t.send("Hello! What can you help me with?");

    t.succeeded();
    for (const tool of stockTools) {
      t.notCalledTool(tool);
    }
  },
});
