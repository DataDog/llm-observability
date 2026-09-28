import { defineEval } from "eve/evals";

function mentionsNvda(input: Record<string, unknown>): boolean {
  return Object.values(input).some((value) =>
    /NVDA|NVIDIA/i.test(String(value)),
  );
}

export default defineEval({
  description:
    "Judge-backed routing: a single-stock research request should call the focused quote, profile, news, and sentiment tools.",
  tags: ["judge", "tools", "stock"],
  timeoutMs: 300_000,
  metadata: {
    expectedOutput:
      "The response should answer a request about NVDA using current quote data, company profile context, recent news, and public sentiment, while making clear it is informational research rather than financial advice.",
  },
  async test(t) {
    if (process.env.RUN_LIVE_STOCK_EVALS !== "true") {
      t.skip(
        "Set RUN_LIVE_STOCK_EVALS=true to run OpenAI-backed live stock tool evals.",
      );
    }

    await t.send(
      [
        "Research NVDA for me using your focused stock tools.",
        "I want the current quote, company profile, recent company news, and public investor sentiment.",
        "Keep the final answer concise and do not run the full watchlist workflow.",
      ].join(" "),
    );

    t.succeeded();
    t.calledTool("get_stock_quote", { input: { ticker: "NVDA" }, count: 1 });
    t.calledTool("get_company_profile", {
      input: { ticker: "NVDA" },
      count: 1,
    });
    t.calledTool("search_company_news", { input: mentionsNvda, count: 1 });
    t.calledTool("search_public_sentiment", { input: mentionsNvda, count: 1 });
    t.notCalledTool("analyze_watchlist");
    t.notCalledTool("delegate_research");

    t.judge(
      [
        "The reply should be about NVDA or NVIDIA.",
        "The reply should include information from current quote, company profile, recent news, and public sentiment research.",
        "The reply should be concise and should not present personalized financial advice.",
      ].join(" "),
    ).atLeast(0.7);
  },
});
