import { defineEval } from "eve/evals";
import { includes } from "eve/evals/expect";

function hasTickers(input: Record<string, unknown>): boolean {
  const tickers = input.tickers;
  return (
    Array.isArray(tickers) &&
    ["AAPL", "NVDA"].every((ticker) => tickers.map((value) => String(value).toUpperCase()).includes(ticker))
  );
}

export default defineEval({
  description: "Live routing: a full watchlist request should call the orchestrated analyze_watchlist tool.",
  tags: ["live", "tools", "watchlist"],
  timeoutMs: 300_000,
  async test(t) {
    if (process.env.RUN_LIVE_STOCK_EVALS !== "true") {
      t.skip("Set RUN_LIVE_STOCK_EVALS=true to run OpenAI-backed live stock workflow evals.");
    }

    const turn = await t.send(
      "Analyze this watchlist: AAPL and NVDA. Skip evaluations and keep the final answer concise.",
    );

    t.succeeded();
    t.calledTool("analyze_watchlist", { input: hasTickers, count: 1 });
    t.check(turn.message, includes(/AAPL|Apple/i));
    t.check(turn.message, includes(/NVDA|NVIDIA/i));
  },
});
