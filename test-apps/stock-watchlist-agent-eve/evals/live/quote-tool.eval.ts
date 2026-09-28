import { defineEval } from "eve/evals";
import { includes } from "eve/evals/expect";

export default defineEval({
  description: "Live routing: a narrow quote request should call get_stock_quote for AAPL.",
  tags: ["live", "tools"],
  timeoutMs: 180_000,
  async test(t) {
    if (process.env.RUN_LIVE_STOCK_EVALS !== "true") {
      t.skip("Set RUN_LIVE_STOCK_EVALS=true to run OpenAI-backed live stock tool evals.");
    }

    const turn = await t.send("Get the current quote for AAPL. Keep the final answer short.");

    t.succeeded();
    t.calledTool("get_stock_quote", { input: { ticker: "AAPL" }, count: 1 });
    t.check(turn.message, includes(/AAPL|Apple/i));
  },
});
