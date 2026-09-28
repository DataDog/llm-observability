import { defineEval } from "eve/evals";
import { includes } from "eve/evals/expect";

export default defineEval({
  description: "The agent asks for ticker symbols before doing ticker-specific research.",
  tags: ["smoke", "behavior"],
  async test(t) {
    const turn = await t.send("Can you research a stock for me?");

    t.succeeded();
    t.notCalledTool("analyze_watchlist");
    t.notCalledTool("delegate_research");
    t.notCalledTool("get_stock_quote");
    t.check(turn.message, includes(/ticker|symbol/i));
  },
});
