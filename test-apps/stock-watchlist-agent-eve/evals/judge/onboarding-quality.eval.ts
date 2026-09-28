import { defineEval } from "eve/evals";

export default defineEval({
  description: "An LLM judge scores whether the agent gives useful stock-watchlist onboarding guidance.",
  tags: ["judge"],
  metadata: {
    expectedOutput:
      "The reply asks for tickers or watchlist/company names, explains supported stock research tasks, and avoids personalized financial advice.",
  },
  async test(t) {
    await t.send(
      "I want to use this agent to monitor stocks, but I'm not sure what information to provide. What should I send you?",
    );

    t.succeeded();
    t.notCalledTool("analyze_watchlist");
    t.judge(
      [
        "The reply should ask for ticker symbols or watchlist/company names.",
        "The reply should explain that the agent can help research quotes, company context, news, sentiment, or a watchlist briefing.",
        "The reply should be concise and avoid presenting itself as personalized financial advice.",
      ].join(" "),
    ).atLeast(0.7);
  },
});
