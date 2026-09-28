import { defineEval } from "eve/evals";
import { equals, satisfies } from "eve/evals/expect";

const requiredTools = [
  "analyze_watchlist",
  "delegate_research",
  "get_stock_quote",
  "search_company_news",
  "search_public_sentiment",
  "get_company_profile",
] as const;

type InfoResponse = {
  tools?: {
    authored?: Array<{ name?: string }>;
  };
};

export default defineEval({
  description: "The stock-watchlist eve app exposes the adapted workflow and researcher tools.",
  tags: ["smoke", "surface"],
  async test(t) {
    const response = await t.target.fetch("/eve/v1/info");
    t.check(response.status, equals(200));

    const info = (await response.json()) as InfoResponse;
    const toolNames = (info.tools?.authored ?? []).map((tool) => tool.name).filter(Boolean);

    t.check(
      toolNames,
      satisfies(
        (names) => Array.isArray(names) && requiredTools.every((tool) => names.includes(tool)),
        "all stock-watchlist tools are discovered",
      ),
    );
  },
});
