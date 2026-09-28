import { z } from "zod";

export const stockAnalysisSchema = {
  type: "object",
  additionalProperties: false,
  properties: {
    ticker: { type: "string" },
    company_name: { type: "string" },
    current_price: { type: "string" },
    price_change: { type: "string" },
    recent_news: { type: "array", items: { type: "string" } },
    sentiment: { type: "string", enum: ["bullish", "bearish", "neutral"] },
    public_sentiment_summary: { type: "string" },
    key_factors: { type: "array", items: { type: "string" } },
    summary: { type: "string" },
  },
  required: [
    "ticker",
    "company_name",
    "current_price",
    "price_change",
    "recent_news",
    "sentiment",
    "public_sentiment_summary",
    "key_factors",
    "summary",
  ],
} as const;

export const researchBatchResultSchema = {
  type: "object",
  additionalProperties: false,
  properties: {
    analyses: { type: "array", items: stockAnalysisSchema },
  },
  required: ["analyses"],
} as const;

export const portfolioBriefingSchema = {
  type: "object",
  additionalProperties: false,
  properties: {
    analyses: { type: "array", items: stockAnalysisSchema },
    market_overview: { type: "string" },
    highlights: { type: "array", items: { type: "string" } },
    generated_at: { type: "string" },
  },
  required: ["analyses", "market_overview", "highlights", "generated_at"],
} as const;

export const StockAnalysisZodSchema = z.object({
  ticker: z.string().min(1),
  company_name: z.string().min(1),
  current_price: z.string().min(1),
  price_change: z.string().min(1),
  recent_news: z.array(z.string()),
  sentiment: z.enum(["bullish", "bearish", "neutral"]),
  public_sentiment_summary: z.string().min(1),
  key_factors: z.array(z.string()),
  summary: z.string().min(1),
});

export const ResearchBatchResultZodSchema = z.object({
  analyses: z.array(StockAnalysisZodSchema),
});

export const PortfolioBriefingZodSchema = z.object({
  analyses: z.array(StockAnalysisZodSchema),
  market_overview: z.string().min(1),
  highlights: z.array(z.string()),
  generated_at: z.string().min(1),
});

export type StockAnalysis = z.infer<typeof StockAnalysisZodSchema>;
export type ResearchBatchResult = z.infer<typeof ResearchBatchResultZodSchema>;
export type PortfolioBriefing = z.infer<typeof PortfolioBriefingZodSchema>;

export function validateResearchBatchResult(result: unknown): ResearchBatchResult {
  return ResearchBatchResultZodSchema.parse(result);
}

export function validatePortfolioBriefing(briefing: unknown): PortfolioBriefing {
  return PortfolioBriefingZodSchema.parse(briefing);
}
