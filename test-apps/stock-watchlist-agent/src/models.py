from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class StockAnalysis(BaseModel):
    """Structured analysis of a single stock ticker."""

    ticker: str
    company_name: str
    current_price: str
    price_change: str
    recent_news: list[str]
    sentiment: Literal["bullish", "bearish", "neutral"]
    public_sentiment_summary: str
    key_factors: list[str]
    summary: str


class ResearchBatchResult(BaseModel):
    """Results from a batch of stock research."""

    analyses: list[StockAnalysis]


class PortfolioBriefing(BaseModel):
    """Synthesized briefing across all analyzed stocks."""

    analyses: list[StockAnalysis]
    market_overview: str
    highlights: list[str]
    generated_at: str
