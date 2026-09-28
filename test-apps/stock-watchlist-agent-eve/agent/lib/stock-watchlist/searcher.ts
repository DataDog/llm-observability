import "./observability";
import { createResponse, extractOutputText } from "./openai-responses";

const SEARCH_INSTRUCTIONS = [
  "Search the web for the requested information and return a concise, factual summary of your findings.",
  "Include specific numbers, dates, and sources where possible.",
  "Do not editorialize.",
].join(" ");

export async function search(query: string): Promise<string> {
  const response = await createResponse({
    model: process.env.OPENAI_SEARCH_MODEL || "gpt-4o-mini",
    instructions: SEARCH_INSTRUCTIONS,
    input: query,
    tools: [{ type: "web_search" }],
  });
  return extractOutputText(response);
}
