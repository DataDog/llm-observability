import OpenAI from "openai";
import type { ResponseCreateParamsNonStreaming } from "openai/resources/responses/responses";
import "./observability";

export type FunctionTool = {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  execute: (input: Record<string, unknown>) => Promise<unknown> | unknown;
};

export type ResponseItem = {
  type?: string;
  name?: string;
  call_id?: string;
  arguments?: string;
  content?: string | Array<{ type?: string; text?: string }>;
  [key: string]: unknown;
};

export type OpenAIResponse = {
  output?: ResponseItem[];
  output_text?: string;
  [key: string]: unknown;
};

let openaiClient: OpenAI | null = null;

function getOpenAIClient(): OpenAI {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is required to run stock watchlist research.");
  }

  openaiClient ??= new OpenAI({ apiKey });
  return openaiClient;
}

export async function createResponse(body: Record<string, unknown>): Promise<OpenAIResponse> {
  const response = await getOpenAIClient().responses.create(body as ResponseCreateParamsNonStreaming);
  return response as unknown as OpenAIResponse;
}

export function extractOutputText(response: OpenAIResponse): string {
  if (typeof response.output_text === "string" && response.output_text.length > 0) {
    return response.output_text;
  }

  const parts: string[] = [];
  for (const item of response.output || []) {
    if (item.type !== "message") continue;
    if (typeof item.content === "string") {
      parts.push(item.content);
      continue;
    }
    for (const content of item.content || []) {
      if (content.type === "output_text" && content.text) {
        parts.push(content.text);
      }
    }
  }
  return parts.join("\n");
}

export function parseJsonOutput(text: string): unknown {
  const trimmed = text.trim();
  const fenced = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/);
  return JSON.parse(fenced ? fenced[1] : trimmed);
}
