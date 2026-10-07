import { openai } from "@ai-sdk/openai";
import { defineAgent } from "eve";

export default defineAgent({
  model: openai("gpt-4o"),
  build: {
    externalDependencies: ["dd-trace"],
  },
});
