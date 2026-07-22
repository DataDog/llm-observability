# AGENTS.md

Guidance for running the Node.js LLM Observability experiments examples in this directory.

Every Node.js experiments SDK feature should have a runnable example in this directory when it is added.

## Live Datadog validation

Use staging credentials from `dd-auth` / `dd-auth-env`; never paste or print API/application keys.

Preferred command shape with a centralized env file:

```sh
cd /Users/mehul.sonowal/dd/llm-observability/experiments/nodejs
EXPERIMENTS_ENV_FILE=/path/to/experiments.env \
  dd-auth --domain dd.datad0g.com -- env DD_SITE=datad0g.com npm run validate:experiments
```

Or put credentials in this directory's `.env` file (`cp .env.example .env`) and run:

```sh
dd-auth --domain dd.datad0g.com -- env DD_SITE=datad0g.com npm run validate:experiments
```

Notes:

- The auth domain is `dd.datad0g.com`.
- Set `DD_SITE=datad0g.com` for `dd-trace-js` experiments API calls. The experiments client builds the API host as `api.${DD_SITE}`; using `DD_SITE=dd.datad0g.com` produces `api.dd.datad0g.com`, which fails TLS validation.
- The scripts auto-load `.env`, or the file pointed to by `EXPERIMENTS_ENV_FILE`. Shell env vars and `dd-auth` values override `.env` values.
- Set `OPENAI_API_KEY` in the env file or shell for the experiment trace examples; they intentionally make real OpenAI chat-completion calls to validate nested LLMObs provider spans. `OPENAI_MODEL` defaults to `gpt-4o-mini`.
- Use `npm run stock-watchlist` for the higher-cost complex workflow example with multiple provider calls in one experiment row.
- The examples print dataset and experiment URLs on `https://dd.datad0g.com/...`. Use those URLs for UI validation of datasets, experiment row spans, nested provider LLM spans, row evaluator metrics, and summary metrics.
