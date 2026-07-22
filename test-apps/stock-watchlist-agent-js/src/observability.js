'use strict'

const path = require('node:path')
const dotenv = require('dotenv')

const appEnvPath = path.resolve(__dirname, '..', '.env')
const cwdEnvPath = path.resolve(process.cwd(), '.env')

// Prefer the app-local .env so this JS sample does not accidentally inherit the
// Python sample's DD_LLMOBS_ML_APP when invoked from a different directory.
dotenv.config({ path: appEnvPath })
if (cwdEnvPath !== appEnvPath) {
  dotenv.config({ path: cwdEnvPath })
}

const mlApp = process.env.DD_LLMOBS_ML_APP || 'stock-watchlist-agent-js'
const site = process.env.DD_SITE || 'datadoghq.com'
const env = process.env.DD_ENV || process.env.NODE_ENV || 'dev'
const explicitEnabled = process.env.DD_LLMOBS_ENABLED
const llmobsEnabled = explicitEnabled == null ? Boolean(process.env.DD_API_KEY) : explicitEnabled !== 'false'

if (llmobsEnabled) {
  process.env.DD_LLMOBS_ENABLED ??= 'true'
  process.env.DD_LLMOBS_ML_APP ??= mlApp
  process.env.DD_LLMOBS_AGENTLESS_ENABLED ??= 'true'
}

const tracer = require('dd-trace').init({
  service: process.env.DD_SERVICE || 'stock-watchlist-agent-js',
  site,
  env,
  // This sample only needs standalone LLMObs agentless intake. Do not try to
  // send regular APM traces to a local Datadog Agent on 127.0.0.1:8126.
  apmTracingEnabled: process.env.DD_APM_TRACING_ENABLED === 'true',
  llmobs: llmobsEnabled
    ? {
        mlApp,
        agentlessEnabled: process.env.DD_LLMOBS_AGENTLESS_ENABLED !== 'false',
      }
    : undefined,
})

const llmobs = tracer.llmobs

function isLLMObsEnabled () {
  return Boolean(llmobs && llmobs.enabled)
}

function traceSpan (options, fn) {
  if (!isLLMObsEnabled()) {
    return fn(null)
  }
  return llmobs.trace(options, fn)
}

function annotate (spanOrOptions, maybeOptions) {
  if (!isLLMObsEnabled()) return
  try {
    if (maybeOptions === undefined) {
      llmobs.annotate(spanOrOptions)
    } else {
      llmobs.annotate(spanOrOptions, maybeOptions)
    }
  } catch (err) {
    // Annotation should never make the sample app fail. Keep the application path
    // equivalent to the Python version's best-effort span-context export.
    console.warn(`LLMObs annotation failed: ${err.message}`)
  }
}

function exportSpan (span) {
  if (!isLLMObsEnabled() || !span) return null
  try {
    return llmobs.exportSpan(span)
  } catch (err) {
    console.warn(`LLMObs span export failed: ${err.message}`)
    return null
  }
}

function submitEvaluation (spanContext, options) {
  if (!isLLMObsEnabled() || !spanContext) return
  llmobs.submitEvaluation(spanContext, options)
}

function flush () {
  if (isLLMObsEnabled()) {
    llmobs.flush()
  }
}

module.exports = {
  tracer,
  llmobs,
  mlApp,
  site,
  env,
  llmobsEnabled,
  isLLMObsEnabled,
  traceSpan,
  annotate,
  exportSpan,
  submitEvaluation,
  flush,
}
