'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

function unquoteEnvValue (value) {
  const trimmed = value.trim()
  if ((trimmed.startsWith('"') && trimmed.endsWith('"')) || (trimmed.startsWith("'") && trimmed.endsWith("'"))) {
    return trimmed.slice(1, -1)
  }
  return trimmed
}

function loadDotEnv () {
  const envPath = process.env.EXPERIMENTS_ENV_FILE || path.resolve(__dirname, '..', '..', '.env')
  if (!fs.existsSync(envPath)) return

  const lines = fs.readFileSync(envPath, 'utf8').split(/\r?\n/)
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) continue

    const match = trimmed.match(/^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$/)
    if (!match) continue

    const [, key, rawValue] = match
    if (process.env[key] === undefined) {
      process.env[key] = unquoteEnvValue(rawValue)
    }
  }
}

loadDotEnv()

function requireEnv (name, fallbackName) {
  const value = process.env[name] || (fallbackName ? process.env[fallbackName] : undefined)
  if (!value) {
    throw new Error(`Missing required env var: ${name}${fallbackName ? ` or ${fallbackName}` : ''}`)
  }
  return value
}

function loadTracer () {
  const tracerPath = process.env.DD_TRACE_JS_PATH
  if (tracerPath) {
    return require(path.resolve(tracerPath))
  }
  return require('dd-trace')
}

function initTracer () {
  requireEnv('DD_API_KEY')
  process.env.DD_APP_KEY = requireEnv('DD_APP_KEY', 'DD_APPLICATION_KEY')

  const tracer = loadTracer()
  tracer.init({
    service: process.env.DD_SERVICE || 'nodejs-experiments-examples',
    llmobs: {
      mlApp: process.env.DD_LLMOBS_PROJECT_NAME || 'nodejs-experiments-examples',
      agentlessEnabled: true,
    },
  })
  return tracer
}

function uniqueName (prefix) {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`
}

async function flushAndWait (tracer, waitMs = 3000) {
  tracer.llmobs.flush()
  await new Promise(resolve => setTimeout(resolve, waitMs))
}

function assertUrl (value, label) {
  assert.equal(typeof value, 'string', `${label} should be a URL string`)
  assert.match(value, /^https:\/\//, `${label} should be an https URL`)
}

module.exports = { assert, assertUrl, flushAndWait, initTracer, requireEnv, uniqueName }
