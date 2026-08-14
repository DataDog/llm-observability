'use strict'

// Enable dd-trace OpenTelemetry interop before anything has a chance to load dd-trace.
process.env.DD_TRACE_OTEL_ENABLED ||= '1'

const { assert, assertUrl, flushAndWait, initTracer, requireEnv, uniqueName } = require('./lib/env')
const { parseJsonObject } = require('./lib/openai')

function requirePackage (name) {
  try {
    return require(name)
  } catch (err) {
    if (err.code === 'MODULE_NOT_FOUND' && err.message.includes(name)) {
      throw new Error(
        `Missing package '${name}'. Install the OTel example dependencies with: ` +
        'npm install openai @opentelemetry/api'
      )
    }
    throw err
  }
}

function otelChatMessages (messages) {
  return JSON.stringify(messages.map(message => ({
    role: message.role,
    parts: [{ type: 'text', content: message.content }],
  })))
}

function setupOpenTelemetry (tracer) {
  const otelApi = requirePackage('@opentelemetry/api')

  // Register dd-trace as the OpenTelemetry TracerProvider. Do not install a separate
  // OpenTelemetry NodeSDK/OTLP exporter here: the goal of this example is for OTel
  // spans created inside the experiment task to become Datadog spans in the same
  // trace as the LLMObs experiment span.
  const tracerProvider = new tracer.TracerProvider()
  tracerProvider.register()

  const serviceName = process.env.OTEL_SERVICE_NAME ||
    process.env.DD_LLMOBS_PROJECT_NAME ||
    process.env.DD_SERVICE ||
    'nodejs-experiments-otel-openai'

  return {
    otelApi,
    otelTracer: otelApi.trace.getTracer('nodejs-experiments-otel-openai'),
    serviceName,
    tracerProvider,
  }
}

function createOtelOpenAIJsonTask (client, otelTracer, otelApi) {
  return async function otel_openai_capital_task (inputData, config, metadata) {
    assert.equal(metadata.instrumentation, 'otel')
    return otelTracer.startActiveSpan('otel.openai_capital_task', async (span) => {
      try {
        const messages = [
          {
            role: 'system',
            content: 'You answer geography questions. Respond only as JSON with shape {"answer":"capital city"}.',
          },
          {
            role: 'user',
            content: `What is the capital of ${inputData.country}?`,
          },
        ]
        span.setAttributes({
          'experiment.input.country': inputData.country,
          'experiment.metadata.case': metadata.case,
          'gen_ai.operation.name': 'chat',
          'gen_ai.system': 'openai',
          'gen_ai.provider.name': 'openai',
          'gen_ai.request.model': config.model,
          'gen_ai.request.temperature': config.temperature,
          'gen_ai.input.messages': otelChatMessages(messages),
        })
        const response = await client.chat.completions.create({
          model: config.model,
          temperature: config.temperature,
          response_format: { type: 'json_object' },
          messages,
        })
        const content = response.choices?.[0]?.message?.content ?? ''
        const answer = String(parseJsonObject(content).answer || '').trim()
        if (response.model) span.setAttribute('gen_ai.response.model', response.model)
        if (response.usage?.prompt_tokens != null) {
          span.setAttribute('gen_ai.usage.input_tokens', response.usage.prompt_tokens)
        }
        if (response.usage?.completion_tokens != null) {
          span.setAttribute('gen_ai.usage.output_tokens', response.usage.completion_tokens)
        }
        if (response.usage?.total_tokens != null) {
          span.setAttribute('gen_ai.usage.total_tokens', response.usage.total_tokens)
        }
        span.setAttribute('gen_ai.output.messages', otelChatMessages([{ role: 'assistant', content }]))
        return { answer }
      } catch (err) {
        span.recordException(err)
        span.setStatus({ code: otelApi.SpanStatusCode.ERROR, message: err.message })
        throw err
      } finally {
        span.end()
      }
    })
  }
}

function exact_match (_inputData, outputData, expectedOutput) {
  return outputData.answer.toLowerCase() === String(expectedOutput).toLowerCase()
}

async function main () {
  const tracer = initTracer()
  const otel = setupOpenTelemetry(tracer)
  const otelApi = otel.otelApi
  const OpenAI = requirePackage('openai')
  const OpenAIClient = OpenAI.default || OpenAI
  const openai = new OpenAIClient({ apiKey: requireEnv('OPENAI_API_KEY') })
  const otelTracer = otel.otelTracer
  const dataset = tracer.llmobs.experiments.createDataset(uniqueName('nodejs-p0-otel-openai'), {
    description: 'P0 Node.js experiment dataset with OpenAI spans instrumented by OpenTelemetry',
    records: [
      {
        inputData: { country: 'Portugal' },
        expectedOutput: 'Lisbon',
        metadata: { case: 'otel-openai-1', instrumentation: 'otel' },
      },
      {
        inputData: { country: 'Kenya' },
        expectedOutput: 'Nairobi',
        metadata: { case: 'otel-openai-2', instrumentation: 'otel' },
      },
    ],
  })

  const experiment = tracer.llmobs.experiments.experiment({
    name: uniqueName('nodejs-p0-otel-openai-exp'),
    dataset,
    task: createOtelOpenAIJsonTask(openai, otelTracer, otelApi),
    evaluators: [exact_match],
    config: {
      model: process.env.OPENAI_MODEL || 'gpt-4o-mini',
      temperature: 0,
      provider: 'openai',
      instrumentation: 'otel',
    },
    tags: { sdk: 'nodejs', example: 'otel-openai', provider: 'openai', instrumentation: 'otel' },
  })

  try {
    const result = await experiment.run({ maxRetries: 1, retryDelay: () => 0, throwOnErrors: true })
    assert.equal(result.rows.length, 2)
    assert.equal(result.rows[0].evaluations.exact_match, true)
    assert.equal(result.rows[1].evaluations.exact_match, true)
    assertUrl(result.url, 'result.url')

    await flushAndWait(tracer)
    await otel.tracerProvider.forceFlush?.().catch(() => {})

    console.log('OpenTelemetry OpenAI experiment P0 validation passed')
    console.log(`Dataset URL       : ${dataset.url()}`)
    console.log(`Experiment URL    : ${result.url}`)
    console.log(`Experiment ID     : ${result.experimentId}`)
    console.log(`OTel service/ml_app: ${otel.serviceName}`)
    console.log('OTel TracerProvider: dd-trace')
    for (const row of result.rows) {
      console.log(`Row ${row.index} span=${row.spanId} trace=${row.traceId}`)
    }
  } catch (err) {
    await otel?.tracerProvider?.shutdown?.().catch(() => {})
    throw err
  }
}

main().catch((err) => {
  console.error(err)
  process.exitCode = 1
})
