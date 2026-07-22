'use strict'

const { assert, assertUrl, flushAndWait, initTracer, uniqueName } = require('./lib/env')
const { callOpenAIJson } = require('./lib/openai')

function createAnswerCapitalTask (llmobs) {
  return async function generate_capital (inputData, config, metadata) {
    assert.equal(metadata.difficulty, 'easy')
    const result = await callOpenAIJson(llmobs, {
      name: 'openai.generate_capital',
      model: config.model,
      temperature: config.temperature,
      messages: [
        {
          role: 'system',
          content: 'You answer geography questions. Respond only as JSON with shape {"answer":"capital city"}.',
        },
        {
          role: 'user',
          content: `What is the capital of ${inputData.country}?`,
        },
      ],
    })
    return { answer: String(result.answer || '').trim() }
  }
}

function exact_match (_inputData, outputData, expectedOutput) {
  return outputData.answer.toLowerCase() === String(expectedOutput).toLowerCase()
}

function contains_answer (_inputData, outputData, expectedOutput) {
  return outputData.answer.toLowerCase().includes(String(expectedOutput).toLowerCase())
}

function accuracy_summary (_inputs, _outputs, _expectedOutputs, evaluatorResults, metadata) {
  assert.deepEqual(metadata.map(item => item.difficulty), ['easy', 'easy'])
  assert.equal(metadata[0].experiment_config.provider, 'openai')
  const values = evaluatorResults.exact_match || []
  return values.filter(Boolean).length / values.length
}

function assertSpanIdentifier (spanId) {
  // Real LLMObs spans export dd-trace span ids in decimal. The experiment
  // fallback path uses 16-char hex ids when no LLMObs SDK span is available.
  assert.match(spanId, /^(?:[0-9]{1,20}|[0-9a-f]{16})$/)
}

async function main () {
  const tracer = initTracer()
  const datasetName = uniqueName('nodejs-p0-basic')
  const dataset = tracer.llmobs.createDataset(datasetName, {
    description: 'P0 Node.js basic experiment dataset with live OpenAI calls',
    records: [
      { inputData: { country: 'France' }, expectedOutput: 'Paris', metadata: { difficulty: 'easy' } },
      { inputData: { country: 'Japan' }, expectedOutput: 'Tokyo', metadata: { difficulty: 'easy' } },
    ],
  })

  const experiment = tracer.llmobs.experiments.experiment({
    name: uniqueName('nodejs-p0-basic-exp'),
    dataset,
    task: createAnswerCapitalTask(tracer.llmobs),
    evaluators: [exact_match, contains_answer],
    summaryEvaluators: [accuracy_summary],
    config: {
      model: process.env.OPENAI_MODEL || 'gpt-4o-mini',
      temperature: 0,
      provider: 'openai',
    },
    tags: { sdk: 'nodejs', example: 'basic', provider: 'openai' },
  })

  const result = await experiment.run({ maxRetries: 1, retryDelay: () => 0, raiseErrors: true })
  assert.equal(result.rows.length, 2)
  assert.equal(result.rows[0].evaluations.exact_match, true)
  assert.equal(result.rows[1].evaluations.contains_answer, true)
  assert.equal(result.summaryEvaluations.accuracy_summary.value, 1)
  assert.equal(result.runs.length, 1)
  assert.equal(result.runs[0].runIteration, 0)
  assert.equal(result.runs[0].rows.length, 2)
  assert.equal(result.runs[0].summaryEvaluations.accuracy_summary.value, 1)
  for (const row of result.rows) {
    assertSpanIdentifier(row.spanId)
    assert.match(row.traceId, /^[0-9a-f]{32}$/)
    assert.equal(row.isError, false)
  }
  assertUrl(result.url, 'result.url')

  await flushAndWait(tracer)

  console.log('Basic experiment P0 validation passed')
  console.log(`Dataset URL   : ${dataset.url()}`)
  console.log(`Experiment URL: ${result.url}`)
  console.log(`Experiment ID : ${result.experimentId}`)
  for (const row of result.rows) {
    console.log(`Row ${row.index} span=${row.spanId} trace=${row.traceId}`)
  }
}

main().catch((err) => {
  console.error(err)
  process.exitCode = 1
})
