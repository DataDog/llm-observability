'use strict'

const { assert, assertUrl, flushAndWait, initTracer, uniqueName } = require('./lib/env')
const { callOpenAIJson } = require('./lib/openai')

function createTask (llmobs, attemptsByQuestion) {
  return async function retrying_openai_task (inputData, config) {
    const attempts = (attemptsByQuestion.get(inputData.question) || 0) + 1
    attemptsByQuestion.set(inputData.question, attempts)
    if (inputData.question === 'retry-once' && attempts === 1) throw new Error('transient task failure')
    if (inputData.question === 'always-fail') throw new Error('permanent task failure')

    const result = await callOpenAIJson(llmobs, {
      name: 'openai.retry_validation',
      model: config.model,
      temperature: config.temperature,
      messages: [
        {
          role: 'system',
          content: 'You are validating retry plumbing. Respond only as JSON with shape {"answer":"ok"}.',
        },
        {
          role: 'user',
          content: `Return the expected answer for test case ${inputData.question}.`,
        },
      ],
    })
    return { answer: String(result.answer || '').trim().toLowerCase() }
  }
}

async function main () {
  const tracer = initTracer()
  const attemptsByQuestion = new Map()
  let evaluatorAttempts = 0

  const dataset = tracer.llmobs.createDataset(uniqueName('nodejs-p0-errors'), {
    description: 'P0 Node.js retry/error handling dataset with live OpenAI calls',
    records: [
      { inputData: { question: 'retry-once' }, expectedOutput: 'ok', metadata: { case: 'transient' } },
      { inputData: { question: 'always-fail' }, expectedOutput: 'ok', metadata: { case: 'permanent' } },
      { inputData: { question: 'evaluator-retry' }, expectedOutput: 'ok', metadata: { case: 'eval-transient' } },
    ],
  })

  function exact_match (_inputData, outputData, expectedOutput) {
    return outputData.answer === expectedOutput
  }

  function flaky_evaluator (inputData) {
    if (inputData.question === 'evaluator-retry') {
      evaluatorAttempts++
      if (evaluatorAttempts === 1) throw new Error('transient evaluator failure')
    }
    return true
  }

  function pass_rate (_inputs, _outputs, _expectedOutputs, evaluatorResults) {
    const values = evaluatorResults.exact_match || []
    return values.filter(Boolean).length / values.length
  }

  const result = await tracer.llmobs.experiment({
    name: uniqueName('nodejs-p0-errors-exp'),
    dataset,
    task: createTask(tracer.llmobs, attemptsByQuestion),
    evaluators: { exact_match, flaky_evaluator },
    summaryEvaluators: { pass_rate },
    config: {
      model: process.env.OPENAI_MODEL || 'gpt-4o-mini',
      temperature: 0,
      provider: 'openai',
    },
    tags: { sdk: 'nodejs', example: 'errors', provider: 'openai' },
  }).run({ maxRetries: 1, retryDelay: () => 0 })

  assert.equal(attemptsByQuestion.get('retry-once'), 2)
  assert.equal(attemptsByQuestion.get('always-fail'), 2)
  assert.equal(evaluatorAttempts, 2)
  assert.equal(result.rows.length, 3)
  assert.equal(result.rows[0].isError, false)
  assert.equal(result.rows[1].isError, true)
  assert.match(result.rows[1].errorMessage, /permanent task failure/)
  assert.equal(result.rows[2].evaluations.flaky_evaluator, true)
  assert.equal(result.rows[2].evaluationErrors.flaky_evaluator, undefined)
  assert.equal(typeof result.summaryEvaluations.pass_rate.value, 'number')
  assert.equal(result.runs.length, 1)
  assert.equal(result.runs[0].summaryEvaluations.pass_rate.value, result.summaryEvaluations.pass_rate.value)
  assertUrl(result.url, 'result.url')

  await assert.rejects(
    () => tracer.llmobs.asyncExperiment({
      name: uniqueName('nodejs-p0-raise-errors-exp'),
      dataset,
      task: () => { throw new Error('raise me') },
    }).run({ raiseErrors: true }),
    /raise me/
  )

  await flushAndWait(tracer)

  console.log('Error/retry/summary P0 validation passed')
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
