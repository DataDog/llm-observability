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
  const evaluatorAttemptsByQuestion = new Map()
  let summaryInputs
  let summaryOutputs
  let summaryExpectedOutputs
  let summaryEvaluatorResults

  const dataset = tracer.llmobs.experiments.createDataset(uniqueName('nodejs-errors'), {
    description: 'Node.js retry/error handling dataset with live OpenAI calls',
    records: [
      { inputData: { question: 'retry-once' }, expectedOutput: 'ok', metadata: { case: 'transient' } },
      { inputData: { question: 'always-fail' }, expectedOutput: 'ok', metadata: { case: 'permanent' } },
      { inputData: { question: 'evaluator-retry' }, expectedOutput: 'ok', metadata: { case: 'eval-transient' } },
      { inputData: { question: 'evaluator-always-fail' }, expectedOutput: 'ok', metadata: { case: 'eval-permanent' } },
    ],
  })

  function exact_match (_inputData, outputData, expectedOutput) {
    return outputData.answer === expectedOutput
  }

  function flaky_evaluator (inputData) {
    const attempts = (evaluatorAttemptsByQuestion.get(inputData.question) || 0) + 1
    evaluatorAttemptsByQuestion.set(inputData.question, attempts)
    if (inputData.question === 'evaluator-retry' && attempts === 1) throw new Error('transient evaluator failure')
    if (inputData.question === 'evaluator-always-fail') throw new Error('permanent evaluator failure')
    return true
  }

  function pass_rate (inputs, outputs, expectedOutputs, evaluatorResults) {
    summaryInputs = inputs
    summaryOutputs = outputs
    summaryExpectedOutputs = expectedOutputs
    summaryEvaluatorResults = evaluatorResults
    const values = evaluatorResults.exact_match || []
    return values.filter(Boolean).length / values.length
  }

  const result = await tracer.llmobs.experiments.experiment({
    name: uniqueName('nodejs-errors-exp'),
    dataset,
    task: createTask(tracer.llmobs, attemptsByQuestion),
    evaluators: { exact_match, flaky_evaluator },
    summaryEvaluators: { pass_rate },
    config: {
      model: process.env.OPENAI_MODEL || 'gpt-5-mini',
      temperature: 0,
      provider: 'openai',
    },
    tags: { sdk: 'nodejs', example: 'errors', provider: 'openai' },
  }).run({ maxRetries: 1, retryDelay: () => 0 })

  assert.equal(attemptsByQuestion.get('retry-once'), 2)
  assert.equal(attemptsByQuestion.get('always-fail'), 2)
  assert.equal(evaluatorAttemptsByQuestion.get('evaluator-retry'), 2)
  assert.equal(evaluatorAttemptsByQuestion.get('evaluator-always-fail'), 2)
  assert.equal(result.rows.length, 4)
  assert.equal(result.rows[0].isError, false)
  assert.equal(result.rows[1].isError, true)
  assert.match(result.rows[1].errorMessage, /permanent task failure/)
  assert.equal(result.rows[2].evaluations.flaky_evaluator, true)
  assert.equal(result.rows[2].evaluationErrors.flaky_evaluator, undefined)
  assert.equal(result.rows[3].evaluations.exact_match, true)
  assert.match(result.rows[3].evaluationErrors.flaky_evaluator, /permanent evaluator failure/)
  assert.deepEqual(summaryInputs.map(input => input.question), [
    'retry-once',
    'always-fail',
    'evaluator-retry',
    'evaluator-always-fail',
  ])
  assert.deepEqual(summaryOutputs.map(output => output?.answer ?? null), ['ok', null, 'ok', 'ok'])
  assert.deepEqual(summaryExpectedOutputs, ['ok', 'ok', 'ok', 'ok'])
  assert.deepEqual(summaryEvaluatorResults.exact_match, [true, null, true, true])
  assert.deepEqual(summaryEvaluatorResults.flaky_evaluator, [true, null, true, null])
  assert.equal(result.summaryEvaluations.pass_rate.value, 3 / 4)
  assert.equal(result.runs.length, 1)
  assert.equal(result.runs[0].summaryEvaluations.pass_rate.value, result.summaryEvaluations.pass_rate.value)
  assertUrl(result.url, 'result.url')

  const bubblingErrorExperiment = tracer.llmobs.experiments.experiment({
    name: uniqueName('nodejs-raise-errors-exp'),
    dataset,
    task: function bubbling_task_error (inputData) {
      assert.equal(inputData.question, 'retry-once')
      throw new Error('raise me')
    },
    tags: { sdk: 'nodejs', example: 'errors', case: 'raise-errors-task' },
  })
  await assert.rejects(
    () => bubblingErrorExperiment.run({ throwOnErrors: true }),
    /raise me/
  )
  assertUrl(bubblingErrorExperiment.url(), 'bubblingErrorExperiment.url()')

  await flushAndWait(tracer)

  console.log('Error/retry/summary validation passed')
  console.log(`Dataset URL   : ${dataset.url()}`)
  console.log(`Experiment URL       : ${result.url}`)
  console.log(`Experiment ID        : ${result.experimentId}`)
  console.log(`Bubbled error URL    : ${bubblingErrorExperiment.url()}`)
  for (const row of result.rows) {
    console.log(`Row ${row.index} span=${row.spanId} trace=${row.traceId}`)
  }
}

main().catch((err) => {
  console.error(err)
  process.exitCode = 1
})
