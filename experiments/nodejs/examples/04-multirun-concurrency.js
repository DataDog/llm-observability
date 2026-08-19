'use strict'

const { assert, assertUrl, flushAndWait, initTracer, uniqueName } = require('./lib/env')
const { callOpenAIJson } = require('./lib/openai')

function sleep (ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function createTracker () {
  let active = 0
  let maxActive = 0

  return {
    async run (fn) {
      active++
      maxActive = Math.max(maxActive, active)
      try {
        await sleep(50)
        return await fn()
      } finally {
        active--
      }
    },
    maxActive () {
      return maxActive
    },
    active () {
      return active
    },
  }
}

function createAnswerCapitalTask (llmobs, taskTracker, taskStartOrder) {
  return async function answer_capital (inputData, config, metadata) {
    assert.equal(config.mode, 'multirun-concurrency')
    assert.equal(config.provider, 'openai')
    assert.equal(typeof metadata.case, 'string')
    taskStartOrder.push(inputData.id)

    return taskTracker.run(() => llmobs.trace({
      kind: 'workflow',
      name: 'capital_answer_workflow',
      tags: { example: 'multirun-concurrency', country: inputData.country },
    }, async (workflowSpan) => {
      llmobs.annotate(workflowSpan, {
        inputData,
        metadata: { country: inputData.country, case: metadata.case },
      })

      const messages = await llmobs.trace({
        kind: 'task',
        name: 'build_capital_prompt',
        tags: { example: 'multirun-concurrency', step: 'prompt' },
      }, async (promptSpan) => {
        const promptMessages = [
          {
            role: 'system',
            content: 'You answer geography questions. Respond only as JSON with shape {"answer":"capital city"}.',
          },
          {
            role: 'user',
            content: `What is the capital of ${inputData.country}?`,
          },
        ]
        llmobs.annotate(promptSpan, {
          inputData,
          outputData: promptMessages,
        })
        return promptMessages
      })

      const answer = await llmobs.trace({
        kind: 'task',
        name: 'lookup_capital_answer',
        tags: { example: 'multirun-concurrency', step: 'lookup' },
      }, async (lookupSpan) => {
        const lookupResult = await callOpenAIJson(llmobs, {
          name: 'openai.lookup_capital',
          model: config.model,
          temperature: config.temperature,
          messages,
        })
        llmobs.annotate(lookupSpan, {
          inputData: messages,
          outputData: lookupResult,
        })
        return lookupResult
      })

      const output = await llmobs.trace({
        kind: 'task',
        name: 'normalize_capital_answer',
        tags: { example: 'multirun-concurrency', step: 'normalize' },
      }, async (normalizeSpan) => {
        const normalized = { answer: String(answer.answer || '').trim() }
        llmobs.annotate(normalizeSpan, {
          inputData: answer,
          outputData: normalized,
        })
        return normalized
      })

      llmobs.annotate(workflowSpan, { outputData: output })
      return output
    }))
  }
}

function assertRunRows (run, expectedRunIteration) {
  assert.equal(run.runIteration, expectedRunIteration)
  assert.equal(run.hasError, false)
  assert.equal(run.rows.length, 4)
  assert.deepEqual(run.rows.map(row => row.index), [0, 1, 2, 3])
  assert.deepEqual(run.rows.map(row => row.output.answer), ['Paris', 'Tokyo', 'Ottawa', 'Berlin'])
  assert.equal(run.summaryEvaluations.exact_match_rate.value, 1)
  assert.equal(run.summaryEvaluations.row_count.value, 4)
  for (const row of run.rows) {
    assert.equal(row.isError, false)
    assert.deepEqual(row.evaluations, {
      exact_match: true,
      contains_answer: true,
    })
  }
}

async function main () {
  const tracer = initTracer()
  const taskTracker = createTracker()
  const evaluatorTracker = createTracker()
  const summaryTracker = createTracker()
  const taskStartOrder = []

  const dataset = tracer.llmobs.experiments.createDataset(uniqueName('nodejs-multirun-concurrency'), {
    description: 'Node.js multirun and concurrency validation dataset',
    records: [
      { inputData: { id: 'france', country: 'France' }, expectedOutput: 'Paris', metadata: { case: 'a' } },
      { inputData: { id: 'japan', country: 'Japan' }, expectedOutput: 'Tokyo', metadata: { case: 'b' } },
      { inputData: { id: 'canada', country: 'Canada' }, expectedOutput: 'Ottawa', metadata: { case: 'c' } },
      { inputData: { id: 'germany', country: 'Germany' }, expectedOutput: 'Berlin', metadata: { case: 'd' } },
    ],
  })

  async function exact_match (_inputData, outputData, expectedOutput) {
    return evaluatorTracker.run(() => outputData.answer === expectedOutput)
  }

  async function contains_answer (_inputData, outputData, expectedOutput) {
    return evaluatorTracker.run(() => outputData.answer.includes(expectedOutput))
  }

  async function exact_match_rate (_inputs, _outputs, _expectedOutputs, evaluatorResults) {
    return summaryTracker.run(() => {
      const values = evaluatorResults.exact_match || []
      return values.filter(Boolean).length / values.length
    })
  }

  async function row_count (inputs) {
    return summaryTracker.run(() => inputs.length)
  }

  const experiment = tracer.llmobs.experiments.experiment({
    name: uniqueName('nodejs-multirun-concurrency-exp'),
    dataset,
    runs: 2,
    task: createAnswerCapitalTask(tracer.llmobs, taskTracker, taskStartOrder),
    evaluators: [exact_match, contains_answer],
    summaryEvaluators: [exact_match_rate, row_count],
    config: {
      mode: 'multirun-concurrency',
      model: process.env.OPENAI_MODEL || 'gpt-5-mini',
      temperature: 0,
      provider: 'openai',
    },
    tags: { sdk: 'nodejs', example: 'multirun-concurrency' },
  })

  const result = await experiment.run({ concurrency: 2, throwOnErrors: true })

  assert.equal(result.runs.length, 2)
  assert.equal(result.rows, result.runs[0].rows)
  assert.equal(result.summaryEvaluations, result.runs[0].summaryEvaluations)
  assertRunRows(result.runs[0], 1)
  assertRunRows(result.runs[1], 2)
  assert.notEqual(result.runs[0].runId, result.runs[1].runId)

  // Runs are sequential, while tasks, evaluators, and summary evaluators are parallel inside each run.
  assert.deepEqual(taskStartOrder.slice(0, 4), ['france', 'japan', 'canada', 'germany'])
  assert.deepEqual(taskStartOrder.slice(4), ['france', 'japan', 'canada', 'germany'])
  assert.equal(taskTracker.maxActive(), 2)
  assert.equal(evaluatorTracker.maxActive(), 2)
  assert.equal(summaryTracker.maxActive(), 2)
  assert.equal(taskTracker.active(), 0)
  assert.equal(evaluatorTracker.active(), 0)
  assert.equal(summaryTracker.active(), 0)

  assertUrl(result.url, 'result.url')
  await flushAndWait(tracer)

  console.log('Multirun/concurrency validation passed')
  console.log(`Dataset URL   : ${dataset.url()}`)
  console.log(`Experiment URL: ${result.url}`)
  console.log(`Experiment ID : ${result.experimentId}`)
  console.log(`Run IDs       : ${result.runs.map(run => run.runId).join(', ')}`)
  console.log(`Task max concurrency      : ${taskTracker.maxActive()}`)
  console.log(`Evaluator max concurrency : ${evaluatorTracker.maxActive()}`)
  console.log(`Summary max concurrency   : ${summaryTracker.maxActive()}`)
  console.log('Each row trace should include nested spans:')
  console.log(
    'experiment row → capital_answer_workflow → build_capital_prompt / ' +
    'lookup_capital_answer → openai.lookup_capital / normalize_capital_answer'
  )
  for (const run of result.runs) {
    console.log(`Run ${run.runIteration} (${run.runId}) rows=${run.rows.length}`)
    for (const row of run.rows) {
      console.log(`  Row ${row.index} span=${row.spanId} trace=${row.traceId}`)
    }
  }
}

main().catch((err) => {
  console.error(err)
  process.exitCode = 1
})
