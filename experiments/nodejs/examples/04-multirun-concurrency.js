'use strict'

const { assert, assertUrl, flushAndWait, initTracer, uniqueName } = require('./lib/env')

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
        return fn()
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

function assertRunRows (run, expectedRunIteration) {
  assert.equal(run.runIteration, expectedRunIteration)
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

  const capitals = {
    France: 'Paris',
    Japan: 'Tokyo',
    Canada: 'Ottawa',
    Germany: 'Berlin',
  }

  async function answer_capital (inputData, config, metadata) {
    assert.equal(config.mode, 'multirun-concurrency')
    assert.equal(typeof metadata.case, 'string')
    taskStartOrder.push(inputData.id)
    return taskTracker.run(() => ({ answer: capitals[inputData.country] }))
  }

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
    task: answer_capital,
    evaluators: [exact_match, contains_answer],
    summaryEvaluators: [exact_match_rate, row_count],
    config: { mode: 'multirun-concurrency' },
    tags: { sdk: 'nodejs', example: 'multirun-concurrency' },
  })

  const result = await experiment.run({ concurrency: 2, throwOnErrors: true })

  assert.equal(result.runs.length, 2)
  assert.equal(result.rows, result.runs[0].rows)
  assert.equal(result.summaryEvaluations, result.runs[0].summaryEvaluations)
  assertRunRows(result.runs[0], 1)
  assertRunRows(result.runs[1], 2)
  assert.notEqual(result.runs[0].runId, result.runs[1].runId)

  // Runs are sequential, while records/evaluators/summary evaluators are parallel inside each run.
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
