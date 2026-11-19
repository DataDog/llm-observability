'use strict'

const { assert, assertUrl, flushAndWait, initTracer, uniqueName } = require('./lib/env')

const structuredOutput = {
  status: 'ok',
  count: 3,
  nested: { a: 1, b: [1, 2, 3] },
}

async function main () {
  const tracer = initTracer()
  const dataset = tracer.llmobs.experiments.createDataset(uniqueName('nodejs-structured-output'), {
    description: 'Demonstrates structured JSON output on Node.js experiment rows.',
    records: [
      { inputData: { prompt: 'smoke test' } },
    ],
  })

  const experiment = tracer.llmobs.experiments.experiment({
    name: uniqueName('nodejs-structured-output-exp'),
    dataset,
    task: async () => structuredOutput,
    tags: { sdk: 'nodejs', example: 'structured-output' },
  })

  const result = await experiment.run({ throwOnErrors: true })
  assert.equal(result.rows.length, 1)
  assert.deepEqual(result.rows[0].output, structuredOutput)
  assert.equal(result.rows[0].isError, false)
  assertUrl(result.url, 'result.url')

  await flushAndWait(tracer)

  console.log('Structured experiment output validation passed')
  console.log(`Dataset URL   : ${dataset.url()}`)
  console.log(`Experiment URL: ${result.url}`)
  console.log(`Experiment ID : ${result.experimentId}`)
  console.log(`Output        : ${JSON.stringify(result.rows[0].output)}`)
  console.log('In the experiment UI, the row Output column should contain expandable JSON fields.')
}

main().catch((err) => {
  console.error(err)
  process.exitCode = 1
})
