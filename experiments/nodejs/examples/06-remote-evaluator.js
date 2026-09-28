'use strict'

const { assert, assertUrl, flushAndWait, initTracer, requireEnv, uniqueName } = require('./lib/env')

async function main () {
  const tracer = initTracer()
  const evalName = requireEnv('DD_LLMOBS_EVALUATOR_NAME')
  const { RemoteEvaluator } = tracer.llmobs

  assert.equal(typeof RemoteEvaluator, 'function', 'This example requires RemoteEvaluator support.')

  const dataset = tracer.llmobs.experiments.createDataset(uniqueName('nodejs-remote-evaluator'), {
    description: 'Demonstrates a managed Datadog evaluator attached to a Node.js experiment.',
    records: [{
      inputData: { answer: 'Paris' },
      expectedOutput: 'Paris',
      metadata: { source: 'remote-evaluator-example' },
    }],
  })

  const experiment = tracer.llmobs.experiments.experiment({
    name: uniqueName('nodejs-remote-evaluator-exp'),
    dataset,
    task: inputData => inputData.answer,
    evaluators: [new RemoteEvaluator({ evalName })],
    description: 'Run a managed evaluator against a deterministic experiment output.',
    tags: { sdk: 'nodejs', example: 'remote-evaluator' },
  })

  const result = await experiment.run({ throwOnErrors: true })
  assert.equal(result.rows.length, 1)
  assert.equal(result.rows[0].isError, false)
  assertUrl(result.url, 'result.url')

  await flushAndWait(tracer)

  console.log('Remote evaluator experiment validation passed')
  console.log(`Dataset URL   : ${dataset.url()}`)
  console.log(`Experiment URL: ${result.url}`)
  console.log(`Experiment ID : ${result.experimentId}`)
  console.log(`Evaluator     : ${evalName}`)
  console.log(`Evaluation    : ${JSON.stringify(result.rows[0].evaluations[evalName])}`)
}

main().catch((err) => {
  console.error(err)
  process.exitCode = 1
})
