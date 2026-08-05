'use strict'

const { assert, assertUrl, flushAndWait, initTracer, uniqueName } = require('./lib/env')

function routeSupportTicket (inputData, config, metadata) {
  assert.equal(config.mode, 'dataset-record-tags')
  assert.equal(typeof metadata.customerTier, 'string')

  const text = `${inputData.subject} ${inputData.body}`.toLowerCase()
  if (text.includes('invoice') || text.includes('billing')) return { category: 'billing' }
  if (text.includes('deploy') || text.includes('error')) return { category: 'technical' }
  return { category: 'general' }
}

function category_match (_inputData, outputData, expectedOutput) {
  return outputData.category === expectedOutput.category
}

function tagged_record_rate (_inputs, _outputs, _expectedOutputs, evaluatorResults, metadata) {
  assert.deepEqual(metadata.map(item => item.customerTier).sort(), ['enterprise', 'startup'])
  const values = evaluatorResults.category_match || []
  return values.filter(Boolean).length / values.length
}

async function main () {
  const tracer = initTracer()
  const datasetName = uniqueName('nodejs-record-tags')

  const dataset = tracer.llmobs.experiments.createDataset(datasetName, {
    description: 'Node.js dataset record tag validation example',
    records: [
      {
        id: 'billing-easy',
        inputData: {
          subject: 'Invoice question',
          body: 'Can you explain why my invoice changed this month?',
        },
        expectedOutput: { category: 'billing' },
        metadata: { customerTier: 'enterprise' },
        tags: ['split:e2e', 'topic:billing'],
      },
      {
        id: 'deploy-error',
        inputData: {
          subject: 'Deploy is failing',
          body: 'The deploy job shows an error after the latest commit.',
        },
        expectedOutput: { category: 'technical' },
        metadata: { customerTier: 'startup' },
        tags: ['split:holdout', 'topic:technical'],
      },
    ],
  })

  const initialPush = await dataset.push()
  assert.deepEqual(initialPush, { pushedCount: 2, totalCount: 2 })
  assert.deepEqual(dataset.records()[0].tags, ['split:e2e', 'topic:billing'])
  assert.deepEqual(dataset.records()[1].tags, ['split:holdout', 'topic:technical'])
  assertUrl(dataset.url(), 'dataset.url()')

  // Exercise the end-to-end tag operation path: add a tag, remove an old tag, and push a new dataset version.
  dataset.addTags(1, ['split:e2e'])
  dataset.removeTags(1, ['split:holdout'])
  await dataset.push()
  assert.deepEqual(dataset.records()[1].tags, ['split:e2e', 'topic:technical'])

  // Pull back only the tagged evaluation slice. The filter tags are persisted on the Dataset object and forwarded
  // with experiment row events so the Datadog UI can show which dataset slice produced the run.
  const pulled = await tracer.llmobs.experiments.pullDataset(datasetName, {
    expectedRecordCount: 2,
    tags: ['split:e2e'],
  })
  assert.deepEqual(pulled.filterTags(), ['split:e2e'])
  assert.equal(pulled.records().length, 2)

  const recordsById = new Map(pulled.records().map(record => [record.id, record]))
  assert.deepEqual(recordsById.get('billing-easy').tags, ['split:e2e', 'topic:billing'])
  assert.deepEqual(recordsById.get('deploy-error').tags, ['split:e2e', 'topic:technical'])

  const experiment = tracer.llmobs.experiments.experiment({
    name: uniqueName('nodejs-record-tags-exp'),
    dataset: pulled,
    task: routeSupportTicket,
    evaluators: { category_match },
    summaryEvaluators: { tagged_record_rate },
    config: { mode: 'dataset-record-tags' },
    tags: { sdk: 'nodejs', example: 'dataset-record-tags', dataset_filter: 'split:e2e' },
  })

  const result = await experiment.run({ throwOnErrors: true })
  assert.equal(result.rows.length, 2)
  assert.deepEqual(result.rows.map(row => row.recordId).sort(), ['billing-easy', 'deploy-error'])
  assert.equal(result.summaryEvaluations.tagged_record_rate.value, 1)
  for (const row of result.rows) {
    assert.equal(row.isError, false)
    assert.equal(row.evaluations.category_match, true)
  }
  assertUrl(result.url, 'result.url')

  await flushAndWait(tracer)

  console.log('Dataset record tag validation passed')
  console.log(`Dataset URL   : ${pulled.url()}`)
  console.log(`Experiment URL: ${result.url}`)
  console.log(`Experiment ID : ${result.experimentId}`)
  console.log(`Filter tags   : ${pulled.filterTags().join(', ')}`)
  for (const record of pulled.records()) {
    console.log(`Record ${record.id} tags=${record.tags.join(', ')}`)
  }
}

main().catch((err) => {
  console.error(err)
  process.exitCode = 1
})
