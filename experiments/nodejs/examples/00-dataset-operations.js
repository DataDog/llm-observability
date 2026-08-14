'use strict'

const { flushAndWait, initTracer, uniqueName } = require('./lib/env')

function answerCapital (inputData) {

  const capitals = {
    France: 'Paris',
    Japan: 'Tokyo',
    Brazil: 'Brasília',
  }
  return { answer: capitals[inputData.country] }
}

function answer_match (_inputData, outputData, expectedOutput) {
  return outputData.answer === expectedOutput
}

function tagged_answer_rate (_inputs, _outputs, _expectedOutputs, evaluatorResults) {
  const values = evaluatorResults.answer_match || []
  return values.filter(Boolean).length / values.length
}

function logPush (label, result) {
  console.log(`${label}: ${result.pushedCount} record(s) pushed; ${result.totalCount} mutation(s) submitted`)
}

function logRecords (label, records) {
  console.log(label)
  for (const record of records) {
    console.log(`  ${record.id}: input=${JSON.stringify(record.input)} tags=${record.tags.join(', ')}`)
  }
}

async function main () {
  const tracer = initTracer()
  const name = uniqueName('nodejs-capitals')

  const dataset = tracer.llmobs.experiments.createDataset(name, {
    description: 'Node.js dataset validation example',
    records: [
      {
        id: 'france',
        inputData: { country: 'France' },
        expectedOutput: 'Paris',
        metadata: { continent: 'Europe' },
        tags: ['split:e2e', 'topic:geography'],
      },
      {
        id: 'japan',
        inputData: { country: 'Japan' },
        expectedOutput: 'Tokyo',
        metadata: { continent: 'Asia' },
        tags: ['split:e2e', 'topic:geography'],
      },
    ],
  })

  // Match the Python SDK flow: create a local dataset, then push it to Datadog.
  const pushResult = await dataset.push()
  logPush('Initial dataset push', pushResult)
  logRecords('Initial records', dataset.records())
  console.log(`Dataset URL: ${dataset.url()}`)

  // Pull the dataset back from Datadog, equivalent to Python's LLMObs.pull_dataset(...).
  const pulled = await tracer.llmobs.experiments.pullDataset(name, { expectedRecordCount: 2 })
  console.log(`Pulled ${pulled.records().length} records`)
  logRecords('Pulled records', pulled.records())

  if (pulled.version() !== null) {
    const pinned = await tracer.llmobs.experiments.pullDataset(
      name,
      { version: pulled.version(), expectedRecordCount: 2 }
    )
    console.log(`Pinned pull: version=${pinned.version()}, records=${pinned.records().length}`)
  }

  dataset.addRecord(
    { country: 'Brazil' },
    'Brasília',
    { continent: 'South America' },
    ['split:holdout', 'topic:geography']
  )
  const incrementalPushResult = await dataset.push()
  logPush('Incremental record push', incrementalPushResult)
  console.log(`Record IDs after incremental push: ${dataset.recordIds().join(', ')}`)
  logRecords('Records after incremental push', dataset.records())

  // Exercise record tag updates on an existing backend record. The next push creates a new dataset version.
  dataset.addTags(2, ['split:e2e'])
  dataset.removeTags(2, ['split:holdout'])
  const tagUpdatePushResult = await dataset.push()
  logPush('Add/remove tag push', tagUpdatePushResult)
  logRecords('Records after add/remove tags', [dataset.records()[2]])

  dataset.replaceTags(2, ['reviewed:true', 'split:e2e', 'topic:geography'])
  const replaceTagsPushResult = await dataset.push()
  logPush('Replace tags push', replaceTagsPushResult)
  logRecords('Records after replacing tags', [dataset.records()[2]])

  const noOpPushResult = await dataset.push()
  logPush('No-op push', noOpPushResult)

  const incrementallyPulled = await tracer.llmobs.experiments.pullDataset(name, { expectedRecordCount: 3 })
  console.log(`Pulled ${incrementallyPulled.records().length} records after tag updates`)
  logRecords('Records after tag updates', incrementallyPulled.records())

  // Pull and run an experiment over the tagged slice so dataset filter tags are visible on experiment row events.
  const taggedPull = await tracer.llmobs.experiments.pullDataset(name, {
    expectedRecordCount: 1,
    tags: ['split:e2e'],
  })
  console.log(`Tagged pull: filter=${taggedPull.filterTags().join(', ')}, records=${taggedPull.records().length}`)
  logRecords('Tagged records', taggedPull.records())

  const experiment = tracer.llmobs.experiments.experiment({
    name: uniqueName('nodejs-capitals-tagged-exp'),
    dataset: taggedPull,
    task: answerCapital,
    evaluators: { answer_match },
    summaryEvaluators: { tagged_answer_rate },
    config: { mode: 'dataset-record-tags' },
    tags: { sdk: 'nodejs', example: 'dataset-record-tags', dataset_filter: 'split:e2e' },
  })

  const result = await experiment.run({ throwOnErrors: true })
  console.log(
    `Tagged experiment: rows=${result.rows.length}, ` +
    `answer_match=${result.summaryEvaluations.tagged_answer_rate.value}`
  )
  for (const row of result.rows) {
    console.log(`  row=${row.index} record=${row.recordId} output=${JSON.stringify(row.output)}`)
  }
  console.log(`Experiment URL: ${result.url}`)

  // Exercise full record mutations on the existing dataset: update multiple records and delete another.
  // All changes are applied locally and sent together by the next push().
  dataset.update(0, {
    input: { country: 'Germany' },
    expectedOutput: 'Berlin',
    metadata: { continent: 'Europe', reviewed: true },
  })
  dataset.update(2, {
    input: { country: 'Argentina' },
    expectedOutput: 'Buenos Aires',
    metadata: { continent: 'South America', reviewed: true },
  })
  dataset.delete(1)
  logRecords('Local records after queued update/delete', dataset.records())

  const mutationPushResult = await dataset.push()
  logPush('Update/delete push', mutationPushResult)

  const mutatedPull = await tracer.llmobs.experiments.pullDataset(name, { expectedRecordCount: 2 })
  console.log(`Pulled ${mutatedPull.records().length} records after update/delete`)
  logRecords('Final records', mutatedPull.records())

  await flushAndWait(tracer)

  console.log('Dataset example completed')
  console.log(`Dataset name    : ${name}`)
  console.log(`Dataset URL     : ${dataset.url()}`)
  console.log(`Record IDs      : ${dataset.recordIds().join(', ')}`)
  console.log(`Final record IDs: ${mutatedPull.recordIds().join(', ')}`)
  console.log(`Filter tags     : ${taggedPull.filterTags().join(', ')}`)
  console.log(`Experiment URL  : ${result.url}`)
}

main().catch((err) => {
  console.error(err)
  process.exitCode = 1
})
