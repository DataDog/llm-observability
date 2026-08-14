'use strict'

const { assert, assertUrl, flushAndWait, initTracer, uniqueName } = require('./lib/env')

function answerCapital (inputData, config, metadata) {
  assert.equal(config.mode, 'dataset-record-tags')
  assert.equal(typeof metadata.continent, 'string')

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

function tagged_answer_rate (_inputs, _outputs, _expectedOutputs, evaluatorResults, metadata) {
  assert.deepEqual(metadata.map(item => item.continent).sort(), ['Asia', 'Europe', 'South America'])
  const values = evaluatorResults.answer_match || []
  return values.filter(Boolean).length / values.length
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
  assert.equal(pushResult.totalCount, 2)
  assert.equal(dataset.records().length, 2)
  assert.deepEqual(dataset.recordIds(), ['france', 'japan'])
  assert.deepEqual(dataset.records()[0].tags, ['split:e2e', 'topic:geography'])
  assert.deepEqual(dataset.records()[1].tags, ['split:e2e', 'topic:geography'])
  assertUrl(dataset.url(), 'dataset.url()')

  // Pull the dataset back from Datadog, equivalent to Python's LLMObs.pull_dataset(...).
  const pulled = await tracer.llmobs.experiments.pullDataset(name, { expectedRecordCount: 2 })
  assert.equal(pulled.records().length, 2)
  const pulledByCountry = new Map(pulled.records().map(record => [record.input.country, record]))
  assert.equal(pulledByCountry.get('France').expectedOutput, 'Paris')
  assert.equal(pulledByCountry.get('Japan').expectedOutput, 'Tokyo')
  assert.equal(pulledByCountry.get('France').id, 'france')
  assert.equal(pulledByCountry.get('Japan').id, 'japan')
  assert.deepEqual(pulledByCountry.get('France').tags, ['split:e2e', 'topic:geography'])
  assert.deepEqual(pulledByCountry.get('Japan').tags, ['split:e2e', 'topic:geography'])

  if (pulled.version() !== null) {
    const pinned = await tracer.llmobs.experiments.pullDataset(
      name,
      { version: pulled.version(), expectedRecordCount: 2 }
    )
    assert.equal(pinned.version(), pulled.version())
    assert.equal(pinned.records().length, 2)
  }

  dataset.addRecord(
    { country: 'Brazil' },
    'Brasília',
    { continent: 'South America' },
    ['split:holdout', 'topic:geography']
  )
  const incrementalPushResult = await dataset.push()
  assert.deepEqual(incrementalPushResult, { pushedCount: 1, totalCount: 1 })
  assert.equal(dataset.records().length, 3)
  const incrementalRecordIds = dataset.recordIds()
  assert.equal(incrementalRecordIds.length, 3)
  assert.deepEqual(incrementalRecordIds.slice(0, 2), ['france', 'japan'])
  assert.notEqual(incrementalRecordIds[2], '')
  assert.deepEqual(dataset.records()[2].tags, ['split:holdout', 'topic:geography'])

  // Exercise record tag updates on an existing backend record. The next push creates a new dataset version.
  dataset.addTags(2, ['split:e2e'])
  dataset.removeTags(2, ['split:holdout'])
  const tagUpdatePushResult = await dataset.push()
  assert.deepEqual(tagUpdatePushResult, { pushedCount: 1, totalCount: 1 })
  assert.deepEqual(dataset.records()[2].tags, ['split:e2e', 'topic:geography'])

  dataset.replaceTags(2, ['reviewed:true', 'split:e2e', 'topic:geography'])
  const replaceTagsPushResult = await dataset.push()
  assert.deepEqual(replaceTagsPushResult, { pushedCount: 1, totalCount: 1 })
  assert.deepEqual(dataset.records()[2].tags, ['reviewed:true', 'split:e2e', 'topic:geography'])

  const noOpPushResult = await dataset.push()
  assert.deepEqual(noOpPushResult, { pushedCount: 0, totalCount: 0 })

  const incrementallyPulled = await tracer.llmobs.experiments.pullDataset(name, { expectedRecordCount: 3 })
  assert.equal(incrementallyPulled.records().length, 3)
  const incrementallyPulledByCountry = new Map(
    incrementallyPulled.records().map(record => [record.input.country, record])
  )
  assert.equal(incrementallyPulledByCountry.get('Brazil').expectedOutput, 'Brasília')
  assert.equal(incrementallyPulledByCountry.get('Brazil').metadata.continent, 'South America')
  assert.equal(incrementallyPulledByCountry.get('Brazil').id, incrementalRecordIds[2])
  assert.deepEqual(incrementallyPulledByCountry.get('Brazil').tags, ['split:e2e', 'topic:geography'])

  // Pull and run an experiment over the tagged slice so dataset filter tags are visible on experiment row events.
  const taggedPull = await tracer.llmobs.experiments.pullDataset(name, {
    expectedRecordCount: 3,
    tags: ['split:e2e'],
  })
  assert.deepEqual(taggedPull.filterTags(), ['split:e2e'])
  assert.equal(taggedPull.records().length, 3)
  for (const record of taggedPull.records()) {
    assert.ok(record.tags.includes('split:e2e'))
    assert.ok(record.tags.includes('topic:geography'))
  }

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
  assert.equal(result.rows.length, 3)
  assert.deepEqual(result.rows.map(row => row.recordId).sort(), [...incrementalRecordIds].sort())
  assert.equal(result.summaryEvaluations.tagged_answer_rate.value, 1)
  for (const row of result.rows) {
    assert.equal(row.isError, false)
    assert.equal(row.evaluations.answer_match, true)
  }
  assertUrl(result.url, 'result.url')

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
  assert.equal(dataset.records().length, 2)
  assert.equal(dataset.records()[0].input.country, 'Germany')
  assert.equal(dataset.records()[0].expectedOutput, 'Berlin')
  assert.equal(dataset.records()[1].input.country, 'Argentina')
  assert.equal(dataset.records()[1].expectedOutput, 'Buenos Aires')

  await dataset.push()

  const mutatedPull = await tracer.llmobs.experiments.pullDataset(name, { expectedRecordCount: 2 })
  assert.equal(mutatedPull.records().length, 2)
  const mutatedByCountry = new Map(mutatedPull.records().map(record => [record.input.country, record]))
  assert.equal(mutatedByCountry.has('Japan'), false)
  assert.equal(mutatedPull.recordIds().includes(incrementalRecordIds[1]), false)
  assert.equal(mutatedByCountry.get('Germany').expectedOutput, 'Berlin')
  assert.equal(mutatedByCountry.get('Germany').metadata.reviewed, true)
  assert.equal(mutatedByCountry.get('Argentina').expectedOutput, 'Buenos Aires')
  assert.equal(mutatedByCountry.get('Argentina').metadata.reviewed, true)
  assert.equal(mutatedByCountry.get('Germany').id, incrementalRecordIds[0])
  assert.equal(mutatedByCountry.get('Argentina').id, incrementalRecordIds[2])

  await flushAndWait(tracer)

  console.log('Dataset validation passed')
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
