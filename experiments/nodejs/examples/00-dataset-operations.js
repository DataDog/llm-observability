'use strict'

const { assert, assertUrl, initTracer, uniqueName } = require('./lib/env')

async function main () {
  const tracer = initTracer()
  const name = uniqueName('nodejs-capitals')

  const dataset = tracer.llmobs.experiments.createDataset(name, {
    description: 'Node.js dataset smoke test',
    records: [
      { id: 'france', inputData: { country: 'France' }, expectedOutput: 'Paris', metadata: { continent: 'Europe' } },
      { id: 'japan', inputData: { country: 'Japan' }, expectedOutput: 'Tokyo', metadata: { continent: 'Asia' } },
    ],
  })

  // Match the Python SDK flow: create a local dataset, then push it to Datadog.
  const pushResult = await dataset.push()
  assert.equal(pushResult.totalCount, 2)
  assert.equal(dataset.records().length, 2)
  assert.deepEqual(dataset.recordIds(), ['france', 'japan'])
  assertUrl(dataset.url(), 'dataset.url()')

  // Pull the dataset back from Datadog, equivalent to Python's LLMObs.pull_dataset(...).
  const pulled = await tracer.llmobs.experiments.pullDataset(name, { expectedRecordCount: 2 })
  assert.equal(pulled.records().length, 2)
  const pulledByCountry = new Map(pulled.records().map(record => [record.input.country, record]))
  assert.equal(pulledByCountry.get('France').expectedOutput, 'Paris')
  assert.equal(pulledByCountry.get('Japan').expectedOutput, 'Tokyo')
  assert.equal(pulledByCountry.get('France').id, 'france')
  assert.equal(pulledByCountry.get('Japan').id, 'japan')

  if (pulled.version() !== null) {
    const pinned = await tracer.llmobs.experiments.pullDataset(
      name,
      { version: pulled.version(), expectedRecordCount: 2 }
    )
    assert.equal(pinned.version(), pulled.version())
    assert.equal(pinned.records().length, 2)
  }

  dataset.addRecord({ country: 'Brazil' }, 'Brasília', { continent: 'South America' })
  const incrementalPushResult = await dataset.push()
  assert.deepEqual(incrementalPushResult, { pushedCount: 1, totalCount: 1 })
  assert.equal(dataset.records().length, 3)
  const incrementalRecordIds = dataset.recordIds()
  assert.equal(incrementalRecordIds.length, 3)
  assert.deepEqual(incrementalRecordIds.slice(0, 2), ['france', 'japan'])
  assert.notEqual(incrementalRecordIds[2], '')

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

  console.log('Dataset validation passed')
  console.log(`Dataset name    : ${name}`)
  console.log(`Dataset URL     : ${dataset.url()}`)
  console.log(`Record IDs      : ${dataset.recordIds().join(', ')}`)
}

main().catch((err) => {
  console.error(err)
  process.exitCode = 1
})
