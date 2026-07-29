'use strict'

const fs = require('node:fs')
const path = require('node:path')

const { assert, assertUrl, initTracer, uniqueName } = require('./lib/env')

function rowsFromCsv (csvPath) {
  const [header, ...lines] = fs.readFileSync(csvPath, 'utf8').trim().split(/\r?\n/)
  const columns = header.split(',')
  return lines.map(line => Object.fromEntries(line.split(',').map((value, index) => [columns[index], value])))
}

async function main () {
  const tracer = initTracer()
  const name = uniqueName('nodejs-p0-capitals')

  const dataset = tracer.llmobs.experiments.createDataset(name, {
    description: 'P0 Node.js dataset smoke test',
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

  const csvName = uniqueName('nodejs-p0-csv-capitals')
  const csvPath = path.resolve(__dirname, 'data', 'capitals.csv')
  const csvRows = rowsFromCsv(csvPath)
  const csvDataset = tracer.llmobs.experiments.createDataset(csvName, {
    description: 'P0 Node.js CSV dataset smoke test',
    records: csvRows.map(row => ({
      id: row.id,
      inputData: { country: row.country, question: row.question },
      expectedOutput: { answer: row.answer },
      metadata: { continent: row.continent, difficulty: row.difficulty },
    })),
  })

  assert.equal(csvDataset.records().length, 3)
  assert.deepEqual(csvDataset.records()[0].input, {
    country: 'France',
    question: 'What is the capital of France?',
  })
  assert.deepEqual(csvDataset.records()[0].expectedOutput, { answer: 'Paris' })
  assert.deepEqual(csvDataset.records()[0].metadata, { continent: 'Europe', difficulty: 'easy' })
  assert.equal(csvDataset.records()[0].id, 'france')

  // CSV datasets follow the same explicit push/pull flow.
  const csvPushResult = await csvDataset.push()
  assert.equal(csvPushResult.totalCount, 3)
  assert.deepEqual(csvDataset.recordIds(), ['france', 'japan', 'brazil'])
  assertUrl(csvDataset.url(), 'csvDataset.url()')

  const csvPulled = await tracer.llmobs.experiments.pullDataset(csvName, { expectedRecordCount: 3 })
  assert.equal(csvPulled.records().length, 3)
  const csvPulledByCountry = new Map(csvPulled.records().map(record => [record.input.country, record]))
  assert.deepEqual(csvPulledByCountry.get('France').expectedOutput, { answer: 'Paris' })
  assert.deepEqual(csvPulledByCountry.get('Japan').expectedOutput, { answer: 'Tokyo' })
  assert.deepEqual(csvPulledByCountry.get('Brazil').expectedOutput, { answer: 'Brasília' })
  assert.equal(csvPulledByCountry.get('Brazil').metadata.difficulty, 'medium')
  assert.equal(csvPulledByCountry.get('France').id, 'france')
  assert.equal(csvPulledByCountry.get('Japan').id, 'japan')
  assert.equal(csvPulledByCountry.get('Brazil').id, 'brazil')

  console.log('Dataset P0 validation passed')
  console.log(`Dataset name    : ${name}`)
  console.log(`Dataset URL     : ${dataset.url()}`)
  console.log(`Record IDs      : ${dataset.recordIds().join(', ')}`)
  console.log(`CSV dataset name: ${csvName}`)
  console.log(`CSV dataset URL : ${csvDataset.url()}`)
  console.log(`CSV record IDs  : ${csvDataset.recordIds().join(', ')}`)
}

main().catch((err) => {
  console.error(err)
  process.exitCode = 1
})
