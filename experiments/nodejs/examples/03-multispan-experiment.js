'use strict'

const { assert, assertUrl, flushAndWait, initTracer, uniqueName } = require('./lib/env')
const { callOpenAIJson } = require('./lib/openai')

function createMultiSpanCapitalTask (llmobs) {
  return async function generate_capital_multispan (inputData, config) {
    return llmobs.trace({
      kind: 'workflow',
      name: 'capital_answer_workflow',
      tags: { example: 'multispan', country: inputData.country },
    }, async (workflowSpan) => {
      llmobs.annotate(workflowSpan, {
        inputData,
        metadata: { country: inputData.country },
      })

      const messages = await llmobs.trace({
        kind: 'task',
        name: 'build_capital_prompt',
        tags: { example: 'multispan', step: 'prompt' },
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

      const result = await callOpenAIJson(llmobs, {
        name: 'openai.generate_capital_multispan',
        model: config.model,
        temperature: config.temperature,
        messages,
      })

      const output = await llmobs.trace({
        kind: 'task',
        name: 'normalize_capital_answer',
        tags: { example: 'multispan', step: 'normalize' },
      }, async (normalizeSpan) => {
        const normalized = { answer: String(result.answer || '').trim() }
        llmobs.annotate(normalizeSpan, {
          inputData: result,
          outputData: normalized,
        })
        return normalized
      })

      llmobs.annotate(workflowSpan, { outputData: output })
      return output
    })
  }
}

function exact_match (_inputData, outputData, expectedOutput) {
  return outputData.answer.toLowerCase() === String(expectedOutput).toLowerCase()
}

function contains_answer (_inputData, outputData, expectedOutput) {
  return outputData.answer.toLowerCase().includes(String(expectedOutput).toLowerCase())
}

function accuracy_summary (_inputs, _outputs, _expectedOutputs, evaluatorResults) {
  const values = evaluatorResults.exact_match || []
  return values.filter(Boolean).length / values.length
}

async function main () {
  const tracer = initTracer()
  const datasetName = uniqueName('nodejs-multispan')
  const dataset = tracer.llmobs.createDataset(datasetName, {
    description: 'Node.js multispan experiment dataset with live OpenAI calls',
    records: [
      { inputData: { country: 'France' }, expectedOutput: 'Paris', metadata: { difficulty: 'easy' } },
      { inputData: { country: 'Japan' }, expectedOutput: 'Tokyo', metadata: { difficulty: 'easy' } },
    ],
  })

  const experiment = tracer.llmobs.experiment({
    name: uniqueName('nodejs-multispan-exp'),
    dataset,
    task: createMultiSpanCapitalTask(tracer.llmobs),
    evaluators: [exact_match, contains_answer],
    summaryEvaluators: [accuracy_summary],
    config: {
      model: process.env.OPENAI_MODEL || 'gpt-4o-mini',
      temperature: 0,
      provider: 'openai',
    },
    tags: { sdk: 'nodejs', example: 'multispan', provider: 'openai' },
  })

  const result = await experiment.run({ maxRetries: 1, retryDelay: () => 0, raiseErrors: true })
  assert.equal(result.rows.length, 2)
  assert.equal(result.rows[0].evaluations.exact_match, true)
  assert.equal(result.rows[1].evaluations.contains_answer, true)
  assert.equal(result.summaryEvaluations.accuracy_summary.value, 1)
  assert.equal(result.runs.length, 1)
  assertUrl(result.url, 'result.url')

  await flushAndWait(tracer)

  console.log('Multispan experiment validation passed')
  console.log(`Dataset URL   : ${dataset.url()}`)
  console.log(`Experiment URL: ${result.url}`)
  console.log(`Experiment ID : ${result.experimentId}`)
  console.log('Each row trace should include nested spans:')
  console.log('experiment row → capital_answer_workflow → build_capital_prompt / openai.generate_capital_multispan / normalize_capital_answer')
  for (const row of result.rows) {
    console.log(`Row ${row.index} span=${row.spanId} trace=${row.traceId}`)
  }
}

main().catch((err) => {
  console.error(err)
  process.exitCode = 1
})
