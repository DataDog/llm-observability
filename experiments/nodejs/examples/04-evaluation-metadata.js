'use strict'

const { assert, assertUrl, flushAndWait, initTracer, uniqueName } = require('./lib/env')

const evaluatorVersion = 'normalization-v1'

function createEvaluators (BaseEvaluator, EvaluatorResult) {
  class ExactMatchEvaluator extends BaseEvaluator {
    constructor () {
      super('exact_match')
    }

    evaluate (context) {
      const passed = context.outputData === context.expectedOutput
      return new EvaluatorResult(passed, {
        reasoning: passed ? 'Normalized output matches the expected output.' : 'Normalized output does not match.',
        assessment: passed ? 'pass' : 'fail',
        metadata: {
          case_id: context.metadata.case_id,
          dataset_split: context.metadata.split,
          evaluator_version: evaluatorVersion,
          normalization: context.metadata.experiment_config.normalization,
        },
        tags: {
          dataset_split: context.metadata.split,
          evaluator_version: evaluatorVersion,
        },
      })
    }
  }

  class LengthDeltaEvaluator extends BaseEvaluator {
    constructor () {
      super('length_delta')
    }

    evaluate (context) {
      const delta = Math.abs(context.outputData.length - context.expectedOutput.length)
      return new EvaluatorResult(delta, {
        reasoning: `Normalized output and expected output differ in length by ${delta} character(s).`,
        assessment: delta === 0 ? 'pass' : 'fail',
        metadata: {
          actual_length: context.outputData.length,
          case_id: context.metadata.case_id,
          evaluator_version: evaluatorVersion,
          expected_length: context.expectedOutput.length,
        },
      })
    }
  }

  return [new ExactMatchEvaluator(), new LengthDeltaEvaluator()]
}

async function main () {
  const tracer = initTracer()
  const { BaseEvaluator, EvaluatorResult } = tracer.llmobs
  assert.equal(typeof BaseEvaluator, 'function', 'This example requires BaseEvaluator support.')
  assert.equal(typeof EvaluatorResult, 'function', 'This example requires EvaluatorResult support.')

  const dataset = tracer.llmobs.experiments.createDataset(uniqueName('nodejs-evaluation-metadata'), {
    description: 'Demonstrates record metadata flowing into class evaluators and evaluation metrics.',
    records: [
      {
        inputData: { answer: ' Paris ' },
        expectedOutput: 'paris',
        metadata: { case_id: 'matching-answer', split: 'evaluation', source: 'synthetic' },
        tags: ['split:evaluation'],
      },
      {
        inputData: { answer: 'Lyon' },
        expectedOutput: 'paris',
        metadata: { case_id: 'different-answer', split: 'evaluation', source: 'synthetic' },
        tags: ['split:evaluation'],
      },
    ],
  })

  const experiment = tracer.llmobs.experiments.experiment({
    name: uniqueName('nodejs-evaluation-metadata-exp'),
    dataset,
    task: (inputData, config, metadata) => {
      assert.equal(metadata.split, 'evaluation')
      assert.equal(config.normalization, 'trim-lowercase')
      return inputData.answer.trim().toLowerCase()
    },
    evaluators: createEvaluators(BaseEvaluator, EvaluatorResult),
    description: 'Propagate dataset and evaluator metadata onto experiment evaluation metrics.',
    config: {
      adapter: 'node',
      generated_by: 'nodejs-example',
      model: 'deterministic-normalizer-v1',
      normalization: 'trim-lowercase',
      purpose: 'demonstrate-evaluation-metadata-propagation',
    },
    tags: {
      adapter: 'node',
      example: 'evaluation-metadata',
      purpose: 'evaluation-metadata-propagation',
    },
  })

  const result = await experiment.run({ throwOnErrors: true })
  assert.deepEqual(result.rows.map(row => row.evaluations.exact_match), [true, false])
  assert.deepEqual(result.rows.map(row => row.evaluations.length_delta), [0, 1])
  for (const row of result.rows) assert.equal(row.isError, false)
  assertUrl(result.url, 'result.url')

  await flushAndWait(tracer)

  console.log('Evaluation metadata experiment validation passed')
  console.log(`Dataset URL   : ${dataset.url()}`)
  console.log(`Experiment URL: ${result.url}`)
  console.log(`Experiment ID : ${result.experimentId}`)
  console.log('Inspect exact_match and length_delta for reasoning, assessment, metadata, and tags.')
}

main().catch((err) => {
  console.error(err)
  process.exitCode = 1
})
