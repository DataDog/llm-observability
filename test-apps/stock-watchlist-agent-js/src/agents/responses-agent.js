'use strict'

const { annotate, traceSpan } = require('../observability')
const OpenAI = require('openai')

const client = new OpenAI()
const DEFAULT_MODEL = process.env.OPENAI_MODEL || 'gpt-4o'

function jsonSchemaFormat (name, schema) {
  return {
    type: 'json_schema',
    name,
    strict: true,
    schema,
  }
}

function toolDefinition ({ name, description, parameters }) {
  return {
    type: 'function',
    name,
    description,
    strict: true,
    parameters,
  }
}

function extractOutputText (response) {
  if (response.output_text) return response.output_text

  const parts = []
  for (const item of response.output || []) {
    if (item.type !== 'message') continue
    for (const content of item.content || []) {
      if (content.type === 'output_text' && content.text) {
        parts.push(content.text)
      }
    }
  }
  return parts.join('\n')
}

function parseJsonOutput (text) {
  const trimmed = text.trim()
  const fenced = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/)
  return JSON.parse(fenced ? fenced[1] : trimmed)
}

function stringifyToolOutput (value) {
  return typeof value === 'string' ? value : JSON.stringify(value)
}

async function runResponsesAgent ({
  name,
  systemPrompt,
  userPrompt,
  outputSchema,
  outputSchemaName,
  tools,
  model = DEFAULT_MODEL,
  maxTurns = 12,
}) {
  const toolDefinitions = tools.map(toolDefinition)
  const toolMap = new Map(tools.map(tool => [tool.name, tool]))
  const input = [{ role: 'user', content: userPrompt }]

  annotate({
    inputData: userPrompt,
    toolDefinitions: toolDefinitions.map(tool => ({
      name: tool.name,
      description: tool.description,
      schema: tool.parameters,
    })),
    metadata: { model, agent: name },
  })

  for (let turn = 0; turn < maxTurns; turn++) {
    const response = await client.responses.create({
      model,
      instructions: systemPrompt,
      input,
      tools: toolDefinitions,
      parallel_tool_calls: true,
      text: {
        format: jsonSchemaFormat(outputSchemaName || name, outputSchema),
      },
    })

    const functionCalls = (response.output || []).filter(item => item.type === 'function_call')
    if (functionCalls.length === 0) {
      const outputText = extractOutputText(response)
      const parsed = parseJsonOutput(outputText)
      annotate({ outputData: parsed })
      return parsed
    }

    input.push(...response.output)

    const toolOutputs = await Promise.all(functionCalls.map(async call => {
      const tool = toolMap.get(call.name)
      if (!tool) {
        return {
          type: 'function_call_output',
          call_id: call.call_id,
          output: `Unknown tool: ${call.name}`,
        }
      }

      try {
        const args = call.arguments ? JSON.parse(call.arguments) : {}
        const output = await traceSpan({ kind: 'tool', name: call.name }, async span => {
          annotate(span, { inputData: args, metadata: { agent: name, tool: call.name } })
          const result = await tool.execute(args)
          annotate(span, { outputData: result })
          return result
        })
        return {
          type: 'function_call_output',
          call_id: call.call_id,
          output: stringifyToolOutput(output),
        }
      } catch (err) {
        return {
          type: 'function_call_output',
          call_id: call.call_id,
          output: `Tool ${call.name} failed: ${err.message}`,
        }
      }
    }))

    input.push(...toolOutputs)
  }

  throw new Error(`${name} did not produce final structured output after ${maxTurns} turns`)
}

module.exports = {
  runResponsesAgent,
  jsonSchemaFormat,
}
