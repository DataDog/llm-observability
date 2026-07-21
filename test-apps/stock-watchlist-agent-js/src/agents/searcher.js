'use strict'

require('../observability')
const OpenAI = require('openai')

const client = new OpenAI()

const SEARCH_INSTRUCTIONS = [
  'Search the web for the requested information and return a concise, factual summary of your findings.',
  'Include specific numbers, dates, and sources where possible.',
  'Do not editorialize.',
].join(' ')

async function search (query) {
  const response = await client.responses.create({
    model: process.env.OPENAI_SEARCH_MODEL || 'gpt-4o-mini',
    instructions: SEARCH_INSTRUCTIONS,
    input: query,
    tools: [{ type: 'web_search' }],
  })
  return response.output_text || ''
}

module.exports = {
  search,
}
