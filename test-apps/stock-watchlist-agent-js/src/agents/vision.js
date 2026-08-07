'use strict'

const fs = require('node:fs')
const path = require('node:path')
const { annotate, traceSpan } = require('../observability')
const { tickerFromImageSchema, validateTickerFromImage } = require('../models')
const { jsonSchemaFormat } = require('./responses-agent')
const OpenAI = require('openai')

const client = new OpenAI()
const DEFAULT_VISION_MODEL = process.env.OPENAI_VISION_MODEL || process.env.OPENAI_MODEL || 'gpt-5.4-nano'

const IMAGE_MIME_TYPES = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
}

const VISION_PROMPT = `\
<scope>
You identify the publicly traded company shown in an image and return its stock ticker symbol.
</scope>

<approach>
1. Read every signal in the image: logos, wordmarks, product names, storefronts, chart labels, ticker tape text
2. Decide which single public company the image refers to
3. Map that company to the ticker symbol of its primary US listing (use the local exchange symbol if it is not US-listed)
4. Report how confident you are and what evidence you used
</approach>

<rules>
- Return the ticker in uppercase with no exchange prefix (AAPL, not NASDAQ:AAPL)
- Prefer the parent company's ticker when the image shows a brand or product (e.g. Instagram -> META)
- If the image shows no identifiable public company, set ticker to "UNKNOWN" and confidence to "low"
</rules>

<output>
Return JSON only matching the TickerFromImage schema.
</output>`

function looksLikeImage (value) {
  return Object.keys(IMAGE_MIME_TYPES).includes(path.extname(value).toLowerCase())
}

const IMAGE_QUESTION = 'Which publicly traded company does this image show? Return its ticker symbol.'

function loadImage (imageInput) {
  const filePath = path.resolve(imageInput)
  const extension = path.extname(filePath).toLowerCase()
  const mimeType = IMAGE_MIME_TYPES[extension]
  if (!mimeType) {
    throw new Error(`Unsupported image type "${extension}" for ${imageInput}`)
  }
  if (!fs.existsSync(filePath)) {
    throw new Error(`Image not found: ${imageInput}`)
  }
  return { mimeType, base64: fs.readFileSync(filePath).toString('base64') }
}

async function identifyTicker (imageInput, model = DEFAULT_VISION_MODEL) {
  // Annotated as an `llm` span with imageParts: the LLMObs SDK only renders images
  // on manually annotated llm-kind messages, not from provider auto-instrumentation.
  return traceSpan({ kind: 'llm', name: 'identify_ticker', modelName: model, modelProvider: 'openai' }, async span => {
    const { mimeType, base64 } = loadImage(imageInput)
    annotate(span, {
      inputData: [
        { role: 'system', content: VISION_PROMPT },
        {
          role: 'user',
          content: IMAGE_QUESTION,
          imageParts: [{ mimeType, content: base64 }],
        },
      ],
      metadata: { model, image_input: imageInput },
    })

    const response = await client.responses.create({
      model,
      instructions: VISION_PROMPT,
      input: [
        {
          role: 'user',
          content: [
            { type: 'input_text', text: IMAGE_QUESTION },
            { type: 'input_image', image_url: `data:${mimeType};base64,${base64}`, detail: 'auto' },
          ],
        },
      ],
      text: { format: jsonSchemaFormat('TickerFromImage', tickerFromImageSchema) },
    })

    const result = validateTickerFromImage(JSON.parse((response.output_text || '').trim()))
    const identified = { ...result, ticker: result.ticker.toUpperCase(), source: imageInput }
    annotate(span, { outputData: [{ role: 'assistant', content: JSON.stringify(identified) }] })
    return identified
  })
}

async function resolveTickersFromImages (imageInputs) {
  return traceSpan({ kind: 'workflow', name: 'resolve_tickers_from_images' }, async span => {
    annotate(span, { inputData: imageInputs })
    const identifications = await Promise.all(imageInputs.map(image => identifyTicker(image)))
    const recognized = identifications.filter(item => item.ticker !== 'UNKNOWN')
    const tickers = [...new Set(recognized.map(item => item.ticker))]
    annotate(span, { outputData: { tickers, identifications } })
    return { tickers, identifications }
  })
}

module.exports = {
  identifyTicker,
  resolveTickersFromImages,
  looksLikeImage,
  VISION_PROMPT,
}
