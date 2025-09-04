'use strict'

const fs = require('node:fs');
const tracerPackageString = fs.existsSync('/dd-trace-js') ? '/dd-trace-js' : 'dd-trace';

const tracer = require(tracerPackageString).init();
const llmobs = tracer.llmobs;
const telemetry = require(`${tracerPackageString}/packages/dd-trace/src/telemetry`);

const express = require('express');

// proxy server url
const proxyServerUrl = process.env.PROXY_LLM_SERVER_URL;
function getProxyUrl (provider) {
  return `${proxyServerUrl}/${provider}`;
}

// OpenAI
const OpenAI = require('openai');
const client = new OpenAI({
  baseURL: getProxyUrl('openai')
});

// VertexAI
const { VertexAI } = require('@google-cloud/vertexai');
const vertexai = new VertexAI({
  project: 'datadog-sandbox',
  location: 'us-central1',
});

// Google GenAI
const { GoogleGenAI } = require('@google/genai');
const genai = new GoogleGenAI({
  httpOptions: {
    baseUrl: getProxyUrl('genai')
  }
});

// Anthropic
const { Anthropic } = require('@anthropic-ai/sdk');
const anthropic = new Anthropic({
  baseURL: getProxyUrl('anthropic')
});



const app = express();
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

const tracerVersion = require(`${tracerPackageString}/package.json`).version;
app.get("/sdk/info", (_req, res) => {
  res.json({ version: tracerVersion });
});

app.post('/sdk/trace', async (req, res) => {
  try {
    const maybeExportedSpanCtx = await createTrace(req.body.trace_structure);
    res.json(maybeExportedSpanCtx ?? {});
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  } finally {
    telemetry.appClosing();
  }
});

app.post('/sdk/export_span_with_fake_span', async (req, res) => {
  const fakeSpan = 5;
  const maybeExportedSpanCtx = llmobs.exportSpan(fakeSpan);
  res.json(maybeExportedSpanCtx ?? {});
});

app.post('/sdk/submit_evaluation_metric', async (req, res) => {
  try {
    const { trace_id, span_id, label, metric_type, value, tags, ml_app, timestamp_ms } = req.body;
    const spanContext = {
      traceId: trace_id,
      spanId: span_id,
    }
    
    llmobs.submitEvaluation(spanContext, {
      label,
      metricType: metric_type,
      value,
      tags,
      mlApp: ml_app,
      timestampMs: timestamp_ms,
    })

    res.json({});
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  } finally {
    telemetry.appClosing();
  }
});

app.post('/sdk/flush', async (req, res) => {
  llmobs.flush();
  telemetry.appClosing()
  res.json({});
});

app.post('/openai/chat_completion', async (req, res) => {
  try {
    const { prompt, model } = req.body;
    const parameters = req.body.parameters ?? {};
    const stream = parameters.stream ?? false;

    if (stream) {
      parameters.stream_options = {
        include_usage: true
      };
    } else {
      delete parameters.stream;
    }

    const tools = req.body.tools ? { tools: req.body.tools, tool_choice: "auto" } : {};

    const response = await client.chat.completions.create({
      model,
      messages: [{ role: 'user', content: prompt }],
      ...tools,
      ...parameters
    });

    if (stream) {
      for await (const part of response) {} // consume the stream
    }

    res.json({ response });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

app.post('/openai/completion', async (req, res) => {
  try {
    const { prompt, parameters, model } = req.body;
    const response = await client.completions.create({
      model,
      prompt: [prompt],
      ...parameters
    });
    res.json({ response });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

app.post('/openai/embedding', async (req, res) => {
  try {
    const { input, model } = req.body;
    const response = await client.embeddings.create({
      model,
      input
    });
    res.json({ response });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

app.post('/openai/responses/create', async (req, res) => {
  try {
    const { input, model } = req.body;
    const parameters = req.body.parameters ?? {};
    const stream = parameters.stream ?? false;

    const tools = req.body.tools ? { tools: req.body.tools, tool_choice: "auto" } : {};

    if (!stream) {
      delete parameters.stream;
    }

    const response = await client.responses.create({
      model,
      input,
      ...tools,
      ...parameters
    });

    if (stream) {
      for await (const part of response) {} // consume the stream
    }

    res.json({ response });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

app.post('/vertexai/completion', async (req, res) => {
  try {
    const { system_instructions, prompt, parameters } = req.body;

    const { generation_config, stream } = parameters;

    const vertexModel = vertexai.getGenerativeModel({
      model: 'gemini-1.5-flash-002',
      systemInstructions: system_instructions,
      generationConfig: normalizeSnakeCaseConfigToCamelCase(generation_config),
    });

    const throwError = !!parameters.candidate_count;
    const request = throwError ? { contents: prompt } : getVertexAIGenerateContentRequest(prompt); // trigger an error

    let tools, toolsConfig;
    if (req.body.tools) {
      const tool = req.body.tools[0].function_declarations[0];
      tools = [{
        function_declarations: {
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters
        }
      }];

      toolsConfig = {
        functionCallingConfig: {
          mode: 'ANY',
          allowedFunctionNames: [tool.name],
        }
      };

      request.tools = tools;
      request.toolsConfig = toolsConfig;
    }

    if (stream) {
      const streamingResult = await vertexModel.generateContentStream(request);
      for await (const _ of streamingResult.stream) {} // consume the stream
      const response = await streamingResult.response;
      res.json({ response });
    } else {
      const response = await vertexModel.generateContent(request);
      res.json({ response });
    }

  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

app.post("/vertexai/chat_completion", async (req, res) => {
  try {
    const { system_instructions, prompt, parameters } = req.body;

    const { generation_config, stream } = parameters;

    const vertexModel = vertexai.getGenerativeModel({
      model: 'gemini-1.5-flash-002',
      systemInstructions: system_instructions,
      generationConfig: normalizeSnakeCaseConfigToCamelCase(generation_config),
    });

    const throwError = !!parameters.candidate_count;
    const request = throwError ? { contents: prompt } : prompt; // trigger an error

    let chatConfig = {};
    if (req.body.tools) {
      const tool = req.body.tools[0].function_declarations[0];
      chatConfig.tools = [{
        function_declarations: {
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters
        }
      }];

      chatConfig.toolsConfig = {
        functionCallingConfig: {
          mode: 'ANY',
          allowedFunctionNames: [tool.name],
        }
      };
    }

    const chat = vertexModel.startChat(chatConfig);

    if (stream) {
      const streamingResult = await chat.sendMessageStream(request);
      for await (const _ of streamingResult.stream) {} // consume the stream
      const response = await streamingResult.response;
      res.json({ response });
    } else {
      const response = await chat.sendMessage(request);
      res.json({ response });
    }
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
})

// TODO(sabrenner): normalize any configs to change snake_case to camelCase
app.post("/genai/generate_content", async (req, res) => {
  try {
    const { model, contents, config = {} } = req.body;

    const stream = config.stream;
    delete config.stream;

    const options = {
      model,
      contents: normalizeGoogleGenAiContents(contents),
      config: normalizeGoogleGenAiConfig(config)
    }

    if (stream) {
      const stream = await genai.models.generateContentStream(options);
      for await (const _ of stream) {} // consume the stream
    } else {
      await genai.models.generateContent(options);
    }

    res.json({});
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

app.post("/genai/embed_content", async (req, res) => {
  try {
    const { model, contents } = req.body;
    await genai.models.embedContent({ model, contents });
    res.json({});
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

app.post("/anthropic/create", async (req, res) => {
  try {
    const {
      model,
      messages,
      system,
      parameters,
      stream,
      stream_as_method
    } = req.body;
  
    const tools = req.body.tools ? { tools: req.body.tools } : {};

    const httpOptions = {};

    if (parameters.extra_headers) {
      httpOptions.headers = parameters.extra_headers;
      delete parameters.extra_headers;
    }
  
    const options = {
      model,
      messages,
      ...parameters,
      ...tools
    };
  
    if (system) {
      options.system = system;
    }
  
    if (stream) {
      if (stream_as_method) {
        const response = await anthropic.messages.stream(options, httpOptions);
        for await (const _ of response) {} // consume the stream
      } else {
        Object.assign(options, { stream: true });
        const response = await anthropic.messages.create(options, httpOptions);
        for await (const _ of response) {} // consume the stream
      }
    } else {
      await anthropic.messages.create(options, httpOptions);
    }

    res.json({});
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

async function createTrace (traceStructure) {
  const isLlmObs = traceStructure.sdk === 'llmobs';
  const makeTrace = traceStructure.sdk === 'llmobs' ? llmobs.trace.bind(llmobs) : tracer.trace.bind(tracer);
  
  let args
  if (isLlmObs) {
    const options = { kind: traceStructure.kind, name: traceStructure.name };
    if (traceStructure.session_id) options.sessionId = traceStructure.session_id;
    if (traceStructure.ml_app) options.mlApp = traceStructure.ml_app;
    if (traceStructure.model_name) options.modelName = traceStructure.model_name;
    if (traceStructure.model_provider) options.modelProvider = traceStructure.model_provider;
    args = [options];
  } else {
    args = [traceStructure.name];
  }

  let exportedSpanCtx;

  const annotations = traceStructure.annotations;
  const annotateAfter = traceStructure.annotate_after;
  let span;

  await makeTrace(...args, async (_span) => {
    span = _span;

    // apply annotations
    if (annotations && !annotateAfter) {
      applyAnnotations(span, annotations);
    }

    // apply export span
    const exportSpan = traceStructure.export_span;
    if (exportSpan) {
      const args = exportSpan === 'explicit' ? [span] : [];
      exportedSpanCtx = llmobs.exportSpan(...args);
    }

    // trace children
    const children = traceStructure.children;
    if (!children) return;

    for (const child of children) {
      if (!Array.isArray(child)) {
        await createTrace(child);
      } else {
        // process all of the array in parallel/async
        await Promise.all(child.map(createTrace));
      }
    }
  })

  if (annotateAfter) {
    // this case should always throw
    applyAnnotations(span, annotations, true);
  }

  return exportedSpanCtx;
}

function applyAnnotations (span, annotations, annotateAfter = false) {
  for (const annotation of annotations) {
    const inputData = annotation.input_data;
    const outputData = annotation.output_data;
    const metadata = annotation.metadata;
    const metrics = annotation.metrics;
    const tags = annotation.tags;

    const args = [];

    if (annotation.explicit_span || annotateAfter) {
      args.push(span);
    }

    args.push({ inputData, outputData, metadata, metrics, tags });

    llmobs.annotate(...args);
  }
}

function getVertexAIGenerateContentRequest (prompt) {
  if (!Array.isArray(prompt)) {
    prompt = [prompt];
  }

  return {
    contents: prompt.map(text => ({ role: 'user', parts: [{ text }] }))
  };
}

function normalizeSnakeCaseConfigToCamelCase (generationConfig) {
  const normalizedConfig = {};
  for (const key of Object.keys(generationConfig)) {
    // turn keys into camelCase
    const camelKey = key.replace(/_([a-z])/g, (_, p1) => p1.toUpperCase());
    normalizedConfig[camelKey] = generationConfig[key];
    if (key !== camelKey) {
      delete normalizedConfig[key];
    }
  }

  return normalizedConfig;
}

function normalizeGoogleGenAiContents (contents) {
  if (!Array.isArray(contents)) {
    contents = [contents];
  }

  return contents.map(content => {
    if (typeof content === 'string') return content;

    const normalizeContent = content;

    if (content.parts) {
      normalizeContent.parts = content.parts.map(part => {
        if (typeof part === 'string') return part;

        return normalizeSnakeCaseConfigToCamelCase(part);
      })

      return normalizeContent;
    } else {
      return normalizeSnakeCaseConfigToCamelCase(content);
    }
  })
}

function normalizeGoogleGenAiConfig (config) {
  const normalizedConfig = normalizeSnakeCaseConfigToCamelCase(config);

  if (normalizedConfig.thinkingConfig) {
    normalizedConfig.thinkingConfig = normalizeSnakeCaseConfigToCamelCase(normalizedConfig.thinkingConfig);
  }

  if (Array.isArray(normalizedConfig.tools)) {
    normalizedConfig.tools = normalizedConfig.tools.map(tool => {
      const normalizedTool = {};

      if (tool.function_declarations) {
        normalizedTool.functionDeclarations = tool.function_declarations;
      }

      if (tool.code_execution) {
        normalizedTool.codeExecution = tool.code_execution;
      }

      return normalizedTool;
    })
  }

  return normalizedConfig;
}

app.listen(process.env.PORT, '0.0.0.0', () => {
  console.log('Server listening on port ' + process.env.PORT);
})
