import io.javalin.Javalin;
import io.javalin.http.Context;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.JsonProcessingException;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.List;

import datadog.trace.api.llmobs.LLMObs;
import datadog.trace.api.llmobs.LLMObsSpan;
import datadog.trace.api.GlobalTracer;
import datadog.trace.bootstrap.instrumentation.api.AgentSpan;
import datadog.trace.bootstrap.instrumentation.api.AgentTracer;
import datadog.trace.bootstrap.instrumentation.api.AgentScope;

// OpenAI imports
import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.models.completions.*;
import com.openai.models.chat.completions.*;
import com.openai.models.embeddings.*;
import com.openai.models.responses.*;
import com.openai.core.JsonValue;

// Anthropic imports
import com.anthropic.client.AnthropicClient;
import com.anthropic.client.okhttp.AnthropicOkHttpClient;
import com.anthropic.models.messages.Message;
import com.anthropic.models.messages.MessageCreateParams;
import com.anthropic.models.messages.Model;
import com.anthropic.models.messages.RawMessageStreamEvent;
import com.anthropic.models.messages.TextBlockParam;

public class SingleFileServer {
  private static final ObjectMapper objectMapper = new ObjectMapper();

  public static void main(String[] args) {
    Integer port = Integer.parseInt(System.getenv("PORT"));
    Javalin app = Javalin.create(config -> {
      config.plugins.enableDevLogging();
    }).start(port);

    OpenAIClient openaiClient = OpenAIOkHttpClient.builder()
      .fromEnv()
      .baseUrl(getProxyUrl("openai"))
      .build();

    AnthropicClient anthropicClient = AnthropicOkHttpClient.builder()
      .fromEnv()
      .baseUrl(getProxyUrl("anthropic"))
      .build();

    // SDK endpoints
    app.get("/sdk/info", ctx -> {
      String responseJson = doGetSDKInfo(ctx);
      ctx.result(responseJson).contentType("application/json");
    });

    app.post("/sdk/trace", ctx -> {
      String requestBody = ctx.body();
      JSONObject payload = new JSONObject(requestBody);
      JSONObject traceStructure = payload.getJSONObject("trace_structure");

      String responseJson = doCreateTrace(traceStructure);
      ctx.result(responseJson).contentType("application/json");
    });

    app.post("/sdk/submit_evaluation_metric", ctx -> {
      String responseJson = doSubmitEvaluationMetric(ctx);
      ctx.result(responseJson).contentType("application/json");
    });

    // OpenAI endpoints
    app.post("/openai/completion", ctx -> {
      String responseJson = doOpenAICompletion(ctx, openaiClient);
      ctx.result(responseJson).contentType("application/json");
    });

    app.post("/openai/chat_completion", ctx -> {
      String responseJson = doOpenAIChatCompletion(ctx, openaiClient);
      ctx.result(responseJson).contentType("application/json");
    });

    app.post("/openai/embedding", ctx -> {
      String responseJson = doOpenAIEmbedding(ctx, openaiClient);
      ctx.result(responseJson).contentType("application/json");
    });

    app.post("/openai/responses/create", ctx -> {
      String responseJson = doOpenAIResponsesCreate(ctx, openaiClient);
      ctx.result(responseJson).contentType("application/json");
    });

    // Anthropic endpoints
    app.post("/anthropic/create", ctx -> {
      String responseJson = doAnthropicCreate(ctx, anthropicClient);
      ctx.result(responseJson).contentType("application/json");
    });
  }

  private static String getProxyUrl (String provider) {
    return System.getenv("PROXY_LLM_SERVER_URL") + "/" + provider;
  }

  private static String doOpenAICompletion (Context ctx, OpenAIClient openaiClient) {
    String requestBody = ctx.body();
    JSONObject payload = new JSONObject(requestBody);

    JSONObject parameters = payload.optJSONObject("parameters");

    var builder = CompletionCreateParams.builder();
    builder.model(payload.optString("model"));
    builder.prompt(payload.optString("prompt"));

    if (!Double.isNaN(parameters.optDouble("max_tokens"))) {
      builder.maxTokens(parameters.optLong("max_tokens"));
    }
    if (!Double.isNaN(parameters.optDouble("temperature"))) {
      builder.temperature(parameters.optDouble("temperature"));
    }

    openaiClient.completions().create(builder.build());

    return toJson(new HashMap<String, String>());
  }

  @SuppressWarnings("unchecked")
	private static String doOpenAIChatCompletion (Context ctx, OpenAIClient openaiClient) {
    String requestBody = ctx.body();
    JSONObject payload = new JSONObject(requestBody);

    JSONArray tools = payload.optJSONArray("tools");
    JSONObject parameters = payload.optJSONObject("parameters");
    boolean stream = parameters.optBoolean("stream");

    var builder = ChatCompletionCreateParams.builder();
    builder.model(payload.optString("model"));
    builder.addUserMessage(payload.optString("prompt"));

    if (!Double.isNaN(parameters.optDouble("max_tokens"))) {
      // using deprecated maxTokens for sake of cassettes
      // maxCompletionTokens is preferable
      builder.maxTokens(parameters.optLong("max_tokens"));
    }
    if (!Double.isNaN(parameters.optDouble("temperature"))) { 
      builder.temperature(parameters.optDouble("temperature"));
    }
    if (tools != null) {
      for (int i = 0; i < tools.length(); i++) {
        JSONObject tool = tools.getJSONObject(i);
        
         JSONObject functionObj = tool.getJSONObject("function");
         Map<String, Object> functionMap = functionObj.toMap();
         
         ChatCompletionTool toolObject = ChatCompletionTool.builder()
           .type(JsonValue.from(tool.getString("type")))
           .function(JsonValue.from(functionMap))
           .build();
         builder.addTool(toolObject);
      }

      builder.toolChoice(JsonValue.from("auto"));
    }

    if (stream) {
      builder.streamOptions(ChatCompletionStreamOptions.builder().includeUsage(true).build());
      try (com.openai.core.http.StreamResponse<ChatCompletionChunk> streamResponse = openaiClient.chat().completions().createStreaming(builder.build())) {
        streamResponse.stream().forEach(chunk -> {
            // consume the stream
        });
      }
    } else {
      openaiClient.chat().completions().create(builder.build());
    }

    return toJson(new HashMap<String, String>());
  }

  private static String doOpenAIEmbedding (Context ctx, OpenAIClient openaiClient) {
    String requestBody = ctx.body();
    JSONObject payload = new JSONObject(requestBody);

    String model = payload.getString("model");
    String input = payload.getString("input");

    EmbeddingCreateParams.Builder builder = EmbeddingCreateParams.builder();
    builder.model(model);
    builder.input(input);

    openaiClient.embeddings().create(builder.build());
    
    return toJson(new HashMap<String, String>());
  }

  @SuppressWarnings("unchecked")
  private static String doOpenAIResponsesCreate (Context ctx, OpenAIClient openaiClient) {
    String requestBody = ctx.body();
    JSONObject payload = new JSONObject(requestBody);

    String model = payload.getString("model");
    var input = payload.get("input"); // string or JSONArray<JSONObject>
    JSONObject parameters = payload.optJSONObject("parameters");
    JSONArray tools = payload.optJSONArray("tools");

    boolean stream = parameters.optBoolean("stream");

    ResponseCreateParams.Builder builder = ResponseCreateParams.builder();

    builder.model(model);
    if (input instanceof String) {
      builder.input(JsonValue.from(input));
    } else {
      List<Object> inputList = (List<Object>) deepConvertJsonToJava(input);
      builder.input(JsonValue.from(inputList));
    }

    if (!Double.isNaN(parameters.optDouble("max_output_tokens"))) {
      builder.maxOutputTokens((long) parameters.getDouble("max_output_tokens"));
    }

    if (!Double.isNaN(parameters.optDouble("temperature"))) { 
      builder.temperature(parameters.getDouble("temperature"));
    }

    if (!parameters.optString("instructions").isEmpty()) {
      builder.instructions(parameters.getString("instructions"));
    }

    if (parameters.optJSONObject("reasoning") != null) {
      JSONObject reasoning = parameters.getJSONObject("reasoning");
      builder.reasoning(JsonValue.from(reasoning.toMap()));
    }

    if (tools != null) {
      List<Object> toolsList = (List<Object>) deepConvertJsonToJava(tools);
      
      builder.tools(JsonValue.from(toolsList));
      builder.toolChoice(JsonValue.from("auto"));
    }

    if (stream) {
      try (com.openai.core.http.StreamResponse<ResponseStreamEvent> streamResponse = openaiClient.responses().createStreaming(builder.build())) {
        streamResponse.stream().forEach(chunk -> {
            // consume the stream
        });
      } 
    } else {
      openaiClient.responses().create(builder.build());
    }

    return toJson(new HashMap<String, String>());
  }

  @SuppressWarnings("unchecked")
  private static String doAnthropicCreate (Context ctx, AnthropicClient anthropicClient) {
    String requestBody = ctx.body();
    JSONObject payload = new JSONObject(requestBody);

    String model = payload.getString("model");
    var messages = payload.optJSONArray("messages");
    var system = payload.opt("system");
    JSONObject parameters = payload.optJSONObject("parameters");
    JSONArray tools = payload.optJSONArray("tools");
    boolean stream = payload.optBoolean("stream");


    MessageCreateParams.Builder builder = MessageCreateParams.builder();

    JSONObject extraHeaders = parameters.optJSONObject("extra_headers");
    if (extraHeaders != null) {
      for (String key : extraHeaders.keySet()) {
        builder.putAdditionalHeader(key, (String) extraHeaders.get(key));
      }
    }

    if (system instanceof String) {
      builder.system((String) system);
    } else if (system instanceof JSONArray) {
      List<Object> systemList = (List<Object>) deepConvertJsonToJava(system);
      builder.system(com.anthropic.core.JsonValue.from(systemList));
    }
    
    builder.model(model);

    if (messages != null) {
      List<Object> messagesList = (List<Object>) deepConvertJsonToJava(messages);
      builder.messages(com.anthropic.core.JsonValue.from(messagesList));
    }

    if (!Double.isNaN(parameters.optDouble("temperature"))) {
      builder.temperature(parameters.getDouble("temperature"));
    }

    if (!Double.isNaN(parameters.optDouble("max_tokens"))) {
      builder.maxTokens(parameters.getLong("max_tokens"));
    }

    // do tools
    if (tools != null && tools != JSONObject.NULL) {
      List<Object> toolsList = (List<Object>) deepConvertJsonToJava(tools);
      builder.tools(com.anthropic.core.JsonValue.from(toolsList));
    }

    if (stream) {
      try (com.anthropic.core.http.StreamResponse<RawMessageStreamEvent> streamResponse = anthropicClient.messages().createStreaming(builder.build())) {
        streamResponse.stream().forEach(chunk -> {
          // consume the stream
        });
      }
    } else {
      anthropicClient.messages().create(builder.build());
    }

    return toJson(new HashMap<String, String>());
  }

  private static String doGetSDKInfo (Context ctx) {
    Package tracerPackage = GlobalTracer.class.getPackage();
    String version = tracerPackage.getImplementationVersion();

    Map<String, String> responseMap = Map.of(
      "version", version
    );
    return toJson(responseMap);
  }

  private static String doCreateTrace (JSONObject traceStructure) {
    boolean isLLMObs = "llmobs".equals(traceStructure.getString("sdk"));
    String kind = isLLMObs ? traceStructure.getString("kind") : null;

    String name = traceStructure.optString("name");
    String modelName = traceStructure.optString("model_name", null);
    String modelProvider = traceStructure.optString("model_provider", null);
    String mlApp = traceStructure.optString("ml_app", null);
    String sessionId = traceStructure.optString("session_id", null);

    JSONObject exportedSpanCtx = null;

    JSONArray annotations = traceStructure.optJSONArray("annotations");
    boolean annotateAfter = traceStructure.optBoolean("annotate_after");

    if (isLLMObs) {
      LLMObsSpan span = startLLMObsSpan(kind, name, modelName, modelProvider, mlApp, sessionId);

      JSONArray children = traceStructure.optJSONArray("children");
      doTraceChildren(children);

      if (annotateAfter) {
        // to trigger an exception for not being able to apply annotations after the span is finished
        span.finish();
        doApplyAnnotations(span, kind, annotations);
      } else {
        doApplyAnnotations(span, kind, annotations);
        span.finish();
      }
    } else {
      AgentSpan span = AgentTracer
          .get()
          .buildSpan(name)
          .start();

      AgentScope scope = AgentTracer
        .get()
        .activateSpan(span);

      JSONArray children = traceStructure.optJSONArray("children");
      doTraceChildren(children);

      span.finish();
      scope.close();
    }

    return toJson(
      Map.of(
        "foo", "bar"
      )
    );
  }

  private static void doTraceChildren(JSONArray children) {
    if (children == null) { 
      return; 
    }

    for (int i = 0; i < children.length(); i++) {
      JSONObject child = children.getJSONObject(i);
      doCreateTrace(child);
    }
  }

  private static void doApplyAnnotations (LLMObsSpan span, String kind, JSONArray annotations) {
    if (annotations == null) {
      return;
    }

    for (int i = 0; i < annotations.length(); i++) {
      JSONObject annotation = annotations.getJSONObject(i);

      // apply IO annotations - could be a string or a JSON object
      Object inputDataObject = annotation.opt("input_data");
      Object outputDataObject = annotation.opt("output_data");

      if (inputDataObject instanceof String && outputDataObject instanceof String) {
        String inputData = (String) inputDataObject;
        String outputData = (String) outputDataObject;

        span.annotateIO(inputData, outputData);
      } else if ("llm".equals(kind)) {
        JSONArray inputDataMessages = null;
        if (!(inputDataObject instanceof JSONArray)) {
          inputDataMessages = new JSONArray();
          inputDataMessages.put(inputDataObject);
        } else {
          inputDataMessages = (JSONArray) inputDataObject;
        }

        JSONArray outputDataMessages = null;
        if (!(outputDataObject instanceof JSONArray)) {
          outputDataMessages = new JSONArray();
          outputDataMessages.put(outputDataObject);
        } else {
          outputDataMessages = (JSONArray) outputDataObject;
        }

        List<LLMObs.LLMMessage> inputData = new ArrayList<>();
        for (int j = 0; j < inputDataMessages.length(); j++) {
          JSONObject message = inputDataMessages.optJSONObject(j);
          if (message == null) {
            continue;
          }

          String role = message.optString("role");
          String content = message.optString("content");
          // TODO: add tool calls when tests are added

          inputData.add(LLMObs.LLMMessage.from(role, content));
        }

        List<LLMObs.LLMMessage> outputData = new ArrayList<>();
        for (int j = 0; j < outputDataMessages.length(); j++) {
          JSONObject message = outputDataMessages.optJSONObject(j);
          if (message == null) {
            continue;
          }

          String role = message.optString("role");
          String content = message.optString("content");
          // TODO: add tool calls when tests are added

          outputData.add(LLMObs.LLMMessage.from(role, content));
        }

        span.annotateIO(inputData, outputData);
      }

      // apply metadata annotations
      JSONObject metadata = annotation.optJSONObject("metadata");
      if (metadata != null) {
        span.setMetadata(metadata.toMap());
      }

      // apply metrics annotations
      JSONObject metrics = annotation.optJSONObject("metrics");
      if (metrics != null) {
        Map<String, Number> metricsMap = new HashMap<>();
        for (String key : metrics.keySet()) {
          metricsMap.put(key, metrics.getDouble(key));
        }
        span.setMetrics(metricsMap);
      }

      // apply tags annotations
      JSONObject tags = annotation.optJSONObject("tags");
      if (tags != null) {
        span.setTags(tags.toMap());
      }
    }
  }

  private static LLMObsSpan startLLMObsSpan(String kind, String name, String modelName, String modelProvider, String mlApp, String sessionId) {
    if ("llm".equals(kind)) {
      return LLMObs.startLLMSpan(name, modelName, modelProvider, mlApp, sessionId);
    } else if ("task".equals(kind)) {
      return LLMObs.startTaskSpan(name, mlApp, sessionId);
    } else if ("agent".equals(kind)) {
      return LLMObs.startAgentSpan(name, mlApp, sessionId);
    } else if ("workflow".equals(kind)) {
      return LLMObs.startWorkflowSpan(name, mlApp, sessionId);
    } else if ("tool".equals(kind)) {
      return LLMObs.startToolSpan(name, mlApp, sessionId);
    } else {
      // TODO: add embedding and retrieval spans once support is added
      throw new RuntimeException("Unsupported kind: " + kind);
    }
  }

  private static String doSubmitEvaluationMetric(Context ctx) {
    String requestBody = ctx.body();
    JSONObject payload = new JSONObject(requestBody);

    // TODO: unlike the Python and Node.js SDKs, the Java SDK
    // does not take a dictionary for a span context, it needs the span itself.
    // if it is update to do so, remove this and use payload.trace_id and payload.span_id instead.
    LLMObsSpan span = LLMObs.startTaskSpan("test-task", null, null);

    String label = payload.optString("label");
    var value = payload.optString("value"); // keep the type dynamic

    JSONObject tagsObject = payload.optJSONObject("tags");
    var tags = tagsObject != null ? tagsObject.toMap() : null;

    String mlApp = payload.optString("ml_app");

    if (mlApp != null) {
      LLMObs.SubmitEvaluation(
        span,
        label,
        value,
        mlApp,
        tags
      );
    } else {
      LLMObs.SubmitEvaluation(
        span,
        label,
        value,
        tags
      );
    }

    span.finish();

    String responseJson = toJson(new HashMap<String, String>());
    return responseJson;
  }

  private static String toJson (Object data) {
    try {
      return objectMapper.writeValueAsString(data);
    } catch (JsonProcessingException e) {
      throw new RuntimeException(e);
    }
  }

  private static Object deepConvertJsonToJava(Object obj) {
    if (obj == JSONObject.NULL) {
      return null;
    }
    if (obj instanceof JSONObject) {
      JSONObject jsonObj = (JSONObject) obj;
      Map<String, Object> map = new HashMap<>();
      for (String key : jsonObj.keySet()) {
        map.put(key, deepConvertJsonToJava(jsonObj.get(key)));
      }
      return map;
    } else if (obj instanceof JSONArray) {
      JSONArray jsonArray = (JSONArray) obj;
      List<Object> list = new ArrayList<>();
      for (int i = 0; i < jsonArray.length(); i++) {
        list.add(deepConvertJsonToJava(jsonArray.get(i)));
      }
      return list;
    } else {
      return obj;
    }
  }
}
