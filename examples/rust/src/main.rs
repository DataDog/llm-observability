use core::panic;
use std::{collections::HashMap, time::Duration};

use crate::{
    client::DatadogLlmClient,
    config::DatadogLlmConfig,
    models::{Message, Metrics, Span, SpansRequest},
    span_extraction::{
        extract_span_id, extract_trace_id, generate_span_id, generate_trace_id,
        get_current_timestamp_ns,
    },
};

mod client;
mod config;
mod models;
mod span_extraction;

#[tokio::main]
async fn main() {
    // Create the DatadogLlm client by parsing environment variables, and then using that config to create an LLM Client
    let client = match DatadogLlmConfig::from_env() {
        Some(config) => match DatadogLlmClient::new(config) {
            Ok(client) => {
                tracing::info!("Datadog LLM Observability enabled");
                Some(client)
            }
            Err(e) => {
                tracing::warn!("Failed to initialize LLM observability: {}", e);
                None
            }
        },
        None => {
            tracing::warn!("LLM observability enabled but DD_API_KEY not set");
            None
        }
    };

    // Fail quickly if client is not configured.
    if client.is_none() {
        tracing::info!("Datadog LLM Observability disabled");
        return;
    }

    let client = client.unwrap();

    // Simulate an LLM call, at the point of call collect the System and User prompt
    let input_messages = vec![
        Message::system("this is a test message"),
        Message::user("and the user prompt"),
    ];

    let start_ns = get_current_timestamp_ns();

    tokio::time::sleep(Duration::from_secs(5)).await; // Simulate LLM call;

    let duration_ns = get_current_timestamp_ns() - start_ns;

    let trace_id = extract_trace_id().unwrap_or(generate_trace_id());
    let span_id = extract_span_id().unwrap_or(generate_span_id());

    // Build LLM span
    let mut span = Span::new_llm_span(
        "bedrock_invoke_model".to_string(),
        span_id,
        trace_id,
        None, // parent_id - using None for root span
        start_ns,
        duration_ns,
    )
    .with_input_messages(input_messages);

    // Simulate the output from the LLM
    let output_messages = vec![Message::assistant("this is the response from the LLM")];

    // Add the output message to the span
    span = span
        .with_output_messages(output_messages)
        // And also add token metrics. How you get these metrics will depend on your LLM provider.
        .with_metrics(Metrics {
            input_tokens: Some(123.0),
            output_tokens: Some(900.0),
            total_tokens: Some(123.0 + 900.0),
        });

    // Add metadata (model parameters)
    let mut metadata = HashMap::new();
    metadata.insert(
        "temperature".to_string(),
        serde_json::Value::Number(
            serde_json::Number::from_f64(0.7 as f64).unwrap_or(serde_json::Number::from(0)),
        ),
    );
    metadata.insert(
        "max_tokens".to_string(),
        serde_json::Value::Number(serde_json::Number::from(1024)),
    );
    metadata.insert(
        "model_name".to_string(),
        serde_json::Value::String("gpt-4o".to_string()),
    );
    metadata.insert(
        "model_provider".to_string(),
        serde_json::Value::String("openai".to_string()),
    );

    span = span.with_metadata(metadata);

    // Use the DatadogLlmClient to send the span to Datadog
    let res = client
        .send_spans(SpansRequest::new("rust_llm_obs".to_string(), span))
        .await;

    if res.is_err() {
        panic!("Failed to send spans to Datadog: {}", res.err().unwrap());
    } else {
        tracing::info!("Successfully sent spans to Datadog");
    }
}
