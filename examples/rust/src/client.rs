use super::config::DatadogLlmConfig;
use super::models::SpansRequest;
use reqwest;
use std::time::Duration;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LlmObsError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Serialization failed: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("API returned error status: {status}, body: {body}")]
    ApiError { status: u16, body: String },

    #[error("Client not configured properly: {0}")]
    ConfigurationError(String),
}

#[derive(Clone)]
pub struct DatadogLlmClient {
    client: reqwest::Client,
    config: DatadogLlmConfig,
    endpoint: String,
}

impl DatadogLlmClient {
    /// Create a new Datadog LLM Observability client
    pub fn new(config: DatadogLlmConfig) -> Result<Self, LlmObsError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .build()
            .map_err(|e| {
                LlmObsError::ConfigurationError(format!("Failed to build HTTP client: {}", e))
            })?;

        let endpoint = config.endpoint_url();

        tracing::info!(
            endpoint = %endpoint,
            ml_app = %config.ml_app,
            "Datadog LLM Observability client initialized"
        );

        Ok(Self {
            client,
            endpoint,
            config,
        })
    }

    /// Send spans to Datadog LLM Observability API
    pub async fn send_spans(&self, request: SpansRequest) -> Result<(), LlmObsError> {
        tracing::debug!(
            span_count = request.data.attributes.spans.len(),
            ml_app = %self.config.ml_app,
            "Sending spans to Datadog LLM Observability"
        );

        let response = self
            .client
            .post(&self.endpoint)
            .header("DD-API-KEY", &self.config.api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        let status = response.status();

        if status.is_success() {
            tracing::debug!(
                status = status.as_u16(),
                "Successfully sent spans to Datadog"
            );
            Ok(())
        } else {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "Unable to read response body".to_string());

            tracing::warn!(
                status = status.as_u16(),
                body = %body,
                "Failed to send spans to Datadog"
            );

            Err(LlmObsError::ApiError {
                status: status.as_u16(),
                body,
            })
        }
    }
}
