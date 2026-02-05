use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Top-level request wrapper for sending spans to Datadog
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpansRequest {
    pub data: SpansRequestData,
}

/// Request data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpansRequestData {
    #[serde(rename = "type")]
    pub data_type: String, // Always "span"
    pub attributes: SpansPayload,
}

/// Payload containing the actual spans and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpansPayload {
    pub ml_app: String,
    pub spans: Vec<Span>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

/// Individual span representing an LLM operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    pub name: String,
    pub span_id: String,
    pub trace_id: String,
    pub parent_id: String, // "undefined" for root spans
    pub start_ns: u64,     // Nanoseconds since epoch
    pub duration: u64,     // Duration in nanoseconds
    pub meta: Meta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>, // "ok" or "error"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Metrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
}

/// Core content of a span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meta {
    pub kind: String, // "llm", "agent", "workflow", "tool", "task", "embedding", "retrieval"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<IO>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<IO>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<Error>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Input/Output data for a span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IO {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<Message>>,
}

/// Message in a conversation (for LLM spans)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String, // "system", "user", "assistant"
    pub content: String,
}

/// Error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Error {
    pub message: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub error_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
}

/// Metrics for LLM spans (token counts)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<f64>,
}

impl SpansRequest {
    /// Create a new spans request with a single span
    pub fn new(ml_app: String, span: Span) -> Self {
        Self {
            data: SpansRequestData {
                data_type: "span".to_string(),
                attributes: SpansPayload {
                    ml_app,
                    spans: vec![span],
                    tags: None,
                    session_id: None,
                },
            },
        }
    }

    /// Create a new spans request with multiple spans
    pub fn with_spans(ml_app: String, spans: Vec<Span>) -> Self {
        Self {
            data: SpansRequestData {
                data_type: "span".to_string(),
                attributes: SpansPayload {
                    ml_app,
                    spans,
                    tags: None,
                    session_id: None,
                },
            },
        }
    }
}

impl Span {
    /// Create a new LLM span
    pub fn new_llm_span(
        name: String,
        span_id: String,
        trace_id: String,
        parent_id: Option<String>,
        start_ns: u64,
        duration_ns: u64,
    ) -> Self {
        Self {
            name,
            span_id,
            trace_id,
            parent_id: parent_id.unwrap_or_else(|| "undefined".to_string()),
            start_ns,
            duration: duration_ns,
            meta: Meta {
                kind: "llm".to_string(),
                input: None,
                output: None,
                error: None,
                metadata: None,
            },
            status: Some("ok".to_string()),
            metrics: None,
            tags: None,
        }
    }

    /// Set input messages
    pub fn with_input_messages(mut self, messages: Vec<Message>) -> Self {
        self.meta.input = Some(IO {
            value: None,
            messages: Some(messages),
        });
        self
    }

    /// Set output messages
    pub fn with_output_messages(mut self, messages: Vec<Message>) -> Self {
        self.meta.output = Some(IO {
            value: None,
            messages: Some(messages),
        });
        self
    }

    /// Set error information
    pub fn with_error(mut self, error: Error) -> Self {
        self.meta.error = Some(error);
        self.status = Some("error".to_string());
        self
    }

    /// Set metadata (model parameters, etc.)
    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.meta.metadata = Some(metadata);
        self
    }

    /// Set metrics (token counts)
    pub fn with_metrics(mut self, metrics: Metrics) -> Self {
        self.metrics = Some(metrics);
        self
    }
}

impl Message {
    /// Create a new message
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

impl Error {
    /// Create a new error
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            error_type: None,
            stack: None,
        }
    }

    /// Create an error with type
    pub fn with_type(mut self, error_type: impl Into<String>) -> Self {
        self.error_type = Some(error_type.into());
        self
    }
}
