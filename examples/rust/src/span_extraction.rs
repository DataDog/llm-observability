use opentelemetry::trace::TraceContextExt;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use uuid::Uuid;

/// Extract the span ID from the current OpenTelemetry span context
/// Returns None if no active span context is found
pub fn extract_span_id() -> Option<String> {
    let current_span = Span::current();
    let context = current_span.context();
    let span = context.span();
    let span_context = span.span_context();

    if span_context.is_valid() {
        Some(format!("{:x}", span_context.span_id()))
    } else {
        None
    }
}

/// Extract the trace ID from the current OpenTelemetry span context
/// Returns None if no active span context is found
pub fn extract_trace_id() -> Option<String> {
    let current_span = Span::current();
    let context = current_span.context();
    let span = context.span();
    let span_context = span.span_context();

    if span_context.is_valid() {
        Some(format!("{:x}", span_context.trace_id()))
    } else {
        None
    }
}

/// Generate a new unique span ID
pub fn generate_span_id() -> String {
    let uuid = Uuid::new_v4();
    format!("{:016x}", uuid.as_u128() & 0xFFFFFFFFFFFFFFFF)
}

/// Generate a new unique trace ID
pub fn generate_trace_id() -> String {
    let uuid = Uuid::new_v4();
    format!("{:032x}", uuid.as_u128())
}

/// Get the current timestamp in nanoseconds since Unix epoch
pub fn get_current_timestamp_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("System time before Unix epoch")
        .as_nanos() as u64
}

/// Get span ID from context or generate a new one
pub fn get_or_generate_span_id() -> String {
    extract_span_id().unwrap_or_else(generate_span_id)
}

/// Get trace ID from context or generate a new one
pub fn get_or_generate_trace_id() -> String {
    extract_trace_id().unwrap_or_else(generate_trace_id)
}
