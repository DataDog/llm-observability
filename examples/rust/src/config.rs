use std::env;

#[derive(Debug, Clone)]
pub struct DatadogLlmConfig {
    pub api_key: String,
    pub site: String,
    pub ml_app: String,
    pub enabled: bool,
    pub timeout_ms: u64,
}

impl DatadogLlmConfig {
    /// Load configuration from environment variables
    /// Returns None if required fields are missing or enabled is false
    pub fn from_env() -> Option<Self> {
        // Check if enabled first
        let enabled = env::var("DD_LLMOBS_ENABLED")
            .ok()
            .and_then(|v| match v.to_lowercase().as_str() {
                "1" | "true" | "yes" => Some(true),
                "0" | "false" | "no" | "" => Some(false),
                _ => None,
            })
            .unwrap_or(false);

        if !enabled {
            return None;
        }

        // API key is required
        let api_key = env::var("DD_API_KEY").ok()?;

        // Optional fields with defaults
        let site = env::var("DD_SITE").unwrap_or_else(|_| "datadoghq.com".to_string());
        let ml_app = env::var("DD_LLMOBS_ML_APP").unwrap_or_else(|_| "talkforge".to_string());
        let timeout_ms = env::var("DD_LLMOBS_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(5000);

        Some(Self {
            api_key,
            site,
            ml_app,
            enabled,
            timeout_ms,
        })
    }

    /// Get the full API endpoint URL
    pub fn endpoint_url(&self) -> String {
        format!(
            "https://api.{}/api/intake/llm-obs/v1/trace/spans",
            self.site
        )
    }
}
