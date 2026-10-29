package com.maddensgpt.aiplayers.core.ai;

import com.google.gson.JsonObject;

/**
 * Connection settings for an OpenAI-compatible Chat Completions endpoint.
 * OpenAI and DeepSeek both speak this format; only the base URL, model and
 * a few request fields differ. Public fields so Gson can read the config file.
 */
public final class ProviderConfig {
    /** Free-form label shown in logs, e.g. "openai" or "deepseek". */
    public String provider = "deepseek";

    /** Base URL without the trailing "/chat/completions". */
    public String baseUrl = "https://api.deepseek.com";

    /** Model ID sent in each request. Check the provider's docs for current IDs. */
    public String model = "deepseek-v4-flash";

    /** API key written directly in the config. Prefer {@link #apiKeyEnv} instead. */
    public String apiKey = "";

    /** Environment variable that holds the API key; used when {@link #apiKey} is blank. */
    public String apiKeyEnv = "DEEPSEEK_API_KEY";

    /** Sampling temperature, or null to use the provider's default (some models reject it). */
    public Double temperature = null;

    /** Upper bound on generated tokens per request. */
    public int maxOutputTokens = 1024;

    /**
     * Request field that carries {@link #maxOutputTokens}. OpenAI deprecated
     * "max_tokens" in favour of "max_completion_tokens"; DeepSeek uses "max_tokens".
     */
    public String maxTokensField = "max_tokens";

    /**
     * Send the model's reasoning_content back with its earlier assistant messages.
     * DeepSeek's thinking mode returns HTTP 400 on tool-calling requests without it.
     */
    public boolean replayReasoning = true;

    /** Extra top-level JSON fields merged into every request, e.g. {"thinking": {"type": "enabled"}}. */
    public JsonObject extraBody = new JsonObject();

    public int connectTimeoutSeconds = 10;
    public int requestTimeoutSeconds = 90;

    /** Retries for network errors, HTTP 429 and 5xx responses. */
    public int maxRetries = 2;

    public static ProviderConfig deepSeek() {
        return new ProviderConfig();
    }

    /** OpenAI preset. The model is left blank on purpose: set it to a current model ID. */
    public static ProviderConfig openAi() {
        ProviderConfig config = new ProviderConfig();
        config.provider = "openai";
        config.baseUrl = "https://api.openai.com/v1";
        config.model = "";
        config.apiKeyEnv = "OPENAI_API_KEY";
        config.maxTokensField = "max_completion_tokens";
        return config;
    }

    /** Resolves the key from {@link #apiKey} or the {@link #apiKeyEnv} environment variable. */
    public String resolveApiKey() {
        if (apiKey != null && !apiKey.isBlank()) {
            return apiKey.trim();
        }
        if (apiKeyEnv != null && !apiKeyEnv.isBlank()) {
            String fromEnv = System.getenv(apiKeyEnv.trim());
            if (fromEnv != null && !fromEnv.isBlank()) {
                return fromEnv.trim();
            }
        }
        return "";
    }

    /** Returns a list of problems that would stop requests from working; empty when usable. */
    public java.util.List<String> validate() {
        java.util.List<String> problems = new java.util.ArrayList<>();
        if (baseUrl == null || !(baseUrl.startsWith("https://") || baseUrl.startsWith("http://"))) {
            problems.add("baseUrl must start with https:// or http://");
        }
        if (model == null || model.isBlank()) {
            problems.add("model is not set; put a current model ID from your provider's docs in the config");
        }
        if (resolveApiKey().isEmpty()) {
            problems.add("no API key: set apiKey, or set the environment variable named by apiKeyEnv ("
                    + apiKeyEnv + ")");
        }
        if (maxOutputTokens <= 0) {
            problems.add("maxOutputTokens must be positive");
        }
        if (maxTokensField == null || maxTokensField.isBlank()) {
            problems.add("maxTokensField must be set (max_tokens or max_completion_tokens)");
        }
        return problems;
    }
}
