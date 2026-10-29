package com.maddensgpt.aiplayers.core.ai;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.net.ProxySelector;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;

/**
 * Client for the Chat Completions API ({@code POST {baseUrl}/chat/completions}) as offered by
 * OpenAI and by OpenAI-compatible providers such as DeepSeek.
 */
public final class OpenAiCompatibleClient implements ChatClient, AutoCloseable {
    private static final Gson GSON = new GsonBuilder().serializeNulls().disableHtmlEscaping().create();
    private static final long MAX_RETRY_DELAY_MILLIS = 30_000;
    private static final int MAX_ERROR_DETAIL_CHARS = 300;

    private final ProviderConfig config;
    private final HttpClient http;
    private final long retryBaseDelayMillis;

    public OpenAiCompatibleClient(ProviderConfig config) {
        this(config, defaultHttpClient(config), 1_000);
    }

    OpenAiCompatibleClient(ProviderConfig config, HttpClient http, long retryBaseDelayMillis) {
        this.config = config;
        this.http = http;
        this.retryBaseDelayMillis = retryBaseDelayMillis;
    }

    private static HttpClient defaultHttpClient(ProviderConfig config) {
        HttpClient.Builder builder = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(Math.max(1, config.connectTimeoutSeconds)));
        // Honour the JVM's proxy settings (https.proxyHost etc.); HttpClient ignores them otherwise.
        ProxySelector proxySelector = ProxySelector.getDefault();
        if (proxySelector != null) {
            builder.proxy(proxySelector);
        }
        return builder.build();
    }

    @Override
    public CompletableFuture<ChatResponse> complete(List<ChatMessage> messages, List<ToolSpec> tools) {
        List<String> problems = config.validate();
        if (!problems.isEmpty()) {
            return CompletableFuture.failedFuture(
                    new AiException("Provider config is not usable: " + String.join("; ", problems), 0));
        }
        HttpRequest request = HttpRequest.newBuilder(endpoint("/chat/completions"))
                .timeout(Duration.ofSeconds(Math.max(1, config.requestTimeoutSeconds)))
                .header("Authorization", "Bearer " + config.resolveApiKey())
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(buildRequestBody(messages, tools)))
                .build();
        return sendWithRetry(request, 0).thenApply(OpenAiCompatibleClient::parseChatResponse);
    }

    /** Lists the model IDs the key can use ({@code GET {baseUrl}/models}), sorted. */
    public CompletableFuture<List<String>> listModels() {
        if (config.resolveApiKey().isEmpty()) {
            return CompletableFuture.failedFuture(new AiException("No API key configured", 0));
        }
        HttpRequest request = HttpRequest.newBuilder(endpoint("/models"))
                .timeout(Duration.ofSeconds(Math.max(1, config.requestTimeoutSeconds)))
                .header("Authorization", "Bearer " + config.resolveApiKey())
                .header("Accept", "application/json")
                .GET()
                .build();
        return sendWithRetry(request, 0).thenApply(OpenAiCompatibleClient::parseModelList);
    }

    @Override
    public void close() {
        http.shutdownNow();
    }

    String buildRequestBody(List<ChatMessage> messages, List<ToolSpec> tools) {
        JsonObject body = new JsonObject();
        body.addProperty("model", config.model.trim());

        JsonArray messageArray = new JsonArray();
        for (ChatMessage message : messages) {
            messageArray.add(message.toJson(config.replayReasoning));
        }
        body.add("messages", messageArray);

        // An empty "tools" array is rejected by some providers, so omit it entirely.
        if (!tools.isEmpty()) {
            JsonArray toolArray = new JsonArray();
            for (ToolSpec tool : tools) {
                toolArray.add(tool.toJson());
            }
            body.add("tools", toolArray);
        }

        body.addProperty(config.maxTokensField.trim(), config.maxOutputTokens);
        if (config.temperature != null) {
            body.addProperty("temperature", config.temperature);
        }
        if (config.extraBody != null) {
            for (Map.Entry<String, JsonElement> entry : config.extraBody.entrySet()) {
                body.add(entry.getKey(), entry.getValue().deepCopy());
            }
        }
        return GSON.toJson(body);
    }

    private URI endpoint(String path) {
        String base = config.baseUrl.trim();
        while (base.endsWith("/")) {
            base = base.substring(0, base.length() - 1);
        }
        return URI.create(base + path);
    }

    private CompletableFuture<String> sendWithRetry(HttpRequest request, int attempt) {
        return http.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .handle((response, error) -> {
                    if (error != null) {
                        Throwable cause = unwrap(error);
                        return retryOrFail(request, attempt,
                                new AiException("Request to " + config.provider + " failed: " + describe(cause), cause),
                                OptionalLong.empty());
                    }
                    if (response.statusCode() / 100 == 2) {
                        return CompletableFuture.completedFuture(response.body());
                    }
                    AiException failure = new AiException(
                            "HTTP " + response.statusCode() + " from " + config.provider + ": "
                                    + errorDetail(response.body()),
                            response.statusCode());
                    return retryOrFail(request, attempt, failure, retryAfterSeconds(response));
                })
                .thenCompose(next -> next);
    }

    private CompletableFuture<String> retryOrFail(HttpRequest request, int attempt, AiException failure,
                                                  OptionalLong retryAfterSeconds) {
        if (!failure.isRetryable() || attempt >= config.maxRetries) {
            return CompletableFuture.failedFuture(failure);
        }
        long delayMillis = retryAfterSeconds.isPresent()
                ? retryAfterSeconds.getAsLong() * 1_000
                : retryBaseDelayMillis << Math.min(attempt, 10);
        delayMillis = Math.min(MAX_RETRY_DELAY_MILLIS, Math.max(0, delayMillis));
        Executor delayed = CompletableFuture.delayedExecutor(delayMillis, TimeUnit.MILLISECONDS);
        return CompletableFuture.supplyAsync(() -> attempt + 1, delayed)
                .thenCompose(nextAttempt -> sendWithRetry(request, nextAttempt));
    }

    private static OptionalLong retryAfterSeconds(HttpResponse<String> response) {
        return response.headers().firstValue("Retry-After").map(value -> {
            try {
                return OptionalLong.of(Long.parseLong(value.trim()));
            } catch (NumberFormatException e) {
                return OptionalLong.empty(); // HTTP-date form; fall back to exponential backoff
            }
        }).orElse(OptionalLong.empty());
    }

    private static Throwable unwrap(Throwable error) {
        Throwable current = error;
        while (current instanceof CompletionException && current.getCause() != null) {
            current = current.getCause();
        }
        return current;
    }

    private static String describe(Throwable cause) {
        String name = cause instanceof IOException ? cause.getClass().getSimpleName() : cause.getClass().getName();
        return cause.getMessage() == null ? name : name + ": " + cause.getMessage();
    }

    /** Pulls error.message out of an OpenAI-style error body, falling back to the raw text. */
    static String errorDetail(String body) {
        if (body == null || body.isBlank()) {
            return "(empty body)";
        }
        try {
            JsonElement root = JsonParser.parseString(body);
            if (root.isJsonObject()) {
                JsonElement error = root.getAsJsonObject().get("error");
                if (error != null && error.isJsonObject()) {
                    String message = stringOrNull(error.getAsJsonObject(), "message");
                    if (message != null) {
                        return truncate(message);
                    }
                } else if (error != null && error.isJsonPrimitive()) {
                    return truncate(error.getAsString());
                }
            }
        } catch (JsonParseException | IllegalStateException e) {
            // not JSON; use the raw body
        }
        return truncate(body.strip());
    }

    private static String truncate(String text) {
        return text.length() <= MAX_ERROR_DETAIL_CHARS ? text : text.substring(0, MAX_ERROR_DETAIL_CHARS) + "...";
    }

    static ChatResponse parseChatResponse(String body) {
        try {
            JsonObject root = JsonParser.parseString(body).getAsJsonObject();
            JsonArray choices = root.has("choices") && root.get("choices").isJsonArray()
                    ? root.getAsJsonArray("choices") : null;
            if (choices == null || choices.isEmpty()) {
                throw new AiException("Response has no choices: " + errorDetail(body), -1);
            }
            JsonObject choice = choices.get(0).getAsJsonObject();
            JsonObject message = choice.getAsJsonObject("message");
            if (message == null) {
                throw new AiException("Response choice has no message", -1);
            }

            List<ToolCall> toolCalls = new ArrayList<>();
            JsonElement callsElement = message.get("tool_calls");
            if (callsElement != null && callsElement.isJsonArray()) {
                JsonArray calls = callsElement.getAsJsonArray();
                for (int i = 0; i < calls.size(); i++) {
                    JsonObject call = calls.get(i).getAsJsonObject();
                    JsonObject function = call.getAsJsonObject("function");
                    if (function == null) {
                        continue; // not a function call (e.g. a custom tool type we never offer)
                    }
                    String id = stringOrNull(call, "id");
                    String name = stringOrNull(function, "name");
                    JsonElement args = function.get("arguments");
                    String arguments = args == null || args.isJsonNull() ? ""
                            : args.isJsonPrimitive() ? args.getAsString() : args.toString();
                    toolCalls.add(new ToolCall(id != null ? id : "call_" + i, name != null ? name : "", arguments));
                }
            }

            ChatMessage assistant = ChatMessage.assistant(
                    textContent(message.get("content")),
                    stringOrNull(message, "reasoning_content"),
                    toolCalls);

            int promptTokens = -1;
            int completionTokens = -1;
            JsonElement usage = root.get("usage");
            if (usage != null && usage.isJsonObject()) {
                promptTokens = intOr(usage.getAsJsonObject(), "prompt_tokens", -1);
                completionTokens = intOr(usage.getAsJsonObject(), "completion_tokens", -1);
            }
            return new ChatResponse(assistant, stringOrNull(choice, "finish_reason"), promptTokens, completionTokens);
        } catch (JsonParseException | IllegalStateException | ClassCastException e) {
            throw new AiException("Could not parse response: " + e.getMessage(), e);
        }
    }

    static List<String> parseModelList(String body) {
        try {
            JsonObject root = JsonParser.parseString(body).getAsJsonObject();
            List<String> ids = new ArrayList<>();
            JsonElement data = root.get("data");
            if (data != null && data.isJsonArray()) {
                for (JsonElement entry : data.getAsJsonArray()) {
                    if (entry.isJsonObject()) {
                        String id = stringOrNull(entry.getAsJsonObject(), "id");
                        if (id != null) {
                            ids.add(id);
                        }
                    }
                }
            }
            ids.sort(String::compareTo);
            return ids;
        } catch (JsonParseException | IllegalStateException e) {
            throw new AiException("Could not parse model list: " + e.getMessage(), e);
        }
    }

    /** Content is normally a string, but tolerate the array-of-parts form. */
    private static String textContent(JsonElement content) {
        if (content == null || content.isJsonNull()) {
            return null;
        }
        if (content.isJsonPrimitive()) {
            return content.getAsString();
        }
        if (content.isJsonArray()) {
            StringBuilder text = new StringBuilder();
            for (JsonElement part : content.getAsJsonArray()) {
                if (part.isJsonObject()) {
                    String partText = stringOrNull(part.getAsJsonObject(), "text");
                    if (partText != null) {
                        text.append(partText);
                    }
                }
            }
            return text.isEmpty() ? null : text.toString();
        }
        return null;
    }

    private static String stringOrNull(JsonObject object, String key) {
        JsonElement value = object.get(key);
        return value != null && value.isJsonPrimitive() ? value.getAsString() : null;
    }

    private static int intOr(JsonObject object, String key, int fallback) {
        JsonElement value = object.get(key);
        if (value != null && value.isJsonPrimitive() && value.getAsJsonPrimitive().isNumber()) {
            return value.getAsInt();
        }
        return fallback;
    }
}
