package com.maddensgpt.aiplayers.core.ai;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.http.HttpClient;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class OpenAiCompatibleClientTest {
    private record Canned(int status, String body, String retryAfter) {
    }

    private record Seen(String method, String path, String authorization, String body) {
    }

    private HttpServer server;
    private final Deque<Canned> responses = new ArrayDeque<>();
    private final List<Seen> requests = new ArrayList<>();
    private final HttpClient http = HttpClient.newBuilder().proxy(HttpClient.Builder.NO_PROXY).build();

    @BeforeEach
    void startServer() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", exchange -> {
            String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            synchronized (requests) {
                requests.add(new Seen(exchange.getRequestMethod(), exchange.getRequestURI().getPath(),
                        exchange.getRequestHeaders().getFirst("Authorization"), body));
            }
            Canned canned;
            synchronized (responses) {
                canned = responses.isEmpty() ? new Canned(500, "{\"error\":{\"message\":\"no canned response\"}}", null)
                        : responses.removeFirst();
            }
            byte[] bytes = canned.body().getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            if (canned.retryAfter() != null) {
                exchange.getResponseHeaders().add("Retry-After", canned.retryAfter());
            }
            exchange.sendResponseHeaders(canned.status(), bytes.length);
            try (OutputStream out = exchange.getResponseBody()) {
                out.write(bytes);
            }
        });
        server.start();
    }

    @AfterEach
    void stopServer() {
        server.stop(0);
    }

    private ProviderConfig config() {
        ProviderConfig config = ProviderConfig.deepSeek();
        config.baseUrl = "http://127.0.0.1:" + server.getAddress().getPort() + "/v1/";
        config.apiKey = "test-key";
        config.maxRetries = 2;
        return config;
    }

    private OpenAiCompatibleClient client(ProviderConfig config) {
        return new OpenAiCompatibleClient(config, http, 1);
    }

    private void respond(int status, String body) {
        responses.add(new Canned(status, body, null));
    }

    private static String toolCallResponse() {
        return """
                {"id":"x","choices":[{"index":0,"finish_reason":"tool_calls","message":{
                  "role":"assistant","content":null,"reasoning_content":"I should get wood.",
                  "tool_calls":[
                    {"id":"call_a","type":"function","function":{"name":"say","arguments":"{\\"message\\":\\"getting wood\\"}"}},
                    {"id":"call_b","type":"function","function":{"name":"collect_blocks","arguments":"{\\"block\\":\\"oak_log\\",\\"count\\":4}"}}
                  ]}}],
                 "usage":{"prompt_tokens":120,"completion_tokens":30,"total_tokens":150}}
                """;
    }

    private static List<ToolSpec> tools() {
        JsonObject params = JsonParser.parseString(
                "{\"type\":\"object\",\"properties\":{\"message\":{\"type\":\"string\"}},\"required\":[\"message\"]}")
                .getAsJsonObject();
        return List.of(new ToolSpec("say", "Say something", params));
    }

    @Test
    void sendsChatCompletionRequestAndParsesToolCalls() throws Exception {
        respond(200, toolCallResponse());
        ProviderConfig config = config();
        config.extraBody.add("thinking", JsonParser.parseString("{\"type\":\"enabled\"}"));

        ChatResponse response = client(config).complete(
                List.of(ChatMessage.system("sys"), ChatMessage.user("status")), tools()).get(5, TimeUnit.SECONDS);

        Seen seen = requests.get(0);
        assertEquals("POST", seen.method());
        assertEquals("/v1/chat/completions", seen.path());
        assertEquals("Bearer test-key", seen.authorization());

        JsonObject body = JsonParser.parseString(seen.body()).getAsJsonObject();
        assertEquals("deepseek-v4-flash", body.get("model").getAsString());
        assertEquals(1024, body.get("max_tokens").getAsInt());
        assertFalse(body.has("temperature"), "temperature is omitted when null");
        assertEquals("enabled", body.getAsJsonObject("thinking").get("type").getAsString());
        JsonArray messages = body.getAsJsonArray("messages");
        assertEquals("system", messages.get(0).getAsJsonObject().get("role").getAsString());
        assertEquals("status", messages.get(1).getAsJsonObject().get("content").getAsString());
        JsonObject tool = body.getAsJsonArray("tools").get(0).getAsJsonObject();
        assertEquals("function", tool.get("type").getAsString());
        assertEquals("say", tool.getAsJsonObject("function").get("name").getAsString());

        assertEquals("tool_calls", response.finishReason());
        assertEquals(120, response.promptTokens());
        assertEquals(30, response.completionTokens());
        ChatMessage message = response.message();
        assertNull(message.content());
        assertEquals("I should get wood.", message.reasoningContent());
        assertEquals(2, message.toolCalls().size());
        assertEquals(new ToolCall("call_a", "say", "{\"message\":\"getting wood\"}"), message.toolCalls().get(0));
        assertEquals("collect_blocks", message.toolCalls().get(1).name());
    }

    @Test
    void replaysAssistantMessageWithReasoningAndToolResults() throws Exception {
        respond(200, toolCallResponse());
        respond(200, "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}");
        OpenAiCompatibleClient client = client(config());

        ChatMessage first = client.complete(List.of(ChatMessage.user("hi")), tools()).get(5, TimeUnit.SECONDS).message();
        List<ChatMessage> next = List.of(ChatMessage.user("hi"), first,
                ChatMessage.toolResult("call_a", "OK: Said in chat."),
                ChatMessage.toolResult("call_b", "OK: Mined 4 oak_log."));
        ChatResponse second = client.complete(next, tools()).get(5, TimeUnit.SECONDS);
        assertEquals("ok", second.message().content());

        JsonArray messages = JsonParser.parseString(requests.get(1).body()).getAsJsonObject().getAsJsonArray("messages");
        JsonObject assistant = messages.get(1).getAsJsonObject();
        assertEquals("assistant", assistant.get("role").getAsString());
        assertTrue(assistant.has("content") && assistant.get("content").isJsonNull(), "null content is sent as null");
        assertEquals("I should get wood.", assistant.get("reasoning_content").getAsString());
        JsonObject call = assistant.getAsJsonArray("tool_calls").get(1).getAsJsonObject();
        assertEquals("call_b", call.get("id").getAsString());
        assertEquals("function", call.get("type").getAsString());
        assertEquals("{\"block\":\"oak_log\",\"count\":4}",
                call.getAsJsonObject("function").get("arguments").getAsString());
        JsonObject result = messages.get(3).getAsJsonObject();
        assertEquals("tool", result.get("role").getAsString());
        assertEquals("call_b", result.get("tool_call_id").getAsString());
        assertEquals("OK: Mined 4 oak_log.", result.get("content").getAsString());
    }

    @Test
    void omitsReasoningWhenReplayIsOff() {
        ProviderConfig config = config();
        config.replayReasoning = false;
        String body = client(config).buildRequestBody(
                List.of(ChatMessage.assistant("hi", "secret thoughts", List.of())), List.of());
        JsonObject assistant = JsonParser.parseString(body).getAsJsonObject().getAsJsonArray("messages")
                .get(0).getAsJsonObject();
        assertFalse(assistant.has("reasoning_content"));
        assertFalse(JsonParser.parseString(body).getAsJsonObject().has("tools"), "empty tools are omitted");
    }

    @Test
    void openAiPresetUsesMaxCompletionTokensAndTemperatureWhenSet() {
        ProviderConfig config = ProviderConfig.openAi();
        config.model = "some-model";
        config.temperature = 0.7;
        JsonObject body = JsonParser.parseString(client(config).buildRequestBody(List.of(ChatMessage.user("x")),
                List.of())).getAsJsonObject();
        assertEquals(1024, body.get("max_completion_tokens").getAsInt());
        assertFalse(body.has("max_tokens"));
        assertEquals(0.7, body.get("temperature").getAsDouble());
    }

    @Test
    void retriesRateLimitsAndServerErrors() throws Exception {
        responses.add(new Canned(429, "{\"error\":{\"message\":\"slow down\"}}", "0"));
        respond(503, "overloaded");
        respond(200, "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"done\"}}]}");

        ChatResponse response = client(config()).complete(List.of(ChatMessage.user("x")), List.of())
                .get(5, TimeUnit.SECONDS);
        assertEquals("done", response.message().content());
        assertEquals(3, requests.size());
        assertEquals(-1, response.promptTokens(), "usage missing is reported as -1");
    }

    @Test
    void givesUpAfterMaxRetries() {
        respond(500, "{\"error\":{\"message\":\"boom\"}}");
        respond(500, "{\"error\":{\"message\":\"boom\"}}");
        respond(500, "{\"error\":{\"message\":\"boom\"}}");
        respond(200, "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"too late\"}}]}");

        ExecutionException e = assertThrows(ExecutionException.class,
                () -> client(config()).complete(List.of(ChatMessage.user("x")), List.of()).get(5, TimeUnit.SECONDS));
        AiException cause = assertInstanceOf(AiException.class, e.getCause());
        assertEquals(500, cause.statusCode());
        assertEquals(3, requests.size(), "1 attempt + 2 retries");
    }

    @Test
    void doesNotRetryAuthErrorsAndReportsProviderMessage() {
        respond(401, "{\"error\":{\"message\":\"Invalid API key\",\"type\":\"authentication_error\"}}");

        ExecutionException e = assertThrows(ExecutionException.class,
                () -> client(config()).complete(List.of(ChatMessage.user("x")), List.of()).get(5, TimeUnit.SECONDS));
        AiException cause = assertInstanceOf(AiException.class, e.getCause());
        assertEquals(401, cause.statusCode());
        assertFalse(cause.isRetryable());
        assertTrue(cause.getMessage().contains("Invalid API key"), cause.getMessage());
        assertFalse(cause.getMessage().contains("test-key"), "the key never appears in errors");
        assertEquals(1, requests.size());
    }

    @Test
    void rejectsUnusableConfigWithoutSendingAnything() {
        ProviderConfig config = ProviderConfig.openAi();
        config.apiKey = "k";
        ExecutionException e = assertThrows(ExecutionException.class,
                () -> client(config).complete(List.of(ChatMessage.user("x")), List.of()).get(5, TimeUnit.SECONDS));
        AiException cause = assertInstanceOf(AiException.class, e.getCause());
        assertEquals(0, cause.statusCode());
        assertTrue(cause.getMessage().contains("model is not set"), cause.getMessage());
        assertTrue(requests.isEmpty());
    }

    @Test
    void reportsMalformedResponses() {
        respond(200, "{\"choices\":[]}");
        ExecutionException e = assertThrows(ExecutionException.class,
                () -> client(config()).complete(List.of(ChatMessage.user("x")), List.of()).get(5, TimeUnit.SECONDS));
        assertInstanceOf(AiException.class, e.getCause());
    }

    @Test
    void toleratesMissingIdsObjectArgumentsAndContentParts() {
        ChatResponse response = OpenAiCompatibleClient.parseChatResponse("""
                {"choices":[{"message":{"role":"assistant",
                  "content":[{"type":"text","text":"part one, "},{"type":"text","text":"part two"}],
                  "tool_calls":[{"type":"function","function":{"name":"wait","arguments":{"seconds":5}}}]}}]}
                """);
        assertEquals("part one, part two", response.message().content());
        ToolCall call = response.message().toolCalls().get(0);
        assertEquals("call_0", call.id());
        assertEquals("{\"seconds\":5}", call.arguments());
    }

    @Test
    void listsModels() throws Exception {
        respond(200, "{\"object\":\"list\",\"data\":[{\"id\":\"zeta\"},{\"id\":\"alpha\"}]}");
        List<String> models = client(config()).listModels().get(5, TimeUnit.SECONDS);
        assertEquals(List.of("alpha", "zeta"), models);
        assertEquals("GET", requests.get(0).method());
        assertEquals("/v1/models", requests.get(0).path());
    }

    @Test
    void readsApiKeyFromConfiguredEnvironmentVariableName() {
        ProviderConfig config = ProviderConfig.deepSeek();
        config.apiKey = "";
        config.apiKeyEnv = "AIPLAYERS_TEST_SURELY_UNSET_VARIABLE";
        assertEquals("", config.resolveApiKey());
        assertTrue(config.validate().stream().anyMatch(problem -> problem.contains("no API key")));
        config.apiKey = "  direct  ";
        assertEquals("direct", config.resolveApiKey());
    }
}
