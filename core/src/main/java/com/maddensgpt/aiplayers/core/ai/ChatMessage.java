package com.maddensgpt.aiplayers.core.ai;

import com.google.gson.JsonArray;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;

import java.util.List;
import java.util.Objects;

/**
 * A message in a Chat Completions conversation.
 */
public final class ChatMessage {
    public enum Role {
        SYSTEM("system"), USER("user"), ASSISTANT("assistant"), TOOL("tool");

        private final String wireName;

        Role(String wireName) {
            this.wireName = wireName;
        }

        public String wireName() {
            return wireName;
        }
    }

    private final Role role;
    private final String content;
    private final String reasoningContent;
    private final List<ToolCall> toolCalls;
    private final String toolCallId;

    private ChatMessage(Role role, String content, String reasoningContent, List<ToolCall> toolCalls,
                        String toolCallId) {
        this.role = Objects.requireNonNull(role, "role");
        this.content = content;
        this.reasoningContent = reasoningContent;
        this.toolCalls = toolCalls == null ? List.of() : List.copyOf(toolCalls);
        this.toolCallId = toolCallId;
    }

    public static ChatMessage system(String content) {
        return new ChatMessage(Role.SYSTEM, Objects.requireNonNull(content), null, null, null);
    }

    public static ChatMessage user(String content) {
        return new ChatMessage(Role.USER, Objects.requireNonNull(content), null, null, null);
    }

    public static ChatMessage assistant(String content, String reasoningContent, List<ToolCall> toolCalls) {
        return new ChatMessage(Role.ASSISTANT, content, reasoningContent, toolCalls, null);
    }

    public static ChatMessage toolResult(String toolCallId, String content) {
        return new ChatMessage(Role.TOOL, Objects.requireNonNull(content), null, null,
                Objects.requireNonNull(toolCallId));
    }

    public Role role() {
        return role;
    }

    /** Text content; null for an assistant message that only calls tools. */
    public String content() {
        return content;
    }

    /** Reasoning text returned by thinking models (DeepSeek's reasoning_content); usually null. */
    public String reasoningContent() {
        return reasoningContent;
    }

    public List<ToolCall> toolCalls() {
        return toolCalls;
    }

    public String toolCallId() {
        return toolCallId;
    }

    /** Serializes for a request body. Null content is written as JSON null, which the API accepts. */
    JsonObject toJson(boolean includeReasoning) {
        JsonObject json = new JsonObject();
        json.addProperty("role", role.wireName());
        if (content == null) {
            json.add("content", JsonNull.INSTANCE);
        } else {
            json.addProperty("content", content);
        }
        if (role == Role.ASSISTANT) {
            if (includeReasoning && reasoningContent != null) {
                json.addProperty("reasoning_content", reasoningContent);
            }
            if (!toolCalls.isEmpty()) {
                JsonArray calls = new JsonArray();
                for (ToolCall call : toolCalls) {
                    calls.add(call.toJson());
                }
                json.add("tool_calls", calls);
            }
        }
        if (role == Role.TOOL) {
            json.addProperty("tool_call_id", toolCallId);
        }
        return json;
    }
}
