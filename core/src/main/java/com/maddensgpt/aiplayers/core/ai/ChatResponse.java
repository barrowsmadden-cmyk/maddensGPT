package com.maddensgpt.aiplayers.core.ai;

import java.util.Objects;

/**
 * The model's reply to one request.
 *
 * @param message          the assistant message, including any tool calls
 * @param finishReason     e.g. "stop", "tool_calls", "length"; may be null
 * @param promptTokens     input tokens billed, or -1 if the provider did not report usage
 * @param completionTokens output tokens billed, or -1 if the provider did not report usage
 */
public record ChatResponse(ChatMessage message, String finishReason, int promptTokens, int completionTokens) {
    public ChatResponse {
        Objects.requireNonNull(message, "message");
    }
}
