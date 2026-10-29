package com.maddensgpt.aiplayers.core.ai;

import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Sends a conversation to a model. Implementations must not block the calling thread.
 */
public interface ChatClient {
    /**
     * @return a future that completes with the reply, or exceptionally with {@link AiException}
     */
    CompletableFuture<ChatResponse> complete(List<ChatMessage> messages, List<ToolSpec> tools);
}
