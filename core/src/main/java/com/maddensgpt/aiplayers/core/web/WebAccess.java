package com.maddensgpt.aiplayers.core.web;

import java.util.concurrent.CompletableFuture;

/**
 * Internet access for a bot. Both methods return text for the model and must not block the
 * calling thread; failures complete the future with a {@link WebException}.
 */
public interface WebAccess {
    /** Searches the web and returns the results formatted as text. */
    CompletableFuture<String> search(String query);

    /** Downloads a public web page and returns its text from character {@code start} on. */
    CompletableFuture<String> read(String url, int start);

    /** Whether {@link #read} is allowed by the settings. */
    boolean canReadPages();
}
