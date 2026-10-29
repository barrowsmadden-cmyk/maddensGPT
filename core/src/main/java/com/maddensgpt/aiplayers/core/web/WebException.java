package com.maddensgpt.aiplayers.core.web;

/**
 * A web search or page read that failed. The message is written for the model.
 */
public final class WebException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    public WebException(String message) {
        super(message);
    }

    public WebException(String message, Throwable cause) {
        super(message, cause);
    }
}
