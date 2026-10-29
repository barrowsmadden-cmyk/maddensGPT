package com.maddensgpt.aiplayers.core.ai;

/**
 * A failed API request. {@link #statusCode()} is -1 for network or parsing failures.
 */
public final class AiException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    private final int statusCode;

    public AiException(String message, int statusCode) {
        super(message);
        this.statusCode = statusCode;
    }

    public AiException(String message, Throwable cause) {
        super(message, cause);
        this.statusCode = -1;
    }

    public int statusCode() {
        return statusCode;
    }

    /** True for failures worth retrying: network errors, rate limits and server errors. */
    public boolean isRetryable() {
        return statusCode == -1 || statusCode == 408 || statusCode == 409 || statusCode == 429
                || statusCode >= 500;
    }
}
