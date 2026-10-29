package com.maddensgpt.aiplayers.core.action;

/**
 * The model called a function with arguments that cannot be used. The message is returned to the
 * model as the function result.
 */
public final class InvalidArgsException extends Exception {
    private static final long serialVersionUID = 1L;

    public InvalidArgsException(String message) {
        super(message);
    }
}
