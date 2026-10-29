package com.maddensgpt.aiplayers.core.action;

import java.util.Objects;

/**
 * Outcome of one function call, reported back to the model.
 *
 * @param success whether the action achieved what was asked
 * @param message what happened, written for the model (e.g. "Mined 5 oak_log, picked up 5.")
 */
public record ActionResult(boolean success, String message) {
    public ActionResult {
        Objects.requireNonNull(message, "message");
    }

    public static ActionResult ok(String message) {
        return new ActionResult(true, message);
    }

    public static ActionResult fail(String message) {
        return new ActionResult(false, message);
    }

    /** Text sent as the tool message content. */
    public String toToolContent() {
        return (success ? "OK: " : "FAILED: ") + message;
    }
}
