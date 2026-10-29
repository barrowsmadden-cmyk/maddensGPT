package com.maddensgpt.aiplayers.core.action;

/**
 * A game action in progress, advanced once per server tick on the server thread.
 */
public interface RunningAction {
    /**
     * Advances the action by one tick.
     *
     * @return null while the action is still running, otherwise its final result
     */
    ActionResult tick();

    /**
     * Stops the action early (interrupt, timeout, or the bot being removed). Must release anything
     * the action holds, such as movement input or a block being mined. Not called after
     * {@link #tick()} has returned a result.
     */
    void cancel();

    /** Short description for status displays, e.g. "walking to 10 64 -3". */
    default String describe() {
        return getClass().getSimpleName();
    }
}
