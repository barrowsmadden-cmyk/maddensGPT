package com.maddensgpt.aiplayers.core.brain;

import com.maddensgpt.aiplayers.core.world.Observation;

/**
 * What {@link BotBrain} needs from the game. Implemented by the Minecraft side; every method is
 * called on the server thread.
 */
public interface BotEnvironment {
    /** Captures what the bot can perceive right now. */
    Observation observe();

    /** Sends one line to the in-game chat as the bot, the way a player's message appears. */
    void sendChat(String line);

    /** Server log; the brain logs the model's private notes, errors and rate-limit pauses here. */
    void log(String message);

    /** Server log, warning level. */
    void warn(String message);
}
