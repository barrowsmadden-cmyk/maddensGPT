package com.maddensgpt.aiplayers.core.brain;

import com.google.gson.annotations.SerializedName;
import com.maddensgpt.aiplayers.core.ai.ProviderConfig;
import com.maddensgpt.aiplayers.core.web.WebSettings;

import java.util.ArrayList;
import java.util.List;

/**
 * Everything configurable about the bots. Public fields so Gson can read the config file;
 * {@link #sanitize()} repairs missing or out-of-range values after loading.
 */
public final class BotSettings {
    public enum Chattiness {
        /** Speaks when something notable happens or when spoken to. */
        @SerializedName(value = "quiet", alternate = "QUIET") QUIET,
        /** Narrates new activities and reactions; skips turns with nothing new. */
        @SerializedName(value = "normal", alternate = "NORMAL") NORMAL,
        /** Thinks out loud most turns. */
        @SerializedName(value = "chatty", alternate = "CHATTY") CHATTY
    }

    public ProviderConfig provider = ProviderConfig.deepSeek();

    /** Internet access (web search and page reading). Off until a search API key is configured. */
    public WebSettings web = new WebSettings();

    /** Personality description given to the model. */
    public String personality = "Friendly, curious and a little goofy. Loves exploring and building, "
            + "gets nervous around creepers, and likes helping other players.";

    public Chattiness chattiness = Chattiness.NORMAL;

    /** Players whose requests the bot follows. Empty means anyone. Everyone can still chat with it. */
    public List<String> trustedPlayers = new ArrayList<>();

    /** Additional rules appended to the system prompt. */
    public String extraInstructions = "";

    /** Earlier turns kept in the conversation sent to the model. More context costs more tokens. */
    public int maxHistoryTurns = 10;

    /** Function calls accepted from one reply; extras are skipped. */
    public int maxActionsPerTurn = 8;

    /** Long-term notes kept per bot; the oldest is dropped when full. */
    public int maxNotes = 20;

    /** Minimum time between two API requests from one bot. */
    public double minSecondsBetweenRequests = 2.0;

    /** Hard cap on API requests per bot per rolling hour, to bound cost. */
    public int maxRequestsPerHour = 240;

    /** Pause before thinking again after a turn with no game actions (only talking or notes). */
    public int idleThinkSeconds = 20;

    /** A game action running longer than this is cancelled and reported as timed out. */
    public int actionTimeoutSeconds = 120;

    /** Stop the current action and respond when someone mentions the bot's name in chat. */
    public boolean interruptOnMention = true;

    /** Stop the current action and respond when the bot takes damage. */
    public boolean interruptWhenHurt = true;

    /** Minimum time between two interrupts, so constant damage cannot stop the bot every tick. */
    public int interruptCooldownSeconds = 8;

    /** Wake an idle bot for any chat message, not only ones that mention it. Costs more requests. */
    public boolean respondToAllChat = false;

    public int chatMaxMessageLength = 200;
    public int chatMinGapMillis = 1_200;
    public int chatMaxQueuedLines = 6;

    /** Chat lines and events kept for the next turn; older ones are dropped. */
    public int maxPendingChatLines = 20;
    public int maxPendingEvents = 20;

    /** Replaces null or out-of-range values (e.g. from a hand-edited config) with usable ones. */
    public BotSettings sanitize() {
        BotSettings defaults = new BotSettings();
        if (provider == null) {
            provider = defaults.provider;
        }
        if (provider.extraBody == null) {
            provider.extraBody = new com.google.gson.JsonObject();
        }
        web = WebSettings.sanitize(web);
        if (personality == null) {
            personality = "";
        }
        if (chattiness == null) {
            chattiness = defaults.chattiness;
        }
        if (trustedPlayers == null) {
            trustedPlayers = new ArrayList<>();
        }
        trustedPlayers.removeIf(name -> name == null || name.isBlank());
        if (extraInstructions == null) {
            extraInstructions = "";
        }
        maxHistoryTurns = clamp(maxHistoryTurns, 0, 100);
        maxActionsPerTurn = clamp(maxActionsPerTurn, 1, 32);
        maxNotes = clamp(maxNotes, 0, 200);
        if (!(minSecondsBetweenRequests >= 0)) {
            minSecondsBetweenRequests = defaults.minSecondsBetweenRequests;
        }
        maxRequestsPerHour = clamp(maxRequestsPerHour, 1, 100_000);
        idleThinkSeconds = clamp(idleThinkSeconds, 1, 3_600);
        actionTimeoutSeconds = clamp(actionTimeoutSeconds, 5, 3_600);
        interruptCooldownSeconds = clamp(interruptCooldownSeconds, 0, 3_600);
        chatMaxMessageLength = clamp(chatMaxMessageLength, 20, 256);
        chatMinGapMillis = clamp(chatMinGapMillis, 0, 60_000);
        chatMaxQueuedLines = clamp(chatMaxQueuedLines, 1, 50);
        maxPendingChatLines = clamp(maxPendingChatLines, 1, 200);
        maxPendingEvents = clamp(maxPendingEvents, 1, 200);
        return this;
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }
}
