package com.maddensgpt.aiplayers.core.brain;

import com.maddensgpt.aiplayers.core.action.ActionArgs;
import com.maddensgpt.aiplayers.core.action.ActionCatalog;
import com.maddensgpt.aiplayers.core.action.ActionHandler;
import com.maddensgpt.aiplayers.core.action.ActionResult;
import com.maddensgpt.aiplayers.core.action.InvalidArgsException;
import com.maddensgpt.aiplayers.core.action.RunningAction;
import com.maddensgpt.aiplayers.core.ai.AiException;
import com.maddensgpt.aiplayers.core.ai.ChatClient;
import com.maddensgpt.aiplayers.core.ai.ChatMessage;
import com.maddensgpt.aiplayers.core.ai.ChatResponse;
import com.maddensgpt.aiplayers.core.ai.ToolCall;
import com.maddensgpt.aiplayers.core.ai.ToolSpec;
import com.maddensgpt.aiplayers.core.chat.ChatPacer;
import com.maddensgpt.aiplayers.core.world.Observation;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.CompletionException;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Pattern;

/**
 * The think-act loop for one bot.
 *
 * <p>Each turn the brain sends the conversation plus a fresh status report to the model, then
 * runs the function calls it gets back in order (one game action at a time, over as many ticks
 * as it takes), and reports every result on the next turn. Talking happens through the
 * {@code say} function, paced by {@link ChatPacer} so it reads like a person typing.
 *
 * <p>Not thread-safe: call every method on the server thread. Only the API response is handed
 * over from the HTTP thread, through an atomic reference that {@link #tick(long)} drains.
 */
public final class BotBrain {
    public enum State { IDLE, THINKING, ACTING, PAUSED }

    /** Snapshot for status commands. */
    public record Status(State state, String goal, String plan, int notes, String currentAction,
                         int requestsLastHour, int totalRequests, long promptTokens, long completionTokens,
                         String lastError) {
    }

    private static final long HOUR_MILLIS = 3_600_000L;
    private static final int MAX_IDLE_BACKOFF_SHIFT = 3;
    private static final int BAD_REQUEST_RESETS_AFTER = 3;

    private static final class Turn {
        final String fullMessage;
        final String briefMessage;
        final List<String> consumedChat;
        final List<String> consumedEvents;
        ChatMessage assistant;
        final List<ChatMessage> results = new ArrayList<>();
        final List<ToolCall> overflow = new ArrayList<>();

        Turn(String fullMessage, String briefMessage, List<String> consumedChat, List<String> consumedEvents) {
            this.fullMessage = fullMessage;
            this.briefMessage = briefMessage;
            this.consumedChat = consumedChat;
            this.consumedEvents = consumedEvents;
        }
    }

    private record Outcome(int generation, ChatResponse response, Throwable error) {
    }

    private final class WaitAction implements RunningAction {
        private final long until;
        private final int seconds;

        WaitAction(int seconds) {
            this.seconds = seconds;
            this.until = now + seconds * 1_000L;
        }

        @Override
        public ActionResult tick() {
            return now >= until ? ActionResult.ok("Waited " + seconds + " seconds.") : null;
        }

        @Override
        public void cancel() {
        }

        @Override
        public String describe() {
            return "waiting " + seconds + "s";
        }
    }

    private final String botName;
    private final BotSettings settings;
    private final ChatClient client;
    private final ActionHandler actions;
    private final BotEnvironment env;
    private final ChatPacer pacer;
    private final List<ToolSpec> tools = ActionCatalog.all();
    private final String systemPrompt;
    private final Pattern mentionPattern;

    // Conversation
    private final Deque<Turn> history = new ArrayDeque<>();
    private Turn current;

    // Memory the bot keeps across turns
    private String goal = "";
    private String plan = "";
    private final List<String> notes = new ArrayList<>();

    // Input gathered between turns
    private final Deque<String> pendingChat = new ArrayDeque<>();
    private final Deque<String> pendingEvents = new ArrayDeque<>();
    private boolean wakeRequested;
    private String interruptReason;

    // Scheduling and accounting
    private State state = State.IDLE;
    private long now;
    private long nextThinkAt;
    private long lastRequestAt = Long.MIN_VALUE;
    private long lastInterruptAt = Long.MIN_VALUE;
    private final Deque<Long> requestTimes = new ArrayDeque<>();
    private boolean rateLimitWarned;
    private int idleStreak;
    private int consecutiveFailures;
    private int consecutiveBadRequests;
    private String lastError = "";
    private int totalRequests;
    private long totalPromptTokens;
    private long totalCompletionTokens;

    // In-flight request
    private int generation;
    private final AtomicReference<Outcome> arrived = new AtomicReference<>();

    // Running the current turn's calls
    private final Deque<ToolCall> queuedCalls = new ArrayDeque<>();
    private ToolCall runningCall;
    private RunningAction running;
    private long runningDeadline;
    private boolean turnFailed;
    private boolean turnUsedWorldTime;

    public BotBrain(String botName, BotSettings settings, ChatClient client, ActionHandler actions,
                    BotEnvironment env) {
        this.botName = Objects.requireNonNull(botName, "botName");
        this.settings = Objects.requireNonNull(settings, "settings").sanitize();
        this.client = Objects.requireNonNull(client, "client");
        this.actions = Objects.requireNonNull(actions, "actions");
        this.env = Objects.requireNonNull(env, "env");
        this.pacer = new ChatPacer(botName, settings.chatMaxMessageLength, settings.chatMinGapMillis,
                settings.chatMaxQueuedLines);
        this.systemPrompt = PromptBuilder.systemPrompt(botName, settings);
        this.mentionPattern = Pattern.compile("(?<![A-Za-z0-9_])" + Pattern.quote(botName) + "(?![A-Za-z0-9_])",
                Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE);
    }

    // ------------------------------------------------------------------ inputs

    /**
     * A chat message seen by the bot. The bot's own messages are ignored.
     *
     * @param direct true for a private message (/msg) to the bot
     */
    public void onChat(String sender, String message, boolean direct) {
        if (sender == null || message == null || sender.equalsIgnoreCase(botName)) {
            return;
        }
        boolean toBot = direct || mentionPattern.matcher(message).find();
        String line = (direct ? "[private] " : "") + "<" + sender + "> " + message.strip()
                + (toBot && !direct ? "  (mentions you)" : "");
        addBounded(pendingChat, line, settings.maxPendingChatLines);
        if (toBot) {
            wakeRequested = true;
            if (settings.interruptOnMention) {
                requestInterrupt(sender + " is talking to you");
            }
        } else if (settings.respondToAllChat) {
            wakeRequested = true;
        }
    }

    /** Something that happened to or near the bot, e.g. "Picked up 3 oak_log" or "It is getting dark". */
    public void onEvent(String event) {
        if (event != null && !event.isBlank()) {
            addBounded(pendingEvents, event.strip(), settings.maxPendingEvents);
        }
    }

    /** The bot took damage. Wakes it and, if enabled, interrupts the current action. */
    public void onHurt(String description) {
        onEvent(description);
        wakeRequested = true;
        if (settings.interruptWhenHurt) {
            requestInterrupt(description);
        }
    }

    /** Sets the long-term goal given by a player or operator; "" clears it. */
    public void setGoal(String newGoal) {
        goal = newGoal == null ? "" : newGoal.strip();
        onEvent(goal.isEmpty() ? "Your goal was cleared." : "You were given a new goal: " + goal);
        wakeRequested = true;
        requestInterruptIgnoringCooldown("your goal changed");
    }

    /** Stops thinking and acting until {@link #resume()}. Any request in flight is discarded. */
    public void pause() {
        abandonTurn();
        state = State.PAUSED;
    }

    public void resume() {
        if (state == State.PAUSED) {
            state = State.IDLE;
            nextThinkAt = now;
            idleStreak = 0;
        }
    }

    /** Forgets the conversation history (keeps goal, plan and notes). */
    public void resetConversation() {
        boolean wasPaused = state == State.PAUSED;
        abandonTurn();
        history.clear();
        state = wasPaused ? State.PAUSED : State.IDLE;
        nextThinkAt = now;
    }

    /** Cancels everything; call when the bot is removed or the server stops. */
    public void shutdown() {
        abandonTurn();
        pacer.clear();
        state = State.PAUSED;
    }

    // ------------------------------------------------------------------ loop

    /** Advances the brain. Call once per server tick with the current wall-clock time. */
    public void tick(long nowMillis) {
        now = nowMillis;
        pacer.poll(now).ifPresent(env::sendChat);

        Outcome outcome = arrived.getAndSet(null);
        if (outcome != null && outcome.generation() == generation && state == State.THINKING) {
            if (outcome.error() != null) {
                onRequestFailed(outcome.error());
            } else {
                onResponse(outcome.response());
            }
        }

        switch (state) {
            case IDLE -> maybeThink();
            case ACTING -> stepActions();
            case THINKING, PAUSED -> {
            }
        }
    }

    private void maybeThink() {
        if (wakeRequested) {
            wakeRequested = false;
            idleStreak = 0;
            nextThinkAt = Math.min(nextThinkAt, now);
        }
        if (now < nextThinkAt) {
            return;
        }
        long minGap = (long) (settings.minSecondsBetweenRequests * 1_000);
        if (lastRequestAt != Long.MIN_VALUE && now < lastRequestAt + minGap) {
            nextThinkAt = lastRequestAt + minGap;
            return;
        }
        while (!requestTimes.isEmpty() && requestTimes.peekFirst() <= now - HOUR_MILLIS) {
            requestTimes.removeFirst();
        }
        if (requestTimes.size() >= settings.maxRequestsPerHour) {
            nextThinkAt = requestTimes.peekFirst() + HOUR_MILLIS;
            if (!rateLimitWarned) {
                rateLimitWarned = true;
                env.warn(botName + " reached maxRequestsPerHour (" + settings.maxRequestsPerHour
                        + "); pausing until the hourly window frees up.");
            }
            return;
        }
        rateLimitWarned = false;
        startThinking();
    }

    private void startThinking() {
        Observation observation;
        try {
            observation = Objects.requireNonNull(env.observe(), "observe() returned null");
        } catch (RuntimeException e) {
            env.warn(botName + ": could not observe the world: " + e);
            nextThinkAt = now + 5_000;
            return;
        }

        List<String> chat = new ArrayList<>(pendingChat);
        List<String> events = new ArrayList<>(pendingEvents);
        pendingChat.clear();
        pendingEvents.clear();

        String full = PromptBuilder.turnMessage(observation.render(), goal, plan, List.copyOf(notes), events, chat,
                history.isEmpty() ? "This is your first turn. Say hi if anyone is around, then get going." : null);
        String brief = PromptBuilder.turnMessage(observation.renderBrief(), null, null, null, events, chat, null);
        current = new Turn(full, brief, chat, events);

        List<ChatMessage> messages = new ArrayList<>();
        messages.add(ChatMessage.system(systemPrompt));
        for (Turn turn : history) {
            messages.add(ChatMessage.user(turn.briefMessage));
            messages.add(turn.assistant);
            messages.addAll(turn.results);
        }
        messages.add(ChatMessage.user(full));

        state = State.THINKING;
        lastRequestAt = now;
        requestTimes.addLast(now);
        totalRequests++;
        int requestGeneration = ++generation;
        try {
            client.complete(messages, tools).whenComplete((response, error) ->
                    arrived.set(new Outcome(requestGeneration, response, error)));
        } catch (RuntimeException e) {
            arrived.set(new Outcome(requestGeneration, null, e));
        }
    }

    private void onResponse(ChatResponse response) {
        consecutiveFailures = 0;
        consecutiveBadRequests = 0;
        lastError = "";
        if (response.promptTokens() > 0) {
            totalPromptTokens += response.promptTokens();
        }
        if (response.completionTokens() > 0) {
            totalCompletionTokens += response.completionTokens();
        }

        ChatMessage assistant = response.message();
        current.assistant = assistant;
        if (assistant.content() != null && !assistant.content().isBlank()) {
            env.log(botName + " (private): " + assistant.content().strip());
        }

        List<ToolCall> calls = assistant.toolCalls();
        int accepted = Math.min(calls.size(), settings.maxActionsPerTurn);
        queuedCalls.clear();
        queuedCalls.addAll(calls.subList(0, accepted));
        current.overflow.addAll(calls.subList(accepted, calls.size()));
        turnFailed = false;
        turnUsedWorldTime = false;
        interruptReason = null;

        if (calls.isEmpty()) {
            finishTurn();
        } else {
            state = State.ACTING;
            stepActions();
        }
    }

    private void onRequestFailed(Throwable error) {
        Throwable cause = error instanceof CompletionException && error.getCause() != null ? error.getCause() : error;
        consecutiveFailures++;
        lastError = cause.getMessage() == null ? cause.toString() : cause.getMessage();
        env.warn(botName + ": AI request failed (" + consecutiveFailures + " in a row): " + lastError);

        // Put the turn's chat and events back so the next attempt still sees them.
        if (current != null) {
            List<String> chatBack = new ArrayList<>(current.consumedChat);
            chatBack.addAll(pendingChat);
            pendingChat.clear();
            chatBack.forEach(line -> addBounded(pendingChat, line, settings.maxPendingChatLines));
            List<String> eventsBack = new ArrayList<>(current.consumedEvents);
            eventsBack.addAll(pendingEvents);
            pendingEvents.clear();
            eventsBack.forEach(event -> addBounded(pendingEvents, event, settings.maxPendingEvents));
        }
        current = null;
        state = State.IDLE;

        int status = cause instanceof AiException ai ? ai.statusCode() : -1;
        long delay;
        if (status == 0 || status == 401 || status == 403 || status == 404) {
            delay = 60_000; // configuration problem: retrying quickly will not help
        } else if (status == 400 || status == 422) {
            delay = 10_000;
            if (++consecutiveBadRequests >= BAD_REQUEST_RESETS_AFTER && !history.isEmpty()) {
                env.warn(botName + ": the provider keeps rejecting the conversation; starting a fresh one.");
                history.clear();
                consecutiveBadRequests = 0;
            }
        } else {
            delay = Math.min(120_000L, 5_000L << Math.min(consecutiveFailures - 1, 5));
        }
        nextThinkAt = now + delay;
    }

    private void stepActions() {
        if (interruptReason != null) {
            String reason = interruptReason;
            interruptReason = null;
            lastInterruptAt = now;
            if (running != null) {
                safeCancel(running);
                addResult(runningCall, ActionResult.fail("Interrupted because " + reason + "."));
                running = null;
                runningCall = null;
            }
            while (!queuedCalls.isEmpty()) {
                addResult(queuedCalls.pollFirst(), ActionResult.fail("Skipped: interrupted because " + reason + "."));
            }
            turnUsedWorldTime = true; // think again right away
            finishTurn();
            return;
        }

        if (running != null) {
            ActionResult result;
            if (now >= runningDeadline) {
                safeCancel(running);
                result = ActionResult.fail("Timed out after " + settings.actionTimeoutSeconds + " seconds.");
            } else {
                try {
                    result = running.tick();
                } catch (RuntimeException e) {
                    env.warn(botName + ": action " + runningCall.name() + " crashed: " + e);
                    safeCancel(running);
                    result = ActionResult.fail("Something went wrong: " + e.getMessage());
                }
            }
            if (result == null) {
                return;
            }
            addResult(runningCall, result);
            turnFailed |= !result.success();
            running = null;
            runningCall = null;
        }

        while (running == null && !queuedCalls.isEmpty()) {
            ToolCall call = queuedCalls.pollFirst();
            if (turnFailed) {
                addResult(call, ActionResult.fail("Skipped because an earlier action this turn failed."));
                continue;
            }
            startCall(call);
        }
        if (running == null && queuedCalls.isEmpty()) {
            finishTurn();
        }
    }

    private void startCall(ToolCall call) {
        ActionResult immediate;
        try {
            ActionArgs args = ActionArgs.parse(call.arguments());
            immediate = switch (call.name()) {
                case ActionCatalog.SAY -> {
                    int lines = pacer.enqueue(args.getString("message"), now);
                    yield lines > 0 ? ActionResult.ok("Said in chat.")
                            : ActionResult.ok("Nothing was sent: the message was empty or you are talking too much.");
                }
                case ActionCatalog.REMEMBER -> {
                    if (settings.maxNotes == 0) {
                        yield ActionResult.fail("Notes are disabled on this server.");
                    }
                    notes.add(args.getString("note"));
                    while (notes.size() > settings.maxNotes) {
                        notes.remove(0);
                    }
                    yield ActionResult.ok("Saved to notes.");
                }
                case ActionCatalog.SET_PLAN -> {
                    plan = args.optString("plan").orElse("");
                    yield ActionResult.ok(plan.isEmpty() ? "Plan cleared." : "Plan updated.");
                }
                case ActionCatalog.WAIT -> {
                    begin(call, new WaitAction(args.getInt("seconds", 1, 60)));
                    yield null;
                }
                default -> {
                    if (!ActionCatalog.isKnown(call.name())) {
                        yield ActionResult.fail("There is no function named '" + call.name() + "'.");
                    }
                    RunningAction action = actions.start(call.name(), args);
                    if (action == null) {
                        yield ActionResult.fail(call.name() + " is not available right now.");
                    }
                    begin(call, action);
                    yield null;
                }
            };
        } catch (InvalidArgsException e) {
            immediate = ActionResult.fail("Invalid arguments for " + call.name() + ": " + e.getMessage());
        } catch (RuntimeException e) {
            env.warn(botName + ": could not start " + call.name() + ": " + e);
            immediate = ActionResult.fail("Could not start " + call.name() + ": " + e.getMessage());
        }
        if (immediate != null) {
            addResult(call, immediate);
            turnFailed |= !immediate.success();
        }
    }

    private void begin(ToolCall call, RunningAction action) {
        runningCall = call;
        running = action;
        long timeout = settings.actionTimeoutSeconds * 1_000L;
        if (action instanceof WaitAction wait) {
            timeout = Math.max(timeout, wait.until - now + 5_000);
        }
        runningDeadline = now + timeout;
        turnUsedWorldTime = true;
    }

    private void finishTurn() {
        for (ToolCall extra : current.overflow) {
            addResult(extra, ActionResult.fail("Skipped: at most " + settings.maxActionsPerTurn
                    + " actions per turn."));
        }
        history.addLast(current);
        while (history.size() > settings.maxHistoryTurns) {
            history.removeFirst();
        }
        current = null;
        state = State.IDLE;
        interruptReason = null;
        if (turnUsedWorldTime || turnFailed) {
            // Something happened, or the model made a mistake it should get to correct: think again soon.
            idleStreak = 0;
            nextThinkAt = now;
        } else {
            // Nothing happened in the world, so back off: 1x, 2x, 4x, 8x the idle delay.
            nextThinkAt = now + ((settings.idleThinkSeconds * 1_000L) << Math.min(idleStreak, MAX_IDLE_BACKOFF_SHIFT));
            idleStreak++;
        }
    }

    private void addResult(ToolCall call, ActionResult result) {
        current.results.add(ChatMessage.toolResult(call.id(), result.toToolContent()));
    }

    private void requestInterrupt(String reason) {
        if (lastInterruptAt != Long.MIN_VALUE && now - lastInterruptAt < settings.interruptCooldownSeconds * 1_000L) {
            return;
        }
        requestInterruptIgnoringCooldown(reason);
    }

    private void requestInterruptIgnoringCooldown(String reason) {
        // Only a turn that is acting can be interrupted; while thinking, the news waits for the next turn.
        if (state == State.ACTING && interruptReason == null) {
            interruptReason = reason;
        }
    }

    /** Drops the current turn without adding it to history, keeping the conversation consistent. */
    private void abandonTurn() {
        generation++;
        if (running != null) {
            safeCancel(running);
        }
        running = null;
        runningCall = null;
        queuedCalls.clear();
        current = null;
        interruptReason = null;
        if (state != State.PAUSED) {
            state = State.IDLE;
        }
    }

    private void safeCancel(RunningAction action) {
        try {
            action.cancel();
        } catch (RuntimeException e) {
            env.warn(botName + ": error cancelling action: " + e);
        }
    }

    private static void addBounded(Deque<String> deque, String value, int max) {
        deque.addLast(value);
        while (deque.size() > max) {
            deque.removeFirst();
        }
    }

    // ------------------------------------------------------------------ queries

    public String botName() {
        return botName;
    }

    public State state() {
        return state;
    }

    public String goal() {
        return goal;
    }

    public List<String> notes() {
        return List.copyOf(notes);
    }

    public Status status() {
        int lastHour = 0;
        for (long time : requestTimes) {
            if (time > now - HOUR_MILLIS) {
                lastHour++;
            }
        }
        String action = running != null ? running.describe()
                : state == State.THINKING ? "thinking" : state.name().toLowerCase(Locale.ROOT);
        return new Status(state, goal, plan, notes.size(), action, lastHour, totalRequests,
                totalPromptTokens, totalCompletionTokens, lastError);
    }
}
