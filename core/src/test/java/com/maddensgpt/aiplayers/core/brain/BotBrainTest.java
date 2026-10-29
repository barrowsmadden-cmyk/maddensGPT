package com.maddensgpt.aiplayers.core.brain;

import com.maddensgpt.aiplayers.core.action.ActionArgs;
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
import com.maddensgpt.aiplayers.core.world.Observation;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BotBrainTest {
    private static final long TICK = 50;

    /** Model stand-in: hands out scripted futures and records every conversation it is sent. */
    private static final class ScriptedClient implements ChatClient {
        final Deque<CompletableFuture<ChatResponse>> script = new ArrayDeque<>();
        final List<List<ChatMessage>> requests = new ArrayList<>();
        final List<Long> requestTimes = new ArrayList<>();
        long[] clock;

        @Override
        public CompletableFuture<ChatResponse> complete(List<ChatMessage> messages, List<ToolSpec> tools) {
            requests.add(List.copyOf(messages));
            requestTimes.add(clock[0]);
            CompletableFuture<ChatResponse> next = script.pollFirst();
            return next != null ? next : new CompletableFuture<>(); // unscripted: never answers
        }

        void reply(ToolCall... calls) {
            reply(null, null, calls);
        }

        void reply(String content, String reasoning, ToolCall... calls) {
            script.add(CompletableFuture.completedFuture(
                    new ChatResponse(ChatMessage.assistant(content, reasoning, List.of(calls)), "tool_calls", 100, 20)));
        }

        void fail(int status) {
            script.add(CompletableFuture.failedFuture(new AiException("HTTP " + status + " from test: nope", status)));
        }

        List<ChatMessage> last() {
            return requests.get(requests.size() - 1);
        }

        String lastUserMessage() {
            List<ChatMessage> messages = last();
            return messages.get(messages.size() - 1).content();
        }
    }

    /** A game action that finishes after a fixed number of ticks (or never, if ticks < 0). */
    private static final class TimedAction implements RunningAction {
        final int ticks;
        final ActionResult result;
        int ticked;
        boolean cancelled;

        TimedAction(int ticks, ActionResult result) {
            this.ticks = ticks;
            this.result = result;
        }

        @Override
        public ActionResult tick() {
            ticked++;
            return ticks >= 0 && ticked >= ticks ? result : null;
        }

        @Override
        public void cancel() {
            cancelled = true;
        }
    }

    private static final class FakeActions implements ActionHandler {
        final Map<String, Function<ActionArgs, TimedAction>> behaviours = new HashMap<>();
        final List<TimedAction> started = new ArrayList<>();
        final List<String> startedNames = new ArrayList<>();

        @Override
        public RunningAction start(String name, ActionArgs args) throws InvalidArgsException {
            Function<ActionArgs, TimedAction> behaviour = behaviours.get(name);
            if (behaviour == null) {
                throw new InvalidArgsException("no fake for " + name);
            }
            args.getInt("x"); // every fake used here takes x; proves args are passed through
            TimedAction action = behaviour.apply(args);
            started.add(action);
            startedNames.add(name);
            return action;
        }
    }

    private static final class FakeEnv implements BotEnvironment {
        final List<String> chat = new ArrayList<>();
        final List<String> logs = new ArrayList<>();
        final List<String> warnings = new ArrayList<>();

        @Override
        public Observation observe() {
            return new Observation(new Observation.Vec3(0, 64, 0), "minecraft:overworld", "minecraft:plains", 1_000,
                    "clear", 20, 20, 20, 0, "", List.of(), List.of(), 36, "minecraft:grass_block", "", List.of(),
                    List.of(), List.of("Steve"));
        }

        @Override
        public void sendChat(String line) {
            chat.add(line);
        }

        @Override
        public void log(String message) {
            logs.add(message);
        }

        @Override
        public void warn(String message) {
            warnings.add(message);
        }
    }

    private final long[] clock = {0};
    private ScriptedClient client;
    private FakeActions actions;
    private FakeEnv env;
    private BotSettings settings;
    private BotBrain brain;

    @BeforeEach
    void setUp() {
        client = new ScriptedClient();
        client.clock = clock;
        actions = new FakeActions();
        env = new FakeEnv();
        settings = new BotSettings();
        actions.behaviours.put("move_to", args -> new TimedAction(3, ActionResult.ok("Arrived.")));
    }

    private BotBrain brain() {
        brain = new BotBrain("Robo", settings, client, actions, env);
        return brain;
    }

    private void tick() {
        brain.tick(clock[0]);
        clock[0] += TICK;
    }

    private void runUntil(long millis) {
        while (clock[0] <= millis) {
            tick();
        }
    }

    private static ToolCall call(String id, String name, String json) {
        return new ToolCall(id, name, json);
    }

    private static List<ChatMessage> toolResults(List<ChatMessage> messages) {
        return messages.stream().filter(m -> m.role() == ChatMessage.Role.TOOL).toList();
    }

    @Test
    void runsCallsInOrderNarratesAndReportsResults() {
        client.reply("private: wood first", "I need wood.",
                call("c1", "say", "{\"message\":\"heading out to grab some wood\"}"),
                call("c2", "move_to", "{\"x\":10,\"y\":64,\"z\":5}"),
                call("c3", "say", "{\"message\":\"ok made it\"}"));
        brain();

        runUntil(1_000);
        assertEquals(1, client.requests.size());
        assertEquals(List.of("move_to"), actions.startedNames);
        assertEquals(3, actions.started.get(0).ticked);
        assertEquals(List.of("heading out to grab some wood"), env.chat, "first line typed out, second still pacing");
        assertTrue(env.logs.contains("Robo (private): private: wood first"));

        // Line 1 queued at 50, sent at 930; line 2 queued at 200 waits for the 1.2 s gap, sent at 2630.
        runUntil(2_600);
        assertEquals(List.of("heading out to grab some wood"), env.chat);
        runUntil(2_700);
        assertEquals(List.of("heading out to grab some wood", "ok made it"), env.chat);
        assertEquals(2, client.requests.size(), "thinks again once the min gap has passed");
        assertEquals(2_000, client.requestTimes.get(1));

        List<ChatMessage> second = client.last();
        assertEquals(ChatMessage.Role.SYSTEM, second.get(0).role());
        assertTrue(second.get(0).content().contains("You are Robo"));
        assertEquals(ChatMessage.Role.USER, second.get(1).role());
        ChatMessage assistant = second.get(2);
        assertEquals("I need wood.", assistant.reasoningContent(), "reasoning kept for replay");
        List<ChatMessage> results = toolResults(second);
        assertEquals(List.of("c1", "c2", "c3"), results.stream().map(ChatMessage::toolCallId).toList());
        assertEquals("OK: Said in chat.", results.get(0).content());
        assertEquals("OK: Arrived.", results.get(1).content());
        assertEquals(ChatMessage.Role.USER, second.get(second.size() - 1).role());
        assertTrue(client.lastUserMessage().startsWith("[Status]\nPosition (0.0, 64.0, 0.0)"));
        assertFalse(client.requests.get(0).get(1).content().isEmpty());
        assertTrue(client.requests.get(0).get(1).content().contains("This is your first turn"));
        assertFalse(client.lastUserMessage().contains("This is your first turn"));
    }

    @Test
    void failureSkipsTheRestOfTheTurn() {
        actions.behaviours.put("mine_block", args -> new TimedAction(1, ActionResult.fail("No block there.")));
        client.reply(call("a", "mine_block", "{\"x\":1,\"y\":2,\"z\":3}"),
                call("b", "say", "{\"message\":\"got it!\"}"));
        brain();
        runUntil(2_100);

        List<ChatMessage> results = toolResults(client.last());
        assertEquals("FAILED: No block there.", results.get(0).content());
        assertEquals("FAILED: Skipped because an earlier action this turn failed.", results.get(1).content());
        assertTrue(env.chat.isEmpty(), "never claims success after a failure");
    }

    @Test
    void reportsBadArgumentsAndUnknownFunctions() {
        client.reply(call("a", "fly", "{}"));
        client.reply(call("b", "move_to", "{oops"));
        client.reply(call("c", "move_to", "{\"y\":1}"));
        brain();
        runUntil(6_100);

        assertEquals("FAILED: There is no function named 'fly'.", toolResults(client.requests.get(1)).get(0).content());
        assertEquals("FAILED: Invalid arguments for move_to: arguments are not valid JSON",
                toolResults(client.requests.get(2)).get(1).content());
        assertEquals("FAILED: Invalid arguments for move_to: missing required argument 'x'",
                toolResults(client.requests.get(3)).get(2).content());
    }

    @Test
    void mentionInterruptsCurrentActionAndIsSeenNextTurn() {
        actions.behaviours.put("move_to", args -> new TimedAction(-1, null));
        client.reply(call("a", "move_to", "{\"x\":100,\"y\":64,\"z\":0}"), call("b", "say", "{\"message\":\"there\"}"));
        brain();
        runUntil(500);
        assertEquals(BotBrain.State.ACTING, brain.state());

        brain.onChat("Robo", "talking to myself, Robo", false); // own message: ignored
        brain.onChat("Alex", "nice weather", false);           // not addressed: no interrupt
        tick();
        assertFalse(actions.started.get(0).cancelled);

        brain.onChat("Steve", "hey robo, come here", false);
        tick();
        assertTrue(actions.started.get(0).cancelled);
        runUntil(2_100);

        List<ChatMessage> results = toolResults(client.last());
        assertEquals("FAILED: Interrupted because Steve is talking to you.", results.get(0).content());
        assertEquals("FAILED: Skipped: interrupted because Steve is talking to you.", results.get(1).content());
        String status = client.lastUserMessage();
        assertTrue(status.contains("<Alex> nice weather\n"), status);
        assertTrue(status.contains("<Steve> hey robo, come here  (mentions you)"), status);
        assertFalse(status.contains("talking to myself"), status);
    }

    @Test
    void privateMessagesAreMarkedAndWakeTheBot() {
        client.reply(call("a", "say", "{\"message\":\"hi\"}"));
        brain();
        runUntil(2_500);
        assertEquals(1, client.requests.size(), "idle after a talk-only turn");

        brain.onChat("Steve", "psst", true);
        runUntil(3_000);
        assertEquals(2, client.requests.size());
        assertTrue(client.lastUserMessage().contains("[private] <Steve> psst"));
    }

    @Test
    void talkOnlyTurnsBackOffUntilSomethingHappens() {
        settings.idleThinkSeconds = 10;
        client.reply(call("a", "say", "{\"message\":\"hmm\"}"));
        client.reply(call("b", "say", "{\"message\":\"hmmm\"}"));
        client.reply(call("c", "say", "{\"message\":\"hmmmm\"}"));
        brain();
        runUntil(40_000);
        // Turn 1 at 0, done at 50 -> next at 10_050; done at 10_100 -> next at 30_100 (2x).
        assertEquals(List.of(0L, 10_050L, 30_100L), client.requestTimes);

        brain.onEvent("A zombie appeared");
        runUntil(60_000);
        assertEquals(3, client.requests.size(), "plain events wait for the next scheduled turn");
        brain.onChat("Steve", "Robo!", false);
        tick();
        tick();
        assertEquals(4, client.requests.size(), "being spoken to wakes the bot and resets the backoff");
        assertTrue(client.lastUserMessage().contains("- A zombie appeared"));
    }

    @Test
    void failedRequestBacksOffAndKeepsChatForTheRetry() {
        client.fail(503);
        client.reply(call("a", "say", "{\"message\":\"sorry, lagging\"}"));
        brain();
        brain.onChat("Steve", "robo you there?", false);
        runUntil(4_900);
        assertEquals(1, client.requests.size());
        assertTrue(env.warnings.get(0).contains("AI request failed (1 in a row): HTTP 503"), env.warnings.toString());
        assertTrue(brain.status().lastError().contains("503"));

        runUntil(5_200);
        assertEquals(2, client.requests.size(), "retried 5 s after the failure");
        assertTrue(client.lastUserMessage().contains("<Steve> robo you there?"));
        assertEquals(2, client.last().size(), "failed turn is not in the history");
    }

    @Test
    void repeatedBadRequestsStartAFreshConversation() {
        client.reply(call("a", "move_to", "{\"x\":1,\"y\":1,\"z\":1}"));
        client.fail(400);
        client.fail(400);
        client.fail(400);
        brain();
        runUntil(40_000);
        assertEquals(5, client.requests.size());
        // system + one history turn (user, assistant, tool result) + current user
        assertEquals(5, client.requests.get(3).size(), "history still sent before the third failure");
        assertEquals(2, client.requests.get(4).size(), "history cleared after three 400s in a row");
        assertTrue(env.warnings.stream().anyMatch(w -> w.contains("starting a fresh one")));
    }

    @Test
    void enforcesHourlyRequestCap() {
        settings.maxRequestsPerHour = 2;
        settings.idleThinkSeconds = 1;
        settings.minSecondsBetweenRequests = 0;
        client.reply();
        client.reply();
        client.reply();
        brain();
        runUntil(120_000);
        assertEquals(2, client.requests.size());
        assertEquals(1, env.warnings.stream().filter(w -> w.contains("maxRequestsPerHour")).count());
        assertEquals(2, brain.status().requestsLastHour());
    }

    @Test
    void limitsActionsPerTurnAndTimesOutStuckActions() {
        settings.maxActionsPerTurn = 1;
        settings.actionTimeoutSeconds = 5;
        actions.behaviours.put("move_to", args -> new TimedAction(-1, null));
        client.reply(call("a", "move_to", "{\"x\":1,\"y\":1,\"z\":1}"), call("b", "say", "{\"message\":\"x\"}"));
        brain();
        runUntil(4_000);
        assertEquals(1, client.requests.size());
        runUntil(7_100);
        assertTrue(actions.started.get(0).cancelled);
        List<ChatMessage> results = toolResults(client.last());
        assertEquals("FAILED: Timed out after 5 seconds.", results.get(0).content());
        assertEquals("FAILED: Skipped: at most 1 actions per turn.", results.get(1).content());
    }

    @Test
    void notesPlanAndGoalAppearInTheStatus() {
        client.reply(call("a", "remember", "{\"note\":\"base is at 0 64 0\"}"),
                call("b", "set_plan", "{\"plan\":\"1. logs 2. table\"}"),
                call("c", "wait", "{\"seconds\":2}"));
        brain();
        brain.setGoal("build a house");
        runUntil(3_000);
        assertEquals(2, client.requests.size());
        String status = client.lastUserMessage();
        assertTrue(status.contains("Your goal: build a house"), status);
        assertTrue(status.contains("Your plan: 1. logs 2. table"), status);
        assertTrue(status.contains("Your notes:\n1. base is at 0 64 0"), status);
        assertEquals("OK: Waited 2 seconds.", toolResults(client.last()).get(2).content());
    }

    @Test
    void newGoalInterruptsCurrentAction() {
        actions.behaviours.put("move_to", args -> new TimedAction(-1, null));
        client.reply(call("a", "move_to", "{\"x\":1,\"y\":1,\"z\":1}"));
        brain();
        runUntil(500);
        brain.setGoal("come back");
        runUntil(2_100);
        assertEquals("FAILED: Interrupted because your goal changed.", toolResults(client.last()).get(0).content());
        assertTrue(client.lastUserMessage().contains("You were given a new goal: come back"));
    }

    @Test
    void pauseDropsTheRequestInFlightAndResumeStartsOver() {
        CompletableFuture<ChatResponse> slow = new CompletableFuture<>();
        client.script.add(slow);
        brain();
        tick();
        assertEquals(BotBrain.State.THINKING, brain.state());

        brain.pause();
        slow.complete(new ChatResponse(ChatMessage.assistant(null, null,
                List.of(call("a", "move_to", "{\"x\":1,\"y\":1,\"z\":1}"))), "tool_calls", 1, 1));
        runUntil(5_000);
        assertTrue(actions.started.isEmpty(), "stale reply ignored");
        assertEquals(BotBrain.State.PAUSED, brain.state());

        brain.resume();
        tick();
        assertEquals(2, client.requests.size());
        assertEquals(2, client.last().size(), "abandoned turn left no trace in the history");
    }

    @Test
    void historyIsTrimmedToWholeTurns() {
        settings.maxHistoryTurns = 2;
        settings.minSecondsBetweenRequests = 0;
        for (int i = 0; i < 4; i++) {
            client.reply(call("m" + i, "move_to", "{\"x\":" + i + ",\"y\":1,\"z\":1}"));
        }
        brain();
        runUntil(2_000);
        assertEquals(5, client.requests.size());
        List<ChatMessage> last = client.last();
        // system + 2 turns x (user, assistant, tool) + current user
        assertEquals(1 + 2 * 3 + 1, last.size());
        assertEquals(ChatMessage.Role.USER, last.get(1).role());
        assertEquals("m2", last.get(2).toolCalls().get(0).id());
        assertEquals("m3", last.get(6).toolCallId());
    }

    @Test
    void statusReportsCurrentActionAndTokens() {
        actions.behaviours.put("move_to", args -> new TimedAction(-1, null));
        client.reply(call("a", "move_to", "{\"x\":1,\"y\":1,\"z\":1}"));
        brain();
        runUntil(200);
        BotBrain.Status status = brain.status();
        assertEquals(BotBrain.State.ACTING, status.state());
        assertEquals("TimedAction", status.currentAction());
        assertEquals(1, status.totalRequests());
        assertEquals(100, status.promptTokens());
        assertEquals(20, status.completionTokens());
        assertEquals("", status.lastError());
    }
}
