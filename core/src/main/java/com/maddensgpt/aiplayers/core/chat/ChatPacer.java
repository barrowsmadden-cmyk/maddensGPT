package com.maddensgpt.aiplayers.core.chat;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Optional;
import java.util.regex.Pattern;

/**
 * Turns what the model wants to say into chat lines that read like a person typing: cleans up
 * formatting, splits long text at sentence or word boundaries, and releases lines with a short
 * typing delay and a minimum gap so the bot never floods chat.
 */
public final class ChatPacer {
    private static final Pattern FORMATTING_CODE = Pattern.compile("§.?");
    private static final Pattern CONTROL_CHARS = Pattern.compile("[\\p{Cntrl}&&[^\\n\\t]]");
    private static final Pattern MARKDOWN = Pattern.compile("(\\*\\*|__|`+|^#+\\s*)", Pattern.MULTILINE);
    private static final Pattern WHITESPACE = Pattern.compile("\\s+");

    private static final long BASE_TYPING_MILLIS = 300;
    private static final long TYPING_MILLIS_PER_CHAR = 20;
    private static final long MAX_TYPING_MILLIS = 3_000;

    private record Pending(String line, long readyAt) {
    }

    private final String botName;
    private final int maxLineLength;
    private final long minGapMillis;
    private final int maxQueued;
    private final Deque<Pending> queue = new ArrayDeque<>();
    private long lastScheduledAt = Long.MIN_VALUE;

    public ChatPacer(String botName, int maxLineLength, long minGapMillis, int maxQueued) {
        this.botName = botName;
        this.maxLineLength = Math.max(20, maxLineLength);
        this.minGapMillis = Math.max(0, minGapMillis);
        this.maxQueued = Math.max(1, maxQueued);
    }

    /**
     * Queues a message. Lines beyond the queue limit are dropped.
     *
     * @return the number of lines queued (0 if the text was empty after cleanup or the queue was full)
     */
    public int enqueue(String text, long nowMillis) {
        String cleaned = clean(text);
        if (cleaned.isEmpty()) {
            return 0;
        }
        int queued = 0;
        for (String line : split(cleaned, maxLineLength)) {
            if (queue.size() >= maxQueued) {
                break;
            }
            long earliest = lastScheduledAt == Long.MIN_VALUE ? nowMillis
                    : Math.max(nowMillis, lastScheduledAt + minGapMillis);
            long readyAt = earliest + typingDelay(line);
            queue.addLast(new Pending(line, readyAt));
            lastScheduledAt = readyAt;
            queued++;
        }
        return queued;
    }

    /** Returns the next line if its send time has come. Call once per tick. */
    public Optional<String> poll(long nowMillis) {
        Pending head = queue.peekFirst();
        if (head == null || head.readyAt() > nowMillis) {
            return Optional.empty();
        }
        queue.removeFirst();
        return Optional.of(head.line());
    }

    public boolean isEmpty() {
        return queue.isEmpty();
    }

    public void clear() {
        queue.clear();
    }

    /** Strips Minecraft formatting codes, control characters, markdown, quotes and a leading "Name:". */
    String clean(String text) {
        if (text == null) {
            return "";
        }
        String cleaned = FORMATTING_CODE.matcher(text).replaceAll("");
        cleaned = CONTROL_CHARS.matcher(cleaned).replaceAll("");
        cleaned = MARKDOWN.matcher(cleaned).replaceAll("");
        cleaned = WHITESPACE.matcher(cleaned).replaceAll(" ").strip();

        // Models sometimes write "<Bot> hi" or "Bot: hi" as if formatting a chat log.
        for (String prefix : List.of("<" + botName + ">", botName + ":")) {
            if (cleaned.regionMatches(true, 0, prefix, 0, prefix.length())) {
                cleaned = cleaned.substring(prefix.length()).strip();
            }
        }
        if (cleaned.length() >= 2 && cleaned.startsWith("\"") && cleaned.endsWith("\"")) {
            cleaned = cleaned.substring(1, cleaned.length() - 1).strip();
        }
        return cleaned;
    }

    /** Splits at sentence ends, then spaces, then hard-cuts words longer than a line. */
    static List<String> split(String text, int maxLength) {
        List<String> lines = new ArrayList<>();
        String rest = text;
        while (rest.length() > maxLength) {
            int cut = lastSentenceEnd(rest, maxLength);
            if (cut <= 0) {
                cut = rest.lastIndexOf(' ', maxLength);
            }
            if (cut <= 0) {
                cut = maxLength;
            }
            lines.add(rest.substring(0, cut).strip());
            rest = rest.substring(cut).strip();
        }
        if (!rest.isEmpty()) {
            lines.add(rest);
        }
        return lines;
    }

    /** Index just past the last ". ", "! " or "? " that keeps the line within maxLength. */
    private static int lastSentenceEnd(String text, int maxLength) {
        for (int i = Math.min(maxLength, text.length() - 1) - 1; i > maxLength / 3; i--) {
            char c = text.charAt(i);
            if ((c == '.' || c == '!' || c == '?') && text.charAt(i + 1) == ' ') {
                return i + 1;
            }
        }
        return -1;
    }

    private static long typingDelay(String line) {
        return Math.min(MAX_TYPING_MILLIS, BASE_TYPING_MILLIS + TYPING_MILLIS_PER_CHAR * line.length());
    }
}
