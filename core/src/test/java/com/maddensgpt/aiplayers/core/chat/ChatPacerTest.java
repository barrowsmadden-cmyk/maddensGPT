package com.maddensgpt.aiplayers.core.chat;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ChatPacerTest {
    @Test
    void cleansFormattingMarkdownAndSelfPrefix() {
        ChatPacer pacer = new ChatPacer("Robo", 200, 0, 10);
        assertEquals("found iron finally", pacer.clean("§aRobo: **found** `iron`\n\n finally"));
        assertEquals("hey", pacer.clean("<robo> \"hey\""));
        assertEquals("", pacer.clean("   \u0007 "));
        assertEquals("", pacer.clean(null));
    }

    @Test
    void splitsAtSentencesThenWords() {
        assertEquals(List.of("one two three.", "four five six."), ChatPacer.split("one two three. four five six.", 20));
        assertEquals(List.of("aaaa bbbb", "cccc"), ChatPacer.split("aaaa bbbb cccc", 10));
        assertEquals(List.of("abcdefghij", "klm"), ChatPacer.split("abcdefghijklm", 10));
        assertEquals(List.of("short"), ChatPacer.split("short", 10));
    }

    @Test
    void releasesLinesWithTypingDelayAndGap() {
        ChatPacer pacer = new ChatPacer("Robo", 200, 1_000, 10);
        assertEquals(1, pacer.enqueue("hi", 0));          // ready at 0 + 300 + 2*20 = 340
        assertEquals(1, pacer.enqueue("going mining", 0)); // ready at 340 + 1000 + 300 + 12*20 = 1880

        assertEquals(Optional.empty(), pacer.poll(339));
        assertEquals(Optional.of("hi"), pacer.poll(340));
        assertEquals(Optional.empty(), pacer.poll(1_879));
        assertEquals(Optional.of("going mining"), pacer.poll(1_880));
        assertTrue(pacer.isEmpty());

        // Much later, the next line only waits for its typing delay.
        pacer.enqueue("ok", 10_000);
        assertEquals(Optional.of("ok"), pacer.poll(10_340));
    }

    @Test
    void dropsLinesBeyondQueueLimit() {
        ChatPacer pacer = new ChatPacer("Robo", 20, 0, 2);
        assertEquals(2, pacer.enqueue("first sentence here. second sentence here. third sentence.", 0));
        assertEquals(0, pacer.enqueue("more", 0));
        assertEquals(0, pacer.enqueue("§r", 0), "nothing left after cleanup");
    }
}
