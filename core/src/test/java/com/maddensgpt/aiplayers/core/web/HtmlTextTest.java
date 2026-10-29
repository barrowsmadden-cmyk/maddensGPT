package com.maddensgpt.aiplayers.core.web;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class HtmlTextTest {
    @Test
    void extractsTitleAndReadableText() {
        HtmlText.Page page = HtmlText.extract("""
                <!DOCTYPE html><html><HEAD><title>Crafting Table &ndash; Minecraft Wiki</title>
                <style>body { color: red }</style><script>var x = "<p>not text</p>";</script></head>
                <body><!-- comment <b>hidden</b> --><h1>Crafting Table</h1>
                <p>A <b>crafting table</b> lets you craft 3&times;3 recipes.<br>Made from 4 planks.</p>
                <ul><li>Fast &amp; easy</li><li>Costs &#52; &#x50;lanks</li></ul>
                <svg><text>icon</text></svg><noscript>enable js</noscript>
                <p>Price &lt; 1 diamond &unknown; and a < b</p></body></html>
                """);
        assertEquals("Crafting Table - Minecraft Wiki", page.title());
        assertEquals("""
                Crafting Table

                A crafting table lets you craft 3x3 recipes.
                Made from 4 planks.

                Fast & easy
                Costs 4 Planks

                Price < 1 diamond &unknown; and a < b""", page.text());
    }

    @Test
    void handlesMalformedMarkup() {
        assertEquals("unclosed", HtmlText.extract("unclosed<script>forever and ever").text());
        assertEquals("text", HtmlText.extract("text<div class=\"x").text());
        assertEquals("after", HtmlText.extract("<svg/>after").text());
        assertEquals("", HtmlText.extract("").text());
        assertEquals("", HtmlText.extract("<title>never closed").title());
    }

    @Test
    void inlinesSnippets() {
        assertEquals("How to make a brewing stand in Minecraft",
                HtmlText.inline("How to make a <strong>brewing stand</strong>\n in Minecraft"));
        assertEquals("", HtmlText.inline(null));
    }

    @Test
    void findsCaseInsensitively() {
        assertEquals(3, HtmlText.indexOfIgnoreCase("abcDEF", "def", 0));
        assertEquals(-1, HtmlText.indexOfIgnoreCase("abc", "abcd", 0));
    }
}
