package com.maddensgpt.aiplayers.core.web;

import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Turns HTML into readable plain text: drops scripts, styles and markup, keeps line breaks at
 * block elements and decodes common character references. A single linear scan, so large or
 * malformed pages cannot make it slow.
 */
final class HtmlText {
    record Page(String title, String text) {
    }

    private static final Set<String> SKIPPED = Set.of(
            "script", "style", "noscript", "svg", "template", "head", "iframe", "object", "canvas");
    private static final Set<String> BLOCKS = Set.of(
            "p", "br", "div", "li", "ul", "ol", "tr", "td", "th", "table", "h1", "h2", "h3", "h4", "h5", "h6",
            "section", "article", "header", "footer", "nav", "aside", "main", "blockquote", "pre", "hr", "dt", "dd",
            "figcaption", "form");
    private static final Map<String, String> NAMED = Map.ofEntries(
            Map.entry("amp", "&"), Map.entry("lt", "<"), Map.entry("gt", ">"), Map.entry("quot", "\""),
            Map.entry("apos", "'"), Map.entry("nbsp", " "), Map.entry("ndash", "-"), Map.entry("mdash", "-"),
            Map.entry("hellip", "..."), Map.entry("lsquo", "'"), Map.entry("rsquo", "'"), Map.entry("ldquo", "\""),
            Map.entry("rdquo", "\""), Map.entry("copy", "(c)"), Map.entry("reg", "(R)"), Map.entry("times", "x"),
            Map.entry("middot", "-"), Map.entry("bull", "-"), Map.entry("deg", "°"));
    private static final Pattern ENTITY = Pattern.compile("&(#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[a-zA-Z]{2,8});");
    private static final Pattern SPACES = Pattern.compile("[ \\t\\x0B\\f\\r\\u00A0]+");
    private static final Pattern BLANK_LINES = Pattern.compile("\\n{3,}");

    private HtmlText() {
    }

    static Page extract(String html) {
        String title = "";
        int titleStart = indexOfIgnoreCase(html, "<title", 0);
        if (titleStart >= 0) {
            int open = html.indexOf('>', titleStart);
            int close = open < 0 ? -1 : indexOfIgnoreCase(html, "</title", open + 1);
            if (close > open) {
                title = inline(html.substring(open + 1, close));
            }
        }
        return new Page(title, tidy(decode(stripTags(html))));
    }

    /** Text of a short snippet on one line, e.g. a search result description containing &lt;strong&gt;. */
    static String inline(String html) {
        if (html == null) {
            return "";
        }
        return tidy(decode(stripTags(html))).replace('\n', ' ').strip();
    }

    private static String stripTags(String html) {
        StringBuilder out = new StringBuilder(Math.min(html.length(), 1 << 20));
        int length = html.length();
        int i = 0;
        while (i < length) {
            int lt = html.indexOf('<', i);
            if (lt < 0) {
                out.append(html, i, length);
                break;
            }
            out.append(html, i, lt);
            if (html.startsWith("<!--", lt)) {
                int end = html.indexOf("-->", lt + 4);
                i = end < 0 ? length : end + 3;
                continue;
            }
            int j = lt + 1;
            boolean closing = j < length && html.charAt(j) == '/';
            if (closing) {
                j++;
            }
            int nameStart = j;
            while (j < length && Character.isLetterOrDigit(html.charAt(j))) {
                j++;
            }
            String name = html.substring(nameStart, j).toLowerCase(Locale.ROOT);
            boolean declaration = !closing && j < length && (html.charAt(j) == '!' || html.charAt(j) == '?');
            if (name.isEmpty() && !declaration) {
                out.append('<'); // a literal "<" in text, e.g. "a < b"
                i = lt + 1;
                continue;
            }
            int gt = html.indexOf('>', j);
            if (gt < 0) {
                break; // unterminated tag at the end: drop it
            }
            if (!closing && SKIPPED.contains(name) && html.charAt(gt - 1) != '/') {
                int close = indexOfIgnoreCase(html, "</" + name, gt + 1);
                int closeEnd = close < 0 ? -1 : html.indexOf('>', close);
                i = closeEnd < 0 ? length : closeEnd + 1;
                out.append(' ');
                continue;
            }
            // One line break per block boundary: "</li><li>" must not leave an empty line between items.
            if (BLOCKS.contains(name) && (out.isEmpty() || out.charAt(out.length() - 1) != '\n')) {
                out.append('\n');
            }
            i = gt + 1;
        }
        return out.toString();
    }

    private static String decode(String text) {
        Matcher matcher = ENTITY.matcher(text);
        StringBuilder out = new StringBuilder(text.length());
        while (matcher.find()) {
            String entity = matcher.group(1);
            String replacement = null;
            if (entity.charAt(0) == '#') {
                try {
                    int codePoint = entity.length() > 1 && (entity.charAt(1) == 'x' || entity.charAt(1) == 'X')
                            ? Integer.parseInt(entity.substring(2), 16)
                            : Integer.parseInt(entity.substring(1));
                    if (Character.isValidCodePoint(codePoint) && codePoint != 0
                            && !(codePoint >= 0xD800 && codePoint <= 0xDFFF)) {
                        replacement = new String(Character.toChars(codePoint));
                    }
                } catch (NumberFormatException e) {
                    // leave it as written
                }
            } else {
                replacement = NAMED.get(entity.toLowerCase(Locale.ROOT));
            }
            matcher.appendReplacement(out, Matcher.quoteReplacement(replacement != null ? replacement : matcher.group()));
        }
        matcher.appendTail(out);
        return out.toString();
    }

    private static String tidy(String text) {
        String spaced = SPACES.matcher(text).replaceAll(" ");
        StringBuilder out = new StringBuilder(spaced.length());
        for (String line : spaced.split("\n", -1)) {
            out.append(line.strip()).append('\n');
        }
        return BLANK_LINES.matcher(out.toString()).replaceAll("\n\n").strip();
    }

    static int indexOfIgnoreCase(String text, String needle, int from) {
        int last = text.length() - needle.length();
        for (int i = Math.max(0, from); i <= last; i++) {
            if (text.regionMatches(true, i, needle, 0, needle.length())) {
                return i;
            }
        }
        return -1;
    }
}
