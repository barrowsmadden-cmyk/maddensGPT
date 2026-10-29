package com.maddensgpt.aiplayers.core.web;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Internet access for the bots: web search through a search API, and reading public web pages.
 * Off by default; it needs a search API key.
 */
public final class WebSettings {
    /** Offer web_search (and read_webpage) to the model. */
    public boolean enabled = false;

    /** "tavily" or "brave". */
    public String searchProvider = "tavily";

    /** Search API key written directly in the config. Prefer {@link #apiKeyEnv} instead. */
    public String apiKey = "";

    /** Environment variable holding the search API key, e.g. TAVILY_API_KEY or BRAVE_API_KEY. */
    public String apiKeyEnv = "TAVILY_API_KEY";

    /** Results returned per search, 1 to 10. */
    public int maxResults = 5;

    /** Also offer read_webpage, which downloads a page and returns its text. */
    public boolean allowPageReading = true;

    /** Characters of page text returned per read_webpage call; the bot can ask for more with "start". */
    public int maxPageChars = 4_000;

    /** Largest download accepted per page; anything beyond is ignored. */
    public int maxPageBytes = 2_000_000;

    public int timeoutSeconds = 20;

    /** Cap on web requests (searches plus page reads) per bot per rolling hour, to bound cost. */
    public int maxRequestsPerHour = 60;

    /** Domains the bot may never read, e.g. "example.com" (also blocks subdomains). */
    public List<String> blockedDomains = new ArrayList<>();

    public enum Provider { TAVILY, BRAVE }

    public Provider provider() {
        String name = searchProvider == null ? "" : searchProvider.trim().toLowerCase(Locale.ROOT);
        return switch (name) {
            case "brave" -> Provider.BRAVE;
            case "tavily" -> Provider.TAVILY;
            default -> null;
        };
    }

    public String resolveApiKey() {
        if (apiKey != null && !apiKey.isBlank()) {
            return apiKey.trim();
        }
        if (apiKeyEnv != null && !apiKeyEnv.isBlank()) {
            String fromEnv = System.getenv(apiKeyEnv.trim());
            if (fromEnv != null && !fromEnv.isBlank()) {
                return fromEnv.trim();
            }
        }
        return "";
    }

    /** Problems that stop web access from working; empty when usable. */
    public List<String> validate() {
        List<String> problems = new ArrayList<>();
        if (provider() == null) {
            problems.add("web.searchProvider must be \"tavily\" or \"brave\", got \"" + searchProvider + "\"");
        }
        if (resolveApiKey().isEmpty()) {
            problems.add("no search API key: set web.apiKey, or set the environment variable named by "
                    + "web.apiKeyEnv (" + apiKeyEnv + ")");
        }
        return problems;
    }

    WebSettings sanitize() {
        maxResults = clamp(maxResults, 1, 10);
        maxPageChars = clamp(maxPageChars, 500, 50_000);
        maxPageBytes = clamp(maxPageBytes, 10_000, 20_000_000);
        timeoutSeconds = clamp(timeoutSeconds, 1, 120);
        maxRequestsPerHour = clamp(maxRequestsPerHour, 1, 10_000);
        if (blockedDomains == null) {
            blockedDomains = new ArrayList<>();
        }
        blockedDomains.removeIf(domain -> domain == null || domain.isBlank());
        return this;
    }

    /** Called by BotSettings.sanitize(). */
    public static WebSettings sanitize(WebSettings settings) {
        return (settings == null ? new WebSettings() : settings).sanitize();
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }
}
