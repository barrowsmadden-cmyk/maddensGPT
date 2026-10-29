package com.maddensgpt.aiplayers.core.web;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.http.HttpClient;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class HttpWebAccessTest {
    private record Route(int status, String contentType, byte[] body, String location) {
    }

    private record Seen(String method, String path, String query, Map<String, String> headers, String body) {
    }

    private HttpServer server;
    private String base;
    private final Map<String, Route> routes = new HashMap<>();
    private final List<Seen> seen = new ArrayList<>();
    private final long[] clock = {0};
    private final HttpClient http = HttpClient.newBuilder().proxy(HttpClient.Builder.NO_PROXY)
            .followRedirects(HttpClient.Redirect.NEVER).build();

    @BeforeEach
    void start() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", this::handle);
        server.start();
        base = "http://127.0.0.1:" + server.getAddress().getPort();
    }

    @AfterEach
    void stop() {
        server.stop(0);
    }

    private void handle(HttpExchange exchange) throws IOException {
        Map<String, String> headers = new HashMap<>();
        exchange.getRequestHeaders().forEach((name, values) -> headers.put(name.toLowerCase(), values.get(0)));
        synchronized (seen) {
            seen.add(new Seen(exchange.getRequestMethod(), exchange.getRequestURI().getPath(),
                    exchange.getRequestURI().getRawQuery(), headers,
                    new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8)));
        }
        Route route = routes.getOrDefault(exchange.getRequestURI().getPath(),
                new Route(404, "text/plain", "missing".getBytes(StandardCharsets.UTF_8), null));
        if (route.contentType() != null) {
            exchange.getResponseHeaders().add("Content-Type", route.contentType());
        }
        if (route.location() != null) {
            exchange.getResponseHeaders().add("Location", route.location());
        }
        exchange.sendResponseHeaders(route.status(), route.body().length == 0 ? -1 : route.body().length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(route.body());
        }
    }

    private void route(String path, int status, String contentType, String body) {
        routes.put(path, new Route(status, contentType, body.getBytes(StandardCharsets.UTF_8), null));
    }

    private void redirect(String path, String location) {
        routes.put(path, new Route(302, null, new byte[0], location));
    }

    private WebSettings settings(String provider) {
        WebSettings settings = new WebSettings();
        settings.enabled = true;
        settings.searchProvider = provider;
        settings.apiKey = "search-key";
        settings.maxResults = 3;
        settings.maxPageChars = 500;
        return settings;
    }

    private HttpWebAccess web(WebSettings settings) {
        return new HttpWebAccess(settings, http, new UrlGuard(settings.blockedDomains, true),
                base + "/tavily", base + "/brave", () -> clock[0]);
    }

    private static String get(java.util.concurrent.CompletableFuture<String> future) throws Exception {
        return future.get(10, TimeUnit.SECONDS);
    }

    private static WebException failure(java.util.concurrent.CompletableFuture<String> future) {
        ExecutionException e = assertThrows(ExecutionException.class, () -> future.get(10, TimeUnit.SECONDS));
        return assertInstanceOf(WebException.class, e.getCause());
    }

    @Test
    void searchesWithTavily() throws Exception {
        route("/tavily", 200, "application/json", """
                {"query":"brewing stand","results":[
                  {"title":"Brewing Stand","url":"https://minecraft.wiki/w/Brewing_Stand","content":"Made with a blaze rod and 3 cobblestone.","score":0.9},
                  {"title":"Guide","url":"https://example.org/guide","content":""}]}
                """);
        String text = get(web(settings("tavily")).search("  how to make a brewing stand  "));

        Seen request = seen.get(0);
        assertEquals("POST", request.method());
        assertEquals("Bearer search-key", request.headers().get("authorization"));
        JsonObject body = JsonParser.parseString(request.body()).getAsJsonObject();
        assertEquals("how to make a brewing stand", body.get("query").getAsString());
        assertEquals(3, body.get("max_results").getAsInt());
        assertEquals("basic", body.get("search_depth").getAsString());

        assertEquals(HttpWebAccess.UNTRUSTED_NOTE + "\n"
                + "Web results for \"how to make a brewing stand\":\n"
                + "1. Brewing Stand - https://minecraft.wiki/w/Brewing_Stand\n"
                + "   Made with a blaze rod and 3 cobblestone.\n"
                + "2. Guide - https://example.org/guide", text);
    }

    @Test
    void searchesWithBrave() throws Exception {
        route("/brave", 200, "application/json", """
                {"type":"search","web":{"results":[
                  {"title":"Netherite &amp; you","url":"https://a.example/n","description":"Combine with <strong>gold</strong>."}]}}
                """);
        String text = get(web(settings("brave")).search("netherite ingot recipe"));

        Seen request = seen.get(0);
        assertEquals("GET", request.method());
        assertEquals("q=netherite+ingot+recipe&count=3", request.query());
        assertEquals("search-key", request.headers().get("x-subscription-token"));
        assertFalse(request.headers().containsKey("authorization"), "Brave uses its own header");
        assertTrue(text.endsWith("1. Netherite & you - https://a.example/n\n   Combine with gold."), text);
    }

    @Test
    void reportsSearchProblemsInPlainWords() throws Exception {
        route("/tavily", 401, "application/json", "{\"detail\":{\"error\":\"Unauthorized\"}}");
        assertEquals("the search service answered HTTP 401 (the server's search API key was rejected)",
                failure(web(settings("tavily")).search("x")).getMessage());

        route("/tavily", 200, "application/json", "{\"results\":[]}");
        assertTrue(get(web(settings("tavily")).search("nothing")).endsWith("none found."));

        assertEquals("the search query is empty", failure(web(settings("tavily")).search(" ")).getMessage());

        WebSettings noKey = settings("tavily");
        noKey.apiKey = "";
        noKey.apiKeyEnv = "AIPLAYERS_TEST_SURELY_UNSET_VARIABLE";
        assertEquals("web search is not set up on this server", failure(web(noKey).search("x")).getMessage());
    }

    @Test
    void readsPagesAsTextInChunks() throws Exception {
        String paragraph = "Iron golems protect villagers. ".repeat(30).strip();
        route("/page", 200, "text/html; charset=utf-8",
                "<html><head><title>Iron Golem</title><script>evil()</script></head><body><p>" + paragraph
                        + "</p></body></html>");
        HttpWebAccess web = web(settings("tavily"));

        String first = get(web.read(base + "/page", 0));
        assertTrue(first.startsWith(HttpWebAccess.UNTRUSTED_NOTE + "\nPage: " + base + "/page\nTitle: Iron Golem\n"
                + "Characters 0 to 500 of " + paragraph.length() + ":\nIron golems protect villagers."), first);
        assertFalse(first.contains("evil"));
        assertTrue(first.endsWith("[More text follows: call read_webpage again with start=500]"), first);
        assertEquals("AIPlayersMod", seen.get(0).headers().get("user-agent").split("/")[0]);

        String second = get(web.read(base + "/page", 500));
        assertTrue(second.contains("Characters 500 to " + paragraph.length()), second);
        assertFalse(second.contains("More text follows"));

        String past = get(web.read(base + "/page", 99_999));
        assertTrue(past.contains("(nothing after character 99999"), past);
    }

    @Test
    void followsRedirectsCheckingEachHop() throws Exception {
        route("/final", 200, "text/plain", "plain text page");
        redirect("/hop1", "/hop2");
        redirect("/hop2", base + "/final");
        String text = get(web(settings("tavily")).read(base + "/hop1", 0));
        assertTrue(text.contains("Page: " + base + "/final"), text);
        assertTrue(text.endsWith("plain text page"), text);

        redirect("/to-file", "file:///etc/passwd");
        assertEquals("only http and https links can be opened",
                failure(web(settings("tavily")).read(base + "/to-file", 0)).getMessage());

        WebSettings blocking = settings("tavily");
        blocking.blockedDomains = new ArrayList<>(List.of("blocked.test"));
        redirect("/to-blocked", "http://blocked.test/");
        assertEquals("this server does not allow opening blocked.test",
                failure(web(blocking).read(base + "/to-blocked", 0)).getMessage());

        redirect("/loop", "/loop");
        assertEquals("the page redirected too many times",
                failure(web(settings("tavily")).read(base + "/loop", 0)).getMessage());
    }

    @Test
    void refusesNonPagesAndCapsDownloadSize() throws Exception {
        routes.put("/image", new Route(200, "image/png", new byte[] {1, 2, 3}, null));
        assertEquals("that link is not a web page (image/png)",
                failure(web(settings("tavily")).read(base + "/image", 0)).getMessage());

        route("/gone", 404, "text/html", "<p>nope</p>");
        assertEquals("the site answered HTTP 404", failure(web(settings("tavily")).read(base + "/gone", 0)).getMessage());

        WebSettings small = settings("tavily");
        small.maxPageBytes = 10_000; // the minimum allowed
        route("/huge", 200, "text/plain", "x".repeat(50_000));
        String text = get(web(small).read(base + "/huge", 0));
        assertTrue(text.contains("of 10000 (page was too large; only the first part was downloaded)"), text);

        WebSettings off = settings("tavily");
        off.allowPageReading = false;
        assertEquals("reading web pages is turned off on this server",
                failure(web(off).read(base + "/huge", 0)).getMessage());
    }

    @Test
    void limitsWebRequestsPerHour() throws Exception {
        route("/tavily", 200, "application/json", "{\"results\":[]}");
        WebSettings settings = settings("tavily");
        settings.maxRequestsPerHour = 2;
        HttpWebAccess web = web(settings);
        get(web.search("a"));
        get(web.search("b"));
        assertEquals("you have used this hour's web lookups (2); try again later",
                failure(web.search("c")).getMessage());
        clock[0] += 3_600_000;
        get(web.search("d"));
        assertEquals(3, seen.size());
    }
}
