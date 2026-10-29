package com.maddensgpt.aiplayers.core.web;

import org.junit.jupiter.api.Test;

import java.net.InetAddress;
import java.net.URI;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class UrlGuardTest {
    private final UrlGuard guard = new UrlGuard(List.of("Blocked.example", ".also-blocked.test"), false);

    private static boolean isPublic(String literal) throws Exception {
        return UrlGuard.isPublic(InetAddress.getByName(literal));
    }

    @Test
    void classifiesAddresses() throws Exception {
        for (String blocked : List.of("127.0.0.1", "127.8.9.10", "10.1.2.3", "172.16.0.1", "172.31.255.255",
                "192.168.1.1", "169.254.169.254", "100.64.0.1", "100.127.255.255", "0.0.0.0", "0.1.2.3",
                "224.0.0.1", "255.255.255.255", "198.18.0.1", "192.0.2.5", "203.0.113.9",
                "::", "::1", "fe80::1", "fc00::1", "fd12:3456::1", "ff02::1", "::ffff:127.0.0.1", "::ffff:10.0.0.1",
                "::127.0.0.1", "64:ff9b::a9fe:a9fe", "2002:7f00:1::", "2001:0:4136:e378::1", "2001:db8::1")) {
            assertFalse(isPublic(blocked), blocked);
        }
        for (String allowed : List.of("8.8.8.8", "1.1.1.1", "172.32.0.1", "100.128.0.1", "93.184.216.34",
                "2606:4700:4700::1111", "2002:808:808::", "64:ff9b::808:808")) {
            assertTrue(isPublic(allowed), allowed);
        }
    }

    @Test
    void allowsPublicHttpUrls() {
        assertEquals(URI.create("https://8.8.8.8/a/c"), guard.check("https://8.8.8.8/a/b/../c"));
        assertEquals(URI.create("https://1.1.1.1"), guard.check("1.1.1.1"), "https:// is assumed");
        assertEquals("http", guard.check("http://8.8.8.8:80/").getScheme());
    }

    @Test
    void refusesUnsafeUrls() {
        assertMessage("only http and https links can be opened", "file:///etc/passwd");
        assertMessage("only http and https links can be opened", "ftp://8.8.8.8/");
        assertMessage("only http and https links can be opened", "javascript://8.8.8.8/%0aalert(1)");
        assertMessage("links with a username or password cannot be opened", "https://user:pw@8.8.8.8/");
        assertMessage("only standard web ports (80 and 443) can be opened", "https://8.8.8.8:8080/");
        assertMessage("that address is on a private or local network and cannot be opened", "http://127.0.0.1/");
        assertMessage("that address is on a private or local network and cannot be opened", "http://localhost/");
        assertMessage("that address is on a private or local network and cannot be opened",
                "http://169.254.169.254/latest/meta-data/");
        assertMessage("that address is on a private or local network and cannot be opened", "http://[::1]/");
        // Java does not read hex IP literals, so this is looked up as a host name and not found.
        assertThrows(WebException.class, () -> guard.check("http://0x7f000001/"));
        assertMessage("this server does not allow opening blocked.example", "https://www.BLOCKED.example./x");
        assertMessage("this server does not allow opening also-blocked.test", "https://also-blocked.test");
        assertMessage("no URL given", " ");
        assertMessage("that is not a valid URL", "https://exa mple.com/");
        assertMessage("the URL is too long", "https://8.8.8.8/" + "a".repeat(3_000));
    }

    private void assertMessage(String expected, String url) {
        WebException e = assertThrows(WebException.class, () -> guard.check(url), url);
        assertEquals(expected, e.getMessage(), url);
    }
}
