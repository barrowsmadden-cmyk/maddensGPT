package com.maddensgpt.aiplayers.core.web;

import java.net.Inet4Address;
import java.net.Inet6Address;
import java.net.InetAddress;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.util.List;
import java.util.Locale;

/**
 * Decides which URLs a bot may download. Players can ask the bot to open any link, so only public
 * http(s) sites on standard ports are allowed: anything that resolves to a loopback, private,
 * link-local (including cloud metadata at 169.254.169.254) or otherwise reserved address is refused,
 * which keeps the bot away from the server's own network.
 *
 * <p>The check resolves the host itself; the HTTP client resolves it again when connecting, so a
 * DNS answer that changes between the two lookups is not caught. Redirects are checked one by one.
 */
final class UrlGuard {
    private static final int MAX_URL_LENGTH = 2_048;

    private final List<String> blockedDomains;
    private final boolean allowPrivateAddresses;

    /** @param allowPrivateAddresses only for tests against a local server */
    UrlGuard(List<String> blockedDomains, boolean allowPrivateAddresses) {
        this.blockedDomains = blockedDomains.stream()
                .map(domain -> domain.trim().toLowerCase(Locale.ROOT))
                .map(domain -> domain.startsWith(".") ? domain.substring(1) : domain)
                .toList();
        this.allowPrivateAddresses = allowPrivateAddresses;
    }

    /** Parses and checks a URL given by the model. Blocking: may perform a DNS lookup. */
    URI check(String url) {
        if (url == null || url.isBlank()) {
            throw new WebException("no URL given");
        }
        String trimmed = url.strip();
        if (trimmed.length() > MAX_URL_LENGTH) {
            throw new WebException("the URL is too long");
        }
        if (!trimmed.contains("://")) {
            trimmed = "https://" + trimmed;
        }
        try {
            return check(new URI(trimmed));
        } catch (URISyntaxException e) {
            throw new WebException("that is not a valid URL");
        }
    }

    /** Checks an already-parsed URL, e.g. a redirect target. Blocking: may perform a DNS lookup. */
    URI check(URI uri) {
        String scheme = uri.getScheme() == null ? "" : uri.getScheme().toLowerCase(Locale.ROOT);
        if (!scheme.equals("http") && !scheme.equals("https")) {
            throw new WebException("only http and https links can be opened");
        }
        if (uri.getRawUserInfo() != null) {
            throw new WebException("links with a username or password cannot be opened");
        }
        String host = uri.getHost();
        if (host == null || host.isEmpty()) {
            throw new WebException("the link has no host name");
        }
        int port = uri.getPort();
        if (port != -1 && port != 80 && port != 443 && !allowPrivateAddresses) {
            throw new WebException("only standard web ports (80 and 443) can be opened");
        }

        String bareHost = host.startsWith("[") && host.endsWith("]") ? host.substring(1, host.length() - 1) : host;
        String lowerHost = bareHost.toLowerCase(Locale.ROOT);
        while (lowerHost.endsWith(".")) {
            lowerHost = lowerHost.substring(0, lowerHost.length() - 1);
        }
        for (String domain : blockedDomains) {
            if (lowerHost.equals(domain) || lowerHost.endsWith("." + domain)) {
                throw new WebException("this server does not allow opening " + domain);
            }
        }

        if (!allowPrivateAddresses) {
            InetAddress[] addresses;
            try {
                addresses = InetAddress.getAllByName(bareHost);
            } catch (UnknownHostException e) {
                throw new WebException("could not find the site " + bareHost);
            }
            for (InetAddress address : addresses) {
                if (!isPublic(address)) {
                    throw new WebException("that address is on a private or local network and cannot be opened");
                }
            }
        }
        return uri.normalize();
    }

    /** True for globally routable unicast addresses. */
    static boolean isPublic(InetAddress address) {
        if (address.isAnyLocalAddress() || address.isLoopbackAddress() || address.isLinkLocalAddress()
                || address.isSiteLocalAddress() || address.isMulticastAddress()) {
            return false;
        }
        byte[] bytes = address.getAddress();
        if (address instanceof Inet4Address) {
            return isPublicIpv4(bytes, 0);
        }
        if (address instanceof Inet6Address v6) {
            int b0 = bytes[0] & 0xff;
            int b1 = bytes[1] & 0xff;
            if ((b0 & 0xfe) == 0xfc) {
                return false; // unique local fc00::/7
            }
            if (v6.isIPv4CompatibleAddress()) {
                return isPublicIpv4(bytes, 12); // ::a.b.c.d
            }
            if (b0 == 0x00 && b1 == 0x64 && (bytes[2] & 0xff) == 0xff && (bytes[3] & 0xff) == 0x9b) {
                return isPublicIpv4(bytes, 12); // NAT64 64:ff9b::/96
            }
            if (b0 == 0x20 && b1 == 0x02) {
                return isPublicIpv4(bytes, 2); // 6to4 2002::/16
            }
            if (b0 == 0x20 && b1 == 0x01 && bytes[2] == 0 && bytes[3] == 0) {
                return false; // Teredo 2001::/32 hides the real address
            }
            if (b0 == 0x20 && b1 == 0x01 && (bytes[2] & 0xff) == 0x0d && (bytes[3] & 0xff) == 0xb8) {
                return false; // documentation 2001:db8::/32
            }
            return (b0 & 0xe0) == 0x20; // global unicast is 2000::/3
        }
        return false;
    }

    private static boolean isPublicIpv4(byte[] bytes, int offset) {
        int a = bytes[offset] & 0xff;
        int b = bytes[offset + 1] & 0xff;
        int c = bytes[offset + 2] & 0xff;
        if (a == 0 || a == 10 || a == 127 || a >= 224) {
            return false; // "this" network, private, loopback, multicast and reserved/broadcast
        }
        if (a == 100 && (b & 0xc0) == 64) {
            return false; // carrier-grade NAT 100.64.0.0/10
        }
        if (a == 169 && b == 254) {
            return false; // link-local, includes cloud metadata services
        }
        if (a == 172 && (b & 0xf0) == 16) {
            return false; // private 172.16.0.0/12
        }
        if (a == 192 && b == 168) {
            return false; // private 192.168.0.0/16
        }
        if (a == 192 && b == 0 && (c == 0 || c == 2)) {
            return false; // protocol assignments and documentation
        }
        if (a == 198 && (b == 18 || b == 19)) {
            return false; // benchmarking 198.18.0.0/15
        }
        if ((a == 198 && b == 51 && c == 100) || (a == 203 && b == 0 && c == 113)) {
            return false; // documentation
        }
        return true;
    }
}
