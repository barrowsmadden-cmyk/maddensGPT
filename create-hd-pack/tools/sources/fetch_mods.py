"""Resolve the pack's mods on Modrinth at their exact versions and download them.

    python3 fetch_mods.py <download_dir>

Writes <download_dir>/resolved.json. A mod resolves only when a Modrinth version for
Minecraft 1.21.1 + NeoForge has a version number containing the listed version, so a
near-miss never stands in for the real file. Unresolved mods are listed at the end.
"""
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

API = "https://api.modrinth.com/v2"
HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "maddensGPT/create-hd-pack texture tooling"}


def get(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def versions(slug):
    q = urllib.parse.urlencode({"game_versions": '["1.21.1"]', "loaders": '["neoforge"]'})
    return get(f"{API}/project/{slug}/version?{q}")


def _tokens(s):
    return [t for t in re.split(r"[+\-_\s]+", s.lower()) if t]


def matches(v, want):
    """True when the listed version appears as whole tokens in the version number,
    e.g. "6.0.10" in "6.0.10+mc1.21.1" or "mc1.21.1-6.0.10", but not "6.0.1"."""
    have, w = _tokens(v["version_number"]), _tokens(want)
    return any(have[i:i + len(w)] == w for i in range(len(have) - len(w) + 1))


def resolve(name, version):
    hits = get(f"{API}/search?" + urllib.parse.urlencode({
        "query": name, "limit": 8,
        "facets": json.dumps([["versions:1.21.1"], ["project_type:mod"]])}))["hits"]
    for h in hits:
        for v in versions(h["slug"]):
            if matches(v, version):
                f = next((f for f in v["files"] if f["primary"]), v["files"][0])
                return dict(slug=h["slug"], title=h["title"], version=v["version_number"],
                            url=f["url"], filename=f["filename"], sha512=f["hashes"]["sha512"])
    return None


def main(out):
    os.makedirs(out, exist_ok=True)
    mods = json.load(open(os.path.join(HERE, "modlist.json")))
    resolved, missing = {}, []
    for name, version, namespaces in mods:
        r = resolve(name, version)
        if not r:
            missing.append(f"{name} {version}")
            print("MISSING ", name, version)
            continue
        path = os.path.join(out, r["filename"])
        if not os.path.exists(path):
            urllib.request.urlretrieve(r["url"], path)
        if hashlib.sha512(open(path, "rb").read()).hexdigest() != r["sha512"]:
            raise SystemExit(f"hash mismatch {path}")
        r.update(name=name, wanted=version, namespaces=namespaces, path=path)
        resolved[name] = r
        print("ok      ", name, "->", r["slug"], r["version"])
    json.dump(dict(resolved=resolved, missing=missing), open(os.path.join(out, "resolved.json"), "w"), indent=1)
    print(len(resolved), "resolved,", len(missing), "missing")


if __name__ == "__main__":
    main(sys.argv[1])
