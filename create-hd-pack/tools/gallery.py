"""Review gallery: per mod, original vs HD (lit) for a sample of its textures.

    python3 gallery.py <src_dir> <plan.json> <pack_dir> <template.html> <out.html> [per_mod]

Samples are spread across each mod's textures (blocks first, then items) with a
fixed seed so the same textures show on every rebuild.
"""
import base64
import io
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

from render import tile_view


def uri(img, fmt="JPEG", **kw):
    buf = io.BytesIO()
    if fmt == "JPEG":
        img = img.convert("RGB")
    img.save(buf, fmt, **kw)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def thumb_pair(src, pack, t, size=128):
    rel = t["path"][:-4]
    base = f"{pack}/assets/{t['ns']}/{rel}"
    maps = [np.asarray(Image.open(base + s + ".png").convert("RGBA")) for s in ("", "_n", "_s")]
    w = maps[0].shape[1]
    maps = [m[:w] for m in maps]
    lit = tile_view(maps)
    bg = Image.new("RGBA", lit.size, (40, 40, 44, 255))
    bg.alpha_composite(lit)
    orig = Image.open(f"{src}/assets/{t['ns']}/{t['path']}").convert("RGBA")
    orig = orig.crop((0, 0, orig.width, orig.width))
    ob = Image.new("RGBA", orig.size, (40, 40, 44, 255))
    ob.alpha_composite(orig)
    return (uri(ob.resize((size, size), Image.NEAREST), "PNG", optimize=True),
            uri(bg.resize((size, size), Image.LANCZOS), quality=85), w)


def main(src, plan_path, pack, template, out, per_mod=12):
    plan = json.load(open(plan_path))["plan"]
    by_mod = defaultdict(list)
    for t in plan:
        by_mod[t["mod"]].append(t)
    rng = random.Random(7)
    mods = []
    # hand-made textures first, so they're easy to review: everything in the
    # hand-built overlay, grouped; buckets (130+, one design) are sampled
    hand_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "pack", "assets")
    hand = set()
    for root, _, files in os.walk(hand_root):
        for f in files:
            if f.endswith(".png") and not f.endswith(("_n.png", "_s.png")):
                ns, rel = os.path.relpath(os.path.join(root, f), hand_root).split(os.sep, 1)
                hand.add(f"{ns}/{rel.replace(os.sep, '/')}")
    groups = defaultdict(list)
    for t in plan:
        if f"{t['ns']}/{t['path']}" not in hand:
            continue
        if t["kind"] == "block":
            g = "Hand-made: blocks"
        elif t["path"].endswith("bucket.png"):
            g = "Hand-made: buckets (sample)"
        elif t["ns"] != "minecraft":
            g = "Hand-made: Create & addon items"
        else:
            g = "Hand-made: vanilla items"
        groups[g].append(t)
    for g in ("Hand-made: blocks", "Hand-made: Create & addon items", "Hand-made: vanilla items",
              "Hand-made: buckets (sample)"):
        feat = sorted(groups[g], key=lambda t: (t["ns"], t["path"]))
        total = len(feat)
        if "buckets" in g:
            feat = rng.sample(feat, min(24, len(feat)))
        cards = []
        for t in feat:
            try:
                o, h, w = thumb_pair(src, pack, t)
                cards.append(dict(name=t["path"].split("/")[-1][:-4], kind=t["kind"], tier=t["tier"], px=w, orig=o, hd=h))
            except Exception:
                pass
        mods.append(dict(mod=g, count=total, cards=cards))
        print(g, len(cards), "of", total)
    for mod in sorted(by_mod, key=lambda m: (m != "Minecraft 1.21.1", not m.startswith("Create"), m)):
        ts = by_mod[mod]
        blocks = [t for t in ts if t["kind"] == "block" and t["w"] <= 64]
        items = [t for t in ts if t["kind"] == "item"]
        heroes = [t for t in ts if t["tier"] == "hero"]
        pick = rng.sample(heroes, min(len(heroes), per_mod // 3))
        pick += rng.sample([t for t in blocks if t not in pick], min(len(blocks), per_mod - len(pick) - per_mod // 4))
        pick += rng.sample(items, min(len(items), per_mod - len(pick)))
        cards = []
        for t in pick:
            try:
                o, h, w = thumb_pair(src, pack, t)
            except Exception as e:
                continue
            cards.append(dict(name=t["path"].split("/")[-1][:-4], kind=t["kind"], tier=t["tier"], px=w, orig=o, hd=h))
        mods.append(dict(mod=mod, count=len(ts), cards=cards))
        print(mod, len(cards))
    html = open(template).read().replace("/*__GALLERY__*/null", json.dumps(mods))
    open(out, "w").write(html)
    print(out, f"{len(html) / 1e6:.1f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:6], *(int(x) for x in sys.argv[6:7]))
