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
    """(original, HD) thumbnails. Items show the HD as the game draws it in hand
    (extruded sprite, nearest-magnified); blocks show a lit tile."""
    rel = t["path"][:-4]
    base = f"{pack}/assets/{t['ns']}/{rel}"
    maps = [np.asarray(Image.open(base + s + ".png").convert("RGBA")) for s in ("", "_n", "_s")]
    w = maps[0].shape[1]
    maps = [m[:w] for m in maps]
    if t["kind"] == "item":
        from ingame_preview import in_hand, to8
        hd = Image.fromarray(to8(in_hand(maps[0].astype(np.float32) / 255, size)))
    else:
        lit = tile_view(maps)
        hd = Image.new("RGBA", lit.size, (40, 40, 44, 255))
        hd.alpha_composite(lit)
        hd = hd.resize((size, size), Image.LANCZOS)
    orig = Image.open(f"{src}/assets/{t['ns']}/{t['path']}").convert("RGBA")
    orig = orig.crop((0, 0, orig.width, orig.width))
    ob = Image.new("RGBA", orig.size, (40, 40, 44, 255))
    ob.alpha_composite(orig)
    return (uri(ob.resize((size, size), Image.NEAREST), "PNG", optimize=True),
            uri(hd.convert("RGB"), quality=85), w)


CASINGS = {"andesite_casing", "brass_casing", "copper_casing", "industrial_iron_block", "brass_block"}
MODEL_SHOWCASE = """item_vault fluid_tank/block_single_window gearbox/block encased_fan/block mechanical_press/block
mechanical_mixer/block basin/block millstone/block depot/block deployer/horizontal chute/block smart_chute/block
spout/block item_drain mechanical_pump/block mechanical_saw/horizontal mechanical_drill/block mechanical_crafter/block
steam_engine/block weighted_ejector/block cogwheel large_cogwheel windmill_bearing mechanical_bearing clutch/block
gearshift/block encased_chain_drive/single fluid_valve/block_horizontal_open funnel/block_horizontal
belt_tunnel/window cart_assembler/block packager/block linear_chassis rope_pulley/block content_observer/block
threshold_switch/block_wall portable_storage_interface/block sequenced_gearshift/idle copper_tiles
weathered_copper_shingles red_postbox_open yellow_toolbox red_table_cloth seat rose_quartz_lamp metal_girder/block_x
copper_scaffolding refined_radiance_casing shadow_steel_casing track_station/block""".split()


def model_cards(src_assets, pack_assets, size=128):
    """Create's machines rendered on their models: original textures | HD."""
    from mcmodel import render
    cards = []
    for m in MODEL_SHOWCASE:
        mid = "create:block/" + m
        try:
            a = render(mid, src_assets, src_assets, size * 2)
            b = render(mid, src_assets, [pack_assets], size * 2)
        except Exception:
            continue
        def card(img):
            bg = Image.new("RGBA", img.size, (62, 86, 110, 255))
            bg.alpha_composite(img)
            return bg.resize((size, size), Image.LANCZOS)
        cards.append(dict(name=m.replace("/block", "").replace("/", " "), kind="model", tier="block", px=256,
                          orig=uri(card(a), quality=88), hd=uri(card(b), quality=88)))
    return cards


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
        if t["kind"] == "block" and t["ns"] == "create" and "/vault/" not in t["path"] and \
                os.path.basename(t["path"])[:-4] not in CASINGS:
            continue                       # shown on their models below
        if t["kind"] == "block":
            g = "Hand-made: blocks"
        elif t["path"].endswith("bucket.png"):
            g = "Hand-made: buckets (sample)"
        elif t["ns"] != "minecraft":
            g = "Hand-made: Create & addon items"
        else:
            g = "Hand-made: vanilla items"
        groups[g].append(t)
    mc = model_cards(os.path.join(src, "assets"), os.path.join(pack, "assets"))
    mods.append(dict(mod="Hand-made: Create machines (on their models)", count=len(mc), cards=mc))
    print("models", len(mc))
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
