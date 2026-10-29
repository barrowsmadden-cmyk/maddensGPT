"""Write the hand-made textures into the hand-built
overlay pack, at the item tier size from plan.py.

    python3 build_handmade.py <handbuilt_pack_dir> [size] [src_dir] [plan.json]
"""
import os
import sys

from PIL import Image

from families import load_rgba
from handmade import bricks, tools

SRC = None  # vanilla source root, for palettes taken from the original textures


def write(pack, ns, rel, maps):
    base = os.path.join(pack, "assets", ns, rel[:-4])
    os.makedirs(os.path.dirname(base), exist_ok=True)
    for suffix, arr in zip(("", "_n", "_s"), maps):
        Image.fromarray(arr, "RGBA").save(base + suffix + ".png")


def main(pack, size=64, src=None, plan=None):
    n = 0
    if src and plan:
        import json
        from handmade import armor, items
        from handmade import buckets, create_items, glass, rocks, storage, surfaces
        buckets.prepare(os.path.join(src, "assets"))
        for t in json.load(open(plan))["plan"]:
            if t["kind"] == "block" and t["frames"] == 1 and rocks.rock_of(t["path"], t["ns"]):
                img = load_rgba(os.path.join(src, "assets", t["ns"], t["path"]))
                write(pack, t["ns"], t["path"], rocks.render(t["path"], img, t["w"] * t["scale"], t["ns"]))
                n += 1
                continue
            if t["kind"] == "block" and t["frames"] == 1 and storage.storage_of(t["path"], t["ns"]):
                img = load_rgba(os.path.join(src, "assets", t["ns"], t["path"]))
                write(pack, t["ns"], t["path"], storage.render(t["path"], img, t["w"] * t["scale"], t["ns"]))
                n += 1
                continue
            if t["kind"] == "block" and t["frames"] == 1 and glass.glass_of(t["path"], t["ns"]):
                img = load_rgba(os.path.join(src, "assets", t["ns"], t["path"]))
                write(pack, t["ns"], t["path"], glass.render(t["path"], img, t["w"] * t["scale"]))
                n += 1
                continue
            if t["kind"] == "block" and t["frames"] == 1 and surfaces.surface_of(t["path"], t["ns"]):
                img = load_rgba(os.path.join(src, "assets", t["ns"], t["path"]))
                write(pack, t["ns"], t["path"], surfaces.render(t["path"], img, t["w"] * t["scale"], t["ns"]))
                n += 1
                continue
            if t["kind"] != "item" or t["frames"] > 1 or t["w"] != t["h"]:
                continue
            if t["ns"] == "minecraft" and any(t["path"] == rel for rel, *_ in tools.items()):
                continue
            if t["path"].endswith("bucket.png"):
                img = load_rgba(os.path.join(src, "assets", t["ns"], t["path"]))
                if buckets.bucket_of(t["path"], t["ns"], img):
                    write(pack, t["ns"], t["path"], buckets.render(t["path"], img, t["w"] * t["scale"], t["ns"]))
                    n += 1
                    continue
            if create_items.create_item_of(t["path"], t["ns"]) is not None:
                img = load_rgba(os.path.join(src, "assets", t["ns"], t["path"]))
                write(pack, t["ns"], t["path"], create_items.render(t["path"], img, t["w"] * t["scale"], t["ns"]))
                n += 1
                continue
            if items.archetype_of(t["path"], t["ns"]) is not None:
                renderer = items.render
            elif armor.armor_of(t["path"], t["ns"]) is not None:
                renderer = armor.render
            else:
                continue
            img = load_rgba(os.path.join(src, "assets", t["ns"], t["path"]))
            write(pack, t["ns"], t["path"], renderer(t["path"], img, t["w"] * t["scale"]))
            n += 1
    if src:
        for rel, layout, opts in bricks.textures():
            base = os.path.join(pack, "assets", "minecraft", rel[:-4])
            os.makedirs(os.path.dirname(base), exist_ok=True)
            maps = bricks.brick_wall(load_rgba(os.path.join(src, "assets", "minecraft", rel)), 256, layout, seed=3, **opts)
            for suffix, arr in zip(("", "_n", "_s"), maps):
                Image.fromarray(arr, "RGBA").save(base + suffix + ".png")
            n += 1
    for rel, tool, mat in tools.items():
        base = os.path.join(pack, "assets", "minecraft", rel[:-4])
        os.makedirs(os.path.dirname(base), exist_ok=True)
        a8, n8, s8 = tools.render(tool, mat, size)
        Image.fromarray(a8, "RGBA").save(base + ".png")
        Image.fromarray(n8, "RGBA").save(base + "_n.png")
        Image.fromarray(s8, "RGBA").save(base + "_s.png")
        n += 1
    print(n, "hand-made textures written")


if __name__ == "__main__":
    main(sys.argv[1], *(int(x) for x in sys.argv[2:3]), *sys.argv[3:5])
