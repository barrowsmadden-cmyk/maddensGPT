"""Finish the full pack: overlay hand-built textures, add metadata, validate, zip.

    python3 assemble.py <plan.json> <full_dir> <handbuilt_pack_dir>

Validation checks every planned texture has its colour, _n and _s files at the
planned size, that _n height (alpha) never drops to 0 on visible pixels, and that
_s uses only valid LabPBR values. Writes validation.json next to the pack.
"""
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

PACK_FORMAT = 34  # Minecraft 1.21 - 1.21.1
HERE = os.path.dirname(os.path.abspath(__file__))


def overlay(handbuilt, pack):
    n = 0
    src_assets = os.path.join(handbuilt, "assets")
    for root, _, files in os.walk(src_assets):
        for f in files:
            s = os.path.join(root, f)
            d = os.path.join(pack, "assets", os.path.relpath(s, src_assets))
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy(s, d)
            n += 1
    return n


def validate(plan, pack, handbuilt_paths):
    problems = []
    for t in plan:
        base = os.path.join(pack, "assets", t["ns"], t["path"][:-4])
        key = f"{t['ns']}/{t['path']}"
        try:
            a = Image.open(base + ".png")
            n = Image.open(base + "_n.png")
            s = Image.open(base + "_s.png")
        except FileNotFoundError as e:
            problems.append(f"missing {key}: {e.filename}")
            continue
        want = (t["w"] * t["scale"], t["h"] * t["scale"])
        if key not in handbuilt_paths and a.size != want:
            problems.append(f"size {key}: {a.size} != {want}")
        if not (a.size == n.size == s.size):
            problems.append(f"map sizes differ {key}")
            continue
        a8, n8, s8 = (np.asarray(x.convert("RGBA")) for x in (a, n, s))
        vis = a8[..., 3] > 0
        if (n8[..., 3][vis] < 1).any():
            problems.append(f"height 0 on visible pixels {key}")
        g = s8[..., 1][vis]
        if g.size and np.isin(g[g >= 230], [230, 231, 232, 233, 234, 235, 236, 237, 255], invert=True).any():
            problems.append(f"unknown metal id {key}")
    return problems


def main(plan_path, full, handbuilt):
    plan = json.load(open(plan_path))["plan"]
    pack = os.path.join(full, "pack")
    hb = set()
    for root, _, files in os.walk(os.path.join(handbuilt, "assets")):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), os.path.join(handbuilt, "assets"))
            parts = rel.split(os.sep)
            hb.add(f"{parts[0]}/{'/'.join(parts[1:])}")
    print("hand-built files overlaid:", overlay(handbuilt, pack))
    with open(os.path.join(pack, "pack.mcmeta"), "w") as f:
        json.dump({"pack": {"pack_format": PACK_FORMAT,
                            "description": "Create HD: 256x/64x LabPBR for the Create+ pack"}}, f, indent=2)
    for name in ("pack.png",):
        if os.path.exists(os.path.join(handbuilt, name)):
            shutil.copy(os.path.join(handbuilt, name), os.path.join(pack, name))
    shutil.copy(os.path.join(HERE, "..", "README.md"), os.path.join(pack, "README.md"))
    problems = validate(plan, pack, hb)
    json.dump(problems, open(os.path.join(full, "validation.json"), "w"), indent=1)
    print(len(problems), "validation problems")
    for p in problems[:15]:
        print("  ", p)


if __name__ == "__main__":
    main(*sys.argv[1:4])
