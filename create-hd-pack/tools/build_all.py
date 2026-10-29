"""Build every planned texture into the resource pack.

    python3 build_all.py <src_dir> <plan.json> <out_dir> [--only NAMESPACE ...] [--jobs N]

Routes each texture: hand-built Create textures (blocks.py) win, then family
generators (families.py), then the automatic remaster (remaster.py). Writes
<out_dir>/pack/assets/<ns>/textures/... with _n and _s, rescales animation
frame sizes in .mcmeta, and logs failures (those fall back to a plain upscale so
the pack is never missing a texture).
"""
import json
import os
import shutil
import sys
import time
import traceback
import zlib
from multiprocessing import Pool

import numpy as np
from PIL import Image

from families import family_of, generate, load_rgba
from remaster import remaster, remaster_item
from texgen import derive


def out_base(out, t):
    return os.path.join(out, "pack", "assets", t["ns"], t["path"][:-4])


def save_maps(base, maps):
    os.makedirs(os.path.dirname(base), exist_ok=True)
    a8, n8, s8 = maps
    Image.fromarray(a8, "RGBA").save(base + ".png", optimize=False, compress_level=6)
    Image.fromarray(n8, "RGBA").save(base + "_n.png", optimize=False, compress_level=6)
    Image.fromarray(s8, "RGBA").save(base + "_s.png", optimize=False, compress_level=6)


def scale_mcmeta(src_meta, dst_meta, scale):
    meta = json.load(open(src_meta, encoding="utf-8", errors="replace"))
    anim = meta.get("animation")
    if isinstance(anim, dict):
        for k in ("width", "height"):
            if k in anim:
                anim[k] = int(anim[k]) * scale
    json.dump(meta, open(dst_meta, "w"))


def fallback(src_path, scale):
    """Nearest-neighbour upscale with flat normals: used only if a texture fails."""
    im = Image.open(src_path).convert("RGBA")
    big = np.asarray(im.resize((im.width * scale, im.height * scale), Image.NEAREST))
    h, w = big.shape[:2]
    n8 = np.zeros((h, w, 4), np.uint8)
    n8[..., 0] = n8[..., 1] = 128
    n8[..., 2] = n8[..., 3] = 255
    s8 = np.zeros((h, w, 4), np.uint8)
    s8[..., 1] = 10
    s8[..., 3] = 255
    return big, n8, s8


def work(args):
    src, out, t = args
    rel = t["path"]
    src_path = os.path.join(src, "assets", t["ns"], rel)
    base = out_base(out, t)
    seed = zlib.crc32(f"{t['ns']}:{rel}".encode()) % 100000
    route = "remaster"
    err = None
    try:
        fam = family_of(rel)
        maps = None
        if fam and t["kind"] == "block" and t["frames"] == 1 and t["w"] == t["h"]:
            try:
                src_img = load_rgba(src_path)
                m, opts = generate(rel, src_img, t["w"] * t["scale"], seed=seed)
                maps = derive(m, **opts)
                route = "family:" + fam
            except ValueError:
                maps = None  # layout not recognised (e.g. no mortar grid): remaster it
        if maps is not None:
            pass
        elif t["kind"] == "item":
            route = "item"
            maps = remaster_item(src_path, t["scale"], rel, seed=seed)
        else:
            maps = remaster(src_path, t["scale"], rel, seed=seed)
    except Exception:
        err = traceback.format_exc(limit=3)
        route = "fallback"
        maps = fallback(src_path, t["scale"])
    save_maps(base, maps)
    if os.path.exists(src_path + ".mcmeta"):
        try:
            scale_mcmeta(src_path + ".mcmeta", base + ".png.mcmeta", t["scale"])
        except Exception:
            shutil.copy(src_path + ".mcmeta", base + ".png.mcmeta")
    return t["ns"], rel, route, err


def main():
    src, plan_path, out = sys.argv[1:4]
    only, jobs = None, 4
    rest = sys.argv[4:]
    if "--only" in rest:
        i = rest.index("--only")
        only = set(x for x in rest[i + 1:] if not x.startswith("--"))
    paths = None
    if "--paths" in rest:
        # rebuild just these textures, given as ns:textures/...png
        i = rest.index("--paths")
        paths = {tuple(x.split(":", 1)) for x in rest[i + 1:] if not x.startswith("--")}
    if "--jobs" in rest:
        jobs = int(rest[rest.index("--jobs") + 1])
    skip = set()
    if "--handbuilt" in rest:
        # textures the hand-built pack provides are overlaid later; don't build them
        hb = os.path.join(rest[rest.index("--handbuilt") + 1], "assets")
        for root, _, files in os.walk(hb):
            for f in files:
                if f.endswith(".png") and not f.endswith(("_n.png", "_s.png")):
                    rel = os.path.relpath(os.path.join(root, f), hb).split(os.sep)
                    skip.add((rel[0], "/".join(rel[1:])))
    plan = json.load(open(plan_path))["plan"]
    kinds = None
    if "--kind" in rest:
        kinds = {rest[rest.index("--kind") + 1]}
    todo = [t for t in plan if (only is None or t["ns"] in only) and (kinds is None or t["kind"] in kinds)
            and (t["ns"], t["path"]) not in skip and (paths is None or (t["ns"], t["path"]) in paths)]
    print(f"{len(todo)} to build, {len(skip)} provided by the hand-built pack", flush=True)
    # biggest first so the pool finishes evenly
    todo.sort(key=lambda t: -t["texels"])
    log = {"routes": {}, "failures": []}
    t0 = time.time()
    with Pool(jobs) as pool:
        for i, (ns, rel, route, err) in enumerate(pool.imap_unordered(work, [(src, out, t) for t in todo], chunksize=4)):
            log["routes"][route] = log["routes"].get(route, 0) + 1
            if err:
                log["failures"].append(dict(ns=ns, path=rel, error=err))
            if (i + 1) % 250 == 0 or i + 1 == len(todo):
                el = time.time() - t0
                print(f"{i + 1}/{len(todo)}  {el / 60:.1f} min  eta {el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min  "
                      f"failures {len(log['failures'])}", flush=True)
    os.makedirs(out, exist_ok=True)
    name = "build_log.json" if only is None and kinds is None and paths is None else "build_log_partial.json"
    json.dump(log, open(os.path.join(out, name), "w"), indent=1)
    print("routes:", log["routes"])


if __name__ == "__main__":
    main()
