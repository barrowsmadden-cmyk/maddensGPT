"""Extract every block/item texture (plus models, blockstates and animation
metadata) from the vanilla client jar and the downloaded mod jars.

    python3 extract.py <download_dir> <out_dir>

Nested jar-in-jar files (e.g. Create Aeronautics bundles) are opened too.
Writes <out_dir>/assets/<namespace>/... and <out_dir>/index.json with one entry
per texture: namespace, path, kind (block/item), size, frames, source mod.
Only data files are read; nothing from the jars is executed.
"""
import io
import json
import os
import sys
import zipfile

from PIL import Image

KEEP_DIRS = ("textures/block/", "textures/item/", "models/", "blockstates/", "atlases/",
             "optifine/", "fusion/")
PBR_SUFFIXES = ("_n.png", "_s.png", "_e.png")


def wanted(path):
    parts = path.split("/")
    if len(parts) < 3 or parts[0] != "assets":
        return False
    rest = "/".join(parts[2:])
    return rest.startswith(KEEP_DIRS)


def safe_join(root, member):
    dest = os.path.normpath(os.path.join(root, member))
    if not dest.startswith(os.path.normpath(root) + os.sep):
        raise ValueError(f"unsafe path in jar: {member}")
    return dest


def walk(z, mod, out, index, owners):
    names = set(z.namelist())
    for n in z.namelist():
        if n.endswith(".jar"):
            walk(zipfile.ZipFile(io.BytesIO(z.read(n))), mod, out, index, owners)
            continue
        if n.endswith("/") or not wanted(n):
            continue
        dest = safe_join(out, n)
        if n in owners and owners[n] != mod:
            # two jars ship the same file (compat textures); first one wins, note it
            index.setdefault("_conflicts", []).append([n, owners[n], mod])
            continue
        owners[n] = mod
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(z.read(n))
        p = n.split("/")
        if p[2] == "textures" and p[3] in ("block", "item") and n.endswith(".png") and not n.endswith(PBR_SUFFIXES):
            try:
                w, h = Image.open(dest).size
            except Exception:
                continue
            meta = None
            if n + ".mcmeta" in names:
                try:
                    meta = json.loads(z.read(n + ".mcmeta").decode("utf-8", "replace"))
                except Exception:
                    meta = {}
            frames = 1
            if meta is not None and "animation" in meta:
                fw = meta["animation"].get("width", min(w, h))
                fh = meta["animation"].get("height", fw)
                frames = max(1, (w // max(fw, 1)) * (h // max(fh, 1)))
            index["textures"].append(dict(ns=p[1], path="/".join(p[2:]), kind=p[3], w=w, h=h,
                                          frames=frames, mod=mod))


def main(dl, out):
    os.makedirs(out, exist_ok=True)
    res = json.load(open(os.path.join(dl, "mods", "resolved.json")))
    jars = [("Minecraft 1.21.1", os.path.join(dl, "vanilla", "client-1.21.1.jar"))]
    jars += [(name, r["path"]) for name, r in res["resolved"].items()]
    index, owners = {"textures": []}, {}
    for mod, path in jars:
        walk(zipfile.ZipFile(path), mod, out, index, owners)
        print("extracted", mod)
    index["missing_mods"] = res["missing"]
    json.dump(index, open(os.path.join(out, "index.json"), "w"), indent=0)
    print(len(index["textures"]), "textures,", len(index.get("_conflicts", [])), "duplicate files skipped")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
