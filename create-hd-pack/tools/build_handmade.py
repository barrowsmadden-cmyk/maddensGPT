"""Write the hand-made textures (tools and weapons so far) into the hand-built
overlay pack, at the item tier size from plan.py.

    python3 build_handmade.py <handbuilt_pack_dir> [size] [src_dir]
"""
import os
import sys

from PIL import Image

from families import load_rgba
from handmade import bricks, tools

SRC = None  # vanilla source root, for palettes taken from the original textures


def main(pack, size=64, src=None):
    n = 0
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
    main(sys.argv[1], *(int(x) for x in sys.argv[2:3]), *sys.argv[3:4])
