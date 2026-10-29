"""Build the resource pack folder and zip.

    python3 build_pack.py <out_dir>

Writes <out_dir>/pack/ (the unzipped pack) and <out_dir>/CreateHD.zip.
Target: Minecraft 1.21.1 (pack_format 34), Create 6.0.10, Iris + LabPBR, Continuity.
"""
import json
import os
import shutil
import sys

from PIL import Image

import blocks
from render import cube

PACK_FORMAT = 34  # Minecraft 1.21 - 1.21.1
DESCRIPTION = "Create HD: 256x LabPBR textures for Create (samples)"


def main(out):
    pack = os.path.join(out, "pack")
    if os.path.isdir(pack):
        shutil.rmtree(pack)
    texdir = os.path.join(pack, "assets", "create", "textures", "block")
    built = blocks.build(texdir)
    with open(os.path.join(pack, "pack.mcmeta"), "w") as f:
        json.dump({"pack": {"pack_format": PACK_FORMAT, "description": DESCRIPTION}}, f, indent=2)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "README.md"), os.path.join(pack, "README.md"))
    icon = cube(built["andesite_casing"], size=256)
    icon.resize((128, 128), Image.LANCZOS).save(os.path.join(pack, "pack.png"))
    zpath = shutil.make_archive(os.path.join(out, "CreateHD"), "zip", pack)
    print(zpath, f"{os.path.getsize(zpath) / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
