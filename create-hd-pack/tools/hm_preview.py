"""Original vs hand-made (albedo, lit) for item/block textures handled by a hand-made module.

    python3 hm_preview.py <src_assets_dir> <dest.png> <size> <module> ns/path ...
"""
import importlib
import sys

from PIL import Image

from families import load_rgba
from render import tile_view

BG = (40, 40, 44, 255)


def main(src, dest, size, module, items, cols=8):
    mod = importlib.import_module(module)
    if hasattr(mod, "prepare"):
        mod.prepare(src)
    cells = []
    for it in items:
        ns, rel = it.split("/", 1)
        img = load_rgba(f"{src}/{ns}/{rel}")
        maps = mod.render(rel, img, size, ns)
        o = Image.open(f"{src}/{ns}/{rel}").convert("RGBA")
        o = o.crop((0, 0, o.width, o.width)).resize((192, 192), Image.NEAREST)
        a = Image.new("RGBA", (192, 192), BG); a.alpha_composite(o)
        b = Image.new("RGBA", (192, 192), BG); b.alpha_composite(Image.fromarray(maps[0]).resize((192, 192), Image.LANCZOS))
        lit = tile_view(maps); c = Image.new("RGBA", lit.size, BG); c.alpha_composite(lit)
        cells.append((a, b, c.resize((192, 192), Image.LANCZOS)))
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (min(len(cells), cols) * 200 + 8, rows * 600), (28, 28, 30))
    for i, cs in enumerate(cells):
        for j, c in enumerate(cs):
            sheet.paste(c.convert("RGB"), (8 + (i % cols) * 200, (i // cols) * 600 + 4 + j * 198))
    sheet.save(dest)
    print(dest)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5:])
