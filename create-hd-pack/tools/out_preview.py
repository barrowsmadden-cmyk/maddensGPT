"""Grid of built textures (lit render) straight from a built pack, for review."""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from render import tile_view

FONT = ImageFont.load_default(size=12)


def main(pack_assets, ns, dest, names, cell=192):
    cols = 6
    rows = (len(names) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cell + 8) + 8, rows * (cell + 26) + 8), (28, 28, 30))
    dr = ImageDraw.Draw(sheet)
    for i, rel in enumerate(names):
        base = f"{pack_assets}/{ns}/textures/{rel[:-4] if rel.endswith('.png') else rel}"
        try:
            maps = [np.asarray(Image.open(base + s + ".png").convert("RGBA")) for s in ("", "_n", "_s")]
        except FileNotFoundError:
            continue
        w = maps[0].shape[1]
        maps = [m[:w] for m in maps]  # first frame of strips
        lit = tile_view(maps)
        bg = Image.new("RGBA", lit.size, (48, 48, 52, 255))
        bg.alpha_composite(lit)
        x, y = 8 + (i % cols) * (cell + 8), 8 + (i // cols) * (cell + 26)
        sheet.paste(bg.convert("RGB").resize((cell, cell), Image.LANCZOS), (x, y + 16))
        dr.text((x, y), rel.split("/")[-1][:30] + f"  {w}px", fill=(255, 214, 120), font=FONT)
    sheet.save(dest)
    print(dest)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
