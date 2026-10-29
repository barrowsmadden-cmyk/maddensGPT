"""Contact sheet of remastered textures: original | albedo | normal | lit."""
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import zlib
from remaster import remaster, classify
from families import family_of, generate, load_rgba
from texgen import derive
from render import tile_view

FONT = ImageFont.load_default(size=14)


def main(src_root, dest, scale, rels):
    cw, pad, head = 256, 10, 22
    W = pad + 4 * (cw + pad)
    H = len(rels) * (cw + head + pad) + pad
    sheet = Image.new("RGB", (W, H), (28, 28, 30))
    dr = ImageDraw.Draw(sheet)
    y = pad
    for rel in rels:
        t = time.time()
        fam = family_of(rel)
        if fam:
            try:
                src = load_rgba(f"{src_root}/{rel}")
                generate(rel, src, src.shape[1] * scale, seed=0)
            except ValueError:
                fam = None
        if fam:
            src = load_rgba(f"{src_root}/{rel}")
            m, opts = generate(rel, src, src.shape[1] * scale, seed=zlib.crc32(rel.encode()) % 10000)
            maps = derive(m, **opts)
        else:
            maps = remaster(f"{src_root}/{rel}", scale, rel)
        a8, n8, s8 = maps
        a8, n8, s8 = a8[:a8.shape[1]], n8[:n8.shape[1]], s8[:s8.shape[1]]  # first frame
        dt = time.time() - t
        orig = Image.open(f"{src_root}/{rel}").convert("RGBA")
        orig = orig.crop((0, 0, orig.width, orig.width)).resize((cw, cw), Image.NEAREST)
        tiles = [orig, Image.fromarray(a8, "RGBA"),
                 Image.fromarray(np.dstack([n8[..., 0], n8[..., 1], np.full_like(n8[..., 0], 255), a8[..., 3]]), "RGBA"),
                 tile_view((a8, n8, s8))]
        dr.text((pad, y), f"{rel.split('/')[-1]}   [{('family: ' + fam) if fam else classify(rel)}]   {dt:.1f}s", fill=(255, 214, 120), font=FONT)
        y += head
        for i, im in enumerate(tiles):
            bg = Image.new("RGBA", (cw, cw), (48, 48, 52, 255))
            bg.alpha_composite(im.resize((cw, cw), Image.LANCZOS) if im.width != cw else im)
            sheet.paste(bg.convert("RGB"), (pad + i * (cw + pad), y))
        y += cw + pad
    sheet.save(dest)
    print(dest)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:])
