"""Contact sheet: albedo | normal | specular | height | lit tile | lit block."""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import blocks
from render import tile_view, cube, render_box

FONT = ImageFont.load_default(size=16)


def checker(w, h, s=16):
    yy, xx = np.mgrid[0:h, 0:w]
    v = np.where(((xx // s) + (yy // s)) % 2, 60, 48).astype(np.uint8)
    return Image.fromarray(np.dstack([v, v, v, np.full_like(v, 255)]), "RGBA")


def channel_views(maps):
    a8, n8, s8 = maps
    alb = Image.fromarray(a8, "RGBA")
    nrm = Image.fromarray(np.dstack([n8[..., 0], n8[..., 1], np.full_like(n8[..., 0], 255), a8[..., 3]]), "RGBA")
    spec = Image.fromarray(np.dstack([s8[..., 0], s8[..., 1], s8[..., 2] * 3, a8[..., 3]]).clip(0, 255).astype(np.uint8), "RGBA")
    hgt = Image.fromarray(np.dstack([n8[..., 3]] * 3 + [a8[..., 3]]), "RGBA")
    return [alb, nrm, spec, hgt]


def main(outdir, sheet_path):
    built = blocks.build(outdir)
    rows = []
    names = ["andesite_casing", "brass_casing", "copper_casing", "industrial_iron_block", "brass_block"]
    for name in names:
        maps = built[name]
        rows.append((name, channel_views(maps) + [tile_view(maps)], cube(maps, 300)))
    shaft = render_box({
        "up": (built["axis_top"], (6, 6, 10, 10)),
        "south": (built["axis"], (6, 0, 10, 16)),
        "east": (built["axis"], (6, 0, 10, 16)),
    }, size=300, scale=10, frm=(6, 0, 6), to=(10, 16, 10))
    rows.append(("axis / axis_top (shaft)", channel_views(built["axis"]) + [tile_view(built["axis"])], shaft))

    labels = ["albedo", "_n normal (RG)", "_s smooth/F0/poro", "_n height (A)", "lit (raking light)", "lit block"]
    cw, pad, head = 256, 12, 28
    W = pad + 5 * (cw + pad) + 300 + pad
    H = head + len(rows) * (cw + head + pad) + pad
    sheet = Image.new("RGBA", (W, H), (28, 28, 30, 255))
    dr = ImageDraw.Draw(sheet)
    for i, lab in enumerate(labels):
        x = pad + i * (cw + pad)
        dr.text((x, 6), lab, fill=(200, 200, 205, 255), font=FONT)
    y = head
    for name, tiles, iso in rows:
        dr.text((pad, y + 4), name, fill=(255, 214, 120, 255), font=FONT)
        y += head
        for i, t in enumerate(tiles):
            x = pad + i * (cw + pad)
            bg = checker(cw, cw)
            bg.alpha_composite(t.resize((cw, cw)))
            sheet.alpha_composite(bg, (x, y))
        sheet.alpha_composite(iso, (pad + 5 * (cw + pad), y - 22))
        y += cw + pad
    sheet.convert("RGB").save(sheet_path)
    print("sheet", sheet_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
