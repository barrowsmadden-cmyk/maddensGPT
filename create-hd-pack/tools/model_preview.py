"""Create models rendered with the original textures and with the HD pack's.

    python3 model_preview.py <src_assets> <pack_dir> <dest.png> model_id ... [--size 220] [--cols 4]
"""
import sys

from PIL import Image, ImageDraw

from mcmodel import render

BG = (62, 86, 110)


def main(src, pack, dest, models, size=220, cols=4):
    cells = []
    for m in models:
        a = render(m, src, src, size)
        b = render(m, src, [f"{p}/assets" for p in pack.split(",")], size)
        cell = Image.new("RGB", (size * 2 + 6, size + 16), BG)
        cell.paste(a, (0, 0), a)
        cell.paste(b, (size + 6, 0), b)
        ImageDraw.Draw(cell).text((4, size + 2), m.split(":")[1][:40] + "   (original | HD)", fill=(230, 230, 230))
        cells.append(cell)
    cw, ch = cells[0].width + 8, cells[0].height + 8
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cw * min(cols, len(cells)), ch * rows), (30, 30, 34))
    for i, c in enumerate(cells):
        sheet.paste(c, ((i % cols) * cw, (i // cols) * ch))
    sheet.save(dest)
    print(dest)


if __name__ == "__main__":
    args = sys.argv[1:]
    size = int(args[args.index("--size") + 1]) if "--size" in args else 220
    cols = int(args[args.index("--cols") + 1]) if "--cols" in args else 4
    args = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))]
    main(args[0], args[1], args[2], args[3:], size, cols)
