"""Zoomed, gridded view of original textures for tracing: python3 zoom.py <dest.png> <cell_px> <png> ..."""
import sys
from PIL import Image, ImageDraw

dest, Z = sys.argv[1], int(sys.argv[2])
ims = [Image.open(p).convert("RGBA") for p in sys.argv[3:]]
ims = [im.crop((0, 0, im.width, im.width)) for im in ims]
W = sum(im.width * Z + 12 for im in ims)
H = max(im.height * Z for im in ims) + 18
sheet = Image.new("RGB", (W, H), (50, 50, 58))
d = ImageDraw.Draw(sheet)
x = 0
for im, p in zip(ims, sys.argv[3:]):
    big = im.resize((im.width * Z, im.height * Z), Image.NEAREST)
    sheet.paste(big, (x, 0), big)
    for g in range(0, im.width + 1):
        c = (110, 110, 120) if g % 4 == 0 else (70, 70, 80)
        d.line([(x + g * Z, 0), (x + g * Z, im.height * Z)], fill=c)
        d.line([(x, g * Z), (x + im.width * Z, g * Z)], fill=c)
    d.text((x, im.height * Z + 3), p.split("/")[-1], fill=(255, 255, 255))
    x += im.width * Z + 12
sheet.save(dest)
