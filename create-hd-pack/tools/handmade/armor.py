"""Hand-made armour items: helmet, chestplate, leggings, boots.

House style (handmade/style.py). Shapes traced from the vanilla textures
(texel (x, y) spans x..x+1, y..y+1); every material (chainmail, iron, golden,
diamond, netherite) shares them and takes its tones from its own original.
Chainmail gets a crisp ring-mail pattern in the original's tones.
"""
import re

import cv2
import numpy as np

from handmade.tools import Canvas
from handmade.style import part, render_item, tones_from


def helmet(cv, m):
    shell = cv.polygon([(5.0, 3.0), (11.0, 3.0), (13.0, 5.0), (13.0, 11.0), (12.0, 12.0), (10.0, 12.0), (10.0, 11.0),
                        (6.0, 11.0), (6.0, 12.0), (4.0, 12.0), (3.0, 11.0), (3.0, 5.0)])
    opening = cv.polygon([(6.0, 7.0), (10.0, 7.0), (11.0, 8.0), (11.0, 11.0), (5.0, 11.0), (5.0, 8.0)])
    return [part(shell, m, "bevel", bevel=1.0, h=0.88),
            part(opening, m, "flat", h=0.7, glint=False)]


def chestplate(cv, m):
    body = cv.polygon([(1.0, 2.0), (6.0, 2.0), (8.0, 5.0), (10.0, 2.0), (15.0, 2.0), (15.0, 8.0), (13.0, 8.0),
                       (13.0, 13.0), (11.0, 15.0), (5.0, 15.0), (3.0, 13.0), (3.0, 8.0), (1.0, 8.0)])
    collar = cv.stroke((5.4, 6.5), (8.6, 6.5), 0.9)
    return [part(body, m, "bevel", bevel=1.0, h=0.88),
            part(collar, m, "flat", h=0.9, inner_line=False)]


def leggings(cv, m):
    body = cv.polygon([(4.0, 2.0), (12.0, 2.0), (13.0, 3.0), (13.0, 14.0), (9.0, 14.0), (9.0, 7.0), (7.0, 7.0),
                       (7.0, 14.0), (3.0, 14.0), (3.0, 3.0)])
    band = cv.polygon([(3.6, 2.6), (12.4, 2.6), (12.4, 4.2), (3.6, 4.2)])
    return [part(body, m, "bevel", bevel=1.0, h=0.86),
            part(band, m, "flat", h=0.9, inner_line=False)]


def boots(cv, m):
    left = cv.polygon([(4.0, 3.0), (7.0, 3.0), (7.0, 12.0), (5.0, 12.0), (5.0, 13.0), (1.0, 13.0), (1.0, 10.0),
                       (2.0, 9.0), (3.0, 9.0), (3.0, 4.0)])
    right = cv.polygon([(9.0, 3.0), (12.0, 3.0), (13.0, 4.0), (13.0, 9.0), (14.0, 9.0), (15.0, 10.0), (15.0, 13.0),
                        (11.0, 13.0), (11.0, 12.0), (9.0, 12.0)])
    return [part(left, m, "bevel", bevel=0.9, h=0.86), part(right, m, "bevel", bevel=0.9, h=0.86)]


PIECES = {"helmet": helmet, "chestplate": chestplate, "leggings": leggings, "boots": boots}
MAT = {"chainmail": "iron", "iron": "iron", "golden": "gold", "diamond": "gem", "netherite": "dark_metal"}
PATTERN = re.compile(r"^(chainmail|iron|golden|diamond|netherite)_(helmet|chestplate|leggings|boots)$")


def armor_of(path, ns):
    name = path.split("/")[-1].rsplit(".", 1)[0]
    if ns != "minecraft" or not path.startswith("textures/item/"):
        return None
    m = PATTERN.match(name)
    return (m.group(1), m.group(2)) if m else None


def _rings(cv, cover, scale=1.0):
    """Ring-mail: rows of interlocking links (ellipses), alternate rows offset
    half a link; returns the link and gap masks."""
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    px, py = 1.7 * scale, 1.3 * scale
    row = np.floor(yy / py)
    fx = ((xx / px + 0.5 * (row % 2)) % 1 - 0.5) * px
    fy = ((yy / py) % 1 - 0.5) * py
    r = np.hypot(fx / (0.78 * scale), fy / (0.6 * scale))
    ring = (np.abs(r - 1.0) < 0.36) & cover
    return ring, cover & ~ring


def render(path, src, size):
    name = path.split("/")[-1].rsplit(".", 1)[0]
    mat, piece = PATTERN.match(name).groups()
    cv = Canvas(size)
    parts = PIECES[piece](cv, MAT[mat])
    if mat == "chainmail":
        # mail over the armour's own light and shade: links catch light, gaps fall dark
        scale = 0.8 if piece in ("boots", "helmet") else 1.0
        for p in parts:
            if p["shade"] != "bevel":
                continue
            inner = cv2.erode(p["mask"].astype(np.uint8), np.ones((int(cv.k * 1.2),) * 2, np.uint8)) > 0
            ring, gap = _rings(cv, inner, scale)
            p["overlay"] = [(ring, 1.1, 0.02), (gap, 0.66, -0.03)]
    return render_item(parts, cv, src, size)
