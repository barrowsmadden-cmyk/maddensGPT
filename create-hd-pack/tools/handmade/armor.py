"""Hand-made armour items: helmet, chestplate, leggings, boots.

Shapes follow the vanilla item silhouettes (16-unit grid); colours come from the
texture being replaced, so chainmail/iron/golden/diamond/netherite (and mod
armour using the same names) all share one design per piece.
"""
import math
import re

import cv2
import numpy as np

from texgen import Material, derive, IRON, GOLD, ALBEDO_METAL, DIELECTRIC
from handmade.tools import Canvas, _shade_part
from handmade.items import palette_from, material_of, _outline, LUMW


def _seam(cv, pts, dark, width=0.45):
    """A thin recessed seam line drawn over a part."""
    m = cv.blank()
    p = cv.pts(pts)
    cv2.polylines(m, [p], False, 255, thickness=max(1, int(round(width * cv.k))), lineType=cv2.LINE_8)
    col = np.broadcast_to(np.array(dark, float) / 255 * 0.85, (cv.n, cv.n, 3))
    return col, np.full((cv.n, cv.n), 0.6), m > 0


def _chain(col, mask, cv, dark):
    """Ring-mail pattern: small rings in offset rows, dark gaps between."""
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    pitch = 1.15
    ry = yy / pitch
    row = np.floor(ry)
    xo = xx / pitch + 0.5 * (row % 2)
    fx, fy = xo - np.floor(xo) - 0.5, ry - row - 0.5
    r = np.hypot(fx, fy)
    ring = np.exp(-((r - 0.32) / 0.09) ** 2)
    gap = r < 0.18
    d = np.array(dark, float) / 255
    out = col * (0.75 + 0.45 * ring[..., None])
    out = np.where((gap & mask)[..., None], d[None, None, :] * 0.8, out)
    return np.clip(out, 0, 1), ring


def helmet(cv, pal):
    light, mid, dark = pal
    import math
    dome = [(8.0 + 5.0 * math.cos(a), 7.6 - 4.9 * math.sin(a)) for a in np.linspace(math.pi, 0, 20)]
    shell = cv.polygon([(3.0, 11.6)] + dome + [(13.0, 11.6), (10.8, 11.6), (10.8, 8.2), (5.2, 8.2), (5.2, 11.6)])
    inside = cv.polygon([(5.4, 8.0), (10.6, 8.0), (10.6, 11.6), (5.4, 11.6)])
    d = tuple(np.array(dark) * 0.45)
    return [
        (np.broadcast_to(np.array(d) / 255, (cv.n, cv.n, 3)).copy(), np.full((cv.n, cv.n), 0.4), inside > 0),
        _shade_part(shell, cv, pal, 1.6),
        _seam(cv, [(3.3, 7.3), (12.7, 7.3)], dark, 0.4),
    ]


def chestplate(cv, pal):
    body = cv.polygon([(1.0, 2.0), (6.0, 2.0), (6.3, 3.7), (8.0, 5.2), (9.7, 3.7), (10.0, 2.0), (15.0, 2.0),
                       (15.0, 7.8), (13.0, 8.3), (13.0, 13.0), (11.8, 14.2), (4.2, 14.2), (3.0, 13.0),
                       (3.0, 8.3), (1.0, 7.8)])
    _, _, dark = pal
    return [
        _shade_part(body, cv, pal, 2.2),
        _seam(cv, [(8.0, 5.4), (8.0, 13.8)], dark, 0.35),
        _seam(cv, [(3.2, 8.2), (5.2, 6.6)], dark, 0.35),
        _seam(cv, [(12.8, 8.2), (10.8, 6.6)], dark, 0.35),
        _seam(cv, [(3.4, 11.6), (12.6, 11.6)], dark, 0.35),
    ]


def leggings(cv, pal):
    body = cv.polygon([(3.0, 2.0), (13.0, 2.0), (13.0, 13.6), (9.4, 13.6), (9.4, 7.0), (8.0, 6.3), (6.6, 7.0),
                       (6.6, 13.6), (3.0, 13.6)])
    _, _, dark = pal
    return [
        _shade_part(body, cv, pal, 1.5),
        _seam(cv, [(3.2, 4.4), (12.8, 4.4)], dark, 0.4),
        _seam(cv, [(8.0, 2.2), (8.0, 4.4)], dark, 0.3),
    ]


def boots(cv, pal):
    left = cv.polygon([(3.0, 3.0), (6.6, 3.0), (6.6, 12.1), (1.3, 12.4), (1.3, 10.4), (3.0, 9.0)])
    right = cv.polygon([(9.4, 3.0), (13.0, 3.0), (13.0, 9.0), (14.7, 10.4), (14.7, 12.4), (9.4, 12.1)])
    _, _, dark = pal
    return [
        _shade_part(left, cv, pal, 1.3),
        _shade_part(right, cv, pal, 1.3),
    ]


PIECES = {"helmet": helmet, "chestplate": chestplate, "leggings": leggings, "boots": boots}
PATTERN = re.compile(r"^(chainmail|iron|golden|diamond|netherite)_(helmet|chestplate|leggings|boots)$")


def armor_of(path, ns):
    name = path.split("/")[-1].rsplit(".", 1)[0]
    if ns != "minecraft" or not path.startswith("textures/item/"):
        return None
    m = PATTERN.match(name)
    return (m.group(1), m.group(2)) if m else None


def render(path, src, size):
    name = path.split("/")[-1].rsplit(".", 1)[0]
    mat, piece = PATTERN.match(name).groups()
    pal = palette_from(src)
    cv = Canvas(size)
    parts = PIECES[piece](cv, pal)
    col = np.zeros((cv.n, cv.n, 3))
    h = np.zeros((cv.n, cv.n))
    cover = np.zeros((cv.n, cv.n), bool)
    for c, hh, m in parts:
        col = np.where(m[..., None], c, col)
        h = np.where(m, hh, h)
        cover |= m
    if mat == "chainmail":
        col, ring = _chain(col, cover, cv, pal[2])
        h = h + 0.05 * ring * cover
    rim, rim_col = _outline(cover, cv, pal[2])
    col = np.where(rim[..., None], rim_col[None, None, :], col)
    h = np.where(rim, h - 0.05, h)
    down = lambda a: cv2.resize(a.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    alpha = down(cover.astype(np.float32))
    rgb = np.dstack([down(col[..., i] * cover) for i in range(3)]) / np.maximum(alpha, 1e-4)[..., None]
    height = down(h * cover) / np.maximum(alpha, 1e-4)
    m = Material(size)
    solid = alpha > 0.5
    m.albedo, m.alpha = np.clip(rgb, 0, 1), alpha
    m.height = np.where(solid, np.clip(height + (0.99 - height[solid].max()), 0, 0.99), 0.0)
    f0, sm = {"chainmail": (IRON, 0.6), "iron": (IRON, 0.7), "golden": (GOLD, 0.8),
              "diamond": (44, 0.9), "netherite": (ALBEDO_METAL, 0.62)}[mat]
    lum = rgb @ LUMW
    metal = solid & (lum > 0.08)
    m.f0 = np.where(metal, f0, DIELECTRIC).astype(float)
    m.smooth = np.where(solid, sm, 0.0)
    return derive(m, edge_wear=0.0, edges="reflect", normal_strength=4.0)
