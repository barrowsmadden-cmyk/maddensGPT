"""Hand-made item archetypes: ingot, nugget, gems, raw ore, rod.

Each archetype is one hand-authored design in the vanilla 16-unit item grid,
laid out to match the vanilla silhouette. Colours come from the texture being
replaced (its light/mid/dark tones), so a single ingot design serves vanilla
iron/gold/copper/netherite and every mod ingot (brass, zinc, steel, ...).
"""
import math
import re

import cv2
import numpy as np

from texgen import Material, derive, IRON, GOLD, COPPER, ALBEDO_METAL, DIELECTRIC
from handmade.tools import Canvas, _shade_part, HANDLE

LUMW = np.array([0.2126, 0.7152, 0.0722])


def palette_from(src):
    """(light, mid, dark) sRGB 0-255 from a source texture, skipping its outline."""
    rgb, a = src[..., :3], src[..., 3] > 0.5
    px = rgb[a]
    if len(px) == 0:
        return (200, 200, 200), (140, 140, 140), (60, 60, 60)
    lum = px @ LUMW
    order = np.argsort(lum)
    px, lum = px[order], lum[order]
    body = px[int(len(px) * 0.2):] if len(px) > 10 else px  # drop the darkest 20% (outline)
    n = len(body)
    light = body[int(n * 0.8):].mean(0)
    mid = body[int(n * 0.35):int(n * 0.65) + 1].mean(0)
    dark = px[: max(1, int(len(px) * 0.25))].mean(0)
    return tuple((light * 255).round()), tuple((mid * 255).round()), tuple((dark * 255).round())


def material_of(name):
    """(_s green value, smoothness) by name."""
    if re.search(r"gold", name):
        return GOLD, 0.82
    if re.search(r"copper", name):
        return COPPER, 0.7
    if re.search(r"iron|steel|chain", name):
        return IRON, 0.72
    if re.search(r"diamond", name):
        return 44, 0.94
    if re.search(r"emerald|amethyst|quartz|gem|crystal|lapis|ruby|sapphire", name):
        return 40, 0.9
    if re.search(r"netherite|brass|zinc|bronze|lead|tin|nickel|silver|aluminum|electrum|ingot|nugget|sheet|plate", name):
        return ALBEDO_METAL, 0.7
    return DIELECTRIC, 0.35


def _flat(mask, cv, colour, dark, height, edge_w=0.35, grad=0.0, axis=(0, -1)):
    """Flat facet: one colour with a slight gradient, darker rim, given height."""
    c = np.array(colour, float) / 255
    d = np.array(dark, float) / 255
    m = mask > 0
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    g = (xx * axis[0] + yy * axis[1]) / 16.0
    col = c[None, None, :] * (1 + grad * (g - g[m].mean() if m.any() else 0))[..., None]
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5) / cv.k
    rim = dist < edge_w
    col = np.where(rim[..., None], col * 0.82, col)
    h = np.broadcast_to(height, m.shape).astype(float) if np.ndim(height) == 0 else height
    return np.clip(col, 0, 1), h - 0.03 * rim, m


def _ramp(mask_face, mask_top, cv, top_h, drop):
    """Height for a side facet: falls away from the edge it shares with the top face."""
    inv = (~(mask_top > 0)).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5) / cv.k
    return top_h - drop * np.clip(dist / 4.0, 0, 1)


def _outline(cover, cv, dark, width=0.55):
    """Dark outline band just inside the silhouette (like the vanilla art)."""
    dist = cv2.distanceTransform(cover.astype(np.uint8), cv2.DIST_L2, 5) / cv.k
    return dist < width, np.array(dark, float) / 255 * 0.7


# ------------------------------------------------------------- designs ---

def ingot(cv, pal):
    light, mid, dark = pal
    top = cv.polygon([(1.0, 6.4), (10.6, 2.3), (15.3, 6.1), (5.7, 10.3)])
    front = cv.polygon([(5.7, 10.3), (15.3, 6.1), (15.6, 9.9), (6.7, 13.9)])
    left = cv.polygon([(1.0, 6.4), (5.7, 10.3), (6.7, 13.9), (0.4, 10.7)])
    end_col = tuple((np.array(light) * 0.55 + np.array(mid) * 0.45).round())
    return [
        _flat(left, cv, end_col, dark, _ramp(left, top, cv, 0.95, 0.35), grad=0.15, axis=(-1, 0)),
        _flat(front, cv, mid, dark, _ramp(front, top, cv, 0.95, 0.35), grad=0.25, axis=(1, -1)),
        _flat(top, cv, light, dark, 0.95, grad=0.18, axis=(-1, -1)),
    ]


def nugget(cv, pal):
    light, mid, dark = pal
    parts = []
    for (cx, cy, r) in ((7.2, 7.9, 2.3), (9.6, 7.6, 1.8), (8.2, 6.4, 1.5)):
        parts.append(_shade_part(cv.circle((cx, cy), r), cv, (light, mid, dark), r * 0.8))
    return parts


def diamond(cv, pal):
    light, mid, dark = pal
    outline = [(6.0, 2.0), (10.0, 2.0), (13.5, 5.5), (14.0, 10.0), (11.0, 14.0), (5.0, 14.0), (2.0, 10.0), (2.5, 5.5)]
    table = [(6.6, 4.0), (9.6, 4.0), (11.6, 6.4), (11.2, 9.6), (8.2, 11.4), (5.0, 9.6), (4.6, 6.4)]
    parts = [_flat(cv.polygon(outline), cv, mid, dark, 0.7)]
    # crown facets between the table and the girdle, alternating tones
    tones = [light, mid, light, dark, mid, dark, mid, light]
    for i in range(len(outline)):
        a, b = outline[i], outline[(i + 1) % len(outline)]
        c_, d_ = table[min(i, len(table) - 1)], table[(i + 1) % len(table)] if i + 1 < len(table) else table[0]
        facet = cv.polygon([a, b, d_, c_])
        parts.append(_flat(facet, cv, tones[i], dark, 0.8 + 0.04 * (i % 2)))
    table_col = tuple((np.array(light) * 0.6 + np.array(mid) * 0.4).round())
    parts.append(_flat(cv.polygon(table), cv, table_col, dark, 0.95, grad=0.35, axis=(-1, -1)))
    parts.append(_flat(cv.polygon([(6.7, 4.7), (7.8, 4.7), (6.2, 6.6)]), cv, light, light, 0.97, edge_w=0.0))
    return parts


def emerald(cv, pal):
    light, mid, dark = pal
    rings = [
        ([(6.0, 2.0), (10.0, 2.0), (13.0, 5.0), (13.0, 11.0), (10.0, 14.0), (6.0, 14.0), (3.0, 11.0), (3.0, 5.0)], dark, 0.7),
        ([(6.4, 3.2), (9.6, 3.2), (11.8, 5.4), (11.8, 10.6), (9.6, 12.8), (6.4, 12.8), (4.2, 10.6), (4.2, 5.4)], mid, 0.8),
        ([(6.8, 4.4), (9.2, 4.4), (10.6, 5.8), (10.6, 10.2), (9.2, 11.6), (6.8, 11.6), (5.4, 10.2), (5.4, 5.8)], light, 0.9),
    ]
    parts = [_flat(cv.polygon(p), cv, c, dark, h, grad=0.25, axis=(-1, -1)) for p, c, h in rings]
    parts.append(_flat(cv.polygon([(6.9, 4.9), (8.0, 4.9), (6.0, 7.6), (5.9, 6.2)]), cv, (240, 255, 240), light, 0.95))
    return parts


def raw_ore(cv, pal):
    """Angular rock chunk: faceted planes lit from the top-left, rough surface."""
    light, mid, dark = pal
    L, M, D = (np.array(c, float) for c in (light, mid, dark))
    tone = lambda t: tuple(np.clip(D + (L - D) * t, 0, 255).round())
    facets = [
        ([(0.6, 4.6), (2.4, 1.4), (6.6, 1.0), (8.0, 5.6), (3.6, 7.4)], 0.95, 0.95),
        ([(6.6, 1.0), (10.2, 2.0), (13.6, 3.6), (12.2, 7.0), (8.0, 5.6)], 0.85, 0.92),
        ([(3.6, 7.4), (8.0, 5.6), (12.2, 7.0), (12.0, 10.6), (8.4, 12.6), (3.6, 12.8)], 0.68, 0.86),
        ([(0.6, 4.6), (3.6, 7.4), (3.6, 12.8), (0.8, 11.2)], 0.52, 0.8),
        ([(13.6, 3.6), (15.4, 6.6), (15.6, 11.0), (12.0, 10.6), (12.2, 7.0)], 0.5, 0.8),
        ([(12.0, 10.6), (15.6, 11.0), (14.6, 14.4), (10.4, 14.8), (8.4, 12.6)], 0.32, 0.74),
    ]
    rng = np.random.default_rng(11)
    noise = cv2.GaussianBlur(rng.random((cv.n, cv.n)).astype(np.float32), (0, 0), cv.k * 0.35)
    noise = (noise - noise.mean()) / (noise.std() + 1e-6)
    parts = []
    for poly, t, h in facets:
        c, hh, m = _flat(cv.polygon(poly), cv, tone(t), dark, h, edge_w=0.25)
        c = np.clip(c * (1 + 0.06 * noise[..., None]), 0, 1)
        parts.append((c, hh + 0.015 * noise, m))
    return parts


def rod(cv, pal):
    light, mid, dark = pal
    # lift the body toward the light tone: pixel-art sticks are drawn very dark
    body = tuple((np.array(light) * 0.55 + np.array(mid) * 0.45).round())
    return [_shade_part(cv.stroke((2.7, 13.3), (13.4, 2.6), 2.7), cv, (light, body, dark), 1.15,
                        axis=((2.7, 13.3), (13.4, 2.6)), grain=True, seed=5)]


METALS = (r"iron|gold|golden|copper|zinc|brass|tin|lead|silver|nickel|aluminum|aluminium|osmium|uranium|"
          r"platinum|tungsten|steel|bronze|electrum|invar|constantan|cobalt|titanium")

ARCHETYPES = [
    ("ingot", re.compile(r"ingot$"), ingot),
    ("nugget", re.compile(r"nugget$"), nugget),
    ("diamond", re.compile(r"^diamond$"), diamond),
    ("emerald", re.compile(r"^emerald$"), emerald),
    ("raw_ore", re.compile(r"^raw_(" + METALS + r")$"), raw_ore),
    ("rod", re.compile(r"^(stick|blaze_rod|breeze_rod)$|^(" + METALS + r")_rod$"), rod),
]


def archetype_of(path, ns="minecraft"):
    name = path.split("/")[-1].rsplit(".", 1)[0]
    if not path.startswith("textures/item/") or name.startswith("empty_slot"):
        return None  # empty_slot_* are the smithing table's ghost outlines
    for key, pat, fn in ARCHETYPES:
        if pat.search(name):
            # bare gem names in other mods are their own items (e.g. wrench variants)
            if key in ("diamond", "emerald") and ns != "minecraft":
                return None
            return key
    return None


def render(path, src, size):
    """Hand-made maps for an item texture, coloured from its source."""
    name = path.split("/")[-1].rsplit(".", 1)[0]
    fn = next(f for key, pat, f in ARCHETYPES if pat.search(name))
    pal = palette_from(src)
    cv = Canvas(size)
    parts = fn(cv, pal)
    col = np.zeros((cv.n, cv.n, 3))
    h = np.zeros((cv.n, cv.n))
    cover = np.zeros((cv.n, cv.n), bool)
    for c, hh, m in parts:
        col = np.where(m[..., None], c, col)
        h = np.where(m, hh, h)
        cover |= m
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
    f0, sm = material_of(name)
    if name in ("stick",):
        f0, sm = DIELECTRIC, 0.3
    m.f0 = np.where(solid, f0, DIELECTRIC).astype(float)
    m.smooth = np.where(solid, sm, 0.0)
    if name == "blaze_rod":
        lum = rgb @ LUMW
        m.emissive = np.clip((lum - 0.45) * 2, 0, 1) * solid
    return derive(m, edge_wear=0.0, edges="reflect", normal_strength=4.0)
