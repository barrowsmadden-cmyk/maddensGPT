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
# House style (handmade/style.py). Shapes are traced from the originals: texel
# (x, y) spans x..x+1, y..y+1 in the 16-unit grid. Each design takes the
# original texture; parts take their tones from the texels they cover.

def _rp(pts, r=0.5):
    from handmade.style import round_poly
    return round_poly(pts, r)


def mat_of(name):
    """House-style material for an item name."""
    if re.search(r"gold", name):
        return "gold"
    if re.search(r"copper", name):
        return "copper"
    if re.search(r"iron|steel|chain", name):
        return "iron"
    if re.search(r"diamond", name):
        return "gem"
    if re.search(r"emerald|amethyst|quartz|gem|crystal|lapis|ruby|sapphire|experience", name):
        return "crystal"
    if re.search(r"netherite|brass|zinc|bronze|lead|tin|nickel|silver|aluminum|electrum|osmium|platinum|"
                 r"uranium|ingot|nugget", name):
        return "metal"
    if re.search(r"stick|rod", name):
        return "wood"
    return "stone"


def ingot_vanilla(cv, src, m):
    """Long isometric bar: top face, left end, front face, bright edges between them."""
    from handmade.style import part
    top = cv.polygon([(0.0, 5.9), (12.0, 2.0), (16.0, 6.0), (4.6, 9.3)])
    left = cv.polygon([(0.0, 5.9), (4.6, 9.3), (4.6, 13.9), (3.0, 14.0), (0.0, 11.0)])
    front = cv.polygon([(4.6, 9.3), (16.0, 6.0), (16.0, 10.0), (7.0, 14.0), (4.6, 13.9)])
    return [part(front, m, "bevel", bevel=0.7, h=0.8, glint=False, inner_line=False),
            part(left, m, "bevel", bevel=0.7, h=0.8, glint=False, inner_line=False),
            part(top, m, "bevel", bevel=0.8, h=0.88, inner_line=False),
            part(cv.stroke((1.0, 6.4), (4.4, 9.0), 0.75), m, "flat", h=0.9, inner_line=False),
            part(cv.stroke((5.0, 9.0), (13.6, 6.4), 0.75), m, "flat", h=0.9, inner_line=False),
            part(cv.stroke((4.6, 9.6), (4.6, 12.6), 0.75), m, "flat", h=0.86, inner_line=False)]


def ingot_create(cv, src, m):
    """Create's taller ingot: same faces, thicker bar."""
    from handmade.style import part
    top = cv.polygon([(1.0, 5.8), (12.0, 2.0), (15.0, 5.0), (13.6, 6.5), (5.0, 9.0), (1.0, 6.3)])
    left = cv.polygon([(1.0, 6.2), (5.0, 9.0), (5.0, 14.0), (3.0, 14.0), (1.0, 12.0)])
    front = cv.polygon([(5.0, 9.0), (13.6, 6.5), (15.0, 5.0), (15.0, 10.0), (13.5, 11.0), (6.0, 14.0), (5.0, 14.0)])
    return [part(front, m, "bevel", bevel=0.7, h=0.8, glint=False, inner_line=False),
            part(left, m, "bevel", bevel=0.7, h=0.8, glint=False, inner_line=False),
            part(top, m, "bevel", bevel=0.8, h=0.88, inner_line=False),
            part(cv.stroke((2.4, 6.4), (4.6, 8.6), 0.75), m, "flat", h=0.9, inner_line=False),
            part(cv.stroke((4.6, 9.4), (4.6, 11.6), 0.75), m, "flat", h=0.86, inner_line=False)]


def nugget(cv, src, m):
    """Rounded lump traced from the original, its light pattern kept."""
    from handmade.style import part, silhouette
    return [part(silhouette(src, cv, 0.5), m, "post", bevel=0.9, h=0.85)]


def diamond(cv, src, m):
    """Brilliant cut seen from above: table, crown and pavilion facets with
    straight edges; each facet takes its tone from the original texels it
    covers (bright upper-left crown, dark right side, lit culet)."""
    from handmade.style import part
    body = cv.polygon(_rp([(6.0, 2.0), (10.0, 2.0), (13.0, 5.0), (14.0, 7.0), (14.0, 11.0), (13.0, 13.0), (11.0, 15.0),
                           (5.0, 15.0), (3.0, 13.0), (2.0, 11.0), (2.0, 7.0), (3.0, 5.0)], 0.4))
    facets = [
        [(3.0, 5.0), (6.0, 2.0), (5.2, 4.4), (4.6, 7.0), (5.6, 9.8), (2.0, 10.4), (2.0, 7.0)],    # left crown
        [(6.0, 2.0), (10.0, 2.0), (9.6, 3.6), (5.2, 4.4)],                                        # upper crown
        [(10.0, 2.0), (13.0, 5.0), (14.0, 7.0), (14.0, 10.2), (10.0, 9.6), (10.4, 6.0), (9.6, 3.6)],  # right crown
        [(5.2, 4.4), (9.6, 3.6), (10.4, 6.0), (10.0, 9.6), (5.6, 9.8), (4.6, 7.0)],               # table
        [(2.0, 10.4), (5.6, 9.8), (8.0, 12.4), (5.0, 15.0), (3.0, 13.0), (2.0, 11.0)],            # pavilion left
        [(5.6, 9.8), (10.0, 9.6), (8.0, 12.4)],                                                   # pavilion centre
        [(10.0, 9.6), (14.0, 10.2), (14.0, 11.0), (13.0, 13.0), (11.0, 15.0), (8.0, 12.4)],       # pavilion right
        [(8.0, 12.4), (11.0, 15.0), (5.0, 15.0)],                                                 # culet
    ]
    parts = [part(body, m, "bevel", bevel=0.8, h=0.88)]
    parts += [part(cv.polygon(f), m, "flat", h=0.9, inner_line=False) for f in facets]
    star = cv.polygon([(5.6, 3.6), (6.0, 4.6), (7.0, 5.0), (6.0, 5.4), (5.6, 6.4), (5.2, 5.4), (4.2, 5.0), (5.2, 4.6)])
    parts.append(part(star, m, "flat", h=0.95, inner_line=False, tones=((255, 255, 255), (255, 255, 255), (220, 255, 250))))
    return parts


def emerald(cv, src, m):
    """Step cut: wide bevels round a raised table."""
    from handmade.style import part
    outer = cv.polygon([(6.0, 2.0), (10.0, 2.0), (13.0, 5.0), (13.0, 11.0), (10.0, 14.0), (6.0, 14.0), (3.0, 11.0),
                        (3.0, 5.0)])
    table = cv.polygon([(6.6, 4.4), (9.4, 4.4), (10.6, 5.8), (10.6, 10.2), (9.4, 11.6), (6.6, 11.6), (5.4, 10.2),
                        (5.4, 5.8)])
    return [part(outer, m, "bevel", bevel=2.0, h=0.86),
            part(table, m, "bevel", bevel=0.6, h=0.92, inner_line=False)]


def raw_ore(cv, src, m):
    """Raw ore chunk: the original's lumps and crevices, traced smoothly."""
    from handmade.style import part, silhouette
    return [part(silhouette(src, cv, 0.5), m, "post", bevel=0.8, h=0.85)]


def rod(cv, src, m):
    """A straight bar fitted to the original: its axis and ends from the texels'
    principal direction, its width from their spread across it."""
    from handmade.style import part
    h, w = src.shape[:2]
    ys, xs = np.nonzero(src[..., 3] > 0.5)
    pts = np.stack([xs + 0.5, ys + 0.5], 1) * (16.0 / w)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    u = vt[0]
    t = (pts - c) @ u
    across = (pts - c) @ np.array([-u[1], u[0]])
    width = max(1.2, np.percentile(np.abs(across), 95) * 2 + 16.0 / w)
    a = tuple(c + u * (t.min() + width / 2 - 0.3))
    b = tuple(c + u * (t.max() - width / 2 + 0.3))
    return [part(cv.stroke(a, b, width), m, "cyl", axis=(a, b), grain=1.0 if m == "wood" else 0.0, h=0.8)]


METALS = (r"iron|gold|golden|copper|zinc|brass|tin|lead|silver|nickel|aluminum|aluminium|osmium|uranium|"
          r"platinum|tungsten|steel|bronze|electrum|invar|constantan|cobalt|titanium")

ARCHETYPES = [
    ("ingot", re.compile(r"ingot$"), (ingot_vanilla, ingot_create)),
    ("nugget", re.compile(r"nugget$"), (nugget,)),
    ("diamond", re.compile(r"^diamond$"), (diamond,)),
    ("emerald", re.compile(r"^emerald$"), (emerald,)),
    ("raw_ore", re.compile(r"^raw_(" + METALS + r")$"), (raw_ore,)),
    ("rod", re.compile(r"^(stick|blaze_rod|breeze_rod)$|^(" + METALS + r")_rod$"), (rod,)),
]
MIN_IOU = 0.8   # a traced design must cover the original's silhouette this well, or the texture is left alone


def archetype_of(path, ns="minecraft"):
    name = path.split("/")[-1].rsplit(".", 1)[0]
    if not path.startswith("textures/item/") or name.startswith("empty_slot") or path.count("/") != 2:
        return None  # empty_slot_* are the smithing table's ghost outlines; sub-folders hold model textures
    for key, pat, _ in ARCHETYPES:
        if pat.search(name):
            # bare gem names in other mods are their own items (e.g. wrench variants)
            if key in ("diamond", "emerald") and ns != "minecraft":
                return None
            return key
    return None


def _iou(parts, cv, src):
    cover = np.any([p["mask"] for p in parts], 0).astype(np.float32)
    h, w = src.shape[:2]
    small = cv2.resize(cover, (w, h), interpolation=cv2.INTER_AREA) >= 0.5
    orig = src[..., 3] > 0.5
    return (small & orig).sum() / max((small | orig).sum(), 1)


def render(path, src, size):
    """Hand-made maps for an item texture, or None when no traced design fits its shape."""
    from handmade.style import render_item
    name = path.split("/")[-1].rsplit(".", 1)[0]
    variants = next(v for _, pat, v in ARCHETYPES if pat.search(name))
    m = mat_of(name)
    if name in ("stick",):
        m = "wood"
    elif name in ("blaze_rod", "breeze_rod"):
        m = "glossy"
    cv = Canvas(size)
    best = max(((fn(cv, src, m)) for fn in variants), key=lambda ps: _iou(ps, cv, src))
    if _iou(best, cv, src) < MIN_IOU:
        return None
    if name == "blaze_rod":
        for p in best:
            p["emissive"] = 0.5
    return render_item(best, cv, src, size)
