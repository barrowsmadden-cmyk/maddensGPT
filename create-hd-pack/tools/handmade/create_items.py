"""Hand-made Create items (and the addon items sharing their designs).

Each design is drawn from explicit shapes in the 16-unit item grid, laid out on
the Create silhouettes. Generic designs (sheets, crushed ores, powders, alloy
bars) take their colours from the texture being replaced; one-off items
(precision mechanism, electron tube, propeller, goggles) use fixed palettes
sampled from the Create art.

A part is (colour, height, mask, material) with material = (f0, smooth,
emissive); parts are painted in order, later ones on top.
"""
import math
import re

import cv2
import numpy as np

from texgen import Material, derive, IRON, GOLD, ALBEDO_METAL, DIELECTRIC
from handmade.tools import Canvas, _shade_part
from handmade.items import palette_from, _flat, _ramp, _outline, LUMW

BRASS = ((252, 226, 140), (222, 168, 72), (118, 72, 26))
IRONP = ((236, 236, 238), (176, 176, 182), (78, 78, 88))
DARKP = ((92, 74, 60), (58, 44, 36), (28, 20, 16))
LEATHER = ((150, 104, 70), (112, 74, 48), (54, 34, 22))
ROSE = ((255, 190, 200), (236, 92, 122), (124, 28, 52))
LENS = ((214, 240, 236), (120, 186, 184), (36, 82, 88))


def _mix(a, b, t):
    return tuple((np.array(a, float) * (1 - t) + np.array(b, float) * t).round())


def _noise(cv, scale, seed):
    rng = np.random.default_rng(seed)
    n = cv2.GaussianBlur(rng.random((cv.n, cv.n)).astype(np.float32), (0, 0), cv.k * scale)
    return (n - n.mean()) / (n.std() + 1e-6)


def _with(part, mat):
    return (*part[:3], mat)


def _textured(part, cv, amount, scale, seed, dh=0.01):
    c, h, m = part[:3]
    n = _noise(cv, scale, seed)
    return (np.clip(c * (1 + amount * n[..., None]), 0, 1), h + dh * n, m, *part[3:])


# ------------------------------------------------------------- designs ---

def _round_poly(pts, r=0.8, steps=6):
    """Closed polygon with each corner filleted over `r` units (or less on short edges)."""
    p = [np.array(v, float) for v in pts]
    out = []
    for i, v in enumerate(p):
        prev, nxt = p[i - 1], p[(i + 1) % len(p)]
        e1, e2 = prev - v, nxt - v
        r1 = min(r, np.linalg.norm(e1) * 0.45)
        r2 = min(r, np.linalg.norm(e2) * 0.45)
        a = v + e1 / np.linalg.norm(e1) * r1
        c = v + e2 / np.linalg.norm(e2) * r2
        for t in np.linspace(0, 1, steps):
            out.append(tuple((1 - t) ** 2 * a + 2 * (1 - t) * t * v + t * t * c))
    return out


def _slab(cv, top, depth, pal, top_col, side_l, side_r, top_h=0.95, drop=0.2, rounding=0.8):
    """Rounded isometric slab: the top face and its downward extrusion."""
    _, _, dark = pal
    rt = _round_poly(top, rounding)
    tm = cv.polygon(rt)
    body = cv.blank()
    for dy in np.linspace(0, depth, 12):
        body |= cv.polygon([(x, y + dy) for x, y in rt])
    sides = (body > 0) & ~(tm > 0)
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    B = top[3]
    left = sides & (xx < B[0])
    right = sides & (xx >= B[0])
    pl = _flat(left.astype(np.uint8) * 255, cv, side_l, dark, _ramp(left, tm, cv, top_h, drop), grad=0.12, axis=(-1, 0))
    pr = _flat(right.astype(np.uint8) * 255, cv, side_r, dark, _ramp(right, tm, cv, top_h, drop), grad=0.2, axis=(1, -1))
    pt = _flat(tm, cv, top_col, dark, top_h, grad=0.16, axis=(-1, -1))
    return [pl, pr, pt]


def sheet(cv, pal, name):
    """Pressed plate: thick rounded slab, rolling streaks, bright leading edge;
    the sturdy sheet is a heavier riveted plate."""
    light, mid, dark = pal
    top = [(0.9, 7.6), (9.0, 3.5), (15.1, 8.0), (6.8, 12.2)]
    sturdy = name == "sturdy_sheet"
    parts = _slab(cv, top, 1.9 if sturdy else 1.2, pal, mid, _mix(mid, dark, 0.5), _mix(mid, dark, 0.25), rounding=1.0)
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    along = (xx - 0.9) * 0.89 + (yy - 7.6) * -0.45
    rng = np.random.default_rng(len(name))
    streak = sum(a * np.sin(along * f + rng.uniform(0, 6)) for f, a in ((3.1, 0.5), (7.3, 0.3), (13.0, 0.2)))
    c, h, m = parts[-1]
    c = np.clip(c * (1 + 0.04 * streak[..., None]), 0, 1)
    glint = cv.stroke((2.8, 7.4), (8.8, 4.5), 0.5) > 0
    c = np.where((glint & m)[..., None], np.array(light, float)[None, None, :] / 255, c)
    parts[-1] = (np.clip(c, 0, 1), h + 0.004 * streak, m)
    if sturdy:                                            # four rivets near the corners
        for (u, v) in ((0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)):
            L, T, R, B = (np.array(q) for q in top)
            pnt = L + (T - L) * u + (B - L) * v
            parts.append(_shade_part(cv.circle(tuple(pnt), 0.75), cv, (light, _mix(light, mid, 0.5), dark), 0.6))
    return parts


def alloy(cv, pal, name):
    """Chunky cast bar (andesite alloy, shadow steel, refined radiance)."""
    light, mid, dark = pal
    top = [(2.6, 5.8), (9.4, 2.4), (13.6, 5.4), (6.8, 8.8)]
    parts = _slab(cv, top, 4.0, pal, light, _mix(mid, dark, 0.3), mid, drop=0.3, rounding=0.7)
    if name == "andesite_alloy":                          # speckled andesite with iron flecks
        parts = [_textured(p, cv, 0.12, 0.18, i + 3) for i, p in enumerate(parts)]
        fleck = (_noise(cv, 0.22, 9) > 1.6)
        parts = [(np.where((fleck & p[2])[..., None], np.array(IRONP[0], float)[None, None] / 255, p[0]),
                  p[1] + 0.01 * fleck, p[2]) for p in parts]
    elif name == "shadow_steel":                          # dark steel with a violet sheen
        c, h, m = parts[-1]
        yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
        sheen = np.exp(-(((xx + yy) - 13.0) / 1.4) ** 2)
        parts[-1] = (np.clip(c + np.array([0.18, 0.12, 0.3]) * sheen[..., None], 0, 1), h, m)
    return parts


def crushed(cv, pal, name):
    """Small heap of crushed ore: angular chunks packed into a low mound, each
    with a lit upper facet and a shaded lower facet over a shadowed base;
    some chunks are bright raw metal, the rest the duller gangue."""
    light, mid, dark = pal
    L, M, D = (np.array(c, float) for c in pal)
    tone = lambda t: tuple(np.clip(D + (M - D) * 2 * t if t < 0.5 else M + (L - M) * (2 * t - 1), 0, 255).round())
    heap = cv.polygon(_round_poly([(1.2, 9.4), (3.4, 4.6), (7.0, 3.0), (11.4, 3.4), (14.4, 6.0), (15.0, 10.0),
                                   (12.0, 12.2), (7.2, 12.8), (2.8, 11.8)], 1.5)) > 0
    rng = np.random.default_rng(sum(map(ord, name)))
    centres = []
    for _ in range(4000):
        x, y = rng.uniform(1.6, 14.6), rng.uniform(3.4, 12.4)
        if heap[int(y * cv.k), int(x * cv.k)] and all(math.hypot(x - a, (y - b) * 1.15) > 2.5 for a, b in centres):
            centres.append((x, y))
        if len(centres) >= 16:
            break
    centres.sort(key=lambda c: c[1])                  # back (top of the image) to front
    f0, sm = _material_for(name.replace("crushed_raw_", ""))
    bound = cv2.dilate(heap.astype(np.uint8), np.ones((int(cv.k * 1.6),) * 2, np.uint8)) > 0
    chunks = []
    for i, (cx, cy) in enumerate(centres):
        n, r = 6, rng.uniform(1.7, 2.2)
        ang = np.arange(n) * 2 * math.pi / n + rng.uniform(-0.35, 0.35, n) + rng.uniform(0, 1)
        rad = r * rng.uniform(0.8, 1.1, n)
        poly = [(cx + q * math.cos(a), cy + q * math.sin(a) * 0.85) for a, q in zip(ang, rad)]
        body = (cv.polygon(poly) > 0) & bound
        top = (cv.polygon([(cx - 0.35 + 0.7 * (x - cx), cy - 0.5 + 0.6 * (y - cy)) for x, y in poly]) > 0) & bound
        chunks.append((i, cy, body, top))
    # shadowed base under the chunks: their union, so the outline follows them
    under = cv2.erode(np.any([c[2] for c in chunks], 0).astype(np.uint8), np.ones((3, 3), np.uint8)) | heap
    base = _flat(under.astype(np.uint8) * 255, cv, tone(0.12), dark, 0.7, edge_w=0.0)
    parts = [_with(_textured(base, cv, 0.1, 0.2, 20), (DIELECTRIC, 0.25, 0.0))]
    for i, cy, body, top in chunks:
        ore = rng.random() < 0.4
        lo, hi = (0.55, 1.0) if ore else (0.36, 0.7)
        mat = (f0, sm, 0.0) if ore else (DIELECTRIC, 0.3, 0.0)
        h = 0.8 + 0.06 * (cy / 12)
        b = _flat(body.astype(np.uint8) * 255, cv, tone(lo), dark, h - 0.05, edge_w=0.3, grad=0.2, axis=(1, 1))
        t = _flat(top.astype(np.uint8) * 255, cv, tone(hi), dark, h, edge_w=0.0, grad=0.15, axis=(-1, -1))
        parts += [_with(_textured(b, cv, 0.05, 0.25, 22 + i), mat), _with(_textured(t, cv, 0.05, 0.25, 40 + i), mat)]
    return parts


def powder(cv, pal, name):
    """Soft mound of powder with grainy surface."""
    light, mid, dark = pal
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    cx, base, peak, half = 8.0, 12.8, 4.6, 6.2
    prof = base - (base - peak) * np.clip(1 - ((xx - cx) / half) ** 2, 0, 1) ** 0.75
    lumps = 0.35 * np.sin(xx * 1.9) * np.sin(xx * 0.7 + 1)
    m = ((yy >= prof + lumps) & (yy <= base + 0.9 * np.sqrt(np.clip(1 - ((xx - cx) / half) ** 2, 0, 1)))
         & (np.abs(xx - cx) < half)).astype(np.uint8) * 255
    c, h, mk = _shade_part(m, cv, pal, 4.0)
    return [_textured((c, h, mk), cv, 0.14, 0.12, 31, dh=0.02)]


def paper_grit(cv, pal, name):
    """Sand paper: a square sheet seen at an angle, gritty face, one corner
    folded back over the sheet showing the smooth paper backing."""
    light, mid, dark = pal
    L, T, R, B = (np.array(p) for p in ((0.8, 8.0), (8.0, 2.6), (15.2, 7.8), (8.0, 13.4)))
    p1, p2 = R + (T - R) * 0.42, R + (B - R) * 0.42
    tip = p1 + p2 - R                                         # the corner folded across p1-p2
    sheet_ = _flat(cv.polygon([tuple(L), tuple(T), tuple(p1), tuple(p2), tuple(B)]), cv, mid, dark, 0.9,
                   edge_w=0.3, grad=0.12, axis=(-1, -1))
    grit = _noise(cv, 0.08, 41)
    c, h, m = sheet_
    c = np.clip(c * (1 + 0.16 * grit[..., None]), 0, 1)
    flap_m = cv.polygon([tuple(p1), tuple(p2), tuple(tip)])
    shadow = (cv.polygon([tuple(p1 + (-0.5, 0.5)), tuple(p2 + (-0.5, 0.5)), tuple(tip + (-0.5, 0.5))]) > 0) & m
    c = np.where(shadow[..., None], c * 0.75, c)
    backing = _mix((250, 244, 228), light, 0.3)
    flap = _flat(flap_m, cv, backing, dark, 0.97, edge_w=0.2, grad=0.3, axis=(1, -1))
    return [(c, h + 0.02 * grit, m), flap]


def precision_mechanism(cv, pal, name):
    brass, iron = BRASS, IRONP
    cx, cy = 7.6, 8.4
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    ang = np.arctan2(yy - cy, xx - cx)
    rr = np.hypot(xx - cx, yy - cy)
    teeth = rr < 5.4 + 0.75 * (np.cos(ang * 12) > 0.1)
    rim = (teeth & (rr > 3.6)).astype(np.uint8) * 255
    well = cv.circle((cx, cy), 3.7)
    hub = cv.circle((cx - 0.4, cy - 0.4), 1.25)
    ring = _shade_part(rim, cv, brass, 1.0)
    inner = _flat(well, cv, DARKP[1], DARKP[2], 0.62, edge_w=0.3)
    # small brass cog inside the well, offset up-left
    a2 = np.arctan2(yy - (cy - 0.6), xx - (cx - 0.8))
    r2 = np.hypot(xx - (cx - 0.8), yy - (cy - 0.6))
    cog = ((r2 < 2.3 + 0.5 * (np.cos(a2 * 8) > 0)) & (r2 > 0.9)).astype(np.uint8) * 255
    # iron gear crossing the rim at the bottom right
    a3 = np.arctan2(yy - 11.6, xx - 11.6)
    r3 = np.hypot(xx - 11.6, yy - 11.6)
    gear = ((r3 < 2.0 + 0.55 * (np.cos(a3 * 8) > 0)) & (r3 > 0.7)).astype(np.uint8) * 255
    gear_hub = cv.circle((11.6, 11.6), 0.7)
    gold = (GOLD, 0.8, 0.0)
    if name.startswith("incomplete"):
        return [_with(inner, (DIELECTRIC, 0.3, 0.0)), _with(ring, gold),
                _with(_shade_part(gear, cv, iron, 0.8), (IRON, 0.72, 0.0)),
                _with(_flat(gear_hub, cv, DARKP[1], DARKP[2], 0.85, edge_w=0.0), (IRON, 0.5, 0.0))]
    return [
        _with(inner, (DIELECTRIC, 0.3, 0.0)),
        _with(_shade_part(cog, cv, _mix_pal(brass, 0.12), 0.8), gold),
        _with(_shade_part(hub, cv, iron, 0.9), (IRON, 0.7, 0.0)),
        _with(ring, gold),
        _with(_shade_part(gear, cv, iron, 0.8), (IRON, 0.72, 0.0)),
        _with(_flat(gear_hub, cv, DARKP[1], DARKP[2], 0.85, edge_w=0.0), (IRON, 0.5, 0.0)),
    ]


def _mix_pal(pal, t):
    return tuple(_mix(c, (0, 0, 0), t) for c in pal)


def electron_tube(cv, pal, name):
    """Vacuum tube: clear glass envelope around a glowing rose-quartz core and
    its electrode, on a dark iron base with three pins."""
    env = [(5.2, 9.6), (5.2, 4.6)] + \
          [(8.0 + 2.8 * math.cos(a), 4.6 - 3.3 * math.sin(a)) for a in np.linspace(math.pi, 0, 20)] + [(10.8, 9.6)]
    glass_m = cv.polygon(env)
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    glass_pal = (_mix(ROSE[0], (255, 255, 255), 0.5), _mix(ROSE[1], (255, 255, 255), 0.55), ROSE[2])
    gc, gh, gm = _shade_part(glass_m, cv, glass_pal, 2.4)
    core_m = cv.polygon([(7.0, 8.6), (7.0, 5.0), (8.0, 3.4), (9.0, 5.0), (9.0, 8.6)])
    cc, ch, cm = _shade_part(core_m, cv, ROSE, 1.2)
    glow = np.exp(-(((xx - 8.0) / 2.2) ** 2 + ((yy - 5.6) / 2.8) ** 2))
    cc = np.clip(cc + (np.array([1.0, 0.8, 0.86]) - cc) * (0.35 * glow)[..., None], 0, 1)
    stem = _flat(cv.polygon([(7.6, 8.4), (8.4, 8.4), (8.4, 9.8), (7.6, 9.8)]), cv, IRONP[1], IRONP[2], 0.7, edge_w=0.0)
    glint = cv.stroke((6.3, 3.9), (6.1, 7.6), 0.5) > 0
    gc = np.where((glint & gm)[..., None], np.array([1.0, 0.97, 0.98]), gc)
    dark_iron = _mix_pal(IRONP, 0.35)
    base = _shade_part(cv.polygon(_round_poly([(4.8, 9.4), (11.2, 9.4), (11.2, 12.6), (4.8, 12.6)], 0.6)), cv, dark_iron, 1.3)
    collar = _shade_part(cv.stroke((5.0, 9.7), (11.0, 9.7), 0.9), cv, IRONP, 0.45)
    pins = [_shade_part(cv.stroke((x, 12.4), (x, 14.6), 0.85), cv, IRONP, 0.42) for x in (6.3, 8.0, 9.7)]
    em = np.clip(0.4 + 0.5 * glow, 0, 0.9) * cm
    return ([_with(p, (IRON, 0.6, 0.0)) for p in pins] +
            [_with(base, (IRON, 0.62, 0.0)), _with(collar, (IRON, 0.75, 0.0)),
             (gc, gh, gm, (12, 0.95, 0.0)),
             _with(stem, (IRON, 0.6, 0.0)),
             (cc, ch, cm, (40, 0.9, em)),
             (gc, gh + 0.05, (glint & gm), (12, 0.98, 0.0))])


def propeller(cv, pal, name):
    """Four broad, twisted iron blades around a riveted hub."""
    hub = (7.8, 7.8)
    blades = []
    for (tx, ty) in ((4.6, 1.8), (14.6, 5.6), (10.8, 14.0), (1.0, 9.8)):
        dx, dy = tx - hub[0], ty - hub[1]
        ln = math.hypot(dx, dy)
        nx, ny = -dy / ln, dx / ln
        root = (hub[0] + dx * 0.18, hub[1] + dy * 0.18)
        tip = (tx, ty)
        w_root, w_tip = 0.9, 1.9
        lead = cv.polygon(_round_poly([(root[0] + nx * w_root, root[1] + ny * w_root),
                                       (tip[0] + nx * w_tip, tip[1] + ny * w_tip), tip, root], 0.5))
        trail = cv.polygon(_round_poly([root, tip, (tip[0] - nx * w_tip, tip[1] - ny * w_tip),
                                        (root[0] - nx * w_root, root[1] - ny * w_root)], 0.5))
        blades.append(_flat(trail, cv, IRONP[1], IRONP[2], 0.8, edge_w=0.3, grad=0.25, axis=(dx, dy)))
        blades.append(_flat(lead, cv, IRONP[0], IRONP[2], 0.88, edge_w=0.3, grad=0.25, axis=(dx, dy)))
    hubp = _shade_part(cv.circle(hub, 2.0), cv, _mix_pal(IRONP, 0.2), 1.4)
    cap = _shade_part(cv.circle(hub, 0.9), cv, IRONP, 0.7)
    return [_with(p, (IRON, 0.7, 0.0)) for p in blades + [hubp, cap]]


def goggles(cv, pal, name):
    """Engineer's goggles: wide leather strap, brass-rimmed round lenses."""
    strap = cv.blank()
    pts = [(2.2 + 11.6 * t, 8.6 - 4.8 * math.sin(math.pi * t)) for t in np.linspace(0, 1, 40)]
    cv2.polylines(strap, [cv.pts(pts)], False, 255, thickness=int(round(2.2 * cv.k)), lineType=cv2.LINE_8)
    band = _textured(_shade_part(strap, cv, LEATHER, 1.1), cv, 0.08, 0.15, 51)
    stitch = cv.blank()
    cv2.polylines(stitch, [cv.pts(pts)], False, 255, thickness=max(1, int(round(0.22 * cv.k))), lineType=cv2.LINE_8)
    dash = (np.mgrid[0:cv.n, 0:cv.n][1] / cv.k % 1.2) < 0.6
    stitch = (stitch > 0) & dash
    parts = [_with(band, (DIELECTRIC, 0.25, 0.0)),
             (np.broadcast_to(np.array(LEATHER[0]) / 255 * 1.1, (cv.n, cv.n, 3)), band[1] + 0.01, stitch,
              (DIELECTRIC, 0.3, 0.0))]
    bridge = _shade_part(cv.stroke((6.0, 9.2), (10.0, 9.2), 1.2), cv, BRASS, 0.6)
    parts.append(_with(bridge, (GOLD, 0.75, 0.0)))
    for cx in (4.2, 11.8):
        rim = _shade_part(cv.circle((cx, 9.8), 3.0), cv, BRASS, 1.1)
        lens = _shade_part(cv.circle((cx, 9.8), 1.9), cv, LENS, 1.6)
        c, h, m = lens
        glint = (cv.stroke((cx - 1.0, 9.3), (cx - 0.3, 8.5), 0.5) > 0) & m
        c = np.where(glint[..., None], np.array([0.95, 1.0, 1.0]), c)
        parts += [_with(rim, (GOLD, 0.78, 0.0)), (c, h + 0.02, m, (12, 0.95, 0.0))]
    return parts


def _prism(cv, base, tip, width, pal, h=0.9, tip_len=None):
    """Hexagonal crystal seen side-on: three long faces (shaded, lit, mid) and
    a pointed tip with matching facets."""
    light, mid, dark = pal
    bx, by = base
    tx, ty = tip
    ln = math.hypot(tx - bx, ty - by)
    ux, uy = (tx - bx) / ln, (ty - by) / ln
    nx, ny = -uy, ux
    tl = tip_len if tip_len is not None else width * 0.8
    top = (tx - ux * tl, ty - uy * tl)
    cuts = (-0.5, -0.17, 0.2, 0.5)
    tones = (_mix(mid, dark, 0.35), light, mid)
    parts = []
    for (a, b), t in zip(zip(cuts, cuts[1:]), tones):
        pa = lambda p, k: (p[0] + nx * width * k, p[1] + ny * width * k)
        face = cv.polygon([pa(base, a), pa(top, a), tip, pa(top, b), pa(base, b)])
        parts.append(_flat(face, cv, t, dark, h, edge_w=0.22, grad=0.2, axis=(ux, uy)))
    return parts


def rose_quartz(cv, pal, name):
    """Raw rose quartz: a chunky mass of stubby hexagonal crystals, milky veins."""
    light, mid, dark = pal
    cpal = (_mix(light, mid, 0.45), _mix(mid, dark, 0.1), dark)
    parts = []
    for base, tip, w in (((4.2, 14.2), (2.6, 6.0), 4.2), ((12.0, 14.2), (13.8, 5.0), 4.4),
                         ((8.2, 14.6), (7.2, 1.6), 6.0)):
        parts += _prism(cv, base, tip, w, cpal, tip_len=w * 0.6)
    veins = cv.blank()
    for pts in (((7.0, 12.8), (6.6, 9.8), (7.4, 6.6)), ((12.6, 12.4), (12.9, 9.0)), ((3.6, 12.6), (3.2, 9.8))):
        cv2.polylines(veins, [cv.pts(pts)], False, 255, thickness=max(1, int(round(0.4 * cv.k))), lineType=cv2.LINE_AA)
    cover = np.any([p[2] for p in parts], 0)
    parts.append((np.broadcast_to(np.array(light, float) / 255, (cv.n, cv.n, 3)).copy(),
                  np.full((cv.n, cv.n), 0.9), (veins > 0) & cover))
    return parts


def polished_rose_quartz(cv, pal, name):
    """Polished rose quartz: a smooth rounded gem slab with a glossy highlight."""
    light, mid, dark = pal
    top = [(2.4, 6.4), (9.0, 2.6), (14.0, 6.0), (7.4, 10.0)]
    parts = _slab(cv, top, 4.2, pal, _mix(light, mid, 0.3), _mix(mid, dark, 0.3), mid, rounding=1.6)
    glint = cv.stroke((4.6, 6.2), (8.6, 4.0), 0.7) > 0
    c, h, m = parts[-1]
    parts[-1] = (np.where((glint & m)[..., None], np.array([1.0, 0.96, 0.97]), c), h, m)
    return parts


def super_glue(cv, pal, name):
    """Squeezable glue tube: flattened, ridged crimp at the top right, plump
    body, short white cap at the bottom left."""
    light, mid, dark = pal
    p0, p1 = np.array((13.2, 2.8)), np.array((6.2, 9.8))
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array((-u[1], u[0]))
    q = lambda p, w: tuple(p + n * w)
    body = cv.polygon(_round_poly([q(p0, 2.9), q(p0, -2.9), q(p1 - u * 0.4, -2.2), q(p1, -1.2), q(p1, 1.2),
                                   q(p1 - u * 0.4, 2.2)], 0.8))
    c0 = p0 - u * 0.2
    crimp = cv.polygon([q(c0 - u * 1.6, 3.1), q(c0 - u * 1.6, -3.1), q(c0, -2.9), q(c0, 2.9)])
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    ridges = (np.abs(((xx - c0[0]) * n[0] + (yy - c0[1]) * n[1]) % 1.0 - 0.5) < 0.16)
    cr = _shade_part(crimp, cv, (light, _mix(light, mid, 0.5), dark), 0.8)
    cr = (np.where(ridges[..., None], cr[0] * 0.82, cr[0]), cr[1], cr[2])
    cap = cv.polygon(_round_poly([q(p1 - u * 0.2, 1.5), q(p1 - u * 0.2, -1.5), q(p1 + u * 3.4, -1.1),
                                  q(p1 + u * 3.4, 1.1)], 0.5))
    white = ((250, 250, 246), (216, 216, 212), (110, 110, 112))
    return [_with(_shade_part(body, cv, pal, 2.6), (DIELECTRIC, 0.55, 0.0)),
            _with(cr, (DIELECTRIC, 0.5, 0.0)),
            _with(_shade_part(cap, cv, white, 1.0), (DIELECTRIC, 0.5, 0.0))]


def _quad_uv(cv, L, T, B):
    """(u, v) coordinates of every canvas pixel on the parallelogram L + u(T-L) + v(B-L)."""
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    m = np.array([[T[0] - L[0], B[0] - L[0]], [T[1] - L[1], B[1] - L[1]]])
    inv = np.linalg.inv(m)
    dx, dy = xx - L[0], yy - L[1]
    return inv[0, 0] * dx + inv[0, 1] * dy, inv[1, 0] * dx + inv[1, 1] * dy


def schematic(cv, pal, name):
    """Blueprint paper: a diamond sheet with a faint grid and, when drawn on,
    a white plan of a machine."""
    light, mid, dark = pal
    L, T, R, B = (1.2, 8.0), (8.0, 1.4), (14.8, 8.0), (8.0, 14.6)
    c, h, m = _flat(cv.polygon(_round_poly([L, T, R, B], 0.6)), cv, _mix(mid, light, 0.25), dark, 0.9,
                    edge_w=0.35, grad=0.12, axis=(-1, -1))
    u, v = _quad_uv(cv, L, T, B)
    grid = (np.abs((u * 8) % 1 - 0.5) > 0.44) | (np.abs((v * 8) % 1 - 0.5) > 0.44)
    c = np.where((grid & m)[..., None], c * 1.12, c)
    if name == "schematic":
        lines = cv.blank()
        plan = [[(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75), (0.25, 0.25)],
                [(0.25, 0.5), (0.75, 0.5)], [(0.5, 0.25), (0.5, 0.75)]]
        to_xy = lambda uv: (L[0] + uv[0] * (T[0] - L[0]) + uv[1] * (B[0] - L[0]),
                            L[1] + uv[0] * (T[1] - L[1]) + uv[1] * (B[1] - L[1]))
        for pl in plan:
            cv2.polylines(lines, [cv.pts([to_xy(p) for p in pl])], False, 255,
                          thickness=max(1, int(round(0.45 * cv.k))), lineType=cv2.LINE_AA)
        circ = np.hypot(u - 0.5, v - 0.5)
        ring = np.abs(circ - 0.13) < 0.025
        drawn = ((lines > 0) | ring) & m
        c = np.where(drawn[..., None], np.array([0.94, 0.97, 1.0]), c)
    return [(np.clip(c, 0, 1), h, m)]


def filter_item(cv, pal, name):
    """Filter: a woven cloth panel stretched between two rods with end knobs
    (iron for the list filter, brass for the attribute filter)."""
    rods = BRASS if name == "attribute_filter" else IRONP
    rod_mat = (GOLD if name == "attribute_filter" else IRON, 0.72, 0.0)
    cloth_pal = ((246, 240, 226), (222, 212, 192), (128, 116, 98))
    cloth = cv.polygon([(2.2, 3.0), (13.8, 3.0), (13.8, 13.0), (2.2, 13.0)])
    c, h, m = _flat(cloth, cv, cloth_pal[1], cloth_pal[2], 0.8, edge_w=0.25, grad=0.1, axis=(-1, -1))
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    weave = np.sin(xx * math.pi * 2.5) * np.sin(yy * math.pi * 2.5)
    c = np.clip(c * (1 + 0.07 * weave[..., None]), 0, 1)
    parts = [(c, h + 0.01 * weave, m, (DIELECTRIC, 0.15, 0.0))]
    for y in (3.0, 13.0):
        parts.append(_with(_shade_part(cv.stroke((1.6, y), (14.4, y), 1.4), cv, rods, 0.7), rod_mat))
        for x in (1.4, 14.6):
            parts.append(_with(_shade_part(cv.circle((x, y), 1.15), cv, rods, 0.9), rod_mat))
    return parts


def dough(cv, pal, name):
    """A soft lump of dough, dusted with flour on top."""
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    lump = ((((xx - 8.0) / 6.4) ** 2 + ((yy - 9.4) / 4.4) ** 2 < 1) |
            (((xx - 7.0) / 4.2) ** 2 + ((yy - 7.2) / 3.0) ** 2 < 1)).astype(np.uint8) * 255
    c, h, m = _shade_part(lump, cv, pal, 3.4)
    soft = _noise(cv, 0.5, 72)
    c = np.clip(c * (1 + 0.03 * soft[..., None]), 0, 1)
    dust = (_noise(cv, 0.06, 71) > 1.3) & m & (yy < 9.5)
    c = np.where(dust[..., None], np.minimum(c * 1.1 + 0.04, 1), c)
    return [(c, h + 0.01 * soft, m, (DIELECTRIC, 0.25, 0.0))]


def chocolate_bar(cv, pal, name):
    """Bar of chocolate: a slab moulded into 3x2 raised squares."""
    light, mid, dark = pal
    L, T, R, B = (1.4, 7.6), (9.6, 3.4), (14.8, 7.0), (6.6, 11.2)
    parts = _slab(cv, [L, T, R, B], 2.2, pal, mid, _mix(mid, dark, 0.45), _mix(mid, dark, 0.2), rounding=0.6)
    u, v = _quad_uv(cv, L, T, B)
    gu, gv = (u * 3) % 1, (v * 2) % 1
    groove = (np.minimum(gu, 1 - gu) < 0.07) | (np.minimum(gv, 1 - gv) < 0.09)
    bevel = (np.minimum(gu, 1 - gu) < 0.16) | (np.minimum(gv, 1 - gv) < 0.2)
    c, h, m = parts[-1]
    lit_side = ((gu < 0.16) | (gv < 0.2)) & bevel & ~groove
    c = np.where(groove[..., None], np.array(dark, float) / 255, np.where(lit_side[..., None], c * 1.15, c))
    parts[-1] = (np.clip(c, 0, 1), h - 0.03 * groove - 0.015 * (bevel & ~groove), m)
    return [_with(p, (DIELECTRIC, 0.5, 0.0)) for p in parts]


def sweet_roll(cv, pal, name):
    """Cinnamon roll: a coiled golden bun, the spiral showing as a dark
    cinnamon groove, a glossy pool of icing over the middle."""
    dough_p = ((240, 190, 116), (204, 140, 72), (112, 62, 26))
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    cx, cy = 8.0, 7.4
    ex, ey = (xx - cx) / 6.8, (yy - cy) / 5.0
    r, th = np.hypot(ex, ey), np.arctan2(ey, ex)
    bun = (r < 1).astype(np.uint8) * 255
    c, h, m = _shade_part(bun, cv, dough_p, 3.0)
    side_m = ((np.hypot(ex, (yy - cy - 2.0) / 5.0) < 1) | ((np.abs(ex) < 1) & (yy > cy) & (yy < cy + 2.0))) & ~(r < 1)
    sd = _shade_part(side_m.astype(np.uint8) * 255, cv, _mix_pal(dough_p, 0.15), 1.2)
    phase = (r * 3.6 - th / (2 * math.pi)) % 1                  # archimedean coil
    coil = 0.5 + 0.5 * np.cos(2 * math.pi * phase)              # each turn a rounded ridge
    groove = (np.abs(phase - 0.5) < 0.09) & (r < 0.95)
    c = c * (0.86 + 0.18 * coil[..., None])
    c = np.where(groove[..., None], np.array([0.42, 0.2, 0.08]), c)
    # icing: a ribbon along the ridge of the coil, pooled in the centre
    icing = (((np.abs(phase) < 0.16) | (phase > 0.84)) & (r < 0.86) | (r < 0.2)) & m
    ic = np.array([0.99, 0.97, 0.93]) * (0.92 + 0.08 * coil)[..., None]
    c = np.where(icing[..., None], np.clip(ic, 0, 1), c)
    h = h + 0.03 * coil - 0.05 * groove + 0.05 * icing
    return [(*sd, (DIELECTRIC, 0.35, 0.0)),
            (np.clip(c, 0, 1), h, m, (DIELECTRIC, 0.35, 0.0)),
            (np.clip(c, 0, 1), h, icing, (DIELECTRIC, 0.75, 0.0))]


def blaze_cake(cv, pal, name):
    """Blaze cake: a round layer cake, sponge and dark-red filling bands
    curving round its side, glowing orange glaze swirled on top."""
    sponge = ((236, 140, 52), (204, 100, 34), (110, 40, 14))
    filling = np.array([0.5, 0.12, 0.08])
    glaze = ((255, 214, 92), (246, 150, 36), (140, 60, 10))
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    cx, top_y, bot_y, rx, ry = 8.0, 6.2, 11.2, 6.8, 2.8
    curve = ry * np.sqrt(np.clip(1 - ((xx - cx) / rx) ** 2, 0, 1))
    inx = np.abs(xx - cx) < rx
    side = inx & (yy > top_y) & (yy < bot_y + curve)
    topm = ((xx - cx) / rx) ** 2 + ((yy - top_y) / ry) ** 2 < 1
    depth = yy - top_y - curve                                  # distance below the rim, following the curve
    fill_band = (np.abs(depth - 1.7) < 0.45) | (np.abs(depth - 3.5) < 0.45)
    sp = _shade_part(side.astype(np.uint8) * 255, cv, sponge, 2.5)
    sc = np.where(fill_band[..., None], filling * (0.8 + 0.3 * (sp[0].mean(-1, keepdims=True) / 0.6)), sp[0])
    tp = _shade_part(topm.astype(np.uint8) * 255, cv, glaze, 2.0)
    rr = np.hypot((xx - cx) / rx, (yy - top_y) / ry)
    swirl = (np.abs(np.sin(rr * 10 - np.arctan2(yy - top_y, xx - cx))) > 0.88) & topm & (rr < 0.85)
    tc = np.where(swirl[..., None], np.array(glaze[0]) / 255, tp[0])
    drips = topm | (side & (depth < 0.7 + 0.8 * np.maximum(0, np.sin(xx * 1.7)) ** 4))
    em = 0.4 * topm
    return [(np.clip(sc, 0, 1), sp[1] - 0.02 * fill_band, sp[2], (DIELECTRIC, 0.3, 0.0)),
            (np.where(topm[..., None], tc, np.array(glaze[1]) / 255 * 0.95), np.maximum(tp[1], sp[1]) + 0.02,
             drips, (DIELECTRIC, 0.75, em))]


def honeyed_apple(cv, pal, name):
    """Apple dipped in honey: red apple, glossy amber coat with drips, stem."""
    apple_p = ((236, 92, 60), (186, 36, 30), (92, 14, 14))
    honey_p = ((255, 214, 96), (232, 150, 26), (130, 70, 10))
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    apple = ((np.hypot(xx - 6.4, (yy - 9.6) * 1.05) < 4.8) | (np.hypot(xx - 9.6, (yy - 9.6) * 1.05) < 4.8)) & (yy > 4.4)
    ac = _shade_part(apple.astype(np.uint8) * 255, cv, apple_p, 3.6)
    drip = 9.4 + 1.6 * np.maximum(0, np.sin(xx * 1.3 + 0.6)) ** 3 + 0.4 * np.sin(xx * 0.7)
    coat = apple & (yy < drip)
    hc = _shade_part(coat.astype(np.uint8) * 255, cv, honey_p, 2.0)
    stem = _shade_part(cv.stroke((8.0, 5.8), (7.4, 3.2), 0.9), cv, ((140, 98, 54), (96, 64, 32), (50, 30, 14)), 0.45)
    glint = (cv.stroke((4.8, 7.0), (6.4, 5.8), 0.6) > 0) & coat
    hcol = np.where(glint[..., None], np.array([1.0, 0.96, 0.8]), hc[0])
    return [_with(ac, (DIELECTRIC, 0.6, 0.0)), (hcol, hc[1] + 0.02, hc[2], (DIELECTRIC, 0.88, 0.0)),
            _with(stem, (DIELECTRIC, 0.3, 0.0))]


def brass_hand(cv, pal, name):
    """Mechanical brass hand pointing up and to the right: broad riveted palm,
    jointed index finger, curled fingers, raised thumb, dark iron wrist cuff."""
    br = BRASS
    gold = (GOLD, 0.78, 0.0)
    cuff = cv.polygon(_round_poly([(0.8, 12.4), (3.8, 9.4), (6.8, 12.4), (3.8, 15.4)], 0.6))
    palm = cv.polygon(_round_poly([(3.4, 11.0), (5.4, 5.8), (10.4, 5.2), (12.2, 9.6), (8.0, 13.0)], 1.4))
    index1 = cv.stroke((9.0, 6.0), (11.0, 3.6), 2.5)
    index2 = cv.stroke((11.0, 3.6), (12.6, 1.8), 2.2)
    curled = [cv.stroke((10.6, 7.2), (13.0, 6.8), 2.4), cv.stroke((10.8, 9.6), (12.8, 9.4), 2.3)]
    thumb = cv.stroke((5.6, 7.6), (4.2, 3.6), 2.3)
    parts = [_with(_shade_part(cuff, cv, _mix_pal(IRONP, 0.4), 1.4), (IRON, 0.55, 0.0)),
             _with(_shade_part(thumb, cv, br, 1.1), gold),
             _with(_shade_part(palm, cv, br, 3.0), gold)]
    parts += [_with(_shade_part(m, cv, _mix_pal(br, 0.1), 1.1), gold) for m in curled]
    parts += [_with(_shade_part(index1, cv, br, 1.2), gold), _with(_shade_part(index2, cv, br, 1.1), gold)]
    for c, r in (((11.0, 3.6), 0.55), ((7.4, 9.2), 0.7)):    # knuckle pin and palm rivet
        parts.append(_with(_shade_part(cv.circle(c, r), cv, IRONP, 0.5), (IRON, 0.7, 0.0)))
    return parts


def whisk(cv, pal, name):
    """Balloon whisk: wire loops fanned round the axis, dark handle with a
    metal ferrule."""
    a0, a1 = np.array((5.4, 10.6)), np.array((14.2, 1.8))
    u = (a1 - a0) / np.linalg.norm(a1 - a0)
    n = np.array((-u[1], u[0]))
    L = np.linalg.norm(a1 - a0)
    wires = cv.blank()
    for w in (3.4, 2.1, 0.7, -0.7, -2.1, -3.4):
        pts = [tuple(a0 + u * L * t + n * w * math.sin(math.pi * t) ** 0.8) for t in np.linspace(0, 1, 40)]
        cv2.polylines(wires, [cv.pts(pts)], False, 255, thickness=max(1, int(round(0.5 * cv.k))), lineType=cv2.LINE_8)
    handle = cv.stroke((1.6, 14.4), (4.4, 11.6), 2.2)
    ferrule = cv.stroke((4.2, 11.8), (5.8, 10.2), 1.8)
    wire = _flat(wires, cv, IRONP[0], IRONP[1], 0.85, edge_w=0.0, grad=0.3, axis=(-1, -1))
    return [_with(wire, (IRON, 0.72, 0.0)),
            _with(_shade_part(handle, cv, DARKP, 1.0), (DIELECTRIC, 0.35, 0.0)),
            _with(_shade_part(ferrule, cv, IRONP, 0.8), (IRON, 0.7, 0.0))]


def cardboard(cv, pal, name):
    """A sheet of corrugated cardboard, creased across, flutes on the edge."""
    light, mid, dark = pal
    L, T, R, B = (1.0, 8.0), (8.6, 3.6), (15.0, 8.0), (7.4, 12.4)
    parts = _slab(cv, [L, T, R, B], 1.1, pal, mid, _mix(mid, dark, 0.4), _mix(mid, dark, 0.25), rounding=0.4)
    u, v = _quad_uv(cv, L, T, B)
    c, h, m = parts[-1]
    crease = np.abs(u - 0.55) < 0.025
    c = np.where(crease[..., None], c * 0.8, c)
    fibre = _noise(cv, 0.1, 81)
    parts[-1] = (np.clip(c * (1 + 0.04 * fibre[..., None]), 0, 1), h - 0.02 * crease, m)
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    for i in (0, 1):                                          # flutes along the cut edges
        c2, h2, m2 = parts[i]
        flutes = np.sin((xx + yy * 0.3) * math.pi * 2.6)
        parts[i] = (np.clip(c2 * (1 + 0.12 * flutes[..., None]), 0, 1), h2, m2)
    return [_with(p, (DIELECTRIC, 0.2, 0.0)) for p in parts]


def belt_connector(cv, pal, name):
    """A rolled-up rubber drive belt: the coil's face shows the wound layers,
    the roll recedes to the right and a loose tail hangs off the bottom."""
    light, mid, dark = pal
    rub = (_mix(light, mid, 0.3), mid, dark)
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    fx, fy, rx, ry, depth = 6.0, 7.4, 3.8, 5.0, 5.2
    face = ((xx - fx) / rx) ** 2 + ((yy - fy) / ry) ** 2 < 1
    back = ((xx - fx - depth) / rx) ** 2 + ((yy - fy) / ry) ** 2 < 1
    body = (back | ((xx > fx) & (xx < fx + depth) & (np.abs(yy - fy) < ry))) & ~face
    tail = cv.polygon(_round_poly([(fx + depth - 0.4, fy + ry - 1.6), (15.2, 12.6), (14.6, 14.8),
                                   (fx + 1.0, fy + ry + 0.2)], 0.5)) > 0
    bp = _shade_part(body.astype(np.uint8) * 255, cv, rub, 2.0)
    treads = np.cos(xx * math.pi * 2.2) > 0.4
    bc = np.where(treads[..., None], bp[0] * 0.86, bp[0])
    tp = _shade_part((tail & ~face & ~body).astype(np.uint8) * 255, cv, _mix_pal(rub, 0.15), 0.8)
    fp = _shade_part(face.astype(np.uint8) * 255, cv, _mix_pal(rub, 0.25), 1.5)
    r = np.hypot((xx - fx) / rx, (yy - fy) / ry)
    th = np.arctan2((yy - fy) / ry, (xx - fx) / rx)
    layers = (np.abs(((r * 5 - th / (2 * math.pi)) % 1) - 0.5) < 0.12) & (r > 0.2)
    fc = np.where(layers[..., None], np.array(dark) / 255 * 0.8, fp[0])
    hub = r < 0.2
    fc = np.where(hub[..., None], np.array([0.08, 0.07, 0.07]), fc)
    mat = (DIELECTRIC, 0.35, 0.0)
    return [(*tp, mat), (bc, bp[1], bp[2], mat), (fc, fp[1] - 0.04 * layers - 0.1 * hub, fp[2], mat)]


def experience_nugget(cv, pal, name):
    """Nugget of experience: a small glowing green faceted gem."""
    from handmade.items import emerald
    parts = emerald(cv, pal)
    return [(c, h, m, (12, 0.9, 0.55)) for c, h, m in parts]


def integrated_circuit(cv, pal, name):
    """Circuit board: a slab with copper traces, gold edge contacts, a chip."""
    light, mid, dark = pal
    L, T, R, B = (1.4, 7.8), (8.8, 3.6), (14.8, 7.8), (7.4, 12.0)
    parts = [_with(p, (DIELECTRIC, 0.45, 0.0)) for p in
             _slab(cv, [L, T, R, B], 1.3, pal, mid, _mix(mid, dark, 0.4), _mix(mid, dark, 0.25), rounding=0.4)]
    u, v = _quad_uv(cv, L, T, B)
    c, h, m = parts[-1][:3]
    traces = ((np.abs((u * 6) % 1 - 0.5) < 0.06) & (v > 0.2) & (v < 0.8)) | \
             ((np.abs((v * 5) % 1 - 0.5) < 0.06) & (u > 0.15) & (u < 0.4))
    c = np.where((traces & m)[..., None], np.array(BRASS[1]) / 255, c)
    parts[-1] = (c, h + 0.005 * traces, m, (DIELECTRIC, 0.45, 0.0))
    contacts = m & (v > 0.86) & ((u * 7) % 1 < 0.6)
    parts.append((np.broadcast_to(np.array(BRASS[0]) / 255, (cv.n, cv.n, 3)).copy(), h + 0.01, contacts, (GOLD, 0.85, 0.0)))
    chip = (u > 0.42) & (u < 0.78) & (v > 0.25) & (v < 0.65)
    parts.append((np.broadcast_to(np.array([0.12, 0.12, 0.14]), (cv.n, cv.n, 3)).copy(), h + 0.05, chip & m,
                  (DIELECTRIC, 0.6, 0.0)))
    return parts


THIN = (whisk,)  # wire designs: a full-width outline would swallow the wires
SHEETS = r"(iron|golden|copper|brass|sturdy|lapis|bronze|steel|electrum|zinc)_sheet"
DESIGNS = [
    (re.compile(rf"^(create|alloyed|createaddition):{SHEETS}$"), sheet),
    (re.compile(r"^create:(andesite_alloy|shadow_steel|refined_radiance)$"), alloy),
    (re.compile(r"^create:crushed_raw_\w+$"), crushed),
    (re.compile(r"^(create:(cinder_flour|wheat_flour|powdered_obsidian)|createfood:corn_flour)$"), powder),
    (re.compile(r"^create:(red_)?sand_paper$"), paper_grit),
    (re.compile(r"^create:(incomplete_)?precision_mechanism$"), precision_mechanism),
    (re.compile(r"^create:brass_hand$"), brass_hand),
    (re.compile(r"^create:whisk$"), whisk),
    (re.compile(r"^create:cardboard$"), cardboard),
    (re.compile(r"^create:belt_connector$"), belt_connector),
    (re.compile(r"^create:experience_nugget$"), experience_nugget),
    (re.compile(r"^create:integrated_circuit$"), integrated_circuit),
    (re.compile(r"^create:electron_tube$"), electron_tube),
    (re.compile(r"^create:propeller$"), propeller),
    (re.compile(r"^create:goggles$"), goggles),
    (re.compile(r"^create:rose_quartz$"), rose_quartz),
    (re.compile(r"^create:polished_rose_quartz$"), polished_rose_quartz),
    (re.compile(r"^create:super_glue$"), super_glue),
    (re.compile(r"^create:(empty_)?schematic$"), schematic),
    (re.compile(r"^create:(attribute_)?filter$"), filter_item),
    (re.compile(r"^create:dough$"), dough),
    (re.compile(r"^create:bar_of_chocolate$"), chocolate_bar),
    (re.compile(r"^create:sweet_roll$"), sweet_roll),
    (re.compile(r"^create:blaze_cake$"), blaze_cake),
    (re.compile(r"^create:honeyed_apple$"), honeyed_apple),
]


def create_item_of(path, ns):
    if not path.startswith("textures/item/") or path.count("/") != 2:
        return None  # sub-folders hold model textures (e.g. item/package/), not icons
    key = f"{ns}:{path.split('/')[-1][:-4]}"
    return next((fn for pat, fn in DESIGNS if pat.match(key)), None)


def _material_for(name):
    if re.search(r"gold|brass", name):
        return GOLD if "gold" in name else ALBEDO_METAL, 0.8
    if re.search(r"iron|steel|sturdy", name):
        return IRON if "iron" in name else ALBEDO_METAL, 0.68
    if "copper" in name:
        from texgen import COPPER
        return COPPER, 0.7
    if re.search(r"zinc|bronze|electrum|shadow", name):
        return ALBEDO_METAL, 0.66
    if "rose_quartz" in name:
        return 40, (0.95 if "polished" in name else 0.8)
    if "lapis" in name:
        return 40, 0.6
    if "refined_radiance" in name:
        return DIELECTRIC, 0.7
    return DIELECTRIC, 0.3


def render(path, src, size, ns="create"):
    name = path.split("/")[-1][:-4]
    fn = create_item_of(path, ns)
    pal = palette_from(src)
    cv = Canvas(size)
    parts = fn(cv, pal, name)
    base_mat = _material_for(name) + (0.0,)
    col = np.zeros((cv.n, cv.n, 3))
    h = np.zeros((cv.n, cv.n))
    f0 = np.full((cv.n, cv.n), float(DIELECTRIC))
    sm = np.zeros((cv.n, cv.n))
    em = np.zeros((cv.n, cv.n))
    cover = np.zeros((cv.n, cv.n), bool)
    for layer, p in enumerate(parts):
        c, hh, m = p[:3]
        mf, ms, me = p[3] if len(p) > 3 else base_mat
        col = np.where(m[..., None], c, col)
        h = np.where(m, hh + 0.02 * layer, h)
        f0 = np.where(m, mf, f0)
        sm = np.where(m, ms, sm)
        em = np.where(m, me, em)
        cover |= m
    if name == "refined_radiance":
        em = np.where(cover, 0.6, 0.0)
    rim, rim_col = _outline(cover, cv, pal[2] if fn in (sheet, alloy, crushed, powder, paper_grit) else DARKP[2],
                            width=0.18 if fn in THIN else 0.55)
    col = np.where(rim[..., None], rim_col[None, None, :], col)
    h = np.where(rim, h - 0.05, h)
    down = lambda a: cv2.resize(a.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    alpha = down(cover)
    w = np.maximum(alpha, 1e-4)
    rgb = np.dstack([down(col[..., i] * cover) for i in range(3)]) / w[..., None]
    height = down(h * cover) / w
    m = Material(size)
    solid = alpha > 0.5
    m.albedo, m.alpha = np.clip(rgb, 0, 1), alpha
    m.height = np.where(solid, np.clip(height + (0.99 - height[solid].max()), 0, 0.99), 0.0)
    # metal ids must not be averaged: take the dominant part's value per pixel
    m.f0 = np.where(solid, cv2.resize(f0.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST), DIELECTRIC)
    m.smooth = np.where(solid, down(sm * cover) / w, 0.0)
    m.emissive = np.where(solid, down(em * cover) / w, 0.0)
    return derive(m, edge_wear=0.0, edges="reflect", normal_strength=4.0)
