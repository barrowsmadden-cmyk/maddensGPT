"""Hand-made Create items (and addon items that share their designs).

House style (handmade/style.py): traced from the Create originals (texel
(x, y) spans x..x+1, y..y+1 in the 16-unit grid), colours from the original
texels each part covers, crisp bevel/cel shading, bold outline, hard alpha.
Manufactured things get hand-placed straight-edged parts; natural things
(crushed ore, powders, food) keep the original's own colour pattern, traced
smoothly and posterized to its colours.
"""
import math
import re

import cv2
import numpy as np

from handmade.tools import Canvas
from handmade.style import part, render_item, traced, silhouette, edge_band, round_poly


def _stroke_poly(cv, pts, width):
    m = cv.blank()
    cv2.polylines(m, [cv.pts(pts)], False, 255, thickness=max(1, int(round(width * cv.k))), lineType=cv2.LINE_8)
    return m


# ------------------------------------------------------------- designs ---

def _extremes(src):
    """Leftmost, topmost, rightmost, bottommost points of the original's
    silhouette (texel edges, 16-unit grid)."""
    h, w = src.shape[:2]
    a = src[..., 3] > 0.5
    ys, xs = np.nonzero(a)
    k = 16.0 / w
    L = (xs.min() * k, (ys[xs == xs.min()].mean() + 0.5) * k)
    R = ((xs.max() + 1) * k, (ys[xs == xs.max()].mean() + 0.5) * k)
    T = ((xs[ys == ys.min()].mean() + 0.5) * k, ys.min() * k)
    B = ((xs[ys == ys.max()].mean() + 0.5) * k, (ys.max() + 1) * k)
    return L, T, R, B


def plate(cv, src, name):
    """Pressed sheet: a plate in perspective fitted to the original's extreme
    points; lit top face, its one-texel thickness showing along the front
    edges, a bright rim on the front-left edge."""
    m = {"golden_sheet": "gold", "iron_sheet": "iron", "lapis_sheet": "crystal",
         "sturdy_sheet": "dark_metal"}.get(name, "metal")
    L, T, R, B = (np.array(p) for p in _extremes(src))
    t = 1.0
    top = [L - (0, t * 0.5), T, R - (0, t * 0.5), B - (0, t)]
    c0 = np.mean(top, 0)
    top = [tuple(p + (p - c0) / np.linalg.norm(p - c0) * 0.45) for p in top]   # originals bulge past the corners
    body = cv.polygon(round_poly(top, 0.6))
    for dy in np.linspace(0, t, 6):
        body |= cv.polygon(round_poly([(x, y + dy) for x, y in top], 0.6))
    face = cv.polygon(round_poly(top, 0.6))
    Lt, Bt = np.array(top[0]), np.array(top[3])
    c = np.mean(np.array(top), 0)
    inward = lambda p, k: tuple(np.array(p) + (c - np.array(p)) / np.linalg.norm(c - np.array(p)) * k)
    rim = cv.stroke(inward(Lt, 0.9), inward(Bt, 0.9), 0.8)
    return [part(body, m, "flat", h=0.78, glint=False),
            part(face, m, "bevel", bevel=0.7, h=0.88, inner_line=False),
            part((rim > 0) & (face > 0), m, "flat", h=0.92, inner_line=False,
                 tones=None)]


def curled_sheet(cv, src, name):
    """Copper sheet: a curled plate; its rolled light bands are the original's."""
    return [part(traced(src, cv), "copper", "post", bevel=0.8, h=0.88)]


def alloy(cv, src, name):
    """Cast block (andesite alloy, shadow steel, refined radiance): lit top,
    left end, dark front, bright edges where the faces meet."""
    m = {"andesite_alloy": "stone", "shadow_steel": "dark_metal", "refined_radiance": "glossy"}[name]
    top = cv.polygon([(2.0, 5.5), (8.0, 2.0), (10.0, 2.0), (14.0, 6.0), (6.6, 8.9)])
    left = cv.polygon([(2.0, 5.5), (6.6, 8.9), (6.6, 14.0), (5.0, 14.0), (2.0, 11.0)])
    front = cv.polygon([(6.6, 8.9), (14.0, 6.0), (14.0, 11.0), (11.0, 13.0), (8.0, 14.0), (6.6, 14.0)])
    em = 0.6 if name == "refined_radiance" else 0.0
    return [part(front, m, "bevel", bevel=0.7, h=0.8, glint=False, inner_line=False, emissive=em),
            part(left, m, "bevel", bevel=0.7, h=0.8, glint=False, inner_line=False, emissive=em),
            part(top, m, "bevel", bevel=0.8, h=0.88, inner_line=False, emissive=em),
            part(cv.stroke((2.6, 6.0), (6.4, 8.8), 0.75), m, "flat", h=0.9, inner_line=False, emissive=em),
            part(cv.stroke((6.6, 9.4), (6.6, 12.8), 0.75), m, "flat", h=0.86, inner_line=False, emissive=em)]


def _cog(cv, c, r_out, r_in, teeth, hub=0.0, phase=0.0):
    """Gear wheel mask: `teeth` square-ish teeth between r_in and r_out, optional hole."""
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    ang = np.arctan2(yy - c[1], xx - c[0]) + phase
    rr = np.hypot(xx - c[0], yy - c[1])
    tooth = np.cos(ang * teeth) > 0.15
    m = (rr < np.where(tooth, r_out, r_in)) & (rr >= hub)
    return m.astype(np.uint8) * 255


def tex(src, x, y):
    """Colour (sRGB 0-255) of the original texel at (x, y) in the 16 grid: the eyedropper."""
    h, w = src.shape[:2]
    k = w / 16.0
    return tuple(np.round(src[int(y * k), int(x * k), :3] * 255))


def pal(src, light, face, dark):
    """(light, face, dark) tones picked from three texels of the original."""
    return tex(src, *light), tex(src, *face), tex(src, *dark)


def precision_mechanism(cv, src, name):
    """A raised brass ring round a recess filled by a large brass cog (dark only
    in the gaps between its teeth; an empty dark well when incomplete), an
    iron gear set into the lower right of the rim."""
    c = (7.6, 8.4)
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    ell = lambda rx, ry, cc=c: ((((xx - cc[0]) / rx) ** 2 + ((yy - cc[1]) / ry) ** 2) < 1).astype(np.uint8) * 255
    incomplete = name.startswith("incomplete")
    brass = pal(src, (7, 3), (4, 5), (11, 5)) if not incomplete else pal(src, (7, 3), (2, 10), (9, 12))
    iron = pal(src, (11, 10), (12, 9), (13, 9))
    well = pal(src, (10, 4), (6, 8), (4, 8)) if incomplete else pal(src, (11, 4), (10, 4), (5, 4))
    parts = [part(ell(6.6, 6.2), "gold", "bevel", bevel=1.0, h=0.9, tones=brass, form=False),
             part(ell(4.4 if incomplete else 4.0, 4.1 if incomplete else 3.8, (7.4, 8.1)), "gold", "flat", h=0.72,
                  glint=False, tones=well)]
    if not incomplete:
        parts += [part(_cog(cv, (7.4, 8.0), 4.0, 3.2, 9, phase=0.2), "gold", "bevel", bevel=0.8, h=0.84,
                       tones=brass, form=False),
                  part(ell(1.2, 1.2, (7.4, 8.0)), "gold", "flat", h=0.78, glint=False, tones=well)]
    parts += [part(_cog(cv, (12.1, 10.7), 2.6, 1.9, 6, phase=0.3), "iron", "bevel", bevel=0.7, h=0.92, tones=iron),
              part(ell(0.75, 0.75, (12.1, 10.7)), "iron", "flat", h=0.86, glint=False,
                   tones=((90, 90, 96), (60, 60, 66), (30, 30, 34)))]
    return parts


def _bright(src, q):
    """Original texels in the lightest (1 - q) share: the clean material colour."""
    lab = cv2.cvtColor(src[..., :3].astype(np.float32), cv2.COLOR_RGB2LAB)[..., 0]
    a = src[..., 3] > 0.5
    return a & (lab >= np.quantile(lab[a], q))


def _texels(src, pred):
    """Original-resolution mask of texels (x, y in the 16 grid) where pred(x, y) holds."""
    h, w = src.shape[:2]
    k = 16.0 / w
    return np.array([[pred((x + 0.5) * k, (y + 0.5) * k) for x in range(w)] for y in range(h)])


def electron_tube(cv, src, name):
    """Vacuum tube: glass bulb glowing white-gold at the top round a red
    rose-quartz core, on a grey collar and base with three pins."""
    bulb = cv.polygon(round_poly([(4.0, 9.6), (4.0, 3.2), (5.2, 1.0), (10.8, 1.0), (12.0, 3.2), (12.0, 9.6)], 1.6))
    core = cv.polygon(round_poly([(6.4, 9.6), (6.4, 6.2), (7.4, 4.8), (9.2, 4.8), (10.0, 6.2), (10.0, 9.6)], 0.6))
    glow = cv.polygon(round_poly([(5.4, 5.4), (5.4, 2.6), (6.2, 1.8), (9.8, 1.8), (10.6, 2.6), (10.6, 4.4),
                                  (9.6, 4.8), (7.0, 4.8)], 0.8))
    collar = cv.polygon(round_poly([(3.8, 9.6), (12.2, 9.6), (12.2, 11.2), (3.8, 11.2)], 0.3))
    base = cv.polygon(round_poly([(4.2, 11.2), (11.8, 11.2), (11.8, 13.2), (4.2, 13.2)], 0.3))
    pins = [cv.stroke((x, 13.0), (x, 14.6), 0.9) for x in (5.5, 8.0, 10.5)]
    glass, core_t, glow_t = pal(src, (6, 2), (5, 5), (10, 5)), pal(src, (8, 5), (7, 7), (8, 9)), pal(src, (6, 4), (7, 3), (9, 3))
    collar_t, base_t = pal(src, (6, 11), (5, 10), (9, 10)), pal(src, (5, 13), (7, 13), (9, 13))
    return ([part(p, "iron", "flat", h=0.7, glint=False, tones=base_t) for p in pins] +
            [part(base, "iron", "bevel", bevel=0.6, h=0.78, tones=base_t),
             part(collar, "iron", "bevel", bevel=0.6, h=0.84, tones=collar_t),
             part(bulb, "glass", "cyl", axis=((8.0, 9.6), (8.0, 1.0)), h=0.9, tones=glass),
             part(core, "crystal", "bevel", bevel=0.7, h=0.86, emissive=0.5, inner_line=False, tones=core_t),
             part(glow, "glass", "flat", h=0.92, emissive=0.75, inner_line=False, tones=glow_t)])


def propeller(cv, src, name):
    """Four paddle blades radiating from a small hub, each twisted: the half on
    the leading side lit, the trailing half in shade."""
    lit, shade = pal(src, (12, 7), (11, 6), (5, 3)), pal(src, (2, 8), (3, 8), (6, 8))
    hub_t = pal(src, (7, 9), (6, 8), (9, 7))
    hub = np.array((7.6, 7.8))
    parts = []
    for tip in ((5.4, 2.6), (14.2, 6.8), (1.8, 9.0), (9.8, 13.2)):
        tip = np.array(tip)
        u = (tip - hub) / np.linalg.norm(tip - hub)
        n = np.array((-u[1], u[0]))                           # leading side (rotation sense)
        root = hub + u * 0.6
        wr, wt = 1.5, 2.9
        lead = [root, root + n * wr, tip + n * wt - u * 0.6, tip + n * wt * 0.5 + u * 0.2, tip + u * 0.3]
        trail = [root, tip + u * 0.3, tip - n * wt * 0.5 + u * 0.2, tip - n * wt - u * 0.6, root - n * wr]
        parts.append(part(cv.polygon(round_poly([tuple(p) for p in trail], 0.45)), "iron", "bevel", bevel=0.55,
                          h=0.82, tones=shade, glint=False))
        parts.append(part(cv.polygon(round_poly([tuple(p) for p in lead], 0.45)), "iron", "bevel", bevel=0.55,
                          h=0.86, tones=lit, inner_line=False))
    parts.append(part(cv.circle(tuple(hub), 1.7), "iron", "bevel", bevel=0.6, h=0.92, tones=hub_t))
    return parts


def _rrect(cv, x0, y0, x1, y1, r):
    return cv.polygon(round_poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], r))


def goggles(cv, src, name):
    """Engineer's goggles: thick leather strap loop, rounded-square brass
    frames with blue glass, a dark leather bridge."""
    strap_t, frame_t = pal(src, (12, 4), (10, 4), (6, 4)), pal(src, (3, 8), (2, 10), (6, 10))
    glass_t, bridge_t = pal(src, (11, 10), (3, 9), (4, 9)), pal(src, (8, 10), (7, 9), (7, 8))
    strap = _stroke_poly(cv, [(2.2, 7.8), (2.6, 5.4), (4.6, 3.9), (7.5, 3.5), (10.6, 3.7), (12.8, 4.6), (13.8, 7.0)], 1.9)
    parts = [part(strap, "leather", "cyl", axis=((2.0, 3.0), (14.0, 3.0)), h=0.8, tones=strap_t),
             part(_rrect(cv, 6.4, 7.8, 9.6, 10.6, 0.4), "leather", "bevel", bevel=0.5, h=0.82, tones=bridge_t)]
    for x0 in (1.0, 9.0):
        parts.append(part(_rrect(cv, x0 + 0.2, 7.2, x0 + 5.8, 12.6, 1.6), "gold", "bevel", bevel=0.9, h=0.9, tones=frame_t))
        parts.append(part(_rrect(cv, x0 + 1.8, 8.8, x0 + 5.0, 11.2, 0.6), "glass", "bevel", bevel=0.6, h=0.86,
                          tones=glass_t))
    return parts


def super_glue(cv, src, name):
    """Glue tube lying diagonally: lit green body, darker crimped end with fold
    ridges at the top right, white cap at the bottom left."""
    body_t, crimp_t, cap_t = pal(src, (7, 8), (5, 9), (9, 8)), pal(src, (8, 3), (10, 3), (10, 6)), \
        pal(src, (3, 10), (3, 9), (4, 12))
    body = cv.polygon(round_poly([(3.0, 8.6), (5.0, 7.0), (9.0, 6.2), (13.0, 5.8), (15.0, 6.6), (14.0, 8.4),
                                  (11.0, 10.0), (9.0, 11.0), (5.0, 11.2)], 0.9))
    crimp = cv.polygon(round_poly([(7.0, 2.0), (12.0, 2.0), (15.0, 5.0), (15.0, 6.8), (12.4, 6.4), (9.2, 6.6),
                                   (7.6, 4.4)], 0.5))
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    ridges = (np.abs(((xx - yy) % 1.6) - 0.8) < 0.22) & (crimp > 0)
    cap = cv.stroke((2.0, 12.4), (5.2, 10.4), 2.8)
    return [part(crimp, "rubber", "bevel", bevel=0.6, h=0.84, tones=crimp_t, overlay=[(ridges, 1.25, 0.02)]),
            part(body, "glossy", "cyl", axis=((4.0, 10.0), (14.0, 6.6)), h=0.88, tones=body_t),
            part(cap, "glossy", "cyl", axis=((2.0, 12.4), (5.2, 10.4)), h=0.86, tones=cap_t)]


def filter_item(cv, src, name):
    """Filter: a woven cloth panel stretched between two upright rods (iron for
    the list filter, brass-capped wood for the attribute filter)."""
    rod_m = "wood" if name == "attribute_filter" else "iron"
    cloth = _rrect(cv, 1.8, 1.8, 12.4, 13.2, 0.3)
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    weave = ((np.abs((xx % 1.0) - 0.5) < 0.16) | (np.abs((yy % 1.0) - 0.5) < 0.16)) & (cloth > 0)
    parts = [part(cloth, "cloth", "bevel", bevel=0.5, h=0.8, region=_texels(src, lambda x, y: 3 <= x < 11 and 3 <= y < 12),
                  overlay=[(weave, 0.9, -0.01)], form=False)]
    for x0 in (0.0, 12.2):
        parts.append(part(_rrect(cv, x0, 0.8, x0 + 2.2, 14.2, 0.9), rod_m, "cyl", axis=((x0 + 1.1, 14), (x0 + 1.1, 1)),
                          h=0.9, region=_texels(src, lambda x, y, x0=x0: x0 <= x < x0 + 2.2 and 3 <= y < 12)))
        for y0 in (0.6, 12.6):                               # end caps
            parts.append(part(_rrect(cv, x0 - 0.1, y0, x0 + 2.3, y0 + 1.6, 0.6), rod_m, "bevel", bevel=0.5, h=0.92,
                              region=_texels(src, lambda x, y, x0=x0, y0=y0: x0 <= x < x0 + 2.2 and y0 <= y < y0 + 1.6)))
    return parts


def schematic(cv, src, name):
    """Blueprint paper: a diamond sheet with a darker thickness edge, a faint
    grid and, when drawn on, a white plan of a block."""
    L, T, R, B = (1.0, 7.4), (8.0, 1.0), (15.4, 7.4), (8.0, 14.6)
    sheet = cv.polygon(round_poly([L, T, R, B], 0.6))
    under = cv.polygon(round_poly([(L[0], L[1] + 0.9), T, (R[0], R[1] + 0.9), (B[0], B[1] + 0.9)], 0.6))
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    u, v = (xx - L[0]) + (yy - L[1]), (xx - L[0]) - (yy - L[1])   # along the diamond's edges
    grid = ((np.abs((u / 1.6) % 1 - 0.5) > 0.42) | (np.abs((v / 1.6) % 1 - 0.5) > 0.42)) & (sheet > 0)
    paper_t = pal(src, (2, 8), (9, 5), (3, 7)) if name == "schematic" else pal(src, (8, 3), (6, 5), (8, 4))
    paper = part(sheet, "paper", "bevel", bevel=0.6, h=0.88, tones=paper_t, overlay=[(grid, 0.86, -0.005)], form=False)
    parts = [part(under, "paper", "flat", h=0.8, region=~_bright(src, 0.25) & (src[..., 3] > 0.5)), paper]
    if name == "schematic":                                   # a block drawn on the plan in dark ink
        c = np.array((8.2, 7.6))
        top = [c + (0, -3.0), c + (2.6, -1.5), c + (0, 0), c + (-2.6, -1.5)]
        lines = [top + [top[0]], [c + (-2.6, -1.5), c + (-2.6, 1.5), c + (0, 3.0), c + (2.6, 1.5), c + (2.6, -1.5)],
                 [c, c + (0, 3.0)]]
        ink = cv.blank()
        for ln in lines:
            cv2.polylines(ink, [cv.pts([tuple(p) for p in ln])], False, 255, thickness=max(1, int(0.5 * cv.k)),
                          lineType=cv2.LINE_8)
        parts.append(part((ink > 0) & (sheet > 0), "paper", "flat", h=0.9, inner_line=False,
                          tones=pal(src, (8, 5), (7, 5), (6, 9))))
    return parts


def paper_sheet(cv, src, name):
    """Sand paper and cardboard: a flat sheet fitted to the original, grit or
    paper fibre on its face."""
    parts = plate(cv, src, name)
    for p in parts:
        p["mat"] = "powder" if "sand_paper" in name else "paper"
    if "sand_paper" in name:
        rng = np.random.default_rng(3)
        grit = cv2.GaussianBlur(rng.random((cv.n, cv.n)).astype(np.float32), (0, 0), cv.k * 0.12)
        g = (grit - grit.mean()) / grit.std()
        parts[1]["overlay"] = [(g > 1.0, 0.86, -0.01), (g < -1.1, 1.08, 0.01)]
    return parts


def belt_connector(cv, src, name):
    """A rolled-up drive belt: the roll's face shows the wound layers, the
    roll runs back to the right, a loose tail lies along the bottom; treads
    across the rubber."""
    face_t, body_t, tail_t = pal(src, (4, 5), (3, 6), (6, 8)), pal(src, (10, 9), (9, 11), (8, 13)), \
        pal(src, (4, 10), (6, 12), (8, 14))
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    fx, fy, rx, ry, depth = 5.2, 7.4, 4.4, 5.0, 6.6
    face = ((xx - fx) / rx) ** 2 + ((yy - fy) / ry) ** 2 < 1
    back = ((xx - fx - depth) / rx) ** 2 + ((yy - fy) / ry) ** 2 < 1
    body = (back | ((xx > fx) & (xx < fx + depth) & (np.abs(yy - fy) < ry))) & ~face
    tail = cv.polygon(round_poly([(fx + 1.0, fy + ry - 1.4), (15.4, 12.0), (15.2, 14.6), (fx + 0.6, fy + ry + 1.0)], 0.6)) > 0
    r = np.hypot((xx - fx) / rx, (yy - fy) / ry)
    th = np.arctan2((yy - fy) / ry, (xx - fx) / rx)
    layers = (np.abs(((r * 4.0 - th / (2 * math.pi)) % 1) - 0.5) < 0.13) & (r > 0.24) & face
    hub = (r < 0.24) & face
    treads = (np.abs(((xx * 1.0) % 1.3) - 0.65) < 0.17)
    return [part((tail & ~face & ~body).astype(np.uint8), "rubber", "bevel", bevel=0.6, h=0.78, tones=tail_t,
                 overlay=[(treads, 0.85, -0.01)]),
            part(body.astype(np.uint8), "rubber", "cyl", axis=((fx, fy + ry), (fx, fy - ry)), h=0.84, tones=body_t,
                 overlay=[(treads & body, 0.85, -0.01)]),
            part(face.astype(np.uint8), "rubber", "bevel", bevel=0.7, h=0.88, tones=face_t,
                 overlay=[(layers, 0.7, -0.03), (hub, 0.35, -0.08)])]


def integrated_circuit(cv, src, name):
    """A chip board seen at an angle: purple-blue top, navy edge, gold pins
    along the two front edges."""
    top_t, edge_t, pin_t = pal(src, (6, 4), (7, 6), (5, 7)), pal(src, (12, 8), (11, 9), (3, 9)), \
        pal(src, (1, 11), (13, 9), (6, 12))
    Lp, T, R, B = np.array((1.0, 6.4)), np.array((8.4, 2.0)), np.array((14.8, 6.0)), np.array((7.2, 10.2))
    topm = cv.polygon(round_poly([tuple(Lp), tuple(T), tuple(R), tuple(B)], 0.5))
    body = topm.copy()
    for dy in np.linspace(0, 1.8, 8):
        body |= cv.polygon(round_poly([tuple(Lp + (0, dy)), tuple(T + (0, dy)), tuple(R + (0, dy)), tuple(B + (0, dy))], 0.5))
    pins = cv.blank()
    for a_, b_ in ((Lp, B), (B, R)):
        for t in np.linspace(0.15, 0.85, 4):
            q = a_ + (b_ - a_) * t + (0, 1.6)
            pins |= cv.polygon([tuple(q + (-0.45, 0)), tuple(q + (0.45, 0)), tuple(q + (0.45, 1.5)), tuple(q + (-0.45, 1.5))])
    return [part(pins, "gold", "flat", h=0.75, tones=pin_t, glint=False),
            part(body, "metal", "flat", h=0.8, tones=edge_t),
            part(topm, "glossy", "bevel", bevel=0.7, h=0.9, tones=top_t, inner_line=False)]


def experience_nugget(cv, src, name):
    """A small glowing gem: wide bevels round a raised table."""
    outer = silhouette(src, cv, 0.5)
    table = cv2.erode(outer, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(cv.k * 4.0),) * 2))
    return [part(outer, "crystal", "bevel", bevel=1.8, h=0.86, emissive=0.35),
            part(table, "crystal", "bevel", bevel=0.6, h=0.92, emissive=0.45, inner_line=False)]


def polished_rose_quartz(cv, src, name):
    """Polished rose quartz: a smooth rounded block, pink-white top, mid-pink
    left face, deeper right face, bright edges where the faces meet."""
    top_t, left_t, right_t = pal(src, (6, 4), (5, 5), (9, 4)), pal(src, (6, 9), (4, 9), (3, 8)), \
        pal(src, (12, 8), (10, 9), (8, 8))
    edge_t = pal(src, (6, 11), (6, 9), (5, 8))
    top = cv.polygon(round_poly([(2.0, 5.2), (8.0, 2.0), (10.6, 2.6), (14.0, 5.4), (7.2, 8.0)], 0.9))
    left = cv.polygon(round_poly([(2.0, 5.2), (7.2, 8.0), (7.2, 13.4), (5.2, 14.0), (2.0, 11.2)], 0.9))
    right = cv.polygon(round_poly([(7.2, 8.0), (14.0, 5.4), (14.0, 11.0), (11.0, 13.0), (7.2, 13.4)], 0.9))
    return [part(right, "crystal", "bevel", bevel=0.7, h=0.82, tones=right_t, glint=False, inner_line=False),
            part(left, "crystal", "bevel", bevel=0.7, h=0.84, tones=left_t, glint=False, inner_line=False),
            part(top, "crystal", "bevel", bevel=0.8, h=0.9, tones=top_t, inner_line=False),
            part(cv.stroke((2.8, 5.8), (6.8, 7.9), 0.75), "crystal", "flat", h=0.92, inner_line=False, tones=edge_t),
            part(cv.stroke((7.2, 8.6), (7.2, 12.6), 0.75), "crystal", "flat", h=0.9, inner_line=False, tones=edge_t)]


def brass_hand(cv, src, name):
    """Mechanical brass hand pointing up and to the right: broad palm, jointed
    index finger, curled fingers, raised thumb, grey metal cuff at the wrist."""
    cuff = cv.polygon(round_poly([(1.4, 10.8), (4.0, 8.4), (6.8, 11.4), (4.2, 14.0)], 0.6))
    palm = cv.polygon(round_poly([(3.0, 8.6), (4.2, 5.4), (7.4, 4.4), (10.4, 5.6), (11.0, 9.0), (8.6, 11.8),
                                  (5.2, 11.6)], 1.2))
    fingers = [cv.stroke((10.0, 6.4), (12.4, 6.0), 2.0), cv.stroke((10.2, 8.4), (12.2, 8.3), 1.9),
               cv.stroke((9.6, 10.2), (11.2, 10.6), 1.7)]
    index = cv.stroke((9.2, 5.6), (12.2, 1.8), 2.2)
    thumb = cv.stroke((4.8, 7.6), (3.8, 4.6), 1.9)
    palm_r = _texels(src, lambda x, y: 3 <= x < 9 and 4 <= y < 10)
    finger_r = _texels(src, lambda x, y: x >= 9 and 5 <= y < 11)
    return ([part(cuff, "iron", "bevel", bevel=0.6, h=0.78, region=_texels(src, lambda x, y: x < 6 and y >= 9)),
             part(thumb, "gold", "cyl", axis=((4.8, 7.6), (3.8, 4.6)), h=0.84, region=palm_r),
             part(palm, "gold", "bevel", bevel=1.0, h=0.88, region=palm_r)] +
            [part(f, "gold", "cyl", axis=((10.0, 9.0), (10.0, 5.0)), h=0.86, region=finger_r) for f in fingers] +
            [part(index, "gold", "cyl", axis=((9.2, 5.6), (12.2, 1.8)), h=0.9,
                  region=_texels(src, lambda x, y: x >= 9 and y < 5))])


def whisk(cv, src, name):
    """Balloon whisk seen end-on: the wire loops make a tilted oval with an
    inner loop and the centre wire, a dark handle at the lower left."""
    def loop(c, rx, ry, ang, w):
        m = cv.blank()
        cv2.ellipse(m, tuple(int(v) for v in np.round(np.array(c) * cv.k)), (int(rx * cv.k), int(ry * cv.k)), ang, 0, 360,
                    255, thickness=max(1, int(round(w * cv.k))), lineType=cv2.LINE_8)
        return m
    outer = loop((8.4, 5.6), 5.5, 3.6, -38, 1.7)
    inner = loop((8.6, 5.5), 3.0, 1.7, -38, 1.3)
    axis = cv.stroke((4.4, 9.4), (12.4, 2.2), 1.0)
    back = cv.blank()                                         # far-side wires: the balloon's dark interior
    cv2.ellipse(back, tuple(int(v) for v in np.round(np.array((8.4, 5.6)) * cv.k)), (int(5.5 * cv.k), int(3.6 * cv.k)),
                -38, 0, 360, 255, thickness=-1)
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    far = (np.abs(((xx - yy * 0.8) % 1.5) - 0.75) < 0.2) & (back > 0)
    handle = cv.stroke((1.4, 14.2), (4.6, 10.6), 2.6)
    ferrule = cv.stroke((3.8, 11.4), (5.4, 9.6), 2.2)
    wire_r = _texels(src, lambda x, y: x >= 5 and y < 10) & _bright(src, 0.4)
    dark_r = _texels(src, lambda x, y: x >= 5 and y < 10) & ~_bright(src, 0.5) & (src[..., 3] > 0.5)
    return [part(back, "iron", "flat", h=0.74, region=dark_r, glint=False, overlay=[(far, 1.3, 0.02)]),
            part(axis, "iron", "flat", h=0.8, region=wire_r, inner_line=False),
            part(inner, "iron", "bevel", bevel=0.35, h=0.84, region=wire_r, inner_line=False),
            part(outer, "iron", "bevel", bevel=0.4, h=0.88, region=wire_r, inner_line=False),
            part(handle, "dark_metal", "cyl", axis=((1.4, 14.2), (4.6, 10.6)), h=0.8,
                 region=_texels(src, lambda x, y: x < 5 and y >= 10)),
            part(ferrule, "iron", "cyl", axis=((3.8, 11.4), (5.4, 9.6)), h=0.84, region=wire_r)]


def chocolate_bar(cv, src, name):
    """A bar of chocolate: the fitted slab, its top moulded into segments."""
    parts = plate(cv, src, name)
    for p in parts:
        p["mat"] = "glossy"
    L, T, R, B = (np.array(p) for p in _extremes(src))
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    m = np.array([[R[0] - L[0], B[0] - T[0]], [R[1] - L[1], B[1] - T[1]]])
    inv = np.linalg.inv(m)
    c = (L + R) / 2
    dx, dy = xx - c[0], yy - c[1]
    uu = inv[0, 0] * dx + inv[0, 1] * dy                          # along the bar
    vv = inv[1, 0] * dx + inv[1, 1] * dy                          # across it
    grooves = (np.abs(((uu + 0.5) * 5) % 1 - 0.5) > 0.42) | (np.abs(vv) < 0.04)
    parts[1]["overlay"] = [(grooves, 0.72, -0.03)]
    return parts


def natural(cv, src, name):
    """Natural things (crushed ore, powders, raw crystals, food): the
    original's outline traced smoothly, its colour pattern kept crisp."""
    m = "powder" if re.search(r"flour|powder|cinder", name) else "stone"
    return [part(silhouette(src, cv, 0.45), m, "post", bevel=0.8, h=0.85)]


SHEETS = r"(iron|golden|brass|sturdy|lapis|bronze|steel|electrum|zinc)_sheet"
DESIGNS = [
    (re.compile(rf"^(create|alloyed|createaddition):{SHEETS}$"), plate),
    (re.compile(r"^create:copper_sheet$"), curled_sheet),
    (re.compile(r"^create:(andesite_alloy|shadow_steel|refined_radiance)$"), alloy),
    (re.compile(r"^create:crushed_raw_\w+$"), natural),
    (re.compile(r"^create:(incomplete_)?precision_mechanism$"), precision_mechanism),
    (re.compile(r"^create:electron_tube$"), electron_tube),
    (re.compile(r"^create:propeller$"), propeller),
    (re.compile(r"^create:goggles$"), goggles),
    (re.compile(r"^create:super_glue$"), super_glue),
    (re.compile(r"^create:(attribute_)?filter$"), filter_item),
    (re.compile(r"^create:(empty_)?schematic$"), schematic),
    (re.compile(r"^create:(red_)?sand_paper$|^create:cardboard$"), paper_sheet),
    (re.compile(r"^create:belt_connector$"), belt_connector),
    (re.compile(r"^create:integrated_circuit$"), integrated_circuit),
    (re.compile(r"^create:experience_nugget$"), experience_nugget),
    (re.compile(r"^create:polished_rose_quartz$"), polished_rose_quartz),
    (re.compile(r"^create:brass_hand$"), natural),
    (re.compile(r"^create:whisk$"), whisk),
    (re.compile(r"^create:bar_of_chocolate$"), chocolate_bar),
    (re.compile(r"^create:(rose_quartz|dough|sweet_roll|blaze_cake|honeyed_apple|chocolate_glazed_berries|"
                r"tree_fertilizer|wheat_flour)$"), natural),
    (re.compile(r"^(create:(cinder_flour|wheat_flour|powdered_obsidian)|createfood:corn_flour)$"), natural),
]


def create_item_of(path, ns):
    if not path.startswith("textures/item/") or path.count("/") != 2:
        return None  # sub-folders hold model textures (e.g. item/package/), not icons
    key = f"{ns}:{path.split('/')[-1][:-4]}"
    return next((fn for pat, fn in DESIGNS if pat.match(key)), None)


THIN = {"whisk": 0.5}   # wire designs: a full outline would swallow the wires


def render(path, src, size, ns="create"):
    name = path.split("/")[-1][:-4]
    cv = Canvas(size)
    return render_item(create_item_of(path, ns)(cv, src, name), cv, src, size, outline=THIN.get(name, 0.75))
