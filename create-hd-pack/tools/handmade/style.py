"""House style for hand-made items, tuned for how items look in the game.

The rules come from reading items at inventory size and in hand (see
ingame_preview.py and fidelity.py), and from the high-res style the pack
targets: chunky silhouettes that fill the original's footprint, a bold outline
in each part's own darkest tone, crisp cel/bevel shading (a flat face, a light
bevel on edges facing the light, a dark bevel on edges facing away, a small
glint), colours taken from the original's own pixels under each part, and hard
alpha edges (the game extrudes every non-transparent texel in hand, so soft
edges become halos).

A design returns a list of parts (painted in order, later on top):

    part(mask, mat="iron", shade="bevel", bevel=1.0, ...)

render_item() samples each part's tones from the original texture, shades,
outlines, downsamples with binary alpha and derives the PBR maps.
"""
import math

import cv2
import numpy as np

from texgen import Material, derive, IRON, GOLD, COPPER, ALBEDO_METAL, DIELECTRIC

LIGHT = np.array([-1.0, -1.0]) / math.sqrt(2)     # light from the top left (screen x right, y down)
LUMW = np.array([0.2126, 0.7152, 0.0722])

# LabPBR per material: (_s green: F0 or metal id, smoothness, porosity/SSS)
MATS = {
    "iron": (IRON, 0.68, 0), "gold": (GOLD, 0.78, 0), "copper": (COPPER, 0.66, 0),
    "metal": (ALBEDO_METAL, 0.64, 0), "dark_metal": (ALBEDO_METAL, 0.5, 0),
    "gem": (44, 0.9, 0), "crystal": (40, 0.86, 0), "glass": (12, 0.94, 0),
    "wood": (DIELECTRIC, 0.28, 30), "stone": (DIELECTRIC, 0.22, 25), "cloth": (DIELECTRIC, 0.12, 40),
    "leather": (DIELECTRIC, 0.3, 20), "paper": (DIELECTRIC, 0.18, 40), "organic": (DIELECTRIC, 0.35, 70),
    "glossy": (DIELECTRIC, 0.75, 10), "rubber": (DIELECTRIC, 0.32, 0), "powder": (DIELECTRIC, 0.1, 60),
}


def round_poly(pts, r=0.8, steps=6):
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


def part(mask, mat="iron", shade="bevel", bevel=1.0, tones=None, h=0.8, axis=None, glint=True,
         grain=0.0, emissive=0.0, inner_line=True, region=None, overlay=(), form=True, texture=True):
    """One shape of the design.

    mask      supersampled uint8/bool mask (Canvas.polygon etc.)
    shade     "bevel" (flat face, lit/dark bevels), "cyl" (round bar along `axis`),
              "flat" (single tone + outline)
    bevel     bevel width in vanilla px
    tones     None = sample from the original under this part; or (light, face, dark)
              sRGB 0-255 (outline and glint derived)
    region    optional original-resolution mask to sample tones from instead
    overlay   [(mask, factor, dh)]: patterns that brighten/darken the shaded part
              (e.g. chain links), keeping its light and shadow
    grain     strength of wood-like streaks along `axis` (0 = none)
    """
    return dict(mask=mask > 0, mat=mat, shade=shade, bevel=bevel, tones=tones, h=h, axis=axis,
                glint=glint, grain=grain, emissive=emissive, inner_line=inner_line, region=region,
                overlay=overlay, form=form, texture=texture)


def _lab(rgb):
    return cv2.cvtColor(np.asarray(rgb, np.float32).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)


def tones_from(pixels):
    """(outline, dark, face, light, glint) from original pixels (N x 3, 0-1),
    each an actual-colour average of a lightness band, so hue and saturation
    stay the artist's."""
    px = np.asarray(pixels, float)
    if len(px) == 0:
        px = np.array([[0.6, 0.6, 0.6]])
    L = _lab(px)[:, 0]
    order = np.argsort(L)
    px, L = px[order], L[order]
    n = len(px)
    band = lambda a, b: px[int(a * (n - 1)):max(int(a * (n - 1)) + 1, int(math.ceil(b * (n - 1))) + 1)].mean(0)
    dark, face, light = band(0.12, 0.35), band(0.4, 0.65), band(0.75, 0.92)
    glint = px[-max(1, n // 20):].mean(0)
    outline = band(0.0, 0.08)
    # outline: the part's darkest tone, pushed a little darker but never black-grey
    outline = outline * 0.8 if _lab(outline)[0, 0] > 20 else outline
    return outline, dark, face, light, glint


def _darkest(px):
    L = _lab(px)[:, 0]
    return px[L <= np.percentile(L, 30)].mean(0)


def _lightest(px, keep):
    """Thin parts are mostly outline texels: keep only their lighter share."""
    if len(px) < 3:
        return px
    L = _lab(px)[:, 0]
    return px[L >= np.percentile(L, 100 * (1 - keep))]


def tones_given(light, face, dark):
    light, face, dark = (np.array(c, float) / 255 for c in (light, face, dark))
    return dark * 0.55, dark, face, light, np.minimum(light * 1.12 + 0.05, 1.0)


def _labfield(src, cv):
    """The original's colour (Lab), upscaled smoothly to the canvas."""
    lab = cv2.cvtColor(src[..., :3].astype(np.float32), cv2.COLOR_RGB2LAB)
    a = src[..., 3] > 0.5
    filled = lab.copy()
    for _ in range(4):
        if a.all():
            break
        k = np.ones((3, 3), np.float32)
        c_ = cv2.filter2D(a.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
        grow = ~a & (c_ > 0)
        for ch in range(3):
            s_ = cv2.filter2D(np.where(a, filled[..., ch], 0).astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
            filled[..., ch][grow] = s_[grow] / c_[grow]
        a = a | grow
    up = cv2.resize(filled, (cv.n, cv.n), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(up, (0, 0), cv.k * 0.3)


def reg_post(p, vis, src):
    if p["region"] is not None:
        return p["region"]
    h, w = src.shape[:2]
    return cv2.resize(vis.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.4


def pix_post(src, reg):
    return src[..., :3][reg & (src[..., 3] > 0.5)]


def _clusters(px, k=6):
    """k colour clusters (Lab) of the original's pixels, or fewer if it has fewer colours."""
    lab = _lab(px)
    uniq = np.unique(np.round(lab, 1), axis=0)
    k = max(1, min(k, len(uniq)))
    if k == len(uniq):
        return uniq.astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, _, centres = cv2.kmeans(lab.astype(np.float32), k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    return centres


def traced(src, cv, eps=0.55, alpha=None):
    """The original's silhouette as straight-edged polygons (stair-stepped
    diagonals become straight lines), for manufactured shapes."""
    a = (src[..., 3] if alpha is None else alpha) > 0.5
    h, w = a.shape
    up = 8
    big = cv2.resize(a.astype(np.uint8) * 255, (w * up, h * up), interpolation=cv2.INTER_NEAREST)
    contours, hier = cv2.findContours(big, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    m = cv.blank()
    unit = 16.0 / w / up
    for i, c in enumerate(contours):
        poly = cv2.approxPolyDP(c, eps * up * (w / 16.0), True).reshape(-1, 2) * unit
        hole = hier is not None and hier[0][i][3] >= 0
        cv2.fillPoly(m, [np.round(poly * cv.k).astype(np.int32)], 0 if hole else 255)
    return m


def edge_band(mask, cv, width, facing, cos_min=0.45):
    """Strip of `mask` within `width` vanilla px of an edge whose outward normal
    points along `facing` (screen x right, y down): a plate's thickness, a rim light."""
    m = mask > 0
    d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5) / cv.k
    ds = cv2.GaussianBlur(d.astype(np.float32), (0, 0), cv.k * 0.5)
    gy, gx = np.gradient(ds)
    gn = np.hypot(gx, gy) + 1e-9
    f = np.array(facing, float) / np.linalg.norm(facing)
    cosv = (-gx * f[0] - gy * f[1]) / gn
    return ((d < width) & m & (cosv > cos_min)).astype(np.uint8) * 255


def silhouette(src, cv, smooth=0.45):
    """The original's silhouette, traced smoothly (for organic shapes: lumps, nuggets, food)."""
    a = cv2.resize(src[..., 3].astype(np.float32), (cv.n, cv.n), interpolation=cv2.INTER_NEAREST)
    a = cv2.GaussianBlur(a, (0, 0), cv.k * smooth)
    return (a > 0.5).astype(np.uint8) * 255


def _shade(p, cv, tones):
    """Colour and height of one part with crisp bands."""
    outline, dark, face, light, glint = tones
    m = p["mask"]
    d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5) / cv.k          # vanilla px from the edge
    ds = cv2.GaussianBlur(d.astype(np.float32), (0, 0), cv.k * 0.6)
    gy, gx = np.gradient(ds)
    gn = np.hypot(gx, gy) + 1e-9
    facing = (-gx * LIGHT[0] - gy * LIGHT[1]) / gn             # +1: edge faces the light
    col = np.broadcast_to(face, m.shape + (3,)).copy()
    height = np.full(m.shape, p["h"])
    if p["shade"] == "bevel":
        w = p["bevel"]
        band = (d < w) & m
        # form shadow: on large parts the side away from the light drops a step,
        # a crisp core-shadow band as in cel shading (big flat faces look like plastic)
        if p.get("form", True) and m.sum() / cv.k ** 2 > 18:
            yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
            proj = xx * LIGHT[0] + yy * LIGHT[1]
            cut = np.percentile(proj[m], 32)
            # a gently curved terminator, not a ruler line
            wob = 0.35 * np.sin((xx * LIGHT[1] - yy * LIGHT[0]) * 0.9)
            col[m & (proj + wob < cut)] = face * 0.55 + dark * 0.45
        col[band & (facing > 0.3)] = light
        col[band & (facing < -0.3)] = dark
        height = p["h"] - 0.12 * np.clip(1 - d / max(w, 1e-3), 0, 1) ** 1.5
        if p["glint"]:
            yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
            proj = xx * LIGHT[0] + yy * LIGHT[1]
            top = np.percentile(proj[m], 93) if m.any() else 0
            col[band & (facing > 0.5) & (proj >= top)] = glint
    elif p["shade"] == "post":
        # the original's own colour pattern, upscaled smoothly and snapped to the
        # original's colour clusters: crisp bands, hue variety kept (patina, specks)
        labf, centres = p["_lab"], p["_centres"]
        dist = ((labf[:, :, None, :] - centres[None, None, :, :]) ** 2 * np.array([1.0, 1.6, 1.6])).sum(-1)
        idx = np.argmin(dist, -1).astype(np.uint8)
        idx = cv2.medianBlur(idx, max(3, int(cv.k * 0.5) | 1))
        rgbc = cv2.cvtColor(centres.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_LAB2RGB).reshape(-1, 3)
        col = np.clip(rgbc[idx], 0, 1)
        Lf = labf[..., 0]
        height = p["h"] - 0.1 + 0.1 * np.clip((Lf - Lf[m].min()) / max(np.ptp(Lf[m]), 1e-3), 0, 1) if m.any() else height
        w = p["bevel"]
        height = height - 0.08 * np.clip(1 - d / max(w, 1e-3), 0, 1) ** 1.5
    elif p["shade"] == "cyl":
        (ax, ay), (bx, by) = p["axis"]
        ln = math.hypot(bx - ax, by - ay)
        nx, ny = -(by - ay) / ln, (bx - ax) / ln
        if nx * LIGHT[0] + ny * LIGHT[1] < 0:                  # make +across the lit side
            nx, ny = -nx, -ny
        yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
        across = (xx - ax) * nx + (yy - ay) * ny
        along = ((xx - ax) * (bx - ax) + (yy - ay) * (by - ay)) / ln
        half = max(np.percentile(np.abs(across[m]), 99), 1e-3) if m.any() else 1.0
        t = across / half
        col[m & (t > 0.3)] = light
        col[m & (t < -0.4)] = dark
        if p["glint"]:
            col[m & (t > 0.55) & (t < 0.8)] = glint
        height = p["h"] - 0.12 * np.clip(np.abs(t), 0, 1) ** 2
        if p["grain"]:
            rng = np.random.default_rng(int(ax * 100 + by * 10))
            g = np.zeros_like(xx)
            for f, a in ((1.7, 0.6), (3.3, 0.4)):
                g += a * np.sin(across * f * math.pi + rng.uniform(0, 6) + np.sin(along * 0.8 + rng.uniform(0, 6)) * 0.7)
            streak = m & (g > 0.72) & (np.abs(t) < 0.85)
            col[streak] = col[streak] * (1 - 0.18 * p["grain"])
    return col, height


def render_item(parts, cv, src, size, outline=0.75):
    """Paint the parts, outline them, and return the LabPBR maps (albedo, _n, _s)."""
    n = cv.n
    owner = np.full((n, n), -1, int)
    for i, p in enumerate(parts):
        owner[p["mask"]] = i
    cover = owner >= 0
    # tones: each part samples the original pixels it covers at the original's resolution
    sh, sw = src.shape[:2]
    # the original's outline texels (opaque, touching transparency): the HD draws its
    # own outline, so the faces must not be sampled from these dark pixels
    opaque = src[..., 3] > 0.5
    pad = np.pad(opaque, 1)
    interior = opaque & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    col = np.zeros((n, n, 3))
    height = np.zeros((n, n))
    out_col = np.zeros((n, n, 3))
    f0 = np.full((n, n), float(DIELECTRIC))
    sm = np.zeros((n, n))
    em = np.zeros((n, n))
    por = np.zeros((n, n))
    for i, p in enumerate(parts):
        vis = owner == i
        if not vis.any():
            continue
        if p["tones"] is not None:
            tones = tones_given(*p["tones"])
        else:
            reg = p["region"]
            if reg is None:
                cov = cv2.resize(vis.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
                reg = cov > 0.4
            body = reg & interior
            pix = src[..., :3][body] if body.sum() >= 3 else _lightest(src[..., :3][reg & opaque], 0.7)
            tones = tones_from(pix)
            edge_px = src[..., :3][reg & opaque & ~interior]
            if len(edge_px):                                  # the original's own outline colour
                tones = (_darkest(edge_px),) + tones[1:]
        if p["shade"] == "post":
            p = dict(p, _lab=_labfield(src, cv), _centres=_clusters(pix_post(src, reg_post(p, vis, src)), k=p.get("k", 6)))
        c, h = _shade(p, cv, tones)
        for om, factor, dh in p["overlay"]:
            om = om > 0
            c[om] = np.clip(c[om] * factor, 0, 1)
            h[om] = h[om] + dh
        col[vis], height[vis], out_col[vis] = c[vis], h[vis], tones[0]
        f0[vis], sm[vis], por[vis] = MATS[p["mat"]][0], MATS[p["mat"]][1], MATS[p["mat"]][2]
        em[vis] = p["emissive"] if np.ndim(p["emissive"]) == 0 else p["emissive"][vis]
    # material surface: faint brushed streaks on metal, speckle on stone, grain on wood,
    # fibre on cloth/paper; kept subtle so the inventory read stays clean
    rng = np.random.default_rng(7)
    def field(sx, sy):
        f = cv2.GaussianBlur(rng.standard_normal((n, n)).astype(np.float32), (0, 0), cv.k * sx, sigmaY=cv.k * sy)
        return f / (f.std() + 1e-6)
    tex = {"brushed": field(3.0, 0.06), "speckle": field(0.18, 0.18), "fibre": field(0.5, 0.08)}
    amount = {"iron": ("brushed", 0.03), "gold": ("brushed", 0.02), "copper": ("brushed", 0.03),
              "metal": ("brushed", 0.03), "dark_metal": ("brushed", 0.045), "stone": ("speckle", 0.07),
              "wood": ("fibre", 0.06), "cloth": ("speckle", 0.05), "paper": ("fibre", 0.04),
              "leather": ("speckle", 0.05), "organic": ("speckle", 0.05), "powder": ("speckle", 0.09),
              "rubber": ("speckle", 0.04)}
    for i, p in enumerate(parts):
        kind = amount.get(p["mat"])
        if kind is None or p.get("texture") is False:
            continue
        vis = owner == i
        col[vis] = col[vis] * (1 + kind[1] * tex[kind[0]][vis])[:, None]
        height[vis] = height[vis] + 0.01 * tex[kind[0]][vis]
    # bold outline round the silhouette, and a thinner one where a part overlaps another
    d = cv2.distanceTransform(cover.astype(np.uint8), cv2.DIST_L2, 5) / cv.k
    rim = cover & (d < outline)
    inner = np.zeros((n, n), bool)
    for i, p in enumerate(parts):
        if not p["inner_line"]:
            continue
        vis = owner == i
        below = cover & (owner < i) & (owner >= 0)
        if vis.any() and below.any():
            near = cv2.dilate(vis.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(cv.k * 0.9),) * 2)) > 0
            inner |= near & below
    col = np.where((rim | inner)[..., None], out_col, col)
    height = np.where(rim, height - 0.08, np.where(inner, height - 0.05, height))
    # downsample: colour averaged over covered samples, alpha hard (no halos when extruded)
    down = lambda a: cv2.resize(a.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    cov = down(cover)
    solid = cov >= 0.5
    w = np.maximum(cov, 1e-4)
    rgb = np.dstack([down(col[..., i] * cover) for i in range(3)]) / w[..., None]
    h = down(height * cover) / w
    m = Material(size)
    m.albedo = np.clip(rgb, 0, 1)
    m.alpha = solid.astype(float)
    m.height = np.where(solid, np.clip(h + (0.99 - h[solid].max()), 0.02, 0.99), 0.0)
    nearest = lambda a: cv2.resize(a.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST)
    m.f0 = np.where(solid, nearest(f0), DIELECTRIC)
    m.smooth = np.where(solid, down(sm * cover) / w, 0.0)
    m.porosity = np.where(solid, nearest(por), 0.0)
    m.emissive = np.where(solid, down(em * cover) / w, 0.0)
    return derive(m, edge_wear=0.0, edges="reflect", normal_strength=3.0)
