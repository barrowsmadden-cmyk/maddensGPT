"""Hand-made buckets: one HD iron pail, filled per texture.

Every fluid bucket (vanilla and the ~170 mod buckets) is the vanilla pail with
its contents painted on top. The contents are found by diffing the source
against the vanilla empty bucket: inside the opening they become a fluid
surface, below the rim they become drips running down the pail, each in the
contents' own colours. Buckets holding a creature (pixels outside the pail's
silhouette) or drawn on a different pail are left to the automatic path.
"""
import math
import re

import cv2
import numpy as np

from texgen import Material, derive, IRON, DIELECTRIC
from handmade.tools import Canvas
from handmade.items import palette_from, _outline, LUMW

REF = None  # vanilla empty bucket (16x16 RGBA float), set by prepare()
IRON_PAL = ((236, 236, 238), (182, 182, 188), (84, 84, 92))
RIM_C, RIM_RX, RIM_RY = (8.0, 3.4), 5.4, 2.3
OPEN_RX, OPEN_RY = 4.5, 1.6
GLOW = re.compile(r"lava|molten|blaze|magma|glow|radiance|experience|essence")


def prepare(assets_root):
    global REF
    from families import load_rgba
    REF = load_rgba(f"{assets_root}/minecraft/textures/item/bucket.png")


def _contents(src):
    """Mask of pixels that differ from the empty bucket."""
    diff = np.abs(src - REF)[..., :3].sum(-1) > 40 / 255
    return diff & (src[..., 3] > 0.5)


def bucket_of(path, ns, src):
    name = path.split("/")[-1][:-4]
    if not path.startswith("textures/item/") or path.count("/") != 2 or not name.endswith("bucket"):
        return False
    if src.shape[:2] != (16, 16) or REF is None:
        return False
    if name == "bucket" and ns == "minecraft":
        return True
    outside = (src[..., 3] > 0.5) & (REF[..., 3] < 0.5)
    if outside.sum() > 1:                        # a creature or a differently shaped pail
        return False
    body = REF[..., 3] > 0.5
    lower = body.copy()
    lower[:10] = False
    if (_contents(src) & lower).sum() > 0.5 * lower.sum():
        return False                             # not drawn on the vanilla pail
    return _contents(src).any()


def _pail(cv):
    """The iron pail: tapered body with two raised hoops and a shadow under a
    thick rolled rim; dark opening with the far wall catching light."""
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    top, bot = 4.0, 13.4
    t = np.clip((yy - top) / (bot - top), 0, 1)
    half = 5.1 - 1.5 * t                                     # ~3..13 at the top, ~5..11 at the bottom
    body = (np.abs(xx - 8.0) < half) & (yy > top) & (yy < bot + 0.7 * np.sqrt(np.clip(1 - ((xx - 8) / 3.2) ** 2, 0, 1)))
    u = (xx - 8.0) / np.maximum(half, 1e-3)                  # -1..1 across the body
    light, mid, dark = (np.array(c, float) / 255 for c in IRON_PAL)
    shade = np.clip(0.5 + 0.5 * np.cos((u + 0.3) * 1.5), 0, 1)   # cylinder lit from the left
    col = dark + (mid - dark) * (0.25 + 0.85 * shade)[..., None]
    col = col + (light - mid) * (np.exp(-((u + 0.5) / 0.16) ** 2))[..., None]
    curve = 0.3 * (1 - u ** 2)                               # bands follow the cylinder's curvature
    hoops = np.zeros_like(xx)
    groove = np.zeros_like(xx)
    for y0 in (7.0, 12.0):
        d = yy - y0 - curve
        hoops = np.maximum(hoops, np.exp(-(d / 0.42) ** 2))
        groove = np.maximum(groove, np.exp(-((d - 0.62) / 0.16) ** 2))
    under_rim = np.exp(-np.clip(yy - (RIM_C[1] + RIM_RY) - curve, 0, None) / 0.7) * (yy > RIM_C[1])
    col = col * (1 + 0.16 * hoops - 0.35 * groove - 0.4 * under_rim)[..., None]
    height = 0.7 + 0.16 * np.sqrt(np.clip(1 - u ** 2, 0, 1)) + 0.07 * hoops - 0.04 * groove
    rim_outer = ((xx - RIM_C[0]) / RIM_RX) ** 2 + ((yy - RIM_C[1]) / RIM_RY) ** 2 < 1
    ri = ((xx - RIM_C[0]) / (RIM_RX - 1.0)) ** 2 + ((yy - RIM_C[1]) / (RIM_RY - 0.75)) ** 2
    rim_inner = ri < 1
    rim = rim_outer & ~rim_inner
    # rolled lip: a tube, bright along its crest, dark underneath
    ro = ((xx - RIM_C[0]) / RIM_RX) ** 2 + ((yy - RIM_C[1]) / RIM_RY) ** 2
    across = np.clip((ro - ri * ((RIM_RX - 1.0) / RIM_RX) ** 2) / (1 - ((RIM_RX - 1.0) / RIM_RX) ** 2), 0, 1)
    tube = np.sin(math.pi * np.clip((np.sqrt(ro) - 0.78) / 0.22, 0, 1))
    ang = np.arctan2((yy - RIM_C[1]) / RIM_RY, (xx - RIM_C[0]) / RIM_RX)
    rim_lit = np.clip(0.35 + 0.45 * tube + 0.3 * np.cos(ang + 2.0), 0, 1.15)
    rim_col = dark + (light - dark) * rim_lit[..., None]
    back = np.clip((yy - (RIM_C[1] - RIM_RY)) / (2 * RIM_RY), 0, 1)
    open_col = dark * (0.35 + 0.75 * (1 - back))[..., None]  # far wall lit, near side in shadow
    col = np.where(rim[..., None], rim_col, col)
    col = np.where(rim_inner[..., None], open_col, col)
    height = np.where(rim, 0.9 + 0.09 * tube, np.where(rim_inner, 0.6, height))
    cover = body | rim_outer
    return np.clip(col, 0, 1), height, cover, rim, rim_inner


def _fluid(cv, src, cont, col, height, cover, name):
    """Paint the contents: a surface in the opening coloured from the source's
    own content pixels, a sheet over the rim where it spills, and drips (one
    per run of adjacent source columns) down the pail."""
    from families import rank_map
    top_rows = cont.copy()
    top_rows[6:] = False
    px = src[top_rows if top_rows.sum() >= 4 else cont][:, :3]
    light, mid, dark = (np.array(c, float) / 255 for c in palette_from(src[cont].reshape(-1, 1, 4)))
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    surf = ((xx - RIM_C[0]) / (RIM_RX - 0.9)) ** 2 + ((yy - RIM_C[1] - 0.15) / (RIM_RY - 0.65)) ** 2 < 1
    rng = np.random.default_rng(sum(map(ord, name)))
    blot = cv2.GaussianBlur(rng.random((cv.n, cv.n)).astype(np.float32), (0, 0), cv.k * 0.9)
    field = (blot - blot.mean()) / (blot.std() + 1e-6) + 0.8 * np.exp(-(((xx - 6.2) / 1.8) ** 2 + ((yy - 2.8) / 0.6) ** 2))
    s_col = rank_map(field, px, surf)
    calm = 0.35 if re.search(r"lava|molten|magma", name) else 0.65   # molten fluids keep their blotches
    s_col = s_col * (1 - calm) + (mid * 0.7 + light * 0.3)[None, None, :] * calm
    # spill: each source column's contents below the rim hang down to that
    # column's length; neighbouring columns join down to the shorter of the two
    # (a sheet with drips hanging off it); drips end in rounded beads
    e = np.zeros(16)
    for x in range(16):
        rows = np.nonzero(cont[:, x])[0]
        rows = rows[rows >= 6]
        e[x] = rows.max() + 1.0 if len(rows) else 0.0
    drips = np.zeros(xx.shape, bool)
    for x in range(16):
        if not e[x]:
            continue
        cx0 = x + 0.5
        w = 0.36 + 0.06 * (e[x] - 6)
        drips |= (np.abs(xx - cx0) < min(w, 0.48)) & (yy >= 5.0) & (yy < e[x] - 0.45)
        drips |= np.hypot(xx - cx0, (yy - (e[x] - 0.5)) * 1.1) < min(w, 0.48) + 0.12
        if x + 1 < 16 and e[x + 1]:
            join = min(e[x], e[x + 1]) - 0.6
            drips |= (xx >= cx0) & (xx <= cx0 + 1) & (yy >= 5.0) & (yy < join)
    # melt the per-column shapes into one fluid outline: no stair steps
    drips = cv2.GaussianBlur(drips.astype(np.float32), (0, 0), cv.k * 0.35) > 0.5
    lip = rim_band(xx, yy) & (yy > RIM_C[1]) & spill_any(cont, xx)
    inside = cv2.erode(cover.astype(np.uint8), np.ones((int(cv.k * 1.1),) * 2, np.uint8)) > 0
    over = (lip | drips) & ~surf & inside                   # clear of the outline: no height cliff there
    d_shade = np.clip(0.8 + 0.3 * np.cos((xx - 6.0) / 4), 0.55, 1.1)
    o_col = (mid * 0.6 + light * 0.4) * d_shade[..., None]
    edge = cv2.distanceTransform(over.astype(np.uint8), cv2.DIST_L2, 3) / cv.k < 0.16
    o_col = np.where(edge[..., None], dark, o_col)
    col = np.where(surf[..., None], s_col, col)
    col = np.where(over[..., None], o_col, col)
    height = np.where(surf, 0.8 + 0.01 * field, height)
    soft = cv2.GaussianBlur(height.astype(np.float32), (0, 0), cv.k * 0.7)   # fluid smooths over the hoops
    height = np.where(over, soft + 0.05, height)
    return np.clip(col, 0, 1), height, surf | over


def rim_band(xx, yy):
    out = ((xx - RIM_C[0]) / (RIM_RX + 0.2)) ** 2 + ((yy - RIM_C[1]) / (RIM_RY + 0.2)) ** 2 < 1
    return out


def spill_any(cont, xx):
    """Columns (in canvas units) where the source's contents reach the rim front (row 5)."""
    cols = np.nonzero(cont[5])[0]
    m = np.zeros(xx.shape, bool)
    for x in cols:
        m |= np.abs(xx - (x + 0.5)) <= 0.6
    return m


def render(path, src, size, ns="minecraft"):
    name = path.split("/")[-1][:-4]
    cv = Canvas(size)
    col, height, cover, rim, opening = _pail(cv)
    f0 = np.full(col.shape[:2], float(IRON))
    smooth = np.full(col.shape[:2], 0.62)
    emis = np.zeros(col.shape[:2])
    cont = _contents(src)
    if cont.any() and not (name == "bucket" and ns == "minecraft"):
        col, height, fl = _fluid(cv, src, cont, col, height, cover, name)
        f0 = np.where(fl, DIELECTRIC, f0)
        smooth = np.where(fl, 0.88, smooth)
        if GLOW.search(name):
            emis = np.where(fl, 0.8, 0.0)
    rimm, rim_col = _outline(cover, cv, IRON_PAL[2], width=0.45)
    col = np.where(rimm[..., None], rim_col[None, None, :], col)
    height = np.where(rimm, height - 0.05, height)
    down = lambda a: cv2.resize(a.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    alpha = down(cover)
    w = np.maximum(alpha, 1e-4)
    rgb = np.dstack([down(col[..., i] * cover) for i in range(3)]) / w[..., None]
    m = Material(size)
    solid = alpha > 0.5
    m.albedo, m.alpha = np.clip(rgb, 0, 1), alpha
    h = down(height * cover) / w
    m.height = np.where(solid, np.clip(h + (0.99 - h[solid].max()), 0, 0.99), 0.0)
    m.f0 = np.where(solid, cv2.resize(f0.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST), DIELECTRIC)
    m.smooth = np.where(solid, down(smooth * cover) / w, 0.0)
    m.emissive = np.where(solid, down(emis * cover) / w, 0.0)
    return derive(m, edge_wear=0.0, edges="reflect", normal_strength=4.0)
