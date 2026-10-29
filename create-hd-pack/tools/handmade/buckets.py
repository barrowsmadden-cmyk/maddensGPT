"""Hand-made buckets: one HD iron pail, filled per texture.

House style (handmade/style.py). The pail is traced from the vanilla bucket
(texel (x, y) spans x..x+1, y..y+1): a near-straight cylinder lit from the
left with a reflected rim light on the right, a shadow line under a thick
rim, a dark opening whose far wall catches the light. Its tones are the
vanilla bucket's own texels.

Every fluid bucket (vanilla and the ~170 mod buckets) is that pail with its
contents painted on: the contents are the texels that differ from the empty
bucket; inside the opening they become the fluid surface, below the rim they
become smooth spills and drips, both coloured from the contents' own texels
(posterized, so jam keeps its fruit, lava its blotches). Buckets holding a
creature, or drawn on a different pail, are left to the automatic path.
"""
import re

import cv2
import numpy as np

from handmade.tools import Canvas
from handmade.style import part, render_item, round_poly

REF = None  # vanilla empty bucket (16x16 RGBA float), set by prepare()
GLOW = re.compile(r"lava|molten|blaze|magma|glow|radiance|experience|essence")
C, RX, RY = (8.0, 3.6), 6.0, 2.6            # rim ellipse
OX, OY = 4.9, 1.8                           # opening ellipse


def prepare(assets_root):
    global REF
    from families import load_rgba
    REF = load_rgba(f"{assets_root}/minecraft/textures/item/bucket.png")


def _contents(src):
    """Texels that differ from the empty bucket."""
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


def _tex(img, x, y):
    return tuple(np.round(img[y, x, :3] * 255))


def _ellipse(cv, c, rx, ry):
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    return ((((xx - c[0]) / rx) ** 2 + ((yy - c[1]) / ry) ** 2) < 1)


def _spill(cv, cont, surf):
    """Smooth spills: each source column's contents below the rim hang to that
    column's length, neighbours join down to the shorter one, ends round."""
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    e = np.zeros(16)
    for x in range(16):
        rows = np.nonzero(cont[:, x])[0]
        rows = rows[rows >= 5]
        e[x] = rows.max() + 1.0 if len(rows) else 0.0
    m = np.zeros(xx.shape, bool)
    for x in range(16):
        if not e[x]:
            continue
        cx, w = x + 0.5, min(0.36 + 0.06 * (e[x] - 6), 0.48)
        m |= (np.abs(xx - cx) < w) & (yy >= 4.0) & (yy < e[x] - 0.45)
        m |= np.hypot(xx - cx, (yy - (e[x] - 0.5)) * 1.1) < w + 0.12
        if x + 1 < 16 and e[x + 1]:
            m |= (xx >= cx) & (xx <= cx + 1) & (yy >= 4.0) & (yy < min(e[x], e[x + 1]) - 0.6)
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), cv.k * 0.35) > 0.5
    return m & ~surf


def render(path, src, size, ns="minecraft"):
    name = path.split("/")[-1][:-4]
    cv = Canvas(size)
    yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
    body = cv.polygon(round_poly([(2.0, 3.6), (14.0, 3.6), (14.0, 10.8), (13.2, 12.6), (12.0, 14.2), (11.0, 15.0),
                                  (5.0, 15.0), (4.0, 14.2), (2.8, 12.6), (2.0, 10.8)], 0.6))
    rim = _ellipse(cv, C, RX, RY)
    opening = _ellipse(cv, (C[0], C[1] + 0.2), OX, OY)
    body_t = (_tex(REF, 5, 8), _tex(REF, 8, 8), _tex(REF, 10, 8))
    rim_t = (_tex(REF, 6, 2), _tex(REF, 4, 3), _tex(REF, 5, 5))
    open_t = (_tex(REF, 6, 2), _tex(REF, 8, 4), _tex(REF, 5, 4))
    under = (yy > C[1] + RY - 0.3) & (yy < C[1] + RY + 0.9)          # shadow line under the rim
    rim_light = np.abs(xx - 12.9) < 0.45                              # reflected light at the right edge
    far_wall = yy < C[1] - 0.2
    parts = [part(body, "iron", "cyl", axis=((8.0, 15.0), (8.0, 3.6)), h=0.8, tones=body_t, texture=False,
                  overlay=[(rim_light, 1.3, 0.0), (under, 0.55, -0.03)]),
             part(rim, "iron", "bevel", bevel=0.8, h=0.92, tones=rim_t, form=False, texture=False),
             part(opening, "iron", "flat", h=0.62, tones=open_t, glint=False, texture=False,
                  overlay=[(far_wall, 1.45, 0.0)])]
    cont = _contents(src)
    if cont.any() and not (name == "bucket" and ns == "minecraft"):
        surf = _ellipse(cv, (C[0], C[1] + 0.25), OX - 0.25, OY - 0.2)
        spill = _spill(cv, cont, surf)
        spill &= cv2.erode(body | rim, np.ones((int(cv.k * 1.0),) * 2, np.uint8)).astype(bool)  # clear of the outline
        em = 0.8 if GLOW.search(name) else 0.0
        if spill.any():                                       # spills first: outlined against the pail only
            parts.append(part(spill, "glossy", "post", h=0.9, region=cont, emissive=em))
        parts.append(part(surf, "glossy", "post", h=0.86, region=cont, emissive=em, inner_line=False))
    return render_item(parts, cv, src, size)
