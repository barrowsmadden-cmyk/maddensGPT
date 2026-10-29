"""Hand-made HD tools and weapons.

Every tool is drawn from explicit shapes in the vanilla 16-unit item grid
(x right, y down), laid out to match the vanilla silhouettes: the handle runs
from the bottom-left corner up to the head at the top-right. One design per
tool type, six material palettes, so all wooden/stone/iron/golden/diamond/
netherite swords, pickaxes, axes, shovels and hoes come from these.
"""
import math

import cv2
import numpy as np

from texgen import Material, derive, IRON, GOLD, ALBEDO_METAL, DIELECTRIC

SS = 4  # supersampling

# light, mid, dark (sRGB 0-255) per material
PALETTES = {
    "wooden": ((196, 152, 92), (150, 110, 60), (86, 58, 28)),
    "stone": ((168, 168, 166), (126, 126, 124), (66, 66, 66)),
    "iron": ((244, 244, 244), (206, 206, 210), (112, 112, 118)),
    "golden": ((255, 240, 120), (240, 192, 44), (150, 100, 16)),
    "diamond": ((168, 255, 246), (58, 214, 206), (14, 108, 112)),
    "netherite": ((120, 106, 114), (82, 70, 78), (40, 32, 38)),
}
HANDLE = ((156, 112, 62), (116, 80, 40), (62, 40, 18))
GRIP = ((120, 84, 52), (86, 58, 34), (44, 28, 16))
# LabPBR _s green: metal id, or F0 for dielectrics (diamond ~0.17)
MATERIAL_F0 = {"wooden": DIELECTRIC, "stone": DIELECTRIC, "iron": IRON, "golden": GOLD,
               "diamond": 44, "netherite": ALBEDO_METAL}
MATERIAL_SMOOTH = {"wooden": 0.3, "stone": 0.25, "iron": 0.72, "golden": 0.8, "diamond": 0.92, "netherite": 0.62}


class Canvas:
    def __init__(self, size):
        self.size = size
        self.k = size / 16 * SS  # supersampled px per unit
        self.n = size * SS

    def blank(self):
        return np.zeros((self.n, self.n), np.uint8)

    def pts(self, p):
        return np.round(np.array(p, float) * self.k).astype(np.int32).reshape(-1, 1, 2)

    def polygon(self, p):
        m = self.blank()
        cv2.fillPoly(m, [self.pts(p)], 255, lineType=cv2.LINE_8)
        return m

    def stroke(self, p0, p1, width):
        m = self.blank()
        a, b = self.pts([p0])[0, 0], self.pts([p1])[0, 0]
        t = max(1, int(round(width * self.k)))
        cv2.line(m, tuple(int(v) for v in a), tuple(int(v) for v in b), 255, thickness=t, lineType=cv2.LINE_8)
        for c in (a, b):
            cv2.circle(m, tuple(int(v) for v in c), t // 2, 255, -1)
        return m

    def circle(self, c, r):
        m = self.blank()
        cv2.circle(m, tuple(int(v) for v in self.pts([c])[0, 0]), int(round(r * self.k)), 255, -1)
        return m

    def tapered_curve(self, p0, c, p2, w_mid, w_end, steps=60):
        """Quadratic Bezier band, widest in the middle, tapering to points."""
        m = self.blank()
        for i in range(steps + 1):
            t = i / steps
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t * t * p2[0]
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t * t * p2[1]
            w = w_end + (w_mid - w_end) * math.sin(math.pi * t) ** 0.7
            cv2.circle(m, (int(round(x * self.k)), int(round(y * self.k))), max(1, int(round(w / 2 * self.k))), 255, -1)
        return m


def _shade_part(mask, cv, palette, radius_u, axis=None, fuller=False, grain=False, seed=0):
    """Colour and height for one part: flat body, light from the top-left,
    highlight on the crown, dark outline band inside the edge."""
    light, mid, dark = (np.array(c, float) / 255 for c in palette)
    m = mask > 0
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5) / cv.k  # units
    t = np.clip(dist / radius_u, 0, 1)
    dome = np.sqrt(1 - (1 - t) ** 2)
    sm = cv2.GaussianBlur(dome, (0, 0), cv.k * 0.25)
    gy, gx = np.gradient(sm)
    nrm = np.dstack([-gx * cv.k * 2.2, -gy * cv.k * 2.2, np.ones_like(gx)])
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    L = np.array([-0.55, -0.6, 0.58]); L /= np.linalg.norm(L)
    shade = np.clip(nrm @ L, 0, 1)
    tone = 0.8 + 0.32 * shade
    col = mid[None, None, :] * tone[..., None]
    col = col + (light - mid)[None, None, :] * (0.55 * np.clip((dome - 0.75) / 0.25, 0, 1))[..., None]
    if axis is not None:
        (ax, ay), (bx, by) = axis
        dx, dy = bx - ax, by - ay
        ln = math.hypot(dx, dy)
        yy, xx = np.mgrid[0:cv.n, 0:cv.n].astype(float) / cv.k
        across = ((xx - ax) * -dy + (yy - ay) * dx) / ln
        along = ((xx - ax) * dx + (yy - ay) * dy) / ln
        if fuller:  # bright ridge down the middle of a blade, darker bevel each side
            ridge = np.exp(-(across / 0.28) ** 2)
            col = col + (light - mid)[None, None, :] * (0.8 * ridge)[..., None]
            col = col * (1 - 0.12 * np.clip(np.abs(across) - 0.3, 0, 1))[..., None]
        if grain:  # wood grain along the handle
            rng = np.random.default_rng(seed)
            g = np.zeros_like(across)
            for f, a in ((2.3, 0.5), (4.1, 0.3), (7.7, 0.2)):
                g += a * np.sin(across * f * math.pi + rng.uniform(0, 6) + np.sin(along * 0.9 + rng.uniform(0, 6)) * 0.6)
            col = col * (1 + 0.07 * g)[..., None]
    edge = dist < 0.42
    col = np.where(edge[..., None], dark[None, None, :] * (0.75 + 0.25 * shade[..., None]), col)
    height = 0.55 + 0.3 * dome - 0.08 * edge
    return np.clip(col, 0, 1), height, m


def _compose(parts, cv):
    """Paint parts in order (later parts on top); return supersampled maps."""
    col = np.zeros((cv.n, cv.n, 3))
    h = np.zeros((cv.n, cv.n))
    cover = np.zeros((cv.n, cv.n), bool)
    for layer, (c, hh, m) in enumerate(parts):
        col = np.where(m[..., None], c, col)
        h = np.where(m, hh + 0.03 * layer, h)
        cover |= m
    return col, h, cover


# ---------------------------------------------------------------- designs ---
# Traced from the vanilla textures: coordinates are texel edges in the 16-unit
# grid (texel (x, y) spans x..x+1, y..y+1). Parts use the house style
# (handmade/style.py); their tones come from the original texels they cover.

HEAD_MAT = {"wooden": "wood", "stone": "stone", "iron": "iron", "golden": "gold", "diamond": "gem",
            "netherite": "dark_metal"}


def _rp(pts, r=0.6):
    from handmade.style import round_poly
    return round_poly(pts, r)


def _handle(cv, top, bottom=(2.8, 14.2), width=2.1):
    from handmade.style import part
    return part(cv.stroke(bottom, top, width), "wood", "cyl", axis=(bottom, top), grain=1.0, h=0.7)


def sword(cv, mat):
    from handmade.style import part
    m = HEAD_MAT[mat]
    blade = cv.polygon([(13.0, 0.0), (16.0, 0.0), (16.0, 2.0), (7.75, 10.25), (5.25, 7.75)])
    fuller = cv.stroke((14.6, 1.4), (7.4, 8.6), 0.9)
    guard = cv.polygon(_rp([(1.6, 6.8), (3.2, 5.6), (10.4, 12.8), (9.2, 14.4)], 0.5))
    grip = cv.stroke((5.6, 10.4), (2.4, 13.6), 2.3)
    pommel = cv.polygon(_rp([(0.0, 13.0), (3.0, 13.0), (3.0, 16.0), (0.0, 16.0)], 0.7))
    return [
        part(grip, "wood", "cyl", axis=((5.6, 10.4), (2.4, 13.6)), grain=1.0, h=0.7),
        part(pommel, m, "bevel", bevel=0.9, h=0.75),
        part(blade, m, "bevel", bevel=1.0, h=0.85),
        part(fuller, m, "flat", h=0.82, inner_line=False),
        part(guard, m, "bevel", bevel=0.9, h=0.9),
    ]


def pickaxe(cv, mat):
    from handmade.style import part
    # curved head with two pointed tips (left, bottom), thick at the crown behind the handle
    outer = [(4.4, 3.7), (6.0, 2.1), (9.0, 1.7), (11.6, 1.9), (13.6, 2.6), (14.9, 4.2), (15.4, 6.6), (15.2, 9.4),
             (14.0, 12.6)]
    inner = [(12.6, 10.2), (12.3, 7.4), (11.6, 5.6), (9.8, 4.9), (7.4, 4.7)]
    head = cv.polygon(_rp(outer + inner, 0.35))
    return [part(head, HEAD_MAT[mat], "bevel", bevel=0.9, h=0.85), _handle(cv, (13.3, 3.7))]


def axe(cv, mat):
    from handmade.style import part
    # broad blade: convex cutting edge on the upper left, narrowing to a neck, poll behind the handle
    edge = [(9.2 - 3.4 * t - 1.2 * math.sin(math.pi * t), 0.8 + 6.6 * t - 0.6 * math.sin(math.pi * t))
            for t in np.linspace(0, 1, 12)]
    head = cv.polygon(_rp(edge + [(8.0, 7.2), (9.8, 6.6), (11.0, 8.8), (13.2, 8.9), (14.1, 7.6), (14.0, 5.8),
                                  (12.4, 4.2), (12.0, 2.0), (11.0, 0.9)], 0.45))
    return [part(head, HEAD_MAT[mat], "bevel", bevel=0.9, h=0.85), _handle(cv, (12.6, 4.6))]


def shovel(cv, mat):
    from handmade.style import part
    blade = cv.polygon(_rp([(11.0, 2.0), (14.0, 2.0), (15.0, 3.0), (15.0, 6.0), (12.0, 9.0), (11.0, 9.0), (8.0, 6.0),
                            (8.0, 5.0)], 1.0))
    return [part(blade, HEAD_MAT[mat], "bevel", bevel=0.9, h=0.85), _handle(cv, (10.2, 6.8), bottom=(3.0, 14.2))]


def hoe(cv, mat):
    from handmade.style import part
    head = cv.polygon(_rp([(7.0, 1.0), (10.0, 1.0), (11.0, 2.0), (12.0, 2.6), (13.6, 4.0), (14.0, 5.0), (14.0, 6.0),
                           (13.0, 7.0), (11.0, 7.0), (10.4, 5.2), (9.0, 5.0), (9.0, 4.2), (7.0, 4.0), (6.0, 3.0),
                           (6.0, 2.0)], 0.5))
    return [part(head, HEAD_MAT[mat], "bevel", bevel=0.8, h=0.85), _handle(cv, (12.8, 3.9))]


DESIGNS = {"sword": sword, "pickaxe": pickaxe, "axe": axe, "shovel": shovel, "hoe": hoe}


def render(tool, mat, size, src):
    """(albedo RGBA, _n, _s) uint8 maps for e.g. ('pickaxe', 'iron', 64, original texture)."""
    from handmade.style import render_item
    cv = Canvas(size)
    return render_item(DESIGNS[tool](cv, mat), cv, src, size)


def items():
    """(namespace path, tool, material) for every vanilla tool texture."""
    out = []
    for mat in PALETTES:
        for tool in DESIGNS:
            out.append((f"textures/item/{mat}_{tool}.png", tool, mat))
    return out
