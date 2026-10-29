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
# coordinates in vanilla item pixels (0..16), taken from the vanilla layouts

def sword(cv, mat):
    pal = PALETTES[mat]
    tip, base = (15.4, 0.6), (6.4, 9.6)
    # broad blade (~3px), parallel edges, tapering over the last 3px to the point
    blade = cv.polygon([(5.3, 8.5), (12.6, 1.2), tip, (14.8, 3.4), (7.5, 10.7)])
    guard = cv.stroke((2.9, 6.9), (9.1, 13.1), 2.1)
    grip = cv.stroke((5.2, 10.8), (2.6, 13.4), 2.0)
    pommel = cv.circle((1.75, 14.25), 1.45)
    return [
        _shade_part(grip, cv, GRIP, 0.6, axis=((5.0, 11.0), (2.4, 13.6)), grain=True, seed=1),
        _shade_part(pommel, cv, pal, 0.9),
        _shade_part(blade, cv, pal, 1.1, axis=(base, tip), fuller=True),
        _shade_part(guard, cv, pal, 0.7),
    ]


def handle(cv, p0, p1, width=2.3):
    return _shade_part(cv.stroke(p0, p1, width), cv, HANDLE, 0.75, axis=(p0, p1), grain=True, seed=2)


def pickaxe(cv, mat):
    pal = PALETTES[mat]
    head = cv.tapered_curve((4.4, 3.0), (13.7, 2.3), (14.0, 12.0), 3.0, 1.0)
    return [handle(cv, (2.6, 13.8), (12.0, 4.4)), _shade_part(head, cv, pal, 1.1)]


def axe(cv, mat):
    pal = PALETTES[mat]
    # cutting edge bulges out to the upper left; the head wraps the handle top
    # fan blade: pointed toe and heel, slightly convex cutting edge, narrow neck
    edge = [(5.4 + 5.2 * t - 0.7 * math.sin(math.pi * t), 6.6 - 5.8 * t - 0.7 * math.sin(math.pi * t)) for t in np.linspace(0, 1, 14)]
    head = cv.polygon(edge + [(11.6, 1.9), (12.3, 3.5), (14.0, 4.4), (14.5, 6.6), (12.5, 8.6), (11.0, 7.4), (8.2, 7.3)])
    return [handle(cv, (2.6, 13.8), (12.6, 3.8)), _shade_part(head, cv, pal, 1.2)]


def shovel(cv, mat):
    pal = PALETTES[mat]
    c, u = np.array([11.9, 4.4]), np.array([1, -1]) / math.sqrt(2)
    v = np.array([1, 1]) / math.sqrt(2)
    pts = []
    for a in np.linspace(-math.pi / 2, math.pi / 2, 18):     # rounded tip
        pts.append(c + u * (1.0 + 2.6 * math.cos(a)) + v * 2.7 * math.sin(a))
    pts += [c - u * 2.4 + v * 2.3, c - u * 3.2, c - u * 2.4 - v * 2.3]
    blade = cv.polygon(pts)
    return [handle(cv, (2.6, 13.8), (11.2, 5.4)), _shade_part(blade, cv, pal, 1.2)]


def hoe(cv, mat):
    pal = PALETTES[mat]
    head = cv.polygon([(5.6, 1.2), (10.9, 0.9), (13.9, 2.8), (14.4, 5.6), (12.6, 6.3), (11.4, 4.8), (9.1, 3.9), (6.3, 3.9)])
    return [handle(cv, (2.6, 13.8), (12.6, 3.8)), _shade_part(head, cv, pal, 1.0)]


DESIGNS = {"sword": sword, "pickaxe": pickaxe, "axe": axe, "shovel": shovel, "hoe": hoe}


def render(tool, mat, size):
    """Return (albedo RGBA, _n, _s) uint8 maps for e.g. ('pickaxe', 'iron', 64)."""
    cv = Canvas(size)
    col, h, cover = _compose(DESIGNS[tool](cv, mat), cv)
    down = lambda a: cv2.resize(a.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    alpha = down(cover.astype(np.float32))
    rgb = np.dstack([down(col[..., i] * cover) for i in range(3)]) / np.maximum(alpha, 1e-4)[..., None]
    height = down(h * cover) / np.maximum(alpha, 1e-4)
    m = Material(size)
    solid = alpha > 0.5
    m.albedo = np.clip(rgb, 0, 1)
    m.alpha = alpha
    m.height = np.where(solid, np.clip(height + (0.99 - height[solid].max()), 0, 0.99), 0.0)
    # material per pixel: wood handles stay wood, heads take the tool material
    wood = (rgb[..., 0] > rgb[..., 2] * 1.3) & (mat != "golden") & (mat != "wooden")
    is_head = solid & ~wood
    m.f0 = np.where(is_head, MATERIAL_F0[mat], DIELECTRIC).astype(float)
    m.smooth = np.where(is_head, MATERIAL_SMOOTH[mat], 0.3) * solid
    return derive(m, edge_wear=0.0, edges="reflect", normal_strength=4.0)


def items():
    """(namespace path, tool, material) for every vanilla tool texture."""
    out = []
    for mat in PALETTES:
        for tool in DESIGNS:
            out.append((f"textures/item/{mat}_{tool}.png", tool, mat))
    return out
