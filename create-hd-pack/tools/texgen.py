"""Procedural 256x LabPBR texture toolkit for the Create HD resource pack.

Every texture is built from a height field plus per-pixel material layers
(albedo, smoothness, F0/metal id, porosity). The LabPBR maps are derived from
those layers so albedo, normals, AO and height always line up.
"""
import numpy as np
from PIL import Image, ImageDraw

N = 256

# LabPBR predefined metal ids (_s green channel)
IRON, GOLD, ALUMINUM, CHROME, COPPER, LEAD, PLATINUM, SILVER, ALBEDO_METAL = 230, 231, 232, 233, 234, 235, 236, 237, 255
DIELECTRIC = 10  # ~4% F0, typical for wood/stone/paint

# How much of LabPBR's 25% max depth the 0..1 height range uses.
HEIGHT_RANGE = 0.5


# ---------------------------------------------------------------- noise ---

def _rng(seed):
    return np.random.default_rng(seed)


def vnoise(cx, cy, seed, size=N):
    """Tileable value noise with cx by cy lattice cells."""
    cx, cy = max(1, min(int(cx), size)), max(1, min(int(cy), size))
    g = _rng(seed).random((cy, cx))
    xs = np.arange(size) * cx / size
    ys = np.arange(size) * cy / size
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    fx = xs - x0
    fy = ys - y0
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    x1 = (x0 + 1) % cx
    y1 = (y0 + 1) % cy
    a = g[np.ix_(y0, x0)]
    b = g[np.ix_(y0, x1)]
    c = g[np.ix_(y1, x0)]
    d = g[np.ix_(y1, x1)]
    top = a + (b - a) * fx
    bot = c + (d - c) * fx
    return top + (bot - top) * fy[:, None]


def fbm(cx, cy, octaves, seed, persistence=0.5, size=N):
    total = np.zeros((size, size))
    amp, norm = 1.0, 0.0
    shifts = _rng(seed + 7919).integers(0, size, (octaves, 2))
    for o in range(octaves):
        # shift each octave by a whole number of pixels (still tiles) so the lattice
        # lines, where value noise flattens, never line up across octaves
        octave = vnoise(cx * 2 ** o, cy * 2 ** o, seed + 101 * o, size)
        total += amp * np.roll(octave, tuple(shifts[o]), axis=(0, 1))
        norm += amp
        amp *= persistence
    return total / norm


def worley(npts, seed, size=N, jitter=None):
    """Tileable F1 distance (pixels) to scattered points.

    jitter: if set, points sit on a sqrt(npts) grid jittered by this fraction
    of a cell, which spreads them evenly (hammered metal) instead of clumping.
    """
    r = _rng(seed)
    if jitter is None:
        pts = r.random((npts, 2)) * size
    else:
        k = max(1, int(round(np.sqrt(npts))))
        cell = size / k
        gy, gx = np.mgrid[0:k, 0:k].reshape(2, -1)
        pts = np.stack([gx, gy], 1) * cell + cell * (0.5 + (r.random((k * k, 2)) - 0.5) * jitter)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    best = np.full((size, size), 1e9)
    for px, py in pts:
        dx = np.abs(xx - px)
        dy = np.abs(yy - py)
        dx = np.minimum(dx, size - dx)
        dy = np.minimum(dy, size - dy)
        best = np.minimum(best, np.hypot(dx, dy))
    return best


def blur(a, sigma):
    """Gaussian blur with wrap-around edges (FFT)."""
    if sigma <= 0:
        return a
    h, w = a.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    k = np.exp(-2 * np.pi ** 2 * sigma ** 2 * (fx ** 2 + fy ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(a) * k))


def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def round_profile(d, width):
    """0 at d<=0 rising to 1 at d>=width with a quarter-circle (rounded) shape."""
    t = np.clip(d / width, 0, 1)
    return np.sqrt(1 - (1 - t) ** 2)


def grid():
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    return xx + 0.5, yy + 0.5


def lerp(a, b, t):
    t = np.asarray(t)
    if np.ndim(a) and np.shape(a)[-1] == 3 and t.ndim == 2:
        t = t[..., None]
    return a + (np.asarray(b) - np.asarray(a)) * t


def col(*rgb):
    return np.array(rgb, dtype=float) / 255.0


def scratches(n, seed, length=(8, 40), width=1, angle=None, size=N):
    """Mask of thin random scratch lines (tileable by drawing wrapped copies)."""
    r = _rng(seed)
    img = Image.new("L", (size, size), 0)
    dr = ImageDraw.Draw(img)
    for _ in range(n):
        x, y = r.random(2) * size
        ang = r.random() * np.pi if angle is None else angle + r.normal(0, 0.12)
        ln = r.uniform(*length)
        x2, y2 = x + np.cos(ang) * ln, y + np.sin(ang) * ln
        v = int(r.uniform(120, 255))
        for ox in (-size, 0, size):
            for oy in (-size, 0, size):
                dr.line([(x + ox, y + oy), (x2 + ox, y2 + oy)], fill=v, width=width)
    return blur(np.asarray(img, float) / 255.0, 0.6)


# -------------------------------------------------------------- material ---

class Material:
    """Per-pixel material layers for one texture."""

    def __init__(self, h=N, w=None):
        w = h if w is None else w
        self.albedo = np.zeros((h, w, 3))
        self.height = np.zeros((h, w))
        self.smooth = np.zeros((h, w))
        self.f0 = np.full((h, w), DIELECTRIC, dtype=float)
        self.porosity = np.zeros((h, w))
        self.emissive = np.zeros((h, w))
        self.alpha = np.ones((h, w))
        self.extra_ao = np.ones((h, w))

    def put(self, mask, albedo=None, height=None, smooth=None, f0=None, porosity=None):
        m = np.clip(mask, 0, 1)
        if albedo is not None:
            self.albedo = lerp(self.albedo, albedo, m)
        if height is not None:
            self.height = self.height + (height - self.height) * m
        if smooth is not None:
            self.smooth = self.smooth + (smooth - self.smooth) * m
        if f0 is not None:
            # metal ids are discrete: hard switch at 50% coverage
            self.f0 = np.where(m > 0.5, f0, self.f0)
        if porosity is not None:
            self.porosity = self.porosity + (porosity - self.porosity) * m


def derive(mat, normal_strength=6.0, ao_strength=1.0, bake_ao=0.55, edge_wear=0.0, edges="wrap", pad=None):
    """Turn material layers into (albedo RGBA, _n RGBA, _s RGBA) uint8 arrays.

    edges: "wrap" for textures that tile against themselves; "reflect" for
    connected-texture tiles, whose real neighbour differs from the opposite edge.
    pad: optional (P, canvas), the tile's height with P px of its real neighbours'
    height around it; normals, AO and edge wear are then read across the seams
    exactly as the neighbouring blocks continue (no mirrored or flattened edge).
    """
    h = mat.height
    H, W = h.shape
    if pad is not None:
        P, canvas = pad
        assert canvas.shape == (H + 2 * P, W + 2 * P)
        assert np.allclose(canvas[P:P + H, P:P + W], h)

        def bl(a, sigma):
            # only height is blurred over the canvas; other maps fall back to reflect
            src = canvas if a is h else np.pad(a, P, mode="reflect")
            return blur(src, sigma)[P:P + H, P:P + W]

        cs = blur(canvas, 0.8)
        dx = ((cs[:, 2:] - cs[:, :-2]) * 0.5)[P:P + H, P - 1:P - 1 + W]
        dy = ((cs[2:, :] - cs[:-2, :]) * 0.5)[P - 1:P - 1 + H, P:P + W]
    elif edges == "wrap":
        bl = blur
        hs = bl(h, 0.8)
        dx = (np.roll(hs, -1, 1) - np.roll(hs, 1, 1)) * 0.5
        dy = (np.roll(hs, -1, 0) - np.roll(hs, 1, 0)) * 0.5
    else:
        def bl(a, sigma, p=48):
            return blur(np.pad(a, p, mode="reflect"), sigma)[p:-p, p:-p]
        hs = bl(h, 0.8)
        hp = np.pad(hs, 1, mode="reflect")
        dx = (hp[1:-1, 2:] - hp[1:-1, :-2]) * 0.5
        dy = (hp[2:, 1:-1] - hp[:-2, 1:-1]) * 0.5
    s = normal_strength * W / 16.0  # strength expressed per vanilla pixel (W px = one block face)
    nx, ny, nz = -dx * s, -dy * s, np.ones_like(h)
    ln = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nx, ny = nx / ln, ny / ln  # DirectX (Y-): +y points down the image

    # cavity AO at several radii
    cav = np.zeros_like(h)
    for sigma, w in ((2, 0.9), (6, 0.8), (14, 0.5)):
        cav += w * np.clip(bl(h, sigma) - h, 0, None)
    ao = np.clip(1 - cav * 4.0 * ao_strength, 0.25, 1) * mat.extra_ao
    ao = bl(ao, 0.7)

    albedo = mat.albedo.copy()
    if edge_wear:
        convex = np.clip(-(bl(h, 1.2) - bl(h, 3.5)) * 30, 0, 1)
        albedo = lerp(albedo, np.clip(albedo * 1.35 + 0.05, 0, 1), convex * edge_wear)
        mat.smooth = np.clip(mat.smooth + convex * edge_wear * 0.2, 0, 1)
    albedo = albedo * (1 - bake_ao * (1 - ao))[..., None]

    a8 = np.dstack([np.clip(albedo, 0, 1) * 255, mat.alpha * 255]).round().astype(np.uint8)
    hmap = 255 * (1 - (1 - np.clip(h, 0, 1)) * HEIGHT_RANGE)
    n8 = np.dstack([
        (nx * 0.5 + 0.5) * 255,
        (ny * 0.5 + 0.5) * 255,
        ao * 255,
        np.clip(hmap, 1, 255),
    ]).round().astype(np.uint8)
    em = np.where(mat.emissive > 0, np.clip(mat.emissive * 254, 0, 254), 255)
    s8 = np.dstack([
        np.clip(mat.smooth, 0, 1) * 255,
        np.clip(mat.f0, 0, 255),
        np.clip(mat.porosity, 0, 255),
        em,
    ]).round().astype(np.uint8)
    return a8, n8, s8


def save(name, maps, outdir):
    import os
    os.makedirs(outdir, exist_ok=True)
    a8, n8, s8 = maps
    Image.fromarray(a8, "RGBA").save(f"{outdir}/{name}.png", optimize=True)
    Image.fromarray(n8, "RGBA").save(f"{outdir}/{name}_n.png", optimize=True)
    Image.fromarray(s8, "RGBA").save(f"{outdir}/{name}_s.png", optimize=True)
