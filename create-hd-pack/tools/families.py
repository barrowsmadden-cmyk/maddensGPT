"""Family generators: hand-built HD designs driven by the source texture.

Each generator reads the source's layout (board seams, bark ridges, ring border,
stone sizes, leaf coverage) and its exact colours (rank-mapped onto the HD
detail), so one generator serves every wood or stone type in vanilla and mods.
All output tiles seamlessly (wrap) at any size.
"""
import re

import numpy as np
from PIL import Image

from texgen import (Material, fbm, worley, blur, sstep, round_profile, lerp, _rng, cell_search,
                    DIELECTRIC)

LUM = np.array([0.2126, 0.7152, 0.0722])


# ------------------------------------------------------------ helpers ---

def load_rgba(path):
    return np.asarray(Image.open(path).convert("RGBA"), float) / 255


def grid(size):
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    return xx + 0.5, yy + 0.5


def rank_map(field, colours, mask=None):
    """Colour each pixel by its rank in `field`, drawn from the source colours
    sorted by luminance: the HD texture keeps the original's exact colour mix."""
    out = np.zeros(field.shape + (3,))
    sel = np.ones(field.shape, bool) if mask is None else mask
    if not sel.any() or len(colours) == 0:
        return out
    cols = colours[np.argsort(colours @ LUM)]
    f = field[sel]
    rank = np.argsort(np.argsort(f, kind="stable"), kind="stable") / max(len(f) - 1, 1)
    idx = rank * (len(cols) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(cols) - 1)
    t = (idx - lo)[:, None]
    out[sel] = cols[lo] * (1 - t) + cols[hi] * t
    return out


def wrap_d(a, b, size):
    d = np.abs(a - b) % size
    return np.minimum(d, size - d)


def worley2(n_points, seed, size, jitter=0.9):
    """Tileable F1, F2 distances and nearest-cell id (jittered grid points)."""
    r = _rng(seed)
    k = max(1, int(round(np.sqrt(n_points))))
    cell = size / k
    gy, gx = np.mgrid[0:k, 0:k].reshape(2, -1)
    pts = np.stack([gx, gy], 1) * cell + cell * (0.5 + (r.random((k * k, 2)) - 0.5) * jitter)
    xx, yy = grid(size)
    f1, f2, ids = cell_search(pts, k, size, xx, yy)
    return f1, f2, ids, len(pts)


def opaque_colours(rgba, mask=None):
    sel = rgba[..., 3] > 0.5
    if mask is not None:
        sel &= mask
    return rgba[..., :3][sel]


# ------------------------------------------------------------- planks ---

def plank_layout(src):
    """Board seams (rows) and butt joints (columns per board) from the source.
    Returns (seam_rows, joints, transposed) in source pixel units."""
    lum = src[..., :3] @ LUM
    h, w = lum.shape

    def seams_of(l):
        rows = l.mean(1)
        thr = rows.mean() - 0.6 * rows.std()
        seam = [r for r in range(len(rows)) if rows[r] < thr and rows[r] <= rows[r - 1] and rows[r] <= rows[(r + 1) % len(rows)]]
        return seam, rows.std()

    s_rows, sr = seams_of(lum)
    s_cols, sc = seams_of(lum.T)
    transposed = sc > sr * 1.3 and len(s_cols) >= 2
    if transposed:
        lum = lum.T
        s_rows = s_cols
    if len(s_rows) < 2:
        s_rows = [int(h / 4 * (i + 1)) - 1 for i in range(4)]
    joints = []
    bounds = sorted(s_rows)
    for i, r in enumerate(bounds):
        top = bounds[i - 1] + 1 if i else bounds[-1] + 1 - h
        rows = [(rr % h) for rr in range(top, r)]
        if not rows:
            joints.append([])
            continue
        cols = lum[rows].mean(0)
        thr = cols.mean() - 1.4 * cols.std()
        joints.append([c for c in range(w) if cols[c] < thr and cols[c] <= cols[c - 1] and cols[c] <= cols[(c + 1) % w]])
    return bounds, joints, transposed


def planks(src, size, seed=0):
    h, w = src.shape[:2]
    seams, joints, transposed = plank_layout(src)
    s = size / w  # output px per source px
    xx, yy = grid(size)
    r = _rng(seed)
    m = Material(size)

    # board index per output row; board spans from previous seam (exclusive) to this seam
    bounds = [(b + 1) * s for b in seams]  # bottom edge of each seam row
    n = len(bounds)
    edges = np.array(bounds)
    board = np.searchsorted(edges, yy % size, side="right") % n
    top_edge = np.where(board == 0, edges[-1] - size, edges[(board - 1) % n])
    bot_edge = edges[board]
    pos = (yy - top_edge) % size
    span = (bot_edge - top_edge) % size
    span = np.where(span <= 0, size, span)
    v = pos / span  # 0 top .. 1 bottom of the board

    grain = fbm(2, 64, 4, seed + 1, 0.55, size=size)            # streaks along x
    warp = fbm(2, 6, 3, seed + 2, size=size)
    height = np.zeros((size, size))
    field = np.zeros((size, size))
    joint_d = np.full((size, size), 1e9)
    for b in range(n):
        sel = board == b
        roll = int(r.integers(0, size))
        g = np.roll(grain, roll, axis=0)
        wv = np.roll(warp, roll, axis=0)
        ring = 0.5 + 0.5 * np.sin((v * 2.2 + wv * 3 + np.sin(xx / size * 2 * np.pi * r.integers(1, 3)) * 0.4) * 2 * np.pi)
        late = sstep(0.4, 0.95, ring)
        tone = r.uniform(-0.08, 0.08)
        f = 0.75 * g + 0.25 * (1 - late) + tone
        crown = 0.03 * (1 - (2 * v - 1) ** 2)
        hb = 0.62 + crown + r.uniform(-0.015, 0.015) - (1 - g) * 0.02 - late * 0.006
        # butt joints of this board (source columns -> output x), groove + segment tone
        for c in joints[b] if b < len(joints) else []:
            jx = (c + 0.5) * s
            joint_d = np.where(sel, np.minimum(joint_d, wrap_d(xx, jx, size)), joint_d)
        field[sel] = f[sel]
        height[sel] = hb[sel]
    # segment tone changes across joints: shift field per segment using a hash of segment index
    seam_d = np.minimum(pos, span - pos)
    groove = (1 - np.clip(seam_d / (0.55 * s), 0, 1)) ** 1.5
    jgroove = (1 - np.clip(joint_d / (0.45 * s), 0, 1)) ** 1.5
    height -= 0.16 * groove + 0.12 * jgroove

    # nail holes beside each butt joint, near the board's top and bottom
    nails = np.zeros((size, size))
    for b in range(n):
        for c in joints[b] if b < len(joints) else []:
            jx = (c + 0.5) * s
            for side in (-1, 1):
                for frac in (0.3, 0.7):
                    nx = jx + side * 0.9 * s
                    ny = (edges[b - 1] if b else edges[-1] - size) + ((edges[b] - (edges[b - 1] if b else edges[-1] - size)) % size) * frac
                    d = np.hypot(wrap_d(xx, nx, size), wrap_d(yy, ny % size, size))
                    nails = np.maximum(nails, sstep(0.32 * s, 0.18 * s, d))
    height -= 0.05 * nails

    seam_mask = groove > 0.6
    colours = opaque_colours(src)
    col = rank_map(field, colours)
    col = col * (1 - 0.45 * np.maximum(groove, jgroove))[..., None] * (1 - 0.5 * nails)[..., None]
    if transposed:
        col, height, field = col.transpose(1, 0, 2), height.T, field.T
    m.albedo, m.height = col, np.clip(height, 0.02, 1)
    m.smooth = 0.32 + (field - field.mean()) * 0.15
    m.porosity[:] = 12
    return m, dict(edge_wear=0.1, ao_strength=1.2)


# --------------------------------------------------------------- logs ---

def bark(src, size, seed=0, stripped=False):
    xx, yy = grid(size)
    m = Material(size)
    colours = opaque_colours(src)
    if stripped:
        g = fbm(64, 2, 4, seed + 3, 0.55, size=size)
        streak = fbm(10, 1, 3, seed + 4, size=size)
        field = 0.7 * g + 0.3 * streak
        height = 0.7 + (g - 0.5) * 0.05 + (streak - 0.5) * 0.03
        m.smooth = 0.35 + (g - 0.5) * 0.1
    else:
        ridges = int(_rng(seed).integers(5, 8))
        warp = fbm(3, 2, 3, seed + 5, size=size) - 0.5
        u = (xx / size + warp * 0.35) * ridges
        ridge = np.abs(np.sin(np.pi * u))
        plates = fbm(ridges * 2, 3, 4, seed + 6, size=size)
        crack = (1 - ridge) ** 3
        # horizontal breaks across the ridges
        brk = sstep(0.62, 0.7, fbm(ridges * 2, 10, 2, seed + 7, size=size))
        rough = fbm(48, 16, 3, seed + 8, size=size)
        height = 0.35 + 0.5 * ridge ** 0.6 * (0.7 + 0.3 * plates) - 0.18 * crack - 0.06 * brk * ridge + (rough - 0.5) * 0.06
        field = height + (rough - 0.5) * 0.15
        m.smooth = 0.18 + (rough - 0.5) * 0.08
    m.albedo = rank_map(field, colours)
    m.height = np.clip(height, 0.02, 1)
    m.porosity[:] = 20
    return m, dict(edge_wear=0.05, ao_strength=1.4)


def log_top(src, size, seed=0):
    h, w = src.shape[:2]
    s = size / w
    lab = src[..., :3]
    # bark border width: how many source rings from the edge differ from the inside
    def ring_mean(k):
        sel = np.zeros((h, w), bool)
        sel[k:h - k, k:w - k] = True
        if h - 2 * (k + 1) > 0:
            sel[k + 1:h - k - 1, k + 1:w - k - 1] = False
        return lab[sel].mean(0)
    # bark = the outermost ring(s) when they differ from the ring just inside;
    # growth rings further in also alternate, so only compare neighbours
    diffs = [np.linalg.norm(ring_mean(k) - ring_mean(k + 1)) for k in range(min(4, h // 4))]
    border = 0
    for k in range(min(2, len(diffs) - 1)):
        # a bark ring stands out from its neighbour clearly more than inner rings do
        if diffs[k] > 1.15 * diffs[k + 1]:
            border = k + 1
        else:
            break
    xx, yy = grid(size)
    c = size / 2
    dx, dy = np.abs(xx - c), np.abs(yy - c)
    cheb = np.maximum(dx, dy)
    rad = 0.55 * cheb + 0.45 * np.hypot(dx, dy) * 0.92
    warp = fbm(4, 4, 3, seed + 9, size=size) - 0.5
    rings = 0.5 + 0.5 * np.sin((rad / (0.95 * s) + warp * 2.5) * 2 * np.pi)
    late = sstep(0.45, 0.95, rings)
    fine = fbm(24, 24, 3, seed + 10, size=size)
    edge = np.minimum(np.minimum(xx, size - xx), np.minimum(yy, size - yy))
    bark_w = border * s
    in_bark = edge < bark_w
    # radial checks (drying cracks) from the pith
    ang = np.arctan2(yy - c, xx - c)
    r = _rng(seed)
    cracks = np.zeros((size, size))
    for a0 in r.uniform(-np.pi, np.pi, 3):
        da = np.abs((ang - a0 + np.pi) % (2 * np.pi) - np.pi) * np.hypot(xx - c, yy - c)
        length = r.uniform(0.25, 0.42) * size
        cracks = np.maximum(cracks, 0.7 * sstep(1.1, 0.3, da) * sstep(length, length * 0.6, np.hypot(xx - c, yy - c)))
    pith = sstep(0.8 * s, 0.3 * s, np.hypot(xx - c, yy - c))
    interior_cols = opaque_colours(src, np.pad(np.ones((h - 2 * border, w - 2 * border), bool), border) if border else None)
    bark_cols = opaque_colours(src, ~np.pad(np.ones((h - 2 * border, w - 2 * border), bool), border)) if border else interior_cols
    field = (1 - late) * 0.6 + fine * 0.4 - cracks * 0.5 - pith * 0.4
    col = rank_map(field, interior_cols, ~in_bark)
    if border:
        bf = fbm(32, 32, 4, seed + 11, size=size)
        col = np.where(in_bark[..., None], rank_map(bf, bark_cols, in_bark), col)
    height = 0.6 + (1 - late) * 0.02 + (fine - 0.5) * 0.02 - cracks * 0.12 - pith * 0.04
    if border:
        bf = fbm(32, 32, 4, seed + 11, size=size)
        rim = round_profile(bark_w - edge, 0.5 * s)
        height = np.where(in_bark, 0.72 + 0.2 * rim * (0.6 + 0.4 * bf), height)
    m = Material(size)
    m.albedo, m.height = col, np.clip(height, 0.02, 1)
    m.smooth = np.where(in_bark, 0.18, 0.3)
    m.porosity[:] = 18
    return m, dict(edge_wear=0.05, ao_strength=1.2)


# ------------------------------------------------------------ cobble ---

def cobble(src, size, seed=0):
    rgb = src[..., :3]
    lum = rgb @ LUM
    opaque = src[..., 3] > 0.5
    # moss: clearly green pixels in the source
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-3)
    hue_g = (rgb[..., 1] >= rgb[..., 0]) & (rgb[..., 1] >= rgb[..., 2] * 0.9)
    moss_src = (sat > 0.18) & hue_g & opaque
    stone_px = opaque & ~moss_src
    thr = np.percentile(lum[stone_px], 28) if stone_px.any() else 0.3
    stone_cols = rgb[stone_px & (lum >= thr)]
    mortar_cols = rgb[stone_px & (lum < thr)]
    if len(mortar_cols) == 0:
        mortar_cols = stone_cols * 0.6

    s = size / src.shape[1]
    f1, f2, ids, npts = worley2(26, seed + 20, size)
    e = (f2 - f1) / 2  # distance to the cell border
    mortar_w = 0.16 * s  # half-width: joints ~0.3 vanilla px wide
    stone = sstep(mortar_w - 0.6, mortar_w + 0.6, e)
    r = _rng(seed + 21)
    tones = r.uniform(-0.25, 0.25, npts)
    domes = r.uniform(0.75, 1.0, npts)
    rough = fbm(20, 20, 5, seed + 22, size=size)
    chips = fbm(60, 60, 2, seed + 23, size=size)
    # rounded stones: t runs 0 at the stone's centre to 1 at its edge and is smooth
    # inside the cell, so the cap has no ridges (distance-to-edge domes peak in
    # pyramid-like creases along each cell's medial lines)
    radius = np.zeros(npts)
    np.maximum.at(radius, ids.ravel(), f1.ravel())          # each stone's farthest pixel
    cap = np.sqrt(np.clip(1 - (f1 / np.maximum(radius[ids], 1.0)) ** 2, 0, 1))
    rim = round_profile(e - mortar_w, 0.45 * s)              # narrow rounded edge at the mortar
    dome = (0.6 * cap + 0.4) * rim * domes[ids] * stone
    height = np.where(stone > 0.5, 0.45 + 0.45 * dome + (rough - 0.5) * 0.08, 0.3 + (chips - 0.5) * 0.05)
    field = tones[ids] + (rough - 0.5) * 0.8 + dome * 0.35
    col = rank_map(field, stone_cols, stone > 0.5)
    col = np.where((stone > 0.5)[..., None], col, rank_map(chips, mortar_cols, stone <= 0.5))
    smooth = np.where(stone > 0.5, 0.28 + dome * 0.1, 0.12)
    poro = np.where(stone > 0.5, 35.0, 55.0)
    if moss_src.any():
        big = np.asarray(Image.fromarray(moss_src.astype(np.float32), "F").resize((size, size), Image.BILINEAR), float)
        wobble = fbm(16, 16, 4, seed + 24, size=size)
        moss = sstep(0.35, 0.55, blur(big, s * 0.6) + (wobble - 0.5) * 0.5) * (1 - 0.5 * dome)
        moss_cols = rgb[moss_src]
        col = lerp(col, rank_map(wobble, moss_cols, moss > 0.01), moss)
        height = height + moss * 0.05 * wobble
        smooth = lerp(smooth, 0.1, moss)
        poro = np.where(moss > 0.5, 60.0, poro)
    m = Material(size)
    m.albedo, m.height, m.smooth, m.porosity = col, np.clip(height, 0.02, 1), smooth, poro
    return m, dict(edge_wear=0.1, ao_strength=1.3)


# ------------------------------------------------------------ bricks ---

def brick_mortar(src):
    """Mortar mask from the source: cluster the colours, then pick the cluster
    (or the two darkest/lightest clusters together) that runs along whole rows,
    i.e. the bed joints. None if the texture has no mortar grid."""
    from remaster import material_regions
    rgb = src[..., :3]
    opaque = src[..., 3] > 0.5
    h, w = opaque.shape
    lum = rgb @ LUM
    for k in (2, 3, 4):
        labels = material_regions(rgb, opaque, max_regions=k)
        ids = [i for i in np.unique(labels) if i >= 0]
        cands = [labels == i for i in ids]
        by_lum = sorted(ids, key=lambda i: lum[labels == i].mean())
        if len(by_lum) >= 3:
            cands += [(labels == by_lum[0]) | (labels == by_lum[1]), (labels == by_lum[-1]) | (labels == by_lum[-2])]
        best = None
        for m in cands:
            rows = (m.mean(1) >= 0.7).sum()
            frac = m.mean()
            if 2 <= rows <= h // 2 and 0.08 <= frac <= 0.45:
                score = rows - abs(frac - 0.25) * 4
                if best is None or score > best[0]:
                    best = (score, m)
        if best is not None:
            return best[1] & opaque
    return None


def bricks(src, size, seed=0):
    from vectorize import coverage
    rgb = src[..., :3]
    mortar = brick_mortar(src)
    if mortar is None:
        raise ValueError("no mortar grid")
    h, w = mortar.shape
    s = size / w
    brick_cov = coverage(~mortar, (size, size), s if s == int(s) else int(round(s)))
    if brick_cov.shape != (size, size):
        brick_cov = np.asarray(Image.fromarray(brick_cov.astype(np.float32)).resize((size, size), Image.BILINEAR))
    brick = brick_cov > 0.5
    n, comp = __import__("cv2").connectedComponents(brick.astype(np.uint8), connectivity=4)
    r = _rng(seed + 40)
    tones = r.uniform(-0.3, 0.3, n + 1)
    # distance to mortar for the bevel (capped)
    from remaster import dist_to_edge
    bevel = 0.55 * s
    d = dist_to_edge(np.where(brick, 0, -1), int(bevel) + 2)
    rough = fbm(18, 18, 5, seed + 41, size=size)
    pits = 1 - sstep(0, 0.35 * s, worley(int((size / s) ** 2 * 0.25) + 4, seed + 42, size, jitter=0.9))
    face = round_profile(d, bevel) * brick
    height = np.where(brick, 0.6 + 0.32 * face + (rough - 0.5) * 0.06 - pits * 0.05, 0.28 + (rough - 0.5) * 0.05)
    field = tones[comp] + (rough - 0.5) * 0.9 - pits * 0.4 + face * 0.2
    lum = rgb @ LUM
    brick_cols = rgb[~mortar & (src[..., 3] > 0.5)]
    mortar_cols = rgb[mortar]
    col = rank_map(field, brick_cols, brick)
    col = np.where(brick[..., None], col, rank_map(rough, mortar_cols, ~brick))
    m = Material(size)
    m.albedo, m.height = col, np.clip(height, 0.02, 1)
    m.smooth = np.where(brick, 0.25 + face * 0.08, 0.1)
    m.porosity = np.where(brick, 35.0, 55.0)
    return m, dict(edge_wear=0.1, ao_strength=1.3)


# ------------------------------------------------------------ leaves ---

def leaves(src, size, seed=0):
    coverage = (src[..., 3] > 0.5).mean()
    colours = opaque_colours(src)
    s = size / src.shape[1]
    xx, yy = grid(size)
    r = _rng(seed + 30)
    H = np.full((size, size), -1.0)
    shade = np.zeros((size, size))
    L, W = 3.0 * s, 1.35 * s
    count = 0
    while True:
        cx, cy = r.uniform(0, size, 2)
        ang = r.uniform(0, np.pi)
        z = r.uniform(0.3, 1.0)
        tone = r.uniform(-0.3, 0.3)
        dx = (xx - cx + size / 2) % size - size / 2
        dy = (yy - cy + size / 2) % size - size / 2
        u = dx * np.cos(ang) + dy * np.sin(ang)
        v = -dx * np.sin(ang) + dy * np.cos(ang)
        t = u / L + 0.5
        half = (W / 2) * np.clip(1 - (2 * t - 1) ** 2, 0, 1) ** 0.7
        inside = (t > 0) & (t < 1) & (np.abs(v) < half)
        cup = 1 - (v / np.maximum(half, 1e-3)) ** 2
        hz = z * 0.6 + 0.35 + 0.08 * cup
        take = inside & (hz > H)
        rib = sstep(0.12 * s, 0.03 * s, np.abs(v)) * (t > 0.08)
        H = np.where(take, hz, H)
        shade = np.where(take, tone + 0.45 * z + 0.25 * cup - 0.35 * rib + 0.25 * (1 - t), shade)
        count += 1
        if count % 10 == 0 and (H > -1).mean() >= coverage:
            break
        if count > 2000:
            break
    alpha = H > -1
    m = Material(size)
    m.albedo = rank_map(shade, colours, alpha)
    m.height = np.where(alpha, H, 0.0)
    m.alpha = alpha.astype(float)
    m.smooth = np.where(alpha, 0.42, 0.0)
    m.porosity = np.where(alpha, 200.0, 0.0)  # 65+ = subsurface scattering
    return m, dict(edge_wear=0.0, ao_strength=1.0)


# ------------------------------------------------------------ routing ---

FAMILIES = [
    ("leaves", re.compile(r"leaves")),
    ("log_top", re.compile(r"(_log|_stem|_wood|_hyphae)_top$|log_top$|stem_top$")),
    ("planks", re.compile(r"planks$")),
    ("cobble", re.compile(r"cobblestone$|^cobbled_|mossy_cobblestone$")),
    # bricks: generator kept but not routed; mortar detection is unreliable on mod
    # bricks, so patterned blocks go through the vector remaster instead
    ("bark", re.compile(r"(_log|_stem)$")),
]


def family_of(path):
    name = path.split("/")[-1].rsplit(".", 1)[0].lower()
    for fam, pat in FAMILIES:
        if pat.search(name):
            return fam
    return None


def generate(path, src, size, seed=0):
    fam = family_of(path)
    name = path.split("/")[-1].lower()
    if fam == "planks":
        m, opts = planks(src, size, seed)
    elif fam == "bark":
        m, opts = bark(src, size, seed, stripped="stripped" in name)
    elif fam == "log_top":
        m, opts = log_top(src, size, seed)
    elif fam == "cobble":
        m, opts = cobble(src, size, seed)
    elif fam == "leaves":
        m, opts = leaves(src, size, seed)
    elif fam == "bricks":
        m, opts = bricks(src, size, seed)
    else:
        raise KeyError(path)
    # the block face is the highest point: shift so the top sits just under 1.0
    # (LabPBR parallax measures depth down from 1.0; a low top sinks the whole face)
    vis = m.alpha > 0.5
    m.height = np.where(vis, np.clip(m.height + (0.99 - m.height[vis].max()), 0.0, 0.99), 0.0)
    return m, opts
