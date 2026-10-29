"""Automatic HD remaster of a pixel-art texture into LabPBR maps.

Used for every texture that no hand-built family generator covers. The original
art decides the design; this adds resolution and depth:

1. Colours are grouped into material regions (pixel-art shading ramps of one hue
   stay together), so shapes come from the original layout.
2. Each region's mask is upscaled smoothly and the regions compete per pixel,
   giving clean rounded shapes instead of blocky pixels.
3. Colour inside a region is a smooth upscale of the original restricted to that
   region (no bleeding across edges), plus fine material detail.
4. Height = region level + rounded bevel at region edges + the original's own
   shading + material micro-relief; LabPBR maps come from texgen.derive().
"""
import re

import numpy as np
from PIL import Image

from texgen import (Material, derive, fbm, blur, sstep, round_profile, lerp, scratches, worley as worley_np,
                    IRON, GOLD, COPPER, ALBEDO_METAL, DIELECTRIC)

# ----------------------------------------------------------- materials ---
# keyword -> material class, first match wins (order matters)
CLASS_RULES = [
    ("glass", r"glass|pane|crystal(?!_lamp)|amethyst_block|ice\b|_ice"),
    ("ore", r"_ore\b|_ore_|ore$"),
    ("metal", r"iron|gold|copper|brass|zinc|steel|netherite|chain|anvil|casing|rail|cannon|"
              r"bronze|lead|nickel|aluminum|metal|bolt|gear|cog|pipe|valve|tank|lantern|bell|cauldron|hopper"),
    ("plant", r"leaves|leaf|sapling|flower|grass|fern|vine|moss|crop|stage|berry|bush|tulip|rose|"
              r"mushroom|azalea|roots|weed|petal|lily|kelp|seagrass|cactus|bamboo|sugar_cane"),
    ("fabric", r"wool|carpet|banner|bed|cushion|stool|sack|paper|cloth|curtain|sail|rope|hammock|"
               r"sleeping_bag|backpack|leather"),
    ("wood", r"plank|log|wood|stripped|door|trapdoor|ladder|crate|barrel|chest|sign|fence|table|"
             r"chair|shelf|bench|drawer|cabinet|frame|bookshelf|post|beam|board|stick|bowl|basket|box"),
    ("stone", r"stone|cobble|brick|deepslate|granite|diorite|andesite|tuff|calcite|basalt|blackstone|"
              r"sandstone|tile|pillar|polished|slab|mud|terracotta|concrete|obsidian|prismarine|purpur|"
              r"quartz|clay|gravel|sand|dirt|path|rock|marble|limestone|slate|jasper|shale|myalite"),
    ("food", r"bread|pie|cake|soup|stew|meat|steak|pork|chicken|fish|cod|salmon|apple|carrot|potato|"
             r"beet|melon|pumpkin|cookie|sandwich|burger|pizza|noodle|rice|cheese|egg|honey|jam|food"),
]
EMISSIVE = r"_on\b|_on_|lit\b|_lit|lamp|glow|torch|lantern|fire|lava|magma|shroomlight|froglight|sea_lantern|beacon"

PROPS = {  # smoothness range, f0, porosity/SSS, micro-relief strength
    "metal": dict(smooth=(0.45, 0.75), f0=ALBEDO_METAL, porosity=0, relief=0.010),
    "glass": dict(smooth=(0.85, 0.95), f0=10, porosity=0, relief=0.004),
    "ore": dict(smooth=(0.20, 0.35), f0=DIELECTRIC, porosity=30, relief=0.020),
    "stone": dict(smooth=(0.18, 0.35), f0=DIELECTRIC, porosity=40, relief=0.022),
    "wood": dict(smooth=(0.25, 0.40), f0=DIELECTRIC, porosity=12, relief=0.016),
    "fabric": dict(smooth=(0.08, 0.18), f0=DIELECTRIC, porosity=38, relief=0.012),
    "plant": dict(smooth=(0.25, 0.45), f0=DIELECTRIC, porosity=170, relief=0.010),  # 65+ = subsurface
    "food": dict(smooth=(0.30, 0.55), f0=DIELECTRIC, porosity=110, relief=0.012),
    "generic": dict(smooth=(0.25, 0.45), f0=DIELECTRIC, porosity=10, relief=0.012),
}
# natural materials read as surface texture even where the pixel art looks flat;
# designed materials (glass, metal, machines) keep their clean shapes
NATURAL_PRIOR = {"stone": 0.9, "ore": 0.9, "plant": 0.8, "food": 0.8, "fabric": 0.75, "wood": 0.6}
METAL_IDS = [(r"gold", GOLD), (r"copper", COPPER), (r"iron|chain|anvil|cauldron|hopper|rail", IRON)]


def classify(path):
    name = path.split("/")[-1].lower()
    for cls, pat in CLASS_RULES:
        if re.search(pat, name):
            return cls
    return "generic"


# ------------------------------------------------------------- colour ---

def srgb_to_lab(rgb):
    c = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def material_regions(rgb, opaque, max_regions=6):
    """Label pixels by material: colours are clustered on chroma first, so a
    shading ramp of one hue is one region; very dark outline colours split off."""
    lab = srgb_to_lab(rgb)
    feat = np.concatenate([lab[..., 1:] * 1.0, lab[..., :1] * 0.35], -1)
    pts = feat[opaque]
    labels = np.full(rgb.shape[:2], -1)
    if len(pts) == 0:
        return labels
    uniq = np.unique(np.round(pts, 1), axis=0)
    k = int(min(max_regions, max(1, len(uniq))))
    # k-means++ seeding, a few Lloyd iterations (deterministic)
    rng = np.random.default_rng(0)
    centres = [uniq[rng.integers(len(uniq))]]
    for _ in range(1, k):
        d = np.min(((uniq[:, None] - np.array(centres)[None]) ** 2).sum(-1), 1)
        if d.max() < 36:  # remaining colours are within ~6 dE: stop splitting
            break
        centres.append(uniq[np.argmax(d)])
    centres = np.array(centres)
    for _ in range(12):
        lab_ = np.argmin(((pts[:, None] - centres[None]) ** 2).sum(-1), 1)
        for j in range(len(centres)):
            if (lab_ == j).any():
                centres[j] = pts[lab_ == j].mean(0)
    labels[opaque] = np.argmin(((pts[:, None] - centres[None]) ** 2).sum(-1), 1)
    return labels


def components(labels):
    """8-connected components of a small label map: list of (label, [(y, x), ...]).
    8-connectivity keeps diagonal lines (common in pixel art) in one piece."""
    h, w = labels.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or labels[y, x] < 0:
                continue
            lab, stack, pix = labels[y, x], [(y, x)], []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                pix.append((cy, cx))
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1),
                               (cy + 1, cx + 1), (cy + 1, cx - 1), (cy - 1, cx + 1), (cy - 1, cx - 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and labels[ny, nx] == lab:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append((lab, pix))
    return comps


def merge_speckles(labels, min_size, lab, max_de=14.0):
    """Fold small, low-contrast components into the region around them, so pixel
    noise becomes surface texture instead of tiny embossed blobs. Specks that
    contrast strongly with their surroundings (outlines, studs, gems) are kept:
    they are part of the design."""
    labels = labels.copy()
    h, w = labels.shape
    for _ in range(4):
        changed = False
        for lab_id, pix in sorted(components(labels), key=lambda c: len(c[1])):
            if len(pix) >= min_size:
                continue
            votes = {}
            own = set(pix)
            for y, x in pix:
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in own and labels[ny, nx] >= 0:
                        votes[labels[ny, nx]] = votes.get(labels[ny, nx], 0) + 1
            if votes:
                new = max(votes, key=votes.get)
                own_col = np.mean([lab[y, x] for y, x in pix], 0)
                around = lab[labels == new].mean(0)
                if np.linalg.norm(own_col - around) > max_de:
                    continue
                if new != lab_id:
                    for y, x in pix:
                        labels[y, x] = new
                    changed = True
        if not changed:
            break
    return labels


def resynthesise(src_rgb, labels, big_labels, noise):
    """Per region: map a detail-noise field onto the region's own colours by rank,
    so the HD surface keeps the original's exact colour mix (histogram match)."""
    out = np.zeros(big_labels.shape + (3,))
    lumw = np.array([0.2126, 0.7152, 0.0722])
    for i in np.unique(big_labels):
        if i < 0:
            continue
        cols = src_rgb[labels == i]
        if len(cols) == 0:
            continue
        cols = cols[np.argsort(cols @ lumw)]
        sel = big_labels == i
        n = noise[sel]
        rank = np.argsort(np.argsort(n)) / max(len(n) - 1, 1)
        idx = rank * (len(cols) - 1)
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, len(cols) - 1)
        t = (idx - lo)[:, None]
        out[sel] = cols[lo] * (1 - t) + cols[hi] * t
    return out


# ----------------------------------------------------------- upscale ---

def upscale_regions(labels, scale, softness=0.42):
    """Smooth vector-like upscale of a label map (-1 = transparent)."""
    h, w = labels.shape
    H, W = h * scale, w * scale
    ids = np.unique(labels)
    best = np.full((H, W), -1e9)
    out = np.full((H, W), -1)
    for i in ids:
        m = (labels == i).astype(float)
        big = np.asarray(Image.fromarray(m.astype(np.float32), "F").resize((W, H), Image.NEAREST), float)
        soft = blur_pad(big, scale * softness)
        take = soft > best
        best = np.where(take, soft, best)
        out = np.where(take, i, out)
    return out


def blur_pad(a, sigma, mode="edge"):
    p = int(sigma * 3) + 2
    return blur(np.pad(a, p, mode=mode), sigma)[p:-p, p:-p]


def region_colour(rgb, labels, big_labels, scale):
    """Smooth colour inside each region from the original, without bleeding."""
    h, w = labels.shape
    H, W = big_labels.shape
    out = np.zeros((H, W, 3))
    for i in np.unique(big_labels):
        if i < 0:
            continue
        m = (labels == i).astype(float)
        mb = np.asarray(Image.fromarray(m.astype(np.float32), "F").resize((W, H), Image.NEAREST), float)
        cb = np.stack([np.asarray(Image.fromarray((rgb[..., c] * m).astype(np.float32), "F").resize((W, H), Image.NEAREST), float)
                       for c in range(3)], -1)
        s = scale * 0.55
        num = np.stack([blur_pad(cb[..., c], s) for c in range(3)], -1)
        den = blur_pad(mb, s)[..., None]
        sel = big_labels == i
        out[sel] = (num / np.maximum(den, 1e-4))[sel]
    return np.clip(out, 0, 1)


def dist_to_edge(big_labels, radius):
    """Distance (px, capped) from each pixel to a pixel of another region."""
    H, W = big_labels.shape
    p = radius + 1
    lab = np.pad(big_labels, p, mode="edge")
    d = np.full((H, W), float(radius))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            r = np.hypot(dx, dy)
            if r >= radius:
                continue
            other = lab[p + dy:p + dy + H, p + dx:p + dx + W] != big_labels
            d = np.where(other & (r < d), r, d)
    return d


# -------------------------------------------------------------- main ---

def remaster_frame(rgba, scale, path, seed=0, block_px=None):
    """rgba: float (h, w, 4) in 0..1. Returns (Material, derive options)."""
    h, w = rgba.shape[:2]
    H, W = h * scale, w * scale
    cls = classify(path)
    props = PROPS[cls]
    name = path.split("/")[-1].lower()
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    # translucent art (stained glass, ice, water) keeps its real alpha; cutout art
    # (leaves, plants, rope) stays hard-edged
    translucent = ((alpha > 0.05) & (alpha < 0.95)).mean() > 0.02
    opaque = alpha > (0.02 if translucent else 0.5)
    labels = material_regions(rgb, opaque)
    lab = srgb_to_lab(rgb)
    labels = merge_speckles(labels, max(4, (h * w) // 48), lab)
    big = upscale_regions(labels, scale)
    smooth_col = region_colour(rgb, labels, big, scale)
    m = Material(H, W)

    size = max(H, W)
    rng_seed = seed

    # material detail first: it drives both the resynthesised colour and relief
    def noise(cx, cy, oct, s):
        n = fbm(cx, cy, oct, rng_seed + s, size=size)
        return n[:H, :W]
    vp = size / max(h, w)  # output px per source px
    cells = max(2, int(size / vp * 0.75))
    lum0 = smooth_col @ np.array([0.2126, 0.7152, 0.0722])
    if cls == "wood":
        gx = np.abs(np.diff(rgb.mean(-1), axis=1)).mean()
        gy = np.abs(np.diff(rgb.mean(-1), axis=0)).mean()
        detail = noise(cells * 2, max(2, cells // 6), 4, 11) if gx > gy else noise(max(2, cells // 6), cells * 2, 4, 11)
    elif cls == "metal":
        detail = 0.6 * noise(cells * 3, 2, 3, 12) + 0.4 * scratches(int(30 * size / 256), rng_seed + 13, size=size)[:H, :W]
    elif cls == "fabric":
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        period = max(3.0, vp / 3)
        detail = 0.5 + 0.25 * np.sin(xx * 2 * np.pi / period) * np.sin(yy * 2 * np.pi / period) + 0.25 * (noise(cells, cells, 3, 14) - 0.5)
    elif cls in ("stone", "ore"):
        cracks = 1 - sstep(0, vp * 0.9, worley_np(int(cells * cells * 0.35) + 4, rng_seed + 16, size)[:H, :W])
        detail = 0.75 * noise(cells, cells, 5, 15) + 0.25 * (1 - cracks)
    else:
        detail = noise(cells, cells, 4, 15)

    # colour: the original's colours redistributed over the HD detail, keeping the
    # artist's large-scale shading from the smooth upscale
    resyn = resynthesise(rgb, labels, big, detail + 0.35 * (lum0 - blur_pad(lum0, vp * 3)))
    lowpass = blur_pad(lum0, vp * 2.5)
    region_mean = np.zeros((H, W))
    for i in np.unique(big):
        if i >= 0:
            region_mean[big == i] = lum0[big == i].mean()
    resyn = np.clip(resyn * (lowpass / np.maximum(region_mean, 1e-3))[..., None], 0, 1)
    # how "noisy" each region is in the original: its colour spread in Lab.
    # Noisy regions (stone, grain, food) take the resynthesised surface; flat,
    # designed regions keep the clean smooth upscale.
    noisy = np.zeros((H, W))
    for i in np.unique(big):
        if i < 0:
            continue
        spread = lab[labels == i].std(0).mean() if (labels == i).any() else 0.0
        noisy[big == i] = max(sstep(2.0, 7.0, spread), NATURAL_PRIOR.get(cls, 0.0))
    noisy = blur_pad(noisy, vp * 0.3)
    col = lerp(smooth_col, resyn, noisy)
    lum = col @ np.array([0.2126, 0.7152, 0.0722])

    # region base level: brighter regions sit a little higher (pixel-art convention)
    base = np.zeros((H, W))
    for i in np.unique(big):
        if i < 0:
            continue
        sel = big == i
        base[sel] = 0.55 + 0.2 * lum[sel].mean()
    bevel_px = max(2, int(round(scale * 0.45)))
    d = dist_to_edge(big, bevel_px + 2)
    bevel = round_profile(d, bevel_px)
    shade = blur_pad(lum, scale * 0.3) - blur_pad(lum, scale * 2.0)

    # bevel strength follows contrast: designed edges get a full rounded bevel,
    # boundaries inside noisy material barely any
    height = base * (0.75 + 0.25 * bevel * (1 - 0.8 * noisy)) + shade * 0.35 + (detail - 0.5) * props["relief"] * 4
    height = np.clip(height, 0.02, 1.0)

    albedo = col
    s_lo, s_hi = props["smooth"]
    smooth = s_lo + (s_hi - s_lo) * np.clip(0.5 + (detail - 0.5) * 1.5 + shade, 0, 1)

    f0 = np.full((H, W), float(props["f0"]))
    if cls in ("metal", "ore"):
        metal_id = next((v for pat, v in METAL_IDS if re.search(pat, name)), ALBEDO_METAL)
        chroma_lum = lum
        if cls == "metal":
            metallic = (chroma_lum > 0.12) & (big >= 0)
        else:
            # ore: the bright, saturated specks are the ore; the rest is stone
            sat = col.max(-1) - col.min(-1)
            metallic = (sat > 0.18) | (chroma_lum > 0.75)
            smooth = np.where(metallic, 0.7, smooth)
        f0 = np.where(metallic, metal_id, DIELECTRIC)
    m.albedo, m.height, m.smooth, m.f0 = albedo, height, smooth, f0
    m.porosity = np.full((H, W), float(props["porosity"]))
    if re.search(EMISSIVE, name):
        m.emissive = sstep(0.55, 0.9, lum) * (big >= 0)
    if translucent:
        a = np.asarray(Image.fromarray(alpha.astype(np.float32), "F").resize((W, H), Image.BICUBIC), float)
        m.alpha = np.clip(a, 0, 1) * (big >= 0)
    else:
        m.alpha = (big >= 0).astype(float)
    return m, dict(edge_wear=0.35 if cls == "metal" else 0.15, edges="wrap")


def remaster(src_path, scale, rel_path, seed=0):
    """Remaster a texture file (handles animation strips frame by frame)."""
    im = np.asarray(Image.open(src_path).convert("RGBA"), float) / 255
    h, w = im.shape[:2]
    frames = [im] if h <= w else [im[i * w:(i + 1) * w] for i in range(h // w)]
    outs = []
    for fi, fr in enumerate(frames):
        m, opts = remaster_frame(fr, scale, rel_path, seed=seed)
        outs.append(derive(m, **opts))
    return [np.concatenate([o[k] for o in outs], 0) for k in range(3)]
