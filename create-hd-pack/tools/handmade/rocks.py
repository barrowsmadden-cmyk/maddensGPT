"""Hand-designed natural ground and stone: one rock generator, one recipe per
block. Colours are rank-mapped from each vanilla texture so every stone keeps
its own palette. All tile seamlessly.

Recipe keys (all optional):
  blotch   (scale, weight)        large soft colour patches
  grains   (count, size, weight)  speckle grains (andesite/diorite/granite, sand)
  pits     (count, size, depth)   pores and pits (tuff, end stone)
  cracks   (count, depth)         thin fissures
  strata   (period, weight, warp) horizontal layering (deepslate, tuff)
  pebbles  (count, )              rounded stones (gravel, dirt)
  bumps    (scale, height)        lumpy relief (netherrack)
  smooth   (lo, hi), porosity
"""
import math
import re

import cv2
import numpy as np

from texgen import Material, derive, fbm, worley, sstep, round_profile, _rng, cell_search
from families import rank_map, opaque_colours, grid, worley2

RECIPES = {
    "stone": dict(blotch=(5, 0.6), grains=(260, 0.22, 0.15), cracks=(5, 0.05), smooth=(0.25, 0.35), porosity=30),
    "andesite": dict(blotch=(6, 0.35), grains=(900, 0.2, 0.55), cracks=(2, 0.03), smooth=(0.28, 0.38), porosity=25),
    "diorite": dict(blotch=(5, 0.3), grains=(520, 0.28, 0.75), smooth=(0.3, 0.42), porosity=20),
    "granite": dict(blotch=(5, 0.3), grains=(320, 0.42, 0.7), smooth=(0.3, 0.45), porosity=20),
    "tuff": dict(blotch=(5, 0.4), strata=(2.6, 0.25, 0.8), pits=(220, 0.22, 0.06), smooth=(0.18, 0.26), porosity=50),
    "calcite": dict(blotch=(4, 0.5), grains=(160, 0.5, 0.25), cracks=(3, 0.02), smooth=(0.4, 0.55), porosity=15),
    "deepslate": dict(blotch=(6, 0.3), strata=(1.6, 0.55, 0.4), cracks=(4, 0.04), smooth=(0.25, 0.35), porosity=20),
    "deepslate_top": dict(blotch=(5, 0.5), cracks=(9, 0.05), grains=(200, 0.2, 0.2), smooth=(0.25, 0.35), porosity=20),
    "netherrack": dict(blotch=(7, 0.4), bumps=(9, 0.12), pits=(160, 0.25, 0.05), smooth=(0.2, 0.3), porosity=40),
    "end_stone": dict(blotch=(5, 0.4), pits=(300, 0.3, 0.07), smooth=(0.2, 0.3), porosity=45),
    "blackstone": dict(blotch=(5, 0.5), cracks=(8, 0.06), grains=(200, 0.25, 0.2), smooth=(0.25, 0.38), porosity=25),
    "sand": dict(blotch=(6, 0.25), grains=(2600, 0.12, 0.8), smooth=(0.12, 0.18), porosity=64),
    "red_sand": dict(blotch=(6, 0.25), grains=(2600, 0.12, 0.8), smooth=(0.12, 0.18), porosity=64),
    "suspicious_sand_0": None,
    "gravel": dict(blotch=(8, 0.3), grains=(900, 0.12, 0.4), pebbles=(150,), smooth=(0.2, 0.32), porosity=40),
    "dirt": dict(blotch=(8, 0.5), grains=(900, 0.16, 0.5), pebbles=(16,), smooth=(0.1, 0.16), porosity=55),
    "coarse_dirt": dict(blotch=(6, 0.4), grains=(700, 0.18, 0.5), pebbles=(36,), smooth=(0.12, 0.2), porosity=50),
    "clay": dict(blotch=(4, 0.7), cracks=(2, 0.02), smooth=(0.35, 0.45), porosity=35),
    "snow": dict(blotch=(4, 0.4), grains=(1800, 0.1, 0.35), smooth=(0.3, 0.4), porosity=10),
    "packed_mud": dict(blotch=(5, 0.5), grains=(500, 0.2, 0.35), cracks=(3, 0.03), smooth=(0.18, 0.25), porosity=45),
    "mud": dict(blotch=(5, 0.6), bumps=(6, 0.06), smooth=(0.55, 0.7), porosity=64),
}
RECIPES = {k: v for k, v in RECIPES.items() if v}


def rock_of(path, ns):
    if ns != "minecraft" or not path.startswith("textures/block/"):
        return None
    name = path.split("/")[-1][:-4]
    return name if name in RECIPES else None


def _crack_layer(size, count, seed):
    r = _rng(seed)
    big = np.zeros((size * 3, size * 3), np.uint8)
    s = size / 16
    for _ in range(count):
        x, y = r.uniform(size, 2 * size, 2)
        ang = r.uniform(0, 2 * math.pi)
        pts = []
        for _st in range(int(r.integers(10, 26))):
            pts.append((x, y))
            ang += r.normal(0, 0.5)
            x += math.cos(ang) * 0.4 * s
            y += math.sin(ang) * 0.4 * s
        cv2.polylines(big, [np.round(np.array(pts)).astype(np.int32).reshape(-1, 1, 2)], False, 255,
                      thickness=max(1, int(s * 0.08)), lineType=cv2.LINE_AA)
    return big.reshape(3, size, 3, size).max(axis=(0, 2)).astype(float) / 255


def render(path, src, size, ns="minecraft"):
    name = path.split("/")[-1][:-4]
    rc = RECIPES[name]
    s = size / 16
    xx, yy = grid(size)
    seed = sum(map(ord, name))
    field = np.zeros((size, size))
    height = np.full((size, size), 0.8)
    if "blotch" in rc:
        sc, w = rc["blotch"]
        b = fbm(sc, sc, 5, seed + 1, size=size)
        field += w * b
        height += 0.04 * (b - 0.5)
    fine = fbm(40, 40, 3, seed + 2, size=size)
    field += 0.15 * fine
    height += 0.015 * (fine - 0.5)
    if "grains" in rc:
        n, gs, w = rc["grains"]
        k = max(2, int(round(math.sqrt(n))))
        f1 = worley(k * k, seed + 3, size, jitter=0.95)
        g = np.clip(1 - f1 / (gs * s), 0, 1)
        r = _rng(seed + 4)
        # each grain gets its own tone so speckles span the palette
        f1b, _, ids, npts = worley2(k * k, seed + 3, size, jitter=0.95)
        tones = r.uniform(-1, 1, npts)
        speck = sstep(0.15, 0.45, g)
        field = field * (1 - w * speck) + (0.5 + 0.9 * tones[ids]) * w * speck
        height += 0.008 * speck
    if "strata" in rc:
        per, w, wp = rc["strata"]
        warp = fbm(2, 5, 4, seed + 5, size=size) + 0.35 * fbm(6, 10, 3, seed + 12, size=size)
        phase = yy / (per * s) + warp * wp * 2.2
        band = 0.5 + 0.5 * np.sin(2 * math.pi * phase)
        flake = sstep(0.55, 0.95, fbm(4, 16, 3, seed + 13, size=size))   # broken, discontinuous layers
        ridge = sstep(0.55, 0.85, band) * (0.4 + 0.6 * flake)
        field += w * (0.6 * band + 0.4 * ridge)
        height += 0.05 * ridge
    if "pits" in rc:
        n, ps, d = rc["pits"]
        f1 = worley(n, seed + 6, size, jitter=0.95)
        pit = (1 - sstep(0, ps * s, f1)) ** 1.5 * _rng(seed + 11).uniform(0.3, 1.0)
        field -= 0.25 * pit
        height -= d * pit
    if "bumps" in rc:
        sc, h = rc["bumps"]
        b = fbm(sc, sc, 4, seed + 7, size=size)
        field += 0.3 * b
        height += h * b
    if "pebbles" in rc:
        n = rc["pebbles"][0]
        f1, f2, ids, npts = worley2(n, seed + 8, size, jitter=0.9)
        cell = 16 / math.sqrt(n)                                   # vanilla px per cell
        rr = _rng(seed + 9)
        radii = rr.uniform(0.32, 0.5, npts) * cell * s
        if n < 40:                                                 # sparse stones in soil: smaller, fewer
            radii *= rr.uniform(0.0, 1.0, npts) > 0.35
            radii = np.minimum(radii, 0.65 * s)
        rad = radii[ids]
        cap = np.sqrt(np.clip(1 - (f1 / np.maximum(rad, 1e-3)) ** 2, 0, 1))
        stone = (f1 < rad).astype(float)
        tones = rr.uniform(-0.7, 0.7, npts)
        field = field * (1 - stone) + (0.5 + tones[ids] + 0.35 * cap) * stone
        height = height + 0.14 * cap * stone - (0.05 if n >= 40 else 0.0) * (1 - stone)
    if "cracks" in rc:
        n, d = rc["cracks"]
        c = _crack_layer(size, n, seed + 10) * 0.8
        field -= 0.3 * c
        height -= d * c
    m = Material(size)
    m.albedo = rank_map(field, opaque_colours(src))
    m.height = np.clip(height + (0.99 - height.max()), 0.02, 0.99)
    lo, hi = rc.get("smooth", (0.2, 0.3))
    m.smooth = lo + (hi - lo) * np.clip((field - field.min()) / (np.ptp(field) + 1e-6), 0, 1)
    m.porosity[:] = rc.get("porosity", 30)
    return derive(m, edge_wear=0.05, ao_strength=1.1, edges="wrap")
