"""Hand-made storage blocks: iron, gold, diamond, emerald, netherite, copper (all
oxidation stages, plain and cut), lapis, redstone, coal, Create zinc.

Vanilla layout: a 1px bevelled border around a plate with diagonal sheen. Here:
a raised rounded rim, a recessed plate and a surface per material; colours are
rank-mapped from the source's own rim and plate pixels.
"""
import math
import re

import numpy as np

from texgen import (Material, derive, fbm, worley, sstep, round_profile, IRON, GOLD, COPPER,
                    ALBEDO_METAL, DIELECTRIC, scratches)
from families import rank_map, grid

COPPER_STAGE = {"": 0, "exposed_": 1, "weathered_": 2, "oxidized_": 3}
PATTERN = re.compile(r"^(iron|gold|diamond|emerald|netherite|lapis|redstone|coal|zinc)_block$|"
                     r"^(exposed_|weathered_|oxidized_)?(cut_)?copper(_block)?$")


def storage_of(path, ns):
    if not path.startswith("textures/block/"):
        return None
    name = path.split("/")[-1][:-4]
    if ns == "create" and name != "zinc_block":
        return None
    if ns not in ("minecraft", "create"):
        return None
    return name if PATTERN.match(name) else None


def render(path, src, size, ns="minecraft"):
    name = path.split("/")[-1][:-4]
    s = size / 16
    xx, yy = grid(size)
    ux, uy = xx / s, yy / s
    m_ = PATTERN.match(name)
    cut = "cut_" in name
    stage = COPPER_STAGE.get((re.match(r"^(exposed_|weathered_|oxidized_)?", name).group(1) or ""), 0)
    kind = "copper" if "copper" in name else m_.group(1)

    # plates: one, or 2x2 tiles for cut copper (each with its own rim)
    tiles = 2 if cut else 1
    tw = 16 / tiles
    lx, ly = ux % tw, uy % tw
    d_edge = np.minimum(np.minimum(lx, tw - lx), np.minimum(ly, tw - ly))
    rim = d_edge < 1.0
    bevel = round_profile(1.0 - np.abs(d_edge - 0.5) * 2, 0.6) * rim

    rgb = src[..., :3]
    border = np.zeros(src.shape[:2], bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    if cut:
        border[7:9, :] = True
        border[:, 7:9] = True
    rim_cols, plate_cols = rgb[border], rgb[~border]

    sheen = 0.5 + 0.5 * np.sin((ux + uy) / 16 * 2 * math.pi * 1.5 + fbm(3, 3, 2, 21, size=size) * 2)
    brushed = fbm(2, 64, 3, 22, size=size)
    tone = fbm(6, 6, 4, 23, size=size)
    height = np.where(rim, 0.86 + 0.13 * bevel, 0.84)
    f0, smooth = DIELECTRIC, 0.4
    if kind == "iron" or kind == "zinc":
        rivets = np.zeros_like(ux)
        for cx, cy in ((2.2, 2.2), (13.8, 2.2), (2.2, 13.8), (13.8, 13.8)):
            rr = np.hypot(ux - cx, uy - cy)
            rivets = np.maximum(rivets, np.sqrt(np.clip(1 - (rr / 0.65) ** 2, 0, 1)))
        field = 0.55 * brushed + 0.3 * sheen + 0.15 * tone + 0.6 * rivets
        height = height + 0.06 * rivets + 0.004 * brushed
        f0, smooth = (IRON if kind == "iron" else ALBEDO_METAL), 0.6 + 0.1 * sheen
    elif kind == "gold":
        dents = (1 - sstep(0, 1.6 * s, worley(60, 24, size, jitter=0.9))) ** 2
        field = 0.65 * sheen + 0.35 * tone          # dents show in relief only, not colour
        height = height - 0.01 * dents
        f0, smooth = GOLD, 0.82 + 0.08 * sheen
    elif kind in ("diamond", "emerald"):
        p = 4.0 if kind == "diamond" else 3.0
        u, v = (ux + uy) / p, (ux - uy) / p
        facet = 1 - 2 * np.maximum(np.abs(u - np.floor(u) - 0.5), np.abs(v - np.floor(v) - 0.5))
        field = 0.45 * facet + 0.4 * sheen + 0.15 * tone
        height = np.where(rim, height, 0.82 + 0.06 * facet)
        f0, smooth = (44 if kind == "diamond" else 40), 0.9
    elif kind == "netherite":
        sc = scratches(40, 25, length=(8, 30), size=size)
        field = 0.5 * brushed + 0.3 * tone + 0.2 * sheen + 0.4 * sc
        height = height - 0.008 * sc
        f0, smooth = ALBEDO_METAL, 0.55 + 0.15 * sc
    elif kind == "copper":
        patina = sstep(0.35, 0.75, fbm(8, 8, 5, 26 + stage, size=size))
        field = 0.4 * brushed + 0.3 * sheen + 0.3 * tone + 0.2 * patina * (stage > 0)
        metal_amt = [1.0, 0.65, 0.3, 0.0][stage]
        f0 = np.where(patina * (1 - metal_amt) > 0.5, DIELECTRIC, COPPER)
        smooth = 0.62 * metal_amt + 0.25 * (1 - metal_amt) + 0.05 * sheen
    elif kind == "lapis":
        flecks = 1 - sstep(0, 0.25 * s, worley(140, 27, size, jitter=0.95))
        field = 0.7 * tone + 0.3 * flecks
        smooth = 0.35 + 0.4 * flecks
    elif kind == "redstone":
        dots = 1 - sstep(0, 0.6 * s, worley(36, 28, size, jitter=0.8))
        field = 0.6 * dots + 0.4 * tone
        height = height + 0.03 * dots
        smooth = 0.45
    elif kind == "coal":
        cracks = 1 - sstep(0, 0.2 * s, worley(30, 29, size, jitter=0.9))
        field = 0.7 * tone + 0.3 * fbm(30, 30, 3, 30, size=size) - 0.4 * cracks
        height = height - 0.02 * cracks
        smooth = 0.3 + 0.2 * tone
    col = rank_map(field, plate_cols, ~rim)
    col = np.where(rim[..., None], rank_map(bevel + 0.2 * tone, rim_cols, rim), col)
    m = Material(size)
    m.albedo = np.clip(col, 0, 1)
    m.height = np.clip(height + (0.99 - height.max()), 0.02, 0.99)
    m.f0 = np.broadcast_to(f0, (size, size)).astype(float).copy()
    m.smooth = np.broadcast_to(smooth, (size, size)).astype(float).copy()
    m.porosity[:] = 0 if kind not in ("lapis", "coal", "redstone") else 15
    return derive(m, edge_wear=0.25, ao_strength=0.8, edges="wrap")
