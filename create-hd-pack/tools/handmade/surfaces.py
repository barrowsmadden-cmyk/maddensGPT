"""Hand-designed material surfaces for the colour sets: wool, concrete,
terracotta and concrete powder (16 colours each, plus plain terracotta).

One design per material; colours are rank-mapped from each source texture, so
every colour keeps its exact vanilla tones. All tile seamlessly.
"""
import math
import re

import cv2
import numpy as np

from texgen import Material, derive, fbm, worley, sstep, blur, _rng, DIELECTRIC
from families import rank_map, opaque_colours, grid


def _wrapped_strokes(size, n, length, width, seed, curl=0.6):
    """Tileable layer of short curved strokes (fibres), drawn on a 3x3 canvas."""
    r = _rng(seed)
    big = np.zeros((size * 3, size * 3), np.float32)
    for _ in range(n):
        x, y = r.uniform(size, 2 * size, 2)
        ang = r.uniform(0, 2 * math.pi)
        pts = []
        for _s in range(6):
            pts.append((x, y))
            ang += r.normal(0, curl)
            x += math.cos(ang) * length / 6
            y += math.sin(ang) * length / 6
        v = float(r.uniform(0.4, 1.0))
        cv2.polylines(big, [np.round(np.array(pts)).astype(np.int32).reshape(-1, 1, 2)], False, v,
                      thickness=max(1, int(width)), lineType=cv2.LINE_AA)
    return big.reshape(3, size, 3, size).max(axis=(0, 2))


def wool(src, size, seed=0):
    s = size / 16
    tufts = fbm(10, 10, 4, seed + 1, size=size)                     # felted lumps
    fibres_l = _wrapped_strokes(size, int(900 * (size / 256) ** 2), 1.6 * s, max(1, s * 0.07), seed + 2)
    fibres_d = _wrapped_strokes(size, int(700 * (size / 256) ** 2), 1.3 * s, max(1, s * 0.06), seed + 3)
    # knit: columns of soft V stitches (2 vanilla px wide, 1.5 tall), each stitch a
    # rounded loop; fibres and gentle felting on top
    xx, yy = grid(size)
    col_w, row_h = 2.0 * s, 1.5 * s
    u = (xx / col_w) % 1.0
    v = (yy / row_h + np.abs(u - 0.5) * 0.9) % 1.0
    loop = np.sin(math.pi * v) ** 0.8 * np.sin(math.pi * u) ** 0.4
    field = 0.3 * loop + 0.35 * tufts + 0.22 * fibres_l - 0.15 * fibres_d
    height = 0.75 + 0.06 * loop + 0.05 * tufts + 0.03 * fibres_l - 0.03 * fibres_d
    m = Material(size)
    m.albedo = rank_map(field, opaque_colours(src))
    m.height = height
    m.smooth[:] = 0.06
    m.porosity[:] = 38
    return m


def concrete(src, size, seed=0):
    s = size / 16
    mottle = fbm(6, 6, 5, seed + 4, size=size)
    fine = fbm(48, 48, 3, seed + 5, size=size)
    pores = (1 - sstep(0, 0.09 * s, worley(int(256 * 0.35), seed + 6, size, jitter=0.95))) * 0.7
    trowel = fbm(2, 14, 3, seed + 7, size=size)                     # faint sweeping trowel marks
    field = 0.6 * mottle + 0.25 * fine + 0.15 * trowel - 0.8 * pores
    m = Material(size)
    m.albedo = rank_map(field, opaque_colours(src))
    m.albedo = m.albedo * (1 - 0.25 * pores)[..., None]
    m.height = 0.9 + 0.02 * mottle + 0.01 * trowel - 0.08 * pores
    m.smooth = 0.42 + 0.08 * (mottle - 0.5) - 0.2 * pores
    m.porosity[:] = 8
    return m


def terracotta(src, size, seed=0):
    s = size / 16
    xx, yy = grid(size)
    warp = fbm(4, 4, 3, seed + 8, size=size)
    strata = 0.5 + 0.5 * np.sin(2 * math.pi * (yy / (3.2 * s) + warp * 0.8))
    grain = fbm(40, 40, 4, seed + 9, size=size)
    specks = 1 - sstep(0, 0.1 * s, worley(int(256 * 0.25), seed + 10, size, jitter=0.95))
    field = 0.35 * strata + 0.5 * grain + 0.15 * fbm(8, 8, 4, seed + 11, size=size) - 0.5 * specks
    m = Material(size)
    m.albedo = rank_map(field, opaque_colours(src))
    m.height = 0.85 + 0.03 * strata + 0.04 * grain - 0.05 * specks
    m.smooth = 0.22 + 0.06 * (grain - 0.5)
    m.porosity[:] = 30
    return m


def concrete_powder(src, size, seed=0):
    s = size / 16
    f1 = worley(int(256 * 2.2), seed + 12, size, jitter=0.95)
    grains = np.clip(1 - f1 / (0.28 * s), 0, 1) ** 0.7               # little rounded grains
    clumps = fbm(8, 8, 4, seed + 13, size=size)
    field = 0.55 * grains + 0.45 * clumps
    m = Material(size)
    m.albedo = rank_map(field, opaque_colours(src))
    m.height = 0.7 + 0.18 * grains + 0.08 * clumps
    m.smooth[:] = 0.12
    m.porosity[:] = 60
    return m


COLOURS = (r"white|orange|magenta|light_blue|yellow|lime|pink|gray|light_gray|cyan|purple|blue|brown|green|red|black")
SURFACES = [
    (re.compile(rf"^({COLOURS})_concrete_powder$"), concrete_powder),
    (re.compile(rf"^({COLOURS})_concrete$"), concrete),
    (re.compile(rf"^({COLOURS})_wool$"), wool),
    (re.compile(rf"^(({COLOURS})_)?terracotta$"), terracotta),
]


def surface_of(path, ns):
    if ns != "minecraft" or not path.startswith("textures/block/"):
        return None
    name = path.split("/")[-1].rsplit(".", 1)[0]
    return next((fn for pat, fn in SURFACES if pat.match(name)), None)


def render(path, src, size, ns="minecraft"):
    fn = surface_of(path, ns)
    m = fn(src, size, seed=7)
    m.height = np.clip(m.height + (0.99 - m.height.max()), 0.02, 0.99)
    return derive(m, edge_wear=0.0, ao_strength=1.0, edges="wrap")
