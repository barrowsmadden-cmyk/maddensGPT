"""Hand-made brick walls from explicit vanilla layouts.

Bricks are rectangles in vanilla pixel units (they may wrap around the tile);
everything else (bevels, chips, stone surface, mortar, moss, cracks) is drawn
from those rectangles, with colours taken from the vanilla texture's own brick
and mortar pixels so each variant keeps its palette.
"""
import math

import numpy as np
from PIL import Image

from texgen import Material, derive, fbm, worley, sstep, round_profile, blur, _rng
from families import rank_map, grid, LUM

# stone bricks: two courses of 15px bricks, 1px joints, courses offset by half
STONE_BRICKS = dict(
    bricks=[(0, 0, 15, 7), (8, 8, 23, 15)],          # x0, y0, x1, y1 (x1 may exceed 16: wraps)
    mortar_rows=[7, 15], mortar_cols=[(15, 0, 7), (7, 8, 15)],
)

LAYOUTS = {
    "stone_bricks": STONE_BRICKS,
    "mossy_stone_bricks": STONE_BRICKS,
    "cracked_stone_bricks": STONE_BRICKS,
    "infested_stone_bricks": STONE_BRICKS,
}


def _src_masks(src, layout):
    h, w = src.shape[:2]
    mortar = np.zeros((h, w), bool)
    for r in layout["mortar_rows"]:
        mortar[r, :] = True
    for c, y0, y1 in layout["mortar_cols"]:
        mortar[y0:y1, c] = True
    return mortar


def brick_wall(src, size, layout, seed=0, moss=False, cracks=False):
    s = size / 16.0                       # output px per vanilla px
    xx, yy = grid(size)
    ux, uy = xx / s, yy / s               # vanilla px coordinates
    r = _rng(seed)
    brick_id = np.full((size, size), -1)
    d = np.zeros((size, size))
    for i, (x0, y0, x1, y1) in enumerate(layout["bricks"]):
        for shift in (0, -16, 16):        # wrap horizontally
            bx = ux - shift
            inside = (bx >= x0) & (bx < x1) & (uy >= y0) & (uy < y1)
            di = np.minimum(np.minimum(bx - x0, x1 - bx), np.minimum(uy - y0, y1 - uy))
            sel = inside & (brick_id < 0)
            brick_id = np.where(sel, i, brick_id)
            d = np.where(sel, di, d)
    # irregular, chipped edges: wobble the distance field near the edge
    wob = (fbm(24, 24, 4, seed + 1, size=size) - 0.5) * 0.45
    chips = sstep(0.62, 0.8, fbm(10, 10, 3, seed + 2, size=size)) * 0.35
    d_eff = d + wob - chips
    is_brick = (brick_id >= 0) & (d_eff > 0.02)
    face = round_profile(d_eff, 0.75) * is_brick

    rough = fbm(20, 20, 6, seed + 3, size=size)
    fine = fbm(64, 64, 3, seed + 4, size=size)
    pits = 1 - sstep(0, 0.22 * s, worley(int(256 * 0.6), seed + 5, size, jitter=0.95))
    tones = r.uniform(-0.22, 0.22, len(layout["bricks"]) + 1)
    surface = (rough - 0.5) * 0.06 + (fine - 0.5) * 0.02 - pits * 0.035
    height = np.where(is_brick, 0.58 + 0.34 * face + surface, 0.26 + (fine - 0.5) * 0.06)

    mortar_src = _src_masks(src, layout)
    rgb = src[..., :3]
    brick_cols = rgb[~mortar_src]
    mortar_cols = rgb[mortar_src]
    field = tones[np.maximum(brick_id, 0)] + (rough - 0.5) * 1.1 + (fine - 0.5) * 0.4 - pits * 0.5 + face * 0.15
    col = rank_map(field, brick_cols, is_brick)
    col = np.where(is_brick[..., None], col, rank_map(fine + rough * 0.5, mortar_cols, ~is_brick))
    smooth = np.where(is_brick, 0.22 + face * 0.06, 0.08)
    poro = np.where(is_brick, 38.0, 58.0)

    if cracks:
        import cv2
        cmask = np.zeros((size * 3, size * 3), np.uint8)   # 3x3 tiles so cracks wrap
        for _ in range(4):
            x, y = r.uniform(0, 16), r.uniform(0, 16)
            ang = r.uniform(0, 2 * math.pi)
            pts = []
            for _step in range(26):
                ang += r.normal(0, 0.55)
                x, y = x + math.cos(ang) * 0.42, y + math.sin(ang) * 0.42
                pts.append((x * s + size, y * s + size))
            p_ = np.round(np.array(pts)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(cmask, [p_], False, 255, thickness=max(1, int(s * 0.18)), lineType=cv2.LINE_AA)
        cm = cmask.astype(float).reshape(3, size, 3, size).max(axis=(0, 2)) / 255.0
        crack = cm * is_brick
        height -= 0.16 * crack
        col = col * (1 - 0.6 * crack)[..., None]

    if moss:
        g = rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2])
        sat = rgb.max(-1) - rgb.min(-1)
        moss_src = (g > 0.05) & (sat > 0.12)
        if moss_src.any():
            big = np.asarray(Image.fromarray(moss_src.astype(np.float32), "F").resize((size, size), Image.BILINEAR), float)
            wob2 = fbm(18, 18, 5, seed + 6, size=size)
            clumps = fbm(40, 40, 4, seed + 7, size=size)
            m_mask = sstep(0.42, 0.52, blur(big, s * 0.6) + (wob2 - 0.5) * 0.5) * sstep(0.3, 0.45, clumps)
            # moss settles in the joints and lower edges first
            m_mask = np.clip(m_mask * (1.2 - 0.6 * face), 0, 1)
            moss_cols = rgb[moss_src]
            col = col * (1 - m_mask[..., None]) + rank_map(wob2 + fine * 0.5, moss_cols, m_mask > 0.01) * m_mask[..., None]
            height = height + m_mask * (0.06 + 0.04 * wob2)
            smooth = smooth * (1 - m_mask) + 0.08 * m_mask
            poro = np.where(m_mask > 0.5, 62.0, poro)

    m = Material(size)
    m.albedo = np.clip(col, 0, 1)
    m.height = np.clip(height + (0.99 - height.max()), 0.02, 0.99)
    m.smooth, m.porosity = smooth, poro
    return derive(m, edge_wear=0.08, ao_strength=1.3, edges="wrap")


def textures():
    """(path, layout, options) for every hand-made brick texture."""
    return [
        ("textures/block/stone_bricks.png", STONE_BRICKS, {}),
        ("textures/block/mossy_stone_bricks.png", STONE_BRICKS, {"moss": True}),
        ("textures/block/cracked_stone_bricks.png", STONE_BRICKS, {"cracks": True}),
    ]
